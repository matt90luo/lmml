"""独立 LLM 训练数据准备管线：多源加载、抽样、标准化、分词与 Arrow 落盘。"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from rich.console import Console
from rich.table import Table

from lmml.datas.llm.sft import (
    DEFAULT_CHAT_END_TOKEN,
    DEFAULT_CHAT_START_TOKEN,
    DEFAULT_IGNORE_INDEX,
    encode_chatml_messages,
    normalize_sft_batch,
    resolve_num_proc,
    tokenize_sft_batch,
)

console = Console()

PREPARE_TASKS = ("pretrain", "sft", "dpo", "ppo", "grpo", "agent_rl")


def _normalize_messages(value: Any, *, default_role: str | None = None) -> list[dict[str, str]]:
    """将消息列表或纯文本字段转换为统一的 role/content 消息列表。"""
    if isinstance(value, str):
        text = value.strip()
        return [{"role": default_role, "content": text}] if text and default_role else []
    if not isinstance(value, list):
        return []

    messages: list[dict[str, str]] = []
    for message in value:
        if not isinstance(message, dict):
            continue
        raw_role = message.get("role") or message.get("from") or default_role or "user"
        if raw_role in ("human", "user"):
            role = "user"
        elif raw_role == "system":
            role = "system"
        else:
            role = "assistant"
        content = message.get("content")
        if content is None:
            content = message.get("value", "")
        if content is not None and str(content).strip():
            messages.append({"role": role, "content": str(content)})
    return messages


def _normalize_prompt(item: dict[str, Any]) -> list[dict[str, str]]:
    for key in ("prompt", "messages", "conversations", "query", "instruction"):
        value = item.get(key)
        messages = _normalize_messages(value, default_role="user")
        if not messages:
            continue
        if key == "instruction" and item.get("input") is not None:
            extra = str(item["input"]).strip()
            if extra:
                messages[0]["content"] = f"{messages[0]['content']}\n{extra}"
        return [m for m in messages if m["role"] != "assistant"]
    return []


def normalize_dpo_batch(batch: dict[str, list[Any]]) -> dict[str, list[list[dict[str, str]]]]:
    """批量标准化 DPO chosen/rejected 数据，并过滤任一侧无效的样本。"""
    keys = list(batch.keys())
    if not keys:
        return {"chosen_messages": [], "rejected_messages": []}

    chosen_messages: list[list[dict[str, str]]] = []
    rejected_messages: list[list[dict[str, str]]] = []
    for i in range(len(batch[keys[0]])):
        row = {key: batch[key][i] for key in keys}
        prompt = _normalize_prompt(row)
        chosen = _normalize_messages(row.get("chosen"), default_role="assistant")
        rejected = _normalize_messages(row.get("rejected"), default_role="assistant")
        chosen = [*prompt, *chosen]
        rejected = [*prompt, *rejected]
        if not chosen or not rejected:
            continue
        if not any(m["role"] == "assistant" for m in chosen):
            continue
        if not any(m["role"] == "assistant" for m in rejected):
            continue
        chosen_messages.append(chosen)
        rejected_messages.append(rejected)
    return {"chosen_messages": chosen_messages, "rejected_messages": rejected_messages}


def normalize_ppo_batch(batch: dict[str, list[Any]]) -> dict[str, list[list[dict[str, str]]]]:
    """批量标准化 PPO prompt 数据，保留 system/user 等上下文消息。"""
    keys = list(batch.keys())
    if not keys:
        return {"prompt": []}

    prompts: list[list[dict[str, str]]] = []
    for i in range(len(batch[keys[0]])):
        messages = _normalize_prompt({key: batch[key][i] for key in keys})
        messages = [m for m in messages if m["role"] != "assistant"]
        if messages:
            prompts.append(messages)
    return {"prompt": prompts}


def tokenize_dpo_batch(
    batch: dict[str, list[Any]],
    *,
    tokenizer: Any,
    max_length: int = 2048,
    chat_start_token: str = DEFAULT_CHAT_START_TOKEN,
    chat_end_token: str = DEFAULT_CHAT_END_TOKEN,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    drop_no_target: bool = True,
) -> dict[str, list[Any]]:
    """批量分词 DPO 偏好对，并分别生成 chosen/rejected 的 Prompt Masking labels。"""
    output: dict[str, list[Any]] = {
        "chosen_input_ids": [],
        "chosen_labels": [],
        "rejected_input_ids": [],
        "rejected_labels": [],
        "length": [],
    }
    for chosen, rejected in zip(batch.get("chosen_messages", []), batch.get("rejected_messages", [])):
        chosen_encoded = encode_chatml_messages(
            messages=chosen,
            tokenizer=tokenizer,
            max_length=max_length,
            chat_start_token=chat_start_token,
            chat_end_token=chat_end_token,
            ignore_index=ignore_index,
            drop_no_target=drop_no_target,
        )
        rejected_encoded = encode_chatml_messages(
            messages=rejected,
            tokenizer=tokenizer,
            max_length=max_length,
            chat_start_token=chat_start_token,
            chat_end_token=chat_end_token,
            ignore_index=ignore_index,
            drop_no_target=drop_no_target,
        )
        if chosen_encoded is None or rejected_encoded is None:
            continue
        output["chosen_input_ids"].append(chosen_encoded["input_ids"])
        output["chosen_labels"].append(chosen_encoded["labels"])
        output["rejected_input_ids"].append(rejected_encoded["input_ids"])
        output["rejected_labels"].append(rejected_encoded["labels"])
        output["length"].append(max(chosen_encoded["length"], rejected_encoded["length"]))
    return output


def encode_chatml_prompt(
    messages: list[dict[str, str]],
    tokenizer: Any,
    max_length: int = 2048,
    chat_start_token: str = DEFAULT_CHAT_START_TOKEN,
    chat_end_token: str = DEFAULT_CHAT_END_TOKEN,
) -> dict[str, Any] | None:
    """编码 RL 生成 prompt；优先使用 tokenizer 自带的 ChatML/CoT 模板。"""
    rendered = _render_chat_prompt(messages, tokenizer, add_generation_prompt=True)
    input_ids = tokenizer.encode(rendered, add_special_tokens=False)
    input_ids = input_ids[:max_length]
    if len(input_ids) < 2:
        return None
    return {"input_ids": input_ids, "prompt_length": len(input_ids), "length": len(input_ids)}


def _coerce_messages(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(decoded, list):
            return [item for item in decoded if isinstance(item, dict)]
    return []


def _normalize_agent_message(message: dict[str, Any]) -> dict[str, Any] | None:
    role = str(message.get("role") or message.get("from") or "user").strip().lower()
    if role in {"human", "user"}:
        role = "user"
    elif role in {"gpt", "model", "assistant", "bot"}:
        role = "assistant"
    elif role in {"tool", "function", "observation", "tool_response"}:
        role = "tool"
    content = message.get("content")
    if content is None:
        content = message.get("value")
    out: dict[str, Any] = {"role": role, "content": str(content or "")}
    for key in ("tools", "tool_calls", "tool_call_id"):
        if key in message:
            out[key] = message[key]
    return out if out["content"] or out["role"] == "assistant" else None


def _render_chat_prompt(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    *,
    add_generation_prompt: bool = True,
) -> str:
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except (TypeError, ValueError, KeyError):
            pass
    pieces = []
    for message in messages:
        role = "user" if message.get("role") == "tool" else message.get("role", "user")
        pieces.append(
            f"{DEFAULT_CHAT_START_TOKEN}{role}\n{message.get('content', '')}{DEFAULT_CHAT_END_TOKEN}\n"
        )
    if add_generation_prompt:
        pieces.append(f"{DEFAULT_CHAT_START_TOKEN}assistant\n")
    return "".join(pieces)


def normalize_pretrain_batch(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
    """Normalize plain-text pretraining rows and discard empty documents."""
    keys = list(batch.keys())
    if not keys:
        return {"text": []}
    texts: list[str] = []
    for index in range(len(batch[keys[0]])):
        row = {key: batch[key][index] for key in keys}
        text = row.get("text")
        if text is None:
            text = row.get("content")
        if text is None:
            text = row.get("document")
        text = str(text or "").strip()
        if text:
            texts.append(text)
    return {"text": texts}


def tokenize_pretrain_batch(
    batch: dict[str, list[Any]],
    *,
    tokenizer: Any,
    max_length: int = 512,
) -> dict[str, list[Any]]:
    """Tokenize raw text with MiniMind-style BOS/EOS and ignore-index padding."""
    output: dict[str, list[Any]] = {"input_ids": [], "labels": [], "length": []}
    bos_id = int(tokenizer.bos_token_id)
    eos_id = int(tokenizer.eos_token_id)
    for text in batch.get("text", []):
        tokens = tokenizer(
            str(text),
            add_special_tokens=False,
            truncation=True,
            max_length=max(0, int(max_length) - 2),
        ).input_ids
        ids = [bos_id, *tokens, eos_id]
        output["input_ids"].append(ids)
        output["labels"].append(list(ids))
        output["length"].append(len(ids))
    return output


def normalize_agent_rl_batch(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Normalize agent prompt conversations, tool definitions and GT answers."""
    keys = list(batch.keys())
    if not keys:
        return {"messages": [], "tools": [], "gt": []}
    messages_out: list[list[dict[str, str]]] = []
    tools_out: list[list[dict[str, Any]]] = []
    gt_out: list[list[str]] = []
    for index in range(len(batch[keys[0]])):
        row = {key: batch[key][index] for key in keys}
        raw_messages = row.get("messages") or row.get("conversations")
        messages = [
            message
            for raw in _coerce_messages(raw_messages)
            if (message := _normalize_agent_message(raw)) is not None
        ]
        messages = [
            {"role": "user" if message["role"] == "tool" else message["role"], "content": message["content"]}
            for message in messages
        ]
        if not messages:
            continue
        tools: Any = row.get("tools")
        if tools is None:
            for message in messages:
                if message.get("role") == "system" and message.get("tools"):
                    tools = message["tools"]
                    break
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except json.JSONDecodeError:
                tools = []
        if isinstance(tools, dict):
            tools = [tools]
        if not isinstance(tools, list):
            tools = []
        gt = row.get("gt", [])
        if isinstance(gt, str):
            gt = [gt]
        gt = [str(item) for item in (gt or []) if str(item).strip()]
        # The last assistant turn is the target; prompts exclude it.
        prompt_messages = messages[:-1] if messages[-1].get("role") == "assistant" else messages
        if not prompt_messages:
            continue
        messages_out.append(
            [
                {key: value for key, value in message.items() if key in {"role", "content"}}
                for message in prompt_messages
            ]
        )
        tools_out.append(tools)
        gt_out.append(gt)
    return {"messages": messages_out, "tools": tools_out, "gt": gt_out}


def tokenize_agent_rl_batch(
    batch: dict[str, list[Any]],
    *,
    tokenizer: Any,
    max_length: int = 2048,
    chat_start_token: str = DEFAULT_CHAT_START_TOKEN,
    chat_end_token: str = DEFAULT_CHAT_END_TOKEN,
) -> dict[str, list[Any]]:
    """Prepare plain prompt text and retain tools/gt for online rollouts."""
    output: dict[str, list[Any]] = {
        "prompt": [],
        "messages_json": [],
        "tools_json": [],
        "gt": [],
        "input_ids": [],
        "length": [],
    }
    for messages, tools, gt in zip(
        batch.get("messages", []), batch.get("tools", []), batch.get("gt", [])
    ):
        prompt = _render_chat_prompt(messages, tokenizer, add_generation_prompt=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)[: int(max_length)]
        if len(ids) < 2:
            continue
        output["prompt"].append(prompt)
        output["messages_json"].append(json.dumps(messages, ensure_ascii=False))
        output["tools_json"].append(json.dumps(tools, ensure_ascii=False))
        output["gt"].append(gt)
        output["input_ids"].append(ids)
        output["length"].append(len(ids))
    return output


def tokenize_ppo_batch(
    batch: dict[str, list[Any]],
    *,
    tokenizer: Any,
    max_length: int = 2048,
    chat_start_token: str = DEFAULT_CHAT_START_TOKEN,
    chat_end_token: str = DEFAULT_CHAT_END_TOKEN,
) -> dict[str, list[Any]]:
    """批量编码 PPO prompt。"""
    output: dict[str, list[Any]] = {
        "prompt": [],
        "input_ids": [],
        "prompt_length": [],
        "length": [],
    }
    for messages in batch.get("prompt", []):
        prompt_text = _render_chat_prompt(messages, tokenizer, add_generation_prompt=True)
        encoded = encode_chatml_prompt(
            messages=messages,
            tokenizer=tokenizer,
            max_length=max_length,
            chat_start_token=chat_start_token,
            chat_end_token=chat_end_token,
        )
        if encoded is None:
            continue
        output["prompt"].append(prompt_text)
        output["input_ids"].append(encoded["input_ids"])
        output["prompt_length"].append(encoded["prompt_length"])
        output["length"].append(encoded["length"])
    return output


def _load_single_source(path: str, cache_dir: Path | None = None) -> HFDataset:
    if os.path.isdir(path) and (
        os.path.exists(os.path.join(path, "dataset_info.json"))
        or os.path.exists(os.path.join(path, "dataset_dict.json"))
    ):
        dataset = load_from_disk(path)
    elif os.path.isdir(path):
        dataset = load_dataset(path, cache_dir=str(cache_dir) if cache_dir else None)
    elif path.endswith((".json", ".jsonl")):
        dataset = load_dataset(
            "json",
            data_files=path,
            split="train",
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    elif path.endswith(".parquet"):
        dataset = load_dataset(
            "parquet",
            data_files=path,
            split="train",
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    else:
        dataset = load_dataset(path, split="train", cache_dir=str(cache_dir) if cache_dir else None)

    if isinstance(dataset, (dict, DatasetDict)):
        dataset = dataset.get("train", next(iter(dataset.values())))
    return dataset


def _apply_subsampling(dataset: HFDataset, config: dict[str, Any], seed: int) -> tuple[HFDataset, int]:
    original_len = len(dataset)
    target_len = original_len
    sample_ratio = config.get("sample_ratio")
    max_samples = config.get("max_samples")
    if sample_ratio is not None and 0.0 < float(sample_ratio) < 1.0:
        target_len = int(original_len * float(sample_ratio))
    if max_samples is not None and int(max_samples) > 0:
        target_len = min(target_len, int(max_samples))
    if target_len < original_len:
        dataset = dataset.shuffle(seed=seed).select(range(target_len))
    return dataset, target_len


def _validate_task_columns(batch: dict[str, list[Any]], task: str) -> None:
    """在 map 前校验任务必需字段，避免所有源都被静默过滤后才发现配置错误。"""
    required = {
        "pretrain": {"text", "content", "document"},
        "sft": set(),
        "dpo": {"chosen", "rejected"},
        "ppo": {"prompt", "messages", "conversations", "query", "instruction"},
        "grpo": {"prompt", "messages", "conversations", "query", "instruction"},
        "agent_rl": {"messages", "conversations"},
    }[task]
    if required and not required.intersection(batch):
        raise ValueError(f"data.task={task} 需要数据源包含字段之一: {sorted(required)}")


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _save_dataset_dict(dataset: DatasetDict, output_dir: Path) -> None:
    """先写同盘临时目录，再替换目标目录，避免覆盖中途留下残缺 Arrow 数据。"""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temp_dir = output_dir.with_name(f".{output_dir.name}.tmp-{token}")
    backup_dir = output_dir.with_name(f".{output_dir.name}.backup-{token}")
    backup_created = False
    try:
        dataset.save_to_disk(str(temp_dir))
        if output_dir.exists():
            os.replace(output_dir, backup_dir)
            backup_created = True
        os.replace(temp_dir, output_dir)
        if backup_created:
            _remove_path(backup_dir)
    except BaseException:
        _remove_path(temp_dir)
        if backup_created and backup_dir.exists() and not output_dir.exists():
            os.replace(backup_dir, output_dir)
        raise


def load_prepared_dataset(prepared_path: str | Path) -> DatasetDict:
    """严格加载并校验独立 ``lmml data`` 生成的 DatasetDict。"""
    path = Path(prepared_path)
    if not path.exists() or not (path / "dataset_dict.json").exists():
        raise FileNotFoundError(
            f"未找到预处理数据集: {path}。请先运行 `lmml data experiment=...` 生成训练数据。"
        )
    dataset = load_from_disk(str(path))
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"预处理路径不是 DatasetDict: {path}")
    missing = {"train", "val"} - set(dataset.keys())
    if missing:
        raise ValueError(f"预处理数据集缺少切分 {sorted(missing)}: {path}")
    return dataset


def _prepared_summary(dataset: DatasetDict, output_dir: Path, task: str, overwritten: bool) -> dict[str, Any]:
    return {
        "task": task,
        "prepared_path": str(output_dir.resolve()),
        "train_samples": len(dataset["train"]),
        "val_samples": len(dataset["val"]),
        "overwritten": overwritten,
    }


def prepare_llm_dataset(
    data_configs: list[dict[str, Any]],
    tokenizer: Any,
    output_dir: str | Path,
    *,
    task: str = "sft",
    overwrite: bool = True,
    max_length: int = 2048,
    val_split_ratio: float = 0.02,
    max_val_samples: int | None = 1500,
    preprocessing_num_proc: int | str | None = "auto",
    chat_start_token: str = DEFAULT_CHAT_START_TOKEN,
    chat_end_token: str = DEFAULT_CHAT_END_TOKEN,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    drop_no_target: bool = True,
    wrap_reasoning: bool = False,
    add_system_ratio: float = 0.0,
    empty_think_remove_ratio: float = 0.0,
    seed: int = 42,
) -> dict[str, Any]:
    """构建 SFT / DPO / PPO 的预分词 DatasetDict 并持久化到输出目录。"""
    task = task.strip().lower()
    if task not in PREPARE_TASKS:
        raise ValueError(f"不支持的 data.task={task!r}，可选值: {PREPARE_TASKS}")
    if not isinstance(data_configs, list) or not data_configs:
        raise ValueError("data.sources 必须是非空的数据源字典列表。")

    output_path = Path(output_dir)
    if output_path.exists() and not overwrite:
        dataset = load_prepared_dataset(output_path)
        console.print(f"[bold cyan][Prepare][/bold cyan] 复用已有预处理数据集: {output_path}")
        return _prepared_summary(dataset, output_path, task, overwritten=False)

    normalize_fn: Callable[[dict[str, list[Any]]], dict[str, list[Any]]]
    tokenize_fn: Callable[..., dict[str, list[Any]]]
    tokenize_fn_kwargs: dict[str, Any] = {
        "tokenizer": tokenizer,
        "max_length": int(max_length),
        "chat_start_token": str(chat_start_token),
        "chat_end_token": str(chat_end_token),
    }
    if task == "pretrain":
        normalize_fn = normalize_pretrain_batch
        tokenize_fn = tokenize_pretrain_batch
        tokenize_fn_kwargs = {"tokenizer": tokenizer, "max_length": int(max_length)}
    elif task == "sft":
        normalize_fn = normalize_sft_batch
        tokenize_fn = tokenize_sft_batch
        tokenize_fn_kwargs.update(ignore_index=int(ignore_index), drop_no_target=bool(drop_no_target))
    elif task == "dpo":
        normalize_fn = normalize_dpo_batch
        tokenize_fn = tokenize_dpo_batch
        tokenize_fn_kwargs.update(ignore_index=int(ignore_index), drop_no_target=bool(drop_no_target))
    elif task == "agent_rl":
        normalize_fn = normalize_agent_rl_batch
        tokenize_fn = tokenize_agent_rl_batch
    else:
        normalize_fn = normalize_ppo_batch
        tokenize_fn = tokenize_ppo_batch

    summary_table = Table(title=f"{task.upper()} 数据源抽样与清洗汇总", show_header=True, header_style="bold cyan")
    summary_table.add_column("数据集", style="white")
    summary_table.add_column("原始条数", justify="right")
    summary_table.add_column("抽样后条数", justify="right")
    summary_table.add_column("有效样本数", justify="right", style="green")

    num_proc = resolve_num_proc(preprocessing_num_proc)
    loading_cache_dir = output_path.parent / ".lmml_datasets_cache"
    loaded_datasets: list[HFDataset] = []
    try:
        loading_cache_dir.mkdir(parents=True, exist_ok=True)
        for config in data_configs:
            path = str(config["path"])
            source_name = os.path.basename(path.rstrip("/")) or path
            if not os.path.exists(path):
                console.print(f"[yellow][Warning] 数据路径不存在，已跳过: {path}[/yellow]")
                continue
            try:
                dataset = _load_single_source(path, cache_dir=loading_cache_dir)
                original_len = len(dataset)
                dataset, sampled_len = _apply_subsampling(dataset, config, int(seed))
                if len(dataset) > 0:
                    _validate_task_columns({name: dataset[:1][name] for name in dataset.column_names}, task)
                normalize_kwargs = {
                    "wrap_reasoning": bool(wrap_reasoning),
                    "add_system_ratio": float(add_system_ratio),
                    "empty_think_remove_ratio": float(empty_think_remove_ratio),
                    "seed": int(seed),
                } if task == "sft" else {}
                dataset = dataset.map(
                    normalize_fn,
                    batched=True,
                    batch_size=1000,
                    remove_columns=dataset.column_names,
                    num_proc=num_proc,
                    fn_kwargs=normalize_kwargs,
                    desc=f"Normalizing {source_name}",
                )
                dataset = dataset.map(
                    tokenize_fn,
                    batched=True,
                    batch_size=1000,
                    remove_columns=dataset.column_names,
                    num_proc=num_proc,
                    fn_kwargs=tokenize_fn_kwargs,
                    desc=f"Tokenizing {source_name}",
                )
                if len(dataset) > 0:
                    loaded_datasets.append(dataset)
                summary_table.add_row(
                    source_name,
                    f"{original_len:,}",
                    f"{sampled_len:,}",
                    f"{len(dataset):,}",
                )
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                console.print(f"[red][Error] 读取或处理数据失败 {path}: {exc}[/red]")

        if not loaded_datasets:
            raise ValueError("未成功加载任何可用数据集，请检查 data.sources 路径配置。")
    finally:
        _remove_path(loading_cache_dir)

    console.print(summary_table)
    merged = concatenate_datasets(loaded_datasets).shuffle(seed=int(seed))
    if len(merged) < 2:
        raise ValueError(f"有效样本总数过少 ({len(merged)})，无法切分训练与验证集。")

    val_size = int(len(merged) * float(val_split_ratio))
    if max_val_samples is not None and int(max_val_samples) > 0:
        val_size = min(val_size, int(max_val_samples))
    val_size = max(1, min(val_size, len(merged) - 1))
    split_dataset = merged.train_test_split(test_size=val_size, seed=int(seed))
    prepared = DatasetDict({"train": split_dataset["train"], "val": split_dataset["test"]})
    _save_dataset_dict(prepared, output_path)

    console.print(
        f"[bold green][Prepare][/bold green] 已保存 {task.upper()} 预处理数据集至: {output_path} "
        f"(train={len(prepared['train']):,}, val={len(prepared['val']):,})"
    )
    return _prepared_summary(prepared, output_path, task, overwritten=True)

