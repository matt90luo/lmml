"""独立 LLM 训练数据准备管线：多源加载、抽样、标准化、分词与 Arrow 落盘。"""

from __future__ import annotations

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

PREPARE_TASKS = ("sft", "dpo", "ppo")


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
    for key in ("prompt", "messages", "query", "instruction"):
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
    """编码 PPO 生成 prompt，并追加 assistant 起始标记。"""
    input_ids: list[int] = []
    for message in messages:
        text = f"{chat_start_token}{message['role']}\n{message['content']}{chat_end_token}\n"
        input_ids.extend(tokenizer.encode(text, add_special_tokens=False))
    assistant_prefix = f"{chat_start_token}assistant\n"
    input_ids.extend(tokenizer.encode(assistant_prefix, add_special_tokens=False))
    input_ids = input_ids[:max_length]
    if len(input_ids) < 2:
        return None
    return {"input_ids": input_ids, "prompt_length": len(input_ids), "length": len(input_ids)}


def tokenize_ppo_batch(
    batch: dict[str, list[Any]],
    *,
    tokenizer: Any,
    max_length: int = 2048,
    chat_start_token: str = DEFAULT_CHAT_START_TOKEN,
    chat_end_token: str = DEFAULT_CHAT_END_TOKEN,
) -> dict[str, list[Any]]:
    """批量编码 PPO prompt。"""
    output: dict[str, list[Any]] = {"input_ids": [], "prompt_length": [], "length": []}
    for messages in batch.get("prompt", []):
        encoded = encode_chatml_prompt(
            messages=messages,
            tokenizer=tokenizer,
            max_length=max_length,
            chat_start_token=chat_start_token,
            chat_end_token=chat_end_token,
        )
        if encoded is None:
            continue
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
        "sft": set(),
        "dpo": {"chosen", "rejected"},
        "ppo": {"prompt", "messages", "query", "instruction"},
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
    if task == "sft":
        normalize_fn = normalize_sft_batch
        tokenize_fn = tokenize_sft_batch
        tokenize_fn_kwargs.update(ignore_index=int(ignore_index), drop_no_target=bool(drop_no_target))
    elif task == "dpo":
        normalize_fn = normalize_dpo_batch
        tokenize_fn = tokenize_dpo_batch
        tokenize_fn_kwargs.update(ignore_index=int(ignore_index), drop_no_target=bool(drop_no_target))
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
                dataset = dataset.map(
                    normalize_fn,
                    batched=True,
                    batch_size=1000,
                    remove_columns=dataset.column_names,
                    num_proc=num_proc,
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

