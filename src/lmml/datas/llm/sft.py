"""LLM SFT 数据读取与训练管线：Arrow DatasetDict 加载、按长度分桶采样与动态对齐 Padding。"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from datasets import load_from_disk
from rich.console import Console
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoTokenizer, PreTrainedTokenizerBase

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

console = Console()

DEFAULT_CHAT_START_TOKEN = "<|im_start|>"
DEFAULT_CHAT_END_TOKEN = "<|im_end|>"
DEFAULT_IGNORE_INDEX = -100

_USER_ROLES = {"human", "user"}
_SYSTEM_ROLES = {"system", "developer"}
_TOOL_ROLES = {"tool", "function", "observation", "tool_response", "ipython"}
_ASSISTANT_ROLES = {"assistant", "gpt", "model", "bot"}
_REASONING_KEYS = ("reasoning_content", "reasoning", "chain_of_thought", "cot")
_TOOL_VALUE_KEYS = ("tools", "functions", "available_tools", "tool_definitions")
_PROMPT_PAIR_KEYS = (
    ("instruction", "output"),
    ("question", "response"),
    ("question", "answer"),
    ("problem", "solution"),
    ("prompt", "response"),
)


def resolve_num_proc(num_proc: int | str | None) -> int | None:
    """解析多进程并行数：支持 'auto'、正整数或 None/1（单进程）。"""
    if num_proc is None:
        return None
    if isinstance(num_proc, str):
        val = num_proc.strip().lower()
        if val in ("auto", "default"):
            cpu_cnt = os.cpu_count() or 2
            resolved = max(1, min(cpu_cnt, 8))
            return resolved if resolved > 1 else None
        if val in ("none", "null", "0", "1"):
            return None
        num_proc = int(val)
    n = int(num_proc)
    return n if n > 1 else None


def build_sft_tokenizer(
    model_name_or_path: str,
    chat_end_token: str = DEFAULT_CHAT_END_TOKEN,
    trust_remote_code: bool = True,
) -> PreTrainedTokenizerBase:
    """加载并标准化 ChatML 分词器；MiniMind 路径缺失时回退到内置词表。"""
    tokenizer_source = str(model_name_or_path)
    if not Path(tokenizer_source).exists() and "minimind" in tokenizer_source.lower():
        bundled = files("lmml").joinpath("assets", "minimind")
        with as_file(bundled) as bundled_path:
            tokenizer = AutoTokenizer.from_pretrained(
                str(bundled_path), trust_remote_code=trust_remote_code
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=trust_remote_code,
        )
    im_end_id = tokenizer.convert_tokens_to_ids(chat_end_token)
    if isinstance(im_end_id, int) and im_end_id != getattr(tokenizer, "unk_token_id", None):
        tokenizer.eos_token = chat_end_token
        tokenizer.eos_token_id = im_end_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _stringify(value: Any) -> str:
    """将字符串、JSON 对象或数组稳定转换为可写入 ChatML 文本的内容。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value).strip()


def _first_text(item: dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        text = _stringify(item.get(key))
        if text:
            return text
    return ""


def _coerce_message_list(value: Any) -> list[dict[str, Any]]:
    """兼容 messages/conversations 为列表或 JSON 字符串的数据源。"""
    if isinstance(value, list):
        return [message for message in value if isinstance(message, dict)]
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(decoded, list):
            return [message for message in decoded if isinstance(message, dict)]
    return []


def _format_tool_calls(value: Any) -> str:
    """将 Hermes/OpenAI 风格的 tool_calls 统一为 Qwen ChatML 的 tool_call 标签。"""
    if value is None:
        return ""
    calls = value if isinstance(value, list) else [value]
    blocks: list[str] = []
    for call in calls:
        normalized = call
        if isinstance(call, dict) and isinstance(call.get("function"), dict):
            function = call["function"]
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            normalized = {"name": function.get("name", ""), "arguments": arguments}
        text = _stringify(normalized)
        if not text or "<tool_call>" in text:
            continue
        blocks.append(f"<tool_call>\n{text}\n</tool_call>")
    return "\n".join(blocks)


def _normalize_role(raw_role: Any) -> str:
    role = str(raw_role or "user").strip().lower()
    if role in _USER_ROLES:
        return "user"
    if role in _SYSTEM_ROLES:
        return "system"
    if role in _TOOL_ROLES:
        return "tool"
    if role in _ASSISTANT_ROLES or role in {"assistant", "gpt"}:
        return "assistant"
    return "assistant"


def _normalize_message(
    message: dict[str, Any],
    *,
    wrap_reasoning: bool = False,
) -> dict[str, str] | None:
    role = _normalize_role(message.get("role") or message.get("from"))
    content = _stringify(
        message.get("content") if message.get("content") is not None else message.get("value")
    )
    reasoning = _first_text(message, _REASONING_KEYS)
    if role == "assistant" and reasoning and reasoning not in content:
        if wrap_reasoning:
            content = f"<think>\n{reasoning}\n</think>\n\n{content}".strip()
        else:
            content = f"{reasoning}\n\n{content}".strip()

    tool_calls = _format_tool_calls(message.get("tool_calls") or message.get("function_call"))
    if role == "assistant" and tool_calls and "<tool_call>" not in content:
        content = f"{content}\n{tool_calls}".strip()

    if role == "tool" and content and "<tool_response>" not in content:
        content = f"<tool_response>\n{content}\n</tool_response>"

    return {"role": role, "content": content} if content else None


def _format_tools_block(tools: Any) -> str:
    tools_text = _stringify(tools)
    if not tools_text:
        return ""
    return (
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{tools_text}\n</tools>\n\n"
        "For each function call, return a JSON object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>"
    )


def _inject_system_context(
    messages: list[dict[str, str]],
    *,
    system_prompt: str = "",
    tools: Any = None,
) -> list[dict[str, str]]:
    """补齐或扩展 system 消息，同时避免重复注入 source 已内置的工具定义。"""
    system_prompt = system_prompt.strip()
    tools_block = _format_tools_block(tools)
    if not system_prompt and not tools_block:
        return messages

    output = [dict(message) for message in messages]
    for message in output:
        if message["role"] != "system":
            continue
        additions: list[str] = []
        if system_prompt and system_prompt not in message["content"]:
            additions.append(system_prompt)
        if tools_block and "<tools>" not in message["content"]:
            additions.append(tools_block)
        if additions:
            message["content"] = "\n\n".join((message["content"], *additions)).strip()
        return output
    addition = "\n\n".join(text for text in (system_prompt, tools_block) if text)
    output.insert(0, {"role": "system", "content": addition})
    return output


def normalize_item_format(
    item: dict[str, Any],
    *,
    wrap_reasoning: bool = False,
) -> list[dict[str, str]]:
    """将不同开源数据集清洗为兼容 ChatML、CoT 与工具调用的统一消息格式。"""
    raw_msgs = (
        _coerce_message_list(item.get("messages"))
        or _coerce_message_list(item.get("conversations"))
        or _coerce_message_list(item.get("dialogue"))
    )
    messages = [
        message
        for raw in raw_msgs
        if (message := _normalize_message(raw, wrap_reasoning=wrap_reasoning)) is not None
    ]
    if any(message["role"] == "assistant" for message in messages):
        system_prompt = _first_text(item, ("system_prompt", "system"))
        tools = next((item.get(key) for key in _TOOL_VALUE_KEYS if item.get(key)), None)
        if not tools:
            tools = next(
                (raw_message.get("tools") for raw_message in raw_msgs if raw_message.get("tools")),
                None,
            )
        return _inject_system_context(messages, system_prompt=system_prompt, tools=tools)

    prompt = ""
    response = ""
    for prompt_key, response_key in _PROMPT_PAIR_KEYS:
        prompt = _stringify(item.get(prompt_key))
        response = _stringify(item.get(response_key))
        if prompt and response:
            break
    if not prompt or not response:
        return []

    extra_input = _stringify(item.get("input"))
    if extra_input:
        prompt = f"{prompt}\n{extra_input}"
    reasoning = _first_text(item, _REASONING_KEYS)
    if reasoning and reasoning not in response:
        if wrap_reasoning:
            response = f"<think>\n{reasoning}\n</think>\n\n{response}"
        else:
            response = f"{reasoning}\n\n{response}"

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    system_prompt = _first_text(item, ("system_prompt", "system"))
    tools = next((item.get(key) for key in _TOOL_VALUE_KEYS if item.get(key)), None)
    return _inject_system_context(messages, system_prompt=system_prompt, tools=tools)


def normalize_sft_batch(
    batch: dict[str, list[Any]],
    *,
    wrap_reasoning: bool = False,
    add_system_ratio: float = 0.0,
    empty_think_remove_ratio: float = 0.0,
    seed: int = 42,
) -> dict[str, list[list[dict[str, str]]]]:
    """顶层批量标准化纯函数（供 datasets.Dataset.map(batched=True, num_proc=...) 调用）。

    在单趟批量转换中同时剔除空对话与无 assistant 回复的脏样本，省去二次 filter 扫描。
    """
    keys = list(batch.keys())
    if not keys:
        return {"messages": []}
    n_rows = len(batch[keys[0]])
    out_messages: list[list[dict[str, str]]] = []
    for i in range(n_rows):
        row = {k: batch[k][i] for k in keys}
        msgs = normalize_item_format(row, wrap_reasoning=wrap_reasoning)
        if msgs:
            out_messages.append(msgs)
    return {"messages": out_messages}


def _has_tool_messages(messages: list[dict[str, str]]) -> bool:
    return any(message.get("role") in {"tool", "function", "observation"} for message in messages)


def _template_tools(messages: list[dict[str, Any]]) -> Any:
    for message in messages:
        tools = message.get("tools")
        if tools:
            if isinstance(tools, str):
                try:
                    return json.loads(tools)
                except json.JSONDecodeError:
                    return tools
            return tools
    return None


def _template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        rendered = {"role": message["role"], "content": message.get("content", "")}
        if message.get("tool_calls"):
            tool_calls = message["tool_calls"]
            if isinstance(tool_calls, str):
                try:
                    tool_calls = json.loads(tool_calls)
                except json.JSONDecodeError:
                    pass
            rendered["tool_calls"] = tool_calls
        out.append(rendered)
    return out


def _encode_with_chat_template(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    chat_start_token: str,
    chat_end_token: str,
    ignore_index: int,
    drop_no_target: bool,
) -> dict[str, Any] | None:
    rendered: list[str] = []
    assistant_ends: list[int] = []
    tools = _template_tools(messages)
    for index, message in enumerate(messages):
        current = _template_messages(messages[: index + 1])
        try:
            text = tokenizer.apply_chat_template(
                current,
                tokenize=False,
                add_generation_prompt=False,
                tools=tools,
            )
        except (TypeError, ValueError, KeyError):
            return None
        rendered.append(text)
        if message.get("role") == "assistant":
            assistant_ends.append(len(tokenizer.encode(text, add_special_tokens=False)))
    final_ids = tokenizer.encode(rendered[-1], add_special_tokens=False)
    if not final_ids:
        return None
    input_ids = list(final_ids[:max_length])
    labels = [ignore_index] * len(input_ids)
    for index, message in enumerate(messages):
        current_ids = tokenizer.encode(rendered[index], add_special_tokens=False)
        if message.get("role") == "assistant":
            current_text = rendered[index]
            assistant_header = f"{chat_start_token}assistant\n"
            header_pos = current_text.rfind(assistant_header)
            if header_pos >= 0:
                header_text = current_text[:header_pos] + assistant_header
                header_len = len(tokenizer.encode(header_text, add_special_tokens=False))
                start = max(0, header_len - 1)
                end = min(len(current_ids), len(input_ids))
                if end > start:
                    labels[start:end] = input_ids[start:end]
    if len(input_ids) < 2 or not any(label != ignore_index for label in labels[1:]):
        if drop_no_target:
            return None
        labels[-1] = input_ids[-1]
    return {"input_ids": input_ids, "labels": labels, "length": len(input_ids)}


def encode_chatml_messages(
    messages: list[dict[str, str]],
    tokenizer: Any,
    max_length: int = 2048,
    chat_start_token: str = DEFAULT_CHAT_START_TOKEN,
    chat_end_token: str = DEFAULT_CHAT_END_TOKEN,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    drop_no_target: bool = True,
    use_chat_template: bool = True,
) -> dict[str, Any] | None:
    """将标准化后的多轮对话编码为带有 Prompt Masking 的 input_ids 与 labels。"""
    if (
        use_chat_template
        and getattr(tokenizer, "chat_template", None)
        and (_has_tool_messages(messages) or "<think>" in ", ".join(m.get("content", "") for m in messages))
    ):
        return _encode_with_chat_template(
            messages=messages,
            tokenizer=tokenizer,
            max_length=max_length,
            chat_start_token=chat_start_token,
            chat_end_token=chat_end_token,
            ignore_index=ignore_index,
            drop_no_target=drop_no_target,
        )
    input_ids: list[int] = []
    labels: list[int] = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        # Qwen 的 ChatML 工具响应使用 user 角色包裹 <tool_response>，而不是新增 tool 角色。
        rendered_role = "user" if role == "tool" else role

        # 分别编码前缀与内容，避免 BPE 在边界处因换行符/空格合并导致 prefix_len 偏移
        prefix_text = f"{chat_start_token}{rendered_role}\n"
        body_text = f"{content}{chat_end_token}\n"

        prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
        body_tokens = tokenizer.encode(body_text, add_special_tokens=False)
        turn_tokens = prefix_tokens + body_tokens

        input_ids.extend(turn_tokens)
        if role == "assistant":
            labels.extend([ignore_index] * len(prefix_tokens) + body_tokens)
        else:
            labels.extend([ignore_index] * len(turn_tokens))

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    if len(input_ids) < 2:
        return None

    # 检查位移后的有效监督 token (labels[1:]) 是否全为 ignore_index
    has_valid_target = any(lbl != ignore_index for lbl in labels[1:])
    if not has_valid_target:
        if drop_no_target:
            return None
        labels[-1] = input_ids[-1]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "length": len(input_ids),
    }


def tokenize_sft_batch(
    batch: dict[str, list[Any]],
    *,
    tokenizer: Any,
    max_length: int = 2048,
    chat_start_token: str = DEFAULT_CHAT_START_TOKEN,
    chat_end_token: str = DEFAULT_CHAT_END_TOKEN,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    drop_no_target: bool = True,
) -> dict[str, list[Any]]:
    """顶层批量分词纯函数（支持 datasets.Dataset.map(batched=True, num_proc=...)）。"""
    out_input_ids: list[list[int]] = []
    out_labels: list[list[int]] = []
    out_lengths: list[int] = []

    for messages in batch.get("messages", []):
        encoded = encode_chatml_messages(
            messages=messages,
            tokenizer=tokenizer,
            max_length=max_length,
            chat_start_token=chat_start_token,
            chat_end_token=chat_end_token,
            ignore_index=ignore_index,
            drop_no_target=drop_no_target,
        )
        if encoded is not None:
            out_input_ids.append(encoded["input_ids"])
            out_labels.append(encoded["labels"])
            out_lengths.append(encoded["length"])

    return {
        "input_ids": out_input_ids,
        "labels": out_labels,
        "length": out_lengths,
    }


class StandardSFTDataset(Dataset):
    """兼容预分词 Arrow 数据集与按需实时分词的 SFT 数据集封装。"""

    def __init__(
        self,
        raw_data: Any,
        tokenizer: Any,
        max_length: int = 2048,
        chat_start_token: str = DEFAULT_CHAT_START_TOKEN,
        chat_end_token: str = DEFAULT_CHAT_END_TOKEN,
        ignore_index: int = DEFAULT_IGNORE_INDEX,
        pretokenized: bool = False,
    ):
        self.data = raw_data
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.chat_start_token = chat_start_token
        self.chat_end_token = chat_end_token
        self.ignore_index = int(ignore_index)
        self.pretokenized = bool(pretokenized)
        self._lengths: list[int] | None = None
        if self.pretokenized and hasattr(self.data, "column_names") and "length" in self.data.column_names:
            self._lengths = list(self.data["length"])

    @property
    def lengths(self) -> list[int] | None:
        return self._lengths

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        item = self.data[int(idx)]
        if self.pretokenized or ("input_ids" in item and "labels" in item):
            return {
                "input_ids": list(item["input_ids"]),
                "labels": list(item["labels"]),
            }

        encoded = encode_chatml_messages(
            messages=item["messages"],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            chat_start_token=self.chat_start_token,
            chat_end_token=self.chat_end_token,
            ignore_index=self.ignore_index,
            drop_no_target=False,
        )
        if encoded is None:
            return {"input_ids": [], "labels": []}
        return {
            "input_ids": encoded["input_ids"],
            "labels": encoded["labels"],
        }


class DynamicPadCollator:
    """向量化动态 Padding Collator：预分配连续张量并支持对齐到 8 的倍数以加速 TensorCore。"""

    def __init__(
        self,
        pad_token_id: int,
        ignore_index: int = DEFAULT_IGNORE_INDEX,
        pad_to_multiple_of: int | None = 8,
        max_length: int | None = None,
    ):
        self.pad_token_id = int(pad_token_id)
        self.ignore_index = int(ignore_index)
        self.pad_to_multiple_of = int(pad_to_multiple_of) if pad_to_multiple_of else None
        self.max_length = int(max_length) if max_length else None

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        valid_batch = [item for item in batch if len(item.get("input_ids", ())) > 0]
        if not valid_batch:
            raise ValueError("当前 Batch 中所有样本均为空，请检查数据集。")

        bsz = len(valid_batch)
        max_batch_len = max(len(item["input_ids"]) for item in valid_batch)
        if self.pad_to_multiple_of and self.pad_to_multiple_of > 1:
            rem = max_batch_len % self.pad_to_multiple_of
            if rem != 0:
                max_batch_len += self.pad_to_multiple_of - rem
        if self.max_length is not None:
            max_batch_len = min(max_batch_len, self.max_length)

        input_ids = torch.full((bsz, max_batch_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_batch_len), dtype=torch.long)
        labels = torch.full((bsz, max_batch_len), self.ignore_index, dtype=torch.long)

        for i, item in enumerate(valid_batch):
            ids = item["input_ids"][:max_batch_len]
            lbls = item["labels"][:max_batch_len]
            seq_len = len(ids)
            input_ids[i, :seq_len] = torch.as_tensor(ids, dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = torch.as_tensor(lbls, dtype=torch.long)

        # 兜底防护：确保整个 batch 的 labels[:, 1:] 至少包含 1 个有效监督 token，彻底杜绝 NaN loss
        if max_batch_len > 1 and not torch.any(labels[:, 1:] != self.ignore_index):
            first_len = min(len(valid_batch[0]["input_ids"]), max_batch_len)
            pos = max(1, first_len - 1)
            labels[0, pos] = input_ids[0, pos]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class LengthGroupedSampler(Sampler[int]):
    """按序列长度分桶的随机采样器（Megabatch Sort + Mini-batch Shuffle）。

    - 将长度相近的样本聚合在同一微批次内，大幅减少混合长短数据集的 Padding 算力浪费；
    - 将全局最长微批次排在首个 step，以便在训练启动最初几秒内暴露显存峰值（防中途 OOM）；
    - 对其余微批次顺序做随机打乱，保证梯度期望无偏。
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        mega_batch_mult: int = 50,
        seed: int = 42,
    ):
        self.lengths = [int(x) for x in lengths]
        self.batch_size = max(1, int(batch_size))
        self.mega_batch_mult = max(1, int(mega_batch_mult))
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.lengths)

    def __iter__(self) -> Iterator[int]:
        n = len(self.lengths)
        if n == 0:
            return iter(())

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(n, generator=g).tolist()

        mega_size = min(n, self.batch_size * self.mega_batch_mult)
        full_batches: list[list[int]] = []
        tail_batch: list[int] = []

        for start in range(0, n, mega_size):
            mega = indices[start : start + mega_size]
            mega.sort(key=self.lengths.__getitem__, reverse=True)
            for b_start in range(0, len(mega), self.batch_size):
                chunk = mega[b_start : b_start + self.batch_size]
                if len(chunk) == self.batch_size:
                    full_batches.append(chunk)
                else:
                    tail_batch.extend(chunk)

        if len(tail_batch) >= self.batch_size:
            for b_start in range(0, len(tail_batch), self.batch_size):
                chunk = tail_batch[b_start : b_start + self.batch_size]
                if len(chunk) == self.batch_size:
                    full_batches.append(chunk)
                else:
                    tail_batch = chunk
                    break
            else:
                tail_batch = []

        if full_batches:
            # 将包含最长序列的 mini-batch 放在第 0 个 batch，其余随机打乱
            max_idx = max(
                range(len(full_batches)),
                key=lambda i: max(self.lengths[idx] for idx in full_batches[i]),
            )
            longest_batch = full_batches.pop(max_idx)
            if full_batches:
                perm = torch.randperm(len(full_batches), generator=g).tolist()
                full_batches = [full_batches[p] for p in perm]
            ordered = [longest_batch, *full_batches]
        else:
            ordered = []

        flat = [idx for batch in ordered for idx in batch]
        flat.extend(tail_batch)
        return iter(flat)


class PromptPredictDataset(Dataset):
    """用于 lmml predict 子命令的对话生成推理数据集。"""

    def __init__(
        self,
        prompts: list[str],
        tokenizer: Any,
        max_length: int = 2048,
        chat_start_token: str = DEFAULT_CHAT_START_TOKEN,
        chat_end_token: str = DEFAULT_CHAT_END_TOKEN,
    ):
        self.prompts = list(prompts)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.chat_start_token = chat_start_token
        self.chat_end_token = chat_end_token

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        prompt = self.prompts[int(idx)]
        text = (
            f"{self.chat_start_token}user\n{prompt}{self.chat_end_token}\n"
            f"{self.chat_start_token}assistant\n"
        )
        ids = self.tokenizer.encode(text, add_special_tokens=False)[: self.max_length]
        return {"prompt": prompt, "input_ids": ids}


def collate_prompt_predict(
    batch: list[dict[str, Any]],
    pad_token_id: int,
) -> dict[str, Any]:
    """生成推理专用左填充 Collator。"""
    bsz = len(batch)
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = torch.full((bsz, max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    prompts: list[str] = []
    for i, item in enumerate(batch):
        ids = item["input_ids"]
        seq_len = len(ids)
        # 生成推理采用左填充 (left-padding)
        input_ids[i, max_len - seq_len :] = torch.as_tensor(ids, dtype=torch.long)
        attention_mask[i, max_len - seq_len :] = 1
        prompts.append(item["prompt"])
    return {
        "prompts": prompts,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


class UnifiedSFTDataModule(pl.LightningDataModule):
    """从独立预处理产物读取 SFT 数据，并提供长度分桶采样与动态 Padding。"""

    def __init__(
        self,
        prepared_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 8,
        val_batch_size: int = 8,
        max_length: int = 2048,
        num_workers: int = 4,
        chat_start_token: str = DEFAULT_CHAT_START_TOKEN,
        chat_end_token: str = DEFAULT_CHAT_END_TOKEN,
        ignore_index: int = DEFAULT_IGNORE_INDEX,
        group_by_length: bool = True,
        mega_batch_mult: int = 50,
        pad_to_multiple_of: int | None = 8,
        predict_prompts: list[str] | None = None,
        predict_batch_size: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        if tokenizer is None:
            raise ValueError("必须提供有效的 tokenizer 实例。")
        self.prepared_path = Path(prepared_path)
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.val_batch_size = int(val_batch_size)
        self.max_length = int(max_length)
        self.num_workers = max(0, int(num_workers))
        self.chat_start_token = str(chat_start_token)
        self.chat_end_token = str(chat_end_token)
        self.ignore_index = int(ignore_index)
        self.group_by_length = bool(group_by_length)
        self.mega_batch_mult = int(mega_batch_mult)
        self.pad_to_multiple_of = int(pad_to_multiple_of) if pad_to_multiple_of else None
        self.predict_prompts = list(predict_prompts) if predict_prompts else [
            "Explain the difference between LoRA and full fine-tuning in 2 sentences."
        ]
        self.predict_batch_size = max(1, int(predict_batch_size))
        self.seed = int(seed)

        self.train_dataset: StandardSFTDataset | None = None
        self.val_dataset: StandardSFTDataset | None = None
        self.collator = DynamicPadCollator(
            pad_token_id=int(self.tokenizer.pad_token_id),
            ignore_index=self.ignore_index,
            pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_length,
        )

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return
        if stage == "predict":
            return
        if not self.prepared_path.exists() or not (self.prepared_path / "dataset_dict.json").exists():
            raise FileNotFoundError(
                f"未找到预处理数据集: {self.prepared_path}。"
                "请先运行 `lmml data experiment=...` 生成训练数据。"
            )
        split_ds = load_from_disk(str(self.prepared_path))
        try:
            train_raw = split_ds["train"]
            val_raw = split_ds["val"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"预处理数据集必须包含 train/val 切分: {self.prepared_path}"
            ) from exc

        console.print(f"[bold cyan][DataModule][/bold cyan] 加载预处理数据集: {self.prepared_path}")
        self.train_dataset = StandardSFTDataset(
            train_raw,
            self.tokenizer,
            max_length=self.max_length,
            chat_start_token=self.chat_start_token,
            chat_end_token=self.chat_end_token,
            ignore_index=self.ignore_index,
            pretokenized=True,
        )
        self.val_dataset = StandardSFTDataset(
            val_raw,
            self.tokenizer,
            max_length=self.max_length,
            chat_start_token=self.chat_start_token,
            chat_end_token=self.chat_end_token,
            ignore_index=self.ignore_index,
            pretokenized=True,
        )

    def _loader_perf_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "num_workers": self.num_workers,
            "pin_memory": torch.cuda.is_available(),
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 2
        return kwargs

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        lengths = self.train_dataset.lengths
        if self.group_by_length and lengths is not None and len(lengths) == len(self.train_dataset):
            sampler = LengthGroupedSampler(
                lengths=lengths,
                batch_size=self.batch_size,
                mega_batch_mult=self.mega_batch_mult,
                seed=self.seed,
            )
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                shuffle=False,
                collate_fn=self.collator,
                **self._loader_perf_kwargs(),
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            **self._loader_perf_kwargs(),
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self.collator,
            **self._loader_perf_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def predict_dataloader(self) -> DataLoader:
        pred_ds = PromptPredictDataset(
            prompts=self.predict_prompts,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            chat_start_token=self.chat_start_token,
            chat_end_token=self.chat_end_token,
        )
        pad_id = int(self.tokenizer.pad_token_id)
        return DataLoader(
            pred_ds,
            batch_size=self.predict_batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_prompt_predict(b, pad_token_id=pad_id),
        )
