"""高效 LLM SFT 数据管线：多进程批量标准化与预分词、本地 SSD Arrow 缓存、按长度分桶采样与动态对齐 Padding。"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoTokenizer, PreTrainedTokenizerBase

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

console = Console()

DEFAULT_CHAT_START_TOKEN = "<|im_start|>"
DEFAULT_CHAT_END_TOKEN = "<|im_end|>"
DEFAULT_IGNORE_INDEX = -100


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
    """加载并标准化 Qwen / ChatML 分词器（同步 <|im_end|> 为 EOS，右侧 Padding）。"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
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


def normalize_item_format(item: dict[str, Any]) -> list[dict[str, str]]:
    """将不同开源数据集的单条样本统一转换为 [{'role': '...', 'content': '...'}]。"""
    raw_msgs = (
        item.get("messages")
        if isinstance(item.get("messages"), list)
        else (item.get("conversations") if isinstance(item.get("conversations"), list) else None)
    )
    if raw_msgs:
        mapped: list[dict[str, str]] = []
        for m in raw_msgs:
            if not isinstance(m, dict):
                continue
            raw_role = m.get("role") or m.get("from") or "user"
            if raw_role in ("human", "user"):
                role = "user"
            elif raw_role == "system":
                role = "system"
            else:
                role = "assistant"
            content = m.get("content") if m.get("content") is not None else m.get("value", "")
            if content is not None and str(content).strip() != "":
                mapped.append({"role": role, "content": str(content)})
        if any(m["role"] == "assistant" for m in mapped):
            return mapped

    if item.get("instruction") is not None and item.get("output") is not None:
        inst = str(item["instruction"]).strip()
        out = str(item["output"]).strip()
        if inst and out:
            prompt = inst
            inp = item.get("input")
            if inp is not None and str(inp).strip() != "":
                prompt = f"{prompt}\n{inp}"
            return [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": out},
            ]

    if item.get("question") is not None and item.get("response") is not None:
        q = str(item["question"]).strip()
        r = str(item["response"]).strip()
        if q and r:
            return [
                {"role": "user", "content": q},
                {"role": "assistant", "content": r},
            ]

    if item.get("problem") is not None and item.get("solution") is not None:
        p = str(item["problem"]).strip()
        s = str(item["solution"]).strip()
        if p and s:
            return [
                {"role": "user", "content": p},
                {"role": "assistant", "content": s},
            ]

    return []


def normalize_sft_batch(batch: dict[str, list[Any]]) -> dict[str, list[list[dict[str, str]]]]:
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
        msgs = normalize_item_format(row)
        if msgs:
            out_messages.append(msgs)
    return {"messages": out_messages}


def encode_chatml_messages(
    messages: list[dict[str, str]],
    tokenizer: Any,
    max_length: int = 2048,
    chat_start_token: str = DEFAULT_CHAT_START_TOKEN,
    chat_end_token: str = DEFAULT_CHAT_END_TOKEN,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    drop_no_target: bool = True,
) -> dict[str, Any] | None:
    """将标准化后的多轮对话编码为带有 Prompt Masking 的 input_ids 与 labels。"""
    input_ids: list[int] = []
    labels: list[int] = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # 分别编码前缀与内容，避免 BPE 在边界处因换行符/空格合并导致 prefix_len 偏移
        prefix_text = f"{chat_start_token}{role}\n"
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
    """统一多源 SFT 数据模块：支持多进程批量清洗与预分词、本地 SSD Arrow 缓存及按长度分桶采样。"""

    def __init__(
        self,
        data_configs: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 8,
        val_batch_size: int = 8,
        max_length: int = 2048,
        num_workers: int = 4,
        preprocessing_num_proc: int | str | None = "auto",
        val_split_ratio: float = 0.02,
        max_val_samples: int | None = 1500,
        chat_start_token: str = DEFAULT_CHAT_START_TOKEN,
        chat_end_token: str = DEFAULT_CHAT_END_TOKEN,
        ignore_index: int = DEFAULT_IGNORE_INDEX,
        pretokenize: bool = True,
        group_by_length: bool = True,
        mega_batch_mult: int = 50,
        pad_to_multiple_of: int | None = 8,
        drop_no_target: bool = True,
        cache_dir: str | None = None,
        predict_prompts: list[str] | None = None,
        predict_batch_size: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        if tokenizer is None:
            raise ValueError("必须提供有效的 tokenizer 实例。")
        self.data_configs = [dict(c) for c in data_configs]
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.val_batch_size = int(val_batch_size)
        self.max_length = int(max_length)
        self.num_workers = max(0, int(num_workers))
        self.preprocessing_num_proc = resolve_num_proc(preprocessing_num_proc)
        self.val_split_ratio = float(val_split_ratio)
        self.max_val_samples = int(max_val_samples) if max_val_samples is not None else None
        self.chat_start_token = str(chat_start_token)
        self.chat_end_token = str(chat_end_token)
        self.ignore_index = int(ignore_index)
        self.pretokenize = bool(pretokenize)
        self.group_by_length = bool(group_by_length)
        self.mega_batch_mult = int(mega_batch_mult)
        self.pad_to_multiple_of = int(pad_to_multiple_of) if pad_to_multiple_of else None
        self.drop_no_target = bool(drop_no_target)
        self.cache_dir = str(cache_dir) if cache_dir else None
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

    def _compute_cache_path(self) -> Path | None:
        if not self.cache_dir:
            return None
        meta = {
            "sources": self.data_configs,
            "max_length": self.max_length,
            "val_split_ratio": self.val_split_ratio,
            "max_val_samples": self.max_val_samples,
            "chat_start_token": self.chat_start_token,
            "chat_end_token": self.chat_end_token,
            "ignore_index": self.ignore_index,
            "pretokenize": self.pretokenize,
            "drop_no_target": self.drop_no_target,
            "seed": self.seed,
            "tokenizer": getattr(self.tokenizer, "name_or_path", "unknown"),
            "vocab_size": len(self.tokenizer),
        }
        digest = hashlib.sha256(json.dumps(meta, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
        return Path(self.cache_dir) / f"sft_prepared_{digest}"

    @staticmethod
    def _load_single_source(path: str) -> HFDataset:
        if os.path.isdir(path) and (
            os.path.exists(os.path.join(path, "dataset_info.json"))
            or os.path.exists(os.path.join(path, "dataset_dict.json"))
        ):
            ds = load_from_disk(path)
        elif os.path.isdir(path):
            ds = load_dataset(path)
        elif path.endswith((".json", ".jsonl")):
            ds = load_dataset("json", data_files=path, split="train")
        elif path.endswith(".parquet"):
            ds = load_dataset("parquet", data_files=path, split="train")
        else:
            ds = load_dataset(path, split="train")

        if isinstance(ds, (dict, DatasetDict)):
            ds = ds.get("train", next(iter(ds.values())))
        return ds

    def _apply_subsampling(self, ds: HFDataset, cfg: dict[str, Any]) -> tuple[HFDataset, int, int]:
        orig_len = len(ds)
        target_len = orig_len

        sample_ratio = cfg.get("sample_ratio")
        max_samples = cfg.get("max_samples")

        if sample_ratio is not None and 0.0 < float(sample_ratio) < 1.0:
            target_len = int(orig_len * float(sample_ratio))
        if max_samples is not None and int(max_samples) > 0:
            target_len = min(target_len, int(max_samples))

        if target_len < orig_len:
            ds = ds.shuffle(seed=self.seed).select(range(target_len))
        return ds, orig_len, len(ds)

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        cache_path = self._compute_cache_path()
        if cache_path is not None and (cache_path / "dataset_dict.json").exists():
            console.print(f"[bold cyan][DataModule][/bold cyan] 命中本地 Arrow 预处理缓存: {cache_path}")
            split_ds = load_from_disk(str(cache_path))
            train_raw = split_ds["train"]
            val_raw = split_ds["val"]
            self.train_dataset = StandardSFTDataset(
                train_raw,
                self.tokenizer,
                max_length=self.max_length,
                chat_start_token=self.chat_start_token,
                chat_end_token=self.chat_end_token,
                ignore_index=self.ignore_index,
                pretokenized=self.pretokenize,
            )
            self.val_dataset = StandardSFTDataset(
                val_raw,
                self.tokenizer,
                max_length=self.max_length,
                chat_start_token=self.chat_start_token,
                chat_end_token=self.chat_end_token,
                ignore_index=self.ignore_index,
                pretokenized=self.pretokenize,
            )
            return

        summary_table = Table(title="SFT 数据源抽样与清洗汇总", show_header=True, header_style="bold cyan")
        summary_table.add_column("数据集", style="white")
        summary_table.add_column("原始条数", justify="right")
        summary_table.add_column("抽样后条数", justify="right")
        summary_table.add_column("有效样本数", justify="right", style="green")

        loaded_datasets: list[HFDataset] = []
        for cfg in self.data_configs:
            path = str(cfg["path"])
            ds_name = os.path.basename(path.rstrip("/")) or path
            if not os.path.exists(path):
                console.print(f"[yellow][Warning] 数据路径不存在，已跳过: {path}[/yellow]")
                continue
            try:
                ds = self._load_single_source(path)
                ds, orig_len, sampled_len = self._apply_subsampling(ds, cfg)

                ds = ds.map(
                    normalize_sft_batch,
                    batched=True,
                    batch_size=1000,
                    remove_columns=ds.column_names,
                    num_proc=self.preprocessing_num_proc,
                    desc=f"Normalizing {ds_name}",
                )

                if self.pretokenize:
                    ds = ds.map(
                        tokenize_sft_batch,
                        batched=True,
                        batch_size=1000,
                        remove_columns=ds.column_names,
                        num_proc=self.preprocessing_num_proc,
                        fn_kwargs={
                            "tokenizer": self.tokenizer,
                            "max_length": self.max_length,
                            "chat_start_token": self.chat_start_token,
                            "chat_end_token": self.chat_end_token,
                            "ignore_index": self.ignore_index,
                            "drop_no_target": self.drop_no_target,
                        },
                        desc=f"Tokenizing {ds_name}",
                    )

                if len(ds) > 0:
                    loaded_datasets.append(ds)
                summary_table.add_row(ds_name, f"{orig_len:,}", f"{sampled_len:,}", f"{len(ds):,}")
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                console.print(f"[red][Error] 读取或处理数据失败 {path}: {exc}[/red]")

        if not loaded_datasets:
            raise ValueError("未成功加载任何可用数据集，请检查 data.sources 路径配置。")

        console.print(summary_table)

        merged = concatenate_datasets(loaded_datasets).shuffle(seed=self.seed)
        if len(merged) < 2:
            raise ValueError(f"有效样本总数过少 ({len(merged)})，无法切分训练与验证集。")

        val_size = int(len(merged) * self.val_split_ratio)
        if self.max_val_samples is not None and self.max_val_samples > 0:
            val_size = min(val_size, self.max_val_samples)
        val_size = max(1, min(val_size, len(merged) - 1))

        split_ds = merged.train_test_split(test_size=val_size, seed=self.seed)
        train_raw = split_ds["train"]
        val_raw = split_ds["test"]

        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                DatasetDict({"train": train_raw, "val": val_raw}).save_to_disk(str(cache_path))
                console.print(f"[bold green][DataModule][/bold green] 已保存预处理缓存至: {cache_path}")
            except (OSError, ValueError, RuntimeError) as exc:
                console.print(f"[yellow][Warning] 写入缓存目录失败 ({cache_path}): {exc}[/yellow]")

        console.print(
            f"[bold green][DataModule][/bold green] 数据集准备完毕 -> "
            f"训练集: [bold]{len(train_raw):,}[/bold] 条 | 验证集: [bold]{len(val_raw):,}[/bold] 条"
        )
        self.train_dataset = StandardSFTDataset(
            train_raw,
            self.tokenizer,
            max_length=self.max_length,
            chat_start_token=self.chat_start_token,
            chat_end_token=self.chat_end_token,
            ignore_index=self.ignore_index,
            pretokenized=self.pretokenize,
        )
        self.val_dataset = StandardSFTDataset(
            val_raw,
            self.tokenizer,
            max_length=self.max_length,
            chat_start_token=self.chat_start_token,
            chat_end_token=self.chat_end_token,
            ignore_index=self.ignore_index,
            pretokenized=self.pretokenize,
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
