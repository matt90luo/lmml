"""Datamodules and collators for MiniMind pretraining and online RL stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset

from lmml.datas.llm.prepare import load_prepared_dataset
from lmml.datas.llm.sft import DynamicPadCollator, LengthGroupedSampler


class ArrowTokenDataset(Dataset):
    """Thin map-style wrapper returning lists from an Arrow dataset."""

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[int(index)]


class PretrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        prepared_path: str | Path,
        pad_token_id: int,
        batch_size: int = 8,
        val_batch_size: int = 8,
        max_length: int = 512,
        num_workers: int = 4,
        group_by_length: bool = True,
        pad_to_multiple_of: int | None = 8,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.prepared_path = Path(prepared_path)
        self.pad_token_id = int(pad_token_id)
        self.batch_size = int(batch_size)
        self.val_batch_size = int(val_batch_size)
        self.max_length = int(max_length)
        self.num_workers = max(0, int(num_workers))
        self.group_by_length = bool(group_by_length)
        self.pad_to_multiple_of = pad_to_multiple_of
        self.seed = int(seed)
        self.train_dataset: Any = None
        self.val_dataset: Any = None
        self.collator = DynamicPadCollator(
            self.pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
            max_length=max_length,
        )

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return
        split = load_prepared_dataset(self.prepared_path)
        self.train_dataset = ArrowTokenDataset(split["train"])
        self.val_dataset = ArrowTokenDataset(split["val"])

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
        lengths = [int(value) for value in self.train_dataset.dataset["length"]]
        sampler = (
            LengthGroupedSampler(lengths, self.batch_size, seed=self.seed)
            if self.group_by_length and lengths
            else None
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def predict_dataloader(self) -> DataLoader:
        self.setup("predict")
        rows = [dict(row) for row in self.val_dataset.select(range(min(4, len(self.val_dataset))))]
        prompts = [str(row.get("prompt", "")) for row in rows]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        return DataLoader(
            [{"prompt": prompts[i], "input_ids": encoded["input_ids"][i].tolist(), "attention_mask": encoded["attention_mask"][i].tolist()} for i in range(len(prompts))],
            batch_size=min(self.val_batch_size, len(prompts)),
            collate_fn=lambda batch: {
                "prompt": [row["prompt"] for row in batch],
                "input_ids": torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(row["input_ids"]) for row in batch],
                    batch_first=True,
                    padding_value=int(self.tokenizer.pad_token_id),
                ),
                "attention_mask": torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(row["attention_mask"]) for row in batch],
                    batch_first=True,
                    padding_value=0,
                ),
            },
        )


class DPODataModule(pl.LightningDataModule):
    def __init__(
        self,
        prepared_path: str | Path,
        pad_token_id: int,
        batch_size: int = 2,
        val_batch_size: int = 2,
        max_length: int = 1024,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.prepared_path = Path(prepared_path)
        self.pad_token_id = int(pad_token_id)
        self.batch_size = int(batch_size)
        self.val_batch_size = int(val_batch_size)
        self.max_length = int(max_length)
        self.num_workers = max(0, int(num_workers))
        self.train_dataset: Any = None
        self.val_dataset: Any = None
        self.collator = lambda batch: dpo_collator(batch, self.pad_token_id)

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return
        split = load_prepared_dataset(self.prepared_path)
        self.train_dataset = ArrowTokenDataset(split["train"])
        self.val_dataset = ArrowTokenDataset(split["val"])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset or [],
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset or [],
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=0,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def predict_dataloader(self) -> DataLoader:
        self.setup("predict")
        rows = [dict(row) for row in self.val_dataset.select(range(min(4, len(self.val_dataset))))]
        prompts = [str(row.get("prompt", "")) for row in rows]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        return DataLoader(
            [{"prompt": prompts[i], "input_ids": encoded["input_ids"][i].tolist(), "attention_mask": encoded["attention_mask"][i].tolist()} for i in range(len(prompts))],
            batch_size=min(self.val_batch_size, len(prompts)),
            collate_fn=lambda batch: {
                "prompt": [row["prompt"] for row in batch],
                "input_ids": torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(row["input_ids"]) for row in batch],
                    batch_first=True,
                    padding_value=int(self.tokenizer.pad_token_id),
                ),
                "attention_mask": torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(row["attention_mask"]) for row in batch],
                    batch_first=True,
                    padding_value=0,
                ),
            },
        )


def _pad_rows(
    rows: list[list[int]],
    pad_token_id: int,
    *,
    ignore_index: int | None = None,
    left: bool = False,
    pad_to_multiple_of: int | None = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    max_len = max(len(row) for row in rows)
    if pad_to_multiple_of and max_len % pad_to_multiple_of:
        max_len += pad_to_multiple_of - max_len % pad_to_multiple_of
    input_ids = torch.full((len(rows), max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    labels = (
        torch.full((len(rows), max_len), int(ignore_index), dtype=torch.long)
        if ignore_index is not None
        else None
    )
    for index, row in enumerate(rows):
        length = len(row)
        start = max_len - length if left else 0
        input_ids[index, start : start + length] = torch.tensor(row, dtype=torch.long)
        attention_mask[index, start : start + length] = 1
    return input_ids, attention_mask, labels


def dpo_collator(batch: list[dict[str, Any]], pad_id: int = 0) -> dict[str, torch.Tensor]:
    ignored = -100
    chosen_ids, chosen_masks, chosen_labels = _pad_rows(
        [list(row["chosen_input_ids"]) for row in batch], pad_id, ignore_index=ignored
    )
    rejected_ids, rejected_masks, rejected_labels = _pad_rows(
        [list(row["rejected_input_ids"]) for row in batch], pad_id, ignore_index=ignored
    )
    for row_index, row in enumerate(batch):
        for prefix, labels in (("chosen", chosen_labels), ("rejected", rejected_labels)):
            assert labels is not None
            seq = list(row[f"{prefix}_labels"])
            labels[row_index, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return {
        "chosen_input_ids": chosen_ids,
        "chosen_attention_mask": chosen_masks,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_ids,
        "rejected_attention_mask": rejected_masks,
        "rejected_labels": rejected_labels,
    }


class RLPromptDataModule(pl.LightningDataModule):
    """Prompt-only datamodule used by PPO and GRPO."""

    def __init__(
        self,
        prepared_path: str | Path,
        tokenizer: Any,
        batch_size: int = 2,
        val_batch_size: int = 2,
        max_length: int = 768,
        num_workers: int = 0,
        thinking_ratio: float = 0.9,
        agent_mode: bool = False,
    ) -> None:
        super().__init__()
        self.prepared_path = Path(prepared_path)
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.val_batch_size = int(val_batch_size)
        self.max_length = int(max_length)
        self.num_workers = max(0, int(num_workers))
        self.thinking_ratio = float(thinking_ratio)
        self.agent_mode = bool(agent_mode)
        self.train_dataset: Any = None
        self.val_dataset: Any = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return
        split = load_prepared_dataset(self.prepared_path)
        self.train_dataset = split["train"]
        self.val_dataset = split["val"]

    def _collate(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {key: [row[key] for row in batch] for key in batch[0]}

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self._collate,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

    def predict_dataloader(self) -> DataLoader:
        self.setup("predict")
        rows = [dict(row) for row in self.val_dataset.select(range(min(4, len(self.val_dataset))))]
        prompts = [str(row.get("prompt", "")) for row in rows]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
        )
        return DataLoader(
            [{"prompt": prompts[i], "input_ids": encoded["input_ids"][i].tolist(), "attention_mask": encoded["attention_mask"][i].tolist()} for i in range(len(prompts))],
            batch_size=min(self.val_batch_size, len(prompts)),
            collate_fn=lambda batch: {
                "prompt": [row["prompt"] for row in batch],
                "input_ids": torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(row["input_ids"]) for row in batch],
                    batch_first=True,
                    padding_value=int(self.tokenizer.pad_token_id),
                ),
                "attention_mask": torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(row["attention_mask"]) for row in batch],
                    batch_first=True,
                    padding_value=0,
                ),
            },
        )
