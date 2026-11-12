"""PyTorch Datasets, Collators, and LightningDataModule for Jev decision training and evaluation."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from rich.console import Console
from torch.utils.data import DataLoader, Dataset, Subset

from lmml.core.llm.jev.api import apply_candidate_chat_template, candidate_prompts, compile_request
from lmml.datas.llm.prepare import load_prepared_dataset

console = Console()


def _dataset_column(dataset: Dataset, name: str) -> list[Any] | None:
    """Read a column from the Arrow-backed dataset when it exists."""
    arrow = getattr(dataset, "dataset", dataset)
    columns = getattr(arrow, "column_names", None)
    if columns is None or name not in columns:
        return None
    try:
        return list(arrow[name])
    except Exception:  # noqa: BLE001 - optional sampling metadata must not break training
        return None


def _sample_indices(
    dataset: Dataset,
    max_samples: int,
    *,
    seed: int,
    strategy: str,
) -> list[int]:
    """Deterministically select a compact training/validation index subset.

    ``source_kind_round_robin`` keeps the decision mix balanced by taking one
    row from each (source, kind) bucket in turn. ``random`` is used for
    validation, where an unbiased shuffled subset is preferable.
    """
    total = len(dataset)
    target = min(int(max_samples), total)
    if target >= total:
        return list(range(total))
    rng = random.Random(int(seed))
    if strategy != "source_kind_round_robin":
        indices = list(range(total))
        rng.shuffle(indices)
        return indices[:target]

    sources = _dataset_column(dataset, "source")
    kinds = _dataset_column(dataset, "kind")
    if sources is None or kinds is None:
        indices = list(range(total))
        rng.shuffle(indices)
        return indices[:target]

    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (source, kind) in enumerate(zip(sources, kinds)):
        buckets[(str(source), str(kind))].append(index)
    keys = sorted(buckets)
    for key in keys:
        rng.shuffle(buckets[key])

    selected: list[int] = []
    cursor = 0
    while len(selected) < target:
        progressed = False
        for key in keys:
            if cursor < len(buckets[key]):
                selected.append(buckets[key][cursor])
                progressed = True
                if len(selected) == target:
                    break
        if not progressed:
            break
        cursor += 1
    return selected


def _parse_row(item: dict[str, Any]) -> dict[str, Any]:
    if "record_json" in item and isinstance(item["record_json"], str):
        parsed = json.loads(item["record_json"])
        if isinstance(parsed, dict):
            return parsed
    return dict(item)


class JevArrowDataset(Dataset):
    """Thin wrapper over a prepared Arrow split of Jev records."""

    def __init__(self, arrow_split: Any) -> None:
        self.dataset = arrow_split

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return _parse_row(dict(self.dataset[int(index)]))


class JevCollator:
    """Pad pre-tokenized candidates, with a legacy on-the-fly tokenization fallback."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int = 512,
        strict_length: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.strict_length = bool(strict_length)
        self.tokenizer.padding_side = "right"
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        records: list[dict[str, Any]] = [_parse_row(item) for item in batch]
        if records and all("input_ids_flat" in record for record in records):
            return self._collate_pretokenized(records)
        return self._collate_records(records)

    def _collate_pretokenized(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Pad pre-tokenized candidate IDs without invoking the tokenizer again."""
        pad_id = int(self.tokenizer.pad_token_id)
        lengths_per_row: list[list[int]] = []
        counts: list[int] = []
        kinds: list[str] = []
        targets: list[list[float]] = []
        rows: list[list[list[int]]] = []
        max_width = 0
        for record in records:
            lengths = [int(value) for value in record["candidate_lengths"]]
            offsets = [int(value) for value in record["candidate_offsets"]]
            flat = [int(value) for value in record["input_ids_flat"]]
            if len(offsets) != len(lengths) + 1 or offsets[0] != 0 or offsets[-1] != len(flat):
                raise ValueError(f"invalid pre-tokenized offsets for row {record.get('id', '<unknown>')}")
            candidates: list[list[int]] = []
            for index, length in enumerate(lengths):
                start, end = offsets[index], offsets[index + 1]
                if end - start != length or length <= 0:
                    raise ValueError(f"invalid pre-tokenized length for row {record.get('id', '<unknown>')}")
                candidate = flat[start:end]
                if len(candidate) > self.max_length:
                    if self.strict_length:
                        raise ValueError(
                            f"Input length {len(candidate)} exceeds max_length={self.max_length}; no silent truncation"
                        )
                    candidate = candidate[: self.max_length]
                candidates.append(candidate)
                max_width = max(max_width, len(candidate))
            rows.append(candidates)
            lengths_per_row.append([len(candidate) for candidate in candidates])
            counts.append(len(candidates))
            kinds.append(str(record["kind"]))
            if "target" in record and record["target"] is not None:
                targets.append([float(x) for x in record["target"]])
        input_ids = torch.full((sum(counts), max_width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((sum(counts), max_width), dtype=torch.long)
        lengths: list[int] = []
        cursor = 0
        for candidates, row_lengths in zip(rows, lengths_per_row):
            for candidate, length in zip(candidates, row_lengths):
                if length:
                    input_ids[cursor, :length] = torch.tensor(candidate, dtype=torch.long)
                    attention_mask[cursor, :length] = 1
                lengths.append(length)
                cursor += 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "counts": counts,
            "kinds": kinds,
            "targets": targets,
            "records": records,
        }

    def _collate_records(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        prompts: list[str] = []
        counts: list[int] = []
        kinds: list[str] = []
        targets: list[list[float]] = []

        for record in records:
            entries = candidate_prompts(record)
            counts.append(len(entries))
            kinds.append(str(record["kind"]))
            if "target" in record and record["target"] is not None:
                targets.append([float(x) for x in record["target"]])
            for prompt in entries:
                prompts.append(apply_candidate_chat_template(self.tokenizer, prompt))

        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=not self.strict_length,
            max_length=self.max_length if not self.strict_length else None,
            return_tensors="pt",
        )
        lengths = encoded["attention_mask"].sum(dim=-1)
        max_seen = int(lengths.max().item()) if lengths.numel() > 0 else 0
        if self.strict_length and max_seen > self.max_length:
            raise ValueError(
                f"Input length {max_seen} exceeds max_length={self.max_length}; no silent truncation"
            )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "lengths": lengths,
            "counts": counts,
            "kinds": kinds,
            "targets": targets,
            "records": records,
        }


class JevPredictionCollator:
    """Collate SystemOne inference requests into compiled records and tokenized candidates."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int = 512,
        strict_length: bool = True,
    ) -> None:
        self.record_collator = JevCollator(
            tokenizer,
            max_length=max_length,
            strict_length=strict_length,
        )

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        requests: list[dict[str, Any]] = []
        compiled_groups: list[list[dict[str, Any]]] = []
        flat_records: list[dict[str, Any]] = []

        for item in batch:
            req = item.get("record", item)
            if "questions" in req and isinstance(req["questions"], dict):
                records = compile_request(req["state"], req["questions"])
                requests.append(req)
                compiled_groups.append(records)
                flat_records.extend(records)
            else:
                row = _parse_row(req)
                if "answer_keys" not in row:
                    row["answer_keys"] = (
                        ["false", "true"]
                        if row["kind"] == "noul"
                        else [str(opt) for opt in row["options"]]
                    )
                if row["kind"] == "score" and "legend" not in row:
                    row["legend"] = dict(zip(row["answer_keys"], row["options"]))
                requests.append({"state": row["state"], "questions": {row["id"]: row}})
                compiled_groups.append([row])
                flat_records.append(row)

        encoded = self.record_collator(flat_records)
        encoded["requests"] = requests
        encoded["compiled_groups"] = compiled_groups
        return encoded


class JevDataModule(pl.LightningDataModule):
    """LightningDataModule managing train, validation, calibration, test, test_set_30k, and ood splits."""

    def __init__(
        self,
        prepared_path: str | Path,
        tokenizer: Any,
        *,
        batch_size: int = 1,
        val_batch_size: int = 1,
        num_workers: int = 0,
        max_length: int = 512,
        strict_length: bool = True,
        predict_requests: list[dict[str, Any]] | None = None,
        train_max_samples: int | None = None,
        val_max_samples: int | None = None,
        train_sampling: str = "source_kind_round_robin",
        sample_seed: int = 42,
    ) -> None:
        super().__init__()
        self.prepared_path = Path(prepared_path)
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.val_batch_size = int(val_batch_size)
        self.num_workers = int(num_workers)
        self.max_length = int(max_length)
        self.strict_length = bool(strict_length)
        self.predict_requests = list(predict_requests or [])
        self.train_max_samples = int(train_max_samples) if train_max_samples is not None else None
        self.val_max_samples = int(val_max_samples) if val_max_samples is not None else None
        if self.train_max_samples is not None and self.train_max_samples <= 0:
            raise ValueError("train_max_samples must be positive or null")
        if self.val_max_samples is not None and self.val_max_samples <= 0:
            raise ValueError("val_max_samples must be positive or null")
        self.train_sampling = str(train_sampling)
        if self.train_sampling not in {"source_kind_round_robin", "random"}:
            raise ValueError("train_sampling must be 'source_kind_round_robin' or 'random'")
        self.sample_seed = int(sample_seed)

        self.collator = JevCollator(
            tokenizer,
            max_length=self.max_length,
            strict_length=self.strict_length,
        )
        self.predict_collator = JevPredictionCollator(
            tokenizer,
            max_length=self.max_length,
            strict_length=self.strict_length,
        )

        self.splits: dict[str, JevArrowDataset] = {}
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.calibration_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self.test_30k_dataset: Dataset | None = None
        self.ood_dataset: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return
        loaded = load_prepared_dataset(self.prepared_path)
        for name in loaded:
            self.splits[name] = JevArrowDataset(loaded[name])

        full_train = self.splits["train"]
        full_val = self.splits.get("validation") or self.splits.get("val") or full_train
        self.train_dataset = self._subset(
            full_train,
            self.train_max_samples,
            strategy=self.train_sampling,
            seed=self.sample_seed,
            label="train",
        )
        self.val_dataset = self._subset(
            full_val,
            self.val_max_samples,
            strategy="random",
            seed=self.sample_seed + 1,
            label="validation",
        )
        self.calibration_dataset = self.splits.get("calibration") or full_val
        self.test_dataset = self.splits.get("test") or full_val
        self.test_30k_dataset = self.splits.get("test_set_30k")
        self.ood_dataset = self.splits.get("ood")

    @staticmethod
    def _subset(
        dataset: Dataset,
        max_samples: int | None,
        *,
        strategy: str,
        seed: int,
        label: str,
    ) -> Dataset:
        if max_samples is None or int(max_samples) >= len(dataset):
            return dataset
        indices = _sample_indices(dataset, int(max_samples), seed=seed, strategy=strategy)
        console.print(
            f"[bold cyan][Jev Sample][/bold cyan] {label}: {len(dataset):,} -> {len(indices):,} "
            f"(strategy={strategy}, seed={seed})"
        )
        return Subset(dataset, indices)

    def train_dataloader(self) -> DataLoader:
        self.setup("fit")
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        self.setup("fit")
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
        self.setup("test")
        assert self.test_dataset is not None
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def split_dataloader(
        self,
        split_name: str,
        *,
        max_samples: int | None = None,
        batch_size: int | None = None,
    ) -> DataLoader:
        self.setup("test")
        canonical = "val" if split_name == "validation" and "validation" not in self.splits else split_name
        if canonical not in self.splits:
            raise KeyError(f"Split {split_name!r} not found in prepared dataset {list(self.splits)}")
        dataset: Dataset = self.splits[canonical]
        if max_samples is not None:
            max_samples = int(max_samples)
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            dataset = Subset(dataset, range(min(max_samples, len(dataset))))
        size = int(batch_size if batch_size is not None else self.val_batch_size)
        if size <= 0:
            raise ValueError("batch_size must be positive")
        return DataLoader(
            dataset,
            batch_size=size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def calibration_dataloader(
        self,
        *,
        max_samples: int | None = None,
        batch_size: int | None = None,
    ) -> DataLoader:
        return self.split_dataloader(
            "calibration",
            max_samples=max_samples,
            batch_size=batch_size,
        )

    def predict_dataloader(self) -> DataLoader:
        requests = list(self.predict_requests)
        if not requests:
            self.setup("predict")
            assert self.test_dataset is not None
            requests = [self.test_dataset[i] for i in range(min(4, len(self.test_dataset)))]
        return DataLoader(
            [{"index": idx, "record": req} for idx, req in enumerate(requests)],
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=self.predict_collator,
            num_workers=0,
        )


__all__ = [
    "JevArrowDataset",
    "JevCollator",
    "JevDataModule",
    "JevPredictionCollator",
]
