"""Preparation of Kev frozen suites into lmml Arrow DatasetDicts."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset
from datasets import DatasetDict
from rich.console import Console

from lmml.core.llm.kev.api import materialize_labelled
from lmml.core.llm.kev.model import MAX_BRANCH, MAX_STATE, ContextOverflow, encode
from lmml.datas.llm.prepare import _save_dataset_dict, load_prepared_dataset

console = Console()
KEV_SPLITS = ("train", "calibration", "development", "test")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if "state" not in record or not isinstance(record.get("questions"), dict):
                raise ValueError(f"{path}:{line_number}: record needs state and questions")
            records.append(record)
    return records


def _read_manifest(path: Path) -> dict[str, Any] | None:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _source_files(config: dict[str, Any]) -> list[tuple[str, Path]]:
    path = Path(str(config["path"]))
    split = str(config.get("split", "")).strip().lower()
    if path.is_file():
        return [(split or str(config.get("role", "train")), path)]
    if not path.is_dir():
        raise FileNotFoundError(f"Kev data source does not exist: {path}")
    manifest = _read_manifest(path)
    if manifest and manifest.get("eval_only") and str(config.get("role", "")) == "train":
        raise ValueError(f"eval-only Kev suite cannot be used for training: {path}")
    files: list[tuple[str, Path]] = []
    if split:
        candidate = path / f"{split}.jsonl"
        if candidate.exists():
            files.append((split, candidate))
    else:
        for name in KEV_SPLITS:
            candidate = path / f"{name}.jsonl"
            if candidate.exists():
                files.append((name, candidate))
    if not files:
        jsonl = sorted(path.glob("*.jsonl"))
        if len(jsonl) == 1:
            files.append((str(config.get("role", "train")), jsonl[0]))
    return files


def _state_key(record: dict[str, Any]) -> str:
    meta = record.get("_meta") or {}
    if meta.get("text_sha256"):
        return str(meta["text_sha256"])
    payload = json.dumps(record["state"], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(" ".join(payload.casefold().split()).encode()).hexdigest()


def _is_eval_only(record: dict[str, Any], manifest: dict[str, Any] | None) -> bool:
    if not manifest:
        return False
    source = str((record.get("_meta") or {}).get("source", ""))
    return source in set(manifest.get("eval_only_sources", [])) or source in set(manifest.get("holdout_sources", []))


def _fit_record(record: dict[str, Any], tokenizer: Any, max_state: int, max_branch: int) -> bool:
    try:
        materialized = materialize_labelled(record)
        encode(tokenizer, materialized, max_state=max_state, max_branch=max_branch, strict=True)
        return True
    except (ContextOverflow, ValueError, KeyError, TypeError):
        return False


def _row(
    record: dict[str, Any],
    source: str,
    split: str,
    tokenizer: Any,
    max_state: int,
    max_branch: int,
) -> dict[str, Any]:
    materialized = materialize_labelled(record)
    encoded = encode(
        tokenizer,
        materialized,
        max_state=max_state,
        max_branch=max_branch,
        strict=True,
    )
    meta = record.get("_meta") or {}
    return {
        "record_json": json.dumps(record, ensure_ascii=False),
        "source": source,
        "split": split,
        "group_id": str(meta.get("group_id") or meta.get("id") or _state_key(record)),
        "state_hash": _state_key(record),
        "length": len(encoded["ids"]),
    }


def _sample(records: list[dict[str, Any]], max_samples: int | None, ratio: float | None, seed: int) -> list[dict[str, Any]]:
    target = len(records)
    if ratio is not None and 0 < float(ratio) < 1:
        target = int(target * float(ratio))
    if max_samples is not None and int(max_samples) > 0:
        target = min(target, int(max_samples))
    if target >= len(records):
        return records
    indices = sorted(random.Random(seed).sample(range(len(records)), target))
    return [records[index] for index in indices]


def prepare_kev_dataset(
    data_configs: list[dict[str, Any]],
    tokenizer: Any,
    output_dir: str | Path,
    *,
    overwrite: bool = True,
    max_state: int = MAX_STATE,
    max_branch: int = MAX_BRANCH,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    seed: int = 42,
    fallback_sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Load frozen labelled suites and write train/val/calibration/test Arrow splits."""
    output_path = Path(output_dir)
    if output_path.exists() and not overwrite:
        dataset = load_prepared_dataset(output_path)
        return {
            "task": "kev",
            "prepared_path": str(output_path.resolve()),
            "train_samples": len(dataset["train"]),
            "val_samples": len(dataset["val"]),
            "overwritten": False,
        }
    if not data_configs:
        raise ValueError("Kev data.sources must not be empty")

    collected: dict[str, list[dict[str, Any]]] = {split: [] for split in KEV_SPLITS}
    seen: dict[str, set[str]] = {split: set() for split in KEV_SPLITS}
    reports: list[dict[str, Any]] = []

    def load(configs: list[dict[str, Any]], *, fallback: bool = False) -> None:
        for config in configs:
            source_name = Path(str(config["path"])).name
            try:
                files = _source_files(config)
            except FileNotFoundError as error:
                console.print(f"[yellow][Kev data] source unavailable: {error}[/yellow]")
                if fallback or fallback_sources:
                    continue
                raise
            except ValueError:
                raise
            if not files:
                console.print(f"[yellow][Kev data] no JSONL partitions found under {config['path']}[/yellow]")
                continue
            manifest = _read_manifest(Path(str(config["path"]))) if Path(str(config["path"])).is_dir() else None
            accepted = 0
            rejected = 0
            for file_split, path in files:
                if fallback and file_split != "train":
                    continue
                records = _read_jsonl(path)
                records = _sample(
                    records,
                    config.get("max_samples"),
                    config.get("sample_ratio"),
                    int(config.get("seed", seed)),
                )
                target_split = str(config.get("role") or file_split).lower()
                if target_split not in collected:
                    target_split = file_split
                if target_split not in collected:
                    raise ValueError(f"unknown Kev split role {target_split!r}")
                for record in records:
                    if target_split == "train" and _is_eval_only(record, manifest):
                        rejected += 1
                        continue
                    if not _fit_record(record, tokenizer, max_state, max_branch):
                        rejected += 1
                        continue
                    key = _state_key(record)
                    if key in seen[target_split]:
                        rejected += 1
                        continue
                    seen[target_split].add(key)
                    collected[target_split].append(
                        _row(record, source_name, target_split, tokenizer, max_state, max_branch)
                    )
                    accepted += 1
            reports.append(
                {
                    "path": str(config["path"]),
                    "fallback": fallback,
                    "accepted": accepted,
                    "rejected": rejected,
                }
            )

    load(data_configs)
    if not collected["train"] and fallback_sources:
        console.print("[yellow][Kev data] no training records in primary suites; using configured fallback[/yellow]")
        load(fallback_sources, fallback=True)
    if not collected["train"]:
        raise ValueError("Kev preprocessing produced no training records; check data.sources/fallback_sources")

    random.Random(seed).shuffle(collected["train"])
    if max_train_samples and int(max_train_samples) > 0:
        collected["train"] = collected["train"][: int(max_train_samples)]

    # The trainer always needs a validation split. Prefer development, then
    # calibration, then a deterministic holdout from train.
    if collected["development"]:
        val_records = collected["development"]
    elif collected["calibration"]:
        val_records = collected["calibration"]
    else:
        count = min(max(1, len(collected["train"]) // 10), len(collected["train"]))
        val_records = collected["train"][-count:]
        collected["train"] = collected["train"][:-count]
    if max_val_samples and int(max_val_samples) > 0:
        val_records = val_records[: int(max_val_samples)]
    calibration = collected["calibration"] or val_records
    test = collected["test"] or collected["development"] or val_records

    dataset = DatasetDict(
        {
            "train": HFDataset.from_list(collected["train"]),
            "val": HFDataset.from_list(val_records),
            "calibration": HFDataset.from_list(calibration),
            "test": HFDataset.from_list(test),
        }
    )
    _save_dataset_dict(dataset, output_path)
    summary = {
        "task": "kev",
        "prepared_path": str(output_path.resolve()),
        "train_samples": len(dataset["train"]),
        "val_samples": len(dataset["val"]),
        "calibration_samples": len(dataset["calibration"]),
        "test_samples": len(dataset["test"]),
        "overwritten": True,
        "sources": reports,
    }
    console.print(
        f"[bold green][Kev Prepare][/bold green] saved {output_path} "
        f"(train={summary['train_samples']}, val={summary['val_samples']}, "
        f"calibration={summary['calibration_samples']}, test={summary['test_samples']})"
    )
    return summary


__all__ = ["KEV_SPLITS", "prepare_kev_dataset"]
