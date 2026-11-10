"""Preparation of SargeDev/jev-distill-corpus-v3 splits into lmml Arrow DatasetDicts."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset
from datasets import DatasetDict
from rich.console import Console
from rich.table import Table

from lmml.core.llm.jev.api import apply_candidate_chat_template, candidate_prompts
from lmml.datas.llm.prepare import _save_dataset_dict, load_prepared_dataset

console = Console()

JEV_SPLITS = ("train", "validation", "calibration", "test", "test_set_30k", "ood")
JEV_REQUIRED_KEYS = {"id", "kind", "options", "target", "state", "question"}


def validate_jev_row(row: dict[str, Any], *, location: str = "row") -> dict[str, Any]:
    """Validate and normalize a single row from jev-distill-corpus-v3 or Open-Jev."""
    missing = JEV_REQUIRED_KEYS - set(row.keys())
    if missing:
        raise ValueError(f"{location}: missing required keys {sorted(missing)}")
    kind = str(row["kind"])
    if kind not in ("noul", "choice", "score"):
        raise ValueError(f"{location}: invalid kind={kind!r}; expected noul, choice, or score")
    options = list(row["options"])
    target = [float(x) for x in row["target"]]
    if len(options) != len(target) or len(options) < 2:
        raise ValueError(
            f"{location}: options length ({len(options)}) and target length ({len(target)}) must match and be >= 2"
        )
    if kind == "noul" and len(options) != 2:
        raise ValueError(f"{location}: noul row must have exactly 2 options/targets")
    if any(not math.isfinite(x) or x < 0.0 for x in target):
        raise ValueError(f"{location}: target probabilities must be finite and nonnegative")
    total = sum(target)
    if not math.isclose(total, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        raise ValueError(f"{location}: target probabilities must sum to 1 (got {total})")
    normalized_target = [x / total for x in target]

    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    domain = str(row.get("domain") or metadata.get("domain") or metadata.get("question_id") or kind)
    family = str(row.get("family") or metadata.get("family") or "general")
    source = str(row.get("source") or metadata.get("source") or "jev")
    return {
        "id": str(row["id"]),
        "kind": kind,
        "options": [str(opt) for opt in options],
        "target": normalized_target,
        "state": row["state"],
        "question": str(row["question"]),
        "domain": domain,
        "family": family,
        "source": source,
    }


def _candidate_max_tokens(tokenizer: Any, row: dict[str, Any]) -> int:
    prompts = [apply_candidate_chat_template(tokenizer, p) for p in candidate_prompts(row)]
    encoded = tokenizer(prompts, padding=False, truncation=False)
    input_ids = encoded["input_ids"]
    return max((len(seq) for seq in input_ids), default=0)


def _read_jev_jsonl(
    path: Path,
    *,
    max_samples: int | None = None,
    sample_ratio: float | None = None,
    seed: int = 42,
    balanced: bool = False,
    tokenizer: Any | None = None,
    max_length: int | None = None,
    filter_overlong: bool = True,
) -> tuple[list[dict[str, Any]], int, int]:
    """Stream and sample a Jev JSONL split without reading redundant lines when capped."""
    if not path.is_file():
        raise FileNotFoundError(f"Jev split file does not exist: {path}")

    cap = int(max_samples) if (max_samples is not None and int(max_samples) > 0) else None
    ratio = float(sample_ratio) if (sample_ratio is not None and 0.0 < float(sample_ratio) < 1.0) else None

    # Fast reservoir / capped read when max_samples is set so 655k-row train.jsonl can be smoke-sampled quickly.
    raw_rows: list[dict[str, Any]] = []
    raw_count = 0
    stream_limit = cap * 4 if (cap is not None and ratio is None) else None
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw_count += 1
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            validated = validate_jev_row(parsed, location=f"{path.name}:{line_number}")
            raw_rows.append(validated)
            if stream_limit is not None and len(raw_rows) >= stream_limit:
                break

    rng = random.Random(seed)
    if balanced:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in raw_rows:
            groups[(row["source"], row["kind"])].append(row)
        for key in sorted(groups):
            rng.shuffle(groups[key])
        ordered: list[dict[str, Any]] = []
        keys = sorted(groups)
        index = 0
        while any(index < len(groups[key]) for key in keys):
            for key in keys:
                if index < len(groups[key]):
                    ordered.append(groups[key][index])
            index += 1
        raw_rows = ordered
    else:
        rng.shuffle(raw_rows)

    if ratio is not None:
        target_len = max(1, int(len(raw_rows) * ratio))
        raw_rows = raw_rows[:target_len]

    accepted: list[dict[str, Any]] = []
    rejected_overlong = 0
    for row in raw_rows:
        if tokenizer is not None and max_length is not None and int(max_length) > 0:
            tok_len = _candidate_max_tokens(tokenizer, row)
            if tok_len > int(max_length):
                if filter_overlong:
                    rejected_overlong += 1
                    continue
                raise ValueError(
                    f"{path.name}:{row['id']}: candidate token length {tok_len} exceeds max_length={max_length}"
                )
            row["max_candidate_tokens"] = tok_len
        else:
            row["max_candidate_tokens"] = 0
        row["record_json"] = json.dumps(row, ensure_ascii=False)
        accepted.append(row)
        if cap is not None and len(accepted) >= cap:
            break

    return accepted, raw_count, rejected_overlong


def _resolve_split_files(
    data_configs: list[dict[str, Any]],
) -> list[tuple[str, Path, dict[str, Any]]]:
    resolved: list[tuple[str, Path, dict[str, Any]]] = []
    for cfg in data_configs:
        path = Path(str(cfg["path"]))
        explicit_split = str(cfg.get("split") or cfg.get("role") or "").strip().lower()
        if path.is_file():
            split_name = explicit_split or path.stem.lower()
            resolved.append((split_name, path, cfg))
        elif path.is_dir():
            if explicit_split:
                candidate = path / f"{explicit_split}.jsonl"
                if candidate.is_file():
                    resolved.append((explicit_split, candidate, cfg))
            else:
                for split_name in JEV_SPLITS:
                    candidate = path / f"{split_name}.jsonl"
                    if candidate.is_file():
                        resolved.append((split_name, candidate, cfg))
        else:
            raise FileNotFoundError(f"Jev data source does not exist: {path}")
    return resolved


def prepare_jev_dataset(
    data_configs: list[dict[str, Any]],
    tokenizer: Any,
    output_dir: str | Path,
    *,
    overwrite: bool = True,
    max_length: int = 512,
    filter_overlong: bool = True,
    training_sampling: str = "source_kind_round_robin",
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_calibration_samples: int | None = None,
    max_test_samples: int | None = None,
    max_test_30k_samples: int | None = None,
    max_ood_samples: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Prepare SargeDev/jev-distill-corpus-v3 splits into an Arrow DatasetDict."""
    output_path = Path(output_dir)
    if output_path.exists() and not overwrite:
        loaded = load_prepared_dataset(output_path)
        return {
            "task": "jev",
            "prepared_path": str(output_path.resolve()),
            "splits": {name: len(loaded[name]) for name in loaded},
            "overwritten": False,
        }
    if not data_configs:
        raise ValueError("Jev data.sources must not be empty")

    split_caps: dict[str, int | None] = {
        "train": int(max_train_samples) if max_train_samples else None,
        "validation": int(max_val_samples) if max_val_samples else None,
        "val": int(max_val_samples) if max_val_samples else None,
        "calibration": int(max_calibration_samples) if max_calibration_samples else None,
        "test": int(max_test_samples) if max_test_samples else None,
        "test_set_30k": int(max_test_30k_samples) if max_test_30k_samples else None,
        "ood": int(max_ood_samples) if max_ood_samples else None,
    }

    split_files = _resolve_split_files(data_configs)
    if not split_files:
        raise ValueError("No Jev JSONL split files found in data.sources")

    collected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    table = Table(title="Jev Distill Corpus v3 数据准备汇总", show_header=True, header_style="bold cyan")
    table.add_column("Split", style="white")
    table.add_column("文件", style="white")
    table.add_column("读取行数", justify="right")
    table.add_column("过滤超长", justify="right", style="yellow")
    table.add_column("有效样本数", justify="right", style="green")

    for split_name, file_path, cfg in split_files:
        canonical_split = "validation" if split_name == "val" else split_name
        cap = cfg.get("max_samples")
        if cap is None:
            cap = split_caps.get(canonical_split)
        balanced = (
            training_sampling == "source_kind_round_robin"
            if canonical_split == "train"
            else True
        )
        rows, raw_count, rejected = _read_jev_jsonl(
            file_path,
            max_samples=cap,
            sample_ratio=cfg.get("sample_ratio"),
            seed=int(cfg.get("seed", seed)),
            balanced=balanced,
            tokenizer=tokenizer,
            max_length=max_length,
            filter_overlong=filter_overlong,
        )
        collected[canonical_split].extend(rows)
        table.add_row(
            canonical_split,
            file_path.name,
            f"{raw_count:,}",
            f"{rejected:,}",
            f"{len(rows):,}",
        )

    console.print(table)
    if not collected.get("train"):
        raise ValueError("Jev dataset preparation produced no training rows")

    if not collected.get("validation"):
        if collected.get("calibration"):
            collected["validation"] = list(collected["calibration"])
        else:
            holdout = max(1, len(collected["train"]) // 10)
            collected["validation"] = collected["train"][-holdout:]
            collected["train"] = collected["train"][:-holdout]
    if not collected.get("calibration"):
        collected["calibration"] = list(collected["validation"])
    if not collected.get("test"):
        collected["test"] = list(collected["validation"])

    dataset_dict_payload: dict[str, HFDataset] = {
        "train": HFDataset.from_list(collected["train"]),
        "val": HFDataset.from_list(collected["validation"]),
        "validation": HFDataset.from_list(collected["validation"]),
        "calibration": HFDataset.from_list(collected["calibration"]),
        "test": HFDataset.from_list(collected["test"]),
    }
    if collected.get("test_set_30k"):
        dataset_dict_payload["test_set_30k"] = HFDataset.from_list(collected["test_set_30k"])
    if collected.get("ood"):
        dataset_dict_payload["ood"] = HFDataset.from_list(collected["ood"])

    dataset = DatasetDict(dataset_dict_payload)
    _save_dataset_dict(dataset, output_path)
    summary = {
        "task": "jev",
        "prepared_path": str(output_path.resolve()),
        "train_samples": len(dataset["train"]),
        "val_samples": len(dataset["val"]),
        "calibration_samples": len(dataset["calibration"]),
        "test_samples": len(dataset["test"]),
        "test_set_30k_samples": len(dataset["test_set_30k"]) if "test_set_30k" in dataset else 0,
        "ood_samples": len(dataset["ood"]) if "ood" in dataset else 0,
        "overwritten": True,
    }
    console.print(
        f"[bold green][Jev Prepare][/bold green] saved {output_path} "
        f"(train={summary['train_samples']}, val={summary['val_samples']}, "
        f"calibration={summary['calibration_samples']}, test={summary['test_samples']}, "
        f"test_set_30k={summary['test_set_30k_samples']}, ood={summary['ood_samples']})"
    )
    return summary


__all__ = ["JEV_REQUIRED_KEYS", "JEV_SPLITS", "prepare_jev_dataset", "validate_jev_row"]
