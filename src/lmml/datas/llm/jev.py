"""Streaming, pre-tokenized preparation of SargeDev/jev-distill-corpus-v3 splits."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset
from datasets import DatasetDict, Features, Sequence, Value, concatenate_datasets, load_from_disk
from rich.console import Console

from lmml.core.llm.jev.api import (
    apply_candidate_chat_template,
    candidate_prefix,
    candidate_prompts_from_prefix,
)
from lmml.datas.llm.prepare import _save_dataset_dict
from lmml.datas.llm.sft import build_sft_tokenizer, resolve_num_proc

console = Console()

JEV_SPLITS = ("train", "validation", "calibration", "test", "test_set_30k", "ood")
JEV_REQUIRED_KEYS = {"id", "kind", "options", "target", "state", "question"}
JEV_SCHEMA_VERSION = "jev-pretokenized-v1"

JEV_FEATURES = Features(
    {
        "id": Value("string"),
        "kind": Value("string"),
        "options": Sequence(Value("string")),
        "target": Sequence(Value("float32")),
        "state": Value("string"),
        "question": Value("string"),
        "domain": Value("string"),
        "family": Value("string"),
        "source": Value("string"),
        "input_ids_flat": Sequence(Value("int32")),
        "candidate_offsets": Sequence(Value("int32")),
        "candidate_lengths": Sequence(Value("int32")),
        "max_candidate_tokens": Value("int32"),
    }
)

JEV_RAW_FEATURES = Features(
    {
        "id": Value("string"),
        "kind": Value("string"),
        "options": Sequence(Value("string")),
        "target": Sequence(Value("float32")),
        "state": Value("string"),
        "question": Value("string"),
        "domain": Value("string"),
        "family": Value("string"),
        "source": Value("string"),
        "max_candidate_tokens": Value("int32"),
    }
)


@dataclass(frozen=True)
class _SourceInfo:
    path: str
    line_count: int
    size_bytes: int
    sha256: str
    shards: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class _ShardSpec:
    split: str
    input_path: str
    input_sha256: str
    start_byte: int
    end_byte: int
    row_count: int
    shard_index: int
    total_shards: int
    shard_dir: str
    config: dict[str, Any]


_WORKER_TOKENIZER: Any = None
_WORKER_CONFIG: dict[str, Any] = {}


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json_atomic(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_jev_row(row: dict[str, Any], *, location: str = "row") -> dict[str, Any]:
    """Validate and normalize one row from jev-distill-corpus-v3 or Open-Jev."""
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


def _state_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


def _iter_jsonl_rows(
    path: str | Path,
    *,
    start_byte: int = 0,
    end_byte: int | None = None,
    max_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Fast chunked binary JSONL reader without per-line tell() calls.

    ``readline()`` on the mounted Windows drive is unusually slow; reading
    fixed-size blocks and splitting lines is substantially faster while still
    keeping memory bounded to one chunk plus a partial line.
    """
    path = Path(path)
    block_size = 1024 * 1024
    with path.open("rb") as handle:
        handle.seek(start_byte)
        line_number = 0
        carry = b""
        while max_rows is None or line_number < max_rows:
            if end_byte is not None:
                remaining = end_byte - handle.tell()
                if remaining <= 0:
                    break
                chunk = handle.read(min(block_size, remaining))
            else:
                chunk = handle.read(block_size)
            if not chunk:
                break
            data = carry + chunk
            lines = data.splitlines(keepends=True)
            if lines and not lines[-1].endswith((b"\n", b"\r")):
                carry = lines.pop()
            else:
                carry = b""
            for raw in lines:
                line_number += 1
                if max_rows is not None and line_number > max_rows:
                    return
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"{path}:line {line_number}: invalid JSON: {exc}") from exc
                yield validate_jev_row(row, location=f"{path.name}:line {line_number}")
        if carry and (max_rows is None or line_number < max_rows):
            line_number += 1
            try:
                row = json.loads(carry.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:line {line_number}: invalid JSON: {exc}") from exc
            yield validate_jev_row(row, location=f"{path.name}:line {line_number}")

def _scan_source(path: str | Path, shard_size: int) -> _SourceInfo:
    """Scan one source in fast chunks for its hash, line count and byte ranges."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Jev source does not exist: {path}")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    digest = hashlib.sha256()
    starts = [0]
    line_count = 0
    block_size = 1024 * 1024
    with path.open("rb") as handle:
        base = 0
        carry = b""
        while True:
            chunk = handle.read(block_size)
            if not chunk:
                break
            digest.update(chunk)
            carry_prefix = len(carry)
            data = carry + chunk
            lines = data.splitlines(keepends=True)
            if lines and not lines[-1].endswith((b"\n", b"\r")):
                carry = lines.pop()
            else:
                carry = b""
            position = base - carry_prefix
            for line in lines:
                position += len(line)
                line_count += 1
                if line_count % shard_size == 0:
                    starts.append(position)
            base += len(chunk)
        if carry:
            line_count += 1
            if line_count % shard_size == 0:
                starts.append(base)
        size = base
    starts = sorted({offset for offset in starts if offset < size})
    if not starts:
        starts = [0]
    if starts[-1] < size:
        starts.append(size)
    ranges = tuple(
        (start, starts[index + 1], min(shard_size, line_count - index * shard_size))
        for index, start in enumerate(starts[:-1])
        if start < starts[index + 1] and line_count - index * shard_size > 0
    )
    return _SourceInfo(
        path=str(path.resolve()),
        line_count=line_count,
        size_bytes=size,
        sha256=digest.hexdigest(),
        shards=ranges,
    )

def _count_groups(path: str | Path) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in _iter_jsonl_rows(path):
        counts[(row["source"], row["kind"])] += 1
    return dict(counts)


def _allocate_group_targets(counts: dict[tuple[str, str], int], target: int) -> dict[tuple[str, str], int]:
    if target <= 0:
        return {key: 0 for key in counts}
    total = sum(counts.values())
    if target >= total:
        return dict(counts)
    groups = sorted(counts)
    allocation = {key: 0 for key in groups}
    remaining = target
    while remaining > 0:
        active = [key for key in groups if allocation[key] < counts[key]]
        if not active:
            break
        share = max(1, remaining // len(active))
        progressed = False
        for key in active:
            if remaining <= 0:
                break
            amount = min(counts[key] - allocation[key], share, remaining)
            if amount > 0:
                allocation[key] += amount
                remaining -= amount
                progressed = True
        if not progressed:
            break
    return allocation


def _reservoir_sample(
    path: str | Path,
    target: int,
    *,
    seed: int,
    balanced: bool,
    progress_every: int,
) -> list[dict[str, Any]]:
    """Uniform reservoir sampling; optionally preserve (source, kind) balance."""
    if target <= 0:
        return []
    rng = random.Random(seed)
    if not balanced:
        reservoir: list[dict[str, Any]] = []
        for seen, row in enumerate(_iter_jsonl_rows(path), start=1):
            if len(reservoir) < target:
                reservoir.append(row)
            else:
                replace = rng.randrange(seen)
                if replace < target:
                    reservoir[replace] = row
            if progress_every > 0 and seen % progress_every == 0:
                print(f"[Jev sample] {Path(path).name}: seen={seen:,} reservoir={len(reservoir):,}", flush=True)
        rng.shuffle(reservoir)
        return reservoir

    counts = _count_groups(path)
    allocation = _allocate_group_targets(counts, target)
    reservoirs: dict[tuple[str, str], list[dict[str, Any]]] = {key: [] for key in counts}
    seen_by_group: dict[tuple[str, str], int] = defaultdict(int)
    total_seen = 0
    for row in _iter_jsonl_rows(path):
        key = (row["source"], row["kind"])
        total_seen += 1
        seen_by_group[key] += 1
        cap = allocation.get(key, 0)
        if cap <= 0:
            continue
        reservoir = reservoirs[key]
        if len(reservoir) < cap:
            reservoir.append(row)
        else:
            replace = rng.randrange(seen_by_group[key])
            if replace < cap:
                reservoir[replace] = row
        if progress_every > 0 and total_seen % progress_every == 0:
            print(f"[Jev sample] {Path(path).name}: seen={total_seen:,} balanced_groups={len(counts)}", flush=True)
    for key in sorted(reservoirs):
        rng.shuffle(reservoirs[key])
    ordered: list[dict[str, Any]] = []
    index = 0
    keys = sorted(reservoirs)
    while True:
        added = False
        for key in keys:
            if index < len(reservoirs[key]):
                ordered.append(reservoirs[key][index])
                added = True
        if not added:
            break
        index += 1
    return ordered


def _write_selected_rows(
    rows: Iterable[dict[str, Any]],
    path: str | Path,
) -> tuple[int, str]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    digest = hashlib.sha256()
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False) + "\n"
            digest.update(line.encode("utf-8"))
            handle.write(line)
            count += 1
    temporary.replace(path)
    return count, digest.hexdigest()


def _tokenizer_fingerprint(tokenizer: Any, tokenizer_name_or_path: str | None, chat_end_token: str) -> dict[str, Any]:
    source = str(tokenizer_name_or_path or getattr(tokenizer, "name_or_path", "unknown"))
    files: dict[str, str] = {}
    base = Path(source)
    if base.is_dir():
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"):
            path = base / name
            if path.is_file():
                files[name] = _sha256_file(path)
    return {
        "name_or_path": source,
        "chat_end_token": str(chat_end_token),
        "files_sha256": files,
        "fingerprint": _json_hash([source, str(chat_end_token), files]),
    }


def _tokenize_records(
    tokenizer: Any,
    records: list[dict[str, Any]],
    *,
    max_length: int,
    filter_overlong: bool,
    store_tokenized: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Batch-render and batch-tokenize all candidates for a group of records."""
    prompts: list[str] = []
    candidate_counts: list[int] = []
    for record in records:
        prefix = candidate_prefix(record)
        entries = candidate_prompts_from_prefix(record, prefix)
        candidate_counts.append(len(entries))
        prompts.extend(apply_candidate_chat_template(tokenizer, prompt) for prompt in entries)

    encoded = tokenizer(prompts, padding=False, truncation=False)
    input_ids = encoded["input_ids"]
    if len(input_ids) != len(prompts):
        raise RuntimeError("Tokenizer returned a different number of candidate rows")

    accepted: list[dict[str, Any]] = []
    offset = 0
    stats = {"accepted": 0, "rejected_overlong": 0, "candidates": 0, "tokens": 0}
    for record, count in zip(records, candidate_counts):
        row_ids = input_ids[offset : offset + count]
        offset += count
        lengths = [len(ids) for ids in row_ids]
        max_seen = max(lengths, default=0)
        if max_seen > max_length:
            if filter_overlong:
                stats["rejected_overlong"] += 1
                continue
            raise ValueError(
                f"{record.get('id', '<unknown>')}: candidate token length {max_seen} exceeds max_length={max_length}"
            )
        flat: list[int] = []
        offsets = [0]
        for ids in row_ids:
            flat.extend(int(token) for token in ids)
            offsets.append(len(flat))
        row: dict[str, Any] = {
            "id": record["id"],
            "kind": record["kind"],
            "options": record["options"],
            "target": record["target"],
            "state": _state_text(record["state"]),
            "question": record["question"],
            "domain": record["domain"],
            "family": record["family"],
            "source": record["source"],
            "max_candidate_tokens": max_seen,
        }
        if store_tokenized:
            row.update(
                {
                    "input_ids_flat": flat,
                    "candidate_offsets": offsets,
                    "candidate_lengths": lengths,
                }
            )
        accepted.append(row)
        stats["accepted"] += 1
        stats["candidates"] += count
        stats["tokens"] += len(flat)
    return accepted, stats


def _batched_rows(
    path: str | Path,
    *,
    start_byte: int,
    max_rows: int,
    batch_size: int,
) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in _iter_jsonl_rows(
        path,
        start_byte=start_byte,
        max_rows=max_rows,
    ):
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _worker_init(
    tokenizer_name_or_path: str,
    chat_end_token: str,
    trust_remote_code: bool,
    max_length: int,
    filter_overlong: bool,
    tokenization_batch_size: int,
    progress_every: int,
    store_tokenized: bool,
) -> None:
    global _WORKER_TOKENIZER, _WORKER_CONFIG
    _WORKER_TOKENIZER = build_sft_tokenizer(
        tokenizer_name_or_path,
        chat_end_token=chat_end_token,
        trust_remote_code=trust_remote_code,
    )
    _WORKER_CONFIG = {
        "max_length": int(max_length),
        "filter_overlong": bool(filter_overlong),
        "tokenization_batch_size": int(tokenization_batch_size),
        "progress_every": int(progress_every),
        "store_tokenized": bool(store_tokenized),
    }


def _process_shard(spec: _ShardSpec) -> dict[str, Any]:
    shard_dir = Path(spec.shard_dir)
    done_path = shard_dir / "done.json"
    # The parent validates completed markers before scheduling; any submitted
    # shard is rebuilt atomically, including a stale or partial marker.
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    cache_dir = shard_dir.with_name(f".{shard_dir.name}.cache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    config = _WORKER_CONFIG
    tokenizer = _WORKER_TOKENIZER
    if tokenizer is None:
        raise RuntimeError("Jev worker tokenizer was not initialized")

    totals = {"processed": 0, "accepted": 0, "rejected_overlong": 0, "candidates": 0, "tokens": 0}
    started = time.perf_counter()

    def generator() -> Iterator[dict[str, Any]]:
        for batch in _batched_rows(
            spec.input_path,
            start_byte=spec.start_byte,
            max_rows=spec.row_count,
            batch_size=config["tokenization_batch_size"],
        ):
            rows, batch_stats = _tokenize_records(
                tokenizer,
                batch,
                max_length=config["max_length"],
                filter_overlong=config["filter_overlong"],
                store_tokenized=config["store_tokenized"],
            )
            totals["processed"] += len(batch)
            totals["accepted"] += batch_stats["accepted"]
            totals["rejected_overlong"] += batch_stats["rejected_overlong"]
            totals["candidates"] += batch_stats["candidates"]
            totals["tokens"] += batch_stats["tokens"]
            progress_every = config["progress_every"]
            if progress_every > 0 and totals["processed"] % progress_every < len(batch):
                elapsed = max(1e-9, time.perf_counter() - started)
                print(
                    f"[Jev progress] split={spec.split} shard={spec.shard_index + 1}/{spec.total_shards} "
                    f"processed={totals['processed']:,} accepted={totals['accepted']:,} "
                    f"rejected={totals['rejected_overlong']:,} candidates={totals['candidates']:,} "
                    f"rows/s={totals['processed'] / elapsed:.1f}",
                    flush=True,
                )
            yield from rows

    features = JEV_FEATURES if config["store_tokenized"] else JEV_RAW_FEATURES
    dataset = HFDataset.from_generator(
        generator,
        features=features,
        cache_dir=str(cache_dir),
        writer_batch_size=config["tokenization_batch_size"],
    )
    dataset.save_to_disk(str(shard_dir))
    elapsed = time.perf_counter() - started
    result = {
        "split": spec.split,
        "shard": spec.shard_index,
        "total_shards": spec.total_shards,
        "path": str(shard_dir),
        "input_sha256": spec.input_sha256,
        "pipeline_hash": spec.config["pipeline_hash"],
        "elapsed_seconds": elapsed,
        "rows_per_second": totals["processed"] / max(elapsed, 1e-9),
        **totals,
    }
    _write_json_atomic(done_path, result)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    print(
        f"[Jev shard] split={spec.split} shard={spec.shard_index + 1}/{spec.total_shards} "
        f"rows={totals['accepted']:,} tokens={totals['tokens']:,} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return {"split": spec.split, "shard": spec.shard_index, "reused": False, **result}


def _resolve_split_files(data_configs: list[dict[str, Any]]) -> list[tuple[str, Path, dict[str, Any]]]:
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


def _split_cap(split_name: str, cfg: dict[str, Any], caps: dict[str, int | None]) -> int | None:
    value = cfg.get("max_samples")
    if value is not None:
        return int(value) if int(value) > 0 else None
    return caps.get(split_name)


def _build_split_input(
    split_name: str,
    source_path: Path,
    source_info: _SourceInfo,
    cfg: dict[str, Any],
    *,
    cap: int | None,
    split_staging: Path,
    seed: int,
    training_sampling: str,
    sampling_strategy: str,
    progress_every: int,
) -> tuple[str, str, int | None, dict[str, Any]]:
    ratio = cfg.get("sample_ratio")
    ratio = float(ratio) if ratio is not None else 1.0
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"sample_ratio must be in (0, 1], got {ratio!r} for {source_path}")
    target: int | None = None
    if cap is not None or ratio < 1.0:
        target = source_info.line_count if ratio >= 1.0 else max(1, int(source_info.line_count * ratio))
        if cap is not None:
            target = min(target, cap)
    if target is None:
        return str(source_path), source_info.sha256, None, {"sampled": False}

    if sampling_strategy not in ("reservoir", "hash_ratio"):
        raise ValueError("sampling_strategy must be reservoir or hash_ratio")
    # A ratio-only sample can target hundreds of thousands of rows. Use
    # deterministic streaming hash selection in that case so we never hold a
    # large reservoir in Python; capped samples still use exact reservoir.
    effective_strategy = sampling_strategy
    if cap is None and ratio < 1.0 and sampling_strategy == "reservoir":
        effective_strategy = "hash_ratio"
    selected_path = split_staging / "selected.jsonl"
    selection_meta_path = split_staging / "selection.json"
    expected_meta = {
        "source_sha256": source_info.sha256,
        "target": target,
        "seed": seed,
        "balanced": training_sampling == "source_kind_round_robin" and split_name == "train",
        "strategy": effective_strategy,
    }
    if selected_path.is_file() and selection_meta_path.is_file():
        existing = json.loads(selection_meta_path.read_text(encoding="utf-8"))
        if existing == expected_meta and _sha256_file(selected_path) == existing.get("selected_sha256"):
            return str(selected_path), existing["selected_sha256"], target, {"sampled": True, **existing}

    if effective_strategy == "hash_ratio":
        # Deterministic hash selection, useful when a ratio is requested without
        # requiring a second pass over the source. `max_samples` still caps it.
        threshold = int(ratio * (2**64 - 1))
        selected_count = 0

        def hash_rows() -> Iterator[dict[str, Any]]:
            nonlocal selected_count
            for seen, row in enumerate(_iter_jsonl_rows(source_path), start=1):
                key = hashlib.blake2b(f"{seed}:{row['id']}".encode(), digest_size=8).digest()
                if int.from_bytes(key, "big") <= threshold:
                    selected_count += 1
                    yield row
                    if cap is not None and selected_count >= cap:
                        return
                if progress_every > 0 and seen % progress_every == 0:
                    print(f"[Jev sample] {source_path.name}: hash_seen={seen:,} selected={selected_count:,}", flush=True)

        selected: Iterable[dict[str, Any]] = hash_rows()
    else:
        balanced = training_sampling == "source_kind_round_robin" and split_name == "train"
        selected = _reservoir_sample(
            source_path,
            target,
            seed=seed,
            balanced=balanced,
            progress_every=progress_every,
        )
    rows, selected_sha = _write_selected_rows(selected, selected_path)
    meta = {**expected_meta, "selected_rows": rows, "selected_sha256": selected_sha}
    _write_json_atomic(selection_meta_path, meta)
    return str(selected_path), selected_sha, target, {"sampled": True, **meta}


def _assemble_split(split_name: str, shard_dirs: list[Path]) -> HFDataset:
    if not shard_dirs:
        raise ValueError(f"No completed shards for split {split_name!r}")
    datasets = [load_from_disk(str(path)) for path in sorted(shard_dirs)]
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def _summary_from_manifest(manifest: dict[str, Any], *, reused: bool = True) -> dict[str, Any]:
    splits = manifest.get("splits", {})
    return {
        "task": "jev",
        "schema_version": manifest.get("schema_version", JEV_SCHEMA_VERSION),
        "prepared_path": manifest.get("prepared_path"),
        "train_samples": splits.get("train", {}).get("accepted_rows", 0),
        "val_samples": splits.get("validation", {}).get("accepted_rows", 0),
        "calibration_samples": splits.get("calibration", {}).get("accepted_rows", 0),
        "test_samples": splits.get("test", {}).get("accepted_rows", 0),
        "test_set_30k_samples": splits.get("test_set_30k", {}).get("accepted_rows", 0),
        "ood_samples": splits.get("ood", {}).get("accepted_rows", 0),
        "overwritten": False,
        "reused": reused,
        "manifest": str(Path(manifest["prepared_path"]) / "manifest.json") if manifest.get("prepared_path") else None,
    }


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
    preprocessing_num_proc: int | str | None = "auto",
    tokenization_batch_size: int = 512,
    shard_size: int = 20000,
    progress_every: int = 10000,
    resume: str = "auto",
    sampling_strategy: str = "reservoir",
    store_tokenized: bool = True,
    tokenizer_name_or_path: str | None = None,
    chat_end_token: str = "<|im_end|>",
    trust_remote_code: bool = True,
) -> dict[str, Any]:
    """Prepare Jev corpus as streaming pre-tokenized Arrow shards.

    The source is never materialized as a full Python list. Each worker reads a
    byte range, tokenizes batches of records, and writes one Arrow shard. A
    matching completed output is returned unchanged; incomplete staging is
    resumed automatically.
    """
    output_path = Path(output_dir)
    staging_path = output_path.parent / f".{output_path.name}.jev-staging"
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if tokenization_batch_size <= 0:
        raise ValueError("tokenization_batch_size must be positive")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if progress_every < 0:
        raise ValueError("progress_every must be nonnegative")
    if resume not in ("auto", "off"):
        raise ValueError("resume must be auto or off")
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

    tokenizer_info = _tokenizer_fingerprint(tokenizer, tokenizer_name_or_path, chat_end_token)
    pipeline_config = {
        "schema_version": JEV_SCHEMA_VERSION,
        "tokenizer": tokenizer_info,
        "max_length": int(max_length),
        "filter_overlong": bool(filter_overlong),
        "training_sampling": training_sampling,
        "sampling_strategy": sampling_strategy,
        "tokenization_batch_size": int(tokenization_batch_size),
        "shard_size": int(shard_size),
        "store_tokenized": bool(store_tokenized),
        "seed": int(seed),
        "caps": split_caps,
        "sources": [{"path": str(path), "max_samples": cfg.get("max_samples"), "sample_ratio": cfg.get("sample_ratio")} for _, path, cfg in split_files],
    }
    pipeline_hash = _json_hash(pipeline_config)

    if output_path.exists():
        manifest_path = output_path / "manifest.json"
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("pipeline_hash") == pipeline_hash:
                console.print(f"[bold green][Jev Prepare][/bold green] reusing completed dataset: {output_path}")
                return _summary_from_manifest(existing)
            if not overwrite:
                raise ValueError(
                    "Existing Jev prepared dataset has different source/tokenizer/config. "
                    "Set data.overwrite=true or use a new prepared_path."
                )
        elif not overwrite:
            raise ValueError(f"Existing Jev output is incomplete: {output_path}")
        console.print(f"[yellow][Jev Prepare][/yellow] removing incompatible output: {output_path}")
        shutil.rmtree(output_path)
    if resume == "off" and staging_path.exists():
        shutil.rmtree(staging_path)

    split_meta: dict[str, dict[str, Any]] = {}
    shard_specs: list[_ShardSpec] = []
    staging_path.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()

    for split_name, source_path, cfg in split_files:
        canonical_split = "validation" if split_name == "val" else split_name
        cap = _split_cap(canonical_split, cfg, split_caps)
        console.print(f"[cyan][Jev scan][/cyan] split={canonical_split} source={source_path}")
        info = _scan_source(source_path, shard_size)
        console.print(
            f"[cyan][Jev scan][/cyan] split={canonical_split} rows={info.line_count:,} "
            f"shards={len(info.shards)} sha256={info.sha256[:12]}…"
        )
        split_staging = staging_path / canonical_split
        split_staging.mkdir(parents=True, exist_ok=True)
        input_path, input_sha, sample_target, selection_meta = _build_split_input(
            canonical_split,
            source_path,
            info,
            cfg,
            cap=cap,
            split_staging=split_staging,
            seed=int(cfg.get("seed", seed)),
            training_sampling=training_sampling,
            sampling_strategy=sampling_strategy,
            progress_every=progress_every,
        )
        input_info = _scan_source(input_path, shard_size)
        split_meta[canonical_split] = {
            "source_path": str(source_path),
            "source_sha256": info.sha256,
            "source_bytes": info.size_bytes,
            "raw_rows": info.line_count,
            "input_path": input_path if not selection_meta.get("sampled") else f"reservoir:{source_path.name}",
            "input_sha256": input_sha,
            "sample_target": sample_target,
            "selection": selection_meta,
        }
        for shard_index, (start_byte, end_byte, row_count) in enumerate(input_info.shards):
            shard_specs.append(
                _ShardSpec(
                    split=canonical_split,
                    input_path=input_path,
                    input_sha256=input_sha,
                    start_byte=start_byte,
                    end_byte=end_byte,
                    row_count=row_count,
                    shard_index=shard_index,
                    total_shards=len(input_info.shards),
                    shard_dir=str(split_staging / f"shard-{shard_index:05d}"),
                    config={
                        "pipeline_hash": pipeline_hash,
                        "tokenizer_name_or_path": tokenizer_info["name_or_path"],
                        "chat_end_token": chat_end_token,
                        "trust_remote_code": bool(trust_remote_code),
                        "max_length": int(max_length),
                        "filter_overlong": bool(filter_overlong),
                        "tokenization_batch_size": int(tokenization_batch_size),
                        "progress_every": int(progress_every),
                        "store_tokenized": bool(store_tokenized),
                    },
                )
            )

    if not shard_specs:
        raise ValueError("Jev dataset preparation produced no shards")

    workers = resolve_num_proc(preprocessing_num_proc) or 1
    completed: list[dict[str, Any]] = []
    pending: list[_ShardSpec] = []
    for spec in shard_specs:
        done_path = Path(spec.shard_dir) / "done.json"
        if done_path.is_file():
            meta = json.loads(done_path.read_text(encoding="utf-8"))
            if (
                meta.get("input_sha256") == spec.input_sha256
                and meta.get("pipeline_hash") == pipeline_hash
            ):
                completed.append({**meta, "reused": True})
                continue
        pending.append(spec)

    console.print(
        f"[bold cyan][Jev Prepare][/bold cyan] splits={len(split_meta)} shards={len(shard_specs)} "
        f"pending={len(pending)} workers={workers} staging={staging_path}"
    )
    worker_config = shard_specs[0].config
    if pending:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(
                worker_config["tokenizer_name_or_path"],
                worker_config["chat_end_token"],
                worker_config["trust_remote_code"],
                worker_config["max_length"],
                worker_config["filter_overlong"],
                worker_config["tokenization_batch_size"],
                worker_config["progress_every"],
                worker_config["store_tokenized"],
            ),
        ) as executor:
            futures = {executor.submit(_process_shard, spec): spec for spec in pending}
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    raise RuntimeError(f"Jev shard failed: split={spec.split} shard={spec.shard_index}") from exc
                completed.append(result)
                console.print(
                    f"[bold green][Jev shard done][/bold green] split={result['split']} "
                    f"shard={result['shard'] + 1} accepted={result.get('accepted', 0):,} "
                    f"reused={result.get('reused', False)}"
                )

    # Assemble shard datasets per split and write a single DatasetDict.
    split_datasets: dict[str, HFDataset] = {}
    for split_name in sorted(split_meta):
        shard_dirs = [Path(spec.shard_dir) for spec in shard_specs if spec.split == split_name]
        split_datasets[split_name] = _assemble_split(split_name, shard_dirs)
        split_meta[split_name]["accepted_rows"] = len(split_datasets[split_name])
        split_meta[split_name]["candidate_count"] = sum(
            int(meta.get("candidates", 0)) for meta in completed if meta.get("split") == split_name
        )
        split_meta[split_name]["token_count"] = sum(
            int(meta.get("tokens", 0)) for meta in completed if meta.get("split") == split_name
        )
        split_meta[split_name]["rejected_overlong"] = sum(
            int(meta.get("rejected_overlong", 0)) for meta in completed if meta.get("split") == split_name
        )
        split_meta[split_name]["shard_count"] = len(shard_dirs)
        split_meta[split_name]["reused_shards"] = sum(
            1 for meta in completed if meta.get("split") == split_name and meta.get("reused")
        )

    if not split_datasets.get("train"):
        raise ValueError("Jev dataset preparation produced no training rows")
    if "validation" not in split_datasets:
        if "calibration" in split_datasets:
            split_datasets["validation"] = split_datasets["calibration"]
        else:
            raise ValueError("Jev preparation needs a validation or calibration split")

    payload: dict[str, HFDataset] = {
        "train": split_datasets["train"],
        "val": split_datasets["validation"],
        "validation": split_datasets["validation"],
        "calibration": split_datasets.get("calibration", split_datasets["validation"]),
        "test": split_datasets.get("test", split_datasets["validation"]),
    }
    if "test_set_30k" in split_datasets:
        payload["test_set_30k"] = split_datasets["test_set_30k"]
    if "ood" in split_datasets:
        payload["ood"] = split_datasets["ood"]

    manifest = {
        "schema_version": JEV_SCHEMA_VERSION,
        "prepared_path": str(output_path.resolve()),
        "pipeline_hash": pipeline_hash,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": pipeline_config,
        "tokenizer": tokenizer_info,
        "workers": workers,
        "splits": split_meta,
        "elapsed_seconds": time.perf_counter() - started_all,
    }
    dataset = DatasetDict(payload)
    _save_dataset_dict(dataset, output_path)
    _write_json_atomic(output_path / "manifest.json", manifest)
    if staging_path.exists():
        shutil.rmtree(staging_path)

    summary = _summary_from_manifest(manifest, reused=False)
    summary["overwritten"] = True
    console.print(
        f"[bold green][Jev Prepare][/bold green] saved {output_path} "
        f"(train={len(payload['train']):,}, val={len(payload['val']):,}, "
        f"calibration={len(payload['calibration']):,}, test={len(payload['test']):,}, "
        f"test_set_30k={len(payload.get('test_set_30k', [])):,}, ood={len(payload.get('ood', [])):,})"
    )
    return summary


__all__ = [
    "JEV_FEATURES",
    "JEV_RAW_FEATURES",
    "JEV_REQUIRED_KEYS",
    "JEV_SCHEMA_VERSION",
    "JEV_SPLITS",
    "prepare_jev_dataset",
    "validate_jev_row",
]
