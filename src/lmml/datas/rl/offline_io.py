"""离线转移数据集读写（.npz 必选，.h5/.hdf5 与 .parquet 为可选 extra）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "obs": ("obs", "states", "observations", "state"),
    "action": ("action", "actions", "act"),
    "reward": ("reward", "rewards", "rew"),
    "next_obs": ("next_obs", "next_states", "next_observations", "next_state"),
    "terminated": ("terminated", "dones", "done", "terminals"),
    "truncated": ("truncated", "timeouts", "trunc"),
}


def _pick_key(raw: dict[str, Any], target: str, required: bool = True) -> np.ndarray | None:
    for k in _KEY_ALIASES[target]:
        if k in raw:
            return np.asarray(raw[k])
    if required:
        raise KeyError(
            f"missing required field {target!r} (tried aliases {_KEY_ALIASES[target]}) in {list(raw)}"
        )
    return None


def normalize_transition_dict(raw: dict[str, Any]) -> dict[str, np.ndarray]:
    obs = _pick_key(raw, "obs")
    action = _pick_key(raw, "action").astype(np.int64, copy=False)
    reward = _pick_key(raw, "reward").astype(np.float32, copy=False)
    next_obs = _pick_key(raw, "next_obs")
    terminated = _pick_key(raw, "terminated").astype(bool, copy=False)
    truncated = _pick_key(raw, "truncated", required=False)
    if truncated is None:
        truncated = np.zeros(len(obs), dtype=bool)
    else:
        truncated = truncated.astype(bool, copy=False)
    out = {
        "obs": obs,
        "action": action,
        "reward": reward,
        "next_obs": next_obs,
        "terminated": terminated,
        "truncated": truncated,
    }
    if "discount" in raw:
        out["discount"] = np.asarray(raw["discount"], dtype=np.float32)
    return out


def load_transitions(path: str | Path) -> dict[str, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"offline dataset not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".npz":
        with np.load(p, allow_pickle=False) as f:
            raw = {k: f[k] for k in f.files}
        return normalize_transition_dict(raw)
    if suffix in (".h5", ".hdf5"):
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "Reading .h5/.hdf5 requires h5py; install with `pip install lmml[h5]`"
            ) from e
        with h5py.File(p, "r") as f:
            raw = {k: f[k][...] for k in f}
        return normalize_transition_dict(raw)
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError(
                "Reading .parquet requires pyarrow; install with `pip install lmml[parquet]`"
            ) from e
        tbl = pq.read_table(p)
        raw = {col: tbl[col].to_numpy(zero_copy_only=False) for col in tbl.column_names}
        return normalize_transition_dict(raw)
    raise ValueError(
        f"unsupported offline file format {suffix!r} (expected .npz, .h5/.hdf5, or .parquet)"
    )


def save_transitions(path: str | Path, data: dict[str, np.ndarray]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lower()
    if suffix == ".npz":
        np.savez_compressed(p, **data)
        return
    if suffix in (".h5", ".hdf5"):
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "Writing .h5/.hdf5 requires h5py; install with `pip install lmml[h5]`"
            ) from e
        with h5py.File(p, "w") as f:
            for k, v in data.items():
                f.create_dataset(k, data=np.asarray(v), compression="gzip")
        return
    if suffix == ".parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError(
                "Writing .parquet requires pyarrow; install with `pip install lmml[parquet]`"
            ) from e
        pq.write_table(pa.table({k: np.asarray(v) for k, v in data.items()}), p)
        return
    raise ValueError(
        f"unsupported offline file format {suffix!r} (expected .npz, .h5/.hdf5, or .parquet)"
    )
