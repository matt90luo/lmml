"""观测预处理器：将 Discrete / Box(向量) / Box(C,H,W) 统一转换为张量并报告输出形状。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch import nn


def describe_space(space: spaces.Space) -> dict[str, Any]:
    """将 gym Space 转为可放入 Lightning hparams / YAML 的纯字典。"""
    if isinstance(space, spaces.Discrete):
        return {"type": "discrete", "n": int(space.n)}
    if isinstance(space, spaces.Box):
        return {
            "type": "box",
            "shape": [int(s) for s in space.shape],
            "low": float(np.min(space.low)),
            "high": float(np.max(space.high)),
        }
    raise TypeError(f"unsupported space type: {type(space)}")


def space_from_spec(
    spec: spaces.Space | Mapping[str, Any] | int | tuple[int, ...],
) -> dict[str, Any]:
    if isinstance(spec, spaces.Space):
        return describe_space(spec)
    if isinstance(spec, int):
        return {"type": "box", "shape": [int(spec)], "low": -1e9, "high": 1e9}
    if isinstance(spec, (tuple, list)):
        return {"type": "box", "shape": [int(x) for x in spec], "low": -1e9, "high": 1e9}
    return dict(spec)


class ObsPreprocessor(nn.Module):
    def __init__(self, obs_spec: spaces.Space | Mapping[str, Any] | int | tuple[int, ...]):
        super().__init__()
        spec = space_from_spec(obs_spec)
        self.kind = spec["type"]
        if self.kind == "discrete":
            self.n = int(spec["n"])
            self.out_shape: tuple[int, ...] = (self.n,)
        elif self.kind == "box":
            shape = tuple(int(s) for s in spec["shape"])
            self.n = 0
            self.out_shape = shape
        else:
            raise ValueError(f"unknown obs_spec kind {self.kind!r}")

    @property
    def flat_dim(self) -> int:
        return int(np.prod(self.out_shape))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.kind == "discrete":
            idx = obs.long()
            if idx.dim() > 1 and idx.size(-1) == 1:
                idx = idx.squeeze(-1)
            if idx.dim() == 0:
                idx = idx.unsqueeze(0)
            return F.one_hot(idx, num_classes=self.n).to(torch.float32)
        x = obs.to(torch.float32)
        if x.dim() == len(self.out_shape):
            x = x.unsqueeze(0)
        return x
