"""Common Lightning utilities for RL modules."""

from __future__ import annotations

from typing import Any

import torch

from lmml.core.llm.rl_common.rollout import TorchRolloutEngine


class ReferencePolicyMixin:
    """Keep policy/reference rollouts and optimizer plumbing consistent."""

    def _start_rollout(self, tokenizer: Any, model: torch.nn.Module) -> TorchRolloutEngine:
        return TorchRolloutEngine(model, tokenizer)

    def policy_model(self) -> torch.nn.Module:
        return getattr(self, "policy", getattr(self, "actor", self.model))


__all__ = ["ReferencePolicyMixin"]
