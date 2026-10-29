"""Lightweight reward adapter used by PPO/GRPO and agent rollouts."""

from __future__ import annotations

from dataclasses import dataclass

from lmml.core.llm.rl_common.rewards import RewardScorer


@dataclass
class RewardConfig:
    mode: str = "rule_only"
    model_path: str | None = None
    device: str = "cpu"
    dtype: str = "float16"


class RewardAdapter:
    def __init__(self, config: RewardConfig) -> None:
        self.config = config
        self.scorer = RewardScorer(config.mode, config.model_path)
        if self.scorer.model is not None:
            self.scorer.model = self.scorer.model.to(config.device)

    def score(self, prompt: str, response: str) -> float:
        return float(self.scorer.score(prompt, response))


__all__ = ["RewardAdapter", "RewardConfig"]
