"""Rule reward plus optional external reward-model wrapper."""

from __future__ import annotations

import re

import torch


def _rep_penalty(text: str, n: int = 3, cap: float = 0.5) -> float:
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams))


def calculate_rule_reward(response: str) -> float:
    reward = 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
    answer = response
    if "</think>" in response:
        thinking, answer = response.split("</think>", 1)
        reward += 1.0 if 20 <= len(thinking.strip()) <= 300 else -0.5
        reward += 0.25 if response.count("</think>") == 1 else -0.25
    return reward - _rep_penalty(answer)


class RewardScorer:
    def __init__(self, mode: str = "rule_only", model_path: str | None = None) -> None:
        self.mode = str(mode).lower()
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        if self.mode == "hf":
            if not model_path:
                raise ValueError("reward model mode=hf requires model_path")
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=torch.float16
            ).eval()
            self.model.requires_grad_(False)

    @torch.no_grad()
    def score(self, prompt: str, response: str) -> float:
        total = calculate_rule_reward(response)
        if self.mode == "hf" and self.model is not None and self.tokenizer is not None:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            score = self.model.get_score(self.tokenizer, messages)
            total += float(score)
        elif self.mode == "mock":
            total += 0.01 * len(response)
        return max(-3.0, min(3.0, total))
