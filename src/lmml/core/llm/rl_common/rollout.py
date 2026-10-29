"""Native PyTorch rollout with per-token log-probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class RolloutResult:
    output_ids: torch.Tensor
    completion_ids: torch.Tensor
    per_token_logps: torch.Tensor
    completions: list[str]
    prompt_lens: torch.Tensor
    completion_mask: torch.Tensor


def per_token_logps(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    completion_length: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    unwrapped = getattr(model, "module", model)
    keep = int(completion_length) + 1
    outputs = unwrapped(
        input_ids,
        attention_mask=attention_mask,
        logits_to_keep=keep,
        use_cache=False,
    )
    logits = outputs.logits.float()
    # ``logits_to_keep=R+1`` returns logits for the last R+1 positions. Their
    # first R rows predict the completion tokens.
    logits = logits[:, : int(completion_length), :]
    targets = input_ids[:, -int(completion_length) :]
    values = F.log_softmax(logits, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return values


class TorchRolloutEngine:
    def __init__(self, policy_model: torch.nn.Module, tokenizer: Any) -> None:
        self.policy_model = policy_model
        self.tokenizer = tokenizer

    def update_policy(self, model: torch.nn.Module) -> None:
        self.policy_model = model

    @torch.no_grad()
    def rollout(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_generations: int = 1,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
    ) -> RolloutResult:
        model = getattr(self.policy_model, "module", self.policy_model)
        # Match MiniMind's rollout order: all generations of prompt 0, then
        # all generations of prompt 1, etc.
        prompt_ids.size(0)
        repeated_ids = prompt_ids.repeat_interleave(int(num_generations), dim=0)
        repeated_mask = attention_mask.repeat_interleave(int(num_generations), dim=0)
        generated = model.generate(
            input_ids=repeated_ids,
            attention_mask=repeated_mask,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=50,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # ``generate`` runs under inference_mode; clone to create ordinary
        # tensors that can be used by the subsequent PPO/GRPO loss forward.
        generated = generated.clone()
        prompt_len = int(prompt_ids.size(1))
        completion_ids = generated[:, prompt_len:]
        completion_mask = torch.ones_like(completion_ids, dtype=torch.long)
        eos = int(self.tokenizer.eos_token_id)
        for row in range(completion_ids.size(0)):
            hits = (completion_ids[row] == eos).nonzero(as_tuple=False).flatten()
            if hits.numel():
                first = int(hits[0].item()) + 1
                completion_mask[row, first:] = 0
                completion_ids[row, first:] = int(self.tokenizer.pad_token_id)
        full_mask = torch.cat(
            (repeated_mask, completion_mask), dim=1
        ).to(generated.device)
        if completion_ids.size(1) == 0:
            logps = generated.new_empty((generated.size(0), 0), dtype=torch.float32)
        else:
            logps = per_token_logps(
                self.policy_model,
                generated,
                torch.full((generated.size(0),), prompt_len, device=generated.device),
                completion_ids.size(1),
                attention_mask=full_mask,
            )
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return RolloutResult(
            output_ids=generated,
            completion_ids=completion_ids,
            per_token_logps=logps,
            completions=completions,
            prompt_lens=torch.full(
                (generated.size(0),), prompt_len, dtype=torch.long, device=generated.device
            ),
            completion_mask=completion_mask,
        )
