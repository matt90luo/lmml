"""Shared generation helper for RL Lightning modules."""

from __future__ import annotations

from typing import Any

import torch


@torch.no_grad()
def predict_from_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    batch: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    unwrapped = getattr(model, "module", model)
    input_ids = batch.get("input_ids")
    attention_mask = batch.get("attention_mask")
    if input_ids is None:
        rows = batch.get("prompt_ids") or batch.get("chosen_input_ids")
        if rows is None:
            raise ValueError("prediction batch lacks input_ids/prompt_ids")
        pad = int(tokenizer.pad_token_id)
        width = max(len(row) for row in rows)
        input_ids = torch.full((len(rows), width), pad, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[index, : len(row)] = 1
    input_ids = input_ids.to(next(unwrapped.parameters()).device)
    attention_mask = attention_mask.to(input_ids.device)
    generated = unwrapped.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        top_k=0,
        pad_token_id=int(tokenizer.pad_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
    )
    prompts = batch.get("prompt")
    if prompts is None:
        prompts = [tokenizer.decode(row, skip_special_tokens=True) for row in input_ids.tolist()]
    return {
        "prompts": prompts,
        "generated_token_ids": generated[:, input_ids.size(1) :].cpu().tolist(),
    }


__all__ = ["predict_from_batch"]
