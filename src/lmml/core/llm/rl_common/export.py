"""Checkpoint/export helpers shared by post-training recipes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel


def export_model(model: torch.nn.Module, tokenizer: Any, output_dir: str | Path) -> str:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    unwrapped = getattr(model, "module", model)
    if hasattr(unwrapped, "merge_and_unload"):
        unwrapped = unwrapped.merge_and_unload()
    if isinstance(unwrapped, PreTrainedModel) and hasattr(unwrapped, "save_pretrained"):
        model_to_save = unwrapped
    elif hasattr(unwrapped, "model") and isinstance(unwrapped.model, PreTrainedModel):
        model_to_save = unwrapped.model
    else:
        raise TypeError(f"cannot export model of type {type(unwrapped)}")
    model_to_save.save_pretrained(str(path), safe_serialization=True)
    tokenizer.save_pretrained(str(path))
    return str(path.resolve())


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    return -(probs * torch.log(probs.clamp(min=1e-12))).sum(-1)


__all__ = ["export_model", "token_entropy"]
