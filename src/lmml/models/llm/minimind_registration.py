"""Register MiniMind with Hugging Face AutoClasses exactly once."""

from __future__ import annotations

from transformers import AutoConfig, AutoModelForCausalLM

from lmml.models.llm.minimind import MiniMindConfig, MiniMindForCausalLM

_registered = False


def register_minimind() -> None:
    global _registered
    if _registered:
        return
    AutoConfig.register(MiniMindConfig.model_type, MiniMindConfig, exist_ok=True)
    AutoModelForCausalLM.register(
        MiniMindConfig,
        MiniMindForCausalLM,
        exist_ok=True,
    )
    _registered = True


__all__ = ["register_minimind"]
