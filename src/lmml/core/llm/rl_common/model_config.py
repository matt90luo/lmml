"""Resolve MiniMind model construction options from an OmegaConf algo node."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


def resolve_model_config(model_cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    raw = model_cfg.get("config") if hasattr(model_cfg, "get") else None
    config = OmegaConf.to_container(raw, resolve=True) if raw is not None else {}
    config = dict(config or {})
    for key in ("architecture", "init_mode", "checkpoint_path"):
        value = model_cfg.get(key) if hasattr(model_cfg, "get") else None
        if value is not None:
            config[key] = value
    architecture = str(config.get("architecture") or "auto").lower()
    init_mode = str(config.get("init_mode") or "pretrained").lower()
    if (architecture == "minimind" or init_mode == "scratch") and "config" not in config:
        nested = {
            key: value
            for key, value in config.items()
            if key
            not in {"architecture", "init_mode", "checkpoint_path", "name_or_path", "torch_dtype"}
        }
        config = {
            "architecture": "minimind",
            "init_mode": init_mode,
            "checkpoint_path": config.get("checkpoint_path"),
            "config": nested,
        }
    return config


__all__ = ["resolve_model_config"]
