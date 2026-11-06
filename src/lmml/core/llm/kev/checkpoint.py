"""Save and load Kev checkpoints in the original ``head.pt`` layout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from lmml.core.llm.kev.model import KevDecisionModel


def save_kev_checkpoint(
    model: KevDecisionModel,
    tokenizer: Any,
    output_dir: str | Path,
    *,
    base_model: str,
    mode: str,
    lora_r: int = 0,
    lora_alpha: int = 0,
    pointer_dim: int = 256,
    option_isolation: bool = False,
    temperature: float = 1.0,
    extra: dict[str, Any] | None = None,
) -> str:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    unwrapped = getattr(model, "module", model)
    unwrapped.lm.save_pretrained(str(path), safe_serialization=True)
    tokenizer.save_pretrained(str(path))
    meta = {
        "base": str(base_model),
        "head": {key: value.detach().cpu() for key, value in unwrapped.head.state_dict().items()},
        "lora": int(lora_r if mode == "lora" else 0),
        "lora_alpha": int(lora_alpha),
        "head_dim": int(pointer_dim),
        "option_isolation": bool(option_isolation),
        "weights_dtype": str(unwrapped.torch_dtype).replace("torch.", ""),
        "temperature": float(temperature),
        "weights": "lora" if mode == "lora" else "full",
        "extra": extra or {},
    }
    torch.save(meta, path / "head.pt")
    config_meta = {key: value for key, value in meta.items() if key != "head"}
    (path / "kev_config.json").write_text(
        json.dumps(config_meta, default=str, indent=2), encoding="utf-8"
    )
    return str(path.resolve())


def read_kev_meta(path: str | Path) -> dict[str, Any]:
    meta_path = Path(path) / "head.pt"
    if not meta_path.exists():
        raise FileNotFoundError(f"not a Kev checkpoint (missing head.pt): {path}")
    return torch.load(meta_path, map_location="cpu", weights_only=False)


def load_kev_model(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype | str = torch.float32,
    mode: str | None = None,
    base_model: str | Path | None = None,
) -> tuple[KevDecisionModel, Any]:
    """Load a full-weight checkpoint or a LoRA adapter plus pointer head."""
    path = Path(path)
    meta = read_kev_meta(path)
    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    checkpoint_mode = str(mode or meta.get("weights", "lora")).lower()
    if checkpoint_mode == "full":
        model = KevDecisionModel(
            str(path),
            tokenizer,
            mode="full",
            torch_dtype=torch_dtype,
            pointer_dim=int(meta.get("head_dim", 256)),
            option_isolation=bool(meta.get("option_isolation", False)),
        )
    else:
        model = KevDecisionModel(
            str(base_model or meta["base"]),
            tokenizer,
            mode="lora",
            torch_dtype=torch_dtype,
            lora_r=int(meta.get("lora", 16) or 16),
            lora_alpha=int(meta.get("lora_alpha", 2 * int(meta.get("lora", 16) or 16))),
            pointer_dim=int(meta.get("head_dim", 256)),
            option_isolation=bool(meta.get("option_isolation", False)),
        )
        from peft import load_peft_weights, set_peft_model_state_dict

        weights = load_peft_weights(str(path), device="cpu")
        set_peft_model_state_dict(model.lm, weights)
    model.head.load_state_dict(meta["head"])
    model.head.temperature = float(meta.get("temperature", 1.0))
    model.to(device)
    model.eval()
    return model, tokenizer


def warm_start_kev(
    model: KevDecisionModel,
    path: str | Path,
) -> dict[str, Any]:
    meta = read_kev_meta(path)
    if str(meta.get("weights", "lora")) != "full":
        from peft import load_peft_weights, set_peft_model_state_dict

        weights = load_peft_weights(str(path), device="cpu")
        set_peft_model_state_dict(model.lm, weights)
    else:
        from safetensors.torch import load_file

        checkpoint_path = Path(path)
        index_path = checkpoint_path / "model.safetensors.index.json"
        if index_path.exists():
            import json as _json

            weight_map = _json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
            state = {}
            for shard in sorted(set(weight_map.values())):
                state.update(load_file(str(checkpoint_path / shard), device="cpu"))
        else:
            state = load_file(str(checkpoint_path / "model.safetensors"), device="cpu")
        model.lm.load_state_dict(state, strict=False)
    model.head.load_state_dict(meta["head"])
    return {"resolved": str(Path(path).resolve()), "temperature": float(meta.get("temperature", 1.0))}


__all__ = ["load_kev_model", "read_kev_meta", "save_kev_checkpoint", "warm_start_kev"]
