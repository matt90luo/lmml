"""Qwen3.8-27B independent candidate scoring model with two-stage LoRA + FP32 linear head."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

from lmml.core.llm.jev.api import apply_candidate_chat_template, candidate_prompts
from lmml.models.llm.builders import apply_liger_kernel_if_available, resolve_attn_implementation

DEFAULT_JEV_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
)

BACKBONE_DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def ensure_qwen35_cpu_linear_attention_fallback() -> None:
    """Unwrap FLA/Triton decorators on Qwen3.5/3.8 GatedDeltaNet when no CUDA driver is active."""
    if torch.cuda.is_available():
        return
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35_mod
    except ImportError:
        return

    def _unwrap(fn: Any) -> Any:
        while hasattr(fn, "__wrapped__"):
            fn = fn.__wrapped__
        return fn

    for attr in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        fn = getattr(qwen35_mod, attr, None)
        if fn is not None and hasattr(fn, "__wrapped__"):
            setattr(qwen35_mod, attr, _unwrap(fn))


def _resolve_adapter_dir(checkpoint_dir: str | Path) -> Path:
    path = Path(checkpoint_dir)
    if (path / "adapter" / "adapter_config.json").is_file():
        return path / "adapter"
    if (path / "adapter_config.json").is_file():
        return path
    raise FileNotFoundError(
        f"No PEFT adapter_config.json found in {path} or {path / 'adapter'}"
    )


def _resolve_head_path(checkpoint_dir: str | Path) -> Path | None:
    path = Path(checkpoint_dir)
    for candidate in (path / "head.pt", path.parent / "head.pt"):
        if candidate.is_file():
            return candidate
    return None


def _resolve_temperature(checkpoint_dir: str | Path) -> float | None:
    path = Path(checkpoint_dir)
    for candidate in (path / "temperature.json", path.parent / "temperature.json"):
        if candidate.is_file():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "temperature" in payload:
                return float(payload["temperature"])
    return None


def _extract_language_backbone_and_initial_head(
    base_model_path: str,
    tokenizer: Any,
    *,
    torch_dtype: torch.dtype,
    attn_implementation: str = "sdpa",
    trust_remote_code: bool = True,
) -> tuple[nn.Module, torch.Tensor | None, int]:
    """Load Qwen3.8/3.5 (or CausalLM), extract language_model, and compute Yes-No head init."""
    ensure_qwen35_cpu_linear_attention_fallback()
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=trust_remote_code)
    architectures = getattr(config, "architectures", None) or []
    is_multimodal = (
        getattr(config, "model_type", "") == "qwen3_5"
        or any("ConditionalGeneration" in arch or "ImageTextToText" in arch for arch in architectures)
    )

    actual_attn = resolve_attn_implementation(attn_implementation)
    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "attn_implementation": actual_attn,
        "trust_remote_code": trust_remote_code,
    }
    device_map = os.environ.get("JEV_DEVICE_MAP")
    if device_map:
        load_kwargs["device_map"] = (
            json.loads(device_map) if device_map.lstrip().startswith("{") else device_map
        )

    def _load_full_model() -> Any:
        if is_multimodal:
            return AutoModelForImageTextToText.from_pretrained(base_model_path, **load_kwargs)
        try:
            return AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)
        except (OSError, ValueError, KeyError):
            return AutoModel.from_pretrained(base_model_path, **load_kwargs)

    try:
        full = _load_full_model()
    except (ImportError, ValueError) as exc:
        if actual_attn == "sdpa":
            raise
        print(
            f"[Attention] {actual_attn} 不可用于 {base_model_path}，"
            f"自动回退到 SDPA: {type(exc).__name__}: {exc}"
        )
        load_kwargs["attn_implementation"] = "sdpa"
        full = _load_full_model()

    initial_vector: torch.Tensor | None = None
    out_emb = full.get_output_embeddings() if hasattr(full, "get_output_embeddings") else None
    if out_emb is not None and getattr(out_emb, "weight", None) is not None and tokenizer is not None:
        yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
        no_ids = tokenizer.encode("No", add_special_tokens=False)
        if len(yes_ids) == 1 and len(no_ids) == 1 and out_emb.weight.device.type != "meta":
            weight = out_emb.weight
            if max(yes_ids[0], no_ids[0]) < weight.shape[0]:
                initial_vector = (weight[yes_ids[0]] - weight[no_ids[0]]).detach().float().clone()

    if hasattr(full, "model") and hasattr(full.model, "language_model"):
        backbone = full.model.language_model
    elif hasattr(full, "language_model"):
        backbone = full.language_model
    elif hasattr(full, "model"):
        backbone = full.model
    else:
        backbone = full

    text_cfg = getattr(backbone, "config", config)
    if hasattr(text_cfg, "text_config") and text_cfg.text_config is not None:
        text_cfg = text_cfg.text_config
    hidden_size = int(text_cfg.hidden_size)
    del full
    return backbone, initial_vector, hidden_size


class JevDecisionModel(nn.Module):
    """Independent candidate scorer over Qwen3.8 language_model + LoRA + FP32 Linear(hidden_size, 1)."""

    def __init__(
        self,
        base_model_path: str,
        tokenizer: Any | None = None,
        *,
        lora_config: Any | None = None,
        lora_checkpoint_dir: str | Path | None = None,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] | None = None,
        max_length: int = 512,
        torch_dtype: str | torch.dtype = "bfloat16",
        attn_implementation: str = "sdpa",
        gradient_checkpointing: bool = True,
        trust_remote_code: bool = True,
        is_trainable: bool = True,
        strict_length: bool = True,
        use_liger_kernel: bool = False,
    ) -> None:
        super().__init__()
        from peft import LoraConfig, PeftModel, get_peft_model

        self.base_model_path = str(base_model_path)
        self.max_length = int(max_length)
        self.strict_length = bool(strict_length)
        self.temperature: float = 1.0
        self.last_input_tokens: int = 0

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=trust_remote_code,
            )
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if isinstance(torch_dtype, torch.dtype):
            resolved_dtype = torch_dtype
        else:
            resolved_dtype = BACKBONE_DTYPES.get(str(torch_dtype).lower(), torch.bfloat16)
        self.torch_dtype = resolved_dtype

        # 1. Load and freeze language_model backbone
        backbone, initial_head, hidden_size = _extract_language_backbone_and_initial_head(
            self.base_model_path,
            self.tokenizer,
            torch_dtype=resolved_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        )
        self.hidden_size = hidden_size
        backbone.config.use_cache = False
        for param in backbone.parameters():
            param.requires_grad = False

        # Liger patches the language model before PEFT wraps it. Jev has no LM
        # head, so only RMSNorm/SwiGLU fusion is enabled (FLCE stays off).
        self.liger_enabled = apply_liger_kernel_if_available(
            backbone,
            use_liger_kernel=bool(use_liger_kernel),
            fused_linear_cross_entropy=False,
            rms_norm=True,
            swiglu=True,
        )

        # 2. Initialize FP32 Linear(hidden_size, 1) scalar head
        self.head = nn.Linear(hidden_size, 1, bias=True, dtype=torch.float32)
        with torch.no_grad():
            if initial_head is not None and initial_head.numel() == hidden_size:
                self.head.weight.copy_(initial_head.unsqueeze(0))
            else:
                nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
            self.head.bias.zero_()
        self.head.requires_grad_(bool(is_trainable))

        # 3. Two-stage LoRA branch: load existing checkpoint or initialize fresh LoRA
        ckpt_dir_str = str(lora_checkpoint_dir) if lora_checkpoint_dir else ""
        if ckpt_dir_str and os.path.exists(ckpt_dir_str):
            adapter_dir = _resolve_adapter_dir(ckpt_dir_str)
            print(f"[Init] Loading existing LoRA from: {adapter_dir}")
            self.backbone = PeftModel.from_pretrained(
                backbone,
                str(adapter_dir),
                is_trainable=bool(is_trainable),
            )
            adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
            self.lora_r = int(adapter_cfg.get("r", lora_r))
            self.lora_alpha = int(adapter_cfg.get("lora_alpha", lora_alpha))
            self.lora_dropout = float(adapter_cfg.get("lora_dropout", lora_dropout))
            self.lora_target_modules = list(adapter_cfg.get("target_modules") or [])
            head_path = _resolve_head_path(ckpt_dir_str)
            if head_path is not None:
                head_state = torch.load(head_path, map_location="cpu", weights_only=True)
                if isinstance(head_state, Mapping) and "head" in head_state and isinstance(head_state["head"], Mapping):
                    head_state = head_state["head"]
                self.head.load_state_dict(head_state)
            temp = _resolve_temperature(ckpt_dir_str)
            if temp is not None:
                self.temperature = float(temp)
            self.init_mode = "checkpoint"
            self.loaded_from = str(Path(ckpt_dir_str).resolve())
        else:
            print("[Init] Initializing a brand-new LoRA adapter.")
            if lora_config is None:
                requested_targets = list(lora_target_modules or DEFAULT_JEV_LORA_TARGETS)
                available_leaves = {
                    name.rsplit(".", 1)[-1]
                    for name, module in backbone.named_modules()
                    if isinstance(module, nn.Linear)
                }
                resolved_targets = [t for t in requested_targets if t in available_leaves]
                if not resolved_targets:
                    raise ValueError(
                        f"None of the requested LoRA target modules {requested_targets} exist in backbone "
                        f"(available linear leaves: {sorted(available_leaves)})"
                    )
                lora_config = LoraConfig(
                    r=int(lora_r),
                    lora_alpha=int(lora_alpha),
                    target_modules=resolved_targets,
                    lora_dropout=float(lora_dropout),
                    bias="none",
                )
            else:
                available_leaves = {
                    name.rsplit(".", 1)[-1]
                    for name, module in backbone.named_modules()
                    if isinstance(module, nn.Linear)
                }
                if getattr(lora_config, "target_modules", None):
                    filtered = [t for t in lora_config.target_modules if t in available_leaves]
                    if filtered:
                        lora_config.target_modules = filtered
            self.backbone = get_peft_model(backbone, lora_config)
            self.lora_r = int(lora_config.r)
            self.lora_alpha = int(lora_config.lora_alpha)
            self.lora_dropout = float(lora_config.lora_dropout)
            self.lora_target_modules = list(lora_config.target_modules or [])
            self.init_mode = "fresh"
            self.loaded_from = None

        if gradient_checkpointing and is_trainable:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(self.backbone, "enable_input_require_grads"):
                self.backbone.enable_input_require_grads()

        if hasattr(self.backbone, "print_trainable_parameters"):
            self.backbone.print_trainable_parameters()

        if is_trainable:
            self.train()
        else:
            self.eval()

    def encode_records(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Render candidate prompts with chat_template and tokenize into a padded batch."""
        if not records:
            raise ValueError("encode_records requires at least one record")
        prompts: list[str] = []
        counts: list[int] = []
        for record in records:
            entries = candidate_prompts(record)
            counts.append(len(entries))
            for prompt in entries:
                prompts.append(apply_candidate_chat_template(self.tokenizer, prompt))

        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=not self.strict_length,
            max_length=self.max_length if not self.strict_length else None,
            return_tensors="pt",
        )
        lengths = encoded["attention_mask"].sum(dim=-1)
        max_seen = int(lengths.max().item()) if lengths.numel() > 0 else 0
        if self.strict_length and max_seen > self.max_length:
            raise ValueError(
                f"Input length {max_seen} exceeds max_length={self.max_length}; no silent truncation"
            )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "lengths": lengths,
            "counts": counts,
            "kinds": [str(r["kind"]) for r in records],
        }

    def forward_encoded(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lengths: torch.Tensor,
        counts: Sequence[int],
        kinds: Sequence[str],
    ) -> list[torch.Tensor]:
        """Forward tokenized candidates and return one logit vector per question record.

        Scoring path:
            chat_template(context, question, candidate)
            -> Qwen3.8-27B language_model + LoRA
            -> last non-padding token hidden state
            -> FP32 Linear(hidden_size, 1)
            -> candidate scalar score
        """
        ensure_qwen35_cpu_linear_attention_fallback()
        device = next(self.backbone.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        self.last_input_tokens = int(lengths.sum().item())

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        hdev = hidden_states.device
        batch_indices = torch.arange(input_ids.size(0), device=hdev)
        last_token_indices = (lengths.to(hdev) - 1).clamp(min=0)
        last_hidden = hidden_states[batch_indices, last_token_indices]

        head_device = self.head.weight.device
        scores = self.head(last_hidden.float().to(head_device)).squeeze(-1)

        logits: list[torch.Tensor] = []
        offset = 0
        for kind, count in zip(kinds, counts):
            values = scores[offset : offset + count]
            if kind == "noul":
                values = torch.stack([torch.zeros_like(values[0]), values[0]])
            logits.append(values)
            offset += count
        return logits

    def forward(self, records: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> list[torch.Tensor]:
        if isinstance(records, Mapping) and "input_ids" in records:
            return self.forward_encoded(
                input_ids=records["input_ids"],
                attention_mask=records["attention_mask"],
                lengths=records["lengths"],
                counts=records["counts"],
                kinds=records["kinds"],
            )
        assert isinstance(records, Sequence)
        encoded = self.encode_records(records)
        return self.forward_encoded(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            lengths=encoded["lengths"],
            counts=encoded["counts"],
            kinds=encoded["kinds"],
        )

    @torch.no_grad()
    def probabilities(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        temperature: float | None = None,
    ) -> list[list[float]]:
        was_training = self.training
        self.eval()
        try:
            temp = float(temperature if temperature is not None else self.temperature)
            logits = self.forward(records)
            return [torch.softmax(row.float() / temp, dim=-1).cpu().tolist() for row in logits]
        finally:
            self.train(was_training)

    def save(
        self,
        output_dir: str | Path,
        *,
        temperature: float | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> str:
        """Save Open-Jev compatible checkpoint directory (`adapter/`, `head.pt`, `model.json`, `temperature.json`)."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        adapter_dir = out / "adapter"
        self.backbone.save_pretrained(str(adapter_dir), safe_serialization=True)
        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(str(out))
        head_state = {k: v.detach().cpu() for k, v in self.head.state_dict().items()}
        torch.save(head_state, out / "head.pt")
        temp_value = float(temperature if temperature is not None else self.temperature)
        (out / "temperature.json").write_text(
            json.dumps({"temperature": temp_value}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        model_meta: dict[str, Any] = {
            "model_id": self.base_model_path,
            "max_length": self.max_length,
            "lora_rank": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "torch_dtype": str(self.torch_dtype).replace("torch.", ""),
            "temperature": temp_value,
            "method": "independent_candidate_lora_nll_brier",
            "use_liger_kernel": bool(self.liger_enabled),
            "extra": dict(extra or {}),
        }
        (out / "model.json").write_text(
            json.dumps(model_meta, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return str(out.resolve())

    @classmethod
    def load(
        cls,
        checkpoint_dir: str | Path,
        *,
        base_model_path: str | None = None,
        device: str | torch.device = "cpu",
        torch_dtype: str | torch.dtype | None = None,
        is_trainable: bool = False,
        max_length: int | None = None,
        strict_length: bool = True,
        use_liger_kernel: bool | None = None,
    ) -> JevDecisionModel:
        out = Path(checkpoint_dir)
        model_json = out / "model.json"
        meta = json.loads(model_json.read_text(encoding="utf-8")) if model_json.is_file() else {}
        resolved_base = str(base_model_path or meta.get("model_id") or "")
        if not resolved_base:
            adapter_cfg_path = _resolve_adapter_dir(out) / "adapter_config.json"
            adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
            resolved_base = str(adapter_cfg.get("base_model_name_or_path") or "")
        if not resolved_base:
            raise ValueError(f"Cannot determine base_model_path from checkpoint {out}")

        tokenizer_source = str(out) if (out / "tokenizer_config.json").is_file() else resolved_base
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        resolved_dtype = torch_dtype or meta.get("torch_dtype", "bfloat16")
        resolved_max_len = int(max_length or meta.get("max_length", 512))

        model = cls(
            base_model_path=resolved_base,
            tokenizer=tokenizer,
            lora_checkpoint_dir=out,
            max_length=resolved_max_len,
            torch_dtype=resolved_dtype,
            gradient_checkpointing=bool(is_trainable),
            is_trainable=bool(is_trainable),
            strict_length=strict_length,
            use_liger_kernel=bool(
                meta.get("use_liger_kernel", False) if use_liger_kernel is None else use_liger_kernel
            ),
        )
        model.to(device)
        if not is_trainable:
            model.eval()
        return model


__all__ = [
    "BACKBONE_DTYPES",
    "DEFAULT_JEV_LORA_TARGETS",
    "JevDecisionModel",
    "ensure_qwen35_cpu_linear_attention_fallback",
]
