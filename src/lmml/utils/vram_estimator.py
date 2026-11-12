"""FakeTensor-based VRAM estimator for Qwen LoRA and Jev decision-model training.

The estimator never allocates real model weights on the GPU. It constructs the
model inside :class:`FakeTensorMode`, runs one full forward/backward/AdamW-step,
and tracks live fake storages with ``TorchDispatchMode``. Fake storage bytes are
device-independent, so the result is safe to compute on a CPU-only laptop before
submitting an expensive Colab job.
"""

from __future__ import annotations

import argparse
import gc
import json
import weakref
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    PretrainedConfig,
)

DEFAULT_LORA_TARGETS: tuple[str, ...] = (
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


class PeakMemoryTracker(TorchDispatchMode):
    """Track fake CUDA storages allocated by dispatched PyTorch operators.

    ``weakref.finalize`` is attached to the fake storage object. When the last
    tensor referencing a storage is collected, its bytes are removed from the
    live total. The high-water mark is the estimated tensor peak.
    """

    def __init__(self, target_device_type: str = "cuda") -> None:
        super().__init__()
        self.target_device_type = str(target_device_type)
        self.current_bytes: int = 0
        self.peak_bytes: int = 0
        self.live_storages: set[int] = set()

    def _on_storage_free(self, storage_id: int, nbytes: int) -> None:
        if storage_id in self.live_storages:
            self.live_storages.remove(storage_id)
            self.current_bytes = max(0, self.current_bytes - int(nbytes))

    def _track_tensor(self, obj: Any) -> None:
        if not isinstance(obj, torch.Tensor):
            return
        if obj.device.type != self.target_device_type:
            return
        storage = obj.untyped_storage()
        storage_id = id(storage)
        if storage_id in self.live_storages:
            return
        nbytes = int(storage.nbytes())
        if nbytes <= 0:
            return
        self.live_storages.add(storage_id)
        self.current_bytes += nbytes
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)
        weakref.finalize(storage, self._on_storage_free, storage_id, nbytes)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        tree_map(self._track_tensor, out)
        return out


@contextmanager
def _safe_fake_tensor_swap() -> Iterator[None]:
    """Allow PEFT's dtype/device conversion to work with tracked fake storages.

    PyTorch's ``swap_tensors`` refuses parameters that have a weakref attached
    (which the tracker intentionally does). For fake tensors there is no real
    storage ownership, so an in-place ``data`` assignment is equivalent for
    shape/dtype accounting. The original implementation is restored afterwards.
    """

    original = torch.utils.swap_tensors

    def _swap(t1: torch.Tensor, t2: torch.Tensor) -> None:
        try:
            original(t1, t2)
        except RuntimeError as exc:
            if "weakref" not in str(exc):
                raise
            t1.data = t2.data

    torch.utils.swap_tensors = _swap  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.utils.swap_tensors = original  # type: ignore[assignment]


def _resolve_config(
    model_config: AutoConfig | PretrainedConfig | str,
    *,
    trust_remote_code: bool = True,
) -> AutoConfig:
    if isinstance(model_config, PretrainedConfig):
        return model_config
    return AutoConfig.from_pretrained(str(model_config), trust_remote_code=trust_remote_code)


def _text_config(config: AutoConfig) -> AutoConfig:
    text_config = getattr(config, "text_config", None)
    return text_config if text_config is not None else config


def _is_multimodal_config(config: AutoConfig) -> bool:
    model_type = str(getattr(config, "model_type", ""))
    architectures = getattr(config, "architectures", None) or []
    return model_type in {"qwen3_5", "qwen3_5_moe"} or any(
        "ConditionalGeneration" in str(arch) or "ImageTextToText" in str(arch)
        for arch in architectures
    )


def _from_config(config: AutoConfig, *, dtype: torch.dtype) -> torch.nn.Module:
    kwargs = {
        "dtype": dtype,
        "attn_implementation": "sdpa",
    }
    try:
        if _is_multimodal_config(config):
            return AutoModelForImageTextToText.from_config(config, **kwargs)
        return AutoModelForCausalLM.from_config(config, **kwargs)
    except TypeError:
        # Older transformers versions use torch_dtype rather than dtype.
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = dtype
        if _is_multimodal_config(config):
            return AutoModelForImageTextToText.from_config(config, **kwargs)
        return AutoModelForCausalLM.from_config(config, **kwargs)


def _extract_language_model(full: torch.nn.Module) -> torch.nn.Module:
    if hasattr(full, "model") and hasattr(full.model, "language_model"):
        return full.model.language_model
    if hasattr(full, "language_model"):
        return full.language_model
    if hasattr(full, "model"):
        return full.model
    return full


def _hidden_size(model: torch.nn.Module, config: AutoConfig) -> int:
    model_config = getattr(model, "config", _text_config(config))
    if hasattr(model_config, "text_config") and model_config.text_config is not None:
        model_config = model_config.text_config
    return int(model_config.hidden_size)


def _make_lora_config(
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: Sequence[str] | None,
    *,
    task_type: str | None = None,
) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=list(lora_target_modules or DEFAULT_LORA_TARGETS),
        bias="none",
        task_type=task_type,
    )


def _build_training_model(
    config: AutoConfig,
    *,
    mode: str,
    dtype: torch.dtype,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: Sequence[str] | None,
    use_grad_ckpt: bool,
) -> tuple[torch.nn.Module, Any | None]:
    from peft import get_peft_model

    full = _from_config(config, dtype=dtype)
    if mode == "causal_lm":
        model = full
    else:
        backbone = _extract_language_model(full)
        for param in backbone.parameters():
            param.requires_grad = False
        model = get_peft_model(
            backbone,
            _make_lora_config(
                lora_r,
                lora_alpha,
                lora_dropout,
                lora_target_modules,
                task_type=None,
            ),
        )
        head = torch.nn.Linear(_hidden_size(backbone, config), 1, bias=True, dtype=torch.float32)
        # Jev's head lives outside the PEFT wrapper. Return it so the caller can
        # include its parameters in the optimizer and forward path.
        del full
        if use_grad_ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        return model, head

    if use_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return get_peft_model(
        model,
        _make_lora_config(
            lora_r,
            lora_alpha,
            lora_dropout,
            lora_target_modules,
            task_type="CAUSAL_LM",
        ),
    ), None


def estimate_peak_vram_gb(
    model_config: AutoConfig | PretrainedConfig | str,
    lora_config: Any | None = None,
    mini_batch: int = 1,
    seq_len: int = 512,
    dtype: torch.dtype = torch.bfloat16,
    use_grad_ckpt: bool = True,
    is_qlora_4bit: bool = False,
    *,
    mode: str = "causal_lm",
    candidates_per_question: int = 4,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    lora_target_modules: Sequence[str] | None = None,
    trust_remote_code: bool = True,
) -> float:
    """Estimate one training-step tensor peak in GiB.

    ``mode='causal_lm'`` models a standard CausalLM forward/backward/AdamW
    step. ``mode='jev'`` models the independent candidate scorer: each question
    expands to ``candidates_per_question`` flattened sequences, and the scalar
    head produces one score per candidate.
    """

    if mini_batch <= 0:
        raise ValueError("mini_batch must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if mode not in {"causal_lm", "jev"}:
        raise ValueError("mode must be 'causal_lm' or 'jev'")

    config = _resolve_config(model_config, trust_remote_code=trust_remote_code)
    config._attn_implementation = "sdpa"  # type: ignore[attr-defined]
    try:
        from lmml.core.llm.jev.model import ensure_qwen35_cpu_linear_attention_fallback

        ensure_qwen35_cpu_linear_attention_fallback()
    except ImportError:
        pass
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    gc.collect()

    # Fake CPU avoids requiring a real CUDA primary context during autograd.
    # Storage byte counts are device-independent and are the quantity we need.
    tracker = PeakMemoryTracker("cpu")
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    model: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    try:
        with fake_mode, tracker, _safe_fake_tensor_swap(), torch.device("cpu"):
            if lora_config is not None:
                lora_r = int(getattr(lora_config, "r", lora_r))
                lora_alpha = int(getattr(lora_config, "lora_alpha", lora_alpha))
                lora_dropout = float(getattr(lora_config, "lora_dropout", lora_dropout))
                lora_target_modules = list(getattr(lora_config, "target_modules", lora_target_modules or ()))

            model, head = _build_training_model(
                config,
                mode=mode,
                dtype=dtype,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_target_modules=lora_target_modules,
                use_grad_ckpt=use_grad_ckpt,
            )
            params = [p for p in model.parameters() if p.requires_grad]
            if head is not None:
                params.extend(p for p in head.parameters() if p.requires_grad)
            optimizer = torch.optim.AdamW(params, lr=1e-4)

            vocab_size = int(_text_config(config).vocab_size)
            batch = mini_batch * (candidates_per_question if mode == "jev" else 1)
            input_ids = torch.randint(
                0,
                max(2, min(vocab_size, 1024)),
                (batch, seq_len),
                device="cpu",
                dtype=torch.long,
            )
            attention_mask = torch.ones_like(input_ids)
            if mode == "causal_lm":
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
            else:
                assert head is not None
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                hidden = outputs.last_hidden_state
                # Use the final token for every flattened candidate, matching
                # JevDecisionModel.forward_encoded's readout.
                scores = head(hidden[:, -1, :].float()).squeeze(-1)
                question_scores = scores.reshape(mini_batch, candidates_per_question)
                log_probs = torch.log_softmax(question_scores.float(), dim=-1)
                target = torch.full_like(log_probs, 1.0 / candidates_per_question)
                loss = -(target * log_probs).sum(-1).mean()
                loss = loss + 0.1 * ((question_scores.softmax(-1) - target) ** 2).sum(-1).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            peak_bytes = int(tracker.peak_bytes)
            frozen_params = sum(
                p.numel()
                for p in (model.parameters() if model is not None else ())
                if not p.requires_grad
            )
    finally:
        torch.set_default_dtype(old_default_dtype)
        del model, optimizer
        gc.collect()

    if is_qlora_4bit:
        # Frozen BF16 weights -> approximate NF4 storage (0.55 byte/param).
        saved_bytes = int(frozen_params * (2.0 - 0.55))
        peak_bytes = max(0, peak_bytes - saved_bytes)
    return peak_bytes / (1024**3)


def find_optimal_mini_batch(
    model_name_or_path: str,
    lora_config: Any | None = None,
    seq_len: int = 512,
    gpu_total_vram_gb: float = 80.0,
    cuda_context_overhead_gb: float = 1.5,
    allocator_safety_ratio: float = 0.90,
    use_grad_ckpt: bool = True,
    is_qlora_4bit: bool = False,
    max_search_batch: int = 128,
    *,
    mode: str = "causal_lm",
    candidates_per_question: int = 4,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    lora_target_modules: Sequence[str] | None = None,
    trust_remote_code: bool = True,
) -> dict[str, Any]:
    """Binary-search the largest mini-batch that fits the configured budget."""

    if max_search_batch <= 0:
        raise ValueError("max_search_batch must be positive")
    usable_budget_gb = (float(gpu_total_vram_gb) - float(cuda_context_overhead_gb)) * float(
        allocator_safety_ratio
    )
    if usable_budget_gb <= 0:
        raise ValueError("GPU memory budget is not positive after context overhead")

    config = _resolve_config(model_name_or_path, trust_remote_code=trust_remote_code)
    low, high = 1, int(max_search_batch)
    best_batch = 0
    best_est_vram = 0.0
    while low <= high:
        mid = (low + high) // 2
        est = estimate_peak_vram_gb(
            config,
            lora_config=lora_config,
            mini_batch=mid,
            seq_len=seq_len,
            use_grad_ckpt=use_grad_ckpt,
            is_qlora_4bit=is_qlora_4bit,
            mode=mode,
            candidates_per_question=candidates_per_question,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            trust_remote_code=trust_remote_code,
        )
        print(
            f"[probe] mini_batch={mid:<3d} seq_len={seq_len} mode={mode} "
            f"-> tensor_peak={est:.2f} GiB (budget={usable_budget_gb:.2f} GiB)",
            flush=True,
        )
        if est <= usable_budget_gb:
            best_batch = mid
            best_est_vram = est
            low = mid + 1
        else:
            high = mid - 1

    return {
        "optimal_mini_batch": best_batch,
        "seq_len": int(seq_len),
        "mode": mode,
        "candidates_per_question": int(candidates_per_question) if mode == "jev" else 1,
        "estimated_tensor_vram_gb": round(best_est_vram, 2),
        "estimated_total_vram_gb": round(best_est_vram + float(cuda_context_overhead_gb), 2),
        "usable_budget_gb": round(usable_budget_gb, 2),
        "gpu_total_vram_gb": float(gpu_total_vram_gb),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model id or local config directory")
    parser.add_argument("--mode", choices=("causal_lm", "jev"), default="jev")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--vram-gb", type=float, default=80.0)
    parser.add_argument("--context-overhead-gb", type=float, default=1.5)
    parser.add_argument("--safety-ratio", type=float, default=0.90)
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--no-grad-ckpt", action="store_true")
    parser.add_argument("--qlora-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = find_optimal_mini_batch(
        args.model,
        seq_len=args.seq_len,
        gpu_total_vram_gb=args.vram_gb,
        cuda_context_overhead_gb=args.context_overhead_gb,
        allocator_safety_ratio=args.safety_ratio,
        use_grad_ckpt=not args.no_grad_ckpt,
        is_qlora_4bit=args.qlora_4bit,
        max_search_batch=args.max_batch,
        mode=args.mode,
        candidates_per_question=args.candidates,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
