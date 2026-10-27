"""LLM 基座模型加载、注意力后端解析、梯度检查点与 LoRA / 全量微调装配工具。"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from rich.console import Console
from torch import nn
from transformers import AutoModelForCausalLM

console = Console()


def _ensure_peft_torchao_compat() -> None:
    """兼容 Colab 预装旧版 torchao (<0.16.0) 导致 peft.get_peft_model 抛出 ImportError 的问题。"""
    try:
        import peft.import_utils as peft_iu
        import peft.tuners.lora.torchao as peft_torchao

        peft_iu.is_torchao_available()
    except ImportError as exc:
        console.print(
            f"[yellow][Warning] 检测到不兼容的旧版 torchao ({exc})，已自动禁用 PEFT torchao 分发器。[/yellow]"
        )
        peft_iu.is_torchao_available = lambda: False  # type: ignore[assignment]
        peft_torchao.is_torchao_available = lambda: False  # type: ignore[assignment]


DEFAULT_QWEN35_LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def resolve_attn_implementation(requested_attn: str = "sdpa") -> str:
    """检测当前环境是否支持 FlashAttention-2，不支持则平滑降级至 PyTorch SDPA。"""
    req = str(requested_attn or "sdpa").strip()
    if req == "flash_attention_2":
        if importlib.util.find_spec("flash_attn") is not None:
            console.print("[bold green][Attention][/bold green] 已检测到 flash_attn 库，启用 FlashAttention-2 加速。")
            return "flash_attention_2"
        console.print(
            "[yellow][Warning] 未安装 flash-attn 库，已自动降级为 PyTorch 原生 SDPA (sdpa) 加速。[/yellow]"
        )
        return "sdpa"
    return req


def resolve_torch_dtype(dtype_name: str | torch.dtype | None = "bfloat16") -> torch.dtype | str:
    """将配置中的精度字符串解析为 torch.dtype。"""
    if isinstance(dtype_name, torch.dtype):
        return dtype_name
    name = str(dtype_name or "bfloat16").strip().lower()
    mapping: dict[str, torch.dtype | str] = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "auto": "auto",
    }
    return mapping.get(name, torch.bfloat16)


def resolve_lora_target_modules(
    model: nn.Module,
    requested_modules: Sequence[str] | None = None,
) -> list[str]:
    """根据模型实际包含的子模块名称过滤 target_modules，兼容 Qwen3.5 混合线性注意力与标准 Transformer。"""
    candidates = list(requested_modules or DEFAULT_QWEN35_LORA_TARGET_MODULES)
    all_module_names = [name for name, _ in model.named_modules() if name]
    leaf_names = {name.split(".")[-1] for name in all_module_names}
    matched = [
        mod
        for mod in candidates
        if mod in leaf_names or any(name.endswith(f".{mod}") for name in all_module_names)
    ]
    return matched if matched else candidates


def build_causal_lm(
    model_name_or_path: str,
    mode: str = "lora",
    torch_dtype: str | torch.dtype = "bfloat16",
    attn_implementation: str = "sdpa",
    use_gradient_checkpointing: bool = True,
    eos_token_id: int | None = None,
    trust_remote_code: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Sequence[str] | None = None,
) -> nn.Module:
    """加载 CausalLM 基座并按 mode ('lora' | 'full') 装配可训练参数与显存优化。"""
    finetune_mode = str(mode).strip().lower()
    if finetune_mode not in ("lora", "full"):
        raise ValueError(f"不支持的微调模式 mode={mode!r}，可选值为 'lora' 或 'full'。")

    actual_attn = resolve_attn_implementation(attn_implementation)
    resolved_dtype = resolve_torch_dtype(torch_dtype)

    model: Any = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=resolved_dtype,
        attn_implementation=actual_attn,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )

    if eos_token_id is not None:
        model.config.eos_token_id = int(eos_token_id)
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.eos_token_id = int(eos_token_id)

    if use_gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if finetune_mode == "lora" and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if finetune_mode == "lora":
        _ensure_peft_torchao_compat()
        target_modules = resolve_lora_target_modules(model, lora_target_modules)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)

    else:
        model.train()
        for param in model.parameters():
            param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0
    console.print(
        f"[bold cyan][{finetune_mode.upper()} SFT][/bold cyan] "
        f"可训练参数量: [bold green]{trainable_params:,}[/bold green] / "
        f"总参数量: [bold]{total_params:,}[/bold] ({ratio:.4f}%)"
    )
    return model
