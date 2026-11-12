from __future__ import annotations

from transformers import Qwen2Config

from lmml.datas.llm.synthetic import create_tiny_qwen38_model_and_tokenizer
from lmml.utils.vram_estimator import estimate_peak_vram_gb, find_optimal_mini_batch


def _tiny_qwen2_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )


def test_estimate_causal_lm_peak_scales_and_gradient_checkpointing_helps():
    cfg = _tiny_qwen2_config()
    small = estimate_peak_vram_gb(
        cfg,
        mini_batch=1,
        seq_len=64,
        mode="causal_lm",
        lora_r=8,
        lora_alpha=16,
        lora_target_modules=["q_proj", "v_proj"],
    )
    large = estimate_peak_vram_gb(
        cfg,
        mini_batch=4,
        seq_len=64,
        mode="causal_lm",
        lora_r=8,
        lora_alpha=16,
        lora_target_modules=["q_proj", "v_proj"],
    )
    without_checkpointing = estimate_peak_vram_gb(
        cfg,
        mini_batch=2,
        seq_len=64,
        mode="causal_lm",
        use_grad_ckpt=False,
        lora_r=8,
        lora_alpha=16,
        lora_target_modules=["q_proj", "v_proj"],
    )
    with_checkpointing = estimate_peak_vram_gb(
        cfg,
        mini_batch=2,
        seq_len=64,
        mode="causal_lm",
        use_grad_ckpt=True,
        lora_r=8,
        lora_alpha=16,
        lora_target_modules=["q_proj", "v_proj"],
    )
    assert 0 < small < large
    assert with_checkpointing < without_checkpointing


def test_estimate_jev_and_find_batch(tmp_path):
    model_dir = create_tiny_qwen38_model_and_tokenizer(tmp_path / "tiny_qwen38")
    targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    estimate = estimate_peak_vram_gb(
        str(model_dir),
        mini_batch=2,
        seq_len=64,
        mode="jev",
        lora_r=8,
        lora_alpha=16,
        lora_target_modules=targets,
    )
    assert estimate > 0

    result = find_optimal_mini_batch(
        str(model_dir),
        seq_len=64,
        gpu_total_vram_gb=1.5,
        cuda_context_overhead_gb=1.0,
        allocator_safety_ratio=0.9,
        max_search_batch=4,
        mode="jev",
        lora_r=8,
        lora_alpha=16,
        lora_target_modules=targets,
    )
    assert result["optimal_mini_batch"] >= 1
    assert result["estimated_tensor_vram_gb"] > 0
    assert result["usable_budget_gb"] == 0.45
