"""零下载合成 SFT 数据集与本地微型 Qwen 模型生成器（用于本地 Smoke Test 与自动化测试）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen2Config


def create_synthetic_sft_datasets(
    base_data_dir: str | Path,
    n_ultrachat: int = 36,
    n_python_codes: int = 24,
    n_openr1_math: int = 24,
) -> list[dict[str, Any]]:
    """仿照 Colab 三套开源数据集的目录名称与原始字段格式生成本地合成数据（零网络下载）。

    1. ``ultrachat_200k``: Arrow ``save_to_disk`` 目录，包含多轮 ``messages`` / ``conversations``；
    2. ``python-codes-25k``: 包含 ``instruction`` / ``input`` / ``output`` 的 JSONL/Arrow 数据；
    3. ``OpenR1-Math-220k``: 包含 ``problem`` / ``solution``（及 ``question`` / ``response``）的数学推理数据。
    """
    root = Path(base_data_dir)
    root.mkdir(parents=True, exist_ok=True)

    # 1. ultrachat_200k (Arrow save_to_disk, messages 格式 + 少量空脏数据验证清洗)
    ultra_rows: list[dict[str, Any]] = []
    for i in range(n_ultrachat):
        extra_repeat = (i % 5) + 1
        ultra_rows.append(
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {
                        "role": "user",
                        "content": f"Explain concept number {i} about reinforcement learning and fine tuning.",
                    },
                    {
                        "role": "assistant",
                        "content": (
                            f"Concept {i} connects policy optimization with supervised fine tuning. "
                            * extra_repeat
                        ).strip(),
                    },
                ]
            }
        )
    # 混入 2 条空对话以测试单趟过滤
    ultra_rows.append({"messages": []})
    ultra_rows.append({"messages": [{"role": "user", "content": "Only user message without assistant"}]})
    ultra_dir = root / "ultrachat_200k"
    HFDataset.from_list(ultra_rows).save_to_disk(str(ultra_dir))

    # 2. python-codes-25k (Arrow save_to_disk, instruction + input + output 格式)
    py_rows: list[dict[str, Any]] = []
    for i in range(n_python_codes):
        py_rows.append(
            {
                "instruction": f"Write a Python function to compute value {i} in sequence.",
                "input": f"x = {i}",
                "output": f"def compute_{i}(x):\n    return x + {i} * 2\n",
            }
        )
    py_dir = root / "python-codes-25k"
    HFDataset.from_list(py_rows).save_to_disk(str(py_dir))

    # 3. OpenR1-Math-220k (Arrow save_to_disk, problem + solution 格式)
    math_rows: list[dict[str, Any]] = []
    for i in range(n_openr1_math):
        steps_text = " ".join([f"Step {k}: simplify term {k} for variable x." for k in range(1, (i % 6) + 2)])
        math_rows.append(
            {
                "problem": f"Solve the equation 2 * x + {i} = {3 * i} step by step.",
                "solution": f"{steps_text} Therefore the final answer is x = {i}.",
            }
        )
    math_dir = root / "OpenR1-Math-220k"
    HFDataset.from_list(math_rows).save_to_disk(str(math_dir))

    return [
        {"path": str(ultra_dir), "max_samples": n_ultrachat, "sample_ratio": None},
        {"path": str(py_dir), "max_samples": None, "sample_ratio": 1.0},
        {"path": str(math_dir), "max_samples": n_openr1_math, "sample_ratio": None},
    ]


def create_tiny_qwen_model_and_tokenizer(
    save_dir: str | Path,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
) -> Path:
    """在本地生成微型 Qwen 架构模型与支持 <|im_start|> / <|im_end|> 的分词器（零网络下载）。"""
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_words = [
        "<pad>",
        "<unk>",
        "<|im_start|>",
        "<|im_end|>",
        "\n",
        "system",
        "user",
        "assistant",
        "You",
        "are",
        "a",
        "helpful",
        "AI",
        "Explain",
        "concept",
        "number",
        "about",
        "reinforcement",
        "learning",
        "and",
        "fine",
        "tuning",
        "Concept",
        "connects",
        "policy",
        "optimization",
        "with",
        "supervised",
        "Write",
        "Python",
        "function",
        "to",
        "compute",
        "value",
        "in",
        "sequence",
        "def",
        "return",
        "x",
        "Solve",
        "the",
        "equation",
        "step",
        "by",
        "Step",
        "simplify",
        "term",
        "for",
        "variable",
        "Therefore",
        "final",
        "answer",
        "is",
        "LoRA",
        "full",
        "difference",
        "between",
        "sentences",
        ".",
        ":",
        "+",
        "*",
        "=",
        "(",
        ")",
    ]
    for idx in range(128):
        tok = str(idx)
        if tok not in base_words:
            base_words.append(tok)

    vocab = {w: i for i, w in enumerate(base_words)}
    # 补齐 vocab_size 到 256，方便张量对齐
    while len(vocab) < 256:
        vocab[f"tok_{len(vocab)}"] = len(vocab)

    tok_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok_obj.pre_tokenizer = Whitespace()

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        unk_token="<unk>",
        pad_token="<|im_end|>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    fast_tokenizer.save_pretrained(str(out_dir))

    cfg = Qwen2Config(
        vocab_size=len(vocab),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=2048,
        pad_token_id=vocab["<|im_end|>"],
        eos_token_id=vocab["<|im_end|>"],
        bos_token_id=vocab["<|im_start|>"],
    )
    model = AutoModelForCausalLM.from_config(cfg)
    model.save_pretrained(str(out_dir), safe_serialization=True)

    # 写入元信息便于检查
    (out_dir / "synthetic_meta.json").write_text(
        json.dumps({"synthetic": True, "vocab_size": len(vocab)}, indent=2),
        encoding="utf-8",
    )
    return out_dir


def create_tiny_minimind_model_and_tokenizer(
    save_dir: str | Path,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    num_hidden_layers: int = 1,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    use_moe: bool = False,
    num_experts: int = 2,
    max_position_embeddings: int = 256,
) -> Path:
    """Create an offline MiniMind checkpoint with the bundled 6400-token tokenizer."""
    import shutil
    from importlib.resources import as_file, files

    from lmml.models.llm.minimind import MiniMindConfig, MiniMindForCausalLM
    from lmml.models.llm.minimind_registration import register_minimind

    register_minimind()
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with as_file(files("lmml").joinpath("assets", "minimind")) as asset_dir:
        for name in ("tokenizer.json", "tokenizer_config.json"):
            shutil.copy2(asset_dir / name, out_dir / name)
    config = MiniMindConfig(
        vocab_size=6400,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=hidden_size // num_attention_heads,
        max_position_embeddings=max_position_embeddings,
        use_moe=use_moe,
        num_experts=num_experts,
        moe_intermediate_size=intermediate_size,
        tie_word_embeddings=True,
    )
    MiniMindForCausalLM(config).save_pretrained(str(out_dir), safe_serialization=True)
    return out_dir


def create_synthetic_minimind_datasets(
    root: str | Path,
    samples_per_task: int = 8,
) -> dict[str, Path]:
    """Create small pretrain/SFT/DPO/RLAIF/Agent JSONL fixtures."""
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    paths = {
        "pretrain": base / "pretrain.jsonl",
        "sft": base / "sft.jsonl",
        "dpo": base / "dpo.jsonl",
        "rlaif": base / "rlaif.jsonl",
        "agent_rl": base / "agent_rl.jsonl",
    }
    tools = [{"type": "function", "function": {"name": "calculate_math"}}]
    with (
        paths["pretrain"].open("w", encoding="utf-8") as pretrain,
        paths["sft"].open("w", encoding="utf-8") as sft,
        paths["dpo"].open("w", encoding="utf-8") as dpo,
        paths["rlaif"].open("w", encoding="utf-8") as rlaif,
        paths["agent_rl"].open("w", encoding="utf-8") as agent,
    ):
        for index in range(samples_per_task):
            pretrain.write(json.dumps({"text": f"Document {index}: MiniMind learns language modeling."}, ensure_ascii=False) + "\n")
            sft.write(
                json.dumps(
                    {
                        "conversations": [
                            {"role": "user", "content": f"What is {index}+2?"},
                            {"role": "assistant", "reasoning_content": "Add the numbers.", "content": str(index + 2)},
                        ]
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            dpo.write(
                json.dumps(
                    {
                        "prompt": f"Answer question {index}",
                        "chosen": f"<think>\nReason carefully.\n</think>\n\nAnswer {index}.",
                        "rejected": "I do not know.",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            rlaif.write(
                json.dumps(
                    {"conversations": [{"role": "user", "content": f"Explain concept {index}"}, {"role": "assistant", "content": "A useful explanation."}]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            agent.write(
                json.dumps(
                    {
                        "conversations": [
                            {"role": "system", "content": "Use tools.", "tools": tools},
                            {"role": "user", "content": f"Calculate {index}+2"},
                            {"role": "assistant", "content": str(index + 2)},
                        ],
                        "gt": [str(index + 2)],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return paths
