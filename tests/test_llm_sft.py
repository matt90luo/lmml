from __future__ import annotations

import json

import torch
from hydra import compose, initialize_config_module

from lmml.core.runner import run
from lmml.datas.llm import (
    LengthGroupedSampler,
    UnifiedSFTDataModule,
    build_sft_tokenizer,
    create_synthetic_sft_datasets,
    create_tiny_qwen_model_and_tokenizer,
    encode_chatml_messages,
    normalize_sft_batch,
)


def test_sft_normalization_tokenization_and_datamodule_cache(tmp_path):
    data_root = tmp_path / "datasets"
    model_dir = create_tiny_qwen_model_and_tokenizer(tmp_path / "Qwen3.5-0.8B-Base")
    sources = create_synthetic_sft_datasets(
        data_root,
        n_ultrachat=20,
        n_python_codes=16,
        n_openr1_math=16,
    )
    tokenizer = build_sft_tokenizer(str(model_dir))

    # 1. 验证四类格式批量标准化 + 空样本单趟过滤
    raw_batch = {
        "messages": [
            [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
            [],
            None,
            None,
        ],
        "instruction": [None, None, "Write code", None],
        "input": [None, None, "x=1", None],
        "output": [None, None, "return x", None],
        "problem": [None, None, None, "2+2=?"],
        "solution": [None, None, None, "4"],
    }
    normed = normalize_sft_batch(raw_batch)
    assert len(normed["messages"]) == 3

    # 2. 验证 ChatML Prompt Masking 与无目标截断过滤
    encoded = encode_chatml_messages(
        messages=[
            {"role": "system", "content": "You are AI"},
            {"role": "user", "content": "Explain concept"},
            {"role": "assistant", "content": "Concept connects policy optimization."},
        ],
        tokenizer=tokenizer,
        max_length=128,
    )
    assert encoded is not None
    assert encoded["labels"][0] == -100
    assert any(lbl != -100 for lbl in encoded["labels"][1:])

    # 若截断后仅剩 user prompt，drop_no_target=True 应返回 None
    only_user_truncated = encode_chatml_messages(
        messages=[
            {"role": "user", "content": "Explain concept number about reinforcement learning " * 10},
            {"role": "assistant", "content": "Answer"},
        ],
        tokenizer=tokenizer,
        max_length=8,
        drop_no_target=True,
    )
    assert only_user_truncated is None

    # 3. 验证 UnifiedSFTDataModule 多进程预处理 + 本地 SSD Arrow 缓存 + 8 倍数对齐 Padding + LengthGroupedSampler
    cache_dir = tmp_path / "sft_cache"
    dm = UnifiedSFTDataModule(
        data_configs=sources,
        tokenizer=tokenizer,
        batch_size=4,
        val_batch_size=4,
        max_length=128,
        num_workers=0,
        preprocessing_num_proc=2,
        val_split_ratio=0.1,
        max_val_samples=8,
        pretokenize=True,
        group_by_length=True,
        pad_to_multiple_of=8,
        cache_dir=str(cache_dir),
        seed=42,
    )
    dm.setup("fit")
    assert dm.train_dataset is not None and dm.val_dataset is not None
    assert len(dm.train_dataset) + len(dm.val_dataset) == 50
    assert len(dm.train_dataset) == 45 and len(dm.val_dataset) == 5
    assert any(cache_dir.glob("sft_prepared_*"))

    train_loader = dm.train_dataloader()
    assert isinstance(train_loader.sampler, LengthGroupedSampler)
    first_batch = next(iter(train_loader))
    assert first_batch["input_ids"].shape[0] == 4
    assert first_batch["input_ids"].shape[1] % 8 == 0
    assert torch.any(first_batch["labels"][:, 1:] != -100)

    # 第二次构建命中本地 Arrow 缓存
    dm_cached = UnifiedSFTDataModule(
        data_configs=sources,
        tokenizer=tokenizer,
        batch_size=4,
        val_batch_size=4,
        max_length=128,
        num_workers=0,
        preprocessing_num_proc=1,
        val_split_ratio=0.1,
        max_val_samples=8,
        pretokenize=True,
        group_by_length=True,
        pad_to_multiple_of=8,
        cache_dir=str(cache_dir),
        seed=42,
    )
    dm_cached.setup("fit")
    assert len(dm_cached.train_dataset) == len(dm.train_dataset)


def test_peft_torchao_old_version_fallback(tmp_path, monkeypatch):
    import peft.import_utils as peft_iu
    import peft.tuners.lora.torchao as peft_torchao

    from lmml.models.llm import build_causal_lm

    model_dir = create_tiny_qwen_model_and_tokenizer(tmp_path / "tiny_qwen")

    def _raise_old_torchao():
        raise ImportError(
            "Found an incompatible version of torchao. Found version 0.10.0, "
            "but only versions above 0.16.0 are supported"
        )

    monkeypatch.setattr(peft_iu, "is_torchao_available", _raise_old_torchao)
    monkeypatch.setattr(peft_torchao, "is_torchao_available", _raise_old_torchao)

    model = build_causal_lm(str(model_dir), mode="lora", torch_dtype="float32")
    assert any("lora_" in n for n, p in model.named_parameters() if p.requires_grad)



def test_end_to_end_qwen35_lora_and_full_sft_recipes(tmp_path):
    base_dir = tmp_path / "colab_root"
    data_dir = base_dir / "datasets"
    model_root = base_dir / "models"
    create_tiny_qwen_model_and_tokenizer(model_root / "Qwen3.5-0.8B-Base")
    create_synthetic_sft_datasets(
        data_dir,
        n_ultrachat=16,
        n_python_codes=12,
        n_openr1_math=12,
    )

    # 1. LoRA SFT 端到端训练 + 验证仅保存 LoRA 权重 + 导出 PEFT Adapter
    with initialize_config_module(version_base="1.3", config_module="lmml.configs"):
        cfg_lora = compose(
            config_name="config",
            overrides=[
                "command=train",
                "experiment=qwen35_lora_sft",
                f"paths.base_dir={base_dir}",
                f"paths.base_data_dir={data_dir}",
                f"paths.base_model_dir={model_root}",
                "data.batch_size=4",
                "data.val_batch_size=4",
                "data.max_length=96",
                "data.num_workers=0",
                "data.preprocessing_num_proc=1",
                "algo.model.torch_dtype=float32",
                "trainer.accelerator=cpu",
                "trainer.precision=32-true",
                "trainer.max_epochs=1",
                "trainer.accumulate_grad_batches=1",
                "trainer.val_check_interval=500",
                "trainer.enable_progress_bar=false",
                "trainer.enable_model_summary=false",
                "logger.offline=true",
                f"logger.save_dir={base_dir / 'wandb_lora'}",
            ],
        )
    lora_val_loss = run(cfg_lora)
    assert isinstance(lora_val_loss, float)

    lora_ckpts = list((base_dir / "checkpoints_lora").glob("*.ckpt"))
    assert len(lora_ckpts) == 1
    ckpt_payload = torch.load(lora_ckpts[0], map_location="cpu", weights_only=False)
    assert all("lora_" in k for k in ckpt_payload["state_dict"])

    lora_export_dir = base_dir / "qwen_lora_sft_final"
    assert (lora_export_dir / "adapter_config.json").exists()
    assert (lora_export_dir / "tokenizer.json").exists()

    # 2. Full SFT 端到端训练 + 导出完整 HuggingFace 模型 + test + predict
    with initialize_config_module(version_base="1.3", config_module="lmml.configs"):
        cfg_full = compose(
            config_name="config",
            overrides=[
                "command=train",
                "experiment=qwen35_full_sft",
                f"paths.base_dir={base_dir}",
                f"paths.base_data_dir={data_dir}",
                f"paths.base_model_dir={model_root}",
                "data.batch_size=4",
                "data.val_batch_size=4",
                "data.max_length=96",
                "data.num_workers=0",
                "data.preprocessing_num_proc=1",
                "algo.model.torch_dtype=float32",
                "trainer.accelerator=cpu",
                "trainer.precision=32-true",
                "trainer.max_epochs=1",
                "trainer.accumulate_grad_batches=1",
                "trainer.val_check_interval=500",
                "trainer.enable_progress_bar=false",
                "trainer.enable_model_summary=false",
                "logger.offline=true",
                f"logger.save_dir={base_dir / 'wandb_full'}",
            ],
        )
    full_val_loss = run(cfg_full)
    assert isinstance(full_val_loss, float)

    full_ckpts = list((base_dir / "checkpoints_full").glob("*.ckpt"))
    assert len(full_ckpts) == 1
    full_export_dir = base_dir / "qwen_full_sft_final"
    assert (full_export_dir / "config.json").exists()
    assert (full_export_dir / "model.safetensors").exists()
    assert (full_export_dir / "tokenizer.json").exists()

    # 3. 基于导出的 Full SFT 检查点运行 test 与 predict
    with initialize_config_module(version_base="1.3", config_module="lmml.configs"):
        cfg_test = compose(
            config_name="config",
            overrides=[
                "command=test",
                "experiment=qwen35_full_sft",
                f"ckpt_path={full_ckpts[0]}",
                f"paths.base_dir={base_dir}",
                f"paths.base_data_dir={data_dir}",
                f"paths.base_model_dir={model_root}",
                "data.val_batch_size=4",
                "data.max_length=96",
                "data.num_workers=0",
                "algo.model.torch_dtype=float32",
                "trainer.accelerator=cpu",
                "trainer.precision=32-true",
                "trainer.enable_progress_bar=false",
                "trainer.enable_model_summary=false",
                "logger.offline=true",
                f"logger.save_dir={base_dir / 'wandb_test'}",
            ],
        )
    test_metrics = run(cfg_test)
    assert isinstance(test_metrics, dict) and "test/loss" in test_metrics and "test/ppl" in test_metrics

    pred_file = base_dir / "sft_preds.jsonl"
    with initialize_config_module(version_base="1.3", config_module="lmml.configs"):
        cfg_pred = compose(
            config_name="config",
            overrides=[
                "command=predict",
                "experiment=qwen35_full_sft",
                f"ckpt_path={full_ckpts[0]}",
                f"paths.base_dir={base_dir}",
                f"paths.base_data_dir={data_dir}",
                f"paths.base_model_dir={model_root}",
                "data.max_length=96",
                "data.num_workers=0",
                "algo.model.torch_dtype=float32",
                "algo.max_new_tokens=8",
                f"predict.output={pred_file}",
                "trainer.accelerator=cpu",
                "trainer.precision=32-true",
                "trainer.enable_progress_bar=false",
                "trainer.enable_model_summary=false",
                "logger.offline=true",
                f"logger.save_dir={base_dir / 'wandb_pred'}",
            ],
        )
    pred_res = run(cfg_pred)
    assert pred_file.exists()
    lines = [json.loads(line) for line in pred_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1 and "generated_token_ids" in lines[0]
    assert pred_res["output"] == str(pred_file.resolve())
