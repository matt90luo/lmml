from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from datasets import load_from_disk
from hydra import compose, initialize_config_module

from lmml.cli import main as cli_main
from lmml.core.llm.sft.recipe import CausalLMSFTRecipe
from lmml.core.runner import run
from lmml.datas.llm import (
    LengthGroupedSampler,
    UnifiedSFTDataModule,
    build_sft_tokenizer,
    create_synthetic_sft_datasets,
    create_tiny_qwen_model_and_tokenizer,
    encode_chatml_messages,
    normalize_sft_batch,
    prepare_llm_dataset,
)


def _use_null_logger(monkeypatch):
    monkeypatch.setattr("lmml.core.runner.build_logger", lambda cfg: True)


def test_sft_normalization_tokenization_and_prepared_datamodule(tmp_path):
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

    # 3. 独立预处理命令逻辑：批量清洗 + 预分词 + DatasetDict 落盘
    prepared_path = tmp_path / "prepared" / "qwen_sft_mix"
    summary = prepare_llm_dataset(
        data_configs=sources,
        tokenizer=tokenizer,
        output_dir=prepared_path,
        task="sft",
        max_length=128,
        val_split_ratio=0.1,
        max_val_samples=8,
        preprocessing_num_proc=1,
        seed=42,
    )
    assert (prepared_path / "dataset_dict.json").exists()
    assert summary["train_samples"] == 45 and summary["val_samples"] == 5

    # 4. 验证训练 DataModule 只读取预处理产物 + 8 倍数对齐 Padding + LengthGroupedSampler
    dm = UnifiedSFTDataModule(
        prepared_path=prepared_path,
        tokenizer=tokenizer,
        batch_size=4,
        val_batch_size=4,
        max_length=128,
        num_workers=0,
        group_by_length=True,
        pad_to_multiple_of=8,
        seed=42,
    )
    dm.setup("fit")
    assert dm.train_dataset is not None and dm.val_dataset is not None
    assert len(dm.train_dataset) + len(dm.val_dataset) == 50
    assert len(dm.train_dataset) == 45 and len(dm.val_dataset) == 5
    train_loader = dm.train_dataloader()
    assert isinstance(train_loader.sampler, LengthGroupedSampler)
    first_batch = next(iter(train_loader))
    assert first_batch["input_ids"].shape[0] == 4
    assert first_batch["input_ids"].shape[1] % 8 == 0
    assert torch.any(first_batch["labels"][:, 1:] != -100)

    # 第二次构建直接加载同一 DatasetDict
    dm_loaded = UnifiedSFTDataModule(
        prepared_path=prepared_path,
        tokenizer=tokenizer,
        batch_size=4,
        val_batch_size=4,
        max_length=128,
        num_workers=0,
        group_by_length=True,
        pad_to_multiple_of=8,
        seed=42,
    )
    dm_loaded.setup("fit")
    assert len(dm_loaded.train_dataset) == len(dm.train_dataset)

    # 5. 严格读取：未运行 lmml data 时训练必须立即给出明确提示
    missing_dm = UnifiedSFTDataModule(
        prepared_path=tmp_path / "missing",
        tokenizer=tokenizer,
    )
    with pytest.raises(FileNotFoundError, match="lmml data"):
        missing_dm.setup("fit")

    # 6. overwrite=False 复用已有产物，overwrite=True 可安全覆盖
    reused = prepare_llm_dataset(
        data_configs=sources,
        tokenizer=tokenizer,
        output_dir=prepared_path,
        task="sft",
        overwrite=False,
    )
    assert reused["overwritten"] is False
    overwritten = prepare_llm_dataset(
        data_configs=sources,
        tokenizer=tokenizer,
        output_dir=prepared_path,
        task="sft",
        overwrite=True,
        max_length=128,
        val_split_ratio=0.1,
        max_val_samples=8,
        preprocessing_num_proc=1,
        seed=42,
    )
    assert overwritten["overwritten"] is True


@pytest.mark.parametrize("task", ["dpo", "ppo"])
def test_prepare_dpo_and_ppo_datasets(tmp_path, task):
    model_dir = create_tiny_qwen_model_and_tokenizer(tmp_path / "tiny_qwen")
    tokenizer = build_sft_tokenizer(str(model_dir))
    if task == "dpo":
        rows = [
            {
                "prompt": f"Explain concept {i}",
                "chosen": f"Concept {i} is useful.",
                "rejected": f"Concept {i} is unrelated.",
            }
            for i in range(6)
        ]
    else:
        rows = [{"prompt": f"Solve the equation for x number {i}"} for i in range(6)]
    source = tmp_path / f"{task}.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    output = tmp_path / f"prepared_{task}"
    summary = prepare_llm_dataset(
        data_configs=[{"path": str(source), "max_samples": None, "sample_ratio": 1.0}],
        tokenizer=tokenizer,
        output_dir=output,
        task=task,
        max_length=64,
        val_split_ratio=0.34,
        max_val_samples=2,
        preprocessing_num_proc=1,
        seed=7,
    )
    assert summary["train_samples"] == 4 and summary["val_samples"] == 2
    loaded = load_from_disk(str(output))
    expected = "chosen_input_ids" if task == "dpo" else "input_ids"
    assert expected in loaded["train"].column_names
    assert len(loaded["train"][0][expected]) > 0


def test_prepare_rejects_missing_task_fields(tmp_path):
    model_dir = create_tiny_qwen_model_and_tokenizer(tmp_path / "tiny_qwen")
    tokenizer = build_sft_tokenizer(str(model_dir))
    source = tmp_path / "ppo_data.jsonl"
    source.write_text(
        "\n".join(json.dumps({"prompt": f"Question {i}"}) for i in range(4)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="未成功加载任何可用数据集"):
        prepare_llm_dataset(
            data_configs=[{"path": str(source)}],
            tokenizer=tokenizer,
            output_dir=tmp_path / "prepared_invalid",
            task="dpo",
            preprocessing_num_proc=1,
        )


def test_predict_does_not_require_prepared_dataset(tmp_path):
    model_root = tmp_path / "models"
    create_tiny_qwen_model_and_tokenizer(model_root / "Qwen3.5-0.8B-Base")
    missing = tmp_path / "not_prepared"
    with initialize_config_module(version_base="1.3", config_module="lmml.configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                "command=predict",
                "experiment=qwen35_lora_sft",
                f"paths.base_model_dir={model_root}",
                f"data.prepared_path={missing}",
            ],
        )

    recipe = CausalLMSFTRecipe(cfg)
    datamodule = recipe.build_datamodule()
    datamodule.setup("predict")
    adapted = recipe.adapt_trainer_kwargs({"val_check_interval": 500}, datamodule)
    assert adapted["val_check_interval"] == 500
    assert datamodule.train_dataset is None


def test_data_cli_command_and_overwrite_control(tmp_path, capsys):
    data_dir = tmp_path / "datasets"
    model_root = tmp_path / "models"
    create_tiny_qwen_model_and_tokenizer(model_root / "Qwen3.5-0.8B-Base")
    create_synthetic_sft_datasets(data_dir, n_ultrachat=6, n_python_codes=4, n_openr1_math=4)
    prepared_path = data_dir / "prepared" / "cli_sft"

    overrides = [
        "data",
        "experiment=qwen35_lora_sft",
        f"paths.base_dir={tmp_path}",
        f"paths.base_data_dir={data_dir}",
        f"paths.base_model_dir={model_root}",
        f"data.prepared_path={prepared_path}",
        "data.max_length=64",
        "data.val_split_ratio=0.1",
        "data.max_val_samples=2",
        "data.preprocessing_num_proc=1",
    ]
    cli_main(overrides)
    assert (prepared_path / "dataset_dict.json").exists()

    first_mtime = (prepared_path / "dataset_dict.json").stat().st_mtime_ns
    cli_main([*overrides, "data.overwrite=false"])
    assert (prepared_path / "dataset_dict.json").stat().st_mtime_ns == first_mtime
    assert "复用已有预处理数据集" in capsys.readouterr().out


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


def test_chunked_for_causal_lm_loss_matches_reference():
    from transformers.loss.loss_utils import ForCausalLMLoss

    from lmml.models.llm import chunked_for_causal_lm_loss

    torch.manual_seed(7)
    logits_ref = torch.randn(2, 32, 128, dtype=torch.bfloat16, requires_grad=True)
    logits_chk = logits_ref.detach().clone().requires_grad_(True)
    labels = torch.randint(0, 128, (2, 32))
    labels[:, :10] = -100

    loss_ref = ForCausalLMLoss(logits_ref, labels, vocab_size=128)
    loss_ref.backward()

    loss_chk = chunked_for_causal_lm_loss(logits_chk, labels, vocab_size=128, chunk_size=8)
    loss_chk.backward()

    assert torch.allclose(loss_ref, loss_chk, atol=1e-5)
    assert torch.allclose(logits_ref.grad, logits_chk.grad, atol=1e-4)


def test_lora_and_full_gradient_checkpointing_layers_in_train_mode(tmp_path):
    from transformers.modeling_layers import GradientCheckpointingLayer

    from lmml.models.llm import build_causal_lm

    model_dir = create_tiny_qwen_model_and_tokenizer(tmp_path / "tiny_qwen_gc")
    for mode in ("lora", "full"):
        model = build_causal_lm(
            str(model_dir),
            mode=mode,
            torch_dtype="float32",
            use_gradient_checkpointing=True,
        )
        # 所有子模块必须处于 train() 状态，否则 GradientCheckpointingLayer 会因 self.training=False 跳过检查点
        eval_modules = [name for name, mod in model.named_modules() if not mod.training]
        assert eval_modules == [], f"mode={mode} 下存在处于 eval 模式的子模块: {eval_modules}"
        decoder_layers = [
            mod for _, mod in model.named_modules() if isinstance(mod, GradientCheckpointingLayer)
        ]
        assert len(decoder_layers) == 2 and all(
            layer.training and layer.gradient_checkpointing for layer in decoder_layers
        )

        # 验证前向传播期间每个 GradientCheckpointingLayer 均实际触发了 _gradient_checkpointing_func
        gc_call_count = 0
        for layer in decoder_layers:
            orig_gc_fn = layer._gradient_checkpointing_func

            def _wrapped_gc(fn, *args, _orig=orig_gc_fn, **kwargs):
                nonlocal gc_call_count
                gc_call_count += 1
                return _orig(fn, *args, **kwargs)

            layer._gradient_checkpointing_func = _wrapped_gc

        input_ids = torch.randint(0, 100, (2, 16))
        attn_mask = torch.ones_like(input_ids)
        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids.clone())
        out.loss.backward()
        assert gc_call_count == len(decoder_layers), (
            f"mode={mode} 下期望触发 {len(decoder_layers)} 次梯度检查点，实际触发 {gc_call_count} 次"
        )




def test_end_to_end_qwen35_lora_and_full_sft_recipes(tmp_path, monkeypatch):
    _use_null_logger(monkeypatch)
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

    # 1. 独立数据命令生成预处理 DatasetDict
    with initialize_config_module(version_base="1.3", config_module="lmml.configs"):
        cfg_data = compose(
            config_name="config",
            overrides=[
                "command=data",
                "experiment=qwen35_lora_sft",
                f"paths.base_dir={base_dir}",
                f"paths.base_data_dir={data_dir}",
                f"paths.base_model_dir={model_root}",
                "data.max_length=96",
                "data.preprocessing_num_proc=1",
            ],
        )
    data_summary = run(cfg_data)
    assert data_summary["task"] == "sft" and data_summary["train_samples"] > 0
    assert (Path(data_summary["prepared_path"]) / "dataset_dict.json").exists()

    # 2. LoRA SFT 端到端训练 + 验证仅保存 LoRA 权重 + 导出 PEFT Adapter
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

    # 3. Full SFT 端到端训练 + 导出完整 HuggingFace 模型 + test + predict
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

    # 4. 基于导出的 Full SFT 检查点运行 test 与 predict
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
