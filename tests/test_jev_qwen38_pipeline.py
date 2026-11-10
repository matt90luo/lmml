from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_module
from peft import LoraConfig

from lmml.core.llm.jev import (
    Choice,
    Noul,
    QwenLoraLightningModule,
    Score,
    TypeSafeClient,
    apply_candidate_chat_template,
    candidate_prompts,
    compile_request,
    evaluate_probabilities,
    fit_temperature,
    format_response,
    jev_nll_brier_loss,
)
from lmml.core.runner import run
from lmml.datas.llm.jev import validate_jev_row
from lmml.datas.llm.sft import build_sft_tokenizer
from lmml.datas.llm.synthetic import (
    create_synthetic_jev_distill_v3,
    create_tiny_qwen38_model_and_tokenizer,
)


def test_jev_candidate_prompts_loss_and_metrics(tmp_path):
    model_dir = create_tiny_qwen38_model_and_tokenizer(tmp_path / "tiny_qwen38")
    tokenizer = build_sft_tokenizer(str(model_dir))

    compiled = compile_request(
        state={"user_id": "U-9402", "text": "customer charged twice, refund immediately"},
        questions={
            "asks_for_refund": Noul(instructions="Does the customer explicitly demand a refund?"),
            "routing_queue": Choice(
                instructions="Which department should handle this?",
                criteria={
                    "billing": "Charges and refunds",
                    "account_security": "Password resets",
                },
            ),
            "customer_frustration": Score(
                instructions="Rate frustration.",
                criteria=["Level 0: Calm", "Level 1: Annoyed", "Level 2: Angry"],
            ),
        },
    )
    assert len(compiled) == 3
    noul_prompts = candidate_prompts(compiled[0])
    assert len(noul_prompts) == 1
    assert "Is the answer to this question yes? Answer Yes or No." in noul_prompts[0]

    choice_prompts = candidate_prompts(compiled[1])
    assert len(choice_prompts) == 2
    assert "Proposed answer: billing: Charges and refunds" in choice_prompts[0]

    wrapped = apply_candidate_chat_template(tokenizer, noul_prompts[0])
    assert "<|im_start|>user\n" in wrapped
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\n" in wrapped

    logits = torch.tensor([0.2, 1.4], dtype=torch.float32, requires_grad=True)
    target = [0.15, 0.85]
    total_loss, ce_loss, brier_loss = jev_nll_brier_loss(logits, target, brier_weight=0.1)
    probs = torch.softmax(logits, dim=-1)
    expected_ce = -(0.15 * torch.log(probs[0]) + 0.85 * torch.log(probs[1]))
    expected_brier = (probs[0] - 0.15) ** 2 + (probs[1] - 0.85) ** 2
    assert torch.isclose(ce_loss, expected_ce, atol=1e-6)
    assert torch.isclose(brier_loss, expected_brier, atol=1e-6)
    assert torch.isclose(total_loss, expected_ce + 0.1 * expected_brier, atol=1e-6)
    total_loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    temp = fit_temperature([[0.0, 2.0], [1.5, 0.0]], [[0.1, 0.9], [0.85, 0.15]])
    assert 0.05 <= temp <= 20.0
    eval_res = evaluate_probabilities([[0.0, 1.0], [1.0, 0.0]], [[0.1, 0.9], [0.8, 0.2]])
    assert eval_res["accuracy"] == 1.0
    assert eval_res["coverage"][0]["escalation_rate"] == pytest.approx(0.0)

    formatted = format_response(
        compiled,
        [
            [0.03, 0.97],
            [0.95, 0.05],
            [0.0, 0.1, 0.9],
        ],
    )
    assert formatted["answers"]["asks_for_refund"]["noul"] == pytest.approx(0.97)
    assert formatted["answers"]["routing_queue"]["choice"] == "billing"
    assert formatted["answers"]["customer_frustration"]["score"] == pytest.approx(1.9)


def test_qwen_lora_two_stage_loading_and_forward(tmp_path):
    base_dir = create_tiny_qwen38_model_and_tokenizer(tmp_path / "tiny_qwen38")
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
    )

    # Case B: Fresh initialization (lora_checkpoint_dir=None)
    module_fresh = QwenLoraLightningModule(
        base_model_path=str(base_dir),
        lora_config=lora_cfg,
        lora_checkpoint_dir=None,
        torch_dtype="float32",
        gradient_checkpointing=False,
    )
    assert module_fresh.model.init_mode == "fresh"
    assert module_fresh.model.head.weight.dtype == torch.float32
    trainable = [n for n, p in module_fresh.named_parameters() if p.requires_grad]
    assert any("lora_A" in n for n in trainable)
    assert any(n.startswith("model.head.") for n in trainable)
    assert not any("embed_tokens" in n for n in trainable)

    row = validate_jev_row(
        {
            "id": "r1",
            "kind": "choice",
            "options": ["billing", "technical_support"],
            "target": [0.8, 0.2],
            "state": "Customer charged twice.",
            "question": "Which queue?",
        }
    )
    with torch.no_grad():
        module_fresh.eval()
        logits_before = module_fresh([row])[0].detach().clone()

    saved_dir = tmp_path / "saved_lora_ckpt"
    module_fresh.model.save(saved_dir, temperature=1.25)
    assert (saved_dir / "adapter" / "adapter_config.json").is_file()
    assert (saved_dir / "head.pt").is_file()
    assert (saved_dir / "temperature.json").is_file()

    # Case A: Reload from existing lora_checkpoint_dir with lora_config=None
    module_reloaded = QwenLoraLightningModule(
        base_model_path=str(base_dir),
        lora_config=None,
        lora_checkpoint_dir=str(saved_dir),
        torch_dtype="float32",
        gradient_checkpointing=False,
    )
    assert module_reloaded.model.init_mode == "checkpoint"
    assert module_reloaded.temperature == pytest.approx(1.25)
    reloaded_trainable = [n for n, p in module_reloaded.named_parameters() if p.requires_grad]
    assert set(reloaded_trainable) == set(trainable)

    with torch.no_grad():
        module_reloaded.eval()
        logits_after = module_reloaded([row])[0].detach().clone()
    assert torch.allclose(logits_before, logits_after, atol=1e-5)


def test_jev_end_to_end_data_train_top2_calibrate_test_predict(tmp_path):
    base_model_dir = tmp_path / "models"
    tiny_qwen38 = create_tiny_qwen38_model_and_tokenizer(base_model_dir / "Qwen3.8-27B")
    jev_corpus_dir = create_synthetic_jev_distill_v3(tmp_path / "jev-distill-corpus-v3", samples_per_split=6)
    prepared_dir = tmp_path / "prepared_jev"
    top2_ckpt_dir = tmp_path / "checkpoints_qwen38_27b_jev"
    final_export_dir = tmp_path / "qwen38_27b_jev_final"
    pred_out = tmp_path / "jev_predictions.jsonl"

    common_overrides = [
        "experiment=qwen38_27B_jev_smoke",
        f"paths.base_dir={tmp_path}",
        f"paths.base_model_dir={base_model_dir}",
        f"paths.jev_data_dir={jev_corpus_dir}",
        f"data.prepared_path={prepared_dir}",
        f"algo.model.name_or_path={tiny_qwen38}",
        "algo.model.torch_dtype=float32",
        "algo.model.use_gradient_checkpointing=false",
        "algo.lora_r=8",
        "algo.lora_alpha=16",
        f"algo.export.save_dir={final_export_dir}",
        f"callbacks.checkpoint.dirpath={top2_ckpt_dir}",
        "callbacks.checkpoint.save_top_k=2",
        "trainer.max_epochs=2",
        "+trainer.val_check_interval=2",
        "logger.enabled=false",
        f"logger.save_dir={tmp_path / 'logs'}",
    ]

    with initialize_config_module(version_base="1.3", config_module="lmml.configs"):
        # 1. lmml data
        cfg_data = compose(config_name="config", overrides=["command=data", *common_overrides])
        data_summary = run(cfg_data)
        assert data_summary["task"] == "jev"
        assert data_summary["train_samples"] == 6
        assert data_summary["test_set_30k_samples"] == 6
        assert data_summary["ood_samples"] == 6

        # 2. lmml train (Stage 1 train + Top-2 save + Stage 2 calibration + Stage 3 auto eval in smoke)
        cfg_train = compose(config_name="config", overrides=["command=train", *common_overrides])
        val_metric = run(cfg_train)
        assert val_metric is not None

        ckpt_files = sorted(top2_ckpt_dir.glob("*.ckpt"))
        assert len(ckpt_files) == 2
        top_peft_dirs = sorted(p for p in top2_ckpt_dir.iterdir() if p.is_dir() and p.name.startswith("top"))
        assert len(top_peft_dirs) == 2
        for d in top_peft_dirs:
            assert (d / "adapter" / "adapter_config.json").is_file()
            assert (d / "head.pt").is_file()
            assert (d / "temperature.json").is_file()

        assert (final_export_dir / "adapter" / "adapter_config.json").is_file()
        assert (final_export_dir / "head.pt").is_file()
        assert (final_export_dir / "temperature.json").is_file()
        assert (final_export_dir / "evaluation_summary.json").is_file()
        eval_summary = json.loads((final_export_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
        assert set(eval_summary["splits"].keys()) == {"test", "test_set_30k", "ood"}

        # 3. lmml test (explicit Stage 3 evaluation from exported checkpoint)
        cfg_test = compose(
            config_name="config",
            overrides=[
                "command=test",
                f"ckpt_path={final_export_dir}",
                *common_overrides,
            ],
        )
        test_res = run(cfg_test)
        assert test_res is not None and "test/loss" in test_res

        # 4. lmml predict
        cfg_pred = compose(
            config_name="config",
            overrides=[
                "command=predict",
                f"ckpt_path={final_export_dir}",
                f"predict.output={pred_out}",
                *common_overrides,
            ],
        )
        pred_res = run(cfg_pred)
        assert Path(pred_res["output"]).is_file()
        pred_rows = [json.loads(line) for line in Path(pred_res["output"]).read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(pred_rows) == 1
        answers = pred_rows[0]["answers"]
        assert 0.0 <= answers["asks_for_refund"]["noul"] <= 1.0
        assert answers["routing_queue"]["choice"] in {
            "billing",
            "account_security",
            "technical_support",
            "general_inquiry",
        }
        assert 0.0 <= answers["customer_frustration"]["score"] <= 2.0

    # 5. In-process TypeSafeClient verification
    with TypeSafeClient(checkpoint_path=final_export_dir, device="cpu", torch_dtype="float32") as client:
        response = client.system_one(
            state={"user_id": "U-9402", "text": "charged twice, refund now!", "history_disputes": 0},
            questions={
                "asks_for_refund": Noul(instructions="Does the customer explicitly demand a refund of money?"),
                "routing_queue": Choice(
                    instructions="Which department should handle this request?",
                    criteria={
                        "billing": "Issues with recurring payments, charges, cards, or receipts",
                        "account_security": "Compromised accounts, password resets, 2FA failures",
                    },
                ),
                "customer_frustration": Score(
                    instructions="Rate frustration.",
                    criteria=["Calm", "Annoyed", "Angry"],
                ),
            },
        )
    assert 0.0 <= response.answers["asks_for_refund"].noul <= 1.0
    assert response.answers["routing_queue"].choice in {"billing", "account_security"}
    assert 0.0 <= response.answers["customer_frustration"].score <= 2.0


def test_jev_pretokenized_arrow_sampling_and_resume(tmp_path, monkeypatch):
    from datasets import load_from_disk

    from lmml.core.llm.jev.data import JevCollator
    from lmml.datas.llm import jev as jev_data

    model_dir = create_tiny_qwen38_model_and_tokenizer(tmp_path / "tiny_qwen38")
    tokenizer = build_sft_tokenizer(str(model_dir))
    corpus = create_synthetic_jev_distill_v3(tmp_path / "jev-corpus", samples_per_split=20)
    output = tmp_path / "prepared"

    kwargs = {
        "data_configs": [{"path": str(corpus)}],
        "tokenizer": tokenizer,
        "output_dir": output,
        "max_length": 512,
        "preprocessing_num_proc": 2,
        "tokenization_batch_size": 4,
        "shard_size": 5,
        "progress_every": 5,
        "max_train_samples": 7,
        "max_val_samples": 4,
        "max_calibration_samples": 4,
        "max_test_samples": 4,
        "max_test_30k_samples": 4,
        "max_ood_samples": 4,
        "tokenizer_name_or_path": str(model_dir),
        "overwrite": True,
    }

    # Simulate an interruption after shards complete but before final assembly.
    original_assemble = jev_data._assemble_split

    def fail_assemble(*args, **kwargs):
        raise RuntimeError("simulated interruption before final assembly")

    monkeypatch.setattr(jev_data, "_assemble_split", fail_assemble)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        jev_data.prepare_jev_dataset(**kwargs)
    staging = tmp_path / ".prepared.jev-staging"
    assert staging.exists()

    monkeypatch.setattr(jev_data, "_assemble_split", original_assemble)
    summary = jev_data.prepare_jev_dataset(**kwargs)
    assert summary["schema_version"] == jev_data.JEV_SCHEMA_VERSION
    assert summary["train_samples"] == 7
    assert summary["val_samples"] == 4

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == jev_data.JEV_SCHEMA_VERSION
    assert manifest["splits"]["train"]["reused_shards"] > 0
    assert manifest["splits"]["train"]["candidate_count"] > 0
    assert manifest["splits"]["train"]["token_count"] > 0
    assert not staging.exists()

    dataset = load_from_disk(str(output))
    row = dataset["train"][0]
    assert "record_json" not in row
    assert row["input_ids_flat"]
    assert row["candidate_offsets"][0] == 0
    assert row["candidate_offsets"][-1] == len(row["input_ids_flat"])
    assert len(row["candidate_lengths"]) == len(row["candidate_offsets"]) - 1
    assert max(row["candidate_lengths"]) <= 512

    selected_ids = {dataset["train"][index]["id"] for index in range(len(dataset["train"]))}
    first_ids = {f"train_{index}_{('noul', 'choice', 'score')[index % 3][0]}" for index in range(7)}
    assert selected_ids != first_ids

    collator = JevCollator(tokenizer, max_length=512, strict_length=True)
    pretokenized_batch = collator([row])
    raw_row = {
        key: value
        for key, value in row.items()
        if key not in {"input_ids_flat", "candidate_offsets", "candidate_lengths"}
    }
    raw_row["state"] = json.loads(row["state"]) if row["state"].startswith("{") else row["state"]
    raw_batch = collator([raw_row])
    assert torch.equal(pretokenized_batch["input_ids"], raw_batch["input_ids"])
    assert torch.equal(pretokenized_batch["attention_mask"], raw_batch["attention_mask"])

    # A matching completed output is idempotent.
    reused = jev_data.prepare_jev_dataset(**kwargs)
    assert reused["reused"] is True
    assert reused["train_samples"] == 7


def test_jev_data_config_defaults():
    with initialize_config_module(version_base="1.3", config_module="lmml.configs"):
        full = compose(config_name="config", overrides=["command=data", "experiment=qwen38_27B_jev"])
        assert full.data.task == "jev"
        assert full.data.preprocessing_num_proc == "auto"
        assert full.data.tokenization_batch_size == 512
        assert full.data.shard_size == 20000
        assert full.data.progress_every == 10000
        assert full.data.resume == "auto"
        assert full.data.store_tokenized is True
        assert full.data.max_train_samples is None
        assert full.data.max_val_samples is None
        assert full.data.max_calibration_samples is None

        smoke = compose(config_name="config", overrides=["command=data", "experiment=qwen38_27B_jev_smoke"])
        assert smoke.data.preprocessing_num_proc == 1
        assert smoke.data.max_train_samples == 16
        assert smoke.data.max_val_samples == 8
        assert smoke.data.max_calibration_samples == 8
