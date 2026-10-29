from __future__ import annotations

import torch
from hydra import compose, initialize_config_module
from transformers import AutoModelForCausalLM

from lmml.core.llm.dpo.module import CausalLMDPOLightningModule
from lmml.core.llm.rl_common.rollout import TorchRolloutEngine
from lmml.core.llm.rl_common.tools import execute_tool, parse_tool_calls, safe_math_eval
from lmml.datas.llm import (
    DPODataModule,
    PretrainDataModule,
    build_sft_tokenizer,
    create_synthetic_minimind_datasets,
    encode_chatml_messages,
    normalize_item_format,
    prepare_llm_dataset,
)
from lmml.models.llm.minimind import MiniMindConfig, MiniMindForCausalLM


def test_minimind_dense_and_moe_forward_generate_and_roundtrip(tmp_path):
    tokenizer = build_sft_tokenizer("missing/minimind-3")
    assert len(tokenizer) == 6400
    for use_moe in (False, True):
        config = MiniMindConfig(
            vocab_size=6400,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            use_moe=use_moe,
            num_experts=2,
            moe_intermediate_size=64,
        )
        model = MiniMindForCausalLM(config)
        inputs = torch.randint(0, 6400, (2, 12))
        labels = inputs.clone()
        labels[:, :4] = -100
        outputs = model(inputs, labels=labels)
        assert outputs.logits.shape == (2, 12, 6400)
        assert torch.isfinite(outputs.loss)
        if use_moe:
            assert float(outputs.aux_loss.detach()) > 0
        outputs.loss.backward()
        assert model.lm_head.weight.grad is not None

        generated = model.generate(
            inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=3,
            do_sample=False,
            top_k=0,
            eos_token_id=6400,
            pad_token_id=0,
        )
        assert generated.shape == (2, 15)
        path = tmp_path / f"minimind-{'moe' if use_moe else 'dense'}"
        model.save_pretrained(path)
        loaded = AutoModelForCausalLM.from_pretrained(path)
        assert isinstance(loaded, MiniMindForCausalLM)
        assert loaded.config.use_moe is use_moe


def test_minimind_reference_parameter_counts():
    counts = {}
    for use_moe in (False, True):
        with torch.device("meta"):
            model = MiniMindForCausalLM(
                MiniMindConfig(use_moe=use_moe, max_position_embeddings=2048)
            )
        counts[use_moe] = sum(parameter.numel() for parameter in model.parameters())
    assert counts[False] == 63_912_192
    assert counts[True] == 198_416_640


def test_chatml_cot_tools_are_supervised_and_masked(tmp_path):
    tokenizer = build_sft_tokenizer("missing/minimind-3")
    raw = {
        "conversations": [
            {"role": "system", "content": "Use tools.", "tools": "[{\"type\":\"function\",\"function\":{\"name\":\"calculate_math\"}}]"},
            {"role": "user", "content": "Calculate 2+2"},
            {"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"calculate_math\",\"arguments\":{\"expression\":\"2+2\"}}]"},
            {"role": "tool", "content": "{\"result\":\"4\"}"},
            {"role": "assistant", "reasoning_content": "The tool returned four.", "content": "The answer is 4."},
        ]
    }
    messages = normalize_item_format(raw, wrap_reasoning=True)
    assert "<think>" in messages[-1]["content"]
    encoded = encode_chatml_messages(messages, tokenizer, max_length=512)
    assert encoded is not None
    text = tokenizer.decode(encoded["input_ids"])
    assert "# Tools" in text and "<tool_call>" in text and "<tool_response>" in text
    supervised = tokenizer.decode([token for token, label in zip(encoded["input_ids"], encoded["labels"]) if label != -100])
    assert "<tool_call>" in supervised
    assert "The answer is 4." in supervised
    assert "Calculate 2+2" not in supervised


def test_all_minimind_preprocessing_tasks(tmp_path):
    tokenizer = build_sft_tokenizer("missing/minimind-3")
    paths = create_synthetic_minimind_datasets(tmp_path / "raw", samples_per_task=8)
    expected = {
        "pretrain": "input_ids",
        "sft": "input_ids",
        "dpo": "chosen_input_ids",
        "ppo": "prompt",
        "grpo": "prompt",
        "agent_rl": "messages_json",
    }
    source_for_task = {"ppo": "rlaif", "grpo": "rlaif"}
    for task, column in expected.items():
        output = tmp_path / f"prepared_{task}"
        source = paths[source_for_task.get(task, task)]
        summary = prepare_llm_dataset(
            [{"path": str(source), "max_samples": None, "sample_ratio": 1.0}],
            tokenizer,
            output,
            task=task,
            max_length=128,
            val_split_ratio=0.25,
            preprocessing_num_proc=1,
            wrap_reasoning=task == "sft",
            add_system_ratio=0.2 if task == "sft" else 0.0,
            empty_think_remove_ratio=0.8 if task == "sft" else 0.0,
        )
        assert summary["train_samples"] + summary["val_samples"] == 8
        from datasets import load_from_disk

        dataset = load_from_disk(str(output))
        assert column in dataset["train"].column_names

    dm = PretrainDataModule(tmp_path / "prepared_pretrain", 0, batch_size=2, num_workers=0, max_length=128)
    dm.setup()
    assert dm.train_dataloader() is not None
    dpo = DPODataModule(tmp_path / "prepared_dpo", 0, batch_size=2, max_length=128)
    dpo.setup()
    batch = next(iter(dpo.train_dataloader()))
    assert batch["chosen_input_ids"].shape[0] == batch["rejected_input_ids"].shape[0] == 2
    assert batch["chosen_labels"].shape == batch["chosen_input_ids"].shape
    assert batch["rejected_labels"].shape == batch["rejected_input_ids"].shape


def test_dpo_and_rollout_smoke(tmp_path):
    tokenizer = build_sft_tokenizer("missing/minimind-3")
    paths = create_synthetic_minimind_datasets(tmp_path / "raw", samples_per_task=6)
    prepared = tmp_path / "dpo"
    prepare_llm_dataset(
        [{"path": str(paths["dpo"]), "max_samples": None, "sample_ratio": 1.0}],
        tokenizer,
        prepared,
        task="dpo",
        max_length=128,
        val_split_ratio=0.34,
        preprocessing_num_proc=1,
    )
    dm = DPODataModule(prepared, int(tokenizer.pad_token_id), batch_size=2, max_length=128)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    config = {
        "architecture": "minimind",
        "init_mode": "scratch",
        "config": {
            "vocab_size": 6400,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 256,
        },
    }
    module = CausalLMDPOLightningModule(
        "unused", tokenizer=tokenizer, torch_dtype="float32", use_gradient_checkpointing=False, model_config=config
    )
    loss, margin = module._loss(batch)
    assert torch.isfinite(loss) and margin.shape == (2,)

    inputs = tokenizer(["hello", "world"], return_tensors="pt", padding=True)
    rollout = TorchRolloutEngine(module.policy, tokenizer).rollout(
        inputs.input_ids,
        inputs.attention_mask,
        num_generations=2,
        max_new_tokens=3,
    )
    assert rollout.output_ids.shape[0] == 4
    assert rollout.per_token_logps.shape == rollout.completion_mask.shape


def test_agent_tools_and_reward_primitives():
    assert safe_math_eval("2 + 2 * 3") == 8
    assert execute_tool("calculate_math", {"expression": "6*7"}) == {"result": "42"}
    assert execute_tool("get_current_time", {"timezone": "Asia/Shanghai"})["timezone"] == "Asia/Shanghai"
    calls = parse_tool_calls('<tool_call>{"name":"calculate_math","arguments":{"expression":"6*7"}}</tool_call>')
    assert calls[0]["name"] == "calculate_math"


def test_minimind_experiment_matrix_composes():
    names = [
        "minimind3_pretrain",
        "minimind3_moe_pretrain",
        "minimind3_full_sft",
        "minimind3_lora_sft",
        "minimind3_moe_full_sft",
        "minimind3_moe_lora_sft",
        "minimind3_dpo",
        "minimind3_ppo",
        "minimind3_grpo",
        "minimind3_agent_rl",
        "minimind3_moe_dpo",
        "minimind3_moe_ppo",
        "minimind3_moe_grpo",
        "minimind3_moe_agent_rl",
    ]
    with initialize_config_module(version_base="1.3", config_module="lmml.configs"):
        for name in names:
            cfg = compose(config_name="config", overrides=[f"experiment={name}"])
            assert cfg.algo.model.architecture == "minimind"
            assert cfg.recipe._target_.endswith("Recipe")
            if "moe" in name:
                assert cfg.algo.model.config.use_moe is True
