"""Base Recipe for DPO, PPO, GRPO and Agentic RL."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig

from lmml.core.llm.rl_common.export import export_model
from lmml.core.llm.rl_common.model_config import resolve_model_config
from lmml.core.recipe import Recipe
from lmml.datas.llm.sft import build_sft_tokenizer
from lmml.datas.llm.tasks import DPODataModule, RLPromptDataModule


class CausalLMRLRecipe(Recipe):
    module_class: type[pl.LightningModule]

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        torch.set_float32_matmul_precision(str(cfg.algo.get("matmul_precision", "high")))

    @property
    def task(self) -> str:
        return str(self.cfg.data.get("task", "ppo")).lower()

    def _tokenizer(self) -> Any:
        model_cfg = self.cfg.algo.model
        return build_sft_tokenizer(
            str(self.cfg.data.get("tokenizer_name_or_path") or model_cfg.name_or_path),
            str(self.cfg.data.get("chat_end_token", "<|im_end|>")),
            bool(model_cfg.get("trust_remote_code", True)),
        )

    def build_datamodule(self) -> pl.LightningDataModule:
        tokenizer = self._tokenizer()
        prepared = str(self.cfg.data.prepared_path)
        batch_size = int(self.cfg.data.get("batch_size", 2))
        val_batch_size = int(self.cfg.data.get("val_batch_size", batch_size))
        num_workers = int(self.cfg.data.get("num_workers", 0))
        if self.task == "dpo":
            datamodule = DPODataModule(
                prepared,
                int(tokenizer.pad_token_id),
                batch_size,
                val_batch_size,
                int(self.cfg.data.get("max_length", 1024)),
                num_workers,
            )
        else:
            datamodule = RLPromptDataModule(
                prepared,
                tokenizer,
                batch_size,
                val_batch_size,
                int(self.cfg.data.get("max_length", 768)),
                num_workers,
                float(self.cfg.data.get("thinking_ratio", 0.9)),
                self.task == "agent_rl",
            )
        datamodule.tokenizer = tokenizer
        return datamodule

    def _common_module_kwargs(self, datamodule: pl.LightningDataModule) -> dict[str, Any]:
        cfg = self.cfg
        model_cfg = cfg.algo.model
        algorithm = cfg.algo
        return {
            "model_name_or_path": str(model_cfg.name_or_path),
            "tokenizer": self._tokenizer(),
            "learning_rate": float(algorithm.get("lr", 3e-7)),
            "torch_dtype": str(model_cfg.get("torch_dtype", "bfloat16")),
            "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
            "model_config": resolve_model_config(model_cfg),
            "reward_mode": str(algorithm.get("reward_mode", "rule_only")),
            "reward_model_path": algorithm.get("reward_model_path"),
        }

    def build_module(
        self,
        datamodule: pl.LightningDataModule,
        ckpt_path: str | None = None,
    ) -> pl.LightningModule:
        cfg = self.cfg
        algo = cfg.algo
        kwargs = self._common_module_kwargs(datamodule)
        if self.task in {"ppo", "grpo", "agent_rl"}:
            kwargs["max_new_tokens"] = int(algo.get("max_new_tokens", 128))
            kwargs["max_prompt_length"] = int(cfg.data.get("max_length", 768))
        if self.task == "ppo":
            kwargs.update(
                critic_learning_rate=float(algo.get("critic_lr", 5e-7)),
                gamma=float(algo.get("gamma", 1.0)),
                lam=float(algo.get("lam", 0.95)),
                clip_epsilon=float(algo.get("clip_epsilon", algo.get("epsilon", 0.2))),
                value_clip=float(algo.get("value_clip", 0.2)),
                vf_coef=float(algo.get("vf_coef", 0.5)),
                kl_coef=float(algo.get("kl_coef", 0.02)),
                update_iters=int(algo.get("update_iters", 2)),
                mini_batch_size=int(algo.get("mini_batch_size", 1)),
            )
        elif self.task == "grpo":
            kwargs.update(
                num_generations=int(algo.get("num_generations", 6)),
                beta=float(algo.get("beta", 0.1)),
                epsilon=float(algo.get("epsilon", 0.2)),
                epsilon_high=float(algo.get("epsilon_high", 5.0)),
                loss_type=str(algo.get("loss_type", "cispo")),
            )
        elif self.task == "agent_rl":
            kwargs.update(
                max_turns=int(algo.get("max_turns", 3)),
                num_generations=int(algo.get("num_generations", 4)),
                beta=float(algo.get("beta", 0.1)),
                epsilon_high=float(algo.get("epsilon_high", 5.0)),
            )
        elif self.task == "dpo":
            kwargs = {
                "model_name_or_path": str(algo.model.name_or_path),
                "learning_rate": float(algo.get("lr", 4e-8)),
                "tokenizer": self._tokenizer(),
                "beta": float(algo.get("beta", 0.15)),
                "torch_dtype": str(algo.model.get("torch_dtype", "bfloat16")),
                "trust_remote_code": bool(algo.model.get("trust_remote_code", True)),
                "model_config": resolve_model_config(algo.model),
            }
        module = self.module_class(**kwargs)
        if ckpt_path and os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            module.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
        return module

    def build_callbacks(self, datamodule: pl.LightningDataModule) -> list[pl.Callback]:
        cb_cfg = self.cfg.get("callbacks") or {}
        ckpt_cfg = cb_cfg.get("checkpoint") or {}
        callbacks: list[pl.Callback] = []
        if bool(ckpt_cfg.get("enabled", True)):
            callbacks.append(
                ModelCheckpoint(
                    dirpath=ckpt_cfg.get("dirpath"),
                    filename=str(ckpt_cfg.get("filename", "best-{step:06d}")),
                    monitor=str(ckpt_cfg.get("monitor", "train/loss")),
                    mode=str(ckpt_cfg.get("mode", "min")),
                    save_top_k=int(ckpt_cfg.get("save_top_k", -1)),
                    save_last=bool(ckpt_cfg.get("save_last", True)),
                    auto_insert_metric_name=False,
                )
            )
        if bool((cb_cfg.get("lr_monitor") or {}).get("enabled", True)):
            callbacks.append(LearningRateMonitor(logging_interval="step"))
        return callbacks

    def on_fit_end(
        self,
        trainer: pl.Trainer,
        module: pl.LightningModule,
        datamodule: pl.LightningDataModule,
    ) -> None:
        export_cfg = self.cfg.algo.get("export") or {}
        if not bool(export_cfg.get("enabled", True)):
            return
        save_dir = export_cfg.get("save_dir")
        if not save_dir:
            return
        model = getattr(module, "policy", getattr(module, "actor", getattr(module, "model", module)))
        module.cpu()
        export_model(model, datamodule.tokenizer, Path(str(save_dir)))

    def save_predictions(self, predictions: list[dict[str, Any]], output_path: str) -> str:
        path = Path(output_path)
        if path.suffix == ".npz":
            path = path.with_suffix(".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for batch in predictions:
                for prompt, token_ids in zip(
                    batch.get("prompts", []), batch.get("generated_token_ids", [])
                ):
                    handle.write(
                        json.dumps(
                            {"prompt": str(prompt), "generated_token_ids": token_ids},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        return str(path.resolve())


__all__ = ["CausalLMRLRecipe"]
