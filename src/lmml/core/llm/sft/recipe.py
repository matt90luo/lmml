"""CausalLMSFTRecipe：组装 UnifiedSFTDataModule、CausalLMSFTLightningModule、回调与最优模型导出。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from lmml.core.llm.sft.module import CausalLMSFTLightningModule
from lmml.core.recipe import Recipe
from lmml.datas.llm.sft import UnifiedSFTDataModule, build_sft_tokenizer
from lmml.models.llm.builders import DEFAULT_QWEN35_LORA_TARGET_MODULES

console = Console()


class CausalLMSFTRecipe(Recipe):
    """面向 Qwen / CausalLM 监督微调 (LoRA & Full SFT) 的统一 Recipe。"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        matmul_prec = str(cfg.algo.get("matmul_precision", "high"))
        torch.set_float32_matmul_precision(matmul_prec)

    def build_datamodule(self) -> UnifiedSFTDataModule:
        cfg = self.cfg
        algo = cfg.algo
        data = cfg.data
        model_cfg = algo.model
        pred_cfg = cfg.get("predict") or {}

        chat_start = str(data.get("chat_start_token", "<|im_start|>"))
        chat_end = str(data.get("chat_end_token", "<|im_end|>"))

        tokenizer = build_sft_tokenizer(
            model_name_or_path=str(model_cfg.name_or_path),
            chat_end_token=chat_end,
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        )

        raw_sources = OmegaConf.to_container(data.sources, resolve=True)
        if not isinstance(raw_sources, list):
            raise TypeError("data.sources 必须为数据源字典列表。")

        raw_prompts = pred_cfg.get("prompts")
        prompts = list(OmegaConf.to_container(raw_prompts, resolve=True)) if raw_prompts else None

        return UnifiedSFTDataModule(
            data_configs=raw_sources,
            tokenizer=tokenizer,
            batch_size=int(data.get("batch_size", 8)),
            val_batch_size=int(data.get("val_batch_size", data.get("batch_size", 8))),
            max_length=int(data.get("max_length", 2048)),
            num_workers=int(data.get("num_workers", 4)),
            preprocessing_num_proc=data.get("preprocessing_num_proc", "auto"),
            val_split_ratio=float(data.get("val_split_ratio", 0.02)),
            max_val_samples=data.get("max_val_samples", 1500),
            chat_start_token=chat_start,
            chat_end_token=chat_end,
            ignore_index=int(data.get("ignore_index", -100)),
            pretokenize=bool(data.get("pretokenize", True)),
            group_by_length=bool(data.get("group_by_length", True)),
            mega_batch_mult=int(data.get("mega_batch_mult", 50)),
            pad_to_multiple_of=data.get("pad_to_multiple_of", 8),
            drop_no_target=bool(data.get("drop_no_target", True)),
            cache_dir=data.get("cache_dir"),
            predict_prompts=prompts,
            predict_batch_size=int(pred_cfg.get("batch_size", 4)),
            seed=int(cfg.seed),
        )

    def build_module(
        self,
        datamodule: UnifiedSFTDataModule,
        ckpt_path: str | None = None,
    ) -> CausalLMSFTLightningModule:
        cfg = self.cfg
        algo = cfg.algo
        model_cfg = algo.model
        lora_cfg = algo.get("lora") or {}
        mode = str(algo.get("mode", "lora")).strip().lower()

        raw_targets = lora_cfg.get("target_modules")
        target_modules = (
            list(OmegaConf.to_container(raw_targets, resolve=True))
            if raw_targets is not None
            else list(DEFAULT_QWEN35_LORA_TARGET_MODULES)
        )

        module = CausalLMSFTLightningModule(
            model_name_or_path=str(model_cfg.name_or_path),
            mode=mode,
            learning_rate=float(algo.get("lr", 1e-4 if mode == "lora" else 2e-5)),
            weight_decay=float(algo.get("weight_decay", 0.01)),
            adam_beta1=float(algo.get("adam_beta1", 0.9)),
            adam_beta2=float(algo.get("adam_beta2", 0.999 if mode == "lora" else 0.95)),
            warmup_ratio=float(algo.get("warmup_ratio", 0.05)),
            use_8bit_optimizer=bool(algo.get("use_8bit_optimizer", False)),
            torch_dtype=str(model_cfg.get("torch_dtype", "bfloat16")),
            attn_implementation=str(model_cfg.get("attn_implementation", "sdpa")),
            use_gradient_checkpointing=bool(model_cfg.get("use_gradient_checkpointing", True)),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
            eos_token_id=datamodule.tokenizer.eos_token_id,
            lora_r=int(lora_cfg.get("r", 8)),
            lora_alpha=int(lora_cfg.get("alpha", 16)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            lora_target_modules=target_modules,
            max_new_tokens=int(algo.get("max_new_tokens", 128)),
        )

        if ckpt_path and os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            module.load_state_dict(state_dict, strict=(mode != "lora"))
        return module

    def adapt_trainer_kwargs(
        self,
        trainer_kwargs: dict[str, Any],
        datamodule: UnifiedSFTDataModule,
    ) -> dict[str, Any]:
        """当 val_check_interval 为整数且大于单轮总 batch 数时自动钳制，防止小样本调试报错。"""
        adapted = dict(trainer_kwargs)
        val_interval = adapted.get("val_check_interval")
        if isinstance(val_interval, int) and val_interval > 1:
            datamodule.setup("fit")
            if datamodule.train_dataset is not None:
                num_batches = max(1, len(datamodule.train_dataloader()))
                limit_train = adapted.get("limit_train_batches")
                if isinstance(limit_train, int) and limit_train > 0:
                    num_batches = min(num_batches, limit_train)
                if val_interval > num_batches:
                    adapted["val_check_interval"] = num_batches
        return adapted

    def build_callbacks(self, datamodule: UnifiedSFTDataModule) -> list[pl.Callback]:
        cb_cfg = self.cfg.get("callbacks") or {}
        ckpt_cfg = cb_cfg.get("checkpoint") or {}
        mode = str(self.cfg.algo.get("mode", "lora")).strip().lower()

        callbacks: list[pl.Callback] = []
        if bool(ckpt_cfg.get("enabled", True)):
            monitor = str(ckpt_cfg.get("monitor") or "val/loss")
            ckpt_mode = str(ckpt_cfg.get("mode") or "min")
            save_top_k = int(ckpt_cfg.get("save_top_k", 1))
            save_last = bool(ckpt_cfg.get("save_last", False))
            save_weights_only = bool(ckpt_cfg.get("save_weights_only", mode == "full"))
            filename = str(ckpt_cfg.get("filename") or f"qwen-{mode}-sft-{{epoch:02d}}-{{step:06d}}")

            callbacks.append(
                ModelCheckpoint(
                    dirpath=ckpt_cfg.get("dirpath"),
                    filename=filename,
                    monitor=monitor,
                    mode=ckpt_mode,
                    save_top_k=save_top_k,
                    save_last=save_last,
                    save_weights_only=save_weights_only,
                    save_on_train_epoch_end=False,
                    auto_insert_metric_name=False,
                )
            )

        if bool((cb_cfg.get("lr_monitor") or {}).get("enabled", True)):
            callbacks.append(LearningRateMonitor(logging_interval="step"))

        return callbacks

    def on_fit_end(
        self,
        trainer: pl.Trainer,
        module: CausalLMSFTLightningModule,
        datamodule: UnifiedSFTDataModule,
    ) -> None:
        export_cfg = self.cfg.algo.get("export") or {}
        if not bool(export_cfg.get("enabled", True)):
            return

        save_dir = export_cfg.get("save_dir")
        if not save_dir:
            return
        save_path = Path(str(save_dir))

        ckpt_cb = next((cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)), None)
        best_ckpt_path = ckpt_cb.best_model_path if ckpt_cb is not None else ""

        module.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if best_ckpt_path and os.path.exists(best_ckpt_path):
            console.print(
                f"[bold cyan][Checkpoint][/bold cyan] 正在从最优检查点恢复 {module.mode.upper()} 权重: "
                f"{best_ckpt_path}"
            )
            ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            module.load_state_dict(state_dict, strict=(module.mode != "lora"))
            del ckpt

        save_path.mkdir(parents=True, exist_ok=True)
        lora_cfg = self.cfg.algo.get("lora") or {}
        merge_on_save = bool(lora_cfg.get("merge_on_save", False))

        if module.mode == "lora" and merge_on_save and hasattr(module.model, "merge_and_unload"):
            merged_model = module.model.merge_and_unload()
            merged_model.save_pretrained(
                str(save_path),
                safe_serialization=True,
                max_shard_size=str(export_cfg.get("max_shard_size", "4GB")),
            )
        elif module.mode == "lora":
            module.model.save_pretrained(str(save_path))
        else:
            module.model.save_pretrained(
                str(save_path),
                safe_serialization=True,
                max_shard_size=str(export_cfg.get("max_shard_size", "4GB")),
            )

        datamodule.tokenizer.save_pretrained(str(save_path))
        console.print(
            f"[bold green][Export][/bold green] {module.mode.upper()} SFT 训练完成，"
            f"最优 HuggingFace 模型/Adapter 已导出至: [bold]{save_path.resolve()}[/bold]"
        )

    def save_predictions(self, predictions: list[dict[str, Any]], output_path: str) -> str:
        out = Path(output_path)
        if out.suffix == ".npz":
            out = out.with_suffix(".jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, str]] = []
        for batch in predictions:
            prompts = batch.get("prompts", [])
            token_ids_list = batch.get("generated_token_ids", [])
            for prompt, tok_ids in zip(prompts, token_ids_list):
                records.append({"prompt": str(prompt), "generated_token_ids": tok_ids})

        with out.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return str(out.resolve())
