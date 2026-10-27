"""CausalLM 监督微调 (SFT) LightningModule：统一支持 LoRA 微调与 Full 全量微调。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import lightning.pytorch as pl
import torch
from rich.console import Console
from transformers import get_cosine_schedule_with_warmup

from lmml.models.llm.builders import DEFAULT_QWEN35_LORA_TARGET_MODULES, build_causal_lm

console = Console()


class CausalLMSFTLightningModule(pl.LightningModule):
    """统一的 Qwen / CausalLM 监督微调模块（支持 mode='lora' 与 mode='full'）。"""

    def __init__(
        self,
        model_name_or_path: str,
        mode: str = "lora",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        warmup_ratio: float = 0.05,
        use_8bit_optimizer: bool = False,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        use_gradient_checkpointing: bool = True,
        trust_remote_code: bool = True,
        eos_token_id: int | None = None,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] = DEFAULT_QWEN35_LORA_TARGET_MODULES,
        max_new_tokens: int = 128,
    ):
        super().__init__()
        self.mode = str(mode).strip().lower()
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.warmup_ratio = float(warmup_ratio)
        self.use_8bit_optimizer = bool(use_8bit_optimizer)
        self.max_new_tokens = int(max_new_tokens)
        self.lora_target_modules = list(lora_target_modules)

        self.save_hyperparameters()

        # LoRA 模式下 on_save_checkpoint 仅保存可训练 Adapter 权重，因此关闭 strict_loading
        if self.mode == "lora":
            self.strict_loading = False

        self.model = build_causal_lm(
            model_name_or_path=model_name_or_path,
            mode=self.mode,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            use_gradient_checkpointing=use_gradient_checkpointing,
            eos_token_id=eos_token_id,
            trust_remote_code=trust_remote_code,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=self.lora_target_modules,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> Any:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        bsz = batch["input_ids"].size(0)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=bsz,
        )
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        val_loss = outputs.loss
        bsz = batch["input_ids"].size(0)
        self.log(
            "val/loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=bsz,
        )
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=False,
            sync_dist=True,
            batch_size=bsz,
        )
        return val_loss

    def on_validation_epoch_end(self) -> None:
        val_loss = self.trainer.callback_metrics.get("val/loss")
        if val_loss is not None:
            val_ppl = torch.exp(torch.clamp(val_loss.detach(), max=20.0))
            self.log(
                "val/ppl",
                val_ppl,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        test_loss = outputs.loss
        bsz = batch["input_ids"].size(0)
        self.log(
            "test/loss",
            test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=bsz,
        )
        return test_loss

    def on_test_epoch_end(self) -> None:
        test_loss = self.trainer.callback_metrics.get("test/loss")
        if test_loss is not None:
            test_ppl = torch.exp(torch.clamp(test_loss.detach(), max=20.0))
            self.log(
                "test/ppl",
                test_ppl,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

    def predict_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> dict[str, Any]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        prompts = batch.get("prompts", [""] * input_ids.size(0))
        with torch.no_grad():
            gen_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=getattr(self.model.config, "eos_token_id", 0),
            )
        new_tokens = gen_ids[:, input_ids.size(1) :].detach().cpu().tolist()
        return {
            "prompts": prompts,
            "generated_token_ids": new_tokens,
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """在 LoRA 模式下仅保存可训练的 LoRA 权重，大幅减小检查点体积。"""
        if self.mode == "lora":
            trainable_keys = {name for name, param in self.named_parameters() if param.requires_grad}
            checkpoint["state_dict"] = {
                k: v for k, v in checkpoint["state_dict"].items() if k in trainable_keys
            }

    def configure_optimizers(self) -> dict[str, Any]:
        # 对 Bias 和 LayerNorm / RMSNorm 层禁用 Weight Decay
        no_decay = ("bias", "LayerNorm.weight", "layernorm.weight", "norm.weight")
        decay_params = [
            p
            for n, p in self.model.named_parameters()
            if p.requires_grad and not any(nd in n for nd in no_decay)
        ]
        no_decay_params = [
            p
            for n, p in self.model.named_parameters()
            if p.requires_grad and any(nd in n for nd in no_decay)
        ]
        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g["params"]) > 0]

        betas = (self.adam_beta1, self.adam_beta2)
        if self.use_8bit_optimizer:
            try:
                import bitsandbytes as bnb  # type: ignore[import-not-found]

                optimizer = bnb.optim.PagedAdamW8bit(
                    optimizer_grouped_parameters,
                    lr=self.learning_rate,
                    betas=betas,
                )
                console.print(
                    "[bold green][Optimizer][/bold green] 已启用 bitsandbytes Paged 8-bit AdamW 优化器。"
                )
            except ImportError:
                console.print(
                    "[yellow][Warning] 未安装 bitsandbytes，已回退至标准 torch.optim.AdamW。[/yellow]"
                )
                optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters,
                    lr=self.learning_rate,
                    betas=betas,
                )
        else:
            # Lightning 在开启 AMP (bf16-mixed) + gradient_clip_val 时不兼容 fused=True
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.learning_rate,
                betas=betas,
            )

        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        warmup_steps = int(total_steps * self.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
