"""Two-stage LoRA + FP32 linear head LightningModule for Qwen3.8-27B Jev Decision Model."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from lmml.core.llm.jev.api import format_response
from lmml.core.llm.jev.data import JevDataModule
from lmml.core.llm.jev.metrics import (
    evaluate_probabilities,
    evaluate_split_records,
    fit_temperature,
    softmax,
)
from lmml.core.llm.jev.model import DEFAULT_JEV_LORA_TARGETS, JevDecisionModel


def jev_nll_brier_loss(
    logits: torch.Tensor,
    target: Sequence[float] | torch.Tensor,
    *,
    brier_weight: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute cross-entropy (soft-label NLL) + brier_weight * Brier loss for one question row."""
    logits_fp32 = logits.float()
    if isinstance(target, torch.Tensor):
        target_tensor = target.to(device=logits_fp32.device, dtype=torch.float32)
    else:
        target_tensor = torch.tensor(target, device=logits_fp32.device, dtype=torch.float32)
    log_probs = F.log_softmax(logits_fp32, dim=-1)
    probs = torch.softmax(logits_fp32, dim=-1)
    ce_loss = -(target_tensor * log_probs).sum(dim=-1)
    brier_loss = ((probs - target_tensor) ** 2).sum(dim=-1)
    total_loss = ce_loss + float(brier_weight) * brier_loss
    return total_loss, ce_loss.detach(), brier_loss.detach()


class QwenLoraLightningModule(pl.LightningModule):
    """通用两段式加载模块（Frozen Qwen3.8/3.5 Backbone + LoRA Adapter + FP32 Scalar Head）：

    1. Case A (`lora_checkpoint_dir` 存在)：从已有目录通过 `PeftModel.from_pretrained(..., is_trainable=True)`
       加载 LoRA 权重及 `head.pt` / `temperature.json`；
    2. Case B (`lora_checkpoint_dir` 未提供)：使用 `lora_config` 调用 `get_peft_model` 全新初始化 LoRA。
    """

    def __init__(
        self,
        base_model_path: str,
        tokenizer: Any | None = None,
        *,
        lora_config: Any | None = None,
        lora_checkpoint_dir: str | None = None,
        lr: float = 2e-4,
        head_lr: float | None = None,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.05,
        lr_scheduler_type: str = "cosine",
        min_lr_ratio: float = 0.1,
        brier_weight: float = 0.1,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] | None = None,
        max_length: int = 512,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        gradient_checkpointing: bool = True,
        trust_remote_code: bool = True,
        strict_length: bool = True,
        use_liger_kernel: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=["tokenizer", "lora_config", "lora_checkpoint_dir", "model"]
        )
        self.base_model_path = str(base_model_path)
        self.lora_checkpoint_dir = lora_checkpoint_dir
        self.lr = float(lr)
        self.head_lr = float(head_lr if head_lr is not None else lr)
        self.weight_decay = float(weight_decay)
        self.warmup_ratio = float(warmup_ratio)
        self.lr_scheduler_type = str(lr_scheduler_type).strip().lower()
        self.min_lr_ratio = min(1.0, max(0.0, float(min_lr_ratio)))
        if self.lr_scheduler_type not in {"cosine", "constant", "constant_with_warmup"}:
            raise ValueError(
                "lr_scheduler_type must be 'cosine', 'constant', or 'constant_with_warmup'"
            )
        self.brier_weight = float(brier_weight)

        self.model = JevDecisionModel(
            base_model_path=self.base_model_path,
            tokenizer=tokenizer,
            lora_config=lora_config,
            lora_checkpoint_dir=lora_checkpoint_dir,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules or DEFAULT_JEV_LORA_TARGETS,
            max_length=max_length,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            gradient_checkpointing=gradient_checkpointing,
            trust_remote_code=trust_remote_code,
            is_trainable=True,
            strict_length=strict_length,
            use_liger_kernel=use_liger_kernel,
        )
        self.tokenizer = self.model.tokenizer
        self._val_logits: list[list[float]] = []
        self._val_targets: list[list[float]] = []
        self._test_logits: list[list[float]] = []
        self._test_records: list[dict[str, Any]] = []
        self.calibration_stats: dict[str, Any] = {}
        self.stage3_reports: dict[str, Any] = {}

    @property
    def temperature(self) -> float:
        return float(self.model.temperature)

    @temperature.setter
    def temperature(self, value: float) -> None:
        self.model.temperature = float(value)

    def forward(self, batch_or_records: Any) -> list[torch.Tensor]:
        return self.model(batch_or_records)

    def _compute_batch_loss(
        self,
        batch: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        logits_list = self.model(batch)
        targets = batch["targets"]
        losses: list[torch.Tensor] = []
        ce_losses: list[torch.Tensor] = []
        brier_losses: list[torch.Tensor] = []
        for row_logits, row_target in zip(logits_list, targets):
            loss, ce, brier = jev_nll_brier_loss(
                row_logits,
                row_target,
                brier_weight=self.brier_weight,
            )
            losses.append(loss)
            ce_losses.append(ce)
            brier_losses.append(brier)
        return (
            torch.stack(losses).mean(),
            torch.stack(ce_losses).mean(),
            torch.stack(brier_losses).mean(),
            logits_list,
        )

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, ce_loss, brier_loss, _ = self._compute_batch_loss(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss encountered in Jev decision model")
        bsz = len(batch["counts"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bsz)
        self.log("train/nll", ce_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=bsz)
        self.log("train/brier", brier_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=bsz)
        self.log("train/ce_loss", ce_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=bsz)
        self.log("train/brier_loss", brier_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=bsz)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, ce_loss, brier_loss, logits_list = self._compute_batch_loss(batch)
        bsz = len(batch["counts"])
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bsz)
        self.log("val/nll", ce_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=bsz)
        self.log("val/brier", brier_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=bsz)
        for row_logits, row_target in zip(logits_list, batch["targets"]):
            self._val_logits.append(row_logits.detach().float().cpu().tolist())
            self._val_targets.append([float(x) for x in row_target])
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self._val_logits:
            return
        probs = [softmax(row, temperature=self.temperature) for row in self._val_logits]
        summary = evaluate_probabilities(self._val_targets, probs)
        self._val_logits.clear()
        self._val_targets.clear()
        if summary.get("expected_accuracy") is not None:
            self.log("val/expected_acc", float(summary["expected_accuracy"]), prog_bar=True, sync_dist=True)
        if summary.get("accuracy") is not None:
            self.log("val/acc", float(summary["accuracy"]), prog_bar=False, sync_dist=True)
        self.log("val/ece", float(summary["multiclass_ece"]), prog_bar=True, sync_dist=True)

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _, _, logits_list = self._compute_batch_loss(batch)
        bsz = len(batch["counts"])
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bsz)
        for row_logits, rec in zip(logits_list, batch["records"]):
            self._test_logits.append(row_logits.detach().float().cpu().tolist())
            self._test_records.append(rec)
        return loss

    def on_test_epoch_end(self) -> None:
        if not self._test_logits:
            return
        report = evaluate_split_records(
            self._test_records,
            self._test_logits,
            temperature=self.temperature,
        )
        self._test_logits.clear()
        self._test_records.clear()
        self.stage3_reports["test"] = report
        if report.get("expected_accuracy") is not None:
            self.log("test/expected_acc", float(report["expected_accuracy"]), prog_bar=True, sync_dist=True)
        if report.get("accuracy") is not None:
            self.log("test/acc", float(report["accuracy"]), prog_bar=False, sync_dist=True)
        self.log("test/nll", float(report["nll"]), prog_bar=True, sync_dist=True)
        self.log("test/brier", float(report["brier"]), prog_bar=True, sync_dist=True)
        self.log("test/ece", float(report["multiclass_ece"]), prog_bar=True, sync_dist=True)

    def predict_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> dict[str, Any]:
        start = time.perf_counter()
        logits_list = self.model(batch)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        probs_list = [softmax(row.detach().float().cpu().tolist(), temperature=self.temperature) for row in logits_list]

        results: list[dict[str, Any]] = []
        offset = 0
        for request, group_records in zip(batch["requests"], batch["compiled_groups"]):
            count = len(group_records)
            group_probs = probs_list[offset : offset + count]
            formatted = format_response(group_records, group_probs)
            results.append(
                {
                    "model": request.get("model", "qwen38-27b-jev"),
                    "state": request["state"],
                    "answers": formatted["answers"],
                    "temperature": self.temperature,
                    "usage": {
                        "input_tokens": int(self.model.last_input_tokens),
                        "output_tokens": 0,
                    },
                    "latency_ms": elapsed_ms / max(1, len(batch["requests"])),
                }
            )
            offset += count
        return {"results": results}

    def calibrate(
        self,
        datamodule: JevDataModule,
        *,
        max_samples: int = 4000,
        batch_size: int = 1,
    ) -> float:
        """Stage 2: Fit optimal positive temperature T on calibration.jsonl by batch iteration."""
        was_training = self.training
        self.eval()
        self.model.eval()
        collected_logits: list[list[float]] = []
        collected_targets: list[list[float]] = []
        skipped_batches = 0

        try:
            loader = datamodule.calibration_dataloader(
                max_samples=max_samples,
                batch_size=batch_size,
            )
            with torch.no_grad():
                for batch in loader:
                    try:
                        logits_list = self.model(batch)
                    except ValueError:
                        skipped_batches += 1
                        continue
                    for row_logits, row_target in zip(logits_list, batch["targets"]):
                        collected_logits.append(row_logits.detach().float().cpu().tolist())
                        collected_targets.append([float(x) for x in row_target])

            if not collected_logits:
                raise ValueError("Calibration produced no valid logit rows")

            before = evaluate_probabilities(
                collected_targets,
                [softmax(row, temperature=1.0) for row in collected_logits],
            )
            temperature = fit_temperature(collected_logits, collected_targets)
            after = evaluate_probabilities(
                collected_targets,
                [softmax(row, temperature=temperature) for row in collected_logits],
            )
            self.temperature = temperature
            self.calibration_stats = {
                "status": "completed",
                "temperature": float(temperature),
                "records": len(collected_logits),
                "skipped_batches": skipped_batches,
                "batch_size": int(batch_size),
                "nll_before": float(before["nll"]),
                "nll_after": float(after["nll"]),
                "ece_before": float(before["multiclass_ece"]),
                "ece_after": float(after["multiclass_ece"]),
                "brier_before": float(before["brier"]),
                "brier_after": float(after["brier"]),
            }
            return float(temperature)
        except Exception:
            self.temperature = 1.0
            raise
        finally:
            self.train(was_training)
            self.model.train(was_training)

    def evaluate_stage3(
        self,
        datamodule: JevDataModule,
        *,
        splits: Sequence[str] = ("test", "test_set_30k", "ood"),
        max_samples: int | None = None,
        batch_size: int = 1,
    ) -> dict[str, Any]:
        """Stage 3: Evaluate test (smoke & throughput), test_set_30k (Jev Decision Index), and ood (System 1/2 escalation)."""
        was_training = self.training
        self.eval()
        self.model.eval()
        reports: dict[str, Any] = {}
        try:
            datamodule.setup("test")
            for split_name in splits:
                if split_name not in datamodule.splits:
                    continue
                loader = datamodule.split_dataloader(
                    split_name,
                    max_samples=max_samples,
                    batch_size=batch_size,
                )
                logits_rows: list[list[float]] = []
                records_rows: list[dict[str, Any]] = []
                total_tokens = 0
                start_time = time.perf_counter()
                with torch.no_grad():
                    for batch in loader:
                        out_logits = self.model(batch)
                        total_tokens += int(self.model.last_input_tokens)
                        for row_logits, rec in zip(out_logits, batch["records"]):
                            logits_rows.append(row_logits.detach().float().cpu().tolist())
                            records_rows.append(rec)
                elapsed = max(1e-9, time.perf_counter() - start_time)
                if not records_rows:
                    continue
                split_metrics = evaluate_split_records(
                    records_rows,
                    logits_rows,
                    temperature=self.temperature,
                )
                split_metrics["throughput"] = {
                    "elapsed_seconds": elapsed,
                    "rows_per_second": len(records_rows) / elapsed,
                    "mean_latency_ms": (elapsed * 1000.0) / len(records_rows),
                    "total_input_tokens": total_tokens,
                    "tokens_per_second": total_tokens / elapsed,
                }
                reports[split_name] = split_metrics
            self.stage3_reports.update(reports)
            return reports
        finally:
            self.train(was_training)
            self.model.train(was_training)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Persist only trainable LoRA + scalar head weights and calibration temperature in Lightning .ckpt."""
        trainable_keys = {name for name, param in self.named_parameters() if param.requires_grad}
        checkpoint["state_dict"] = {
            k: v for k, v in checkpoint["state_dict"].items() if k in trainable_keys
        }
        checkpoint["jev_temperature"] = float(self.temperature)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if "jev_temperature" in checkpoint:
            self.temperature = float(checkpoint["jev_temperature"])

    def load_trainable_from_lightning_ckpt(self, ckpt_path: str | Path) -> None:
        raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = raw.get("state_dict", raw)
        self.load_state_dict(state_dict, strict=False)
        if "jev_temperature" in raw:
            self.temperature = float(raw["jev_temperature"])

    def configure_optimizers(self) -> Any:
        """AdamW with decay/no-decay groups + linear warmup + cosine decay.

        LoRA matrices and the FP32 scalar head share the same optimizer but keep
        separate learning rates. Biases and normalization parameters never get
        weight decay.
        """
        head_ids = {id(p) for p in self.model.head.parameters()}
        backbone_named = [
            (name, param)
            for name, param in self.model.named_parameters()
            if param.requires_grad and id(param) not in head_ids
        ]
        head_params = [p for p in self.model.head.parameters() if p.requires_grad]

        def _no_decay(name: str) -> bool:
            lowered = name.lower()
            return "bias" in lowered or "norm" in lowered or "layernorm" in lowered

        backbone_decay = [p for name, p in backbone_named if not _no_decay(name)]
        backbone_no_decay = [p for name, p in backbone_named if _no_decay(name)]

        param_groups: list[dict[str, Any]] = []
        if backbone_decay:
            param_groups.append(
                {
                    "params": backbone_decay,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                    "name": "lora_decay",
                }
            )
        if backbone_no_decay:
            param_groups.append(
                {
                    "params": backbone_no_decay,
                    "lr": self.lr,
                    "weight_decay": 0.0,
                    "name": "lora_no_decay",
                }
            )
        if head_params:
            param_groups.append(
                {
                    "params": head_params,
                    "lr": self.head_lr,
                    "weight_decay": 0.0,
                    "name": "scalar_head",
                }
            )

        optimizer = torch.optim.AdamW(param_groups, lr=self.lr, weight_decay=self.weight_decay)
        if getattr(self, "_trainer", None) is None:
            return optimizer

        total_steps = max(int(getattr(self.trainer, "estimated_stepping_batches", 1)), 1)
        warmup_steps = (
            min(total_steps, max(int(total_steps * self.warmup_ratio), 1))
            if self.warmup_ratio > 0
            else 0
        )

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            if self.lr_scheduler_type in {"constant", "constant_with_warmup"}:
                return 1.0
            decay_steps = max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, float(step - warmup_steps) / decay_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": f"jev_{self.lr_scheduler_type}",
            },
        }


JevLoraLightningModule = QwenLoraLightningModule

__all__ = [
    "JevLoraLightningModule",
    "QwenLoraLightningModule",
    "jev_nll_brier_loss",
]
