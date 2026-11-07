"""Lightning module for Kev decision-model training and prediction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from lmml.core.llm.kev.api import output_tokens, to_answers
from lmml.core.llm.kev.checkpoint import warm_start_kev
from lmml.core.llm.kev.data import KevDataModule
from lmml.core.llm.kev.metrics import fit_temperature, metrics
from lmml.core.llm.kev.model import KevDecisionModel


def question_loss(
    logits: torch.Tensor,
    label: int,
    *,
    target: Sequence[float] | None = None,
    qtype: str = "choice",
    ord_weight: float = 0.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if target is not None:
        target_tensor = torch.tensor(target, dtype=logits.dtype, device=logits.device)
        return -(target_tensor * F.log_softmax(logits, dim=-1)).sum()
    labels = torch.tensor([int(label)], dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits.unsqueeze(0), labels, label_smoothing=label_smoothing)
    if qtype == "score" and ord_weight > 0 and logits.numel() > 1:
        probabilities = F.softmax(logits, dim=-1)
        observed = (torch.arange(len(probabilities), device=logits.device) >= int(label)).to(probabilities.dtype)
        loss = loss + ord_weight * (probabilities.cumsum(-1)[:-1] - observed[:-1]).square().mean()
    return loss


class KevLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: Any,
        *,
        mode: str = "lora",
        learning_rate: float = 1e-4,
        head_learning_rate: float | None = None,
        weight_decay: float = 0.0,
        warmup_ratio: float = 0.1,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        gradient_checkpointing: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Sequence[str] | None = None,
        pointer_dim: int = 256,
        option_isolation: bool = False,
        trust_remote_code: bool = True,
        ord_weight: float = 0.1,
        label_smoothing: float = 0.0,
        init_from: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer", "init_from"])
        self.tokenizer = tokenizer
        self.learning_rate = float(learning_rate)
        self.head_learning_rate = float(head_learning_rate or learning_rate)
        self.weight_decay = float(weight_decay)
        self.warmup_ratio = float(warmup_ratio)
        self.ord_weight = float(ord_weight)
        self.label_smoothing = float(label_smoothing)
        self.model = KevDecisionModel(
            model_name_or_path,
            tokenizer,
            mode=mode,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            gradient_checkpointing=gradient_checkpointing,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            pointer_dim=pointer_dim,
            option_isolation=option_isolation,
            trust_remote_code=trust_remote_code,
        )
        self.init_source = warm_start_kev(self.model, init_from) if init_from else None
        self._val_rows: list[dict[str, Any]] = []
        self._test_rows: list[dict[str, Any]] = []
        self.calibration_stats: dict[str, Any] = {}

    def _batch_loss(
        self, records: list[dict[str, Any]]
    ) -> tuple[torch.Tensor, list[list[torch.Tensor]], list[list[float]]]:
        if not records:
            zero = next(self.model.parameters()).sum() * 0.0
            return zero, [], []
        logits = self.model.forward_batch(records)
        losses: list[torch.Tensor] = []
        probability_rows: list[list[float]] = []
        for record, record_logits in zip(records, logits):
            record_losses = []
            probabilities = []
            for question_index, values in enumerate(record_logits):
                record_losses.append(
                    question_loss(
                        values.float(),
                        int(record["labels"][question_index]),
                        target=record.get("targets", [None] * len(record_logits))[question_index],
                        qtype=record.get("qtypes", ["choice"] * len(record_logits))[question_index],
                        ord_weight=self.ord_weight,
                        label_smoothing=self.label_smoothing,
                    )
                )
                probabilities.append(F.softmax(values.float(), dim=-1).detach().cpu().tolist())
            losses.append(torch.stack(record_losses).mean())
            probability_rows.append(probabilities)
        return torch.stack(losses).mean(), logits, probability_rows

    def _batch_probabilities(self, records: list[dict[str, Any]]) -> list[list[float]]:
        if not records:
            return []
        logits = self.model.forward_batch(records)
        probability_rows = [
            [
                F.softmax(values.float(), dim=-1).detach().cpu().tolist()
                for values in record_logits
            ]
            for record_logits in logits
        ]
        del logits
        return probability_rows

    @staticmethod
    def _metric_rows(records: list[dict[str, Any]], probabilities: list[list[float]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for record, record_probabilities in zip(records, probabilities):
            for index, values in enumerate(record_probabilities):
                rows.append(
                    {
                        "probabilities": values,
                        "label": int(record["labels"][index]),
                        "qtype": record.get("qtypes", ["choice"] * len(record_probabilities))[index],
                        "keys": record.get("keys", [None] * len(record_probabilities))[index],
                    }
                )
        return rows

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        records = batch.get("records", [])
        loss, _, _ = self._batch_loss(records)
        self.log("train/loss", loss, prog_bar=True, batch_size=max(1, len(records)))
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        records = batch.get("records", [])
        raw_temperature = self.model.head.temperature
        self.model.head.temperature = 1.0
        try:
            loss, _, probabilities = self._batch_loss(records)
        finally:
            self.model.head.temperature = raw_temperature
        self._val_rows.extend(self._metric_rows(records, probabilities))
        self.log("val/loss", loss, prog_bar=True, sync_dist=True, batch_size=max(1, len(records)))
        return loss

    def on_validation_epoch_end(self) -> None:
        if self._val_rows:
            values = metrics(self._val_rows)
            for name in ("acc", "brier", "ece", "confident_error_rate", "coverage_at_5pct_error"):
                if name in values:
                    self.log(f"val/{name}", values[name], sync_dist=True)
        self._val_rows.clear()

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        records = batch.get("records", [])
        raw_temperature = self.model.head.temperature
        self.model.head.temperature = 1.0
        try:
            loss, _, probabilities = self._batch_loss(records)
        finally:
            self.model.head.temperature = raw_temperature
        self._test_rows.extend(self._metric_rows(records, probabilities))
        self.log("test/loss", loss, prog_bar=True, sync_dist=True, batch_size=max(1, len(records)))
        return loss

    def on_test_epoch_end(self) -> None:
        if self._test_rows:
            values = metrics(self._test_rows)
            for name in ("acc", "brier", "ece", "nll", "confident_error_rate", "coverage_at_5pct_error", "score_mae"):
                if name in values:
                    self.log(f"test/{name}", values[name], sync_dist=True)
        self._test_rows.clear()

    def predict_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> dict[str, Any]:
        items = batch["items"]
        encodings = [item["encoded"] for item in items]
        probabilities = self.model.probabilities(encodings)
        results = []
        for request, item, record_probabilities in zip(batch["requests"], items, probabilities):
            values = [[float(value) for value in probs.tolist()] for probs in record_probabilities]
            answers = to_answers(values, item["metadata"])
            results.append(
                {
                    "state": request["state"],
                    "answers": answers,
                    "probabilities": values,
                    "usage": {
                        "input_tokens": sum(len(encoded["ids"]) for encoded in [item["encoded"]]),
                        "output_tokens": output_tokens(self.tokenizer, answers),
                    },
                }
            )
        return {"results": results}

    @torch.no_grad()
    def calibrate(
        self,
        datamodule: KevDataModule,
        max_samples: int = 4000,
        batch_size: int = 1,
    ) -> float:
        """Fit one temperature from calibration records in bounded batches."""
        max_samples = int(max_samples)
        batch_size = int(batch_size)
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.calibration_stats = {}
        loader = datamodule.calibration_dataloader(
            max_samples=max_samples,
            batch_size=batch_size,
        )
        selected_records = len(loader.dataset)
        was_training = self.model.training
        self.model.eval()
        self.model.head.temperature = 1.0
        metric_rows: list[dict[str, Any]] = []
        accepted_records = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            for batch in loader:
                records = batch.get("records", [])
                if not records:
                    continue
                probabilities = self._batch_probabilities(records)
                metric_rows.extend(self._metric_rows(records, probabilities))
                accepted_records += len(records)
            if not metric_rows:
                raise ValueError("calibration produced no usable question rows")
            temperature = fit_temperature(metric_rows, aggregation="micro", points=121)
        except BaseException:
            self.model.head.temperature = 1.0
            raise
        finally:
            if was_training:
                self.model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.model.head.temperature = temperature
        self.calibration_stats = {
            "status": "success",
            "selected_records": selected_records,
            "accepted_records": accepted_records,
            "skipped_records": selected_records - accepted_records,
            "question_rows": len(metric_rows),
            "max_samples": max_samples,
            "batch_size": batch_size,
            "temperature": temperature,
        }
        return temperature

    def configure_optimizers(self) -> Any:
        head_parameters = list(self.model.head.parameters())
        head_ids = {id(parameter) for parameter in head_parameters}
        backbone_parameters = [
            parameter
            for parameter in self.model.trainable_parameters()
            if id(parameter) not in head_ids
        ]
        groups = []
        if backbone_parameters:
            groups.append(
                {
                    "params": backbone_parameters,
                    "lr": self.learning_rate,
                    "weight_decay": self.weight_decay,
                }
            )
        groups.append(
            {
                "params": head_parameters,
                "lr": self.head_learning_rate,
                "weight_decay": self.weight_decay,
            }
        )
        optimizer = torch.optim.AdamW(groups)
        total_steps = max(1, int(getattr(self.trainer, "estimated_stepping_batches", 1) or 1))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group["lr"] for group in groups],
            total_steps=total_steps,
            pct_start=max(0.0, min(0.9, self.warmup_ratio)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


__all__ = ["KevLightningModule", "question_loss"]
