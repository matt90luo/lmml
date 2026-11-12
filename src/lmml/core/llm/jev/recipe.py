"""Hydra Recipe for Qwen3.8-27B Jev Decision Model training, Top-2 saving, calibration, Stage-3 evaluation, and prediction."""

from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table

from lmml.core.llm.jev.data import JevDataModule
from lmml.core.llm.jev.model import DEFAULT_JEV_LORA_TARGETS
from lmml.core.llm.jev.module import QwenLoraLightningModule
from lmml.core.llm.kev.recipe import DEFAULT_DEMO_REQUEST
from lmml.core.recipe import Recipe
from lmml.datas.llm.sft import build_sft_tokenizer

console = Console()


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class JevStage3EvaluationCallback(pl.Callback):
    """Run Stage 3 full suite evaluation (test, test_set_30k, ood) at the end of `lmml test`."""

    def __init__(self, recipe: JevDecisionRecipe, datamodule: JevDataModule) -> None:
        super().__init__()
        self.recipe = recipe
        self.datamodule = datamodule

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        assert isinstance(pl_module, QwenLoraLightningModule)
        self.recipe.run_stage3_evaluation(pl_module, self.datamodule, trainer=trainer)


class JevDecisionRecipe(Recipe):
    """Unified Recipe for Qwen3.8-27B Jev Decision Model (`qwen38_27B_jev`)."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        torch.set_float32_matmul_precision(str(cfg.algo.get("matmul_precision", "high")))

    def _tokenizer(self) -> Any:
        model_cfg = self.cfg.algo.model
        source = str(self.cfg.data.get("tokenizer_name_or_path") or model_cfg.name_or_path)
        return build_sft_tokenizer(
            source,
            str(self.cfg.data.get("chat_end_token", "<|im_end|>")),
            bool(model_cfg.get("trust_remote_code", True)),
        )

    def _predict_requests(self) -> list[dict[str, Any]]:
        predict = self.cfg.get("predict") or {}
        configured = predict.get("requests")
        if configured is None:
            configured = self.cfg.data.get("predict_requests")
        if configured is not None:
            requests = OmegaConf.to_container(configured, resolve=True)
            return list(requests or [])
        path = predict.get("requests_path")
        if path:
            with Path(str(path)).open(encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
        return [DEFAULT_DEMO_REQUEST]

    def build_datamodule(self) -> JevDataModule:
        data = self.cfg.data
        prepared_path = data.get("prepared_path")
        if not prepared_path:
            raise ValueError("Jev training requires data.prepared_path; run `lmml data` first")
        tokenizer = self._tokenizer()
        datamodule = JevDataModule(
            str(prepared_path),
            tokenizer,
            batch_size=int(data.get("batch_size", 1)),
            val_batch_size=int(data.get("val_batch_size", data.get("batch_size", 1))),
            num_workers=int(data.get("num_workers", 0)),
            max_length=int(data.get("max_length", 512)),
            strict_length=bool(data.get("strict_length", False)),
            predict_requests=self._predict_requests(),
            train_max_samples=(
                data.get("train_max_samples")
                if data.get("train_max_samples") is not None
                else data.get("max_train_samples")
            ),
            val_max_samples=(
                data.get("val_max_samples")
                if data.get("val_max_samples") is not None
                else data.get("max_val_samples")
            ),
            train_sampling=str(data.get("train_sampling", "source_kind_round_robin")),
            sample_seed=int(data.get("sample_seed", self.cfg.seed)),
        )
        datamodule.tokenizer = tokenizer
        return datamodule

    def build_module(
        self,
        datamodule: pl.LightningDataModule,
        ckpt_path: str | None = None,
    ) -> QwenLoraLightningModule:
        cfg = self.cfg
        algo = cfg.algo
        model_cfg = algo.model
        tokenizer = getattr(datamodule, "tokenizer", None) or self._tokenizer()

        lora_ckpt_dir = algo.get("lora_checkpoint_dir") or algo.get("init_from")
        lightning_ckpt: Path | None = None
        if ckpt_path:
            candidate = Path(str(ckpt_path))
            if candidate.is_dir():
                lora_ckpt_dir = str(candidate)
            elif candidate.is_file() and candidate.suffix == ".ckpt":
                lightning_ckpt = candidate
            else:
                lora_ckpt_dir = str(candidate)

        raw_targets = algo.get("lora_target_modules")
        targets = (
            OmegaConf.to_container(raw_targets, resolve=True)
            if raw_targets is not None
            else list(DEFAULT_JEV_LORA_TARGETS)
        )

        module = QwenLoraLightningModule(
            base_model_path=str(model_cfg.name_or_path),
            tokenizer=tokenizer,
            lora_checkpoint_dir=str(lora_ckpt_dir) if lora_ckpt_dir else None,
            lr=float(algo.get("lr", 2e-4)),
            head_lr=float(algo.get("head_lr", algo.get("lr", 2e-4))),
            weight_decay=float(algo.get("weight_decay", 0.01)),
            warmup_ratio=float(algo.get("warmup_ratio", 0.05)),
            lr_scheduler_type=str(algo.get("lr_scheduler_type", "cosine")),
            min_lr_ratio=float(algo.get("min_lr_ratio", 0.1)),
            brier_weight=float(algo.get("brier_weight", 0.1)),
            lora_r=int(algo.get("lora_r", 32)),
            lora_alpha=int(algo.get("lora_alpha", 64)),
            lora_dropout=float(algo.get("lora_dropout", 0.05)),
            lora_target_modules=targets,
            max_length=int(cfg.data.get("max_length", 512)),
            torch_dtype=str(model_cfg.get("torch_dtype", "bfloat16")),
            attn_implementation=str(model_cfg.get("attn_implementation", "sdpa")),
            gradient_checkpointing=bool(model_cfg.get("use_gradient_checkpointing", True)),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
            strict_length=bool(cfg.data.get("strict_length", False)),
            use_liger_kernel=bool(model_cfg.get("use_liger_kernel", False)),
        )
        if lightning_ckpt is not None:
            module.load_trainable_from_lightning_ckpt(lightning_ckpt)
        return module

    def build_callbacks(self, datamodule: pl.LightningDataModule) -> list[pl.Callback]:
        callbacks: list[pl.Callback] = []
        cb_cfg = self.cfg.get("callbacks") or {}
        ckpt_cfg = cb_cfg.get("checkpoint") or {}
        if bool(ckpt_cfg.get("enabled", True)):
            dirpath = ckpt_cfg.get("dirpath")
            if dirpath:
                Path(str(dirpath)).mkdir(parents=True, exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    dirpath=str(dirpath) if dirpath else None,
                    filename=str(ckpt_cfg.get("filename", "qwen38-27b-jev-{epoch:02d}-{step:06d}")),
                    monitor=str(ckpt_cfg.get("monitor", "val/loss")),
                    mode=str(ckpt_cfg.get("mode", "min")),
                    save_top_k=int(ckpt_cfg.get("save_top_k", 2)),
                    save_last=bool(ckpt_cfg.get("save_last", False)),
                    save_weights_only=bool(ckpt_cfg.get("save_weights_only", True)),
                )
            )
        lr_cfg = cb_cfg.get("lr_monitor") or {}
        if bool(lr_cfg.get("enabled", True)):
            callbacks.append(LearningRateMonitor(logging_interval="step"))
        if isinstance(datamodule, JevDataModule):
            callbacks.append(JevStage3EvaluationCallback(self, datamodule))
        return callbacks

    def adapt_trainer_kwargs(
        self,
        trainer_kwargs: dict[str, Any],
        datamodule: pl.LightningDataModule,
    ) -> dict[str, Any]:
        """Translate optimizer-step validation policy into Lightning batch units."""
        kwargs = dict(trainer_kwargs)
        datamodule.setup("fit")
        train_dataset = getattr(datamodule, "train_dataset", None)
        if train_dataset is None:
            return kwargs

        batch_size = max(1, int(self.cfg.data.get("batch_size", 1)))
        num_batches = max(1, math.ceil(len(train_dataset) / batch_size))
        limit_train = kwargs.get("limit_train_batches")
        if isinstance(limit_train, int) and 0 < limit_train < num_batches:
            num_batches = limit_train

        accumulate_raw = kwargs.get("accumulate_grad_batches", 1)
        if isinstance(accumulate_raw, dict):
            accumulate = int(max(accumulate_raw.values(), default=1))
        else:
            accumulate = max(1, int(accumulate_raw or 1))

        val_every_opt_steps = self.cfg.algo.get("val_every_n_opt_steps")
        if val_every_opt_steps is not None and int(val_every_opt_steps) > 0:
            target = int(val_every_opt_steps) * accumulate
            kwargs["val_check_interval"] = min(num_batches, max(1, target))
        else:
            val_interval = kwargs.get("val_check_interval")
            if isinstance(val_interval, int) and val_interval > num_batches:
                kwargs["val_check_interval"] = num_batches

        max_epochs = int(kwargs.get("max_epochs", 1) or 1)
        total_micro_batches = num_batches * max_epochs
        total_opt_steps = max(1, math.ceil(total_micro_batches / accumulate))
        actual_interval = int(kwargs.get("val_check_interval", num_batches))
        console.print(
            "[bold cyan][Jev Batch Map][/bold cyan] "
            f"micro_batch={batch_size} x accumulate={accumulate} "
            f"=> global_batch={batch_size * accumulate}; "
            f"micro_batches/epoch={num_batches:,}, optimizer_steps/epoch={math.ceil(num_batches / accumulate):,}; "
            f"epochs={max_epochs}, total_optimizer_steps={total_opt_steps:,}; "
            f"val_every={actual_interval:,} micro_batches "
            f"({math.ceil(actual_interval / accumulate):,} optimizer steps)"
        )
        return kwargs

    def _export_top_k_peft_dirs(
        self,
        trainer: pl.Trainer,
        module: QwenLoraLightningModule,
        temperature: float,
        calibration_meta: dict[str, Any],
    ) -> list[str]:
        """Export up to Top-2 best validation checkpoints as PEFT + Head directories."""
        ckpt_cb = next(
            (cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)),
            None,
        )
        if ckpt_cb is None or not ckpt_cb.best_k_models:
            return []

        mode = str(ckpt_cb.mode).lower()
        sorted_items = sorted(
            ckpt_cb.best_k_models.items(),
            key=lambda kv: float(kv[1].item() if hasattr(kv[1], "item") else kv[1]),
            reverse=(mode == "max"),
        )
        ckpt_root = Path(str(ckpt_cb.dirpath)) if ckpt_cb.dirpath else Path(trainer.default_root_dir)
        exported_dirs: list[str] = []

        # Preserve current in-memory trainable state
        current_state = {
            k: v.detach().cpu().clone()
            for k, v in module.state_dict().items()
            if k in {n for n, p in module.named_parameters() if p.requires_grad}
        }
        try:
            for rank_idx, (ckpt_file, score) in enumerate(sorted_items[:2], start=1):
                ckpt_path = Path(ckpt_file)
                if not ckpt_path.is_file():
                    continue
                module.load_trainable_from_lightning_ckpt(ckpt_path)
                score_val = float(score.item() if hasattr(score, "item") else score)
                peft_dir = ckpt_root / f"top{rank_idx}_{ckpt_path.stem}"
                saved_dir = module.model.save(
                    peft_dir,
                    temperature=temperature,
                    extra={
                        "rank": rank_idx,
                        "source_ckpt": str(ckpt_path),
                        "monitor": ckpt_cb.monitor,
                        "monitor_score": score_val,
                        "calibration": calibration_meta,
                    },
                )
                exported_dirs.append(saved_dir)
        finally:
            module.load_state_dict(current_state, strict=False)

        if sorted_items and Path(sorted_items[0][0]).is_file():
            # Keep the #1 best validation checkpoint loaded in module for final export/eval
            module.load_trainable_from_lightning_ckpt(sorted_items[0][0])
        return exported_dirs

    def on_fit_end(
        self,
        trainer: pl.Trainer,
        module: pl.LightningModule,
        datamodule: pl.LightningDataModule,
    ) -> None:
        assert isinstance(module, QwenLoraLightningModule)
        assert isinstance(datamodule, JevDataModule)

        # Load best validation checkpoint before Stage 2 calibration if available
        ckpt_cb = next(
            (cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)),
            None,
        )
        if ckpt_cb is not None and ckpt_cb.best_model_path and Path(ckpt_cb.best_model_path).is_file():
            console.print(
                f"[bold cyan][Jev Best Checkpoint][/bold cyan] restoring best validation weights from {ckpt_cb.best_model_path}"
            )
            device = next(module.parameters()).device
            module.load_trainable_from_lightning_ckpt(ckpt_cb.best_model_path)
            module.to(device)

        calibration = self.cfg.algo.get("calibration") or {}
        temperature = 1.0
        calibration_error: Exception | None = None
        max_samples = int(calibration.get("max_samples", 4000))
        batch_size = int(calibration.get("batch_size", 1))
        calibration_meta: dict[str, Any] = {
            "status": "disabled",
            "temperature": temperature,
            "max_samples": max_samples,
            "batch_size": batch_size,
        }

        if bool(calibration.get("enabled", True)):
            _release_cuda_memory()
            console.print(
                f"[bold cyan][Jev Stage 2 Calibration][/bold cyan] fitting temperature on up to {max_samples} records (batch_size={batch_size})"
            )
            try:
                temperature = module.calibrate(
                    datamodule,
                    max_samples=max_samples,
                    batch_size=batch_size,
                )
                calibration_meta = dict(module.calibration_stats)
                console.print(
                    f"[bold green][Jev Stage 2 Calibration][/bold green] "
                    f"T={temperature:.4f} | ECE: {calibration_meta['ece_before']:.4f} -> {calibration_meta['ece_after']:.4f} | "
                    f"NLL: {calibration_meta['nll_before']:.4f} -> {calibration_meta['nll_after']:.4f}"
                )
            except Exception as exc:  # noqa: BLE001
                calibration_error = exc
                module.temperature = 1.0
                calibration_meta = {
                    "status": "failed",
                    "temperature": 1.0,
                    "max_samples": max_samples,
                    "batch_size": batch_size,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                console.print(
                    f"[bold red][Jev Calibration Failed][/bold red] {type(exc).__name__}: {exc}"
                )

        # Stage 3 optional automatic trigger after training
        eval_cfg = self.cfg.get("eval") or {}
        if bool(eval_cfg.get("after_train", False)) and calibration_error is None:
            self.run_stage3_evaluation(module, datamodule, trainer=trainer)

        # Export Top-2 PEFT directories and final checkpoint on CPU
        export_cfg = self.cfg.algo.get("export") or {}
        save_dir = export_cfg.get("save_dir")
        module.cpu()
        _release_cuda_memory()

        top_k_dirs = self._export_top_k_peft_dirs(trainer, module, temperature, calibration_meta)
        if top_k_dirs:
            console.print(
                f"[bold green][Jev Top-2 Export][/bold green] saved {len(top_k_dirs)} top validation checkpoints: {top_k_dirs}"
            )

        if bool(export_cfg.get("enabled", True)) and save_dir:
            saved = module.model.save(
                str(save_dir),
                temperature=temperature,
                extra={
                    "calibration": calibration_meta,
                    "top_k_checkpoints": top_k_dirs,
                    "stage3_reports": module.stage3_reports,
                },
            )
            console.print(f"[bold green][Jev Final Export][/bold green] {saved}")

        if calibration_error is not None:
            raise calibration_error

    def run_stage3_evaluation(
        self,
        module: QwenLoraLightningModule,
        datamodule: JevDataModule,
        *,
        trainer: pl.Trainer | None = None,
    ) -> dict[str, Any]:
        eval_cfg = self.cfg.get("eval") or {}
        splits = list(eval_cfg.get("splits") or ("test", "test_set_30k", "ood"))
        max_samples = eval_cfg.get("max_samples")
        batch_size = int(eval_cfg.get("batch_size", self.cfg.data.get("val_batch_size", 1)))

        console.print(
            f"[bold cyan][Jev Stage 3 Evaluation][/bold cyan] running splits={splits} (T={module.temperature:.4f})"
        )
        reports = module.evaluate_stage3(
            datamodule,
            splits=splits,
            max_samples=int(max_samples) if max_samples else None,
            batch_size=batch_size,
        )
        self._print_stage3_summary(reports)

        export_cfg = self.cfg.algo.get("export") or {}
        out_dir = Path(str(export_cfg.get("save_dir") or (trainer.default_root_dir if trainer else ".")))
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_file = out_dir / "evaluation_summary.json"
        summary_file.write_text(
            json.dumps(
                {
                    "temperature": module.temperature,
                    "splits": reports,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        console.print(f"[bold green][Jev Stage 3 Summary][/bold green] written to {summary_file}")
        return reports

    @staticmethod
    def _print_stage3_summary(reports: dict[str, Any]) -> None:
        if not reports:
            return
        table = Table(title="Jev Stage 3 验收汇总 (test / test_set_30k / ood)", show_header=True, header_style="bold magenta")
        table.add_column("Split", style="cyan")
        table.add_column("样本数", justify="right")
        table.add_column("Expected Acc", justify="right", style="green")
        table.add_column("NLL", justify="right")
        table.add_column("Brier", justify="right")
        table.add_column("ECE", justify="right")
        table.add_column("ConfErr@0.9", justify="right", style="yellow")
        table.add_column("Escalate@0.8", justify="right")
        table.add_column("吞吐 (rows/s)", justify="right")

        for split_name, metrics in reports.items():
            cov_08 = next(
                (item for item in metrics.get("coverage", []) if abs(float(item["threshold"]) - 0.8) < 1e-6),
                None,
            )
            escalate_str = f"{cov_08['escalation_rate'] * 100:.1f}%" if cov_08 else "-"
            exp_acc = metrics.get("expected_accuracy")
            exp_acc_str = f"{exp_acc * 100:.2f}%" if exp_acc is not None else "-"
            tp = metrics.get("throughput", {})
            tps_str = f"{tp.get('rows_per_second', 0.0):.2f}"
            table.add_row(
                split_name,
                f"{metrics.get('count', 0):,}",
                exp_acc_str,
                f"{metrics.get('nll', 0.0):.4f}",
                f"{metrics.get('brier', 0.0):.4f}",
                f"{metrics.get('multiclass_ece', 0.0):.4f}",
                f"{metrics.get('confident_error_rate', 0.0) * 100:.2f}%",
                escalate_str,
                tps_str,
            )
        console.print(table)

    def save_predictions(self, predictions: list[dict[str, Any]], output_path: str) -> str:
        path = Path(output_path)
        if path.suffix == ".npz":
            path = path.with_suffix(".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        for batch in predictions:
            rows.extend(batch.get("results", []))
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._print_predictions(rows)
        return str(path.resolve())

    @staticmethod
    def _print_predictions(rows: list[dict[str, Any]]) -> None:
        for index, row in enumerate(rows, start=1):
            table = Table(title=f"Jev Decision #{index} (T={row.get('temperature', 1.0):.2f})")
            table.add_column("Question", style="cyan")
            table.add_column("Result", style="green")
            table.add_column("Details", style="white")
            for question_id, answer in row.get("answers", {}).items():
                if answer["type"] == "noul":
                    result = f"{answer['noul']:.2f}"
                    details = "probability"
                elif answer["type"] == "choice":
                    result = str(answer["choice"])
                    details = f"confidence={answer['confidence']:.2f}"
                else:
                    result = f"{answer['score']:.2f}"
                    details = f"confidence={answer['confidence']:.2f}"
                table.add_row(question_id, result, details)
            console.print(table)


__all__ = ["JevDecisionRecipe", "JevStage3EvaluationCallback"]
