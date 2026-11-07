"""Hydra Recipe for Kev decision-model training, testing and prediction."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table

from lmml.core.llm.kev.checkpoint import load_kev_model, save_kev_checkpoint
from lmml.core.llm.kev.data import KevDataModule
from lmml.core.llm.kev.model import MAX_BRANCH, MAX_STATE
from lmml.core.llm.kev.module import KevLightningModule
from lmml.core.recipe import Recipe
from lmml.datas.llm.sft import build_sft_tokenizer

console = Console()


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


DEFAULT_DEMO_REQUEST = {
    "state": {
        "user_id": "U-9402",
        "text": "你们的扣费系统是不是有毛病？上周多扣了我两次费用，马上退款，否则投诉消协！",
        "history_disputes": 0,
    },
    "questions": {
        "asks_for_refund": {
            "type": "noul",
            "instructions": "Does the customer explicitly demand a refund of money?",
        },
        "routing_queue": {
            "type": "choice",
            "instructions": "Which department should handle this request?",
            "criteria": {
                "billing": "Issues with recurring payments, charges, cards, or receipts",
                "account_security": "Compromised accounts, password resets, 2FA failures",
                "technical_support": "App crashes, 5xx errors, API bugs",
                "general_inquiry": "Other non-urgent questions",
            },
        },
        "customer_frustration": {
            "type": "score",
            "instructions": "Rate the level of user emotional frustration and hostility.",
            "criteria": [
                "Level 0: Calm, objective, neutral tone",
                "Level 1: Annoyed, mildly impatient but polite",
                "Level 2: Openly aggressive, angry, threatening legal or external escalations",
            ],
        },
    },
}


class KevDecisionRecipe(Recipe):
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

    def build_datamodule(self) -> KevDataModule:
        data = self.cfg.data
        algo = self.cfg.algo
        prepared_path = data.get("prepared_path")
        if not prepared_path:
            raise ValueError("Kev training requires data.prepared_path; run `lmml data` first")
        tokenizer = self._tokenizer()
        datamodule = KevDataModule(
            str(prepared_path),
            tokenizer,
            batch_size=int(data.get("batch_size", 1)),
            val_batch_size=int(data.get("val_batch_size", data.get("batch_size", 1))),
            num_workers=int(data.get("num_workers", 0)),
            max_state=int(data.get("max_state", MAX_STATE)),
            max_branch=int(data.get("max_branch", MAX_BRANCH)),
            seed=int(self.cfg.seed),
            p_none=float(algo.get("p_none", 0.1)),
            p_none_distract=float(algo.get("p_none_distract", 0.12)),
            p_distract=float(algo.get("p_distract", 0.15)),
            p_none_pair=float(algo.get("p_none_pair", 0.25)),
            predict_requests=self._predict_requests(),
        )
        datamodule.tokenizer = tokenizer
        return datamodule

    def build_module(
        self,
        datamodule: pl.LightningDataModule,
        ckpt_path: str | None = None,
    ) -> KevLightningModule:
        cfg = self.cfg
        algo = cfg.algo
        model_cfg = algo.model
        module = KevLightningModule(
            str(model_cfg.name_or_path),
            datamodule.tokenizer,
            mode=str(algo.get("mode", "lora")),
            learning_rate=float(algo.get("lr", 1e-4)),
            head_learning_rate=algo.get("head_lr"),
            weight_decay=float(algo.get("weight_decay", 0.0)),
            warmup_ratio=float(algo.get("warmup_ratio", 0.1)),
            torch_dtype=str(model_cfg.get("torch_dtype", "bfloat16")),
            attn_implementation=str(model_cfg.get("attn_implementation", "sdpa")),
            gradient_checkpointing=bool(model_cfg.get("use_gradient_checkpointing", False)),
            lora_r=int(algo.get("lora_r", 16)),
            lora_alpha=int(algo.get("lora_alpha", 32)),
            lora_dropout=float(algo.get("lora_dropout", 0.05)),
            lora_target_modules=algo.get("lora_target_modules"),
            pointer_dim=int(algo.get("pointer_dim", 256)),
            option_isolation=bool(algo.get("option_isolation", False)),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
            ord_weight=float(algo.get("ord_weight", 0.1)),
            label_smoothing=float(algo.get("label_smoothing", 0.0)),
            init_from=algo.get("init_from"),
        )
        if ckpt_path:
            path = Path(ckpt_path)
            if path.is_dir() and (path / "head.pt").exists():
                loaded, _ = load_kev_model(
                    path,
                    device="cpu",
                    torch_dtype=model_cfg.get("torch_dtype", "float32"),
                    base_model=str(model_cfg.name_or_path),
                )
                module.model = loaded
            elif path.is_file():
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                module.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
        return module

    def build_callbacks(self, datamodule: pl.LightningDataModule) -> list[pl.Callback]:
        callbacks: list[pl.Callback] = []
        checkpoint = (self.cfg.get("callbacks") or {}).get("checkpoint") or {}
        if bool(checkpoint.get("enabled", True)):
            callbacks.append(
                ModelCheckpoint(
                    dirpath=checkpoint.get("dirpath"),
                    filename=str(checkpoint.get("filename", "kev-{epoch:02d}-{step:06d}")),
                    monitor=str(checkpoint.get("monitor", "val/loss")),
                    mode=str(checkpoint.get("mode", "min")),
                    save_top_k=int(checkpoint.get("save_top_k", 1)),
                    save_last=bool(checkpoint.get("save_last", True)),
                    auto_insert_metric_name=False,
                )
            )
        if bool(((self.cfg.get("callbacks") or {}).get("lr_monitor") or {}).get("enabled", True)):
            callbacks.append(LearningRateMonitor(logging_interval="step"))
        return callbacks

    def on_fit_end(
        self,
        trainer: pl.Trainer,
        module: pl.LightningModule,
        datamodule: pl.LightningDataModule,
    ) -> None:
        assert isinstance(module, KevLightningModule)
        assert isinstance(datamodule, KevDataModule)
        calibration = self.cfg.algo.get("calibration") or {}
        temperature = 1.0
        max_samples = int(calibration.get("max_samples", 4000))
        batch_size = int(calibration.get("batch_size", 1))
        calibration_error: Exception | None = None
        calibration_meta: dict[str, Any] = {
            "status": "disabled",
            "temperature": 1.0,
            "max_samples": max_samples,
            "batch_size": batch_size,
        }
        if bool(calibration.get("enabled", True)):
            _release_cuda_memory()
            console.print(
                "[bold cyan][Kev calibration][/bold cyan] "
                f"fitting on up to {max_samples} records with batch_size={batch_size}"
            )
            try:
                temperature = module.calibrate(
                    datamodule,
                    max_samples=max_samples,
                    batch_size=batch_size,
                )
                calibration_meta = dict(module.calibration_stats)
                console.print(
                    "[bold green][Kev calibration][/bold green] "
                    f"temperature={temperature:.4f} "
                    f"records={calibration_meta.get('accepted_records', 0)} "
                    f"skipped={calibration_meta.get('skipped_records', 0)} "
                    f"batch_size={batch_size}"
                )
            except Exception as exc:  # noqa: BLE001 - preserve the trained checkpoint on any calibration failure
                calibration_error = exc
                module.model.head.temperature = 1.0
                calibration_meta = {
                    "status": "failed",
                    "temperature": 1.0,
                    "max_samples": max_samples,
                    "batch_size": batch_size,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                console.print(
                    "[bold red][Kev calibration failed][/bold red] "
                    f"{type(exc).__name__}: {exc}"
                )
                console.print(
                    "[yellow][Kev export][/yellow] calibration failed; "
                    "exporting the trained model with temperature=1.0"
                )
        export = self.cfg.algo.get("export") or {}
        save_dir = export.get("save_dir")
        if bool(export.get("enabled", True)) and save_dir:
            module.cpu()
            _release_cuda_memory()
            algo = self.cfg.algo
            try:
                saved = save_kev_checkpoint(
                    module.model,
                    module.tokenizer,
                    str(save_dir),
                    base_model=str(algo.model.name_or_path),
                    mode=str(algo.get("mode", "lora")),
                    lora_r=int(algo.get("lora_r", 0)),
                    lora_alpha=int(algo.get("lora_alpha", 0)),
                    pointer_dim=int(algo.get("pointer_dim", 256)),
                    option_isolation=bool(algo.get("option_isolation", False)),
                    temperature=temperature,
                    extra={
                        "init_source": module.init_source,
                        "calibration": calibration_meta,
                        "args": OmegaConf.to_container(self.cfg, resolve=True),
                    },
                )
            except Exception as export_error:
                if calibration_error is not None:
                    raise export_error from calibration_error
                raise
            console.print(f"[bold green][Kev export][/bold green] {saved}")
        else:
            module.cpu()
            _release_cuda_memory()
        if calibration_error is not None:
            raise calibration_error

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
            table = Table(title=f"Kev decision #{index}")
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


__all__ = ["DEFAULT_DEMO_REQUEST", "KevDecisionRecipe"]
