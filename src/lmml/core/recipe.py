"""Recipe 抽象基类：模型 + 数据模块 + 回调的统一组装扩展点。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import lightning.pytorch as pl
from omegaconf import DictConfig


class Recipe(ABC):
    """一个训练/评估/预测任务的构建配方。

    新增任何 RL / LLM / ML / DL 模型时，只需继承 ``Recipe`` 并新增 ``configs/recipe/<name>.yaml``，
    无需改动 CLI 或 runner。
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    @abstractmethod
    def build_datamodule(self) -> pl.LightningDataModule: ...

    @abstractmethod
    def build_module(
        self,
        datamodule: pl.LightningDataModule,
        ckpt_path: str | None = None,
    ) -> pl.LightningModule: ...

    def build_callbacks(self, datamodule: pl.LightningDataModule) -> list[pl.Callback]:
        return []

    def adapt_trainer_kwargs(
        self,
        trainer_kwargs: dict[str, Any],
        datamodule: pl.LightningDataModule,
    ) -> dict[str, Any]:
        """可选钩子：根据数据模块规模自适应修正 Trainer 参数（如小样本下的 val_check_interval）。"""
        return trainer_kwargs

    def on_fit_end(
        self,
        trainer: pl.Trainer,
        module: pl.LightningModule,
        datamodule: pl.LightningDataModule,
    ) -> None:
        """可选钩子：训练结束后的收尾操作（如恢复最优检查点并导出 HuggingFace 权重）。"""
        return

    def save_predictions(self, predictions: list[Any], output_path: str) -> str:
        return output_path
