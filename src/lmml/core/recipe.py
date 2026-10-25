"""Recipe 抽象基类：模型 + 数据模块 + 回调的统一组装扩展点。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import lightning.pytorch as pl
from omegaconf import DictConfig


class Recipe(ABC):
    """一个训练/评估/预测任务的构建配方。

    新增任何 RL / ML / DL 模型时，只需继承 ``Recipe`` 并新增 ``configs/recipe/<name>.yaml``，
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

    def save_predictions(self, predictions: list[Any], output_path: str) -> str:
        return output_path
