"""训练回调：独立评估环境的 EvalCallback 以及全局 RichProgressBar / RichModelSummary 构建器。"""

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from omegaconf import DictConfig, OmegaConf

DEFAULT_RICH_THEME: dict[str, str] = {
    "description": "green_yellow",
    "progress_bar": "cyan",
    "progress_bar_finished": "green",
    "progress_bar_pulse": "magenta",
    "batch_progress": "white",
    "time": "grey54",
    "processing_speed": "bright_yellow",
    "metrics": "yellow",
}


class EvalCallback(pl.Callback):
    """在每个训练 epoch 结束时使用独立 eval_env 评估贪心策略并写入 Logger (WandB)。"""

    def __init__(
        self,
        enabled: bool = True,
        episodes: int = 5,
        every_n_epochs: int = 1,
        seed: int = 777,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.episodes = int(episodes)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.seed = int(seed)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.enabled:
            return
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        dm = getattr(trainer, "datamodule", None)
        if dm is None or not hasattr(dm, "make_eval_env") or not hasattr(pl_module, "evaluate_episodes"):
            return
        env = dm.make_eval_env()
        try:
            stats = pl_module.evaluate_episodes(
                env, n_episodes=self.episodes, seed=self.seed + trainer.current_epoch * 100
            )
        finally:
            env.close()
        pl_module.log("val/episode_return", stats["episode_return"], prog_bar=True, on_epoch=True)
        pl_module.log("val/episode_length", stats["episode_length"], on_epoch=True)
        pl_module.log("val/success_rate", stats["success_rate"], prog_bar=True, on_epoch=True)


def build_rich_callbacks(
    cfg: DictConfig,
    trainer_kwargs: dict[str, Any] | None = None,
) -> list[pl.Callback]:
    """根据 callbacks 与 trainer 配置构建 RichProgressBar 和 RichModelSummary。"""
    raw_cb = cfg.get("callbacks")
    cb_cfg = OmegaConf.to_container(raw_cb, resolve=True) if raw_cb is not None else {}
    if not isinstance(cb_cfg, dict):
        cb_cfg = {}

    t_kwargs = trainer_kwargs or {}
    enable_pbar = bool(t_kwargs.get("enable_progress_bar", True))
    enable_summary = bool(t_kwargs.get("enable_model_summary", True))

    rich_callbacks: list[pl.Callback] = []

    pbar_cfg = cb_cfg.get("rich_progress")
    if not isinstance(pbar_cfg, dict):
        pbar_cfg = {}
    if enable_pbar and bool(pbar_cfg.get("enabled", True)):
        theme_cfg = dict(DEFAULT_RICH_THEME)
        if isinstance(pbar_cfg.get("theme"), dict):
            theme_cfg.update(pbar_cfg["theme"])
        theme = RichProgressBarTheme(**theme_cfg)
        leave = bool(pbar_cfg.get("leave", True))
        rich_callbacks.append(RichProgressBar(theme=theme, leave=leave))

    summary_cfg = cb_cfg.get("rich_summary")
    if not isinstance(summary_cfg, dict):
        summary_cfg = {}
    if enable_summary and bool(summary_cfg.get("enabled", True)):
        max_depth = int(summary_cfg.get("max_depth", 2))
        rich_callbacks.append(RichModelSummary(max_depth=max_depth))

    return rich_callbacks
