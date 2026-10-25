"""训练回调：独立评估环境的贪心策略周期性评估回调 EvalCallback。"""

from __future__ import annotations

import lightning.pytorch as pl


class EvalCallback(pl.Callback):
    """在每个训练 epoch 结束时使用独立 eval_env 评估贪心策略并写入 Logger (TensorBoard)。"""

    def __init__(self, enabled: bool = True, episodes: int = 5, every_n_epochs: int = 1, seed: int = 777):
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
