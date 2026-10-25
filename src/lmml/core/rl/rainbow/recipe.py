"""RainbowRecipe：将 RLReplayDataModule、RainbowLightningModule 与回调组装起来的 Recipe。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from lmml.core.callbacks import EvalCallback
from lmml.core.recipe import Recipe
from lmml.core.rl.rainbow.module import RainbowLightningModule
from lmml.datas.rl.datamodule import RLReplayDataModule


class RainbowRecipe(Recipe):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def build_datamodule(self) -> RLReplayDataModule:
        cfg = self.cfg
        algo = cfg.algo
        data = cfg.data
        buf = algo.buffer
        env_cfg = OmegaConf.to_container(cfg.env, resolve=True)
        predict_cfg = cfg.get("predict") or {}
        return RLReplayDataModule(
            env_cfg=env_cfg,
            mode=str(data.mode),
            capacity=int(buf.capacity),
            batch_size=int(algo.batch_size),
            n_step=int(algo.n_step),
            gamma=float(algo.gamma),
            alpha=float(buf.alpha),
            beta_start=float(buf.beta_start),
            beta_steps=int(buf.beta_steps),
            use_per=bool(algo.get("use_per", buf.get("prioritized", True))),
            steps_per_epoch=int(data.steps_per_epoch),
            offline_path=data.get("offline_path"),
            offline_max_samples=data.get("offline_max_samples"),
            offline_precomputed_nstep=bool(data.get("offline_precomputed_nstep", False)),
            warmup_offline_steps=int(data.get("warmup_offline_steps", 0)),
            online_cold_start_steps=int(data.get("online_cold_start_steps", 500)),
            cold_start_policy=data.get("cold_start_policy", "uniform"),
            eval_episodes=int(data.get("eval_episodes", 10)),
            predict_obs_path=predict_cfg.get("obs_path"),
            predict_batch_size=int(predict_cfg.get("batch_size", 64)),
            seed=int(cfg.seed),
        )

    def build_module(
        self,
        datamodule: RLReplayDataModule,
        ckpt_path: str | None = None,
    ) -> RainbowLightningModule:
        if ckpt_path:
            return RainbowLightningModule.load_from_checkpoint(ckpt_path, weights_only=False)
        cfg = self.cfg
        algo = cfg.algo
        t_up = algo.target_update
        eps = algo.epsilon
        net_cfg = OmegaConf.to_container(cfg.net, resolve=True)
        return RainbowLightningModule(
            obs_spec=datamodule.observation_space,
            action_dim=int(datamodule.action_space.n),
            lr=float(algo.lr),
            gamma=float(algo.gamma),
            n_step=int(algo.n_step),
            num_atoms=int(algo.num_atoms),
            v_min=float(algo.v_min),
            v_max=float(algo.v_max),
            target_update_freq=int(t_up.every),
            target_update_mode=str(t_up.mode),
            target_tau=float(t_up.tau),
            use_double=bool(algo.use_double),
            use_dueling=bool(algo.use_dueling),
            use_noisy=bool(algo.use_noisy),
            noisy_std=float(algo.noisy_std),
            eps_start=float(eps.start),
            eps_end=float(eps.end),
            eps_decay_steps=int(eps.decay_steps),
            env_steps_per_update=int(algo.env_steps_per_update),
            net_cfg=net_cfg,
            seed=int(cfg.seed),
        )

    def build_callbacks(self, datamodule: RLReplayDataModule) -> list[pl.Callback]:
        cb_cfg = self.cfg.get("callbacks") or {}
        eval_cfg = self.cfg.get("eval") or {}
        eval_enabled = bool(eval_cfg.get("enabled", True))
        callbacks: list[pl.Callback] = [
            EvalCallback(
                enabled=eval_enabled,
                episodes=int(eval_cfg.get("episodes", 5)),
                every_n_epochs=int(eval_cfg.get("every_n_epochs", 1)),
                seed=int(self.cfg.seed) + 500,
            )
        ]
        ckpt_cfg = cb_cfg.get("checkpoint")
        if ckpt_cfg is None or bool(ckpt_cfg.get("enabled", True)):
            monitor = (ckpt_cfg or {}).get("monitor", "val/episode_return" if eval_enabled else "train/loss")
            mode = (ckpt_cfg or {}).get("mode", "max" if eval_enabled else "min")
            callbacks.append(
                ModelCheckpoint(
                    dirpath=(ckpt_cfg or {}).get("dirpath"),
                    filename=(ckpt_cfg or {}).get("filename", "best"),
                    monitor=monitor if eval_enabled else None,
                    mode=mode,
                    save_last=True,
                    save_top_k=1 if eval_enabled else 0,
                )
            )
        if bool((cb_cfg.get("lr_monitor") or {}).get("enabled", True)):
            callbacks.append(LearningRateMonitor(logging_interval="step"))
        return callbacks

    def save_predictions(self, predictions: list[dict[str, Any]], output_path: str) -> str:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        merged = {
            k: np.concatenate([batch[k] for batch in predictions], axis=0)
            for k in ("obs", "q_values", "action", "probs")
        }
        np.savez_compressed(out, **merged)
        return str(out.resolve())
