"""核心任务调度器：分发 train / test / predict / collect 子命令。"""

from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
from hydra.utils import instantiate
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from lmml.core.recipe import Recipe
from lmml.datas.rl.datamodule import make_env_from_cfg
from lmml.datas.rl.offline_io import save_transitions
from lmml.envs.collect import collect_offline_data, summarize
from lmml.envs.policies import policy_from_config


def run_collect(cfg: DictConfig) -> dict:
    env_cfg = OmegaConf.to_container(cfg.env, resolve=True)
    env = make_env_from_cfg(env_cfg)
    try:
        col = cfg.collect
        policy = policy_from_config(col.get("policy", "uniform"), env)
        n_transitions = int(col.n_transitions)
        seed = int(col.get("seed", cfg.seed))
        save_path = col.get("save_path")
        npz_direct = save_path if (save_path and str(save_path).endswith(".npz")) else None
        data = collect_offline_data(env, n_transitions, policy, seed=seed, save_path=npz_direct)
        if save_path and not str(save_path).endswith(".npz"):
            save_transitions(save_path, data)
        stats = summarize(data, env)
        stats["save_path"] = str(Path(save_path).resolve()) if save_path else None
        print(f"[lmml collect] summary: {stats}")
        return stats
    finally:
        env.close()


def _build_logger(cfg: DictConfig) -> TensorBoardLogger:
    logger_cfg = OmegaConf.to_container(cfg.logger, resolve=True)
    if isinstance(logger_cfg, dict) and "_target_" in logger_cfg:
        return instantiate(cfg.logger)
    save_dir = (logger_cfg or {}).get("save_dir", "outputs")
    name = (logger_cfg or {}).get("name", "tb")
    return TensorBoardLogger(save_dir=save_dir, name=name, default_hp_metric=False)


def run(cfg: DictConfig) -> float | dict | None:
    pl.seed_everything(int(cfg.seed), workers=True)
    command = str(cfg.command)
    if command == "collect":
        return run_collect(cfg)
    if command not in ("train", "test", "predict"):
        raise ValueError(f"unsupported command {command!r}; expected train, test, predict, or collect")

    recipe: Recipe = instantiate(cfg.recipe, cfg=cfg, _recursive_=False)
    datamodule = recipe.build_datamodule()
    ckpt_path = cfg.get("ckpt_path")
    module = recipe.build_module(datamodule, ckpt_path=ckpt_path if command in ("test", "predict") else None)
    logger = _build_logger(cfg)
    callbacks = recipe.build_callbacks(datamodule)
    trainer_kwargs = dict(OmegaConf.to_container(cfg.trainer, resolve=True))
    trainer = pl.Trainer(**trainer_kwargs, logger=logger, callbacks=callbacks)

    if command == "train":
        trainer.fit(module, datamodule=datamodule, ckpt_path=cfg.get("resume_from"))
        if bool(cfg.get("test_after_train", False)):
            trainer.test(module, datamodule=datamodule)
        metric_key = cfg.get("optimized_metric", "val/episode_return")
        val = trainer.callback_metrics.get(metric_key)
        return float(val.item()) if val is not None else None

    if command == "test":
        results = trainer.test(module, datamodule=datamodule)
        return results[0] if results else None

    # predict
    preds = trainer.predict(module, datamodule=datamodule)
    pred_cfg = cfg.get("predict") or {}
    out_path = str(pred_cfg.get("output", "predictions.npz"))
    saved = recipe.save_predictions(preds, out_path)
    print(f"[lmml predict] saved predictions to: {saved}")
    return {"output": saved}
