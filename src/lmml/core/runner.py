"""核心任务调度器：分发 data / train / test / predict / collect 子命令。"""

from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelSummary, ProgressBar
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

from lmml.core.callbacks import build_rich_callbacks
from lmml.core.logging import build_logger, finish_logger
from lmml.core.recipe import Recipe
from lmml.datas.llm.prepare import prepare_llm_dataset
from lmml.datas.llm.sft import build_sft_tokenizer
from lmml.datas.rl.datamodule import make_env_from_cfg
from lmml.datas.rl.offline_io import save_transitions
from lmml.envs.collect import collect_offline_data, summarize
from lmml.envs.policies import policy_from_config

console = Console()


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
        console.print(f"[bold green][lmml collect][/bold green] summary: {stats}")
        return stats
    finally:
        env.close()


def run_data(cfg: DictConfig) -> dict:
    data = cfg.data
    model_cfg = (cfg.get("algo") or {}).get("model")
    if model_cfg is None:
        raise ValueError("`lmml data` 需要 LLM 实验配置（缺少 algo.model）。")

    chat_end = str(data.get("chat_end_token", "<|im_end|>"))
    tokenizer_name = str(data.get("tokenizer_name_or_path") or model_cfg.name_or_path)
    tokenizer = build_sft_tokenizer(
        model_name_or_path=tokenizer_name,
        chat_end_token=chat_end,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )
    raw_sources = OmegaConf.to_container(data.sources, resolve=True)
    if not isinstance(raw_sources, list):
        raise TypeError("data.sources 必须为数据源字典列表。")
    prepared_path = data.get("prepared_path")
    if not prepared_path:
        raise ValueError("`lmml data` 必须配置 data.prepared_path。")

    summary = prepare_llm_dataset(
        data_configs=raw_sources,
        tokenizer=tokenizer,
        output_dir=str(prepared_path),
        task=str(data.get("task", "sft")),
        overwrite=bool(data.get("overwrite", True)),
        max_length=int(data.get("max_length", 2048)),
        val_split_ratio=float(data.get("val_split_ratio", 0.02)),
        max_val_samples=data.get("max_val_samples", 1500),
        preprocessing_num_proc=data.get("preprocessing_num_proc", "auto"),
        chat_start_token=str(data.get("chat_start_token", "<|im_start|>")),
        chat_end_token=chat_end,
        ignore_index=int(data.get("ignore_index", -100)),
        drop_no_target=bool(data.get("drop_no_target", True)),
        wrap_reasoning=bool(data.get("wrap_reasoning", False)),
        add_system_ratio=float(data.get("add_system_ratio", 0.0)),
        empty_think_remove_ratio=float(data.get("empty_think_remove_ratio", 0.0)),
        seed=int(cfg.seed),
    )
    console.print(f"[bold green][lmml data][/bold green] summary: {summary}")
    return summary


def _merge_callbacks(
    recipe_callbacks: list[pl.Callback],
    rich_callbacks: list[pl.Callback],
) -> list[pl.Callback]:
    merged = list(recipe_callbacks)
    has_pbar = any(isinstance(cb, ProgressBar) for cb in merged)
    has_summary = any(isinstance(cb, ModelSummary) for cb in merged)
    for cb in rich_callbacks:
        if isinstance(cb, ProgressBar) and has_pbar:
            continue
        if isinstance(cb, ModelSummary) and has_summary:
            continue
        merged.append(cb)
    return merged


def run(cfg: DictConfig) -> float | dict | None:
    pl.seed_everything(int(cfg.seed), workers=True)
    command = str(cfg.command)
    if command == "collect":
        return run_collect(cfg)
    if command == "data":
        return run_data(cfg)
    if command not in ("train", "test", "predict"):
        raise ValueError(f"unsupported command {command!r}; expected data, train, test, predict, or collect")

    recipe: Recipe = instantiate(cfg.recipe, cfg=cfg, _recursive_=False)
    datamodule = recipe.build_datamodule()
    ckpt_path = cfg.get("ckpt_path")
    module = recipe.build_module(
        datamodule,
        ckpt_path=ckpt_path if command in ("test", "predict") else None,
    )

    trainer_kwargs = dict(OmegaConf.to_container(cfg.trainer, resolve=True))
    trainer_kwargs = recipe.adapt_trainer_kwargs(trainer_kwargs, datamodule)
    recipe_callbacks = recipe.build_callbacks(datamodule)
    rich_callbacks = build_rich_callbacks(cfg, trainer_kwargs)
    callbacks = _merge_callbacks(recipe_callbacks, rich_callbacks)

    logger = build_logger(cfg)
    try:
        trainer = pl.Trainer(**trainer_kwargs, logger=logger, callbacks=callbacks)

        if command == "train":
            trainer.fit(module, datamodule=datamodule, ckpt_path=cfg.get("resume_from"))
            recipe.on_fit_end(trainer, module, datamodule)
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
        console.print(f"[bold green][lmml predict][/bold green] saved predictions to: {saved}")
        return {"output": saved}
    finally:
        finish_logger()
