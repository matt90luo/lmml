"""WandB 日志与密钥解析工具：支持环境变量与 Google Colab userdata 自动认证。"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import wandb
from hydra.utils import instantiate
from lightning.pytorch.loggers import CSVLogger, Logger, WandbLogger
from omegaconf import DictConfig, OmegaConf
from rich.console import Console

console = Console()

_SENSITIVE_KEYS = frozenset({"api_key", "wandb_api_key", "token", "secret", "password"})


def resolve_wandb_api_key(secret_name: str = "WANDB_API_KEY") -> str | None:
    """按优先级解析 WandB API Key：环境变量 -> google.colab.userdata -> None。

    绝不在代码或配置文件中硬编码明文密钥。
    """
    for env_key in (secret_name, "WANDB_API_KEY"):
        val = os.environ.get(env_key)
        if val and val.strip():
            cleaned = val.strip()
            os.environ["WANDB_API_KEY"] = cleaned
            return cleaned

    with contextlib.suppress(ImportError, AttributeError, RuntimeError, KeyError, OSError):
        from google.colab import userdata  # type: ignore[import-not-found]

        val = userdata.get(secret_name)
        if val and str(val).strip():
            cleaned = str(val).strip()
            os.environ["WANDB_API_KEY"] = cleaned
            return cleaned

    return None


def sanitize_config_dict(obj: Any) -> Any:
    """清理配置字典中的空顶层组与敏感字段，保持 WandB 面板整洁安全。"""
    if isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for k, v in obj.items():
            if str(k).lower() in _SENSITIVE_KEYS:
                continue
            if v is None:
                continue
            cleaned[k] = sanitize_config_dict(v)
        return cleaned
    if isinstance(obj, list):
        return [sanitize_config_dict(x) for x in obj]
    return obj


def build_logger(cfg: DictConfig) -> Logger:
    """根据 Hydra 配置构建 WandbLogger（默认 project='lmml-projects'）。"""
    raw_logger = cfg.get("logger")
    logger_cfg = OmegaConf.to_container(raw_logger, resolve=True) if raw_logger is not None else {}
    if not isinstance(logger_cfg, dict):
        logger_cfg = {}

    if not bool(logger_cfg.get("enabled", True)):
        save_dir = str(logger_cfg.get("save_dir") or "outputs")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        return CSVLogger(save_dir=save_dir, name=str(logger_cfg.get("name") or "lmml"))

    target = logger_cfg.get("_target_")
    if target and target != "lightning.pytorch.loggers.WandbLogger":
        return instantiate(cfg.logger)

    project = str(logger_cfg.get("project") or "lmml-projects")
    name = logger_cfg.get("name")
    save_dir = str(logger_cfg.get("save_dir") or "outputs")
    offline = bool(logger_cfg.get("offline", False))
    log_model = bool(logger_cfg.get("log_model", False))
    group = logger_cfg.get("group")
    job_type = logger_cfg.get("job_type")
    tags = list(logger_cfg.get("tags") or [])
    entity = logger_cfg.get("entity")
    notes = logger_cfg.get("notes")
    secret_name = str(logger_cfg.get("secret_name") or "WANDB_API_KEY")

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if offline:
        os.environ["WANDB_MODE"] = "offline"
    else:
        if os.environ.get("WANDB_MODE") == "offline":
            os.environ.pop("WANDB_MODE", None)
        api_key = resolve_wandb_api_key(secret_name=secret_name)
        if api_key:
            try:
                wandb.login(key=api_key, relogin=False)
            except (RuntimeError, ValueError, OSError) as exc:
                console.print(f"[yellow][WandB] 自动登录告警: {exc}[/yellow]")
        else:
            console.print(
                "[yellow][WandB] 未检测到 WANDB_API_KEY 环境变量或 Colab userdata Secret；"
                "将尝试使用本地已登录凭证 (~/.netrc)。[/yellow]"
            )

    wb_logger = WandbLogger(
        project=project,
        name=name,
        save_dir=save_dir,
        offline=offline,
        log_model=log_model,
        group=group,
        job_type=job_type,
        tags=tags if tags else None,
        entity=entity,
        notes=notes,
    )

    with contextlib.suppress(RuntimeError, ValueError, TypeError, OSError):
        full_cfg = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(full_cfg, dict):
            wb_logger.log_hyperparams(sanitize_config_dict(full_cfg))

    return wb_logger


def finish_logger() -> None:
    """安全关闭当前活跃的 WandB Run，防止进程退出或连续执行时遗留全局状态。"""
    with contextlib.suppress(RuntimeError, OSError, ValueError):
        if wandb.run is not None:
            wandb.finish()
