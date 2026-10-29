from __future__ import annotations

import sys
import types

from hydra import compose, initialize_config_module
from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar

from lmml.core.callbacks import build_rich_callbacks
from lmml.core.logging import resolve_wandb_api_key, sanitize_config_dict


def test_resolve_wandb_api_key_from_env_and_colab_userdata(monkeypatch):
    # 1. 环境变量优先
    monkeypatch.setenv("WANDB_API_KEY", "  env_test_key_123  ")
    assert resolve_wandb_api_key() == "env_test_key_123"

    # 2. 环境变量缺失时，回退读取 google.colab.userdata.get('WANDB_API_KEY')
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    fake_google = types.ModuleType("google")
    fake_colab = types.ModuleType("google.colab")
    fake_userdata = types.ModuleType("google.colab.userdata")
    fake_userdata.get = lambda name: "colab_secret_xyz" if name == "WANDB_API_KEY" else None
    fake_colab.userdata = fake_userdata
    fake_google.colab = fake_colab

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.colab", fake_colab)
    monkeypatch.setitem(sys.modules, "google.colab.userdata", fake_userdata)

    resolved = resolve_wandb_api_key("WANDB_API_KEY")
    assert resolved == "colab_secret_xyz"

    # 3. 均不存在时返回 None
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    fake_userdata.get = lambda name: None
    assert resolve_wandb_api_key("WANDB_API_KEY") is None


def test_sanitize_config_dict_removes_nulls_and_secrets():
    raw = {
        "algo": {"lr": 1e-4, "api_key": "should_not_appear"},
        "env": None,
        "net": None,
        "logger": {"project": "lmml-projects", "secret": "hidden"},
    }
    cleaned = sanitize_config_dict(raw)
    assert "env" not in cleaned
    assert "net" not in cleaned
    assert "api_key" not in cleaned["algo"]
    assert "secret" not in cleaned["logger"]
    assert cleaned["logger"]["project"] == "lmml-projects"


def test_build_rich_callbacks_respects_trainer_flags():
    with initialize_config_module(version_base="1.3", config_module="lmml.configs"):
        cfg = compose(config_name="config")

    # 开启进度条与模型摘要
    cbs_on = build_rich_callbacks(
        cfg,
        {"enable_progress_bar": True, "enable_model_summary": True},
    )
    pbar = next(cb for cb in cbs_on if isinstance(cb, RichProgressBar))
    summary = next(cb for cb in cbs_on if isinstance(cb, RichModelSummary))
    configured = cfg.callbacks.rich_progress.theme
    assert pbar.theme.description == configured.description
    assert pbar.theme.progress_bar == configured.progress_bar
    assert pbar.theme.progress_bar_finished == configured.progress_bar_finished
    assert summary._max_depth == 2

    # 关闭进度条与模型摘要（如 trainer=debug）时不注入，防止 Lightning MisconfigurationException
    cbs_off = build_rich_callbacks(
        cfg,
        {"enable_progress_bar": False, "enable_model_summary": False},
    )
    assert cbs_off == []
