from __future__ import annotations

import numpy as np
from hydra import compose, initialize_config_module

from lmml.cli import main as cli_main
from lmml.core.runner import run


def test_end_to_end_collect_train_modes_test_predict(tmp_path):
    offline_npz = tmp_path / "grid_offline.npz"
    out_dir = tmp_path / "run_out"
    pred_npz = tmp_path / "preds.npz"

    # 1) collect
    cli_main(
        [
            "collect",
            "collect.n_transitions=300",
            "collect.policy=uniform",
            f"collect.save_path={offline_npz}",
            f"hydra.run.dir={out_dir / 'collect'}",
        ]
    )
    assert offline_npz.exists()

    # 2) pure offline train (验证 dm.env is None 且生成 WandB 离线日志与 ckpt)
    with initialize_config_module(version_base="1.3", config_module="lmml.configs"):
        cfg_off = compose(
            config_name="config",
            overrides=[
                "command=train",
                "data=offline",
                f"data.offline_path={offline_npz}",
                "data.steps_per_epoch=10",
                "algo.batch_size=16",
                "trainer=debug",
                "eval.episodes=2",
                "logger.offline=true",
                f"logger.save_dir={out_dir / 'offline'}",
                f"callbacks.checkpoint.dirpath={out_dir / 'offline' / 'ckpts'}",
            ],
        )
    run(cfg_off)
    ckpt_file = out_dir / "offline" / "ckpts" / "last.ckpt"
    assert ckpt_file.exists()
    wandb_runs = list((out_dir / "offline").rglob("offline-run-*"))
    assert len(wandb_runs) >= 1

    # 3) warmup_online (方案 A: 前 8 步离线预热，之后切换在线交互)
    with initialize_config_module(version_base="1.3", config_module="lmml.configs"):
        cfg_warm = compose(
            config_name="config",
            overrides=[
                "command=train",
                "data=warmup_online",
                f"data.offline_path={offline_npz}",
                "data.warmup_offline_steps=8",
                "data.steps_per_epoch=8",
                "algo.batch_size=16",
                "net=transformer",
                "trainer=debug",
                "eval.episodes=2",
                "logger.offline=true",
                f"logger.save_dir={out_dir / 'warmup'}",
            ],
        )
    run(cfg_warm)

    # 4) online + image obs_mode + CNN
    with initialize_config_module(version_base="1.3", config_module="lmml.configs"):
        cfg_cnn = compose(
            config_name="config",
            overrides=[
                "command=train",
                "data=online",
                "data.online_cold_start_steps=32",
                "data.steps_per_epoch=6",
                "algo.batch_size=16",
                "env.kwargs.obs_mode=image",
                "net=cnn",
                "trainer=debug",
                "eval.episodes=2",
                "logger.offline=true",
                f"logger.save_dir={out_dir / 'online_cnn'}",
            ],
        )
    run(cfg_cnn)

    # 5) test from checkpoint
    with initialize_config_module(version_base="1.3", config_module="lmml.configs"):
        cfg_test = compose(
            config_name="config",
            overrides=[
                "command=test",
                f"ckpt_path={ckpt_file}",
                "data.eval_episodes=3",
                "trainer=debug",
                "logger.offline=true",
                f"logger.save_dir={out_dir / 'test'}",
            ],
        )
    test_res = run(cfg_test)
    assert isinstance(test_res, dict) and "test/episode_return" in test_res

    # 6) predict from checkpoint
    with initialize_config_module(version_base="1.3", config_module="lmml.configs"):
        cfg_pred = compose(
            config_name="config",
            overrides=[
                "command=predict",
                f"ckpt_path={ckpt_file}",
                f"predict.output={pred_npz}",
                "trainer=debug",
                "logger.offline=true",
                f"logger.save_dir={out_dir / 'predict'}",
            ],
        )
    run(cfg_pred)
    assert pred_npz.exists()
    pred_data = np.load(pred_npz)
    assert pred_data["q_values"].shape == (25, 5)
    assert pred_data["action"].shape == (25,)
    assert pred_data["probs"].shape == (25, 5, 51)
