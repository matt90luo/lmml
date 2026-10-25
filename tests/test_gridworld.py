from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import lmml  # noqa: F401
from lmml.envs.gridworld import DOWN, LEFT, RIGHT, STAY, UP, GridWorldEnv


def test_gridworld_movements_and_walls():
    env = GridWorldEnv(
        n_rows=3,
        n_cols=3,
        start=(0, 0),
        targets=((2, 2),),
        traps=((1, 1),),
        walls=((0, 1),),
        step_reward=-0.01,
        wall_penalty=-0.1,
        target_reward=1.0,
        trap_reward=-1.0,
        max_steps=10,
        obs_mode="discrete",
    )
    s, _ = env.reset()
    assert s == 0

    # 向上撞上边界 -> 原地不动，reward = step_reward + wall_penalty
    s, r, te, tr, _ = env.step(UP)
    assert s == 0
    assert np.isclose(r, -0.11)
    assert not te and not tr

    # 向右撞内部墙 (0, 1) -> 原地不动，reward = step_reward + wall_penalty
    s, r, te, tr, _ = env.step(RIGHT)
    assert s == 0
    assert np.isclose(r, -0.11)

    # 向下走到 (1, 0) -> 正常移动，reward = step_reward
    s, r, te, tr, _ = env.step(DOWN)
    assert s == 3
    assert np.isclose(r, -0.01)

    # STAY 原地不动（不撞墙）
    s, r, te, tr, _ = env.step(STAY)
    assert s == 3
    assert np.isclose(r, -0.01)

    # 向右进入 trap (1, 1) -> terminated=True, reward=trap_reward
    s, r, te, tr, info = env.step(RIGHT)
    assert s == 4
    assert te and not tr
    assert np.isclose(r, -1.0)
    assert not info["is_success"]


def test_gridworld_target_and_truncation():
    env = GridWorldEnv(
        n_rows=2,
        n_cols=2,
        start=(0, 0),
        targets=((1, 1),),
        traps=(),
        max_steps=3,
        obs_mode="discrete",
    )
    env.reset()
    _, _, te, tr, _ = env.step(DOWN)  # (1, 0)
    assert not te and not tr
    s, r, te, tr, info = env.step(RIGHT)  # (1, 1) target
    assert s == 3 and te and not tr
    assert np.isclose(r, 1.0) and info["is_success"]

    # 步数超限 -> truncated=True
    env.reset()
    for i in range(3):
        _, _, te, tr, _ = env.step(LEFT)
        assert not te
        assert tr == (i == 2)


@pytest.mark.parametrize("obs_mode", ["discrete", "coords", "image"])
def test_gridworld_obs_modes_and_gym_check(obs_mode: str):
    env = gym.make("lmml/GridWorld-v0", obs_mode=obs_mode, n_rows=4, n_cols=5)
    check_env(env.unwrapped, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    if obs_mode == "image":
        assert obs.shape == (4, 4, 5)
        assert obs[0, 0, 0] == 1.0  # agent channel
    env.close()
