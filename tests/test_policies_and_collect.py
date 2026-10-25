from __future__ import annotations

import numpy as np

from lmml.envs.collect import collect_offline_data
from lmml.envs.gridworld import GridWorldEnv
from lmml.envs.policies import _make_policy_callable, policy_from_config


def test_four_policy_forms_and_expert_reaches_target():
    env = GridWorldEnv(
        n_rows=4, n_cols=4, start=(0, 0), targets=((3, 3),), traps=((1, 1),), obs_mode="discrete"
    )
    rng = np.random.default_rng(0)

    # 1) 字符串 "uniform" 与 "expert"
    p_uni = _make_policy_callable("uniform", env)
    assert 0 <= p_uni(0, env, rng) < env.n_actions

    p_exp = _make_policy_callable("expert", env)
    assert 0 <= p_exp(0, env, rng) < env.n_actions
    # 0 噪声 expert 应当稳定走到 target
    p_exp_det = policy_from_config({"type": "expert", "noise": 0.0}, env)
    s, _ = env.reset(seed=0)
    for _ in range(20):
        a = p_exp_det(s, env, rng)
        s, _, te, tr, info = env.step(a)
        if te or tr:
            break
    assert te and info["is_success"]

    # 2) callable(s, env, rng) -> int
    p_call = _make_policy_callable(lambda s, e, r: 1, env)
    assert p_call(0, env, rng) == 1

    # 3) 概率矩阵 (n_states, n_actions)
    probs = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    probs[:, 3] = 1.0
    p_mat = _make_policy_callable(probs, env)
    assert p_mat(0, env, rng) == 3

    # 4) ("eps_greedy", q_table, epsilon)
    q_table = np.zeros((env.n_states, env.n_actions), dtype=np.float32)
    q_table[:, 2] = 5.0
    p_eps = _make_policy_callable(("eps_greedy", q_table, 0.0), env)
    assert p_eps(0, env, rng) == 2


def test_collect_offline_data_and_npz(tmp_path):
    env = GridWorldEnv(n_rows=4, n_cols=4, obs_mode="discrete", max_steps=15)
    save_path = tmp_path / "offline.npz"
    data = collect_offline_data(env, 200, "uniform", seed=7, save_path=save_path)
    assert save_path.exists()
    assert data["obs"].shape == (200,) and data["obs"].dtype == np.int32
    assert data["action"].shape == (200,) and data["action"].dtype == np.int8
    assert data["reward"].shape == (200,) and data["reward"].dtype == np.float32
    assert data["next_obs"].shape == (200,) and data["next_obs"].dtype == np.int32
    assert data["terminated"].dtype == bool and data["truncated"].dtype == bool

    loaded = np.load(save_path)
    assert np.array_equal(loaded["obs"], data["obs"])
    assert np.array_equal(loaded["action"], data["action"])
