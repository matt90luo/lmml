"""批量采样离线数据。"""

from __future__ import annotations

import numpy as np

from lmml.envs.policies import _make_policy_callable


def collect_offline_data(env, n_transitions, policy, *, seed=0, save_path=None):
    base = getattr(env, "unwrapped", env)
    assert base.obs_mode == "discrete", "采样器按离散状态实现"
    rng = np.random.default_rng(seed)
    n = int(n_transitions)

    obs = np.empty(n, dtype=np.int32)
    act = np.empty(n, dtype=np.int8)
    rew = np.empty(n, dtype=np.float32)
    next_obs = np.empty(n, dtype=np.int32)
    term = np.empty(n, dtype=bool)
    trunc = np.empty(n, dtype=bool)

    call_policy = _make_policy_callable(policy, env)
    s, _ = env.reset(seed=seed)
    i = 0
    while i < n:
        a = int(call_policy(s, env, rng))
        s2, r, te, tr, _ = env.step(a)
        obs[i] = s
        act[i] = a
        rew[i] = r
        next_obs[i] = s2
        term[i] = te
        trunc[i] = tr
        i += 1
        if te or tr:
            s, _ = env.reset()  # 新 episode，不重置 RNG
        else:
            s = s2

    data = {
        "obs": obs,
        "action": act,
        "reward": rew,
        "next_obs": next_obs,
        "terminated": term,
        "truncated": trunc,
    }
    if save_path:
        np.savez_compressed(save_path, **data)
    return data


def summarize(data: dict, env) -> dict:
    """离线数据统计：episode 数、平均回报、target/trap 命中率、动作分布。"""
    base = getattr(env, "unwrapped", env)
    ends = data["terminated"] | data["truncated"]
    n_eps = int(ends.sum())
    ep_id = np.concatenate([[0], np.cumsum(ends)[:-1]])
    returns = np.bincount(ep_id, weights=data["reward"].astype(np.float64))
    target_states = {base.to_state(rc) for rc in base.targets}
    trap_states = {base.to_state(rc) for rc in base.traps}
    term = data["terminated"]
    hit_target = int(np.isin(data["next_obs"][term], list(target_states)).sum())
    hit_trap = int(np.isin(data["next_obs"][term], list(trap_states)).sum())
    return {
        "transitions": len(data["obs"]),
        "episodes": n_eps,
        "mean_return": float(returns[: max(n_eps, 1)].mean()) if len(returns) else 0.0,
        "target_hits": hit_target,
        "trap_hits": hit_trap,
        "action_dist": (
            np.bincount(data["action"], minlength=base.n_actions) / len(data["action"])
        )
        .round(3)
        .tolist(),
    }
