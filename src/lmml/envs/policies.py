"""可插拔的行为策略，统一为 callable(s, env, rng) -> action。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np


def _base(env):
    return getattr(env, "unwrapped", env)


def make_epsilon_greedy_policy(q_table: np.ndarray, epsilon: float) -> Callable:
    q_table = np.asarray(q_table)

    def policy(s, env, rng):
        if rng.random() < epsilon:
            return int(rng.integers(q_table.shape[1]))
        row = q_table[int(s)]
        return int(rng.choice(np.flatnonzero(row == row.max())))

    return policy


def make_expert_policy(env, noise: float = 0.1) -> Callable:
    """BFS 最短路专家 (绕开 trap/wall)，以概率 noise 随机动作。"""
    base = _base(env)
    nxt = base.transition_model()
    trap_states = {base.to_state(rc) for rc in base.traps}
    wall_states = {base.to_state(rc) for rc in base.walls}
    dist = np.full(base.n_states, np.inf)
    for rc in base.targets:
        dist[base.to_state(rc)] = 0.0
    changed = True
    while changed:
        changed = False
        for s in range(base.n_states):
            if s in trap_states or s in wall_states or dist[s] == 0.0:
                continue
            best = min((dist[nxt[s, a]] for a in range(base.n_actions) if nxt[s, a] != s), default=np.inf) + 1
            if best < dist[s]:
                dist[s], changed = best, True
    n_actions = base.n_actions

    def policy(s, env, rng):
        s = int(s)
        if rng.random() < noise:
            return int(rng.integers(n_actions))
        scores = np.array([dist[nxt[s, a]] if nxt[s, a] != s else np.inf for a in range(n_actions)])
        if not np.isfinite(scores.min()):
            return int(rng.integers(n_actions))
        return int(rng.choice(np.flatnonzero(scores == scores.min())))

    return policy


def _make_policy_callable(policy, env) -> Callable:
    """policy 可以是:
    1) 字符串 "uniform" / "expert"
    2) callable(s, env, rng) -> int
    3) 概率矩阵 (n_states, n_actions)，每行和为 1
    4) Q 表 + epsilon 的 tuple ("eps_greedy", q_table, epsilon)
    """
    base = _base(env)
    if callable(policy):
        return policy

    if isinstance(policy, str):
        if policy == "uniform":
            return lambda s, env, rng: int(rng.integers(base.action_space.n))
        if policy == "expert":
            return make_expert_policy(env, noise=0.1)
        raise ValueError(policy)

    if isinstance(policy, tuple) and policy[0] == "eps_greedy":
        _, q_table, epsilon = policy
        return make_epsilon_greedy_policy(q_table, epsilon)

    probs = np.asarray(policy, dtype=np.float64)
    assert probs.shape == (base.n_states, base.n_actions)
    probs = probs / probs.sum(axis=1, keepdims=True)

    def p(s, env, rng):
        return int(rng.choice(base.n_actions, p=probs[int(s)]))

    return p


make_policy_callable = _make_policy_callable


def policy_from_config(cfg: Any, env) -> Any:
    """把 YAML/Hydra 配置转成 `_make_policy_callable` 可接受的 policy。

    - "uniform" | "expert"
    - {type: expert, noise: 0.1}
    - {type: probs, path: x.npy}
    - {type: eps_greedy, q_path: q.npy, epsilon: 0.1}
    - {_target_: pkg.mod.factory, ...}  -> 工厂返回 policy (callable/矩阵/...)
    """
    if cfg is None:
        return "uniform"
    if not isinstance(cfg, (str, Mapping)):
        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        else:
            return cfg
    if isinstance(cfg, str):
        return cfg
    cfg = dict(cfg)
    if "_target_" in cfg:
        from hydra.utils import instantiate

        return instantiate(cfg)
    kind = cfg.get("type", "uniform")
    if kind in ("uniform",):
        return "uniform"
    if kind == "expert":
        return make_expert_policy(env, noise=float(cfg.get("noise", 0.1)))
    if kind == "probs":
        return np.load(cfg["path"])
    if kind == "eps_greedy":
        return ("eps_greedy", np.load(cfg["q_path"]), float(cfg.get("epsilon", 0.1)))
    raise ValueError(f"unknown policy type {kind!r}")
