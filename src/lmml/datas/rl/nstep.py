"""离线轨迹向量化 n-step 计算，与在线滑动窗口语义严格一致。"""

from __future__ import annotations

import numpy as np


def compute_nstep(
    obs: np.ndarray,
    action: np.ndarray,
    reward: np.ndarray,
    next_obs: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    n_step: int = 3,
    gamma: float = 0.99,
) -> dict[str, np.ndarray]:
    """将按时间顺序排列的单步转移转化为 n-step 转移。

    对每个下标 i，最多向后看 n_step 步，遇到 ``terminated[j] | truncated[j]`` 或数组末尾即停止（记实际步数为 k）：
    - ``reward_n[i] = sum_{m=0..k-1} gamma^m * reward[i+m]``
    - ``next_obs_n[i] = next_obs[i+k-1]``
    - ``terminated_n[i] = terminated[i+k-1]``（仅当终止而非截断时屏蔽 bootstrap）
    - ``discount[i] = gamma^k``
    """
    n_step = int(n_step)
    if n_step < 1:
        raise ValueError(f"n_step must be >= 1, got {n_step}")
    n = len(obs)
    if n == 0:
        return {
            "obs": obs.copy(),
            "action": action.astype(np.int64, copy=True),
            "reward": reward.astype(np.float32, copy=True),
            "next_obs": next_obs.copy(),
            "terminated": terminated.astype(bool, copy=True),
            "discount": np.empty(0, dtype=np.float32),
        }

    reward_f = np.asarray(reward, dtype=np.float64)
    term_b = np.asarray(terminated, dtype=bool)
    trunc_b = np.asarray(truncated, dtype=bool)
    ep_end = term_b | trunc_b

    out_reward = reward_f.copy()
    out_next_obs = next_obs.copy()
    out_term = term_b.copy()
    alive = ~ep_end  # 第 0 步后是否还能继续往同一 episode 的下一步看
    steps_taken = np.ones(n, dtype=np.int64)

    for m in range(1, n_step):
        valid_len = n - m
        if valid_len <= 0:
            break
        mask = alive[:valid_len]
        if not np.any(mask):
            break
        idx = np.flatnonzero(mask)
        shifted = idx + m
        out_reward[idx] += (gamma**m) * reward_f[shifted]
        out_next_obs[idx] = next_obs[shifted]
        out_term[idx] = term_b[shifted]
        steps_taken[idx] = m + 1
        alive[idx] &= ~ep_end[shifted]

    discount = (gamma ** steps_taken.astype(np.float64)).astype(np.float32)
    return {
        "obs": obs.copy(),
        "action": np.asarray(action, dtype=np.int64),
        "reward": out_reward.astype(np.float32),
        "next_obs": out_next_obs,
        "terminated": out_term,
        "discount": discount,
    }
