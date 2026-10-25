"""优先经验回放 + n-step 缓冲区（支持在线滑窗、离线批量导入、PER 开关）。"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from lmml.datas.rl.sumtree import SumTree


class PrioritizedNStepReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...] = (),
        obs_dtype: np.dtype | type = np.float32,
        n_step: int = 3,
        gamma: float = 0.99,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_steps: int = 20000,
        epsilon: float = 1e-5,
        prio_clip: float = 100.0,
        use_per: bool = True,
        seed: int | None = None,
    ):
        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self.obs_dtype = np.dtype(obs_dtype)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.alpha = float(alpha) if use_per else 0.0
        self.beta_start = float(beta_start)
        self.beta_steps = max(1, int(beta_steps))
        self.beta = float(beta_start) if use_per else 1.0
        self.epsilon = float(epsilon)
        self.prio_clip = float(prio_clip)
        self.use_per = bool(use_per)
        self.max_priority = 1.0
        self.frame = 0
        self.rng = np.random.default_rng(seed)

        self.tree = SumTree(self.capacity)
        self.obs_buf = np.empty((self.capacity, *self.obs_shape), dtype=self.obs_dtype)
        self.next_obs_buf = np.empty((self.capacity, *self.obs_shape), dtype=self.obs_dtype)
        self.action_buf = np.empty(self.capacity, dtype=np.int64)
        self.reward_buf = np.empty(self.capacity, dtype=np.float32)
        self.term_buf = np.empty(self.capacity, dtype=np.float32)
        self.discount_buf = np.empty(self.capacity, dtype=np.float32)

        self.n_step_buffer: deque[tuple] = deque(maxlen=self.n_step)

    def __len__(self) -> int:
        return self.tree.size

    def _current_leaf_priority(self) -> float:
        return (self.max_priority**self.alpha) if self.use_per else 1.0

    def _store_slot(
        self,
        slot: int,
        obs,
        action: int,
        reward: float,
        next_obs,
        terminated: bool,
        discount: float,
    ) -> None:
        self.obs_buf[slot] = obs
        self.action_buf[slot] = int(action)
        self.reward_buf[slot] = float(reward)
        self.next_obs_buf[slot] = next_obs
        self.term_buf[slot] = 1.0 if bool(terminated) else 0.0
        self.discount_buf[slot] = float(discount)

    def _get_n_step_transition(self):
        state, action = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
        reward = 0.0
        steps_taken = 0
        next_state = self.n_step_buffer[-1][3]
        term = False
        for idx, trans in enumerate(self.n_step_buffer):
            r, next_s, te, tr = trans[2], trans[3], trans[4], trans[5]
            reward += (self.gamma**idx) * float(r)
            steps_taken = idx + 1
            next_state = next_s
            term = bool(te)
            if te or tr:
                break
        discount = self.gamma**steps_taken
        return state, int(action), float(reward), next_state, term, float(discount)

    def push(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        terminated: bool,
        truncated: bool = False,
    ) -> None:
        """在线接收单步转移，经过 n-step 窗口；若 episode 结束（terminated|truncated）自动 flush。"""
        self.n_step_buffer.append((state, int(action), float(reward), next_state, bool(terminated), bool(truncated)))
        if bool(terminated) or bool(truncated):
            self.finish_episode()
        elif len(self.n_step_buffer) == self.n_step:
            s, a, r, s2, te, disc = self._get_n_step_transition()
            slot = self.tree.add(self._current_leaf_priority())
            self._store_slot(slot, s, a, r, s2, te, disc)

    def finish_episode(self) -> None:
        while len(self.n_step_buffer) > 0:
            s, a, r, s2, te, disc = self._get_n_step_transition()
            slot = self.tree.add(self._current_leaf_priority())
            self._store_slot(slot, s, a, r, s2, te, disc)
            self.n_step_buffer.popleft()

    def push_direct(
        self,
        state,
        action: int,
        n_step_reward: float,
        next_state,
        terminated: bool,
        discount: float | None = None,
    ) -> None:
        """直接写入已折现样本（绕过在线滑窗）。"""
        disc = (self.gamma**self.n_step) if discount is None else float(discount)
        slot = self.tree.add(self._current_leaf_priority())
        self._store_slot(slot, state, int(action), float(n_step_reward), next_state, bool(terminated), disc)

    def add_batch(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        terminated: np.ndarray,
        discount: np.ndarray | None = None,
    ) -> int:
        """批量写入已计算好 n-step 的转移并一次性 O(N) 重建 SumTree。返回实际写入条数。"""
        n = min(len(obs), self.capacity)
        if n == 0:
            return 0
        if len(obs) > self.capacity:
            # 保留最后 capacity 条
            obs = obs[-n:]
            action = action[-n:]
            reward = reward[-n:]
            next_obs = next_obs[-n:]
            terminated = terminated[-n:]
            if discount is not None:
                discount = discount[-n:]
        self.obs_buf[:n] = obs
        self.action_buf[:n] = np.asarray(action, dtype=np.int64)
        self.reward_buf[:n] = np.asarray(reward, dtype=np.float32)
        self.next_obs_buf[:n] = next_obs
        self.term_buf[:n] = np.asarray(terminated, dtype=np.float32)
        if discount is None:
            self.discount_buf[:n] = self.gamma**self.n_step
        else:
            self.discount_buf[:n] = np.asarray(discount, dtype=np.float32)
        self.tree.size = n
        self.tree.write_ptr = n % self.capacity
        leaf_prio = self._current_leaf_priority()
        self.tree.set_leaves_bulk(np.arange(n, dtype=np.int64), np.full(n, leaf_prio, dtype=np.float64))
        return n

    def current_beta(self, step: int | None = None) -> float:
        if not self.use_per:
            return 1.0
        s = self.frame if step is None else int(step)
        return min(1.0, self.beta_start + s * (1.0 - self.beta_start) / self.beta_steps)

    def sample(self, batch_size: int, step: int | None = None) -> dict[str, torch.Tensor | np.ndarray]:
        if len(self) == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")
        self.beta = self.current_beta(step)
        self.frame += 1

        if not self.use_per:
            data_idx = self.rng.integers(0, len(self), size=batch_size)
            tree_idx = (data_idx + self.tree.leaf_offset).astype(np.int64)
            weights = np.ones((batch_size, 1), dtype=np.float32)
        else:
            p_total = self.tree.total_priority
            edges = np.linspace(0.0, p_total, batch_size + 1, dtype=np.float64)
            values = self.rng.uniform(edges[:-1], edges[1:])
            tree_idx, prios, data_idx = self.tree.get_leaves(values)
            min_prob = max(self.tree.min_priority() / p_total, 1e-12)
            probs = np.maximum(prios / p_total, 1e-12)
            raw_weights = (len(self) * probs) ** (-self.beta)
            max_weight = (len(self) * min_prob) ** (-self.beta)
            weights = (raw_weights / max_weight).astype(np.float32).reshape(batch_size, 1)

        return {
            "obs": torch.as_tensor(self.obs_buf[data_idx]),
            "action": torch.as_tensor(self.action_buf[data_idx], dtype=torch.int64).unsqueeze(1),
            "reward": torch.as_tensor(self.reward_buf[data_idx], dtype=torch.float32).unsqueeze(1),
            "next_obs": torch.as_tensor(self.next_obs_buf[data_idx]),
            "terminated": torch.as_tensor(self.term_buf[data_idx], dtype=torch.float32).unsqueeze(1),
            "discount": torch.as_tensor(self.discount_buf[data_idx], dtype=torch.float32).unsqueeze(1),
            "weights": torch.as_tensor(weights, dtype=torch.float32),
            "indices": tree_idx,
        }

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        if not self.use_per:
            return
        prios = np.asarray(priorities, dtype=np.float64).reshape(-1)
        clipped = np.minimum(np.abs(prios), self.prio_clip) + self.epsilon
        self.tree.update_batch(np.asarray(indices, dtype=np.int64).reshape(-1), clipped**self.alpha)
        if clipped.size > 0:
            self.max_priority = max(self.max_priority, float(clipped.max()))
