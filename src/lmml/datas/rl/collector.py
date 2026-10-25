"""在线环境交互采集器，维护当前 episode 状态并向 ReplayBuffer 写入。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym

from lmml.datas.rl.replay import PrioritizedNStepReplayBuffer


@dataclass
class EpisodeStats:
    episode_return: float
    episode_length: int
    is_success: bool = False


class OnlineCollector:
    def __init__(self, env: gym.Env, buffer: PrioritizedNStepReplayBuffer, seed: int = 0):
        self.env = env
        self.buffer = buffer
        self.seed = int(seed)
        self.current_obs = None
        self.episode_return = 0.0
        self.episode_length = 0
        self._initialized = False

    def reset(self) -> None:
        self.current_obs, _ = self.env.reset(seed=self.seed)
        self.episode_return = 0.0
        self.episode_length = 0
        self._initialized = True

    def step(self, policy_fn: Callable, n_steps: int = 1) -> list[EpisodeStats]:
        if not self._initialized:
            self.reset()
        completed: list[EpisodeStats] = []
        for _ in range(int(n_steps)):
            action = int(policy_fn(self.current_obs))
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.buffer.push(
                self.current_obs,
                action,
                float(reward),
                next_obs,
                bool(terminated),
                bool(truncated),
            )
            self.current_obs = next_obs
            self.episode_return += float(reward)
            self.episode_length += 1
            if terminated or truncated:
                completed.append(
                    EpisodeStats(
                        episode_return=self.episode_return,
                        episode_length=self.episode_length,
                        is_success=bool(info.get("is_success", False)),
                    )
                )
                self.episode_return = 0.0
                self.episode_length = 0
                self.current_obs, _ = self.env.reset()
        return completed
