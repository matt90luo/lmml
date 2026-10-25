"""统一支持纯离线、纯在线、离线预热转在线（方案 A）的 LightningDataModule。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import gymnasium as gym
import lightning.pytorch as pl
import numpy as np
import torch
from gymnasium import spaces
from torch.utils.data import DataLoader, Dataset, IterableDataset

import lmml  # noqa: F401  确保注册 lmml/GridWorld-v0
from lmml.datas.rl.collector import OnlineCollector
from lmml.datas.rl.nstep import compute_nstep
from lmml.datas.rl.offline_io import load_transitions
from lmml.datas.rl.replay import PrioritizedNStepReplayBuffer
from lmml.envs.policies import _make_policy_callable, policy_from_config

_MODE_ALIASES = {
    "offline": "offline",
    "pure_offline": "offline",
    "online": "online",
    "pure_online": "online",
    "warmup_online": "warmup_online",
    "warmup_then_online": "warmup_online",
}


def make_env_from_cfg(env_cfg: Mapping[str, Any] | str) -> gym.Env:
    if isinstance(env_cfg, str):
        return gym.make(env_cfg)
    cfg = dict(env_cfg)
    env_id = cfg.get("id") or cfg.get("name") or "lmml/GridWorld-v0"
    kwargs = dict(cfg.get("kwargs") or {})
    return gym.make(env_id, **kwargs)


def _obs_shape_dtype(obs_space: spaces.Space) -> tuple[tuple[int, ...], np.dtype]:
    if isinstance(obs_space, spaces.Discrete):
        return (), np.dtype(np.int64)
    return tuple(obs_space.shape), np.dtype(np.float32)


class RLReplayDataset(IterableDataset):
    """通过 IterableDataset 周期性从 ReplayBuffer 中采样批次数据。"""

    def __init__(
        self,
        buffer: PrioritizedNStepReplayBuffer,
        batch_size: int,
        steps_per_epoch: int,
        step_getter: Callable[[], int] | None = None,
    ):
        self.buffer = buffer
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.step_getter = step_getter

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            step = self.step_getter() if self.step_getter is not None else None
            yield self.buffer.sample(self.batch_size, step=step)


class _IndexDataset(Dataset):
    def __init__(self, n: int):
        self.n = int(n)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> int:
        return int(idx)


class _ObsTensorDataset(Dataset):
    def __init__(self, obs: np.ndarray):
        self.obs = np.asarray(obs)

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int):
        return self.obs[idx]


class RLReplayDataModule(pl.LightningDataModule):
    """支持纯离线 (`offline`)、纯在线 (`online`)、离线预热转在线 (`warmup_online`) 的 DataModule。"""

    def __init__(
        self,
        env_cfg: Mapping[str, Any] | str = "lmml/GridWorld-v0",
        mode: str = "online",
        capacity: int = 100000,
        batch_size: int = 64,
        n_step: int = 3,
        gamma: float = 0.99,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_steps: int = 20000,
        use_per: bool = True,
        steps_per_epoch: int = 500,
        offline_path: str | None = None,
        offline_max_samples: int | None = None,
        offline_precomputed_nstep: bool = False,
        warmup_offline_steps: int = 2000,
        online_cold_start_steps: int = 1000,
        cold_start_policy: Any = "uniform",
        eval_episodes: int = 10,
        predict_obs_path: str | None = None,
        predict_batch_size: int = 64,
        seed: int = 42,
    ):
        super().__init__()
        if mode not in _MODE_ALIASES:
            raise ValueError(f"mode must be one of {list(_MODE_ALIASES)}, got {mode!r}")
        self.env_cfg = env_cfg
        self.mode = _MODE_ALIASES[mode]
        self.capacity = int(capacity)
        self.batch_size = int(batch_size)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_steps = int(beta_steps)
        self.use_per = bool(use_per)
        self.steps_per_epoch = int(steps_per_epoch)
        self.offline_path = offline_path
        self.offline_max_samples = offline_max_samples
        self.offline_precomputed_nstep = bool(offline_precomputed_nstep)
        self.warmup_offline_steps = int(warmup_offline_steps)
        self.online_cold_start_steps = int(online_cold_start_steps)
        self.cold_start_policy = cold_start_policy
        self.eval_episodes = int(eval_episodes)
        self.predict_obs_path = predict_obs_path
        self.predict_batch_size = int(predict_batch_size)
        self.seed = int(seed)

        # 探测 observation_space / action_space；纯离线模式下不保留训练 env 实例
        if self.mode == "offline":
            spec_env = make_env_from_cfg(self.env_cfg)
            self.observation_space: spaces.Space = spec_env.observation_space
            self.action_space: spaces.Space = spec_env.action_space
            spec_env.close()
            self.env: gym.Env | None = None
        else:
            self.env = make_env_from_cfg(self.env_cfg)
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space

        obs_shape, obs_dtype = _obs_shape_dtype(self.observation_space)
        self.buffer = PrioritizedNStepReplayBuffer(
            capacity=self.capacity,
            obs_shape=obs_shape,
            obs_dtype=obs_dtype,
            n_step=self.n_step,
            gamma=self.gamma,
            alpha=self.alpha,
            beta_start=self.beta_start,
            beta_steps=self.beta_steps,
            use_per=self.use_per,
            seed=self.seed,
        )
        self.collector: OnlineCollector | None = (
            OnlineCollector(self.env, self.buffer, seed=self.seed) if self.env is not None else None
        )
        self.step_getter: Callable[[], int] | None = None
        self._setup_done = False

    def make_eval_env(self) -> gym.Env:
        """创建独立的评估环境（与训练 env 隔离）。"""
        return make_env_from_cfg(self.env_cfg)

    def is_online_active(self, current_global_step: int) -> bool:
        """判断当前训练步是否需要调用 env.step() 产出实时数据。"""
        if self.mode == "offline":
            return False
        if self.mode == "online":
            return True
        return int(current_global_step) >= self.warmup_offline_steps

    def _load_offline_into_buffer(self) -> None:
        if not self.offline_path:
            raise ValueError(f"mode={self.mode!r} requires data.offline_path to be set")
        raw = load_transitions(self.offline_path)
        if self.offline_max_samples is not None:
            m = int(self.offline_max_samples)
            raw = {k: v[:m] for k, v in raw.items()}
        if self.offline_precomputed_nstep:
            disc = raw.get("discount")
            self.buffer.add_batch(
                raw["obs"],
                raw["action"],
                raw["reward"],
                raw["next_obs"],
                raw["terminated"],
                disc,
            )
        else:
            nstep_data = compute_nstep(
                raw["obs"],
                raw["action"],
                raw["reward"],
                raw["next_obs"],
                raw["terminated"],
                raw["truncated"],
                n_step=self.n_step,
                gamma=self.gamma,
            )
            self.buffer.add_batch(
                nstep_data["obs"],
                nstep_data["action"],
                nstep_data["reward"],
                nstep_data["next_obs"],
                nstep_data["terminated"],
                nstep_data["discount"],
            )

    def _cold_start_online(self) -> None:
        assert self.env is not None
        steps = max(self.online_cold_start_steps, self.batch_size)
        rng = np.random.default_rng(self.seed)
        base = getattr(self.env, "unwrapped", self.env)
        use_custom = (
            self.cold_start_policy != "uniform"
            and hasattr(base, "obs_mode")
            and getattr(base, "obs_mode", None) == "discrete"
        )
        policy_fn = (
            _make_policy_callable(policy_from_config(self.cold_start_policy, self.env), self.env)
            if use_custom
            else None
        )
        state, _ = self.env.reset(seed=self.seed)
        for _ in range(steps):
            if policy_fn is not None:
                action = int(policy_fn(state, self.env, rng))
            else:
                action = int(rng.integers(self.action_space.n))
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            self.buffer.push(state, action, float(reward), next_state, bool(terminated), bool(truncated))
            if terminated or truncated:
                state, _ = self.env.reset()
            else:
                state = next_state
        self.buffer.finish_episode()

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit") and not self._setup_done:
            if self.mode in ("offline", "warmup_online"):
                self._load_offline_into_buffer()
            elif self.mode == "online" and len(self.buffer) < max(self.online_cold_start_steps, self.batch_size):
                self._cold_start_online()
            self._setup_done = True

    def train_dataloader(self) -> DataLoader:
        dataset = RLReplayDataset(
            self.buffer,
            batch_size=self.batch_size,
            steps_per_epoch=self.steps_per_epoch,
            step_getter=self.step_getter,
        )
        return DataLoader(dataset, batch_size=None, num_workers=0)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(_IndexDataset(self.eval_episodes), batch_size=1, num_workers=0)

    def predict_dataloader(self) -> DataLoader:
        if self.predict_obs_path:
            data = np.load(self.predict_obs_path, allow_pickle=False)
            obs = data["obs"] if isinstance(data, np.lib.npyio.NpzFile) else np.asarray(data)
        elif isinstance(self.observation_space, spaces.Discrete):
            obs = np.arange(self.observation_space.n, dtype=np.int64)
        else:
            env = self.make_eval_env()
            samples = []
            s, _ = env.reset(seed=self.seed)
            for _ in range(self.predict_batch_size):
                samples.append(s)
                s, _, te, tr, _ = env.step(env.action_space.sample())
                if te or tr:
                    s, _ = env.reset()
            env.close()
            obs = np.asarray(samples, dtype=np.float32)
        return DataLoader(
            _ObsTensorDataset(obs),
            batch_size=self.predict_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: torch.as_tensor(np.asarray(batch)),
        )

    def teardown(self, stage: str | None = None) -> None:
        if self.env is not None:
            self.env.close()


# 兼容参考代码命名
UnifiedRainbowDataModule = RLReplayDataModule
RainbowDataModule = RLReplayDataModule
