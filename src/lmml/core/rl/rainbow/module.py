"""Rainbow DQN 的 LightningModule 实现，与具体 QNetwork 结构及离线/在线模式解耦。"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
from gymnasium import spaces
from torch import optim

from lmml.core.rl.exploration import EpsilonSchedule
from lmml.core.rl.rainbow.loss import compute_rainbow_loss
from lmml.models.rl.base import QNetwork
from lmml.models.rl.preprocess import space_from_spec
from lmml.models.rl.rainbow_net import RainbowQNetwork


class RainbowLightningModule(pl.LightningModule):
    """Rainbow DQN LightningModule。

    既可通过配置构建 ``RainbowQNetwork``（MLP / CNN / Transformer 编码器），
    也可直接注入任意自定义 ``QNetwork`` 子类实例 ``q_net``。
    """

    def __init__(
        self,
        obs_spec: spaces.Space | Mapping[str, Any] | int | tuple[int, ...] = 4,
        action_dim: int = 2,
        lr: float = 1e-3,
        gamma: float = 0.99,
        n_step: int = 3,
        num_atoms: int = 51,
        v_min: float = -2.0,
        v_max: float = 2.0,
        target_update_freq: int = 100,
        target_update_mode: str = "hard",
        target_tau: float = 0.005,
        use_double: bool = True,
        use_dueling: bool = True,
        use_noisy: bool = True,
        noisy_std: float = 0.5,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 10000,
        env_steps_per_update: int = 1,
        net_cfg: Mapping[str, Any] | None = None,
        q_net: QNetwork | None = None,
        seed: int = 42,
    ):
        super().__init__()
        serializable_obs_spec = space_from_spec(obs_spec)
        if net_cfg is not None and not isinstance(net_cfg, dict):
            from omegaconf import DictConfig, OmegaConf

            if isinstance(net_cfg, DictConfig):
                net_cfg = OmegaConf.to_container(net_cfg, resolve=True)  # type: ignore[assignment]
            else:
                net_cfg = dict(net_cfg)
        obs_spec = serializable_obs_spec
        self.save_hyperparameters(ignore=["q_net"])

        self.obs_spec = serializable_obs_spec
        self.action_dim = int(action_dim)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.n_step = int(n_step)
        self.gamma_n = float(gamma) ** int(n_step)
        self.num_atoms = int(num_atoms)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.target_update_freq = int(target_update_freq)
        self.target_update_mode = str(target_update_mode)
        self.target_tau = float(target_tau)
        self.use_double = bool(use_double)
        self.use_dueling = bool(use_dueling)
        self.use_noisy = bool(use_noisy)
        self.noisy_std = float(noisy_std)
        self.env_steps_per_update = int(env_steps_per_update)
        self.eps_schedule = EpsilonSchedule(eps_start, eps_end, eps_decay_steps)
        self._rng = np.random.default_rng(int(seed))

        self.register_buffer("support", torch.linspace(self.v_min, self.v_max, self.num_atoms))

        if q_net is not None:
            if not isinstance(q_net, QNetwork):
                raise TypeError(f"q_net must be a QNetwork instance, got {type(q_net)}")
            self.online_net: QNetwork = q_net
        else:
            cfg = dict(net_cfg or {})
            encoder_cfg = cfg.get("encoder", cfg.get("arch", "mlp"))
            head_cfg = dict(cfg.get("head") or {})
            head_hidden = int(head_cfg.get("hidden_dim", cfg.get("head_hidden_dim", 64)))
            self.online_net = RainbowQNetwork(
                obs_spec=self.obs_spec,
                action_dim=self.action_dim,
                num_atoms=self.num_atoms,
                encoder=encoder_cfg,
                head_hidden_dim=head_hidden,
                use_dueling=self.use_dueling,
                use_noisy=self.use_noisy,
                std_init=self.noisy_std,
            )

        self.target_net: QNetwork = copy.deepcopy(self.online_net)
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad_(False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.online_net(obs)

    @torch.no_grad()
    def select_action(self, state: Any, greedy: bool = False) -> int:
        if not greedy and not self.use_noisy:
            eps = self.eps_schedule.value(self.global_step)
            if self._rng.random() < eps:
                return int(self._rng.integers(self.action_dim))

        was_training = self.online_net.training
        if greedy:
            self.online_net.eval()
        elif self.use_noisy:
            self.online_net.train()
            self.online_net.reset_noise()

        s_tensor = torch.as_tensor(state, device=self.device)
        q_values = self.online_net.q_values(s_tensor, self.support)
        action = int(q_values.argmax(dim=-1).item())
        if was_training:
            self.online_net.train()
        return action

    def on_train_start(self) -> None:
        dm = getattr(self.trainer, "datamodule", None)
        if dm is not None:
            dm.step_getter = lambda: self.global_step
            if getattr(dm, "collector", None) is not None and dm.is_online_active(self.global_step):
                dm.collector.reset()

    def _unpack_batch(self, batch: Any):
        if isinstance(batch, Mapping):
            states = batch["obs"].to(self.device)
            actions = batch["action"].to(self.device)
            rewards = batch["reward"].to(self.device)
            next_states = batch["next_obs"].to(self.device)
            terminated = batch["terminated"].to(self.device)
            discounts = batch.get("discount")
            if discounts is None:
                discounts = torch.full_like(rewards, self.gamma_n)
            else:
                discounts = discounts.to(self.device)
            is_weights = batch["weights"].to(self.device)
            tree_indices = batch["indices"]
            if isinstance(tree_indices, torch.Tensor):
                tree_indices = tree_indices.cpu().numpy()
            return states, actions, rewards, next_states, terminated, discounts, is_weights, tree_indices

        states, actions, rewards, next_states, dones, is_weights, tree_indices = batch
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        terminated = dones.to(self.device)
        discounts = torch.full_like(rewards, self.gamma_n)
        is_weights = is_weights.to(self.device)
        if isinstance(tree_indices, torch.Tensor):
            tree_indices = tree_indices.cpu().numpy()
        return states, actions, rewards, next_states, terminated, discounts, is_weights, tree_indices

    def _sync_target_network(self) -> None:
        if self.target_update_mode == "soft":
            tau = self.target_tau
            with torch.no_grad():
                for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
                    tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)
                for tb, ob in zip(self.target_net.buffers(), self.online_net.buffers()):
                    tb.data.copy_(ob.data)
        elif self.global_step > 0 and self.global_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        dm = getattr(self.trainer, "datamodule", None)
        online_active = bool(dm is not None and dm.is_online_active(self.global_step))

        # 1. 在线交互步（纯离线模式或 warmup 阶段绝对不调用 env.step()）
        if online_active and getattr(dm, "collector", None) is not None:
            completed = dm.collector.step(
                policy_fn=lambda obs: self.select_action(obs, greedy=False),
                n_steps=self.env_steps_per_update,
            )
            for ep in completed:
                self.log("train/episode_reward", ep.episode_return, prog_bar=True, on_step=True)
                self.log("train/episode_return", ep.episode_return, on_step=True)
                self.log("train/episode_length", float(ep.episode_length), on_step=True)

        # 2. 解包批次并计算 Rainbow 损失
        (
            states,
            actions,
            rewards,
            next_states,
            terminated,
            discounts,
            is_weights,
            tree_indices,
        ) = self._unpack_batch(batch)

        if self.use_noisy:
            self.online_net.reset_noise()
            self.target_net.reset_noise()

        loss, per_sample_ce, q_mean = compute_rainbow_loss(
            online_net=self.online_net,
            target_net=self.target_net,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            terminated=terminated,
            discounts=discounts,
            is_weights=is_weights,
            support=self.support,
            v_min=self.v_min,
            v_max=self.v_max,
            use_double=self.use_double,
        )

        # 3. 更新 PER 优先级与同步目标网络
        if dm is not None and getattr(dm, "buffer", None) is not None:
            priorities = per_sample_ce.cpu().numpy()
            dm.buffer.update_priorities(tree_indices, priorities)
            self.log("train/per_beta", float(dm.buffer.beta), on_step=True)
            self.log("train/buffer_size", float(len(dm.buffer)), on_step=True)

        self._sync_target_network()

        self.log("train/loss", loss, prog_bar=True, on_step=True)
        self.log("train/q_mean", q_mean, on_step=True)
        self.log("train/online_active", 1.0 if online_active else 0.0, on_step=True)
        if not self.use_noisy:
            self.log("train/epsilon", self.eps_schedule.value(self.global_step), on_step=True)
        return loss

    @torch.no_grad()
    def evaluate_episodes(self, env, n_episodes: int = 5, seed: int = 100) -> dict[str, float]:
        returns: list[float] = []
        lengths: list[int] = []
        successes: list[float] = []
        for ep in range(int(n_episodes)):
            obs, _ = env.reset(seed=int(seed) + ep)
            ep_ret = 0.0
            ep_len = 0
            succ = 0.0
            while True:
                action = self.select_action(obs, greedy=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_ret += float(reward)
                ep_len += 1
                if terminated or truncated:
                    if info.get("is_success", False):
                        succ = 1.0
                    break
            returns.append(ep_ret)
            lengths.append(ep_len)
            successes.append(succ)
        return {
            "episode_return": float(np.mean(returns)),
            "episode_length": float(np.mean(lengths)),
            "success_rate": float(np.mean(successes)),
        }

    def test_step(self, batch: Any, batch_idx: int) -> dict[str, float]:
        dm = self.trainer.datamodule
        env = dm.make_eval_env()
        try:
            stats = self.evaluate_episodes(env, n_episodes=1, seed=1000 + int(batch_idx))
        finally:
            env.close()
        self.log("test/episode_return", stats["episode_return"], prog_bar=True, on_epoch=True)
        self.log("test/episode_length", stats["episode_length"], on_epoch=True)
        self.log("test/success_rate", stats["success_rate"], prog_bar=True, on_epoch=True)
        return stats

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> dict[str, np.ndarray]:
        obs = batch.to(self.device)
        self.online_net.eval()
        with torch.no_grad():
            probs = self.online_net(obs)
            q_vals = (probs * self.support.view(1, 1, -1)).sum(dim=-1)
            actions = q_vals.argmax(dim=-1)
        return {
            "obs": obs.cpu().numpy(),
            "q_values": q_vals.cpu().numpy(),
            "action": actions.cpu().numpy(),
            "probs": probs.cpu().numpy(),
        }

    def configure_optimizers(self):
        return optim.Adam(self.online_net.parameters(), lr=self.lr)
