"""C51 分布式输出头，支持 Dueling 解耦与 NoisyLinear 参数探索消融开关。"""

from __future__ import annotations

import torch
from torch import nn

from lmml.models.rl.layers import NoisyLinear


def _make_linear(in_dim: int, out_dim: int, use_noisy: bool, std_init: float) -> nn.Module:
    if use_noisy:
        return NoisyLinear(in_dim, out_dim, std_init=std_init)
    return nn.Linear(in_dim, out_dim)


class DistributionalHead(nn.Module):
    """输出 ``[B, action_dim, num_atoms]`` 未归一化 logits。"""

    def __init__(
        self,
        in_dim: int,
        action_dim: int,
        num_atoms: int = 51,
        hidden_dim: int = 64,
        use_dueling: bool = True,
        use_noisy: bool = True,
        std_init: float = 0.5,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.action_dim = int(action_dim)
        self.num_atoms = int(num_atoms)
        self.hidden_dim = int(hidden_dim)
        self.use_dueling = bool(use_dueling)
        self.use_noisy = bool(use_noisy)
        self.std_init = float(std_init)

        if self.use_dueling:
            self.value_stream = nn.Sequential(
                _make_linear(self.in_dim, self.hidden_dim, self.use_noisy, self.std_init),
                nn.ReLU(),
                _make_linear(self.hidden_dim, self.num_atoms, self.use_noisy, self.std_init),
            )
            self.advantage_stream = nn.Sequential(
                _make_linear(self.in_dim, self.hidden_dim, self.use_noisy, self.std_init),
                nn.ReLU(),
                _make_linear(
                    self.hidden_dim, self.action_dim * self.num_atoms, self.use_noisy, self.std_init
                ),
            )
        else:
            self.q_stream = nn.Sequential(
                _make_linear(self.in_dim, self.hidden_dim, self.use_noisy, self.std_init),
                nn.ReLU(),
                _make_linear(
                    self.hidden_dim, self.action_dim * self.num_atoms, self.use_noisy, self.std_init
                ),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        b = features.size(0)
        if self.use_dueling:
            value = self.value_stream(features).view(b, 1, self.num_atoms)
            advantage = self.advantage_stream(features).view(b, self.action_dim, self.num_atoms)
            return value + (advantage - advantage.mean(dim=1, keepdim=True))
        return self.q_stream(features).view(b, self.action_dim, self.num_atoms)
