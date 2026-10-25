"""Q(s, a) 神经网络的抽象基类 QNetwork。

所有不同的 inductive bias（MLP、CNN、Transformer 等）都继承自 QNetwork，
保证训练逻辑（RainbowLightningModule / loss）与数据流完全解耦。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn

from lmml.models.rl.layers import NoisyLinear


class QNetwork(nn.Module, ABC):
    """输出每个动作在支撑原子上的未归一化 logits: ``[B, action_dim, num_atoms]``。"""

    def __init__(self, action_dim: int, num_atoms: int = 51):
        super().__init__()
        self.action_dim = int(action_dim)
        self.num_atoms = int(num_atoms)

    @abstractmethod
    def forward_logits(self, obs: torch.Tensor) -> torch.Tensor:
        """返回形状为 ``[B, action_dim, num_atoms]`` 的原子 logits。"""

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """返回形状为 ``[B, action_dim, num_atoms]`` 的概率分布（softmax）。"""
        return F.softmax(self.forward_logits(obs), dim=-1)

    def log_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """返回形状为 ``[B, action_dim, num_atoms]`` 的对数概率（log_softmax，数值稳定）。"""
        return F.log_softmax(self.forward_logits(obs), dim=-1)

    def q_values(self, obs: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """返回期望 Q 值 ``[B, action_dim]``。"""
        probs = self.forward(obs)
        return (probs * support.view(1, 1, -1)).sum(dim=-1)

    def reset_noise(self) -> None:
        """重置网络内所有 NoisyLinear 层的参数噪声。"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
