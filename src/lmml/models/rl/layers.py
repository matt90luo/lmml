"""Factorized Gaussian NoisyLinear 层，用于参数空间探索。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.std_init = float(std_init)

        self.weight_mu = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.weight_sigma = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias_mu = nn.Parameter(torch.empty(self.out_features))
        self.bias_sigma = nn.Parameter(torch.empty(self.out_features))

        self.register_buffer("weight_epsilon", torch.empty(self.out_features, self.in_features))
        self.register_buffer("bias_epsilon", torch.empty(self.out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        x = torch.randn(size, device=device, dtype=dtype)
        return x.sign().mul_(x.abs().sqrt())

    def reset_noise(self) -> None:
        device = self.weight_epsilon.device
        dtype = self.weight_epsilon.dtype
        eps_in = self._scale_noise(self.in_features, device, dtype)
        eps_out = self._scale_noise(self.out_features, device, dtype)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)
