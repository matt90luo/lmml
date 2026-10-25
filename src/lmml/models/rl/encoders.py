"""不同 inductive bias 的特征编码器（MLP / CNN / Transformer）。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


class Encoder(nn.Module, ABC):
    """输入预处理后的观测张量，输出 ``[B, out_dim]`` 特征向量。"""

    out_dim: int

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class MLPEncoder(Encoder):
    def __init__(
        self,
        in_shape: tuple[int, ...],
        hidden_dims: Sequence[int] = (128,),
        layer_norm: bool = False,
    ):
        super().__init__()
        in_dim = int(np.prod(in_shape))
        layers: list[nn.Module] = [nn.Flatten(start_dim=1)]
        prev = in_dim
        for h in hidden_dims:
            h = int(h)
            layers.append(nn.Linear(prev, h))
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNEncoder(Encoder):
    """针对 (C, H, W) 网格或图像观测的卷积编码器。"""

    def __init__(
        self,
        in_shape: tuple[int, ...],
        channels: Sequence[int] = (32, 64),
        kernel_sizes: Sequence[int] = (3, 3),
        strides: Sequence[int] = (1, 1),
        paddings: Sequence[int] = (1, 1),
        out_dim: int = 128,
    ):
        super().__init__()
        if len(in_shape) == 3:
            c_in, h_in, w_in = (int(x) for x in in_shape)
            self._reshape_3d = None
        else:
            c_in, h_in, w_in = int(np.prod(in_shape)), 1, 1
            self._reshape_3d = (c_in, 1, 1)

        conv_layers: list[nn.Module] = []
        prev_c = c_in
        for c_out, k, s, p in zip(channels, kernel_sizes, strides, paddings):
            k_eff = min(int(k), max(1, h_in), max(1, w_in))
            p_eff = int(p) if k_eff > 1 else 0
            conv_layers.append(
                nn.Conv2d(prev_c, int(c_out), kernel_size=k_eff, stride=int(s), padding=p_eff)
            )
            conv_layers.append(nn.ReLU())
            prev_c = int(c_out)
        self.conv = nn.Sequential(*conv_layers)
        with torch.no_grad():
            dummy = torch.zeros(1, c_in, h_in, w_in)
            flat_dim = int(self.conv(dummy).numel())
        self.proj = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flat_dim, int(out_dim)),
            nn.ReLU(),
        )
        self.out_dim = int(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._reshape_3d is not None:
            x = x.reshape(x.size(0), *self._reshape_3d)
        return self.proj(self.conv(x))


class TransformerEncoder(Encoder):
    """基于自注意力的状态编码器（支持向量 feature token 与 2D 网格 patch token）。"""

    def __init__(
        self,
        in_shape: tuple[int, ...],
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_shape = tuple(int(s) for s in in_shape)
        self.d_model = int(d_model)
        if len(self.in_shape) == 3:
            c, h, w = self.in_shape
            self.is_spatial = True
            self.num_tokens = h * w
            self.patch_proj = nn.Linear(c, self.d_model)
        else:
            d = int(np.prod(self.in_shape))
            self.is_spatial = False
            self.num_tokens = d
            self.feat_weight = nn.Parameter(torch.randn(d, self.d_model) * 0.02)
            self.feat_bias = nn.Parameter(torch.zeros(d, self.d_model))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.num_tokens, self.d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=int(n_layers), enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(self.d_model)
        self.out_dim = self.d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        if self.is_spatial:
            tokens = x.flatten(2).transpose(1, 2)
            tokens = self.patch_proj(tokens)
        else:
            flat = x.reshape(b, self.num_tokens, 1)
            tokens = flat * self.feat_weight.unsqueeze(0) + self.feat_bias.unsqueeze(0)
        cls = self.cls_token.expand(b, -1, -1)
        seq = torch.cat([cls, tokens], dim=1) + self.pos_embed
        out = self.transformer(seq)
        return self.norm(out[:, 0])
