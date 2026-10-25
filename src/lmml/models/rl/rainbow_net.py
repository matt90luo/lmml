"""RainbowQNetwork：将 ObsPreprocessor + Encoder (MLP/CNN/Transformer) + DistributionalHead 组装为 QNetwork。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from gymnasium import spaces

from lmml.models.rl.base import QNetwork
from lmml.models.rl.encoders import CNNEncoder, Encoder, MLPEncoder, TransformerEncoder
from lmml.models.rl.heads import DistributionalHead
from lmml.models.rl.preprocess import ObsPreprocessor

_ARCH_MAP: dict[str, type[Encoder]] = {
    "mlp": MLPEncoder,
    "cnn": CNNEncoder,
    "transformer": TransformerEncoder,
}


def build_encoder(
    in_shape: tuple[int, ...],
    encoder_cfg: Encoder | Callable | Mapping[str, Any] | str | None = None,
) -> Encoder:
    if isinstance(encoder_cfg, Encoder):
        return encoder_cfg
    if callable(encoder_cfg):
        return encoder_cfg(in_shape=in_shape)
    if encoder_cfg is None or isinstance(encoder_cfg, str):
        arch = (encoder_cfg or "mlp").lower()
        if arch not in _ARCH_MAP:
            raise ValueError(f"unknown encoder arch {arch!r}, choose from {list(_ARCH_MAP)}")
        return _ARCH_MAP[arch](in_shape=in_shape)
    cfg = dict(encoder_cfg)
    if "_target_" in cfg:
        from hydra.utils import instantiate

        cfg.pop("_partial_", None)
        return instantiate(cfg, in_shape=in_shape)
    arch = str(cfg.pop("arch", cfg.pop("type", "mlp"))).lower()
    if arch not in _ARCH_MAP:
        raise ValueError(f"unknown encoder arch {arch!r}, choose from {list(_ARCH_MAP)}")
    return _ARCH_MAP[arch](in_shape=in_shape, **cfg)


class RainbowQNetwork(QNetwork):
    """继承自 ``QNetwork`` 的通用 Rainbow 网络，可无缝切换 MLP / CNN / Transformer 特征编码器。"""

    def __init__(
        self,
        obs_spec: spaces.Space | Mapping[str, Any] | int | tuple[int, ...],
        action_dim: int,
        num_atoms: int = 51,
        encoder: Encoder | Callable | Mapping[str, Any] | str | None = None,
        head_hidden_dim: int = 64,
        use_dueling: bool = True,
        use_noisy: bool = True,
        std_init: float = 0.5,
    ):
        super().__init__(action_dim=action_dim, num_atoms=num_atoms)
        self.preprocess = ObsPreprocessor(obs_spec)
        self.encoder = build_encoder(self.preprocess.out_shape, encoder)
        self.head = DistributionalHead(
            in_dim=self.encoder.out_dim,
            action_dim=self.action_dim,
            num_atoms=self.num_atoms,
            hidden_dim=int(head_hidden_dim),
            use_dueling=use_dueling,
            use_noisy=use_noisy,
            std_init=std_init,
        )

    def forward_logits(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(obs)
        features = self.encoder(x)
        return self.head(features)


class RainbowNetwork(RainbowQNetwork):
    """与参考代码签名 ``RainbowNetwork(state_dim, action_dim, num_atoms=51, std_init=0.5)`` 兼容的子类。"""

    def __init__(
        self,
        state_dim: int | tuple[int, ...] | spaces.Space | Mapping[str, Any],
        action_dim: int,
        num_atoms: int = 51,
        std_init: float = 0.5,
        encoder: Encoder | Callable | Mapping[str, Any] | str | None = None,
        head_hidden_dim: int = 64,
        use_dueling: bool = True,
        use_noisy: bool = True,
    ):
        super().__init__(
            obs_spec=state_dim,
            action_dim=action_dim,
            num_atoms=num_atoms,
            encoder=encoder,
            head_hidden_dim=head_hidden_dim,
            use_dueling=use_dueling,
            use_noisy=use_noisy,
            std_init=std_init,
        )
