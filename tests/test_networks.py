from __future__ import annotations

import pytest
import torch
from gymnasium import spaces

from lmml.core.rl.rainbow.module import RainbowLightningModule
from lmml.models.rl.base import QNetwork
from lmml.models.rl.rainbow_net import RainbowQNetwork


@pytest.mark.parametrize("arch", ["mlp", "cnn", "transformer"])
@pytest.mark.parametrize("use_dueling", [True, False])
@pytest.mark.parametrize("use_noisy", [True, False])
def test_rainbow_qnetwork_architectures_and_ablations(arch: str, use_dueling: bool, use_noisy: bool):
    obs_space = spaces.Box(0.0, 1.0, (4, 5, 5)) if arch == "cnn" else spaces.Discrete(25)
    net = RainbowQNetwork(
        obs_spec=obs_space,
        action_dim=5,
        num_atoms=51,
        encoder={"arch": arch},
        use_dueling=use_dueling,
        use_noisy=use_noisy,
    )
    if arch == "cnn":
        obs = torch.rand(6, 4, 5, 5)
    else:
        obs = torch.randint(0, 25, (6,))

    probs = net(obs)
    assert probs.shape == (6, 5, 51)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(6, 5), atol=1e-5)

    if use_noisy:
        net.train()
        net.reset_noise()
        p1 = net(obs)
        net.reset_noise()
        p2 = net(obs)
        assert not torch.allclose(p1, p2)

    net.eval()
    assert torch.allclose(net(obs), net(obs))


def test_custom_qnetwork_subclass_plugs_into_lightning_module():
    class CustomLinearQNet(QNetwork):
        def __init__(self, state_dim: int, action_dim: int, num_atoms: int = 51):
            super().__init__(action_dim=action_dim, num_atoms=num_atoms)
            self.fc = torch.nn.Linear(state_dim, action_dim * num_atoms)

        def forward_logits(self, obs: torch.Tensor) -> torch.Tensor:
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            return self.fc(obs.float()).view(-1, self.action_dim, self.num_atoms)

    custom_net = CustomLinearQNet(state_dim=4, action_dim=2, num_atoms=21)
    module = RainbowLightningModule(obs_spec=4, action_dim=2, num_atoms=21, q_net=custom_net)
    act = module.select_action(torch.zeros(4), greedy=True)
    assert act in (0, 1)
