from __future__ import annotations

import numpy as np
import torch

from lmml.core.rl.rainbow.loss import categorical_projection
from lmml.datas.rl.nstep import compute_nstep
from lmml.datas.rl.replay import PrioritizedNStepReplayBuffer
from lmml.datas.rl.sumtree import SumTree


def test_sumtree_priorities_and_batch_leaves():
    tree = SumTree(capacity=8)
    prios = [1.0, 2.0, 3.0, 4.0, 0.5]
    for p in prios:
        tree.add(p)
    assert tree.size == 5
    assert np.isclose(tree.total_priority, sum(prios))

    vals = np.array([0.5, 1.5, 5.5, 10.2])
    _t_idx, _got_p, d_idx = tree.get_leaves(vals)
    assert list(d_idx) == [0, 1, 2, 4]
    for v, di in zip(vals, d_idx):
        _, _, single_di = tree.get_leaf(float(v))
        assert single_di == di


def test_online_and_offline_nstep_match_exactly():
    rng = np.random.default_rng(123)
    n = 40
    obs = rng.integers(0, 25, size=n, dtype=np.int64)
    next_obs = rng.integers(0, 25, size=n, dtype=np.int64)
    action = rng.integers(0, 5, size=n, dtype=np.int64)
    reward = rng.normal(size=n).astype(np.float32)
    terminated = np.zeros(n, dtype=bool)
    truncated = np.zeros(n, dtype=bool)
    terminated[[5, 18, 39]] = True
    truncated[[11, 27]] = True

    buf = PrioritizedNStepReplayBuffer(
        capacity=100, obs_shape=(), obs_dtype=np.int64, n_step=3, gamma=0.9
    )
    for i in range(n):
        buf.push(
            obs[i],
            int(action[i]),
            float(reward[i]),
            next_obs[i],
            bool(terminated[i]),
            bool(truncated[i]),
        )
    buf.finish_episode()

    vec = compute_nstep(obs, action, reward, next_obs, terminated, truncated, n_step=3, gamma=0.9)
    assert len(buf) == n
    assert np.array_equal(buf.obs_buf[:n], vec["obs"])
    assert np.array_equal(buf.action_buf[:n], vec["action"])
    assert np.allclose(buf.reward_buf[:n], vec["reward"], atol=1e-6)
    assert np.array_equal(buf.next_obs_buf[:n], vec["next_obs"])
    assert np.array_equal(buf.term_buf[:n].astype(bool), vec["terminated"])
    assert np.allclose(buf.discount_buf[:n], vec["discount"], atol=1e-6)
    assert not vec["terminated"][11]
    assert np.isclose(vec["discount"][11], 0.9)


def test_categorical_projection_distribution_properties():
    support = torch.linspace(-2.0, 2.0, 51)
    next_p = torch.softmax(torch.randn(8, 51), dim=-1)
    rewards = torch.tensor([[1.0], [-1.0], [0.0], [0.5], [-0.5], [2.5], [-2.5], [0.2]])
    terminated = torch.tensor([[1.0], [1.0], [0.0], [0.0], [0.0], [1.0], [1.0], [0.0]])
    discounts = torch.full((8, 1), 0.99**3)

    m = categorical_projection(
        next_p, rewards, terminated, discounts, support, v_min=-2.0, v_max=2.0
    )
    assert m.shape == (8, 51)
    assert torch.all(m >= -1e-7)
    assert torch.allclose(m.sum(dim=-1), torch.ones(8), atol=1e-5)
    exp_0 = (m[0] * support).sum()
    assert torch.isclose(exp_0, torch.tensor(1.0), atol=1e-4)
