"""Rainbow C51 Categorical Projection 与 Double DQN 交叉熵损失。"""

from __future__ import annotations

import torch

from lmml.models.rl.base import QNetwork


def categorical_projection(
    next_probs: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    discounts: torch.Tensor,
    support: torch.Tensor,
    v_min: float,
    v_max: float,
) -> torch.Tensor:
    """将目标动作分布 ``next_probs`` [B, N] 经过贝尔曼算子收缩后投影回固定支撑网格 ``support`` [N]。

    返回投影后的目标分布 ``m``，形状 ``[B, N]``，每行概率和为 1。
    """
    batch_size, num_atoms = next_probs.shape
    delta_z = (float(v_max) - float(v_min)) / (num_atoms - 1)

    rewards = rewards.view(batch_size, 1)
    terminated = terminated.view(batch_size, 1)
    discounts = discounts.view(batch_size, 1)

    tz = rewards + (1.0 - terminated) * discounts * support.view(1, -1)
    tz = tz.clamp(float(v_min), float(v_max))

    b = (tz - float(v_min)) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    l[(u > 0) * (l == u)] -= 1
    u[(l < (num_atoms - 1)) * (l == u)] += 1

    m = torch.zeros(batch_size, num_atoms, device=next_probs.device, dtype=next_probs.dtype)
    offset = (
        torch.arange(batch_size, device=next_probs.device, dtype=torch.long).unsqueeze(1) * num_atoms
    ).expand(batch_size, num_atoms)

    m.view(-1).index_add_(
        0,
        (l + offset). reshape(-1),
        (next_probs * (u.to(next_probs.dtype) - b)).reshape(-1),
    )
    m.view(-1).index_add_(
        0,
        (u + offset).reshape(-1),
        (next_probs * (b - l.to(next_probs.dtype))).reshape(-1),
    )
    return m


def compute_rainbow_loss(
    online_net: QNetwork,
    target_net: QNetwork,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    terminated: torch.Tensor,
    discounts: torch.Tensor,
    is_weights: torch.Tensor,
    support: torch.Tensor,
    v_min: float,
    v_max: float,
    use_double: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算 Rainbow 损失。

    返回 ``(loss, per_sample_ce, q_mean)``：
    - ``loss``: 重要性采样加权后的平均交叉熵损失（标量）
    - ``per_sample_ce``: 每个样本的交叉熵损失 ``[B]``（detach，用于更新 SumTree 优先级）
    - ``q_mean``: 当前批次在线网络的平均 Q 值（用于监控）
    """
    batch_size = states.size(0)
    num_atoms = support.numel()

    log_probs_all = online_net.log_probs(states)  # [B, A, N]
    with torch.no_grad():
        q_mean = (log_probs_all.exp() * support.view(1, 1, -1)).sum(dim=-1).mean()

    action_indices = actions.view(batch_size, 1, 1).expand(batch_size, 1, num_atoms)
    log_p = log_probs_all.gather(1, action_indices).squeeze(1)  # [B, N]

    with torch.no_grad():
        next_probs_target = target_net(next_states)  # [B, A, N]
        if use_double:
            next_q_online = online_net.q_values(next_states, support)
            next_best_actions = next_q_online.argmax(dim=1)
        else:
            next_q_target = (next_probs_target * support.view(1, 1, -1)).sum(dim=-1)
            next_best_actions = next_q_target.argmax(dim=1)

        batch_arange = torch.arange(batch_size, device=states.device)
        next_p = next_probs_target[batch_arange, next_best_actions]  # [B, N]

        m = categorical_projection(
            next_probs=next_p,
            rewards=rewards,
            terminated=terminated,
            discounts=discounts,
            support=support,
            v_min=v_min,
            v_max=v_max,
        )

    cross_entropy = -(m * log_p).sum(dim=-1, keepdim=True)  # [B, 1]
    loss = (is_weights.view(batch_size, 1) * cross_entropy).mean()
    return loss, cross_entropy.detach().squeeze(-1), q_mean
