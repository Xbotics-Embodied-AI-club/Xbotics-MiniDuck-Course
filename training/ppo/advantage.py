"""优势估计：广义优势估计（GAE）与一批优势的标准化。

PPO 的数据一半（`ppo.PPOData.prepare`）用它；选修 S04 在课上手写一遍 GAE，最后一格和这里逐位比对。
"""
from __future__ import annotations

import torch


def compute_gae(rewards, dones, values, next_value, gamma, lam):
    """用广义优势估计（GAE）把一段 rollout 换算成每步的优势与价值目标。

    从后往前递推，遇到回合结束（done）时截断，免得把下一个回合的回报算进上一个回合。

    Args:
        rewards: 形状 (T, N) 的即时奖励，T 是采样步数，N 是并行环境数。
        dones: 形状 (T, N)，回合在该步结束则为 1。
        values: 形状 (T, N, 1)，critic 对每步状态的估值。
        next_value: 形状 (N, 1)，最后一步之后那个状态的估值。
        gamma: 折扣因子。
        lam: GAE 系数。

    Returns:
        (advantages, returns)。advantages 形状 (T, N)；returns 形状 (T, N, 1)。
    """
    rewards = rewards.unsqueeze(-1)
    dones = dones.unsqueeze(-1)
    advantages = torch.zeros_like(values)
    last_advantage = torch.zeros_like(next_value)
    for step in reversed(range(rewards.shape[0])):
        next_values = next_value if step == rewards.shape[0] - 1 else values[step + 1]
        not_done = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_values * not_done - values[step]
        last_advantage = delta + gamma * lam * not_done * last_advantage
        advantages[step] = last_advantage
    returns = advantages + values
    return advantages.squeeze(-1), returns


def normalize(advantages: torch.Tensor) -> torch.Tensor:
    """减去这一批的平均、除以标准差：减平均是基线，除标准差让权重的尺度不随奖励大小变。

    Args:
        advantages: 任意形状的一批优势（或回报）。

    Returns:
        同形状的标准化结果。
    """
    return (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
