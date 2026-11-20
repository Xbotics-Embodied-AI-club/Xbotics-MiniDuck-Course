"""策略网络与优势估计，结构照上游走路任务的配置。

上游（microduck_rl 的 `MicroduckRlCfg`）对 actor 与 critic 的声明是同一句话：
512-256-128 三层隐藏层、ELU、观测在线归一化；actor 输出高斯分布，标准差是一组
与状态无关的可训练参数（`std_type="scalar"`），初值 1.0。本文件就是这句话的实现，
数值细节（归一化的更新公式与 eps、权重用 PyTorch 默认初始化）与上游 rsl_rl 5.0.1 一致。
"""
from __future__ import annotations

import torch
from torch import nn

HIDDEN_DIMS = (512, 256, 128)
INIT_STD = 1.0
# rsl_rl 的 EmpiricalNormalization 默认值：(x - mean) / (std + eps)。
NORMALIZER_EPS = 1.0e-2


def compute_gae(rewards, dones, values, next_value, gamma, lam):
    """用广义优势估计（GAE）把一段 rollout 换算成每步的优势与回报目标。

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


class RunningNormalizer(nn.Module):
    """观测的在线归一化：边采样边更新均值与方差。

    强化学习的观测分布随策略一起漂移，一次算好的统计量很快就不适用。
    统计量是 buffer 不是参数 —— 它们跟着权重一起存盘、一起导出，但不参与梯度。
    """

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, x):
        """用一批新观测更新统计量（并行算法合并均值与方差）。

        Args:
            x: 形状 (N, dim) 的一批观测。
        """
        batch_count = x.shape[0]
        self.count += batch_count
        rate = batch_count / self.count
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        delta = batch_mean - self.mean
        self.mean += rate * delta
        self.var += rate * (batch_var - self.var + delta * (batch_mean - self.mean))

    def forward(self, x):
        """按当前统计量标准化。

        Args:
            x: 原始观测。

        Returns:
            标准化后的观测。
        """
        return (x - self.mean) / (torch.sqrt(self.var) + NORMALIZER_EPS)


def build_mlp(input_dim, output_dim):
    """堆一串「线性层 + ELU」，最后一层不带激活。

    Args:
        input_dim: 输入维度。
        output_dim: 输出维度。

    Returns:
        拼好的 `nn.Sequential`。
    """
    layers, last = [], input_dim
    for hidden in HIDDEN_DIMS:
        layers += [nn.Linear(last, hidden), nn.ELU()]
        last = hidden
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """actor 输出高斯动作分布，critic 估状态价值，各带一个观测归一化。"""

    def __init__(self, obs_dim, critic_obs_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.actor_normalizer = RunningNormalizer(obs_dim)
        self.critic_normalizer = RunningNormalizer(critic_obs_dim)
        self.actor = build_mlp(obs_dim, action_dim)
        self.critic = build_mlp(critic_obs_dim, 1)
        self.std = nn.Parameter(torch.full((action_dim,), INIT_STD))

    def actor_parameters(self):
        """actor 一侧的全部可训练参数：网络加标准差。

        上游把 actor 和 critic 的梯度**分开**裁剪到各自的范数上限，所以要能分组取。

        Returns:
            参数列表。
        """
        return [*self.actor.parameters(), self.std]

    def critic_parameters(self):
        """critic 一侧的全部可训练参数。

        Returns:
            参数列表。
        """
        return list(self.critic.parameters())

    def action_distribution(self, obs):
        """给出当前状态下逐关节独立的高斯动作分布。

        Args:
            obs: 一批 actor 观测。

        Returns:
            `torch.distributions.Normal`。
        """
        mean = self.actor(self.actor_normalizer(obs))
        return torch.distributions.Normal(mean, self.std.expand_as(mean))

    def value(self, critic_obs):
        """critic 给状态打分。

        Args:
            critic_obs: 一批 critic 观测。

        Returns:
            形状 (N, 1) 的状态价值。
        """
        return self.critic(self.critic_normalizer(critic_obs))

    def act(self, obs, critic_obs):
        """采样阶段用：采一个动作，并带回 PPO 训练要用的「旧策略」各量。

        Args:
            obs: actor 观测。
            critic_obs: critic 观测。

        Returns:
            (动作, 对数概率, 状态价值, 分布均值, 分布标准差)。
        """
        dist = self.action_distribution(obs)
        actions = dist.sample()
        return (actions, dist.log_prob(actions).sum(dim=-1), self.value(critic_obs),
                dist.mean, dist.stddev)

    def act_inference(self, obs):
        """部署 / 评测用：直接取分布均值，不采样。

        Args:
            obs: actor 观测。

        Returns:
            确定性动作。
        """
        return self.actor(self.actor_normalizer(obs))


class DeployedActor(nn.Module):
    """真机上跑的那部分：归一化 + actor 网络，输入原始观测、输出动作。

    导出 ONNX 与 int8 都从它出发，所以「真机跑的」和「训练时看到的」是同一个计算。
    """

    def __init__(self, policy: ActorCritic):
        super().__init__()
        self.normalizer = policy.actor_normalizer
        self.actor = policy.actor

    def forward(self, obs):
        """原始观测 → 动作。

        Args:
            obs: 形状 (N, 61) 的原始 actor 观测。

        Returns:
            形状 (N, 14) 的动作。
        """
        return self.actor(self.normalizer(obs))
