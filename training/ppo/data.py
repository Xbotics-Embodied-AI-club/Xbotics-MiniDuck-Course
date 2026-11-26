"""在线策略训练的数据一半：用当前网络在仿真里采样，每轮换一批新数据交给 Lightning。

`trainer.fit(model, data)` 里的 data 就是这里的 `RolloutData`：每轮用最新的网络在仿真里走 `steps` 步，
`prepare` 把这段轨迹换成训练样本，`batches` 切成交给模型的批。PPO 的数据（`ppo.PPOData`）继承它；
选修 S03–S05 课上写的 REINFORCE、A2C、PPO 的数据一半也都继承它，只改 `prepare` 与 `batches`。
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable

import lightning as L
import torch
from lightning.pytorch.utilities.warnings import PossibleUserWarning
from torch.utils.data import DataLoader

from training.ppo.network import ActorCritic
from training.sim.env import DuckEnv

Batch = dict[str, torch.Tensor]

# 数据是仿真现采的、不经过加载进程，Lightning「数据加载器没开多进程」的提醒对这里没有信息
warnings.filterwarnings("ignore", category=PossibleUserWarning, message=".*does not have many workers")

# 每一步记下的量，顺序与 `sample_rollout` 里收集的顺序一致
ROLLOUT_KEYS = ("obs", "critic_obs", "actions", "log_probs", "values", "action_means", "action_stds",
                "rewards", "dones", "time_outs")
# 每轮每个环境走几步（上游 microduck_rl 的 num_steps_per_env；50 Hz 下约 0.5 秒）
STEPS_PER_ITERATION = 24


@torch.no_grad()
def sample_rollout(env: DuckEnv, net: ActorCritic, steps: int,
                   on_step: Callable[[torch.Tensor, torch.Tensor, dict], None] | None = None) -> Batch:
    """用当前网络让每个并行环境接着走 `steps` 步，记下这一段轨迹。

    环境不重置：回合跨轮连续推进，走完的回合由环境自己重开。观测归一化的统计量边走边更新，
    跟着策略实际见到的观测分布走。

    Args:
        env: 并行仿真环境。
        net: 策略与价值网络，须已在仿真所在的设备上。
        steps: 每个环境走几步。
        on_step: 每走一步之后调用一次，传入这一步的原始奖励、结束标记与环境的附加信息。

    Returns:
        `ROLLOUT_KEYS` 里各量，形状是（步数 T，环境数 N，…）：log_probs、action_means、action_stds 是
        采样时那个策略给的，values 是采样时 critic 的估值 (T, N, 1)，time_outs 标出因超时被截断的步；
        另有 next_value：最后一步之后那个状态的估值 (N, 1)，轨迹在回合中间截断时用它接上后面的回报。
    """
    obs, critic_obs = env.get_observations()
    record: dict[str, list[torch.Tensor]] = {key: [] for key in ROLLOUT_KEYS}
    for _ in range(steps):
        actions, log_probs, values, means, stds = net.act(obs, critic_obs)
        next_obs, next_critic_obs, rewards, dones, info = env.step(actions)
        if on_step is not None:
            on_step(rewards, dones, info)
        step = (obs, critic_obs, actions, log_probs, values, means, stds,
                rewards, dones.float(), info["time_outs"].float())
        for key, value in zip(ROLLOUT_KEYS, step, strict=True):
            record[key].append(value)
        net.actor_normalizer.update(next_obs)
        net.critic_normalizer.update(next_critic_obs)
        obs, critic_obs = next_obs, next_critic_obs
    rollout = {key: torch.stack(values) for key, values in record.items()}
    rollout["next_value"] = net.value(critic_obs)
    return rollout


def flat(rollout: Batch, *keys: str) -> Batch:
    """把轨迹里的几个量从（步数，环境数，…）展平成一条一条的样本。"""
    return {key: rollout[key].flatten(0, 1) for key in keys}


class RolloutData(L.LightningDataModule, ABC):
    """data 一半：每轮用当前网络采一段轨迹，换成训练样本，切成批。

    Trainer 要设 `reload_dataloaders_every_n_epochs=1`：Lightning 每轮重建一次数据加载器，
    也就是每轮用最新的网络重新采样 —— 在线策略的数据必须来自当前策略。一轮只采一次：
    采样在 `train_dataloader` 里做，交出去的是现成的批，Lightning 预取或重复遍历都不会再推进仿真。
    子类写 `prepare`，需要时改 `batches`；超参数写成子类的类属性。
    """

    steps = STEPS_PER_ITERATION

    def __init__(self, env: DuckEnv, net: ActorCritic):
        super().__init__()
        self.env, self.net = env, net

    def collect(self) -> Batch:
        """采这一轮的轨迹；采样时要顺带记别的（回合统计、行为克隆的标记）就覆盖它。"""
        return sample_rollout(self.env, self.net, self.steps)

    @abstractmethod
    def prepare(self, rollout: Batch) -> Batch:
        """一段轨迹 → 训练样本：各量展平成一条一条的样本（用 `flat`），算出每条样本的权重。"""

    def batches(self, samples: Batch) -> list[Batch]:
        """训练用的批：默认整段数据当一批、只用一遍。"""
        return [samples]

    def train_dataloader(self) -> DataLoader:
        rollout = self.collect()
        reward = rollout["rewards"].mean()   # 这一轮每步的平均奖励，给进度条看
        return DataLoader([{**batch, "reward": reward} for batch in self.batches(self.prepare(rollout))],
                          batch_size=None)
