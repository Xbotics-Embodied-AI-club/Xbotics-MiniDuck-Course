"""在线策略算法的公共部分：采样、每轮重新采样的数据、Lightning 训练骨架。

正式训练器 `train_ppo.py` 的 `PPO` 继承这里的 `OnPolicy`；课上（选修 S03–S05）在 notebook 里一讲加一层，
S03 写 `Reinforce(OnPolicy)`，S04 写 `A2C(Reinforce)`，S05 写 `PPO(A2C)`，写完的 PPO 与正式训练器是同一套算法。
一个算法只写两样：`prepare`（一段轨迹 → 训练用的数据）与 `loss`（一批数据 → 损失）；
PPO 另外改 `batches`，把同一段数据切成小批、反复用几遍。采样时要顺带记别的（回合统计、行为克隆的标记）就覆盖 `collect`。
"""
from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator

import lightning as L
import torch
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.utilities.warnings import PossibleUserWarning
from torch.utils.data import DataLoader, IterableDataset

from env import DuckEnv
from model import ActorCritic

Batch = dict[str, torch.Tensor]

# Lightning 每次开训都打几行横幅（有没有 GPU、云端记录的广告、Tensor Core 的提示）和「数据加载器没开多进程」的提醒。
# 这里的数据是仿真现采的、不经过加载进程，这些提示对训练没有信息：调到只报警告，notebook 与训练日志里只留训练本身的输出。
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=PossibleUserWarning, message=".*does not have many workers")

# 每一步记下的量，顺序与 `sample_rollout` 里收集的顺序一致
ROLLOUT_KEYS = ("obs", "critic_obs", "actions", "log_probs", "values", "action_means", "action_stds",
                "rewards", "dones", "time_outs")


@torch.no_grad()
def sample_rollout(env: DuckEnv, model: ActorCritic, steps: int,
                   on_step: Callable[[torch.Tensor, torch.Tensor, dict], None] | None = None) -> Batch:
    """用当前策略让每个并行环境接着走 `steps` 步，记下这一段轨迹。

    环境不重置：回合跨轮连续推进，走完的回合由环境自己重开。观测归一化的统计量边走边更新，
    跟着策略实际见到的观测分布走。

    Args:
        env: 并行仿真环境。
        model: 策略与价值网络。
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
        actions, log_probs, values, means, stds = model.act(obs, critic_obs)
        next_obs, next_critic_obs, rewards, dones, info = env.step(actions)
        if on_step is not None:
            on_step(rewards, dones, info)
        step = (obs, critic_obs, actions, log_probs, values, means, stds,
                rewards, dones.float(), info["time_outs"].float())
        for key, value in zip(ROLLOUT_KEYS, step, strict=True):
            record[key].append(value)
        model.actor_normalizer.update(next_obs)
        model.critic_normalizer.update(next_critic_obs)
        obs, critic_obs = next_obs, next_critic_obs
    rollout = {key: torch.stack(values) for key, values in record.items()}
    rollout["next_value"] = model.value(critic_obs)
    return rollout


def flat(rollout: Batch, *keys: str) -> Batch:
    """把轨迹里的几个量从（步数，环境数，…）展平成一条一条的样本。"""
    return {key: rollout[key].flatten(0, 1) for key in keys}


def normalize(advantages: torch.Tensor) -> torch.Tensor:
    """减去这一批的平均、除以标准差：减平均是基线，除标准差让权重的尺度不随奖励大小变。"""
    return (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)


class RolloutDataset(IterableDataset):
    """一轮的数据：采一段轨迹，交给算法换成训练用的批；每批带上这一轮的平均奖励，给进度条显示。

    Lightning 每轮重建数据加载器之后会把数据集迭代两次（先预取一批探路，再从头训练），
    所以轨迹缓存在这里，一轮只采一次样；每次都采的话，每轮推进两倍的环境步，按环境步分段的课程表也会快一倍。
    """

    def __init__(self, algo: OnPolicy):
        super().__init__()
        self.algo = algo
        self.rollout: Batch | None = None

    def __iter__(self) -> Iterator[Batch]:
        if self.rollout is None:
            self.rollout = self.algo.collect()
        reward = self.rollout["rewards"].mean()
        for batch in self.algo.batches(self.rollout):
            yield {**batch, "reward": reward}


class OnPolicy(L.LightningModule, ABC):
    """在线策略算法的骨架：一轮 = 用当前策略采一段轨迹，再用它更新网络。

    一个 Lightning epoch 就是一轮；数据加载器每轮重建，也就是每轮用最新的策略重新采样。
    子类写 `prepare` 与 `loss`，需要时改 `batches`；超参数写成子类的类属性。
    """

    learning_rate = 3.0e-4
    max_grad_norm = 1.0

    def __init__(self, env: DuckEnv, model: ActorCritic, steps: int = 24):
        super().__init__()
        self.env, self.model, self.steps = env, model, steps

    def collect(self) -> Batch:
        """采这一轮的轨迹。"""
        return sample_rollout(self.env, self.model, self.steps)

    @abstractmethod
    def prepare(self, rollout: Batch) -> Batch:
        """把一段轨迹换成训练用的数据：各量展平成一条一条的样本（用 `flat`）。"""

    @abstractmethod
    def loss(self, batch: Batch) -> torch.Tensor:
        """一批数据的损失。"""

    def batches(self, rollout: Batch) -> Iterator[Batch]:
        """训练用的批：默认整段数据当一批、只用一遍。"""
        yield self.prepare(rollout)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(RolloutDataset(self), batch_size=None)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        del batch_idx  # 形参是 Lightning 规定的；数据按顺序用，不需要序号
        loss = self.loss(batch)
        self.log_dict({"reward": batch["reward"], "loss": loss}, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # 交出全部参数：损失里用不到 critic 的算法（REINFORCE），critic 拿不到梯度，Adam 不会动它
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        # actor 与 critic 分开裁到各自的上限：合在一起裁，critic 的大梯度会把 actor 的那份一并压小
        del optimizer, gradient_clip_val, gradient_clip_algorithm
        torch.nn.utils.clip_grad_norm_(self.model.actor_parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.model.critic_parameters(), self.max_grad_norm)


def build_trainer(iterations: int, accelerator: str = "gpu", progress_bar: bool = True) -> L.Trainer:
    """建 Lightning Trainer：一个 epoch = 一轮，每轮重建数据加载器 = 每轮用最新的策略重新采样。

    Args:
        iterations: 训练几轮。
        accelerator: 训练用 "gpu"；单测用 "cpu"。
        progress_bar: 进度条；输出写进日志文件的长训练关掉它（满屏回车符），改由训练器自己按时间报进度。

    Returns:
        Trainer。
    """
    # 进度条用文本的 tqdm：Lightning 2.6 装了 rich 就默认用 rich 的进度条，它在没装 ipywidgets 的 notebook 里不显示
    return L.Trainer(accelerator=accelerator, devices=1, max_epochs=iterations, reload_dataloaders_every_n_epochs=1,
                     enable_checkpointing=False, logger=False, enable_model_summary=False,
                     enable_progress_bar=progress_bar, callbacks=[TQDMProgressBar()] if progress_bar else [])


def fit(algo: OnPolicy, iterations: int) -> None:
    """训 `iterations` 轮，训完把网络放回仿真所在的设备。

    Args:
        algo: 要训练的算法（`OnPolicy` 的子类）。
        iterations: 训练几轮；一轮 = 采一次样 + 用这批数据更新。
    """
    build_trainer(iterations).fit(algo)
    algo.model.to(algo.env.device)  # Lightning 训完会把模块放回 CPU，评测和下一轮采样都在仿真所在的设备上
