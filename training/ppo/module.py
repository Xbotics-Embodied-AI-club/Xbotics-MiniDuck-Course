"""在线策略训练的模型一半：一批数据的损失，反传与更新交给 Lightning。

`trainer.fit(model, data)` 里的 model 就是这里的 `PolicyModule`：它握着策略与价值网络，
每拿到一批数据算一次损失；优化器、梯度裁剪、进度条上的数由它统一管。PPO（`ppo.PPO`）继承它；
选修 S03–S05 课上写的 REINFORCE、A2C、PPO 的模型一半也都继承它，只写 `loss`。
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import lightning as L
import torch

from training.ppo.network import ActorCritic

Batch = dict[str, torch.Tensor]

# Lightning 每次开训打几行横幅（有没有 GPU、云端记录的广告、Tensor Core 的提示），对训练没有信息：只留警告
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)


class PolicyModule(L.LightningModule, ABC):
    """model 一半：一批数据 → 损失；Adam 更新，actor 与 critic 的梯度分开裁。

    子类写 `loss`；超参数写成子类的类属性。
    """

    learning_rate = 3.0e-4
    max_grad_norm = 1.0

    def __init__(self, net: ActorCritic):
        super().__init__()
        self.net = net

    @abstractmethod
    def loss(self, batch: Batch) -> torch.Tensor:
        """一批数据的损失。"""

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        del batch_idx  # 形参是 Lightning 规定的；数据按顺序用，不需要序号
        loss = self.loss(batch)
        self.log_dict({"reward": batch["reward"], "loss": loss}, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # 交出全部参数：损失里用不到 critic 的算法（REINFORCE），critic 拿不到梯度，Adam 不会动它
        return torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        # actor 与 critic 分开裁到各自的上限：合在一起裁，critic 的大梯度会把 actor 的那份一并压小
        del optimizer, gradient_clip_val, gradient_clip_algorithm
        torch.nn.utils.clip_grad_norm_(self.net.actor_parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.net.critic_parameters(), self.max_grad_norm)
