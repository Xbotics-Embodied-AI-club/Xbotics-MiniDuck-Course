"""PPO：data 一半 `PPOData`、model 一半 `PPO`，`trainer.fit(PPO(net), PPOData(env, net))` 就是一次训练。

超参数与做法照上游 microduck_rl 的 PPO（rsl_rl 5.0.1）：

| 项 | 取值 / 做法 |
|---|---|
| 优势 | GAE，γ 0.99、λ 0.95；整段一起标准化；超时步上补 γ·V(s) |
| 更新 | 5 遍 × 4 个小批；一次随机划分，5 遍都用这同一份划分 |
| 损失 | clip 0.2 的替代目标 + 1.0 × 裁剪的价值损失 − 0.01 × 熵；上游开了对称性的任务另加镜像损失 |
| 学习率 | Adam 1e-3，每个小批按 KL(旧‖新) 调：> 0.02 则 /1.5，< 0.005 则 ×1.5，夹在 [1e-5, 1e-2] |
| 梯度 | actor 与 critic 分开各裁到范数 1.0（`module.PolicyModule`） |

选修 S05 沿用 S03、S04 的 data/model 拆法，在 notebook 里手写 PPO。
正式训练（`train_ppo.py`）在它之上挂存盘、记录、行为克隆与续训。
"""
from __future__ import annotations

import torch
from mjlab_microduck.tasks.symmetry import microduck_vel_symmetry
from tensordict import TensorDict

from training.ppo.advantage import compute_gae, normalize
from training.ppo.data import Batch, RolloutData, flat
from training.ppo.module import PolicyModule
from training.ppo.network import ActorCritic


class PPOData(RolloutData):
    """PPO 的数据：GAE 优势与价值目标，切成 4 个小批、用 5 遍。"""

    gamma = 0.99
    lam = 0.95
    num_learning_epochs = 5
    num_mini_batches = 4

    def prepare(self, rollout: Batch) -> Batch:
        # 超时只是截断，不是终点：把 critic 对那个状态的估值折扣后补进奖励
        rewards = rollout["rewards"] + self.gamma * rollout["values"].squeeze(-1) * rollout["time_outs"]
        advantages, returns = compute_gae(rewards, rollout["dones"], rollout["values"], rollout["next_value"],
                                          self.gamma, self.lam)
        return {**flat(rollout, "obs", "critic_obs", "actions", "log_probs", "values", "action_means",
                       "action_stds"),
                "advantages": normalize(advantages).flatten(), "returns": returns.flatten(0, 1)}

    def batches(self, samples: Batch) -> list[Batch]:
        # 上游的小批生成器在循环外只打乱一次：5 遍用的是同一份划分
        size = len(samples["actions"]) // self.num_mini_batches
        order = torch.randperm(self.num_mini_batches * size, device=samples["obs"].device)
        picks = [order[i * size:(i + 1) * size] for i in range(self.num_mini_batches)]
        return [{key: value[picked] for key, value in samples.items()}
                for _ in range(self.num_learning_epochs) for picked in picks]


def mirror_loss(net: ActorCritic, obs, critic_obs, means):
    """镜像损失：左右镜像后的观测下的动作均值，与原观测下动作均值的镜像之间的均方差。

    与 rsl_rl 5.0.1 `PPO.update` 的 symmetry 分支（不做数据增广、只加镜像损失）一致：目标一侧不回传梯度。

    Args:
        net: 策略与价值网络。
        obs: 一批 actor 观测（原始，未归一化）。
        critic_obs: 同一批的 critic 观测（上游镜像函数要它，不镜像）。
        means: 原观测下的动作均值。

    Returns:
        标量损失。
    """
    n = obs.shape[0]
    mirrored, _ = microduck_vel_symmetry(None, TensorDict({"actor": obs, "critic": critic_obs}, batch_size=[n]), None)
    mirrored_means = net.action_distribution(mirrored["actor"][n:].detach()).mean
    _, target = microduck_vel_symmetry(None, None, means)
    return torch.nn.functional.mse_loss(mirrored_means, target[n:].detach())


class PPO(PolicyModule):
    """PPO 的模型：裁剪的替代目标、裁剪的价值损失、熵；每个小批之后按 KL 调学习率。"""

    learning_rate = 1.0e-3
    clip_param = 0.2
    value_loss_coef = 1.0
    entropy_coef = 0.01
    desired_kl = 0.01

    def __init__(self, net: ActorCritic, mirror_loss_coeff: float | None = None):
        """建模型。

        Args:
            net: 策略与价值网络（与数据一半共用同一个）。
            mirror_loss_coeff: 镜像损失的系数；上游这个任务没开对称性时为 None。
        """
        super().__init__(net)
        self.mirror_loss_coeff = mirror_loss_coeff
        # 关掉时损失里只留价值项、学习率不按 KL 调：续训时 critic 是新建的，先只训它（train_ppo.CriticWarmup）
        self.train_actor = True
        self.kl = torch.zeros(())
        self.lr = self.learning_rate

    def loss(self, batch: Batch) -> torch.Tensor:
        dist = self.net.action_distribution(batch["obs"])
        log_probs = dist.log_prob(batch["actions"]).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        values = self.net.value(batch["critic_obs"])
        with torch.no_grad():
            old = torch.distributions.Normal(batch["action_means"], batch["action_stds"])
            self.kl = torch.distributions.kl_divergence(old, dist).sum(dim=-1).mean()

        ratio = torch.exp(log_probs - batch["log_probs"])
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate = torch.max(-batch["advantages"] * ratio, -batch["advantages"] * clipped_ratio).mean()
        value_clipped = batch["values"] + (values - batch["values"]).clamp(-self.clip_param, self.clip_param)
        value_loss = torch.max((values - batch["returns"]).pow(2), (value_clipped - batch["returns"]).pow(2)).mean()
        loss = (surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy if self.train_actor
                else self.value_loss_coef * value_loss)   # 只训 critic 时 actor 与标准差拿不到梯度，Adam 不动它们
        terms = {"surrogate": surrogate, "value_loss": value_loss, "entropy": entropy, "kl": self.kl}
        if self.mirror_loss_coeff is not None:
            mirror = mirror_loss(self.net, batch["obs"], batch["critic_obs"], dist.mean)
            if self.train_actor:
                loss = loss + self.mirror_loss_coeff * mirror
            terms["symmetry"] = mirror
        # 每轮的平均进 trainer.callback_metrics，正式训练逐轮记下来（train_ppo.IterationEnd）
        self.log_dict({key: value.detach() for key, value in terms.items()}, on_step=False, on_epoch=True,
                      batch_size=1)
        return loss

    def on_before_optimizer_step(self, optimizer):
        # 调的是 Lightning 传进来、真的会被 step 的那个优化器
        kl = float(self.kl) if self.train_actor else self.desired_kl   # actor 不动时 KL 恒为 0，学习率固定在初值
        if kl > self.desired_kl * 2.0:
            self.lr = max(1.0e-5, self.lr / 1.5)
        elif 0.0 < kl < self.desired_kl / 2.0:
            self.lr = min(1.0e-2, self.lr * 1.5)
        for group in optimizer.param_groups:
            group["lr"] = self.lr
