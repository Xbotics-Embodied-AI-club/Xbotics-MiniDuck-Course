"""PPO 之后的行为克隆：摔倒的帧学站立老师，站着的帧锚定在走路老师上。

照上游 `mjlab_microduck.tasks.distill.PpoWithExpertBc`（走路 + 摔倒恢复用的那套）。
为什么要它：从走路热启动的学生，动作噪声已经很小，躺在地上乱动永远凑不出一次「半起身」，
起身相关的奖励一次都拿不到、没有梯度；而站立老师在同一套 61 维观测上几乎 100% 能爬起来。
所以每轮 PPO 更新之后，在学生**自己**采到的观测上补一遍监督学习：

- 倾斜超过门槛（摔倒）的帧 → 回归到站立老师的动作；老师没见过非零速度指令，喂给它时把速度指令清零；
- 倾斜低于锚定门槛（站着）的帧 → 回归到走路老师的动作，免得克隆把原来的走路改坏
  （没有锚点时，克隆会把走路改坏）；
- 两个门槛之间的帧只归 PPO；站立时带身体姿态指令的帧按上游规则分流（见 `frame_masks`）。

和上游的差别只有一处：老师从哪来。上游写死成他们自己 wandb 上的两次训练，
这里由调用方用 `MINIDUCK_BC_EXPERT` / `MINIDUCK_BC_ANCHOR` 指定我们自己的权重；
门槛、系数、学习率、分流规则都从上游这个任务的配置里读，不另抄一份。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from model import ActorCritic, DeployedActor

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BcSettings:
    """上游 `bc_cfg` 里决定克隆怎么做的那些项（老师的来源除外）。"""

    coef: float
    learning_rate: float
    gate_tilt_deg: float
    epochs: int
    mini_batches: int
    min_mini_batch: int
    min_samples: int
    gravity_slice: tuple[int, int]
    twist_slice: tuple[int, int]
    anchor_coef: float
    anchor_tilt_deg: float
    body_slice: tuple[int, int] | None
    unanchor_turn_in_place: bool
    body_active_axes: tuple[int, ...]
    body_no_teacher_axes: tuple[int, ...]

    @classmethod
    def from_upstream(cls, bc_cfg: dict) -> BcSettings:
        """从上游配置字典取值；缺任何一项都当场 KeyError（上游改了键名要知道）。

        Args:
            bc_cfg: 上游任务配置里的 `algorithm.bc_cfg`。

        Returns:
            克隆设置。
        """
        body = bc_cfg["body_slice"]
        return cls(
            coef=float(bc_cfg["coef"]), learning_rate=float(bc_cfg["learning_rate"]),
            gate_tilt_deg=float(bc_cfg["gate_tilt_deg"]), epochs=int(bc_cfg["epochs"]),
            mini_batches=int(bc_cfg["mini_batches"]), min_mini_batch=int(bc_cfg["min_mini_batch"]),
            min_samples=int(bc_cfg["min_samples"]),
            gravity_slice=tuple(bc_cfg["gravity_slice"]), twist_slice=tuple(bc_cfg["twist_slice"]),
            anchor_coef=float(bc_cfg["anchor_coef"]), anchor_tilt_deg=float(bc_cfg["anchor_tilt_deg"]),
            body_slice=tuple(body) if body is not None else None,
            unanchor_turn_in_place=bool(bc_cfg["unanchor_turn_in_place"]),
            body_active_axes=tuple(bc_cfg["body_active_axes"]),
            body_no_teacher_axes=tuple(bc_cfg["body_no_teacher_axes"]),
        )


def tilted_beyond(obs: torch.Tensor, gravity_slice: tuple[int, int], tilt_deg: float) -> torch.Tensor:
    """机体倾斜是否超过给定角度，从观测里的重力方向读。

    直立时机体系重力是 (0, 0, -1)，倾角的余弦就是 -g_z。观测带噪声与 IMU 安装角随机化，
    先归一化再算。

    Args:
        obs: 一批 actor 观测。
        gravity_slice: 重力方向在观测里的位置。
        tilt_deg: 门槛角度。

    Returns:
        布尔掩码。
    """
    g = obs[:, gravity_slice[0]:gravity_slice[1]]
    g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return -g[:, 2] < math.cos(math.radians(tilt_deg))


def frame_masks(obs: torch.Tensor, s: BcSettings, has_anchor: bool):
    """把一批帧分给站立老师、走路老师，或谁都不给（只归 PPO）。

    指令在观测里不带噪声，所以「速度指令为零」「某个轴为零」是精确判断。

    Args:
        obs: 一批 actor 观测。
        s: 克隆设置。
        has_anchor: 有没有走路老师。

    Returns:
        (给站立老师的帧, 给走路老师的帧, 统计字典)。
    """
    fallen = tilted_beyond(obs, s.gravity_slice, s.gate_tilt_deg)
    upright = (~tilted_beyond(obs, s.gravity_slice, s.anchor_tilt_deg) if has_anchor
               else torch.zeros_like(fallen))
    twist = obs[:, s.twist_slice[0]:s.twist_slice[1]]
    body_teach = torch.zeros_like(fallen)
    if s.body_slice is not None:
        # 站着且有身体姿态指令的帧：走路老师不会响应身体指令，不能拿它锚；
        # 其中只倾转不蹲（z 轴为零）的帧站立老师会做，交给它；要蹲的帧谁都不会，只归 PPO。
        standing = twist.abs().amax(dim=1) <= 1e-6
        body = obs[:, s.body_slice[0]:s.body_slice[1]]
        body_frames = standing & (body[:, list(s.body_active_axes)].abs().amax(dim=1) > 1e-6)
        teachable = body_frames
        if s.body_no_teacher_axes:
            teachable = body_frames & (body[:, list(s.body_no_teacher_axes)].abs().amax(dim=1) <= 1e-6)
        upright = upright & ~body_frames
        body_teach = teachable & ~fallen
    turn_free = torch.zeros_like(fallen)
    if s.unanchor_turn_in_place:
        turn_free = (twist[:, :2].abs().amax(dim=1) <= 1e-6) & (twist[:, 2].abs() > 1e-6)
        upright = upright & ~turn_free
    stats = {"expert_bc_fallen_frac": float(fallen.float().mean()),
             "expert_bc_anchor_frac": float(upright.float().mean()),
             "expert_bc_body_teach_frac": float(body_teach.float().mean()),
             "expert_bc_turn_free_frac": float(turn_free.float().mean())}
    expert = fallen | body_teach
    if int(expert.sum()) < s.min_samples:
        # 这一轮摔倒的帧太少：只做锚定（或什么都不做），不拿几十帧做 20 步回归。
        expert = torch.zeros_like(expert)
    return expert, upright, stats


def load_teacher(path: Path, device) -> DeployedActor:
    """从本仓的 checkpoint 恢复一个冻结的老师（带它自己的观测归一化）。

    Args:
        path: checkpoint 路径。
        device: 放到哪个设备。

    Returns:
        冻结的「归一化 + actor」。
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    settings = ckpt["training_settings"]
    model = ActorCritic(settings["obs_dim"], settings["critic_obs_dim"], settings["action_dim"])
    model.load_state_dict(ckpt["actor_critic"])
    teacher = DeployedActor(model).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


class ExpertBc:
    """每轮 PPO 更新之后跑一遍的行为克隆，自带优化器。

    用独立的 Adam（学习率取上游配置）：PPO 那个优化器的学习率按 KL 自适应、
    动量也和 PPO 共用，克隆步就几乎不起作用。
    """

    def __init__(self, student: ActorCritic, settings: BcSettings, expert: DeployedActor,
                 anchor: DeployedActor | None, max_grad_norm: float):
        self.student = student
        self.s = settings
        self.expert = expert
        self.anchor = anchor
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.Adam(student.actor_parameters(), lr=settings.learning_rate)

    @torch.no_grad()
    def targets(self, obs: torch.Tensor, to_expert: torch.Tensor) -> torch.Tensor:
        """每帧该回归到的动作：站立老师那份把速度指令清零后再喂。

        Args:
            obs: 选中的帧。
            to_expert: 其中哪些归站立老师（其余归走路老师）。

        Returns:
            目标动作。
        """
        target = torch.zeros(obs.shape[0], self.student.action_dim, device=obs.device)
        if to_expert.any():
            expert_obs = obs[to_expert].clone()
            expert_obs[:, self.s.twist_slice[0]:self.s.twist_slice[1]] = 0.0
            target[to_expert] = self.expert(expert_obs)
        if (~to_expert).any():
            if self.anchor is None:
                raise RuntimeError("有帧分给了走路老师，却没有走路老师 —— frame_masks 与配置不一致")
            target[~to_expert] = self.anchor(obs[~to_expert])
        return target

    def update(self, obs: torch.Tensor) -> dict[str, float]:
        """在这一轮学生采到的观测上跑一遍克隆。

        Args:
            obs: 形状 (帧数, 61) 的 actor 观测。

        Returns:
            统计：各类帧占比与平均克隆损失。
        """
        to_expert, to_anchor, stats = frame_masks(obs, self.s, self.anchor is not None)
        idx = (to_expert | to_anchor).nonzero().flatten()
        if idx.numel() == 0:
            stats["expert_bc"] = 0.0
            return stats
        obs, to_expert = obs[idx], to_expert[idx]
        weight = torch.where(to_expert, self.s.coef, self.s.anchor_coef)
        target = self.targets(obs, to_expert)
        n = idx.numel()
        batch = max(self.s.min_mini_batch, n // self.s.mini_batches)
        total, steps = 0.0, 0
        for _ in range(self.s.epochs):
            perm = torch.randperm(n, device=obs.device)
            for start in range(0, n, batch):
                sel = perm[start:start + batch]
                pred = self.student.act_inference(obs[sel])
                loss = (weight[sel] * (pred - target[sel]).pow(2).mean(dim=1)).mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.actor_parameters(), self.max_grad_norm)
                self.optimizer.step()
                total += float(loss.detach())
                steps += 1
        stats["expert_bc"] = total / steps
        return stats
