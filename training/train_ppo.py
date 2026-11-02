"""PPO 训练小鸭子的各个动作。

训哪个动作由 `MINIDUCK_SKILL` 选，在哪个硬件版本上训由 `MINIDUCK_HARDWARE` 选（`<日期>-<舵机>`，
见 sim_model/hardware/），这一次叫什么由 `MINIDUCK_RUN_NAME` 给；三者都必须显式给出。

结构是一条 Lightning 链：在线采样的 `IterableDataset` → `LightningDataModule` →
算 loss 的 `LightningModule` → `trainer.fit`。

**训练方法与上游一致**，「上游」指 microduck_rl 用 `uv run train Mjlab-Velocity-Flat-MicroDuck`
跑的那条路（mjlab 1.3.0 的 train 脚本 + rsl_rl 5.0.1 的 PPO）。逐项对齐的清单：

| 项 | 取值 / 做法 |
|---|---|
| 每轮采样 | 每个环境 24 步 |
| 优势 | GAE，γ 0.99、λ 0.95；整段 rollout 一起标准化 |
| 超时 | 无限时域任务在超时步上补 γ·V(s) |
| 更新 | 5 遍 × 4 个小批；**一次随机划分**，5 遍都用这同一份划分 |
| 损失 | clip 0.2 的替代目标 + 1.0 × 裁剪的价值损失 − 0.01 × 熵 |
| 学习率 | Adam 1e-3，每个小批按 KL(旧‖新) 自适应：> 0.02 则 /1.5，< 0.005 则 ×1.5，夹在 [1e-5, 1e-2] |
| 梯度 | actor 与 critic **分开**各裁到范数 1.0 |
| 开训 | 各环境的回合进度随机打散；TF32 打开；种子 42 同时给环境与 torch |

左右对称增广在上游这几个任务里都是关的，这里也不做；
若哪天上游打开了它，`check_upstream_cfg` 会当场报错，而不是只执行一半的配方。

走路 + 摔倒恢复（`velstand-*`）在上游还多两件，这里照做：

- 在别的策略上**热启动**（`MINIDUCK_WARM_START` 指定源权重）：继承权重、观测归一化与
  优化器状态，迭代号与课程表计数从 0 起；
- 每轮 PPO 更新之后跑一遍**行为克隆**（`distill.py`；站立老师与走路老师由
  `MINIDUCK_BC_EXPERT` / `MINIDUCK_BC_ANCHOR` 指定，其余设置读上游配置）。

**从已发布的策略续训**（`MINIDUCK_INIT_POLICY` 指一个 rsl_rl 导出的 ONNX，
`MINIDUCK_INIT_ITERATION` 给它在源训练里的迭代号）：上游的站起配方从零训学不会翻身，
已发布的 alpha_stand 只有 ONNX，没有 checkpoint。做法是同任务续训，和上游 resume 一致的部分照做：
actor 与观测归一化取自 ONNX，课程表放在源迭代号对应的那一档；ONNX 里没有的三样这样补 ——
critic 从头训，前 CRITIC_WARMUP_ITERS 轮只训 critic、actor 不动、学习率不做 KL 自适应；
动作标准差从 FINETUNE_INIT_STD 起；归一化计数取源训练看过的样本量。

逐轮的指标写进权重目录的 `metrics.jsonl` 与 W&B（离线）；屏幕上只有就位、存盘、
结束各一行，外加至少隔 10 秒一行的进度摘要。
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path

import lightning as L
import torch
import wandb
from mjlab.tasks.registry import load_rl_cfg
from mjlab_microduck.tasks.distill import PpoWithExpertBcCfg
from torch.utils.data import DataLoader, IterableDataset

import hardware
from distill import BcSettings, ExpertBc, load_teacher
from env import NUM_ENVS, OUR_COLLISION_GEOMS, SEED, DuckEnv, Skill, max_iterations, skill_from_env
from export_tools.from_onnx import load_actor_from_onnx
from logs import run_cli
from model import ActorCritic, compute_gae
from paths import checkpoint_dir, file_sha256, run_name

NUM_STEPS_PER_ENV = 24
SAVE_INTERVAL = 250
# 从 ONNX 续训时补 ONNX 里没有的部分。critic 是随机初始化的，它给的优势在头几十轮是噪声，
# 直接拿来更新 actor 会把已学会的动作推散；先只训 critic。100 轮 × 20 个小批 = 2000 步梯度。
CRITIC_WARMUP_ITERS = 100
# 动作标准差 ONNX 里没有（部署取均值）。本仓从零训的站起与走路都收敛到 0.18–0.21，
# 取 0.2；用初值 1.0（弧度）会让采样动作把已学会的动作抖散。
FINETUNE_INIT_STD = 0.2
# 屏幕进度摘要的最短间隔。Python 的 logging 没有自带限频，只能在这里按墙钟时间判。
SUMMARY_INTERVAL_S = 10.0

log = logging.getLogger(__name__)


def check_upstream_cfg(skill: Skill) -> dict:
    """把上游这个任务的 PPO 配置读出来，核对它与本训练器写死的数字一致。

    数字写在本文件里是为了让读的人一眼看到；但它们的真源是上游的配置。
    上游改了某个超参而这里没跟上，训练照跑、结果却不再是「原版的训练方法」——
    所以开训前对一遍，对不上就停。本仓用到的几个任务，这些数字上游都一样。

    Args:
        skill: 动作声明。

    Returns:
        上游的 PPO 配置（字典），一并存进 checkpoint 备查。

    Raises:
        RuntimeError: 任何一项与本训练器不一致，或上游打开了对称性。
    """
    cfg = load_rl_cfg(skill.task_id)
    algo = cfg.algorithm
    expect = {
        "num_steps_per_env": (cfg.num_steps_per_env, NUM_STEPS_PER_ENV),
        "save_interval": (cfg.save_interval, SAVE_INTERVAL),
        "actor.hidden_dims": (tuple(cfg.actor.hidden_dims), (512, 256, 128)),
        "critic.hidden_dims": (tuple(cfg.critic.hidden_dims), (512, 256, 128)),
        "actor.activation": (cfg.actor.activation, "elu"),
        "actor.obs_normalization": (cfg.actor.obs_normalization, True),
        "critic.obs_normalization": (cfg.critic.obs_normalization, True),
        "actor.init_std": (cfg.actor.distribution_cfg["init_std"], 1.0),
        "actor.std_type": (cfg.actor.distribution_cfg["std_type"], "scalar"),
        "clip_actions": (cfg.clip_actions, None),
        "value_loss_coef": (algo.value_loss_coef, PPO.value_loss_coef),
        "use_clipped_value_loss": (algo.use_clipped_value_loss, True),
        "clip_param": (algo.clip_param, PPO.clip_param),
        "entropy_coef": (algo.entropy_coef, PPO.entropy_coef),
        "num_learning_epochs": (algo.num_learning_epochs, PPO.num_learning_epochs),
        "num_mini_batches": (algo.num_mini_batches, PPO.num_mini_batches),
        "learning_rate": (algo.learning_rate, PPO.learning_rate),
        "schedule": (algo.schedule, "adaptive"),
        "gamma": (algo.gamma, PPO.gamma),
        "lam": (algo.lam, PPO.lam),
        "desired_kl": (algo.desired_kl, PPO.desired_kl),
        "max_grad_norm": (algo.max_grad_norm, PPO.max_grad_norm),
        "normalize_advantage_per_mini_batch": (algo.normalize_advantage_per_mini_batch, False),
        # 走路任务用的是带对称性字段的配置类；直接取属性，字段改名时当场报错。
        "symmetry_cfg": (algo.symmetry_cfg, None),
        "seed": (cfg.seed, SEED),
    }
    wrong = {k: v for k, v in expect.items() if v[0] != v[1]}
    if wrong:
        raise RuntimeError(f"上游配置与本训练器不一致（上游值, 本训练器值）：{wrong}")
    return {k: v[0] for k, v in expect.items()}


def train_seed() -> int:
    """这一次训练的随机种子：默认是上游 runner 配置的那个值（`SEED`），换种子复现时用 `MINIDUCK_SEED` 改。

    上游配置的核对（`check_upstream_cfg`）照旧拿 `SEED` 比，换种子不影响它。

    Returns:
        种子。
    """
    return int(os.environ.get("MINIDUCK_SEED") or SEED)


def required_path(var: str) -> Path:
    """从环境变量取一个必须存在的文件路径。

    Args:
        var: 环境变量名。

    Returns:
        路径。

    Raises:
        RuntimeError: 没给，或文件不存在。
    """
    value = os.environ.get(var, "").strip()
    if not value or not Path(value).is_file():
        raise RuntimeError(f"{var}={value!r} 不是一个存在的 checkpoint 文件")
    return Path(value)


def build_expert_bc(skill: Skill, model: ActorCritic, device):
    """上游这个任务要行为克隆时，按上游的设置、用我们自己的老师权重建好克隆器。

    Args:
        skill: 动作声明。
        model: 学生。
        device: 设备。

    Returns:
        (克隆器或 None, 写进训练设置的说明字典)。
    """
    algo = load_rl_cfg(skill.task_id).algorithm
    if not isinstance(algo, PpoWithExpertBcCfg):
        return None, None
    settings = BcSettings.from_upstream(algo.bc_cfg)
    expert_path = required_path("MINIDUCK_BC_EXPERT")
    anchor_path = required_path("MINIDUCK_BC_ANCHOR") if settings.anchor_coef > 0 else None
    bc = ExpertBc(model, settings, load_teacher(expert_path, device),
                  load_teacher(anchor_path, device) if anchor_path else None, PPO.max_grad_norm)
    log.info("行为克隆：倾斜 > %.0f° 的帧学站立老师 %s，< %.0f° 的帧锚定走路老师 %s",
             settings.gate_tilt_deg, expert_path.parent.name + "/" + expert_path.name,
             settings.anchor_tilt_deg,
             anchor_path.parent.name + "/" + anchor_path.name if anchor_path else "（无）")
    return bc, {"settings": dataclasses.asdict(settings), "expert": str(expert_path),
                "anchor": str(anchor_path) if anchor_path else None}


def enable_tf32() -> None:
    """矩阵乘法与 cuDNN 用 TF32，cuDNN 自动选最快的算法 —— 与上游训练入口相同的三项设置。

    上游调的是 mjlab 的 `configure_torch_backends()`，它在 torch 2.9 上走新接口
    （`fp32_precision = "tf32"`）；而 Lightning 启动时用旧接口读这个设置，
    torch 对「新旧接口混用」直接抛 RuntimeError。所以这里用旧接口设同样的值。
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def upstream_source() -> dict:
    """已安装的上游包来自哪个 commit（uv 从 git 装时写下的 direct_url.json）。

    Returns:
        direct_url.json 的内容。

    Raises:
        RuntimeError: 包不是从 git 装的（没有 direct_url.json），说不清它是哪个版本。
    """
    text = importlib.metadata.distribution("mjlab-microduck").read_text("direct_url.json")
    if text is None:
        raise RuntimeError("mjlab-microduck 不是从 git 钉版装的，说不清上游版本 —— 用 `uv sync` 重装")
    return json.loads(text)


class IterationLog:
    """一轮迭代里结束的那些回合：各自的总奖励与步数，以及 mjlab 在重置时报的各分项。

    每轮新建一个，不跨轮清空复用。mjlab 只在有环境重置的那一步刷新 `extras["log"]`
    （其余步里留的是上次重置时的旧值），且报的是「这一步结束的那些回合」的平均 ——
    所以分项只在重置步上记，并按这一步结束的回合数加权，汇总出来是逐回合的平均。
    """

    def __init__(self):
        self.rewards: list[float] = []
        self.lengths: list[float] = []
        self.term_sums: dict[str, float] = {}

    def add(self, rewards, lengths, mjlab_log):
        """记下这一步里结束的回合。

        Args:
            rewards: 这些回合的总奖励。
            lengths: 这些回合的步数。
            mjlab_log: 这一步 mjlab 的 `extras["log"]`（各项是这些回合的平均）。
        """
        count = len(rewards)
        self.rewards += rewards.tolist()
        self.lengths += lengths.tolist()
        for key, value in mjlab_log.items():
            scalar = float(value.mean() if torch.is_tensor(value) else value)
            self.term_sums[key] = self.term_sums.get(key, 0.0) + scalar * count

    def summary(self):
        """本轮的汇总；本轮一个回合都没结束时只给空字典（「没测到」不写成 0）。

        Returns:
            指标字典。
        """
        if not self.rewards:
            return {}
        episodes = len(self.rewards)
        return {
            "episode_reward": sum(self.rewards) / episodes,
            "episode_length": sum(self.lengths) / episodes,
            **{k: v / episodes for k, v in self.term_sums.items()},
        }


class EpisodeTracker:
    """每个环境当前这一回合累计到的奖励与步数，跨迭代持续。"""

    def __init__(self, num_envs, device):
        self.reward_sum = torch.zeros(num_envs, device=device)
        self.length = torch.zeros(num_envs, device=device)

    def step(self, rewards, dones, extras, window: IterationLog):
        """累计一步；有回合结束时把它记进本轮的窗口。

        Args:
            rewards: 本步各环境的原始奖励。
            dones: 本步各环境是否结束回合。
            extras: mjlab 的 extras 字典。
            window: 本轮的 `IterationLog`。
        """
        self.reward_sum += rewards
        self.length += 1
        done = dones.bool()
        if done.any():
            window.add(self.reward_sum[done], self.length[done], extras["log"])
            self.reward_sum[done] = 0.0
            self.length[done] = 0.0


class RolloutDataset(IterableDataset):
    """一轮迭代的在线数据：先采一段轨迹，再按 PPO 的划分吐出全部小批。"""

    def __init__(self, env, model, tracker):
        super().__init__()
        self.env = env
        self.model = model
        self.tracker = tracker
        self.window = IterationLog()
        self.rollout_obs: torch.Tensor | None = None
        self.rollout: dict[str, torch.Tensor] | None = None

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        # 一个数据集对象 = 一轮 = 一段 rollout，迭代几次都只采一次。Lightning 每轮重建
        # dataloader 后先迭代一次预取一个小批探路，轮内训练时再从头迭代一次；每次都采的话
        # 每轮推进 48 个环境步，课程表（按环境步分段）比上游快一倍，训练却照常跑、不报错。
        if self.rollout is None:
            self.rollout = self.sample_rollout()
        yield from self.iter_mini_batches(self.rollout)

    @torch.no_grad()
    def sample_rollout(self):
        """用当前策略采 24 步，记下旧策略的对数概率与分布参数，算好优势。

        Returns:
            整段 rollout，键为各量名，张量形状 (T, N, ...)。
        """
        env, model = self.env, self.model
        obs, critic_obs = env.get_observations()
        steps = {k: [] for k in ("obs", "critic_obs", "actions", "log_probs", "values",
                                 "rewards", "dones", "action_means", "action_stds")}
        for _ in range(NUM_STEPS_PER_ENV):
            actions, log_probs, values, means, stds = model.act(obs, critic_obs)
            next_obs, next_critic_obs, rewards, dones, info = env.step(actions)
            self.tracker.step(rewards, dones, info["mjlab_extras"], self.window)
            # 超时只是截断，不是终点：把 critic 对当前状态的估值折扣后补进奖励。
            # 加在副本上，统计里记的仍是环境给的原始奖励（同上游）。
            rewards = rewards + PPO.gamma * values.squeeze(-1) * info["time_outs"].float()
            for key, value in (("obs", obs), ("critic_obs", critic_obs), ("actions", actions),
                               ("log_probs", log_probs), ("values", values),
                               ("rewards", rewards), ("dones", dones.float()),
                               ("action_means", means), ("action_stds", stds)):
                steps[key].append(value)
            # 归一化统计量用新观测更新（同上游：在处理这一步时、用 step 返回的观测）。
            model.actor_normalizer.update(next_obs)
            model.critic_normalizer.update(next_critic_obs)
            obs, critic_obs = next_obs, next_critic_obs

        rollout = {k: torch.stack(v) for k, v in steps.items()}
        # 行为克隆在 PPO 更新之后、用的就是这一轮学生自己采到的观测。
        self.rollout_obs = rollout["obs"].reshape(-1, rollout["obs"].shape[-1])
        advantages, returns = compute_gae(rollout["rewards"], rollout["dones"], rollout["values"],
                                          model.value(critic_obs), PPO.gamma, PPO.lam)
        rollout["advantages"] = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
        rollout["returns"] = returns
        return rollout

    def iter_mini_batches(self, rollout):
        """把整段 rollout 打散成小批，同一份划分重复用 5 遍。

        Args:
            rollout: `sample_rollout` 的产出。

        Yields:
            一个个小批字典。
        """
        batch_size = rollout["actions"].shape[0] * rollout["actions"].shape[1]
        mini_batch_size = batch_size // PPO.num_mini_batches
        flat = {k: v.reshape(batch_size, *v.shape[2:]) for k, v in rollout.items()
                if k not in ("rewards", "dones")}
        # 一次划分、反复使用：上游的小批生成器在循环外只打乱一次。
        indices = torch.randperm(PPO.num_mini_batches * mini_batch_size, device=flat["obs"].device)
        for _ in range(PPO.num_learning_epochs):
            for i in range(PPO.num_mini_batches):
                selected = indices[i * mini_batch_size:(i + 1) * mini_batch_size]
                yield {k: v[selected] for k, v in flat.items()}


class RolloutData(L.LightningDataModule):
    """持有持久环境，每轮把用最新策略采到的新数据集交给 Trainer。"""

    def __init__(self, env, model):
        super().__init__()
        self.env = env
        self.model = model
        self.tracker = EpisodeTracker(env.num_envs, env.device)
        self.dataset: RolloutDataset | None = None

    def train_dataloader(self):
        """每轮重建一次数据集 = 每轮重新采样（配合 `reload_dataloaders_every_n_epochs=1`）。

        Returns:
            包着在线数据集的 DataLoader；`batch_size=None` 表示数据集自己吐整批。
        """
        self.dataset = RolloutDataset(self.env, self.model, self.tracker)
        return DataLoader(self.dataset, batch_size=None)


class PPO(L.LightningModule):
    """一个小批的 PPO 损失、学习率自适应、分组梯度裁剪、记录与存盘。"""

    clip_param = 0.2
    value_loss_coef = 1.0
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-3
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0

    def __init__(self, model, env, name, ckpt_dir, training_settings, iterations,
                 bc: ExpertBc | None = None, optimizer_state: dict | None = None,
                 critic_warmup_iters: int = 0):
        super().__init__()
        self.model = model
        self.env = env
        self.run_name = name
        self.ckpt_dir = ckpt_dir
        self.training_settings = training_settings
        self.iterations = iterations
        self.bc = bc
        # 热启动时继承源训练的 Adam 状态（同上游：权重、归一化、优化器都留，计数从 0 起）。
        self.optimizer_state = optimizer_state
        # 前这么多轮只训 critic（从 ONNX 续训时 critic 是新的）。
        self.critic_warmup_iters = critic_warmup_iters
        self.latest_kl = torch.zeros(())
        self.latest_lr = self.learning_rate
        self.batch_records: list[dict[str, float]] = []
        self.pending: list[dict[str, float]] = []
        self.pending_since = time.monotonic()
        self.wandb_run = None

    def setup(self, stage):
        """开训前起一个 W&B run（离线、不上屏：出错时它直接抛异常，不靠屏幕提示）。

        Args:
            stage: Lightning 传入的阶段名，只在 "fit" 阶段做事。
        """
        if stage != "fit":
            return
        self.wandb_run = wandb.init(project="miniduck", name=self.run_name,
                                    dir=self.ckpt_dir.as_posix(), config=self.training_settings,
                                    settings=wandb.Settings(silent=True))

    def configure_optimizers(self):
        """actor 与 critic 的全部参数交给一个 Adam。

        Returns:
            Adam 优化器。
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def on_fit_start(self):
        """热启动时把源训练的优化器状态装进刚建好的 Adam。"""
        if self.optimizer_state is not None:
            self.optimizers().optimizer.load_state_dict(self.optimizer_state)

    def on_train_epoch_start(self):
        """每轮开始时换一个新的指标列表。"""
        self.batch_records = []

    def in_critic_warmup(self) -> bool:
        """本轮是否还在 critic 热身期（只训 critic）。"""
        return self.current_epoch < self.critic_warmup_iters

    def training_step(self, batch, batch_idx):
        """算一个小批的 PPO 损失。

        Args:
            batch: 数据集吐出的一个小批。
            batch_idx: Lightning 规定的形参；按顺序消费数据，不需要序号。

        Returns:
            本步的总损失。
        """
        del batch_idx
        dist = self.model.action_distribution(batch["obs"])
        log_probs = dist.log_prob(batch["actions"]).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        values = self.model.value(batch["critic_obs"])
        with torch.no_grad():
            old = torch.distributions.Normal(batch["action_means"], batch["action_stds"])
            self.latest_kl = torch.distributions.kl_divergence(old, dist).sum(dim=-1).mean()

        ratio = torch.exp(log_probs - batch["log_probs"])
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate = torch.max(-batch["advantages"] * ratio,
                              -batch["advantages"] * clipped_ratio).mean()
        value_clipped = batch["values"] + (values - batch["values"]).clamp(-self.clip_param,
                                                                           self.clip_param)
        value_loss = torch.max((values - batch["returns"]).pow(2),
                               (value_clipped - batch["returns"]).pow(2)).mean()
        if self.in_critic_warmup():
            # 损失里不含 actor 的项，actor 与标准差拿不到梯度，Adam 不动它们。
            loss = self.value_loss_coef * value_loss
        else:
            loss = surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        self.batch_records.append({"surrogate": float(surrogate.detach()),
                                   "value_loss": float(value_loss.detach()),
                                   "entropy": float(entropy.detach()), "kl": float(self.latest_kl)})
        return loss

    def on_before_optimizer_step(self, optimizer):
        """这一步真正迈出去之前，按本小批的 KL 调学习率。

        调的是 Lightning 传进来、真的会被 step 的那个优化器。

        Args:
            optimizer: Lightning 传入的优化器。
        """
        kl = float(self.latest_kl)
        if self.in_critic_warmup():
            # actor 不动时 KL 恒为 0，自适应规则会把学习率一路乘到上限；热身期固定在初值。
            kl = self.desired_kl
        if kl > self.desired_kl * 2.0:
            self.latest_lr = max(1.0e-5, self.latest_lr / 1.5)
        elif 0.0 < kl < self.desired_kl / 2.0:
            self.latest_lr = min(1.0e-2, self.latest_lr * 1.5)
        for group in optimizer.param_groups:
            group["lr"] = self.latest_lr

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None,
                                    gradient_clip_algorithm=None):
        """actor 与 critic 分开裁剪梯度，各自的范数上限都是 1.0（同上游）。

        合在一起裁的话，critic 的大梯度会把 actor 的那份一并压小。

        Args:
            optimizer: Lightning 传入的优化器（这里不用，裁剪按参数分组做）。
            gradient_clip_val: Trainer 的统一裁剪值；本训练器不用它，故 Trainer 也不设。
            gradient_clip_algorithm: 同上。
        """
        del optimizer, gradient_clip_val, gradient_clip_algorithm
        torch.nn.utils.clip_grad_norm_(self.model.actor_parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.model.critic_parameters(), self.max_grad_norm)

    def on_train_epoch_end(self):
        """本轮 PPO 更新完：需要时跑行为克隆，再按间隔存盘，指标写进文件与 W&B，够 10 秒报一行进度。"""
        iteration = self.current_epoch + 1
        dataset = self.trainer.datamodule.dataset
        metrics = {k: sum(r[k] for r in self.batch_records) / len(self.batch_records)
                   for k in self.batch_records[0]}
        if self.bc is not None:
            with torch.enable_grad():
                metrics.update(self.bc.update(dataset.rollout_obs))
        if iteration % SAVE_INTERVAL == 0 or iteration == self.iterations:
            self.save(iteration)
        metrics.update(dataset.window.summary())
        metrics["lr"] = self.latest_lr
        metrics["action_std"] = float(self.model.std.detach().mean())
        wandb.log(metrics, step=iteration)
        with (self.ckpt_dir / "metrics.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"iteration": iteration, **metrics}, sort_keys=True) + "\n")
        self.pending.append(metrics)
        if time.monotonic() - self.pending_since >= SUMMARY_INTERVAL_S:
            self.log_progress(iteration)

    def log_progress(self, iteration):
        """屏幕上报一行进度：报的是自上一行以来这一段的平均，不是开训以来的累计。

        Args:
            iteration: 当前迭代号。
        """
        elapsed = time.monotonic() - self.pending_since
        rate = len(self.pending) / elapsed

        def mean(key):
            values = [m[key] for m in self.pending if key in m]
            return sum(values) / len(values) if values else None

        parts = [f"第 {iteration}/{self.iterations} 轮"]
        if iteration <= self.critic_warmup_iters:
            parts.append("critic 热身（actor 不动）")
        if mean("episode_length") is not None:
            parts += [f"回合平均 {mean('episode_length'):.0f} 步",
                      f"回合奖励 {mean('episode_reward'):.1f}"]
        # 只有带速度指令的任务才有这一项；没有就不报，不报成 nan。
        if mean("Metrics/twist/error_vel_xy") is not None:
            parts.append(f"平移跟踪误差 {mean('Metrics/twist/error_vel_xy'):.2f} m/s")
        parts += [f"学习率 {self.latest_lr:.1e}", f"{rate:.1f} 轮/秒",
                  f"预计还要 {(self.iterations - iteration) / rate / 60.0:.0f} 分钟"]
        log.info("%s", " · ".join(parts))
        self.pending = []
        self.pending_since = time.monotonic()

    def save(self, iteration):
        """存一份可评测、可导出的权重。

        课程表进度（`common_step_counter`）一并存下：评测要把环境对齐到训练到的那一档。

        Args:
            iteration: 当前迭代号。
        """
        path = self.ckpt_dir / f"model_{iteration}.pt"
        torch.save({
            "iteration": iteration,
            "common_step_counter": int(self.env.unwrapped.common_step_counter),
            "actor_critic": self.model.state_dict(),
            "optimizer": self.optimizers().optimizer.state_dict(),
            "training_settings": self.training_settings,
        }, path)
        log.info("存了 %s", path.name)

    def teardown(self, stage):
        """训练结束后把 W&B run 正常关掉。

        Args:
            stage: Lightning 传入的阶段名；收尾不分阶段。
        """
        del stage
        if self.wandb_run is not None:
            self.wandb_run.finish()


def init_policy_from_env(warm_from: Path | None) -> tuple[Path, int] | None:
    """读「从已发布的策略续训」的两个环境变量。

    Args:
        warm_from: 跨任务热启动的源权重（与续训互斥）。

    Returns:
        (ONNX 路径, 它在源训练里的迭代号)；没给 MINIDUCK_INIT_POLICY 时为 None。

    Raises:
        RuntimeError: 与热启动同时给，或没给源迭代号（课程表与归一化计数按它定）。
    """
    if not os.environ.get("MINIDUCK_INIT_POLICY"):
        return None
    policy = required_path("MINIDUCK_INIT_POLICY")
    if warm_from is not None:
        raise RuntimeError("MINIDUCK_INIT_POLICY（同任务续训）与 MINIDUCK_WARM_START（跨任务热启动）只能给一个")
    iteration = os.environ.get("MINIDUCK_INIT_ITERATION", "").strip()
    if not iteration.isdigit():
        raise RuntimeError("给了 MINIDUCK_INIT_POLICY 就要给 MINIDUCK_INIT_ITERATION（它在源训练里的迭代号），"
                           "课程表与归一化计数按它定")
    return policy, int(iteration)


def build_trainer(iterations: int, accelerator: str = "gpu") -> L.Trainer:
    """建 Lightning Trainer：一个 epoch = 一轮 PPO 迭代，每轮重建 dataloader = 每轮采新数据。

    Args:
        iterations: 迭代轮数。
        accelerator: 训练用 "gpu"；单测用 "cpu"。

    Returns:
        Trainer。
    """
    return L.Trainer(
        accelerator=accelerator, devices=1, max_epochs=iterations,
        reload_dataloaders_every_n_epochs=1, enable_checkpointing=False, logger=False,
        # 进度条写进日志文件就是满屏回车符；进度改由 log_progress 每 10 秒一行报。
        enable_model_summary=False, enable_progress_bar=False,
    )


def main():
    """起一次完整训练。

    Raises:
        RuntimeError: 这个 run 名的权重目录里已经有权重（不覆盖旧的训练）；要热启动的动作没给
            源权重，或从零训的动作给了；要克隆的动作没给老师权重；给了续训用的 ONNX
            却没给它的源迭代号，或与热启动同时给。
    """
    skill = skill_from_env()
    hw = hardware.from_env()
    iterations = max_iterations(skill)
    seed = train_seed()
    name = run_name()
    ckpt_dir = checkpoint_dir(name)
    if any(ckpt_dir.glob("model_*.pt")):
        raise RuntimeError(f"{ckpt_dir.name}/ 里已经有权重 —— 换一个 MINIDUCK_RUN_NAME，不覆盖旧的训练")
    upstream_cfg = check_upstream_cfg(skill)
    warm_from = required_path("MINIDUCK_WARM_START") if skill.warm_start else None
    if not skill.warm_start and os.environ.get("MINIDUCK_WARM_START"):
        raise RuntimeError(f"{skill.name} 在上游是从零训的，不接受 MINIDUCK_WARM_START")
    init = init_policy_from_env(warm_from)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    enable_tf32()
    # 上游 rsl_rl 关掉了分布参数校验（每次构造 Normal 都要校验一遍，纯开销）。
    torch.distributions.Distribution.set_default_validate_args(False)
    torch.manual_seed(seed)
    env = DuckEnv(skill, hw, num_envs=NUM_ENVS, seed=seed)
    if init is not None:
        # 同任务续训：课程表接着源训练走（同上游 resume 恢复 common_step_counter）。
        env.set_curriculum_step(init[1] * NUM_STEPS_PER_ENV)
    env.reset()
    env.randomize_episode_starts()
    model = ActorCritic(env.obs_dim, env.critic_obs_dim, env.action_dim).to(env.device)
    optimizer_state = None
    init_settings = None
    if init is not None:
        init_policy, init_iteration = init
        normalizer_count = init_iteration * NUM_STEPS_PER_ENV * NUM_ENVS
        load_actor_from_onnx(model, init_policy, normalizer_count)
        with torch.no_grad():
            model.std.fill_(FINETUNE_INIT_STD)
        init_settings = {"policy": str(init_policy), "policy_sha256": file_sha256(init_policy),
                         "source_iteration": init_iteration, "normalizer_count": normalizer_count,
                         "init_std": FINETUNE_INIT_STD, "critic_warmup_iters": CRITIC_WARMUP_ITERS}
        log.info("从 %s 续训：actor 与归一化取自 ONNX（与 onnxruntime 输出已核对），课程表放在源训练第 %d 轮，"
                 "动作标准差从 %.2f 起，前 %d 轮只训 critic",
                 init_policy.name, init_iteration, FINETUNE_INIT_STD, CRITIC_WARMUP_ITERS)
    if warm_from is not None:
        # 热启动 ≠ 续训：权重、观测归一化、优化器状态都继承，迭代号与课程表计数从 0 起
        # （上游配置里继承来的课程表已经直接从最终档起，见它的 WARM_START）。
        source = torch.load(warm_from, map_location="cpu", weights_only=False)
        model.load_state_dict(source["actor_critic"])
        optimizer_state = source["optimizer"]
        log.info("热启动：从 %s/%s（第 %d 轮）继承权重、归一化与优化器状态，计数从 0 起",
                 warm_from.parent.name, warm_from.name, source["iteration"])
    bc, bc_settings = build_expert_bc(skill, model, env.device)

    training_settings = {
        "run_name": name, "skill": skill.name, "task": skill.task_id, "num_envs": NUM_ENVS,
        "max_iterations": iterations, "num_steps_per_env": NUM_STEPS_PER_ENV, "seed": seed,
        "obs_dim": env.obs_dim, "critic_obs_dim": env.critic_obs_dim, "action_dim": env.action_dim,
        "upstream_cfg": upstream_cfg, "upstream_source": upstream_source(),
        **hw.training_record(skill.robot_xml),
        # 读模型时改名的碰撞体（改名不动模型文件、哈希不变，见 env.OUR_COLLISION_GEOMS）
        "collision_geom_renames": dict(OUR_COLLISION_GEOMS),
        "warm_start_from": str(warm_from) if warm_from else None, "expert_bc": bc_settings,
        "init_from_policy": init_settings,
    }
    (ckpt_dir / "training_settings.json").write_text(
        json.dumps(training_settings, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    log.info("开始训练 %s（%s，硬件 %s）：%d 个并行环境 × %d 轮，种子 %d，权重写到 %s/",
             skill.name, skill.task_id, hw.id, NUM_ENVS, iterations, seed, ckpt_dir.name)
    started = time.monotonic()
    trainer = build_trainer(iterations)
    trainer.fit(PPO(model, env, name, ckpt_dir, training_settings, iterations, bc, optimizer_state,
                    critic_warmup_iters=CRITIC_WARMUP_ITERS if init is not None else 0),
                RolloutData(env, model))
    env.close()
    log.info("训练结束：%d 轮，用时 %.1f 小时，最后一份权重 model_%d.pt",
             iterations, (time.monotonic() - started) / 3600.0, iterations)


if __name__ == "__main__":
    run_cli(main)
