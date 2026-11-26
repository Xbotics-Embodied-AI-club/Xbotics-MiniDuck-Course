"""PPO 训练小鸭子的各个动作。

训哪个动作由 `MINIDUCK_SKILL` 选，在哪个硬件版本上训由 `MINIDUCK_HARDWARE` 选（`<日期>-<舵机>`，
见 sim_model/hardware/），这一次叫什么由 `MINIDUCK_RUN_NAME` 给；三者都必须显式给出。

结构：`trainer.fit(PPO(net), TrainingData(env, net))`。PPO 的两半在 `ppo.py`（data 一半采样、算优势、切小批，
model 一半算损失、按 KL 调学习率）；这里只写正式训练在它之上另外要的东西：采样时记回合统计与行为克隆的标记
（`TrainingData`），每轮更新之后的行为克隆、存盘、记录与进度（`IterationEnd`），续训时先只训 critic（`CriticWarmup`），
热启动时接上源训练的优化器状态（`RestoreOptimizer`）。

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

左右对称：上游只有前滚翻（roulade）打开了，用的是**镜像损失**（不做数据增广）：
把观测左右镜像后喂给 actor，它给的动作均值要等于原观测下动作均值的镜像，
两者的均方差 × 系数加进总损失（rsl_rl 5.0.1 `PPO.update` 的 symmetry 分支）。镜像表直接用上游的
`mjlab_microduck.tasks.symmetry.microduck_vel_symmetry`。上游要是换成数据增广或别的镜像函数，
`check_upstream_cfg` 当场报错，而不是只执行一半的配方。

走路 + 摔倒恢复（`velstand-*`）在上游还多两件，这里照做：

- 在别的策略上**热启动**（`MINIDUCK_WARM_START` 指定源权重）：继承权重、观测归一化与
  优化器状态，迭代号与课程表计数从 0 起；
- 每轮 PPO 更新之后跑一遍**行为克隆**（`distill.py`；站立老师与走路老师由
  `MINIDUCK_BC_EXPERT` / `MINIDUCK_BC_ANCHOR` 指定，其余设置读上游配置）。

**从关节更多的策略移植**（`MINIDUCK_TRANSPLANT_FROM` 指同一动作的本仓 checkpoint）：硬件版本锁死了
一部分关节时，切掉那几维、其余照搬（`transplant.py`），优化器从头，课程表放在源训练到的那一档。

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
from pathlib import Path

import lightning as L
import torch
import wandb
from mjlab.tasks.registry import load_rl_cfg
from mjlab_microduck.tasks.distill import PpoWithExpertBcCfg

from training.logs import run_cli
from training.paths import checkpoint_dir, file_sha256, run_name
from training.ppo.data import STEPS_PER_ITERATION, sample_rollout
from training.ppo.distill import (
    BcSettings,
    ExpertBc,
    free_supine,
    load_teacher,
    masked_teacher,
)
from training.ppo.network import ActorCritic
from training.ppo.ppo import PPO, PPOData
from training.resume.from_onnx import load_actor_from_onnx
from training.resume.transplant import layout_of, transplant
from training.sim import getup, hardware
from training.sim.env import (
    ENCODER_ZERO_ERROR_RAD,
    NUM_ENVS,
    OUR_COLLISION_GEOMS,
    SEED,
    SKILLS,
    DuckEnv,
    Skill,
    max_iterations,
    skill_from_env,
    upstream_v8,
)

SAVE_INTERVAL = 250
# 上游对称配置里的镜像函数（按名字核对，见 check_upstream_cfg）。
MIRROR_FUNC = "mjlab_microduck.tasks.symmetry.microduck_vel_symmetry"
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
        "num_steps_per_env": (cfg.num_steps_per_env, STEPS_PER_ITERATION),
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
        "num_learning_epochs": (algo.num_learning_epochs, PPOData.num_learning_epochs),
        "num_mini_batches": (algo.num_mini_batches, PPOData.num_mini_batches),
        "learning_rate": (algo.learning_rate, PPO.learning_rate),
        "schedule": (algo.schedule, "adaptive"),
        "gamma": (algo.gamma, PPOData.gamma),
        "lam": (algo.lam, PPOData.lam),
        "desired_kl": (algo.desired_kl, PPO.desired_kl),
        "max_grad_norm": (algo.max_grad_norm, PPO.max_grad_norm),
        "normalize_advantage_per_mini_batch": (algo.normalize_advantage_per_mini_batch, False),
        "seed": (cfg.seed, SEED),
    }
    wrong = {k: v for k, v in expect.items() if v[0] != v[1]}
    # 走路任务用的是带对称性字段的配置类；直接取属性，字段改名时当场报错。
    symmetry = algo.symmetry_cfg
    if symmetry is not None and (symmetry["use_data_augmentation"] or not symmetry["use_mirror_loss"]
                                 or symmetry["data_augmentation_func"] != MIRROR_FUNC):
        wrong["symmetry_cfg"] = (symmetry, f"只做镜像损失，镜像函数 {MIRROR_FUNC}")
    if wrong:
        raise RuntimeError(f"上游配置与本训练器不一致（上游值, 本训练器值）：{wrong}")
    return {**{k: v[0] for k, v in expect.items()}, "symmetry_cfg": symmetry}


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
    if isinstance(algo, PpoWithExpertBcCfg):
        settings = BcSettings.from_upstream(algo.bc_cfg)
    elif upstream_v8() and skill.task_id == SKILLS["standup"].task_id:
        # 起身在上游是纯 PPO。上游 v8 的教训：难的倒地状态里只靠 PPO，最便宜的解是躺着不动；velstand 的起身
        # 是照站立老师克隆、老师在 75° 以上看不到关节才救回来的。起身照同样的办法：倾斜过克隆门槛的帧学老师
        # （由 MINIDUCK_BC_EXPERT 给），克隆的门槛、系数、学习率取上游 velstand 的设置；没有走路锚点。
        upstream_bc = load_rl_cfg(SKILLS["velstand-flat"].task_id).algorithm.bc_cfg
        settings = dataclasses.replace(BcSettings.from_upstream(upstream_bc), anchor_coef=0.0, body_slice=None,
                                       unanchor_turn_in_place=False)
    else:
        return None, None
    if upstream_v8() and os.environ.get("MINIDUCK_V8_MASK", "1").strip() != "0":
        # MINIDUCK_V8_MASK=0：只要 v8 的宽开局、老师照常看关节。我们上一版起身被遮关节后自己只能救起 24%
        # （倒地循环），遮关节这一半对它不成立；上游是在他们自己的老师上验证的。
        settings = masked_teacher(settings)
    if skill.weak_getup:
        settings = free_supine(settings, getup.SUPINE_EXIT_GX)
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


class TrainingData(PPOData):
    """正式训练的数据：PPO 的数据，采样时顺带记回合统计，以及行为克隆要的每帧标记。"""

    def __init__(self, env: DuckEnv, net: ActorCritic):
        super().__init__(env, net)
        # 每个环境当前回合累计的奖励与步数，跨轮持续；本轮结束的回合记进 window（每轮换一个新的）
        self.tracker = EpisodeTracker(env.num_envs, env.device)
        self.window = IterationLog()
        # 行为克隆在 PPO 更新之后用这一轮学生自己采到的观测，以及每帧能不能学站立老师、走路老师的标记
        self.rollout_obs: torch.Tensor | None = None
        self.rollout_anchor_ok: torch.Tensor | None = None
        self.rollout_expert_ok: torch.Tensor | None = None

    def collect(self):
        self.window = IterationLog()
        expert_ok = self.env.expert_allowed()   # 舵机电压开训时抽定、重置不重抽，整轮不变
        # 这一帧所在的环境是不是平地。坡在重置时才换，所以第 t 帧取的是第 t 步之前那一刻：
        # 开采前取一次，之后每走一步取一次（那是下一帧的），最后多出来的一个不要
        anchor_ok = [self.env.anchor_allowed()]

        def on_step(rewards, dones, info):
            self.tracker.step(rewards, dones, info["mjlab_extras"], self.window)   # 统计记原始奖励（同上游）
            anchor_ok.append(self.env.anchor_allowed())

        rollout = sample_rollout(self.env, self.net, self.steps, on_step)
        self.rollout_obs = rollout["obs"].flatten(0, 1)
        self.rollout_anchor_ok = torch.stack(anchor_ok[:-1]).flatten()
        self.rollout_expert_ok = expert_ok.repeat(self.steps)
        return rollout


class IterationEnd(L.Callback):
    """每轮 PPO 更新之后正式训练要做的事：需要时跑行为克隆，按间隔存盘，指标写进文件与 W&B，够 10 秒报一行进度。"""

    def __init__(self, name: str, ckpt_dir: Path, training_settings: dict, iterations: int,
                 bc: ExpertBc | None = None):
        self.run_name = name
        self.ckpt_dir = ckpt_dir
        self.training_settings = training_settings
        self.iterations = iterations
        self.bc = bc
        self.pending: list[dict[str, float]] = []
        self.pending_since = time.monotonic()
        self.wandb_run = None

    def setup(self, trainer, pl_module, stage):
        # W&B 离线、不上屏：出错时它直接抛异常，不靠屏幕提示
        del trainer, pl_module
        if stage == "fit":
            self.wandb_run = wandb.init(project="miniduck", name=self.run_name, dir=self.ckpt_dir.as_posix(),
                                        config=self.training_settings, settings=wandb.Settings(silent=True))

    def on_train_epoch_end(self, trainer, pl_module):
        data = trainer.datamodule
        iteration = trainer.current_epoch + 1
        # PPO 各项损失的本轮平均（`ppo.PPO.loss` 按轮汇总的那几项；进度条上的逐步数字不算）
        metrics = {key: float(value) for key, value in trainer.callback_metrics.items()
                   if key not in trainer.progress_bar_metrics}
        if self.bc is not None:
            with torch.enable_grad():
                metrics.update(self.bc.update(data.rollout_obs, data.rollout_anchor_ok, data.rollout_expert_ok))
        if iteration % SAVE_INTERVAL == 0 or iteration == self.iterations:
            self.save(trainer, pl_module, iteration)
        metrics.update(data.window.summary())
        metrics["lr"] = pl_module.lr
        metrics["action_std"] = float(pl_module.net.std.detach().mean())
        wandb.log(metrics, step=iteration)
        with (self.ckpt_dir / "metrics.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"iteration": iteration, **metrics}, sort_keys=True) + "\n")
        self.pending.append(metrics)
        if time.monotonic() - self.pending_since >= SUMMARY_INTERVAL_S:
            self.log_progress(iteration, pl_module)

    def log_progress(self, iteration: int, pl_module: PPO):
        """屏幕上报一行进度：报的是自上一行以来这一段的平均，不是开训以来的累计。"""
        elapsed = time.monotonic() - self.pending_since
        rate = len(self.pending) / elapsed

        def mean(key):
            values = [m[key] for m in self.pending if key in m]
            return sum(values) / len(values) if values else None

        parts = [f"第 {iteration}/{self.iterations} 轮"]
        if not pl_module.train_actor:
            parts.append("critic 热身（actor 不动）")
        if mean("episode_length") is not None:
            parts += [f"回合平均 {mean('episode_length'):.0f} 步", f"回合奖励 {mean('episode_reward'):.1f}"]
        # 只有带速度指令的任务才有这一项；没有就不报，不报成 nan。
        if mean("Metrics/twist/error_vel_xy") is not None:
            parts.append(f"平移跟踪误差 {mean('Metrics/twist/error_vel_xy'):.2f} m/s")
        parts += [f"学习率 {pl_module.lr:.1e}", f"{rate:.1f} 轮/秒",
                  f"预计还要 {(self.iterations - iteration) / rate / 60.0:.0f} 分钟"]
        log.info("%s", " · ".join(parts))
        self.pending = []
        self.pending_since = time.monotonic()

    def save(self, trainer, pl_module: PPO, iteration: int):
        """存一份可评测、可导出、可续训的权重；课程表进度一并存下（评测要把环境对齐到训练到的那一档）。"""
        path = self.ckpt_dir / f"model_{iteration}.pt"
        torch.save({
            "iteration": iteration,
            "common_step_counter": int(trainer.datamodule.env.unwrapped.common_step_counter),
            "actor_critic": pl_module.net.state_dict(),
            "optimizer": trainer.optimizers[0].state_dict(),
            "training_settings": self.training_settings,
        }, path)
        log.info("存了 %s", path.name)

    def teardown(self, trainer, pl_module, stage):
        del trainer, pl_module, stage
        if self.wandb_run is not None:
            self.wandb_run.finish()


class CriticWarmup(L.Callback):
    """前几轮只训 critic：续训、移植时 critic 是新建的，它给的优势在头几十轮是噪声，直接拿来更新 actor 会把已学会的动作推散。"""

    def __init__(self, iterations: int):
        self.iterations = iterations

    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.train_actor = trainer.current_epoch >= self.iterations


class RestoreOptimizer(L.Callback):
    """热启动：把源训练的 Adam 状态装进刚建好的优化器（同上游：权重、归一化、优化器都继承，计数从 0 起）。"""

    def __init__(self, state: dict):
        self.state = state

    def on_fit_start(self, trainer, pl_module):
        del pl_module
        trainer.optimizers[0].load_state_dict(self.state)


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


def transplant_source_from_env(skill, warm_from: Path | None, init) -> tuple[Path, dict] | None:
    """读「从关节更多的策略移植」的环境变量 `MINIDUCK_TRANSPLANT_FROM`（源 checkpoint）。

    Args:
        skill: 这次训的动作。
        warm_from: 跨任务热启动的源权重（与移植互斥）。
        init: 从 ONNX 续训的设置（与移植互斥）。

    Returns:
        (源 checkpoint 路径, 读进来的 checkpoint)；没给时为 None。

    Raises:
        RuntimeError: 与热启动或续训同时给，或源策略训的不是同一个动作。
    """
    if not os.environ.get("MINIDUCK_TRANSPLANT_FROM"):
        return None
    if warm_from is not None or init is not None:
        raise RuntimeError("MINIDUCK_TRANSPLANT_FROM 与 MINIDUCK_WARM_START / MINIDUCK_INIT_POLICY 只能给一个")
    path = required_path("MINIDUCK_TRANSPLANT_FROM")
    source = torch.load(path, map_location="cpu", weights_only=False)
    if source["training_settings"]["skill"] != skill.name:
        raise RuntimeError(f"源策略训的是 {source['training_settings']['skill']}，这次训 {skill.name}，不移植")
    return path, source


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
    transplant_from = transplant_source_from_env(skill, warm_from, init)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    enable_tf32()
    # 上游 rsl_rl 关掉了分布参数校验（每次构造 Normal 都要校验一遍，纯开销）。
    torch.distributions.Distribution.set_default_validate_args(False)
    torch.manual_seed(seed)
    source_layout = None
    if transplant_from is not None:
        # 源策略的观测布局要从源硬件版本的环境里取；建一个单环境的、取完就关。
        source_env = DuckEnv(skill, hardware.of_training_settings(transplant_from[1]["training_settings"]),
                             num_envs=1, seed=seed)
        source_layout = layout_of(source_env)
        source_env.close()
    env = DuckEnv(skill, hw, num_envs=NUM_ENVS, seed=seed)
    if init is not None:
        # 同任务续训：课程表接着源训练走（同上游 resume 恢复 common_step_counter）。
        env.set_curriculum_step(init[1] * STEPS_PER_ITERATION)
    if transplant_from is not None:
        # 移植：源策略已经训到课程表末档，环境也放在那一档。
        env.set_curriculum_step(int(transplant_from[1]["common_step_counter"]))
    env.reset()
    env.randomize_episode_starts()
    model = ActorCritic(env.obs_dim, env.critic_obs_dim, env.action_dim).to(env.device)
    optimizer_state = None
    init_settings = None
    if init is not None:
        init_policy, init_iteration = init
        normalizer_count = init_iteration * STEPS_PER_ITERATION * NUM_ENVS
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
    transplant_settings = None
    if transplant_from is not None:
        path, source = transplant_from
        assert source_layout is not None
        target_layout = layout_of(env)
        model.load_state_dict(transplant(source["actor_critic"], source_layout, target_layout))
        transplant_settings = {
            "checkpoint": str(path), "checkpoint_sha256": file_sha256(path), "source_iteration": source["iteration"],
            "source_hardware": hardware.of_training_settings(source["training_settings"]).id,
            "common_step_counter": int(source["common_step_counter"]),
            "dropped_joints": sorted(set(source_layout.joints) - set(target_layout.joints))}
        log.info("移植：从 %s/%s（第 %d 轮，硬件 %s）切掉 %s，其余权重、归一化与动作标准差照搬；"
                 "优化器从头，课程表放在源训练的第 %d 步", path.parent.name, path.name, source["iteration"],
                 transplant_settings["source_hardware"], transplant_settings["dropped_joints"],
                 transplant_settings["common_step_counter"])
    bc, bc_settings = build_expert_bc(skill, model, env.device)

    training_settings = {
        "run_name": name, "skill": skill.name, "task": skill.task_id, "num_envs": NUM_ENVS,
        "reward_weights": dict(skill.reward_weights),   # MINIDUCK_REWARD_WEIGHTS 改过的几项；空 = 全照上游
        "encoder_zero_error_rad": ENCODER_ZERO_ERROR_RAD,   # 没有这一项的旧权重是按上游的 ±0.015 训的
        "max_iterations": iterations, "num_steps_per_env": STEPS_PER_ITERATION, "seed": seed,
        "obs_dim": env.obs_dim, "critic_obs_dim": env.critic_obs_dim, "action_dim": env.action_dim,
        "upstream_cfg": upstream_cfg, "upstream_source": upstream_source(),
        **hw.training_record(skill.robot_xml),
        # 读模型时改名的碰撞体（改名不动模型文件、哈希不变，见 env.OUR_COLLISION_GEOMS）
        "collision_geom_renames": dict(OUR_COLLISION_GEOMS),
        # 上游 v8 的起身改进（遮关节的站立老师 + 更宽的倒地开局；env.upstream_v8、distill.masked_teacher）
        "upstream_v8": upstream_v8(),
        "warm_start_from": str(warm_from) if warm_from else None, "expert_bc": bc_settings,
        "init_from_policy": init_settings, "transplant_from": transplant_settings,
    }
    (ckpt_dir / "training_settings.json").write_text(
        json.dumps(training_settings, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    if transplant_from is not None:
        # 移植完、一轮没训的那份也存下来：「直接砍掉关节不重训还剩多少」用同一套评测脚本量。
        torch.save({"iteration": 0, "common_step_counter": int(env.unwrapped.common_step_counter),
                    "actor_critic": model.state_dict(), "optimizer": None,
                    "training_settings": training_settings}, ckpt_dir / "model_0.pt")

    log.info("开始训练 %s（%s，硬件 %s）：%d 个并行环境 × %d 轮，种子 %d，权重写到 %s/",
             skill.name, skill.task_id, hw.id, NUM_ENVS, iterations, seed, ckpt_dir.name)
    started = time.monotonic()
    callbacks: list[L.Callback] = [IterationEnd(name, ckpt_dir, training_settings, iterations, bc)]
    if init is not None or transplant_from is not None:
        # 移植来的 actor 一开训就会走（锁头模型上直接测：不摔、速度照旧），而新建的优化器
        # 头两轮就能把它推散（训 2 轮后摔一半），所以同续训一样先只训 critic。
        callbacks.append(CriticWarmup(CRITIC_WARMUP_ITERS))
    if optimizer_state is not None:
        callbacks.append(RestoreOptimizer(optimizer_state))
    trainer = L.Trainer(accelerator="gpu", devices=1, max_epochs=iterations, reload_dataloaders_every_n_epochs=1,
                        enable_checkpointing=False, logger=False, enable_model_summary=False,
                        enable_progress_bar=False, callbacks=callbacks)
    trainer.fit(PPO(model, mirror_loss_coeff=(upstream_cfg["symmetry_cfg"] or {}).get("mirror_loss_coeff")),
                TrainingData(env, model))
    env.close()
    log.info("训练结束：%d 轮，用时 %.1f 小时，最后一份权重 model_%d.pt",
             iterations, (time.monotonic() - started) / 3600.0, iterations)


if __name__ == "__main__":
    run_cli(main)
