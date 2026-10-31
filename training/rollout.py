"""评测一份策略，并录一段单只跟拍视频。

评测与录像都接收一个「观测 → 动作」的函数，而不是某种具体的模型：
同一套口径既评 PyTorch 权重，也评导出后的 ONNX 与 int8 —— 导出有没有把策略弄坏，
就看三者在同一批初始状态上的分数是否一致。

口径：512 个并行环境、各跑一个完整回合（回合长度取任务自己的：走路 20 秒、站起 6 秒…）、
确定性动作、按训练的指令与初始姿态分布采样、课程表对齐到权重训练到的那一档。

为什么是 512 而不是更少：GPU 上的仿真不是逐位可复现的，同一份权重重跑，64 个环境里的
摔倒次数实测在 4 到 8 次之间跳，三种导出形态之间的差别会被这个抽样噪声盖住。
仿真时间几乎不随环境数涨，多跑几倍样本是近乎免费的。

硬件版本跟着权重走（训练设置里记的那一版），不另外指定。

跑法（训练机上，仓根目录）：
    MINIDUCK_SKILL=<动作> MINIDUCK_RUN_NAME=<run> uv run python training/rollout.py            # 评最新一档
    MINIDUCK_SKILL=<动作> MINIDUCK_RUN_NAME=<run> uv run python training/rollout.py model_6000.pt
"""
from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

import mediapy as media
import numpy as np
import torch

import hardware
from env import SEED, DuckEnv, Skill, skill_from_env
from hardware import Hardware
from logs import run_cli
from model import ActorCritic
from paths import checkpoint_dir, result_dir, run_name

EVAL_ENVS = 512
VIDEO_STEPS = 500
# 录像时把指令钉住的动作：走路以 0.3 米/秒直走、头与身体取默认姿态。其余动作按任务自己的
# 指令与初始姿态分布录 —— 站起要从躺着开始、坐站要有坐下的指令，钉住反而录不到。
VIDEO_PINNED_SPEED = {"walk": 0.3}

Policy = Callable[[torch.Tensor], torch.Tensor]

log = logging.getLogger(__name__)


def load_checkpoint(path: Path, device) -> tuple[ActorCritic, dict]:
    """从 checkpoint 恢复策略。网络维度从 checkpoint 自带的设置里取，不写死。

    Args:
        path: checkpoint 路径。
        device: 放到哪个设备。

    Returns:
        (策略, checkpoint 字典)。
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    settings = ckpt["training_settings"]
    model = ActorCritic(settings["obs_dim"], settings["critic_obs_dim"], settings["action_dim"])
    model.load_state_dict(ckpt["actor_critic"])
    return model.eval().to(device), ckpt


def trained_hardware(ckpt: dict) -> Hardware:
    """这份权重是在哪个硬件版本上训的：评测、导出都必须用同一版，不看环境变量。

    Args:
        ckpt: checkpoint 字典。

    Returns:
        硬件版本。
    """
    return hardware.of_training_settings(ckpt["training_settings"])


def latest_checkpoint(name: str) -> Path:
    """某次训练里迭代号最大的那份权重（按数值排序，不按字符串）。

    Args:
        name: run 名。

    Returns:
        checkpoint 路径。

    Raises:
        RuntimeError: 目录里一份权重都没有。
    """
    found = sorted(checkpoint_dir(name).glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not found:
        raise RuntimeError(f"{checkpoint_dir(name).name}/ 下没有任何 model_*.pt —— run 名对吗？")
    return found[-1]


def evaluate(policy: Policy, skill: Skill, hw: Hardware, common_step_counter: int,
             num_envs=EVAL_ENVS, device="cuda:0") -> dict:
    """按统一口径跑一个完整回合的确定性 rollout，返回评测摘要。

    「摔倒」只数 `terminated`（被判失败），不数超时（`truncated`）：正好跑一个回合，
    最后一步全体超时，那不是摔。
    各分项取 mjlab 在回合结束时报的那一份（`extras["log"]` 只在有环境重置的
    那一步刷新，报的是这一步结束的那些回合的平均），按每步结束的回合数加权，
    汇总出来是逐回合的平均。

    Args:
        policy: 观测 → 动作的函数（确定性）。
        skill: 动作声明。
        hw: 硬件版本（与权重训练时的一致）。
        common_step_counter: 课程表对齐到哪一档（checkpoint 里存的那个值）。
        num_envs: 并行环境数。
        device: 仿真所在设备。

    Returns:
        评测摘要字典。
    """
    torch.manual_seed(SEED)
    env = DuckEnv(skill, hw, num_envs=num_envs, device=device, seed=SEED)
    try:
        num_steps = env.max_episode_length
        # 顺序是硬的：先对齐课程表再 reset —— 初始状态按 reset 那一刻的档位抽。
        env.set_curriculum_step(common_step_counter)
        obs, _ = env.reset()
        rewards, falls, term_sums, episodes = [], 0, {}, 0
        fell = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        for _ in range(num_steps):
            with torch.no_grad():
                actions = policy(obs)
            obs, _, reward, dones, info = env.step(actions)
            rewards.append(float(reward.mean()))
            terminated = info["terminated"]
            falls += int(terminated.sum())
            fell |= terminated
            ended = int(dones.sum())
            if ended:
                episodes += ended
                for key, value in info["mjlab_extras"]["log"].items():
                    scalar = float(value.mean() if torch.is_tensor(value) else value)
                    term_sums[key] = term_sums.get(key, 0.0) + scalar * ended
    finally:
        env.close()
    return {
        "skill": skill.name,
        "hardware": hw.id,
        "num_envs": num_envs,
        "steps": num_steps,
        "curriculum_step": int(common_step_counter),
        "mean_step_reward": float(np.mean(rewards)),
        "falls": falls,
        # 跑一个完整回合，被判失败过一次的环境占多少。
        "fall_env_fraction": float(fell.float().mean()),
        "episodes": episodes,
        "terms": {k: v / episodes for k, v in sorted(term_sums.items())},
    }


def collect_observations(policy: Policy, skill: Skill, hw: Hardware, common_step_counter: int,
                         num_envs=64, num_steps=500, device="cuda:0") -> np.ndarray:
    """让策略在仿真里走一段，收集它真实遇到的 actor 观测（int8 标定用）。

    Args:
        policy: 观测 → 动作的函数。
        skill: 动作声明。
        hw: 硬件版本（与权重训练时的一致）。
        common_step_counter: 课程表对齐到哪一档。
        num_envs: 并行环境数。
        num_steps: 走多少步。
        device: 仿真所在设备。

    Returns:
        形状 (num_envs × num_steps, 61) 的原始观测。
    """
    torch.manual_seed(SEED)
    env = DuckEnv(skill, hw, num_envs=num_envs, device=device, seed=SEED)
    seen = []
    try:
        env.set_curriculum_step(common_step_counter)
        obs, _ = env.reset()
        for _ in range(num_steps):
            seen.append(obs.detach().cpu().numpy())
            with torch.no_grad():
                obs, *_ = env.step(policy(obs))
    finally:
        env.close()
    return np.concatenate(seen, axis=0)


def record_video(policy: Policy, skill: Skill, hw: Hardware, common_step_counter: int, path: Path,
                 num_steps=VIDEO_STEPS, device="cuda:0") -> int:
    """单只鸭子跟拍录像；走路钉住直走指令，其余动作按任务自己的分布。

    Args:
        policy: 观测 → 动作的函数。
        skill: 动作声明。
        hw: 硬件版本（与权重训练时的一致）。
        common_step_counter: 课程表对齐到哪一档。
        path: mp4 输出路径。
        num_steps: 录多少步（50 步 = 1 秒）。
        device: 仿真所在设备。

    Returns:
        写入的帧数。

    Raises:
        RuntimeError: 一帧都没渲出来（多半是没设 `MUJOCO_GL=egl`）。
    """
    speed = VIDEO_PINNED_SPEED.get(skill.name)
    torch.manual_seed(SEED)
    env = DuckEnv(skill, hw, num_envs=1, device=device, seed=SEED, render=True)
    frames = []
    try:
        env.set_curriculum_step(common_step_counter)
        env.reset()
        if speed is not None:
            env.pin_commands(speed)
        # 观测是缓存的，钉住的指令从下一次 step 起才生效：先空走一步。
        obs, *_ = env.step(torch.zeros(1, env.action_dim, device=env.device))
        for _ in range(num_steps):
            with torch.no_grad():
                obs, *_ = env.step(policy(obs))
            if speed is not None:
                # 每步都钉一次：回合结束重置时指令会被重采样。
                env.pin_commands(speed)
            frame = env.render()
            if frame is not None:
                frame = np.asarray(frame[0] if np.ndim(frame) == 4 else frame)
                frames.append(frame if frame.dtype == np.uint8
                              else (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8))
    finally:
        env.close()
    if not frames:
        raise RuntimeError("一帧都没渲出来 —— 无头机器上要 MUJOCO_GL=egl")
    media.write_video(str(path), frames, fps=50)
    return len(frames)


def main():
    """评测指定（默认最新）的一份权重，写出摘要 json 与视频。"""
    skill = skill_from_env()
    name = run_name()
    ckpt_path = checkpoint_dir(name) / sys.argv[1] if len(sys.argv) > 1 else latest_checkpoint(name)
    model, ckpt = load_checkpoint(ckpt_path, "cuda:0")
    hw = trained_hardware(ckpt)
    out = result_dir(name)
    counter = int(ckpt["common_step_counter"])
    log.info("评测 %s / %s（硬件 %s）：%d 个环境各跑一个回合，课程表对齐到第 %d 步",
             skill.name, ckpt_path.name, hw.id, EVAL_ENVS, counter)
    summary = {"checkpoint": str(ckpt_path), "iteration": ckpt["iteration"],
               **evaluate(model.act_inference, skill, hw, counter)}
    terms = summary["terms"]
    tracking = ""
    if "Metrics/twist/error_vel_xy" in terms:
        tracking = (f"，平移跟踪误差 {terms['Metrics/twist/error_vel_xy']:.2f} m/s，"
                    f"转向跟踪误差 {terms['Metrics/twist/error_vel_yaw']:.2f} rad/s")
    log.info("评测完：被判失败 %d 次（%.0f%% 的环境）%s",
             summary["falls"], 100 * summary["fall_env_fraction"], tracking)
    video = out / f"{skill.name}-{ckpt['iteration']}.mp4"
    summary["video"] = str(video)
    summary["video_frames"] = record_video(model.act_inference, skill, hw, counter, video)
    log.info("录像写到 %s（%d 帧，%.0f 秒）", video.name, summary["video_frames"],
             summary["video_frames"] / 50)
    report = out / f"eval-{skill.name}-{ckpt['iteration']}.json"
    report.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    log.info("评测摘要写到 %s", report.name)


if __name__ == "__main__":
    run_cli(main)
