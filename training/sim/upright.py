"""坡地动作里的直立判据：按真实重力方向算，不按世界 z 轴算。

坡是把每个环境的重力斜过去造的，地面仍是 z = 0 的平面（terrain.py），所以世界 z 轴是坡面法线、不是「上」。
上游起身与走路任务的几项直立奖励和倒地判定，用根四元数算「机体 z 轴与世界 z 轴的夹角」
（cos = R₂₂ = 1 − 2(qx² + qy²)），斜重力下奖励的就成了「垂直于坡面站」：第一版坡地起身
在 10° 下坡上站成了相对坡面法线 4.0°、相对重力 8.6°。观测、翻倒终止（mjlab `bad_orientation`）、
走路的 `upright` 读的是 projected_gravity_b / gravity_vec_w，本来就按重力，不在这里。

这里每一项都照上游（mjlab_microduck `UPSTREAM_COMMIT`，`mjlab_microduck/tasks/mdp.py`，各函数写了行号）逐行同样算，
只把「上」从世界 z 轴换成 −重力。g 是机体系单位重力（projected_gravity_b，由 terrain.Slopes 写进的 gravity_vec_w 现算）：

- cos(倾斜) = −g_z；重力竖直时 g = −Rᵀe_z，−g_z = R₂₂，与上游同一个式子；
- 横滚、俯仰（ZYX，上游 `body_pose_tracking_locomotion` 的约定）= atan2(−g_y, −g_z)、asin(g_x)；
  重力竖直时 −g_y = R₂₁ = 2(qw qx + qy qz)、−g_z = R₂₂、g_x = −R₂₀ = 2(qw qy − qz qx)，与上游同式。
  航向（绕竖直轴）照旧用四元数：它要和脚的朝向（世界系）比，两边得在同一个系里。

两种改法，按上游怎么取倾斜分：

1. 函数体里自己从四元数算倾斜的，照上游函数体复制、只换倾斜那几行：起身的直立线性项、站高处的直立高斯项、
   站立综合分、站高处的角速度阻尼（倾斜门）、头偏置（倾斜门）、身体姿态跟踪的横滚俯仰，走路的直立进度、倒地税、恢复奖金。
2. 通过上游内部辅助函数 `_fallen_mask` 判倒地的（腾空奖励、倒地时动作变化率与力矩变化率打折、上升速度奖励的门、
   倒地太久终止）：不复制函数体，用 `fallen_by_gravity` 包一层，调用期间把上游模块里的 `_fallen_mask`
   换成 `gravity_fallen_mask`、调用完换回。这几个函数都是调用时按模块全局名字取 `_fallen_mask` 的，
   换这一处就换掉了它们的门；只包坡地动作的这几项，平地动作照旧调上游原函数。

`use_gravity_up(cfg)` 按对照表替换配置里的奖励与终止项；`env.skill_env_cfg` 只对 slope=True 的动作调用。
"""
from __future__ import annotations

import contextlib
import dataclasses
import functools
import math
from collections.abc import Callable

import torch
from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import wrap_to_pi
from mjlab_microduck.tasks import mdp as microduck_mdp

UPSTREAM_COMMIT = "ceca01706143d729305baa94cbd8ab990979e325"
_ROBOT = SceneEntityCfg("robot")
# 上游身体姿态跟踪的脚的缺省（同样是定义时建好、不经管理器解析的那一个对象）
_FEET = SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))


def _unit_gravity_b(asset: Entity) -> torch.Tensor:
    g = asset.data.projected_gravity_b
    return g / g.norm(dim=1, keepdim=True).clamp_min(1e-9)


def cos_tilt(asset: Entity) -> torch.Tensor:
    """机体 z 轴与「上」（−重力）夹角的余弦，形状 (N,)。"""
    return -_unit_gravity_b(asset)[:, 2]


def roll_pitch(asset: Entity) -> tuple[torch.Tensor, torch.Tensor]:
    """机体相对重力的横滚、俯仰（ZYX 欧拉角，弧度）。"""
    g = _unit_gravity_b(asset)
    return torch.atan2(-g[:, 1], -g[:, 2]), torch.asin(torch.clamp(g[:, 0], -1.0, 1.0))


def _trunk_z(env: ManagerBasedRlEnv, asset: Entity) -> torch.Tensor:
    return torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)


def _smoothstep(t: torch.Tensor) -> torch.Tensor:
    return t * t * (3.0 - 2.0 * t)


def _tilt_gate(asset: Entity, tilt_full_deg: float, tilt_zero_deg: float) -> torch.Tensor:
    """倾斜 ≤ tilt_full_deg 时 1、≥ tilt_zero_deg 时 0、中间平滑过渡（上游两处倾斜门的同一个式子）。"""
    tilt_deg = torch.rad2deg(torch.acos(cos_tilt(asset).clamp(-1.0, 1.0)))
    s = torch.clamp((tilt_zero_deg - tilt_deg) / max(tilt_zero_deg - tilt_full_deg, 1e-6), 0.0, 1.0)
    return _smoothstep(s)


def gravity_fallen_mask(env: ManagerBasedRlEnv, asset: Entity, gate_z_below: float,
                        gate_tilt_above_deg: float) -> torch.Tensor:
    """上游 `_fallen_mask`（第 552 行）：躯干低于 gate_z_below 或倾斜超过 gate_tilt_above_deg 记 1.0。"""
    fallen = (_trunk_z(env, asset) < gate_z_below) | (cos_tilt(asset) < math.cos(math.radians(gate_tilt_above_deg)))
    return fallen.float()


@contextlib.contextmanager
def _upstream_fallen_mask_by_gravity():
    upstream = microduck_mdp._fallen_mask
    microduck_mdp._fallen_mask = gravity_fallen_mask
    try:
        yield
    finally:
        microduck_mdp._fallen_mask = upstream


def fallen_by_gravity(func: Callable) -> Callable:
    """包一层：调用 func 期间，上游模块里的 `_fallen_mask` 换成按重力算的 `gravity_fallen_mask`。"""
    @functools.wraps(func)
    def wrapped(env, **params):
        with _upstream_fallen_mask_by_gravity():
            return func(env, **params)
    wrapped.fallen_mask_by_gravity = True
    return wrapped


def body_upright_linear(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT, gate_z_below: float | None = None,
                        gate_tilt_above_deg: float = 40.0) -> torch.Tensor:
    """上游第 825 行：cos(倾斜)，站直 +1、躺平 0、倒立 −1；给了 gate_z_below 就只在倒地时算。"""
    asset = env.scene[asset_cfg.name]
    reward = cos_tilt(asset)
    if gate_z_below is not None:
        reward = reward * gravity_fallen_mask(env, asset, gate_z_below, gate_tilt_above_deg)
    return reward


def upright_gaussian_at_height(env: ManagerBasedRlEnv, std: float, height_low: float, height_high: float,
                               asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """上游第 876 行：exp(−(1 − cos 倾斜)/std²)，乘躯干高度的平滑门。"""
    asset = env.scene[asset_cfg.name]
    tilt_sq = 1.0 - cos_tilt(asset)
    upright_g = torch.exp(-tilt_sq / (std * std))
    t = torch.clamp((_trunk_z(env, asset) - height_low) / max(height_high - height_low, 1e-6), 0.0, 1.0)
    return upright_g * _smoothstep(t)


def body_ang_vel_at_height(env: ManagerBasedRlEnv, height_low: float, height_high: float,
                           asset_cfg: SceneEntityCfg = _ROBOT, tilt_full_deg: float | None = None,
                           tilt_zero_deg: float = 45.0) -> torch.Tensor:
    """上游第 905 行：躯干 sum(ω_xy²)，乘高度门与（可选的）倾斜门。"""
    asset = env.scene[asset_cfg.name]
    ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :].squeeze(1)
    cost = torch.sum(torch.square(ang_vel[:, :2]), dim=1)
    t = torch.clamp((_trunk_z(env, asset) - height_low) / max(height_high - height_low, 1e-6), 0.0, 1.0)
    gate = _smoothstep(t)
    if tilt_full_deg is not None:
        gate = gate * _tilt_gate(asset, tilt_full_deg, tilt_zero_deg)
    return cost * gate


def standing_composite_score(env: ManagerBasedRlEnv, target_height: float, height_std: float, upright_std: float,
                             pose_std: float, joint_indices: list, target_overrides: dict | None = None,
                             asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """上游第 952 行：高度、直立、关节姿态三个高斯相乘。"""
    asset = env.scene[asset_cfg.name]
    height_score = torch.exp(-((_trunk_z(env, asset) - target_height) / height_std) ** 2)
    upright_score = torch.exp(-(1.0 - cos_tilt(asset)) / (upright_std * upright_std))
    target = microduck_mdp._servo_default_joint_pos(env, asset).clone()
    if target_overrides:
        for idx, val in target_overrides.items():
            target[:, idx] = val
    joint_pos = microduck_mdp._servo_joint_pos(env, asset)[:, joint_indices]
    pose_err_sq = ((joint_pos - target[:, joint_indices]) ** 2).mean(dim=-1)
    pose_score = torch.exp(-pose_err_sq / (pose_std * pose_std))
    return height_score * upright_score * pose_score


def head_pose_bias_penalty(env: ManagerBasedRlEnv, command_name: str = "head_pose", tau_s: float = 1.0,
                           gate_height_low: float | None = None, gate_height_high: float = 0.11,
                           gate_tilt_full_deg: float = 20.0, gate_tilt_zero_deg: float = 45.0,
                           asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """上游第 5579 行：头颈跟踪误差的时间平均（EMA）取负；给了 gate_height_low 就乘站高、站直门。"""
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    if not hasattr(env, "_head_pose_neck_ids"):
        microduck_mdp.head_pose_tracking(env, command_name=command_name, asset_cfg=asset_cfg)
    neck_ids = env._head_pose_neck_ids
    joint_pos = asset.data.joint_pos
    measured = joint_pos[:, neck_ids] + joint_pos[:, env._head_pose_bl_ids] * env._head_pose_bl_mask
    err = (measured - asset.data.default_joint_pos[:, neck_ids]) - cmd
    if gate_height_low is not None:
        t = torch.clamp((_trunk_z(env, asset) - gate_height_low) / max(gate_height_high - gate_height_low, 1e-6),
                        0.0, 1.0)
        gate = _smoothstep(t) * _tilt_gate(asset, gate_tilt_full_deg, gate_tilt_zero_deg)
        err = err * gate.unsqueeze(-1)
    else:
        gate = None
    if not hasattr(env, "_head_bias_ema"):
        env._head_bias_ema = torch.zeros_like(err)
    env._head_bias_ema[env.episode_length_buf <= 1] = 0.0
    alpha = min(1.0, float(env.step_dt) / max(tau_s, 1e-6))
    env._head_bias_ema = (1.0 - alpha) * env._head_bias_ema + alpha * err
    out = -env._head_bias_ema.abs().mean(dim=-1)
    if gate is not None:
        out = out * gate
    return out


def body_pose_tracking_locomotion(
        env: ManagerBasedRlEnv, command_name: str = "body_pose", nominal_height: float = 0.105,
        xy_std: float = 0.02, z_std: float = 0.03, angle_std: float = math.radians(30),
        axis_weights: tuple[float, float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        vel_gate_command_name: str | None = None, vel_gate_std: float = 0.1,
        standing_gate_command_name: str | None = None, asset_cfg: SceneEntityCfg = _ROBOT,
        feet_cfg: SceneEntityCfg = _FEET) -> torch.Tensor:
    """上游第 5755 行：6 维身体姿态跟踪（各轴高斯的加权平均）；横滚、俯仰相对重力，其余照上游。"""
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    dx, dy, dz = cmd[:, 0], cmd[:, 1], cmd[:, 2]
    droll, dpitch, dyaw = cmd[:, 3], cmd[:, 4], cmd[:, 5]
    pos_w = asset.data.root_link_pos_w
    quat = asset.data.root_link_quat_w
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    trunk_yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    roll, pitch = roll_pitch(asset)
    foot_pos = asset.data.site_pos_w[:, feet_cfg.site_ids]
    foot_quat = asset.data.site_quat_w[:, feet_cfg.site_ids]
    feet_centroid = foot_pos.mean(dim=1)
    dx_w = pos_w[:, 0] - feet_centroid[:, 0]
    dy_w = pos_w[:, 1] - feet_centroid[:, 1]
    cos_y, sin_y = torch.cos(trunk_yaw), torch.sin(trunk_yaw)
    x_body = cos_y * dx_w + sin_y * dy_w
    y_body = -sin_y * dx_w + cos_y * dy_w
    z_world = torch.nan_to_num(pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)
    fqw, fqx, fqy, fqz = foot_quat[..., 0], foot_quat[..., 1], foot_quat[..., 2], foot_quat[..., 3]
    foot_yaws = torch.atan2(2.0 * (fqw * fqz + fqx * fqy), 1.0 - 2.0 * (fqy * fqy + fqz * fqz))
    mean_foot_yaw = torch.atan2(torch.sin(foot_yaws).mean(dim=1), torch.cos(foot_yaws).mean(dim=1))
    r_x = torch.exp(-((x_body - dx) / xy_std) ** 2)
    r_y = torch.exp(-((y_body - dy) / xy_std) ** 2)
    r_z = torch.exp(-((z_world - (nominal_height + dz)) / z_std) ** 2)
    r_r = torch.exp(-((roll - droll) / angle_std) ** 2)
    r_p = torch.exp(-((pitch - dpitch) / angle_std) ** 2)
    r_w = torch.exp(-(wrap_to_pi(trunk_yaw - mean_foot_yaw - dyaw) / angle_std) ** 2)
    wx, wy, wz, wr, wp, wyaw = axis_weights
    total_w = wx + wy + wz + wr + wp + wyaw
    reward = (wx * r_x + wy * r_y + wz * r_z + wr * r_r + wp * r_p + wyaw * r_w) / max(total_w, 1e-6)
    if vel_gate_command_name is not None:
        vel_cmd = env.command_manager.get_command(vel_gate_command_name)
        reward = reward * torch.exp(-(torch.linalg.vector_norm(vel_cmd[:, :2], dim=-1) / vel_gate_std) ** 2)
    if standing_gate_command_name is not None:
        standing = getattr(env.command_manager.get_term(standing_gate_command_name), "is_standing_env", None)
        reward = reward * (standing.float() if standing is not None else 0.0)
    return reward


def upright_progress(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """上游第 591 行：cos(倾斜) 每步的增量（势函数塑形）。"""
    c = torch.nan_to_num(cos_tilt(env.scene[asset_cfg.name]), nan=1.0)
    if not hasattr(env, "_upright_potential_prev"):
        env._upright_potential_prev = c.clone()
    fresh = env.episode_length_buf <= 1
    env._upright_potential_prev[fresh] = c[fresh]
    delta = c - env._upright_potential_prev
    env._upright_potential_prev = c.clone()
    return delta


def fallen_state_penalty(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT,
                         gate_tilt_above_deg: float = 40.0, release_tilt_below_deg: float | None = None,
                         release_z_above: float | None = None) -> torch.Tensor:
    """上游第 649 行：倒地时每步 1.0；给了 release_* 就一直扣到真站起来（带滞回）。"""
    asset = env.scene[asset_cfg.name]
    fallen = gravity_fallen_mask(env, asset, 0.0, gate_tilt_above_deg).bool()
    if release_tilt_below_deg is None:
        return fallen.float()
    up = cos_tilt(asset) > math.cos(math.radians(release_tilt_below_deg))
    if release_z_above is not None:
        up &= _trunk_z(env, asset) > release_z_above
    if not hasattr(env, "_fallen_tax_armed"):
        env._fallen_tax_armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._fallen_tax_armed[env.episode_length_buf <= 1] = False
    env._fallen_tax_armed |= fallen
    env._fallen_tax_armed &= ~up
    return env._fallen_tax_armed.float()


def recovery_success(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT, fallen_tilt_deg: float = 40.0,
                     min_fallen_s: float = 0.5, up_tilt_deg: float = 25.0, up_z: float = 0.105) -> torch.Tensor:
    """上游第 691 行：倒地满 min_fallen_s 以后真站起来的那一帧记 1.0（一次性，带滞回）。"""
    asset = env.scene[asset_cfg.name]
    c = cos_tilt(asset)
    fallen = c < math.cos(math.radians(fallen_tilt_deg))
    up = (c > math.cos(math.radians(up_tilt_deg))) & (_trunk_z(env, asset) > up_z)
    if not hasattr(env, "_recovery_fallen_s"):
        env._recovery_fallen_s = torch.zeros(env.num_envs, device=env.device)
        env._recovery_armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    fresh = env.episode_length_buf <= 1
    env._recovery_fallen_s[fresh] = 0.0
    env._recovery_armed[fresh] = False
    env._recovery_fallen_s = torch.where(fallen, env._recovery_fallen_s + env.step_dt,
                                         torch.zeros_like(env._recovery_fallen_s))
    env._recovery_armed |= env._recovery_fallen_s >= min_fallen_s
    fired = env._recovery_armed & up
    env._recovery_armed &= ~fired
    return fired.float()


# 上游函数体里自己从四元数算倾斜的 → 这里同样算、只换倾斜的版本。
REIMPLEMENTED: dict[Callable, Callable] = {
    microduck_mdp.body_upright_linear: body_upright_linear,
    microduck_mdp.upright_gaussian_at_height: upright_gaussian_at_height,
    microduck_mdp.body_ang_vel_at_height: body_ang_vel_at_height,
    microduck_mdp.standing_composite_score: standing_composite_score,
    microduck_mdp.head_pose_bias_penalty: head_pose_bias_penalty,
    microduck_mdp.body_pose_tracking_locomotion: body_pose_tracking_locomotion,
    microduck_mdp.upright_progress: upright_progress,
    microduck_mdp.fallen_state_penalty: fallen_state_penalty,
    microduck_mdp.recovery_success: recovery_success,
}
# 经上游 `_fallen_mask` 判倒地的 → 包一层 `fallen_by_gravity`。
GATED_BY_FALLEN_MASK: tuple[Callable, ...] = (
    microduck_mdp.feet_air_time_upright,
    microduck_mdp.action_rate_l2_fallen_scaled,
    microduck_mdp.joint_torque_rate_l2_fallen_scaled,
    microduck_mdp.com_upward_velocity,
    microduck_mdp.fallen_too_long,
)


def use_gravity_up(cfg) -> dict[str, Callable]:
    """把配置里按世界 z 轴算直立、倒地的奖励与终止项换成按重力算的（文件头），换新的项对象、不原地改。

    Args:
        cfg: mjlab 的环境配置（原地改它的 rewards、terminations 两张表）。

    Returns:
        换掉的项名 → 上游原来的函数（测试拿它和替换后的比）。
    """
    replaced = {}
    for section in (cfg.rewards, cfg.terminations):
        for name, term in list(section.items()):
            if term.func in REIMPLEMENTED:
                func = REIMPLEMENTED[term.func]
            elif term.func in GATED_BY_FALLEN_MASK:
                func = fallen_by_gravity(term.func)
            else:
                continue
            replaced[name] = term.func
            section[name] = dataclasses.replace(term, func=func)
    return replaced
