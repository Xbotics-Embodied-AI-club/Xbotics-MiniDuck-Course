"""小鸭子各个动作的仿真环境：上游的任务，我们这版硬件的物理。

整个 training/ 里**唯一与机器人有关**的文件。它做三件事：

1. 从 mjlab 的任务注册表取出上游的任务（奖励、观测、域随机化、BAM 舵机模型都在那里）；
2. 把任务里的机器人模型换成所选硬件版本的那一份（`hardware.Hardware.model_dir` 下），舵机换成
   这一版的电机与供电（`servos.build_actuators`）—— 我们自己的硬件；
3. 建好环境后，核对躯干质量与质心、舵机模型与刚度、每个环境抽到的舵机电压，
   确认仿真里跑的确实是我们的硬件。

第 3 步不是多余的。换模型靠的是替换任务配置里的一个函数引用，上游哪天把模型的
加载方式改了，替换就会**静默失效**：训练照跑、曲线照涨，训出来的却是上游旧硬件的策略。
核对失败就停下，不要让这件事有机会发生。

训哪个动作由 `SKILLS` 表声明，入口用环境变量 `MINIDUCK_SKILL` 选（见 `skill_from_env`）；
在哪个硬件版本上训由训练入口的 `MINIDUCK_HARDWARE` 选（见 `hardware.py`），导出与评测跟着权重里记的那一版走。
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import logging
import math
import os
import re
import warnings
from pathlib import Path
from typing import Any

# Task registration prints upstream patch banners; keep failures visible, not successful patch notices.
with contextlib.redirect_stdout(io.StringIO()) as _registration_output:
    import mjlab_microduck.tasks  # noqa: F401
import mujoco
import numpy as np
import torch
import warp as wp
from mjlab.envs import mdp as mjlab_mdp
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.envs.mdp import dr
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.robot.microduck_constants import name_servo_collision_geoms
from mjlab_microduck.tasks import mdp as microduck_mdp

from training.sim import getup, servos, terrain, upright
from training.sim.hardware import Hardware

log = logging.getLogger(__name__)
for _line in _registration_output.getvalue().splitlines():
    if _line.startswith("[mdp] Patch") and " active:" in _line:
        log.debug("%s", _line)
    else:
        log.warning("%s", _line)
del _registration_output


@dataclasses.dataclass(frozen=True)
class Skill:
    """一个动作：上游的哪个任务、换成本仓的哪份模型、训多少轮。

    Attributes:
        name: 本仓里的叫法，也是 `MINIDUCK_SKILL` 的取值与导出目录名。
        task_id: 上游 mjlab 注册表里的任务名。
        robot_xml: 硬件版本机体结构目录（`Hardware.model_dir`）下的模型文件名，与上游这个任务读的那份同名。
        upstream_spec_fn: 上游这个任务原本用来读模型的函数名。建环境时核对：
            上游换了读法（换了文件、加了后处理），这里当场报错，而不是继续用旧的换法。
        iterations: 训练轮数，取上游实际部署那份策略的量级（见 README 的动作表）。
        rename_servo_collisions: 上游读完模型后还要给舵机外壳的碰撞体起名
            （全碰撞模型系列，接触传感器按名字找它们），这里照做。
        warm_start: 上游这个任务的配置是按「在别的策略上热启动」写的（继承来的课程表
            直接从最终档起），从零训就不是原版的训练方法，所以必须给热启动权重。
        slope: 在坡上训：每个环境的重力各自斜过去，坡度按 `terrain.SlopeSchedule` 取（见 terrain.py）。
        bumpy: 在起伏地面上训（`terrain.GROUNDS[bumpy_ground]`），求解器迭代数照上游崎岖地形的设置。
        vin_range_v: 没有线束压降模型的旧硬件可用动作自己的等效弱舵机范围；有线束模型时按硬件声明。
        bumpy_ground: 起伏地面用 `terrain.GROUNDS` 里的哪一份；只在 `bumpy` 时有用。
        slope_schedule: 坡度怎么取，`terrain.SCHEDULES` 里的哪一份：ramp = 按步数线性抬高上限，adaptive = 每个环境按
            上一个坡上回合的成败升降（自适应版里坡上的环境摔倒就终止，顶推与躺着开局只留给平地环境，见 terrain.py）。
        weak_getup: 起身在弱舵机下训（要配 `vin_range_v`）：critic 看舵机力气、奖励记离开仰躺的进度
            （`getup.configure`），还躺在背上的帧不学站立老师（`distill.free_supine`）。
        reward_weights: 改上游奖励表里几项的权重，（项名, 权重）；空 = 全照上游。用 `with_reward_weights` 改
            （命令行入口读 `MINIDUCK_REWARD_WEIGHTS`），课上「改一项奖励权重、训一个自己的版本」用它，不用去改装好的上游包。
    """

    name: str
    task_id: str
    robot_xml: str
    upstream_spec_fn: str
    iterations: int
    rename_servo_collisions: bool = False
    warm_start: bool = False
    slope: bool = False
    bumpy: bool = False
    vin_range_v: tuple[float, float] | None = None
    bumpy_ground: str = "box"
    slope_schedule: str = "ramp"
    weak_getup: bool = False
    reward_weights: tuple[tuple[str, float], ...] = ()

    def with_reward_weights(self, **weights: float) -> Skill:
        """同一个动作，改几项奖励的权重，例：`SKILLS["walk"].with_reward_weights(air_time=0.0)`。

        项名是上游奖励表里的名字，建环境时核对（写错了报错并列出有哪些）；再调一次会把之前改的覆盖掉。

        Returns:
            新的动作声明。
        """
        return dataclasses.replace(self, reward_weights=tuple((name, float(w)) for name, w in weights.items()))

    def vin_override(self, hardware: Hardware) -> tuple[float, float] | None:
        """带线束压降的硬件用声明的稳压范围，不叠加早期按供电模拟的弱舵机工况。"""
        return self.vin_range_v if hardware.supply["load_sag_v_per_nm"] is None else None


SKILLS = {s.name: s for s in (
    # 走路：上游部署的那份取自第 3750 轮，playbook 给步态类 4000–6000 轮。
    Skill("walk", "Mjlab-Velocity-Flat-MicroDuck", "robot_walk.xml", "get_walk_spec", 6000),
    # 课程从零学平地直走，不把起身、转圈和摆头也塞进同一个短训练。
    Skill("course-walk", "Mjlab-Velocity-Flat-MicroDuck", "robot_walk.xml", "get_walk_spec", 6000),
    Skill("course-walk-elective", "Mjlab-Velocity-Flat-MicroDuck", "robot_walk.xml", "get_walk_spec", 1500),
    # 站起：上游的站立老师（alpha_stand）取自第 9750 轮。上游配方从零训学不会从趴/仰姿势翻身
    # 所以从 alpha_stand.onnx 续训 2000 轮：
    # MINIDUCK_INIT_POLICY=<alpha_stand.onnx> MINIDUCK_INIT_ITERATION=9750。
    Skill("standup", "Mjlab-StandUp-Flat-MicroDuck", "robot_groundcontact.xml",
          "get_standup_spec", 2000),
    # 同一个站起任务，换成全碰撞模型（每个零件、含舵机外壳都能碰地，同 velstand 与倒地循环用的那份）。
    # 在精简碰撞模型上训出的起身，放进全碰撞的倒地循环里都不如上一版起身。
    Skill("standup-full", "Mjlab-StandUp-Flat-MicroDuck", "robot_allcollisions.xml",
          "get_standup_spec", 2000, rename_servo_collisions=True),
    # 坐↔站：上游先训 1250 轮、再续训到 2250 轮（加坐姿扰动，按全局步数排程）。
    Skill("sitstand", "Mjlab-SitStand-Flat-MicroDuck", "robot_groundcontact.xml",
          "get_standup_spec", 2750),
    # 走路 + 摔倒恢复：在走路上热启动，平地 6000 轮，再在崎岖地面 + 齿隙上续训。
    Skill("velstand-flat", "Mjlab-VelStand-Flat-MicroDuck", "robot_allcollisions.xml",
          "get_allcollisions_spec", 6000, rename_servo_collisions=True, warm_start=True),
    Skill("velstand-rough-backlash", "Mjlab-VelStand-Rough-Backlash-MicroDuck",
          "robot_allcollisions_backlash.xml", "get_allcollisions_backlash_spec", 4000,
          rename_servo_collisions=True, warm_start=True),
    # 坡地版：同一个任务放到斜坡（0–15°、坡向任意）与起伏地面上，在平地版上热启动。
    # 不用上游的崎岖地形：它的金字塔坡与台阶有净高差，按出生点高度判摔倒的项在下坡时误判（见 terrain.py）。
    Skill("velstand-slope", "Mjlab-VelStand-Flat-MicroDuck", "robot_allcollisions.xml",
          "get_allcollisions_spec", 1500, rename_servo_collisions=True, warm_start=True, slope=True, bumpy=True),
    # 坡地最终版的配方：纯平面 + 斜重力；上坡 / 下坡 / 横坡各有一个难度级别、各自按成败升降，
    # 20% 的上下坡回合直接取陡坡，一半坡上回合整个回合往前走并记前进进度（terrain.SCHEDULES["directional-go"]）。
    # 一路试过的其余配方（时间斜坡、共用级别的自适应、分方向各步）的参数仍在 terrain.SCHEDULES 里，要复跑就加一行动作。
    Skill("velstand-slope-dirg", "Mjlab-VelStand-Flat-MicroDuck", "robot_allcollisions.xml",
          "get_allcollisions_spec", 1500, rename_servo_collisions=True, warm_start=True, slope=True,
          slope_schedule="directional-go"),
    # 起身与零指令站立的坡地版：固件站着不动时用的就是起身策略，坡上站不住同样要它学。
    # 舵机电压在 3.9–6.1 V 里随机（BAM 里力矩、转速正比于电压 ≈ 标称的 0.65–1.0 倍）：现行起身要满力气，
    # 舵机弱两成时仰躺第一次只翻起一半，真机舵机比仿真跟得慢；
    # 卡在从背上翻到侧面这一步，按弱舵机重新学这一段（weak_getup，见 getup.py）。
    # 地面是纯平面 + 斜重力、不带起伏：倒地循环与真机都在平地上，起伏地面里的方块平地与纯平面接触不同（terrain.py）。
    Skill("standup-slope", "Mjlab-StandUp-Flat-MicroDuck", "robot_allcollisions.xml",
          "get_standup_spec", 1000, rename_servo_collisions=True, slope=True, vin_range_v=(3.9, 6.1),
          weak_getup=True),
    # 下面三个是一次性的动作（从站姿开局，做完回到站姿），上游都从零训。轮数 = 课程表最后一档的轮数
    # + 上游 playbook（AGENTS.md）给「一次性动作」的约 1000 轮，让最后一档的权重与随机化练够。
    # 低头用嘴尖贴近地面再站回去：课程表到第 2000 轮（质心随机化放到最大）。
    Skill("ground-pick", "Mjlab-GroundPick-Flat-MicroDuck", "robot_groundcontact.xml",
          "get_ground_pick_spec", 3000),
    # 右脚把 70 mm / 15 g 的球往前踢（策略看不到球，靠开局位置）：课程表到第 1500 轮。
    Skill("ball-kick", "Mjlab-BallKick-Flat-MicroDuck", "robot_groundcontact.xml",
          "get_standup_spec", 2500),
    # 前滚翻：从头顶滚过去、落回脚上。课程表到第 6000 轮（中途开局的比例最后一次下调）；
    # 上游打开了左右对称的镜像损失（train_ppo 照做）。
    Skill("roulade", "Mjlab-Roulade-Flat-MicroDuck", "robot_groundcontact.xml",
          "get_standup_spec", 7000),
)}

# 每个环境独立抽取各关节的标零偏差；偏差同时作用于编码器读数和电机位置目标。
# 学员肉眼对齐机械零位；六个动作统一用 ±5°。
ENCODER_ZERO_ERROR_RAD = math.radians(5)

# 照上游：README 的训练命令是 4096 个环境；种子是 runner 配置的默认值。
NUM_ENVS = int(os.environ.get("MINIDUCK_NUM_ENVS") or 4096)
SEED = 42


def skill_from_env() -> Skill:
    """入口用：从 `MINIDUCK_SKILL` 取这一次跑哪个动作；给了 `MINIDUCK_REWARD_WEIGHTS` 就改那几项奖励的权重。

    Returns:
        动作声明。

    Raises:
        RuntimeError: 没给，或给的不在 `SKILLS` 里。
    """
    name = os.environ.get("MINIDUCK_SKILL", "").strip()
    if name not in SKILLS:
        raise RuntimeError(f"MINIDUCK_SKILL={name!r} 不对，可选：{', '.join(SKILLS)}")
    return SKILLS[name].with_reward_weights(**reward_weights_from_env())


def reward_weights_from_env() -> dict[str, float]:
    """读 `MINIDUCK_REWARD_WEIGHTS`：逗号分开的「项名=权重」，例如 `air_time=0,upright=2.5`；没给就是空。

    项名是上游奖励表里的名字（建环境时核对，写错了报错并列出有哪些）。

    Returns:
        项名 → 权重。

    Raises:
        RuntimeError: 某一项不是「项名=数」的写法。
    """
    text = os.environ.get("MINIDUCK_REWARD_WEIGHTS", "").strip()
    if not text:
        return {}
    weights = {}
    for item in text.split(","):
        name, _, value = item.partition("=")
        try:
            weight = float(value)          # 没有「=」时 value 是空串，同样走这里报错
        except ValueError:
            weight = None
        if weight is None or not name.strip():
            raise RuntimeError(f"MINIDUCK_REWARD_WEIGHTS 里的 {item!r} 不是「项名=权重」，例：air_time=0")
        weights[name.strip()] = weight
    return weights


def max_iterations(skill: Skill) -> int:
    """这一次训多少轮：默认取动作声明里的轮数，冒烟测试时用 `MINIDUCK_MAX_ITERATIONS` 压小。

    Args:
        skill: 动作声明。

    Returns:
        轮数。
    """
    return int(os.environ.get("MINIDUCK_MAX_ITERATIONS") or skill.iterations)


# 上游实体配置的碰撞筛选（FULL_COLLISION）只留名字以 `_collision` 结尾的碰撞体，其余有名字的一律关掉、不报错。
# 我们给模型加的电池盒与电池支架碰撞体在 XML 里叫 `*_col`，读完改成上游认的名字；不改的话训练里电池盒穿地，
# 仰面躺时身体平贴外壳，学会的翻身到了真机（背着电池盒）上翻不过去。
OUR_COLLISION_GEOMS = {"battery_box_col": "battery_box_collision",
                       "battery_bracket_col": "battery_bracket_collision"}


def name_our_collision_geoms(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """把本仓模型自己加的碰撞体改成上游碰撞筛选认的名字。

    Args:
        spec: 刚读进来的模型。

    Returns:
        同一个 spec。

    Raises:
        RuntimeError: 模型里缺了其中某个碰撞体（模型改过名或删过，这张表要跟着改）。
    """
    names = {g.name for g in spec.geoms}
    missing = sorted(OUR_COLLISION_GEOMS.keys() - names)
    if missing:
        raise RuntimeError(f"模型里没有碰撞体 {missing}：OUR_COLLISION_GEOMS 与模型对不上")
    for old, new in OUR_COLLISION_GEOMS.items():
        spec.geom(old).name = new
    return spec


def _axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """绕单位轴转 angle 弧度的四元数 (w, x, y, z)。"""
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def lock_joints(spec: mujoco.MjSpec, locked: dict[str, float]) -> mujoco.MjSpec:
    """把不通电、锁死的关节从模型里删掉：子刚体按锁定角度焊在父刚体上，驱动它的执行器一起删。

    铰链关节转 q 时，子刚体在父刚体系里的位姿是「先按刚体自身的 pos/quat 放好，再绕关节轴
    （过关节点）转 q」。把这一转并进刚体的 pos/quat，删掉关节，刚体就固定在 q 处了。

    Args:
        spec: 刚读进来的模型。
        locked: 关节名 → 锁定角度（弧度），来自硬件版本的 `locked_joints`。

    Returns:
        同一个 spec。

    Raises:
        RuntimeError: 模型里没有某个关节，或它不是铰链。
    """
    for name, angle in locked.items():
        joint = spec.joint(name)
        if joint is None:
            raise RuntimeError(f"硬件版本要锁的关节 {name!r} 不在模型里")
        if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
            raise RuntimeError(f"关节 {name!r} 不是铰链，锁不了")
        body = joint.parent
        turn = _axis_angle_quat(np.asarray(joint.axis, dtype=float), angle)
        rot_body, rot_turn = np.zeros(9), np.zeros(9)
        mujoco.mju_quat2Mat(rot_body, np.asarray(body.quat, dtype=float))
        mujoco.mju_quat2Mat(rot_turn, turn)
        jpos = np.asarray(joint.pos, dtype=float)
        body.pos = np.asarray(body.pos, dtype=float) + rot_body.reshape(3, 3) @ (jpos - rot_turn.reshape(3, 3) @ jpos)
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, np.asarray(body.quat, dtype=float), turn)
        body.quat = quat
        for actuator in [a for a in spec.actuators if a.target == name]:
            spec.delete(actuator)
        spec.delete(joint)
    return spec


def robot_spec_fn(skill: Skill, hardware: Hardware):
    """读本仓模型的函数，代替上游任务配置里的 `spec_fn`。

    Args:
        skill: 动作声明。
        hardware: 硬件版本（模型取它的机体结构；声明了锁死的关节就在这里删掉）。

    Returns:
        无参函数，返回本仓模型的 MjSpec（本仓加的碰撞体已改名；需要时也给舵机碰撞体起好名）。
    """
    path = hardware.model_dir / skill.robot_xml

    def load() -> mujoco.MjSpec:
        spec = name_our_collision_geoms(mujoco.MjSpec.from_file(str(path)))
        spec = lock_joints(spec, hardware.locked_joints)
        return name_servo_collision_geoms(spec) if skill.rename_servo_collisions else spec

    return load


def check_collisions_kept(spec_fn, mj_model: mujoco.MjModel) -> None:
    """核对：模型里写了参与碰撞的有名字的几何体，到训练环境里一个都没被关掉。

    上游碰撞筛选按名字决定留谁，名字不合它的规矩就静默关掉。这里拿同一个 spec 单独编译一份
    （碰撞属性照 XML），与环境实际用的模型逐个比。

    Args:
        spec_fn: 读模型的无参函数（`robot_spec_fn` 的返回值）。
        mj_model: 环境编译出来的模型。

    Raises:
        RuntimeError: 有几何体在 XML 里参与碰撞、训练里却被关掉了。
    """
    declared = spec_fn().compile()
    used = {}
    for g in range(mj_model.ngeom):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if name:
            used[name.split("/")[-1]] = bool(mj_model.geom_contype[g] or mj_model.geom_conaffinity[g])
    dropped = []
    for g in range(declared.ngeom):
        name = mujoco.mj_id2name(declared, mujoco.mjtObj.mjOBJ_GEOM, g)
        if name and (declared.geom_contype[g] or declared.geom_conaffinity[g]) and not used.get(name, False):
            dropped.append(name)
    if dropped:
        raise RuntimeError(
            f"这些碰撞体在模型里参与碰撞，训练环境里却被关掉了：{dropped}。上游碰撞筛选只留 *_collision —— "
            "改名（OUR_COLLISION_GEOMS）或核对上游的实体配置")


def skill_env_cfg(skill: Skill, hardware: Hardware):
    """取上游这个任务的环境配置，把机器人换成这个硬件版本的模型、舵机换成它的电机与供电。

    模型只替换 `spec_fn`：关节、碰撞选择器都写在上游的实体配置里，按名字匹配到模型上。
    我们的模型改了躯干的质量与电池几何；电池的两个碰撞体名字不合上游碰撞筛选，读模型时改名
    （`name_our_collision_geoms`），建环境时 `check_collisions_kept` 核对没有碰撞体被静默关掉。

    Args:
        skill: 动作声明。
        hardware: 硬件版本。

    Returns:
        mjlab 的环境配置。

    Raises:
        RuntimeError: 上游这个任务读模型的函数已不是声明里的那个；或 `skill.reward_weights` 里有奖励表没有的项。
    """
    cfg = load_env_cfg(skill.task_id)
    # 每个上游任务都有这一项随机化；哪天哪个任务没有了，直接 KeyError，不静默跳过
    cfg.events["encoder_bias"].params["bias_range"] = (-ENCODER_ZERO_ERROR_RAD, ENCODER_ZERO_ERROR_RAD)
    robot = cfg.scene.entities["robot"]
    if robot.spec_fn.__name__ != skill.upstream_spec_fn:
        raise RuntimeError(
            f"{skill.task_id} 读模型的函数是 {robot.spec_fn.__name__}，声明里写的是 "
            f"{skill.upstream_spec_fn} —— 上游改了，先核对它读哪份模型、有没有后处理，再改 SKILLS")
    # 上游的执行器配置是模块级共享对象，换成新对象而不原地改。
    articulation = dataclasses.replace(
        robot.articulation, actuators=servos.build_actuators(robot.articulation.actuators, hardware,
                                                             skill.vin_override(hardware)))
    spec_fn = robot_spec_fn(skill, hardware)
    init_state = robot.init_state
    if hardware.locked_joints:
        if skill.name not in LOCKED_JOINT_SKILLS:
            raise RuntimeError(f"锁死关节的硬件版本 {hardware.id} 只验证过 {LOCKED_JOINT_SKILLS}，"
                               f"{skill.name} 的任务怎么跟着改还没定")
        init_state = _init_state_without(init_state, hardware.locked_joints, spec_fn())
        _drop_head_pose_command(cfg, hardware.locked_joints)
        _drop_head_terms(cfg, hardware.locked_joints)
        _remap_servo_index_params(cfg, servo_joints(mujoco.MjSpec.from_file(str(hardware.model_dir / skill.robot_xml))),
                                  hardware.locked_joints)
        if skill.name in TRUNK_FLOOR_SKILLS:
            cfg.terminations["trunk_too_low"] = TerminationTermCfg(
                func=mjlab_mdp.root_height_below_minimum, params={"minimum_height": LOCKED_MIN_TRUNK_HEIGHT})
    cfg.scene.entities["robot"] = dataclasses.replace(robot, spec_fn=spec_fn, articulation=articulation,
                                                      init_state=init_state)
    if upstream_v8():
        _apply_upstream_v8_spawns(cfg, skill)
    if skill.slope:
        schedule = terrain.SCHEDULES[skill.slope_schedule]
        slope_event = EventTermCfg(func=terrain.reset_slopes, mode="reset", params={"schedule": schedule})
        if isinstance(schedule, terrain.AdaptiveSlope | terrain.DirectionalSlope):
            _walk_only_on_slopes(cfg, slope_event, schedule)
        else:
            cfg.events["slope"] = slope_event
    if skill.bumpy:
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = terrain.GROUNDS[skill.bumpy_ground]
        # 同上游崎岖地形（microduck_velocity_env_cfg 的 rough 分支）：多个零件同时碰起伏时 10 次迭代解不开接触。
        cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 200)
        cfg.sim.mujoco.iterations = 30
        cfg.sim.mujoco.ls_iterations = 50
    if skill.slope:
        # 斜重力下世界 z 轴是坡面法线：直立类奖励与倒地判定改按真实重力方向算（upright.py）
        upright.use_gravity_up(cfg)
    if skill.weak_getup:
        if skill.vin_range_v is None:
            raise RuntimeError(f"{skill.name}：weak_getup 要在舵机电压随机（vin_range_v）的训练里用")
        getup.configure(cfg, nominal_v=sum(hardware.supply["vin_range_v"]) / 2)
    if skill.name in ("course-walk", "course-walk-elective"):
        _configure_course_walk(cfg)
    if skill.name == "course-walk-elective":
        cfg.events["reset_robot_joints"].func = _course_elective_reset_joints
    for name, weight in skill.reward_weights:
        if name not in cfg.rewards:
            raise RuntimeError(f"{skill.task_id} 的奖励表里没有 {name!r}，有：{', '.join(cfg.rewards)}")
        cfg.rewards[name].weight = weight
    return cfg


_COURSE_TRUNK_FLOOR = 0.075
_COURSE_STANDING_HEIGHT = 0.116


def _course_trunk_height(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Give a dense upright-height signal without rewarding jumps above HOME."""
    z = env.scene["robot"].data.root_link_pos_w[:, 2]
    return ((torch.nan_to_num(z, nan=0.0) - _COURSE_TRUNK_FLOOR)
            / (_COURSE_STANDING_HEIGHT - _COURSE_TRUNK_FLOOR)).clamp_(0.0, 1.0)


def _course_air_time(env: ManagerBasedRlEnv, sensor_name: str, threshold_min: float,
                     threshold_max: float, command_name: str, command_threshold: float) -> torch.Tensor:
    """Reward short swing on a planted support foot, not hopping or holding a foot up."""
    feet = env.scene[sensor_name].data
    command = env.command_manager.get_command(command_name)
    assert command is not None
    assert feet.found is not None
    assert feet.current_air_time is not None
    t = feet.current_air_time
    single_support = (feet.found > 0).sum(dim=1) == 1
    swing = ((t > threshold_min) & (t < threshold_max)).any(dim=1)
    return (single_support & swing & (command[:, 0] > command_threshold)).float()



class _CourseYawDrift:
    """Price sustained yaw drift without taxing the cyclic motion of a stride."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        del cfg
        self._yaw_rate = torch.zeros(env.num_envs, device=env.device)
        # Same time-averaged L1 shaping as the upstream head-pose bias term.
        self._alpha = env.step_dt

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._yaw_rate[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        yaw_rate = env.scene["robot"].data.root_link_ang_vel_w[:, 2]
        self._yaw_rate.lerp_(torch.nan_to_num(yaw_rate, nan=0.0, posinf=0.0, neginf=0.0), self._alpha)
        return self._yaw_rate.abs().clamp_max_(1.0)


def _course_encoder_bias_at_reset(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    bias_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Sample a fresh calibration after the opening phase, keeping it fixed within the episode."""
    if env.common_step_counter > 12_000:
        dr.encoder_bias(env, env_ids, bias_range=bias_range, asset_cfg=asset_cfg)


def _course_elective_reset_joints(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Practise correcting small HOME pose errors after basic stepping is learned."""
    if env.common_step_counter > 12_000:
        position_range = (-0.05, 0.05)
    mjlab_mdp.reset_joints_by_offset(
        env, env_ids, position_range=position_range, velocity_range=velocity_range, asset_cfg=asset_cfg,
    )


def _configure_course_walk(cfg) -> None:
    """Make forward steps discoverable before adding balance disturbances."""
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (0.18, 0.40)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    twist.rel_turn_in_place_envs = twist.rel_standing_envs = 0.0
    # Preserve the observation layout for the same ActorCritic and export path.
    for name in ("head_pose", "body_pose"):
        pose = cfg.commands[name]
        pose.ranges = ((0.0, 0.0),) * len(pose.ranges)
        del cfg.curriculum[f"{name}_range"]
    del cfg.curriculum["head_pose_bias_weight"], cfg.rewards["head_pose_bias"]
    cfg.rewards["head_pose_tracking"].weight = 0.5

    # A crouch at zero speed otherwise collects most of the posture reward and
    # a large share of the velocity reward without having to balance on one foot.
    tracking = cfg.rewards["track_linear_velocity"]
    tracking.weight = 5.0
    tracking.params["std"] = 0.20
    # Keep upstream instantaneous angular tracking; its velocity Gaussian
    # alone cannot distinguish periodic yaw from a persistent turning bias.
    cfg.rewards["yaw_drift"] = RewardTermCfg(func=_CourseYawDrift, weight=-2.0)
    cfg.curriculum["yaw_drift_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={"reward_name": "yaw_drift", "weight_stages": [
            {"step": 0, "weight": -2.0}, {"step": 6_000, "weight": -4.0},
        ]})
    cfg.rewards["trunk_height"] = RewardTermCfg(func=_course_trunk_height, weight=3.0)
    cfg.terminations["trunk_too_low"] = TerminationTermCfg(
        func=mjlab_mdp.root_height_below_minimum, params={"minimum_height": _COURSE_TRUNK_FLOOR})
    cfg.rewards["air_time"].func = _course_air_time
    cfg.rewards["air_time"].params["threshold_min"] = 0.06

    # These clocks are environment steps; the three milestones are 500, 1000 and 1250 rollout rounds.
    cfg.curriculum["standing_envs"].params["standing_stages"] = [
        {"step": 0, "rel_standing_envs": 0.0}, {"step": 12_000, "rel_standing_envs": 0.05},
    ]
    cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
        {"step": 0, "weight": -0.1}, {"step": 12_000, "weight": -0.2}, {"step": 24_000, "weight": -0.3},
    ]
    # Train more combinations of encoder calibration and CoM without widening either range.
    cfg.events["episode_encoder_bias"] = EventTermCfg(
        func=_course_encoder_bias_at_reset,
        mode="reset",
        params=cfg.events["encoder_bias"].params.copy(),
    )
    # Keep the upstream per-reset CoM curriculum, without its wide startup event.
    # Start the assay's ±10 mm stage at 750 rounds so the short notebook trains
    # on it for half its budget, without changing the opening or final ranges.
    del cfg.events["base_com"]
    for name in ("com_range", "head_com_range"):
        for stage in cfg.curriculum[name].params["range_stages"]:
            if stage["step"] == 24_000:
                stage["step"] = 18_000
    push_stages = [
        {"step": step, "velocity_range": {"x": (-limit, limit), "y": (-limit, limit)}}
        for step, limit in ((0, 0.0), (12_000, 0.05), (24_000, 0.10), (30_000, 0.15))
    ]
    cfg.events["push_robot"].params["velocity_range"] = push_stages[0]["velocity_range"]
    cfg.curriculum["push_velocity"] = CurriculumTermCfg(
        func=microduck_mdp.push_curriculum, params={"event_name": "push_robot", "push_stages": push_stages})


def _walk_only_on_slopes(cfg, slope_event: EventTermCfg,
                         schedule: terrain.AdaptiveSlope | terrain.DirectionalSlope) -> None:
    """自适应坡度版的配置：坡上的环境只走路，摔倒恢复那一套（顶推、躺着开局、摔倒不终止）只留给平地环境。

    - 坡上的环境躯干倾斜过 70° 就终止回合（`terrain.fell_on_slope`）：摔倒后不再躺 8 秒才重置；
    - 摔倒用的大顶推（topple_push）与躺着开局（random_prone_init）只作用在平地环境（`terrain.flat_only`），
      `slope_pushes=False` 时常规顶推（push_robot）也只给平地环境；
    - 坡事件排在所有重置事件最前面（方位相对朝向的版本排在「重置机器人位姿」之后）：后面的事件按这一轮新抽的坡判断哪些是平地环境；
    - 课程项 `slope_level_deg` 把平均坡度级别记进训练日志；
    - `forward_fraction` > 0 的分坡向课程：一部分坡上回合的指令整个回合都是往前走（`terrain.SlopeForwardCommand`），
      `progress_weight` > 0 时这些回合另加一项前进进度奖励（`terrain.slope_progress`）。
    """
    flat_only_events = ("topple_push", "random_prone_init") + (() if schedule.slope_pushes else ("push_robot",))
    for name in flat_only_events:
        term = cfg.events[name]
        cfg.events[name] = dataclasses.replace(term, func=terrain.flat_only(term.func))
    if isinstance(schedule, terrain.DirectionalSlope):   # 坡的方位相对出生朝向取，要排在「重置机器人位姿」之后
        events = {}
        for name, term in cfg.events.items():
            events[name] = term
            if name == "reset_base":
                events["slope"] = slope_event
        cfg.events = events
    else:
        cfg.events = {"slope": slope_event, **cfg.events}
    cfg.terminations["fell_on_slope"] = TerminationTermCfg(
        func=terrain.fell_on_slope, time_out=False, params={"limit_angle": math.radians(70.0)})
    if isinstance(schedule, terrain.DirectionalSlope) and schedule.slope_track_std is not None:
        track = cfg.rewards["track_linear_velocity"]
        cfg.rewards["track_linear_velocity"] = dataclasses.replace(
            track, func=terrain.track_linear_velocity_slope,
            params={**track.params, "slope_std": schedule.slope_track_std})
    if isinstance(schedule, terrain.DirectionalSlope) and schedule.forward_fraction > 0:
        cfg.commands["twist"] = terrain.forward_command_cfg(cfg.commands["twist"], schedule)
    if isinstance(schedule, terrain.DirectionalSlope) and schedule.progress_weight > 0:
        cfg.rewards["slope_progress"] = RewardTermCfg(
            func=terrain.slope_progress, weight=schedule.progress_weight, params={"command_name": "twist"})
    if isinstance(schedule, terrain.DirectionalSlope):
        for klass, name in ((terrain.UPHILL, "up"), (terrain.DOWNHILL, "down"), (terrain.CROSS, "cross")):
            cfg.curriculum[f"slope_level_{name}_deg"] = CurriculumTermCfg(
                func=terrain.class_level_deg, params={"klass": klass, "schedule": schedule})
    else:
        cfg.curriculum["slope_level_deg"] = CurriculumTermCfg(
            func=terrain.mean_slope_deg, params={"schedule": schedule})


def upstream_v8() -> bool:
    """`MINIDUCK_UPSTREAM_V8=1`：用上游 v8（improve_standup_velstand 分支 2ee6dc8）的起身改进训练。

    我们锁定的上游 ceca017 里没有这个提交。它两件事：站立老师在倾斜过 75° 时看不到关节（`distill.masked_teacher`），
    以及更宽的倒地开局（这里）。缺省关，旧的训练照旧。
    """
    return os.environ.get("MINIDUCK_UPSTREAM_V8", "").strip() == "1"


# 上游 v8 的更宽倒地开局（microduck_velstand_env_cfg.py 的 ENABLE_WIDE_SPAWNS 一节）：倒地那一份开局里，
# 一半把舵机关节（含头）在行程中间 90% 里均匀随机，四分之一身体朝向在三维里均匀随机（关节一律随机、多留 4 cm），
# 另加 5% 的坐姿开局。上游实测：v7 在随机关节的仰躺开局卡住 26%，原因就是只练过站姿关节的 ±90° 开局。
V8_JOINT_RANDOM_PROB = 0.5
V8_JOINT_RANGE_FRAC = 0.9
V8_JOINT_RANDOM_EXTRA_Z = 0.06
V8_SO3_PROB = 0.25
V8_SO3_EXTRA_Z = 0.04
V8_SIT_PROB = 0.05
V8_SIT_PARAMS = {
    "sitting_joint_overrides": {1: 0.0, 2: -0.4079, 3: 1.35, 4: 0.0, 10: 0.0, 11: 0.4079, 12: -1.35, 13: 0.0},
    "sitting_joint_noise_std": 0.15,
    "sitting_tilt_max": math.radians(20),
    "sitting_z_min": 0.06,
    "sitting_z_max": 0.09,
}


def _uniform_random_quat(n: int, device) -> torch.Tensor:
    """均匀随机的单位四元数 (w, x, y, z)，Shoemake 法（照上游 v8）。"""
    a, b, c = torch.rand(n, 3, device=device).unbind(1)
    return torch.stack([
        torch.sqrt(1 - a) * torch.sin(2 * math.pi * b), torch.sqrt(1 - a) * torch.cos(2 * math.pi * b),
        torch.sqrt(a) * torch.sin(2 * math.pi * c), torch.sqrt(a) * torch.cos(2 * math.pi * c),
    ], dim=1)


def _scatter_lying(env, lying: torch.Tensor, joint_random_prob: float, joint_range_frac: float,
                   joint_random_extra_z: float, so3_prob: float, so3_extra_z: float) -> None:
    """倒地开局里一部分转成三维随机朝向、一部分关节随机（照上游 v8 `maybe_set_random_prone_orientation`）。"""
    if len(lying) == 0:
        return
    asset_cfg = SceneEntityCfg("robot")
    so3 = torch.rand(len(lying), device=env.device) < so3_prob
    if so3.any():
        ids = lying[so3]
        env.sim.data.qpos[ids, 3:7] = _uniform_random_quat(len(ids), env.device)
        env.sim.data.qpos[ids, 2] += so3_extra_z
    jr = lying[(torch.rand(len(lying), device=env.device) < joint_random_prob) | so3]
    if len(jr) > 0:
        microduck_mdp.randomize_servo_joints_uniform(env, jr, asset_cfg=asset_cfg, range_frac=joint_range_frac)
        env.sim.data.qpos[jr, 2] += joint_random_extra_z


def prone_orientation_v8(env, env_ids, asset_cfg=None, prone_prob: float = 0.0, face_down_prob: float = 0.5,
                         prone_z_min: float = 0.20, prone_z_max: float = 0.25, crouch_prob: float = 0.0,
                         side_prob: float = 0.0, joint_random_prob: float = 0.0, joint_range_frac: float = 0.8,
                         joint_random_extra_z: float = 0.06, so3_prob: float = 0.0, so3_extra_z: float = 0.04,
                         sit_prob: float = 0.0, sit_params: dict | None = None):
    """上游 v8 的 `maybe_set_random_prone_orientation`（ceca017 版本再加三维随机朝向与坐姿两片），逐行照搬。"""
    asset_cfg = asset_cfg or SceneEntityCfg("robot")
    if prone_prob <= 0.0 and crouch_prob <= 0.0 and sit_prob <= 0.0:
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if len(env_ids) == 0:
        return
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    u = torch.rand(len(env_ids), device=env.device)
    selected = env_ids[u < prone_prob]
    crouch_selected = env_ids[(u >= prone_prob) & (u < prone_prob + crouch_prob)]
    sit_selected = env_ids[(u >= prone_prob + crouch_prob) & (u < prone_prob + crouch_prob + sit_prob)]
    if len(selected) > 0:
        microduck_mdp.set_random_prone_orientation(env, selected, asset_cfg=asset_cfg, face_down_prob=face_down_prob,
                                                   side_prob=side_prob)
        z = torch.rand(len(selected), device=env.device) * (prone_z_max - prone_z_min) + prone_z_min
        env.sim.data.qpos[selected, 2] = z + microduck_mdp._env_origin_z(env, selected)
        _scatter_lying(env, selected, joint_random_prob, joint_range_frac, joint_random_extra_z, so3_prob, so3_extra_z)
    if len(crouch_selected) > 0:
        microduck_mdp.set_random_crouch_state(env, crouch_selected, asset_cfg=asset_cfg)
    if len(sit_selected) > 0:
        microduck_mdp.set_random_ground_state(env, sit_selected, asset_cfg=asset_cfg, face_down_prob=0.0,
                                              face_up_prob=0.0, sitting_prob=1.0, standing_prob=0.0,
                                              **(sit_params or {}))


def ground_state_v8(env, env_ids, upstream, **params):
    """起身任务的开局（趴 / 仰 / 坐 / 站），倒地的那些照上游 v8 的比例放宽：一半关节随机、四分之一三维随机朝向。

    起身任务在上游没有这一步；它与 velstand 用同一个起身动作（velstand 的起身就是照站立老师克隆的），
    倒地开局放宽到同样的比例，起身练到的姿势才盖得住推倒后真实落地的那些。
    """
    upstream(env, env_ids, **params)
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    q = env.sim.data.qpos[env_ids, 3:7]
    up = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)   # 机体 z 轴的竖直分量：趴、仰约 0，坐、站约 1
    _scatter_lying(env, env_ids[up < 0.5], V8_JOINT_RANDOM_PROB, V8_JOINT_RANGE_FRAC, V8_JOINT_RANDOM_EXTRA_Z,
                   V8_SO3_PROB, V8_SO3_EXTRA_Z)


def _apply_upstream_v8_spawns(cfg, skill: Skill) -> None:
    """按动作接上 v8 的更宽倒地开局：velstand 换开局函数，起身在原开局之后再放宽。"""
    if "random_prone_init" in cfg.events:
        term = cfg.events["random_prone_init"]
        cfg.events["random_prone_init"] = dataclasses.replace(term, func=prone_orientation_v8, params={
            **term.params, "joint_random_prob": V8_JOINT_RANDOM_PROB, "joint_range_frac": V8_JOINT_RANGE_FRAC,
            "joint_random_extra_z": V8_JOINT_RANDOM_EXTRA_Z, "so3_prob": V8_SO3_PROB, "so3_extra_z": V8_SO3_EXTRA_Z,
            "sit_prob": V8_SIT_PROB, "sit_params": V8_SIT_PARAMS})
    elif skill.task_id == SKILLS["standup"].task_id:
        term = cfg.events["set_ground_state"]
        cfg.events["set_ground_state"] = dataclasses.replace(term, func=ground_state_v8,
                                                             params={**term.params, "upstream": term.func})
    else:
        raise RuntimeError(f"MINIDUCK_UPSTREAM_V8 只用在 velstand 与起身上，{skill.name} 没有倒地开局")

# 上游的头部姿态指令（4 维，相对站姿的偏移）依次对应这四个关节，奖励按这个顺序逐个比（mdp.head_pose_tracking）。
HEAD_POSE_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")

# 锁死关节的硬件版本能训的动作（建环境核对过的）。velstand 系列每轮还要对 14 关节的老师做行为克隆，
# 老师的维数对不上，没放进来。
LOCKED_JOINT_SKILLS = ("walk", "sitstand", "standup")
# 加躯干高度下限的动作：只有走路。起身、坐站从躺着 / 坐着开局，躯干本来就低。
TRUNK_FLOOR_SKILLS = ("walk",)

# 头部锁死后走路的躯干高度下限（米，平地）。上游走路任务没有高度约束，14 关节时靠头部姿态指令间接撑着：
# 要大幅点头、转头，跪下去头会碰地。头一锁，这层约束没了，策略从第 250 轮起就跪着走（躯干 5 cm，
# 踝关节折到 ±95°）。HOME 站姿 11.6 cm、14 关节走路 11.9 cm，下限取 9 cm。
LOCKED_MIN_TRUNK_HEIGHT = 0.09


def _init_state_without(init_state, locked: dict[str, float], spec: mujoco.MjSpec):
    """初始姿态里去掉只指向锁死关节的条目：模型里已经没有这些关节，留着会匹配不到。

    Args:
        init_state: 上游实体配置的初始状态。
        locked: 锁死的关节。
        spec: 锁死之后的模型（剩下哪些关节以它为准）。

    Returns:
        新的初始状态。
    """
    remaining = [j.name for j in spec.joints]
    joint_pos = {}
    for pattern, value in init_state.joint_pos.items():
        hits_locked = any(re.fullmatch(pattern, name) for name in locked)
        hits_remaining = any(re.fullmatch(pattern, name) for name in remaining)
        if hits_remaining or not hits_locked:
            joint_pos[pattern] = value
    return dataclasses.replace(init_state, joint_pos=joint_pos)


def _drop_head_pose_command(cfg, locked: dict[str, float]) -> None:
    """头部有关节锁死后，任务里「跟着头部姿态指令摆头」的指令、观测、奖励、课程表一起去掉。

    指令是 4 维、按四个头部关节逐个比（上游奖励函数写死了四个），锁掉其中任何一个就对不上了。
    只锁一部分时，剩下的头部关节不再跟指令，由策略自己用（比如脖子俯仰调重心）；上游的站姿奖励
    本来就不管头部关节，所以它们只受动作变化率那一项约束。

    Args:
        cfg: mjlab 的环境配置（原地改）。
        locked: 锁死的关节。
    """
    if not set(HEAD_POSE_JOINTS) & set(locked) or "head_pose" not in cfg.commands:
        return
    del cfg.commands["head_pose"]
    events = [n for n, t in cfg.events.items() if (t.params or {}).get("command_name") == "head_pose"]
    for name in events:
        del cfg.events[name]
    for name in [n for n, t in cfg.curriculum.items() if (t.params or {}).get("event_name") in events]:
        del cfg.curriculum[name]
    for group in cfg.observations.values():
        for name in [n for n, t in group.terms.items() if (t.params or {}).get("command_name") == "head_pose"]:
            del group.terms[name]
    dropped = [n for n, t in cfg.rewards.items() if (t.params or {}).get("command_name") == "head_pose"]
    for name in dropped:
        del cfg.rewards[name]
    for name in [n for n, t in cfg.curriculum.items()
                 if (t.params or {}).get("command_name") == "head_pose" or (t.params or {}).get("reward_name") in dropped]:
        del cfg.curriculum[name]


def check_trunk_matches_table(mj_model: mujoco.MjModel, mass_table: Path) -> None:
    """核对仿真里躯干的质量与质心，必须等于质量属性表。

    表是 SolidWorks 报告的原样数字（单位 kg 与 mm），这里按表里声明的长度单位换算后比较。

    Args:
        mj_model: 环境编译出来的 MuJoCo 模型（标称值，不含域随机化）。
        mass_table: 这个硬件版本的躯干质量属性表。

    Raises:
        RuntimeError: 找不到躯干，或质量 / 质心与表对不上。
    """
    table = json.loads(mass_table.read_text(encoding="utf-8"))
    want = table["bodies"]["trunk_base"]
    length_scale = float(table["input"]["length_scale"])
    names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
             for i in range(mj_model.nbody)]
    hits = [i for i, name in enumerate(names) if name.split("/")[-1] == "trunk_base"]
    if len(hits) != 1:
        raise RuntimeError(f"仿真模型里应当恰好有一个 trunk_base，实际找到 {len(hits)} 个")
    body = hits[0]
    mass = float(mj_model.body_mass[body])
    ipos = np.asarray(mj_model.body_ipos[body], dtype=float)
    want_ipos = np.asarray(want["ipos"], dtype=float) * length_scale
    if abs(mass - float(want["mass"])) > 1e-9 or not np.allclose(ipos, want_ipos, atol=1e-9):
        raise RuntimeError(
            f"仿真里的躯干与质量属性表对不上：质量 {mass} vs {want['mass']}，"
            f"质心 {ipos.tolist()} vs {want_ipos.tolist()}。"
            "多半是机器人模型没换成本仓的 sim_model/ —— 训下去就是在训别的硬件。"
        )


def check_bodies_match_table(mj_model: mujoco.MjModel, bodies_table: Path) -> None:
    """结构目录里有逐刚体的质量属性表（mass/bodies.json，2026-11-05 起）时，核对每个刚体的质量与质心。

    只核躯干时，别的刚体的 inertial 被改了也照样通过；有了全表就逐个比。没有这张表的旧结构跳过。

    Args:
        mj_model: 环境编译出来的 MuJoCo 模型。
        bodies_table: 逐刚体质量属性表（SI，刚体系）。

    Raises:
        RuntimeError: 表里的刚体在模型里找不到，或质量 / 质心对不上。
    """
    if not bodies_table.is_file():
        return
    table = json.loads(bodies_table.read_text(encoding="utf-8"))["bodies"]
    names = {(mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i) or "").split("/")[-1]: i
             for i in range(mj_model.nbody)}
    wrong = []
    for body, want in table.items():
        if body not in names:
            wrong.append(f"{body}（模型里没有）")
            continue
        i = names[body]
        if (abs(float(mj_model.body_mass[i]) - want["mass"]) > 1e-8
                or not np.allclose(mj_model.body_ipos[i], want["ipos"], atol=1e-8)):
            wrong.append(body)
    if wrong:
        raise RuntimeError(f"这些刚体与 {bodies_table.name} 对不上：{wrong} —— 模型不是这个硬件版本的那份")


# 上游按头部四个关节的名字逐个取的奖励（mdp._NECK_JOINT_PATTERNS），锁掉任何一个就取不全：
# 后仰时脖子受力的惩罚（坐站任务）。
HEAD_TERM_FUNCS = ("neck_torque_when_tilted",)
# 上游动头部指令的事件（参数里不写指令名，函数里默认取 head_pose）：坐着时突然甩头（坐站任务）。
HEAD_EVENT_FUNCS = ("seated_head_command_jerk",)
# 上游写成「14 舵机下标」的参数（mdp._servo_joint_ids 的说明）：下标列表与以下标为键的目标角表。
SERVO_INDEX_LISTS = ("joint_indices",)
SERVO_INDEX_DICTS = ("sit_overrides", "sitting_joint_overrides", "target_overrides")
# 以整数为键、但不是关节下标的参数：质心偏移的 x / y / z 范围。
NOT_SERVO_INDEX = ("ranges",)


def servo_joints(spec: mujoco.MjSpec) -> list[str]:
    """模型里由舵机驱动的关节名，按模型顺序（= 上游 14 舵机下标的顺序，passive_* 不算）。"""
    return [j.name for j in spec.joints
            if j.type == mujoco.mjtJoint.mjJNT_HINGE and not j.name.startswith("passive_")]


def _drop_head_terms(cfg, locked: dict[str, float]) -> None:
    """头部有关节锁死后，去掉按头部关节取值的奖励、动头部指令的事件，并关掉复合奖励里的头部因子。

    Args:
        cfg: mjlab 的环境配置（原地改）。
        locked: 锁死的关节。
    """
    if not set(HEAD_POSE_JOINTS) & set(locked):
        return
    for name in [n for n, t in cfg.rewards.items() if getattr(t.func, "__name__", "") in HEAD_TERM_FUNCS]:
        del cfg.rewards[name]
    events = [n for n, t in cfg.events.items() if getattr(t.func, "__name__", "") in HEAD_EVENT_FUNCS]
    for name in events:
        del cfg.events[name]
    for name in [n for n, t in cfg.curriculum.items() if (t.params or {}).get("event_name") in events]:
        del cfg.curriculum[name]
    for term in cfg.rewards.values():
        # 坐稳的复合评分里「头在指令位」那个因子（mdp.posture_composite 的 head_std）：指令已经没了。
        if "head_std" in (term.params or {}):
            term.params["head_std"] = None


def _remap_servo_index_params(cfg, full: list[str], locked: dict[str, float]) -> None:
    """把上游按 14 舵机下标写的参数换算到锁死之后的关节顺序，锁死的关节从里面去掉。

    其余以小整数为键或为值的参数一律报错：说不清它是不是关节下标，就不让它带着错位的下标训下去
    （下标对不上时多半不报错，只是罚错了关节）。

    Args:
        cfg: mjlab 的环境配置（原地改）。
        full: 锁死之前的舵机关节顺序。
        locked: 锁死的关节。

    Raises:
        RuntimeError: 有没认领的、看起来像关节下标的参数。
    """
    remaining = [j for j in full if j not in locked]
    new_index = {i: remaining.index(name) for i, name in enumerate(full) if name in remaining}
    unknown = []
    for section in ("rewards", "events", "terminations", "curriculum"):
        for term_name, term in getattr(cfg, section).items():
            params = term.params or {}
            for key, value in list(params.items()):
                if key in SERVO_INDEX_LISTS:
                    params[key] = type(value)(new_index[i] for i in value if i in new_index)
                elif key in SERVO_INDEX_DICTS and value:
                    params[key] = {new_index[i]: v for i, v in value.items() if i in new_index}
                elif key not in NOT_SERVO_INDEX and (
                        (isinstance(value, dict) and value and all(isinstance(k, int) for k in value))
                        or (isinstance(value, (list, tuple)) and len(value) > 1
                            and all(isinstance(v, int) and not isinstance(v, bool) for v in value)
                            and max(value) < len(full))):
                    unknown.append(f"{section}.{term_name}.{key}")
    if unknown:
        raise RuntimeError(f"这些参数像是按 14 舵机下标写的，没有换算规则：{unknown} —— 认清后加进 SERVO_INDEX_*")


def check_locked_joints(env: ManagerBasedRlEnv, hardware: Hardware, skill: Skill) -> None:
    """核对：锁死的关节在环境里确实没了，动作维数 = 剩下的关节数，头部姿态指令跟着去掉了。

    Args:
        env: 建好的 mjlab 环境。
        hardware: 硬件版本。
        skill: 动作声明。

    Raises:
        RuntimeError: 任何一项不符。
    """
    robot = env.scene["robot"]
    still_there = sorted(set(hardware.locked_joints) & set(robot.joint_names))
    if still_there:
        raise RuntimeError(f"声明锁死的关节 {still_there} 还在环境里 —— 模型没换成锁死后的那份")
    action_dim = int(env.single_action_space.shape[0])
    driven = [j for j in robot.joint_names if not j.startswith("passive_")]
    if action_dim != len(driven):
        raise RuntimeError(f"动作 {action_dim} 维，舵机驱动的关节有 {len(driven)} 个：{driven}")
    if hardware.locked_joints and "head_pose" in env.command_manager.active_terms:
        raise RuntimeError("头部关节锁死了，头部姿态指令却还在")
    if (hardware.locked_joints and skill.name in TRUNK_FLOOR_SKILLS
            and "trunk_too_low" not in env.termination_manager.active_terms):
        raise RuntimeError("关节锁死的版本没有躯干高度下限，策略会学成跪着走（LOCKED_MIN_TRUNK_HEIGHT）")


def split_actor_critic_obs(obs):
    """把 mjlab 按组返回的观测拆成 actor 那份和 critic 那份。

    critic 只在训练时存在，可以多看仿真里的「天眼」信息（基座真实线速度、脚的接触力等）；
    actor 要带到真机上，只看得到真机拿得到的。

    Args:
        obs: mjlab 返回的观测字典，键为观测组名。

    Returns:
        (actor 观测, critic 观测) 两个张量。
    """
    return obs["actor"], obs["critic"]


class DuckEnv:
    """一个动作的并行仿真环境。

    所有动作共用一份观测与动作约定：actor 观测 61 维（48 维本体感觉 + 13 维指令，
    某个任务用不到的指令槽填零），动作 14 维（14 个舵机的目标角）。硬件版本锁死了关节时，
    每锁一个关节少一维位置、一维速度、一维动作；头部四个全锁时再少 4 维头部指令（锁死头部是 45 维观测、10 维动作）。
    """

    def __init__(self, skill: Skill, hardware: Hardware, num_envs=NUM_ENVS, device="cuda:0", seed=SEED,
                 render=False, view: dict | None = None):
        """建环境。

        Args:
            skill: 动作声明。
            hardware: 硬件版本（`hardware.load` / `hardware.from_env` / `hardware.of_training_settings`）。
            num_envs: 并行环境数。
            device: 仿真所在设备。要 GPU 而机器上没有时报错，不静默落到 CPU ——
                MuJoCo Warp 在 CPU 上会慢两个数量级，而进度条照样在动。
            seed: 随机种子。
            render: 是否开渲染（只有录像要）。开了就用一套固定的跟拍机位。
            view: 在固定机位之上再改的录像设置（展示录像用，见 `showcase.py`）：mjlab `ViewerConfig`
                的字段名 → 值（`origin_type` 给枚举名字符串），外加 `env_spacing`（并行环境之间隔多少米）。
        """
        if device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(
                f"要求 device={device!r} 但 CUDA 不可用。不做静默降级 —— 确实要用 CPU 就传 device=\"cpu\"。"
            )
        self.skill = skill
        self.hardware = hardware
        cfg = skill_env_cfg(skill, hardware)
        cfg.scene.num_envs = num_envs
        cfg.seed = seed
        if render:
            # mjlab 默认 320×240、相机 5 米外近乎水平：鸭子只有几十像素。
            cfg.viewer.width, cfg.viewer.height = 1280, 960
            cfg.viewer.distance = 0.9
            cfg.viewer.elevation = -15.0
            cfg.viewer.origin_type = cfg.viewer.OriginType.ASSET_ROOT
            cfg.viewer.entity_name = "robot"
            # 「25 厘米的机器人 + 无边地面」让阴影贴图精度不够，渲出横条纹。
            cfg.viewer.enable_shadows = False
            cfg.viewer.enable_reflections = False
            for command in cfg.commands.values():
                if hasattr(command, "debug_vis"):
                    command.debug_vis = False
            for key, value in (view or {}).items():
                if key == "env_spacing":
                    cfg.scene.env_spacing = value
                elif key == "origin_type":
                    cfg.viewer.origin_type = cfg.viewer.OriginType[value]
                elif hasattr(cfg.viewer, key):
                    setattr(cfg.viewer, key, value)
                else:
                    raise ValueError(f"录像设置里没有 {key!r} 这一项（mjlab ViewerConfig 的字段或 env_spacing）")

        # Warp 初始化时打的版本横幅与每个内核的加载提示，用它自己的开关关掉。
        # mjlab 导入时按 MJLAB_WARP_QUIET 设过一次这个开关，这里在它之后，以这里为准。
        wp.config.quiet = True
        # mjlab 建环境时用 print 打十几张管理器表格（各管理器有哪些项），没有开关可关。
        # 收下来记成一条 DEBUG：屏幕上不出现，要查任务配置时调到 DEBUG 还看得到。
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), warnings.catch_warnings():
            # 上游实体配置里执行器的名字规则同时匹配到 7 个 site（imu、两只脚、相机……）：site 没有传动，执行器照样只接 14 个关节，
            # mjlab 却每建一次环境就警告一遍。只在建环境这一步静音这一条。
            warnings.filterwarnings("ignore", message="Actuator config matched .* site")
            self._env = ManagerBasedRlEnv(cfg=cfg, device=device,
                                          render_mode="rgb_array" if render else None)
        log.debug("mjlab 建环境时的输出：\n%s", captured.getvalue())
        check_trunk_matches_table(self._env.sim.mj_model, hardware.mass_table)
        check_bodies_match_table(self._env.sim.mj_model, hardware.structure_dir / "mass" / "bodies.json")
        check_collisions_kept(robot_spec_fn(skill, hardware), self._env.sim.mj_model)
        vin_override = skill.vin_override(hardware)
        servos.check_actuators(self._env, hardware, vin_override)
        check_locked_joints(self._env, hardware, skill)
        log.info("环境就位：%s，%d 个并行环境，硬件 %s，机器人模型 %s（躯干与质量表一致），舵机 %s%s%s",
                 skill.name, num_envs, hardware.id, skill.robot_xml, servos.describe(hardware, vin_override),
                 f"，锁死 {', '.join(hardware.locked_joints)}" if hardware.locked_joints else "",
                 f"，奖励权重改了 {dict(skill.reward_weights)}" if skill.reward_weights else "")
        self.device = torch.device(self._env.device)
        self.num_envs = self._env.num_envs
        self.action_dim = int(self._env.single_action_space.shape[0])
        # 有限时域的任务里「时间到」就是终点，不该拿 critic 的估值去补后面的回报；
        # 无限时域里时间到只是截断。上游的环境包装器按这个开关决定给不给 time_outs。
        self.is_finite_horizon = bool(cfg.is_finite_horizon)
        obs, critic_obs = self.get_observations()
        self.obs_dim = int(obs.shape[1])
        self.critic_obs_dim = int(critic_obs.shape[1])

    @property
    def unwrapped(self) -> Any:
        """底层的 mjlab 环境，导出元数据时要用。

        Returns:
            未经包装的 mjlab 环境。
        """
        return self._env

    @property
    def max_episode_length(self) -> int:
        """一个回合的步数（走路是 20 秒 × 50 赫兹 = 1000 步）。

        Returns:
            回合长度（控制步数）。
        """
        return int(self._env.max_episode_length)

    def randomize_episode_starts(self) -> None:
        """把各环境的回合进度打散到 [0, 回合长度) 里的随机位置。

        上游开训前做这一步：不打散的话 4096 个环境会在同一步一起超时重置，
        每轮采到的数据就成批地集中在回合的同一阶段。
        """
        buf = self._env.episode_length_buf
        self._env.episode_length_buf = torch.randint_like(buf, high=self.max_episode_length)

    def set_curriculum_step(self, step: int) -> None:
        """把课程表进度搬到指定的环境步，评测时与训练到的那一档对齐。

        走路任务的部分惩罚权重与随机化幅度按 `common_step_counter` 分段，新建环境从 0 开始；
        不对齐的话，评测面对的是第 0 档 —— 任务被系统性地测简单了，而且不报错。

        Args:
            step: 环境步数，通常是 `迭代数 × 每轮步数`。
        """
        self._env.common_step_counter = int(step)

    def pin_commands(self, lin_vel_x: float, lin_vel_y: float = 0.0, ang_vel_z: float = 0.0) -> None:
        """把指令钉死：机体系速度取给定值，头部与身体姿态取默认（零偏移），且不再重采样。

        录像要它：指令本来每隔几秒随机重采，头部姿态随课程放大到明显的点头转头，
        速度指令里还有一成多是「原地转圈」—— 那一档不受速度区间约束，改区间钉不住。
        所以直接改写指令对象里的值，并关掉世界系 / 朝向 / 站立三种模式。
        评测打分时不要钉，按训练的分布采。

        ⚠️ 观测是缓存的（mjlab 为了不重复压入延迟缓冲，`compute()` 在同一步内返回同一份），
        钉住的指令**从下一次 `step` 起**才进观测；钉完立刻取观测拿到的还是旧指令。

        Args:
            lin_vel_x: 前进速度（米/秒）。
            lin_vel_y: 侧移速度（米/秒）。
            ang_vel_z: 转向角速度（弧度/秒）。
        """
        commands = self._env.command_manager
        twist = commands.get_term("twist")
        twist.vel_command_b[:] = torch.tensor([lin_vel_x, lin_vel_y, ang_vel_z], device=self.device)
        for flag in ("is_world_env", "is_heading_env", "is_standing_env"):
            getattr(twist, flag)[:] = False
        # 头部锁死的硬件版本没有 head_pose 指令（`_drop_head_pose_command`）。
        poses = [n for n in ("head_pose", "body_pose") if n in commands.active_terms]
        for name in poses:
            commands.get_term(name)._command[:] = 0.0
        for name in ("twist", *poses):
            commands.get_term(name).time_left[:] = float("inf")

    def reset(self):
        """重置全部并行环境。

        Returns:
            (actor 观测, critic 观测)。
        """
        obs, _ = self._env.reset()
        return split_actor_critic_obs(obs)

    def get_observations(self):
        """取当前观测，不推进仿真。

        Returns:
            (actor 观测, critic 观测)。
        """
        return split_actor_critic_obs(self._env.observation_manager.compute())

    def step(self, actions):
        """推进一个控制周期（50 赫兹，20 毫秒）。

        Args:
            actions: 形状 (N, 14) 的关节目标量。

        Returns:
            (actor 观测, critic 观测, 奖励, done, 附加信息)。附加信息里两个原始信号都保留：
            `time_outs`（无限时域时才有值，PPO 用它给截断的回合补一段自举回报；
            有限时域时全为假）与 `terminated`（评测用它把「摔了」和「时间到了」分开）。
        """
        obs, rewards, terminated, truncated, extras = self._env.step(actions)
        dones = torch.logical_or(terminated, truncated)
        time_outs = torch.zeros_like(truncated) if self.is_finite_horizon else truncated
        actor_obs, critic_obs = split_actor_critic_obs(obs)
        return actor_obs, critic_obs, rewards, dones, {
            "time_outs": time_outs,
            "terminated": terminated,
            "mjlab_extras": extras,
        }

    def anchor_allowed(self) -> torch.Tensor:
        """哪些环境此刻在平地上（逐环境布尔）；不在坡上训的动作全为真。"""
        return terrain.anchor_allowed(self._env)

    def expert_allowed(self) -> torch.Tensor:
        """哪些环境的舵机接近标称（逐环境布尔，`distill.frame_masks` 的 expert_ok）；没放弱舵机的动作全为真。

        舵机电压每个执行器组各抽一个，取这个环境里最低的那组。接近标称 = 不低于硬件版本声明电压下限的 95%。
        """
        if self.skill.vin_override(self.hardware) is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        vin = torch.stack([a.vin_tensor[:, 0] for a in self._env.scene["robot"]._actuators
                           if getattr(a, "vin_tensor", None) is not None]).amin(dim=0)
        return vin >= 0.95 * self.hardware.supply["vin_range_v"][0]

    def render(self):
        """渲染一帧（只在 `render=True` 时有画面）。

        Returns:
            一帧 RGB 图像。
        """
        return self._env.render()

    def close(self):
        """关闭仿真、释放显存。"""
        self._env.close()
