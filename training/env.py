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
import os
from pathlib import Path
from typing import Any

# 导入即注册：执行 mjlab_microduck 里所有 register_mjlab_task(...)，之后注册表才认得
# 下面的 task id。mjlab 导入时也会按 entry point 自动加载它；这里显式导入，
# 让「任务从哪来」在代码里看得见。
import mjlab_microduck.tasks  # noqa: F401
import mujoco
import numpy as np
import torch
import warp as wp
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.robot.microduck_constants import name_servo_collision_geoms

import servos
from hardware import Hardware

log = logging.getLogger(__name__)


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
    """

    name: str
    task_id: str
    robot_xml: str
    upstream_spec_fn: str
    iterations: int
    rename_servo_collisions: bool = False
    warm_start: bool = False


SKILLS = {s.name: s for s in (
    # 走路：上游部署的那份取自第 3750 轮，playbook 给步态类 4000–6000 轮。
    Skill("walk", "Mjlab-Velocity-Flat-MicroDuck", "robot_walk.xml", "get_walk_spec", 6000),
    # 站起：上游的站立老师（alpha_stand）取自第 9750 轮。上游配方从零训学不会从趴/仰姿势翻身
    # 所以从 alpha_stand.onnx 续训 2000 轮：
    # MINIDUCK_INIT_POLICY=<alpha_stand.onnx> MINIDUCK_INIT_ITERATION=9750。
    Skill("standup", "Mjlab-StandUp-Flat-MicroDuck", "robot_groundcontact.xml",
          "get_standup_spec", 2000),
    # 坐↔站：上游先训 1250 轮、再续训到 2250 轮（加坐姿扰动，按全局步数排程）。
    Skill("sitstand", "Mjlab-SitStand-Flat-MicroDuck", "robot_groundcontact.xml",
          "get_standup_spec", 2750),
    # 走路 + 摔倒恢复：在走路上热启动，平地 6000 轮，再在崎岖地面 + 齿隙上续训。
    Skill("velstand-flat", "Mjlab-VelStand-Flat-MicroDuck", "robot_allcollisions.xml",
          "get_allcollisions_spec", 6000, rename_servo_collisions=True, warm_start=True),
    Skill("velstand-rough-backlash", "Mjlab-VelStand-Rough-Backlash-MicroDuck",
          "robot_allcollisions_backlash.xml", "get_allcollisions_backlash_spec", 4000,
          rename_servo_collisions=True, warm_start=True),
)}

# 照上游：README 的训练命令是 4096 个环境；种子是 runner 配置的默认值。
NUM_ENVS = int(os.environ.get("MINIDUCK_NUM_ENVS") or 4096)
SEED = 42


def skill_from_env() -> Skill:
    """入口用：从 `MINIDUCK_SKILL` 取这一次跑哪个动作。

    Returns:
        动作声明。

    Raises:
        RuntimeError: 没给，或给的不在 `SKILLS` 里。
    """
    name = os.environ.get("MINIDUCK_SKILL", "").strip()
    if name not in SKILLS:
        raise RuntimeError(f"MINIDUCK_SKILL={name!r} 不对，可选：{', '.join(SKILLS)}")
    return SKILLS[name]


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


def robot_spec_fn(skill: Skill, hardware: Hardware):
    """读本仓模型的函数，代替上游任务配置里的 `spec_fn`。

    Args:
        skill: 动作声明。
        hardware: 硬件版本（模型取它的机体结构）。

    Returns:
        无参函数，返回本仓模型的 MjSpec（本仓加的碰撞体已改名；需要时也给舵机碰撞体起好名）。
    """
    path = hardware.model_dir / skill.robot_xml

    def load() -> mujoco.MjSpec:
        spec = name_our_collision_geoms(mujoco.MjSpec.from_file(str(path)))
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
        RuntimeError: 上游这个任务读模型的函数已不是声明里的那个。
    """
    cfg = load_env_cfg(skill.task_id)
    robot = cfg.scene.entities["robot"]
    if robot.spec_fn.__name__ != skill.upstream_spec_fn:
        raise RuntimeError(
            f"{skill.task_id} 读模型的函数是 {robot.spec_fn.__name__}，声明里写的是 "
            f"{skill.upstream_spec_fn} —— 上游改了，先核对它读哪份模型、有没有后处理，再改 SKILLS")
    # 上游的执行器配置是模块级共享对象，换成新对象而不原地改。
    articulation = dataclasses.replace(
        robot.articulation, actuators=servos.build_actuators(robot.articulation.actuators, hardware))
    cfg.scene.entities["robot"] = dataclasses.replace(robot, spec_fn=robot_spec_fn(skill, hardware),
                                                      articulation=articulation)
    return cfg


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
    某个任务用不到的指令槽填零），动作 14 维（14 个舵机的目标角）。
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
        with contextlib.redirect_stdout(captured):
            self._env = ManagerBasedRlEnv(cfg=cfg, device=device,
                                          render_mode="rgb_array" if render else None)
        log.debug("mjlab 建环境时的输出：\n%s", captured.getvalue())
        check_trunk_matches_table(self._env.sim.mj_model, hardware.mass_table)
        check_collisions_kept(robot_spec_fn(skill, hardware), self._env.sim.mj_model)
        servos.check_actuators(self._env, hardware)
        log.info("环境就位：%s，%d 个并行环境，硬件 %s，机器人模型 %s（躯干与质量表一致），舵机 %s",
                 skill.name, num_envs, hardware.id, skill.robot_xml, servos.describe(hardware))
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
        for name in ("head_pose", "body_pose"):
            commands.get_term(name)._command[:] = 0.0
        for name in ("twist", "head_pose", "body_pose"):
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

    def render(self):
        """渲染一帧（只在 `render=True` 时有画面）。

        Returns:
            一帧 RGB 图像。
        """
        return self._env.render()

    def close(self):
        """关闭仿真、释放显存。"""
        self._env.close()
