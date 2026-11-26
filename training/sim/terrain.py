"""坡地与起伏地面：每个并行环境的重力各自斜过去（= 站在一个均匀斜坡上），地面加几毫米到一厘多的起伏。

**坡为什么用斜重力而不用地形生成器的坡。** 上游的崎岖地形里坡是金字塔：出生点在坡顶的平台上，
起身、站立这类不走动的任务一直站在平地上，碰不到坡；而且上游按「躯干高度 − 出生点高度」判摔倒
（倒地门槛 0.10 m，站着约 0.115 m），往下坡走 15 cm 就被当成摔倒：罚分、8 s 后重置，等于教策略别往下坡走。
重力斜过去，地面仍是 z = 0 的平面，按高度判的项照旧成立；物理上与均匀斜坡完全等价（接触、摩擦、重力的
沿坡分量都一样），坡向、坡度每个环境各取各的。

实体数据的 `gravity_vec_w` 这里同步改成每个环境的真实重力方向：观测里的 projected_gravity、翻倒终止
（mjlab `bad_orientation`）、行为克隆的分帧都由它现算，真机的 IMU 量到的就是它，所以策略在坡上看到的
「身体相对重力歪了多少」与真机一致。上游起身与走路任务的几项直立奖励和倒地判定不读它，而是拿根四元数算
机体 z 轴与世界 z 轴（这里是坡面法线）的夹角，坡地动作里由 `upright.use_gravity_up` 换成按重力算的版本。
机器人出生时垂直于坡面（像人把它放在坡上），相对重力已经歪了一个坡角。

**坡度怎么升。** 时间斜坡（`SlopeSchedule`）按环境步把坡度上限线性抬高，不看策略学得怎样：第一版
训到第 370 轮，上限升到 10.8°，摔倒帧占比从 16% 涨到 32%、回合奖励从 62 掉到 2、策略熵从 1.6 涨到 3.5，
难度比学习快。自适应课程（`AdaptiveSlope`）改成每个环境各有一个坡度级别：上一个坡上回合撑满 20 秒就升一级、
摔倒就降一级，数据始终落在「刚好撑得住一半多」的坡度上（Rudin 等 2021，Learning to Walk in Minutes 的地形课程）。
自适应版里坡上的环境走「只走路」的那一套：躯干倾斜过 70° 就终止回合（`fell_on_slope`），顶推与躺着开局只留给平地环境
（`flat_only`）——否则摔倒后躺 8 秒才重置，一次摔倒浪费 400 帧，也没有机会重试最难的起步那一秒。
自适应版的级别各方向共用，被横坡（7° 以上必摔）拉在 4.5°，上下坡练不到 10–15°；分坡向版（`DirectionalSlope`）
把级别拆成上坡 / 下坡 / 横坡三个，各自按这个坡向的回合成败升降，横坡不封顶。

**起伏**用高度场随机起伏，平均高度就是出生点高度，按高度判的项最多差半个起伏幅度（≤ 0.8 cm）。
台阶、金字塔这类有净高差的地形不用：理由同上。
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, fields, replace

import mjlab.terrains as terrain_gen
import torch
import warp as wp
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import rewards as mdp_rewards
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab_microduck.tasks.mdp import (
    VelocityCommandCommandOnly,
    VelocityCommandCommandOnlyCfg,
)

GRAVITY = 9.81
_ROBOT = SceneEntityCfg("robot")
# 坡度在这以下算平地：行为克隆的走路老师只锚定平地上的帧（见 `anchor_allowed`）。
FLAT_BELOW_DEG = 0.5

# 起伏地面：一半平地，一半两档起伏（高度场每 5 cm 一格，起伏的「波长」10–15 cm，与鞋底长度同量级）。
# 上游崎岖地形里随机方格最高 1 cm、台阶 1.5 cm，按「这只鸭子抬脚 1–2 cm」定的；这里取同一量级。
BUMPY_GROUND = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=4,
    num_cols=8,
    curriculum=False,
    sub_terrains={
        "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.5),
        "bumps_low": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25, noise_range=(0.0, 0.008), noise_step=0.002, downsampled_scale=0.10,
            horizontal_scale=0.05, vertical_scale=0.001),
        "bumps": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25, noise_range=(0.0, 0.016), noise_step=0.002, downsampled_scale=0.15,
            horizontal_scale=0.05, vertical_scale=0.001),
    },
    add_lights=False,
)

# 平地验收以纯平面为准（现行走路策略在纯平面上训的，真机平地走得好），而 BUMPY_GROUND 里半数「平地」是方块
# （BoxFlatTerrainCfg：1 m 厚的方块，不是纯平面）。基线实测：同一份现行策略直走 0.2 m/s，
# 偏航漂移（中位数）纯平面 0.04 rad/s、高度场地块 0.05、方块平地块 0.26——方块地面的接触与纯平面不同，
# 在上面训出来的走法可能带偏。这一份把方块平地换成起伏 ≤ 2 mm 的高度场（高度场至少要有一级起伏才建得出来），
# 起伏地块不变。
BUMPY_GROUND_HF = replace(BUMPY_GROUND, sub_terrains={
    **BUMPY_GROUND.sub_terrains,
    "flat": terrain_gen.HfRandomUniformTerrainCfg(
        proportion=0.5, noise_range=(0.0, 0.002), noise_step=0.001, downsampled_scale=0.10,
        horizontal_scale=0.05, vertical_scale=0.001),
})

# `Skill.bumpy_ground` 的取值 → 地形配置。
GROUNDS = {"box": BUMPY_GROUND, "hf": BUMPY_GROUND_HF}


@dataclass(frozen=True)
class SlopeSchedule:
    """训练时坡度怎么取：每次重置，一部分环境平地，其余在 [0, 当前上限] 里均匀取坡度、坡向任意。

    上限从 `start_deg` 按环境步线性升到 `max_deg`：热启动的策略只会平地，一开始就上 15° 会摔成一片，
    摔倒的数据把走路的数据挤掉（上游 velstand 的教训是走路数据少于一半就会退化）。

    Attributes:
        max_deg: 坡度上限（度）。
        start_deg: 开训时的上限（度）。
        ramp_steps: 上限升到 `max_deg` 用的环境步数（= 轮数 × 24）。
        flat_fraction: 每次重置时取平地的比例，守住平地上的走路。
    """

    max_deg: float = 15.0
    start_deg: float = 4.0
    ramp_steps: int = 600 * 24
    flat_fraction: float = 0.2

    def limit_deg(self, step: int) -> float:
        """第 `step` 个环境步时的坡度上限（度）。"""
        frac = min(max(step / self.ramp_steps, 0.0), 1.0) if self.ramp_steps > 0 else 1.0
        return self.start_deg + frac * (self.max_deg - self.start_deg)


@dataclass(frozen=True)
class AdaptiveSlope:
    """训练时坡度怎么取（自适应）：每个环境各有一个级别（0–1，乘 `max_deg` 就是坡度），按上一个坡上回合的结果升降。

    一个回合结束时：时间到了（撑满 20 秒）且至少走了 `min_survive_steps` 步 → 级别升 `up`；
    被终止（摔倒）且至少撑过 `min_fail_steps` 步 → 降 `down`；其余不动。升降比 up : down = 1 : 2，
    撑住的概率落在 2/3 附近时级别不再漂移。只有「按级别取坡」的回合算数，`flat_fraction` 抽到平地的回合不算；
    按级别取坡的回合坡度再小（级别接近 0）也算数——不能按坡度是否小于 `FLAT_BELOW_DEG` 来判，
    否则级别一旦降到 3° 以下，环境就被当成平地、再也升降不了，级别永远出不来。

    Attributes:
        max_deg: 级别 1 对应的坡度（度）。
        start_level: 环境第一次分到坡时的级别。训练条件（常规顶推、随机指令）下，现行策略在 4.5° 的坡上 22 秒内
            摔过的环境近一半（check_adaptive.py），所以从 3° 起步，级别自己往撑得住的地方走。
        up: 撑住一次升多少级。
        down: 摔倒一次降多少级。
        min_survive_steps: 时间到了且回合至少这么长（环境步）才算撑住：开局把回合进度打散，第一个回合可能很短。
        min_fail_steps: 摔得比这还快（环境步）的不算：躺着开局被直接终止不是策略的错。
        max_fail_steps: 摔得比这还晚（环境步）的也不降级别（中立）：回合撑过一半就摔，算「差不多撑住了」，
            同 Rudin 等的做法——走了不到一半的路程才降级，走了一半以上不降也不升。缺省不设（任何时刻摔倒都降）。
        flat_fraction: 每次重置时取平地的比例。
        slope_pushes: 坡上的环境要不要照常受常规顶推（±0.3 m/s，每 3–6 秒一次）。常规顶推本身就让现行策略 20 秒内
            摔倒的概率不小（平地侧推 0.35 m/s 摔 30%），级别会被顶推而不是坡度压在很低的地方；关掉它，
            级别才反映「坡」撑得住撑不住，顶推由平地环境练。
    """

    max_deg: float = 15.0
    start_level: float = 0.2
    up: float = 0.1
    down: float = 0.2
    min_survive_steps: int = 500
    min_fail_steps: int = 25
    max_fail_steps: int = 10**9
    flat_fraction: float = 0.2
    slope_pushes: bool = True


# 坡向：机器人出生朝向相对下坡方向。上坡 = 正对上坡走（下坡在身后），下坡 = 正对下坡走，横坡 = 下坡在左手边或右手边。
UPHILL, DOWNHILL, CROSS, FLAT = range(4)
# 下坡方向相对出生朝向的角度（弧度）：上坡 π，下坡 0，横坡 ±π/2（左右随机，见 `directional_angles`）
RELATIVE_AZIMUTH = (math.pi, 0.0, math.pi / 2)


@dataclass(frozen=True)
class DirectionalSlope:
    """训练时坡度怎么取（分坡向）：上坡、下坡、横坡各有一个难度级别，各自按这个坡向的回合成败升降。

    legged_gym 的地形课程是每种地形各自升级（Rudin 等 2021）；级别各方向共用时，横坡一摔级别就降，
    上下坡永远练不到 10–15°（自适应版的平均级别被横坡压在 4.5°）。这里每次重置按 `mix` 抽一个坡向
    （上坡 / 下坡 / 横坡 / 平地），坡度 = 这个坡向的级别 × `max_deg` × U(`slope_floor`, 1)，下坡方向相对
    机器人出生朝向取（朝向在「重置机器人位姿」之后读，所以坡事件要排在那个事件之后）。

    一个回合在非平地坡向上结束时：时间到了且至少走了 `min_survive_steps` 步 → 这个坡向升 `step`；
    被终止且在 [`min_fail_steps`, `max_fail_steps`) 步内 → 降 `step`；其余不动。更新按回合累加，
    一轮里每个坡向约有十几个回合结束，所以级别每轮最多动 十几 × `step`，不用逐环境各存一份。

    Attributes:
        max_deg: 级别 1 对应的坡度（度），三个坡向相同；横坡不封顶，让它自己按成败升降。
        mix: 每次重置抽到（上坡、下坡、横坡、平地）的概率。
        start_levels: 三个坡向各自的起点级别：上下坡 6°；横坡现行 5° 摔 10%、6° 摔 26%，起点 4.5°。
        step: 每个撑住（早摔）的回合把这个坡向的级别升（降）多少。6e-4：一轮约 17 个同向回合全撑住升 0.01 级
            （0.15°），从 6° 升到 15° 要 60 轮左右，策略来得及跟上；撑住与早摔一样多时级别不动。
        slope_floor: 取坡时坡度的下界比例，让同一级别里坡度有点散。
        min_survive_steps / min_fail_steps / max_fail_steps: 同 `AdaptiveSlope`。
        heading_jitter_deg: 坡向相对朝向的随机偏角（正态标准差，度），免得每个回合都正对 / 正侧。
        slope_pushes: 同 `AdaptiveSlope`。
        slope_track_std: 给了就把坡上环境的线速度跟踪奖励换成更紧的 std（平地环境照旧 0.316）。上坡 5° / 10° 时现行策略的
            换脚频率与平地一样（每秒 4 次）、96% 时间单脚着地，前进速度却从 0.15 掉到 0.06 / −0.03：原地踏步，
            既拿得到抬脚奖励又不冒险，而课程只要求撑满 20 秒，没人逼它往前走。紧一点的 std 让前进多拿的奖励
            大得值得冒险（误差 0.15 m/s 时奖励 0.32 对 0.80）。None = 不换。
        hard_fraction: 上坡、下坡回合里有这么大比例直接取陡坡（`max_deg` 的 `hard_floor`–1 倍），不管级别、
            也不计入级别的升降——这些回合是探索，摔了不拉低级别。训练里级别只爬到 7° 左右（训练条件下早摔与撑住
            各一半的坡度），测 10° 上坡走不动、15° 上下坡现行与分坡向版都摔 80–100%，因为 9° 以上根本没练到。
            横坡不加：横坡 9° 以上几乎立刻摔，没有可学的东西，它自己的级别已经按成败升降。
        hard_floor: 陡坡回合的坡度下界比例（0.6 → 9–15°，10° 与 15° 都在里面）。
        forward_fraction: 坡上回合里有这么大比例，整个回合的指令都是「一路往前走」（`forward_ranges`），不再每 3–8 秒随机重抽。
            训练的随机指令（前后各 0.4 m/s、侧向 ±0.3、偏航 ±1 rad/s）几秒内就把机器人转到相对坡任意的朝向、一半时间在后退或站着，
            评测的「正对上坡以 0.2 m/s 直走」只占训练里很小一部分，上坡 5° 的前进速度 0.07（平地 0.16）、10° 走不动与此一致。
        forward_ranges: 往前走回合的指令范围（前进速度、侧向速度、偏航角速度，各取均匀分布）；前进速度围着评测的 0.2 m/s。
        progress_weight: 给了（> 0）就在往前走的坡上回合里加一项「前进进度」奖励（`slope_progress`）：机体前进速度除以指令的前进速度，
            夹在 [−0.5, 1]，权重取这个值。线速度跟踪奖励 exp(−误差²/std²) 在速度为 0 附近很平，原地踏步（误差 = 指令）与慢走得分差不多，
            踏步又照样拿抬脚奖励（权重 3.0）、不冒摔倒的险，所以 5° / 10° 上坡的前进速度只有 0.07 / −0.02（平地 0.15）；
            这一项对前进速度是线性的，每多走一点就多得一点。
    """

    max_deg: float = 15.0
    mix: tuple[float, float, float, float] = (0.30, 0.30, 0.25, 0.15)
    start_levels: tuple[float, float, float] = (0.4, 0.4, 0.3)
    step: float = 6.0e-4
    slope_floor: float = 0.6
    min_survive_steps: int = 500
    min_fail_steps: int = 25
    max_fail_steps: int = 400
    heading_jitter_deg: float = 10.0
    slope_pushes: bool = False
    slope_track_std: float | None = None
    hard_fraction: float = 0.0
    hard_floor: float = 0.8
    forward_fraction: float = 0.0
    forward_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.1, 0.3), (-0.05, 0.05), (-0.15, 0.15))
    progress_weight: float = 0.0


class Slopes:
    """一个环境里每个并行环境的坡：重力数组换成逐环境的一份，并记着各自的坡度、坡向。"""

    def __init__(self, env: ManagerBasedRlEnv):
        sim = env.sim
        model = sim.wp_model
        if model.opt.gravity.shape[0] != env.num_envs:
            per_world = wp.array([model.opt.gravity.numpy()[0]] * env.num_envs, dtype=wp.vec3,
                                 device=model.opt.gravity.device)
            model.opt.gravity = per_world
            # 换了数组就要重新录 CUDA 图：录好的图还指着旧数组，新值会被静默忽略（mjlab Simulation 的说明）。
            sim._model_bridge.clear_cache()
            sim.create_graph()
        self.gravity = wp.to_torch(model.opt.gravity)
        self.robot_gravity = env.scene["robot"].data.gravity_vec_w
        self.angle_deg = torch.zeros(env.num_envs, device=self.gravity.device)
        self.downhill_rad = torch.zeros(env.num_envs, device=self.gravity.device)
        # 自适应课程用：每个环境的坡度级别，以及上次分到坡时的环境步数（-1 = 还没分过）
        self.level = torch.zeros(env.num_envs, device=self.gravity.device)
        self.assigned_at = torch.full((env.num_envs,), -1, dtype=torch.long, device=self.gravity.device)
        self.leveled = torch.zeros(env.num_envs, dtype=torch.bool, device=self.gravity.device)   # 这个回合的坡是按级别取的
        # 分坡向课程用：这个回合的坡向（UPHILL / DOWNHILL / CROSS / FLAT），以及三个坡向各自的难度级别（`DirectionalSlope`）
        self.klass = torch.full((env.num_envs,), FLAT, dtype=torch.long, device=self.gravity.device)
        self.class_level: torch.Tensor | None = None
        # 这个回合的指令是不是「一路往前走」（`SlopeForwardCommand` 读它；`DirectionalSlope.forward_fraction`）
        self.forward = torch.zeros(env.num_envs, dtype=torch.bool, device=self.gravity.device)

    def set(self, env_ids: torch.Tensor, angle_deg: torch.Tensor, downhill_rad: torch.Tensor) -> None:
        """给这些环境设坡：坡度（度）与下坡方向（世界系方位角，弧度；东西滑落的方向）。"""
        theta = torch.deg2rad(angle_deg)
        g = torch.stack([torch.sin(theta) * torch.cos(downhill_rad),
                         torch.sin(theta) * torch.sin(downhill_rad),
                         -torch.cos(theta)], dim=1)
        self.gravity[env_ids] = g * GRAVITY
        self.robot_gravity[env_ids] = g
        self.angle_deg[env_ids] = angle_deg
        self.downhill_rad[env_ids] = downhill_rad


def slopes_of(env: ManagerBasedRlEnv) -> Slopes:
    """这个环境的坡；第一次取时把重力换成逐环境的一份。"""
    slopes = getattr(env, "_miniduck_slopes", None)
    if slopes is None:
        slopes = Slopes(env)
        env._miniduck_slopes = slopes
    return slopes


def heading_of(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> torch.Tensor:
    """这些环境的机器人刚写进仿真的朝向（偏航角，弧度）：读 qpos 里自由关节的四元数（w, x, y, z），不读过期的派生量。"""
    q = env.sim.data.qpos[env_ids, 3:7]
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def adaptive_angles(env: ManagerBasedRlEnv, slopes: Slopes, env_ids: torch.Tensor,
                    schedule: AdaptiveSlope) -> torch.Tensor:
    """按这些环境上一个回合的结果升降它们的坡度级别，再给出新一轮的坡度（度）。

    重置事件在 mjlab 的 `_reset_idx` 里先于终止管理器的重置，所以此刻 `terminated` / `time_outs` 还是刚结束的那个回合的。
    """
    step = int(env.common_step_counter)
    fresh = slopes.assigned_at[env_ids] < 0
    slopes.level[env_ids[fresh]] = schedule.start_level
    age = step - slopes.assigned_at[env_ids]
    counted = slopes.leveled[env_ids] & ~fresh
    survived = counted & env.termination_manager.time_outs[env_ids] & (age >= schedule.min_survive_steps)
    fell = (counted & env.termination_manager.terminated[env_ids]
            & (age >= schedule.min_fail_steps) & (age < schedule.max_fail_steps))
    level = slopes.level[env_ids]
    level = torch.where(survived, level + schedule.up, torch.where(fell, level - schedule.down, level))
    slopes.level[env_ids] = level.clamp(0.0, 1.0)
    slopes.assigned_at[env_ids] = step
    flat_draw = torch.rand(len(env_ids), device=env_ids.device) < schedule.flat_fraction
    slopes.leveled[env_ids] = ~flat_draw
    angle = slopes.level[env_ids] * schedule.max_deg
    angle[flat_draw] = 0.0
    return angle


def directional_angles(env: ManagerBasedRlEnv, slopes: Slopes, env_ids: torch.Tensor,
                       schedule: DirectionalSlope) -> tuple[torch.Tensor, torch.Tensor]:
    """按这些环境上一个回合的结果升降各坡向的级别，再给它们抽新一轮的坡向与坡度。

    Returns:
        （坡度（度），下坡方向相对出生朝向的角度（弧度））。
    """
    device, n = env_ids.device, len(env_ids)
    if slopes.class_level is None:
        slopes.class_level = torch.tensor(schedule.start_levels, device=device)
    step = int(env.common_step_counter)
    fresh = slopes.assigned_at[env_ids] < 0
    age = step - slopes.assigned_at[env_ids]
    klass = slopes.klass[env_ids]
    counted = (klass != FLAT) & ~fresh
    survived = counted & env.termination_manager.time_outs[env_ids] & (age >= schedule.min_survive_steps)
    fell = (counted & env.termination_manager.terminated[env_ids]
            & (age >= schedule.min_fail_steps) & (age < schedule.max_fail_steps))
    up = torch.bincount(klass[survived], minlength=4)[:3]
    down = torch.bincount(klass[fell], minlength=4)[:3]
    slopes.class_level = (slopes.class_level + schedule.step * (up - down)).clamp(0.0, 1.0)
    slopes.assigned_at[env_ids] = step
    edges = torch.cumsum(torch.tensor(schedule.mix, device=device), dim=0)[:-1]
    new_klass = torch.bucketize(torch.rand(n, device=device), edges)
    slopes.klass[env_ids] = new_klass
    slopes.forward[env_ids] = (new_klass != FLAT) & (torch.rand(n, device=device) < schedule.forward_fraction)
    angle = (slopes.class_level[new_klass.clamp(max=CROSS)] * schedule.max_deg
             * (schedule.slope_floor + (1.0 - schedule.slope_floor) * torch.rand(n, device=device)))
    angle[new_klass == FLAT] = 0.0
    hard = (new_klass <= DOWNHILL) & (torch.rand(n, device=device) < schedule.hard_fraction)   # 上坡、下坡（UPHILL, DOWNHILL = 0, 1）
    angle[hard] = schedule.max_deg * (schedule.hard_floor + (1.0 - schedule.hard_floor) * torch.rand(int(hard.sum()),
                                                                                                         device=device))
    slopes.klass[env_ids[hard]] = FLAT   # 陡坡回合不计入级别的升降（`counted` 只认三个坡向）
    base = torch.tensor(RELATIVE_AZIMUTH, device=device)[new_klass.clamp(max=CROSS)]
    side = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)
    base = torch.where(new_klass == CROSS, side * base, base)
    jitter = torch.randn(n, device=device) * math.radians(schedule.heading_jitter_deg)
    return angle, base + jitter


def reset_slopes(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None,
                 schedule: SlopeSchedule | AdaptiveSlope | DirectionalSlope) -> None:
    """重置事件：给被重置的环境重新抽坡（mjlab EventTermCfg，mode="reset"）。"""
    slopes = slopes_of(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=slopes.gravity.device)
    env_ids = torch.as_tensor(env_ids, device=slopes.gravity.device, dtype=torch.long)
    n = len(env_ids)
    if n == 0:
        return
    if isinstance(schedule, DirectionalSlope):
        angle, relative = directional_angles(env, slopes, env_ids, schedule)
        downhill = heading_of(env, env_ids) + relative   # 本事件要排在「重置机器人位姿」那个事件之后
    else:
        if isinstance(schedule, AdaptiveSlope):
            angle = adaptive_angles(env, slopes, env_ids, schedule)
        else:
            limit = schedule.limit_deg(int(env.common_step_counter))
            angle = torch.rand(n, device=env_ids.device) * limit
            angle[torch.rand(n, device=env_ids.device) < schedule.flat_fraction] = 0.0
        downhill = torch.rand(n, device=env_ids.device) * (2 * math.pi)
    slopes.set(env_ids, angle, downhill)


def mean_slope_deg(env: ManagerBasedRlEnv, env_ids: torch.Tensor, schedule: AdaptiveSlope) -> torch.Tensor:
    """课程项：已经分到过坡的环境的平均坡度级别换算成度，训练日志里记成 `Curriculum/slope_level_deg`。"""
    del env_ids
    slopes = slopes_of(env)
    assigned = slopes.assigned_at >= 0
    if not assigned.any():
        return torch.tensor(schedule.start_level * schedule.max_deg)
    return slopes.level[assigned].mean().cpu() * schedule.max_deg


def class_level_deg(env: ManagerBasedRlEnv, env_ids: torch.Tensor, klass: int,
                    schedule: DirectionalSlope) -> torch.Tensor:
    """课程项：这个坡向现在的级别换算成度，训练日志里记成 `Curriculum/slope_level_<坡向>_deg`。"""
    del env_ids
    levels = slopes_of(env).class_level
    level = torch.tensor(schedule.start_levels[klass]) if levels is None else levels[klass].cpu()
    return level * schedule.max_deg


def track_linear_velocity_slope(env: ManagerBasedRlEnv, std: float, slope_std: float, command_name: str,
                                asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
    """mjlab 的线速度跟踪奖励 exp(−误差²/std²)，坡上的环境用更紧的 `slope_std`（平地环境照旧）。

    exp(−e/s²) 换分母等于取幂：奖励 r = exp(−e/std²) 时，exp(−e/slope_std²) = r^(std²/slope_std²)。
    """
    base = mdp_rewards.track_linear_velocity(env, std=std, command_name=command_name, asset_cfg=asset_cfg)
    return torch.where(anchor_allowed(env), base, base ** (std**2 / slope_std**2))


def fell_on_slope(env: ManagerBasedRlEnv, limit_angle: float) -> torch.Tensor:
    """终止项：坡上的环境躯干相对重力倾斜超过 `limit_angle`（弧度）就结束回合；平地环境不管。"""
    gravity = env.scene["robot"].data.projected_gravity_b
    tilt = torch.acos((-gravity[:, 2]).clamp(-1.0, 1.0))
    return (tilt > limit_angle) & ~anchor_allowed(env)


def flat_only(func):
    """把一个事件函数包成只作用在平地环境上的版本（坡上的环境走「只走路」的那一套）。

    被包的函数签名不变（`functools.wraps`），mjlab 按签名核对参数，课程项照旧能按名字改它的参数。
    """

    @functools.wraps(func)
    def wrapped(env, env_ids, *args, **kwargs):
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        keep = env_ids[anchor_allowed(env)[env_ids]]
        if len(keep) > 0:
            func(env, keep, *args, **kwargs)

    return wrapped


@dataclass(kw_only=True)
class SlopeForwardCommandCfg(VelocityCommandCommandOnlyCfg):
    """上游的走路指令项，外加：坡上标了「往前走」的回合（`Slopes.forward`）指令改成一路往前走。"""

    forward_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.1, 0.3), (-0.05, 0.05), (-0.15, 0.15))

    def build(self, env: ManagerBasedRlEnv) -> SlopeForwardCommand:
        return SlopeForwardCommand(self, env)


class SlopeForwardCommand(VelocityCommandCommandOnly):
    """每次重抽指令后，把标了「往前走」的坡上环境的指令改成 `forward_ranges` 里的均匀抽样。

    重置时指令项在坡事件之后才重抽（mjlab `_reset_idx`：事件先于指令管理器的重置），所以读得到这一轮新抽的标记；
    回合中途每 3–8 秒的重抽同样读这个标记，整个回合都往前走。
    """

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        slopes = getattr(self._env, "_miniduck_slopes", None)
        if slopes is None:
            return
        ids = env_ids[slopes.forward[env_ids]]
        if len(ids) == 0:
            return
        r = torch.empty(len(ids), device=self.device)
        for column, (low, high) in enumerate(self.cfg.forward_ranges):
            self.vel_command_b[ids, column] = r.uniform_(low, high)
        self.is_standing_env[ids] = False
        self.vel_command_w[ids] = self.vel_command_b[ids]


def slope_progress(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """往前走的坡上回合的前进进度奖励：机体前进速度 / 指令的前进速度，夹在 [−0.5, 1]；其余环境为 0。"""
    slopes = slopes_of(env)
    command = env.command_manager.get_command(command_name)[:, 0].clamp(min=0.05)
    speed = env.scene["robot"].data.root_link_lin_vel_b[:, 0]
    return torch.where(slopes.forward, (speed / command).clamp(-0.5, 1.0), torch.zeros_like(speed))


def forward_command_cfg(cfg: VelocityCommandCommandOnlyCfg, schedule: DirectionalSlope) -> SlopeForwardCommandCfg:
    """把上游的走路指令项配置换成 `SlopeForwardCommandCfg`：字段照搬，另带这个课程的往前走范围。"""
    return SlopeForwardCommandCfg(**{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init},
                                  forward_ranges=schedule.forward_ranges)


# `Skill.slope_schedule` 的取值 → 坡度怎么取。
# 动作表里在用的只有 ramp（velstand-slope、standup-slope）与 directional-go（velstand-slope-dirg，坡地最终版）；
# 其余是一路试过的配方，留在这里方便复跑。
SCHEDULES = {"ramp": SlopeSchedule(), "adaptive": AdaptiveSlope(),
             "adaptive-quiet": AdaptiveSlope(slope_pushes=False),
             # 前一个的升降太慢（第 140 轮级别才 2.5°，一个回合 20 秒、每个环境每 42 轮才升降一次）：从 6° 起步、
             # 升降各 0.15（2.25°）、前 8 秒内摔倒才降（撑过 8 秒再摔中立），撑住与早摔相等的地方就是平衡点
             "adaptive-edge": AdaptiveSlope(slope_pushes=False, start_level=0.4, up=0.15, down=0.15,
                                            max_fail_steps=400),
             # 前一个的级别各方向共用，被横坡拉在 4.5°（横坡 7° 以上必摔），上下坡练不到 10–15°：
             # 上坡 / 下坡 / 横坡各有一个级别，各自按成败升降
             "directional": DirectionalSlope(),
             # 上一个在上坡 5° / 10° 原地踏步（速度 0.06 / −0.03）：坡上环境的线速度跟踪 std 从 0.316 收到 0.17
             "directional-track": DirectionalSlope(slope_track_std=0.17),
             # 上一个的级别只爬到 7°，10° 上坡走不动、15° 上下坡没练到：再加 20% 的上下坡回合直接取 9–15°（不计入级别）
             "directional-hard": DirectionalSlope(slope_track_std=0.17, hard_fraction=0.2, hard_floor=0.6),
             # 训练的随机指令几秒内就把机器人转到相对坡任意的朝向、一半时间在后退或站着，评测的「正对上坡直走 0.2 m/s」
             # 只占训练里很小一部分：再让一半的坡上回合整个回合都往前走（指令 0.1–0.3 m/s）
             "directional-fwd": DirectionalSlope(slope_track_std=0.17, hard_fraction=0.2, hard_floor=0.6,
                                                 forward_fraction=0.5),
             # 上一个在 10° 上坡仍走不动（速度 −0.02…+0.03）：线速度跟踪奖励在速度 0 附近平，原地踏步也拿抬脚奖励，
             # 往前走的坡上回合再加一项对前进速度线性的进度奖励（权重 2.0）
             "directional-go": DirectionalSlope(slope_track_std=0.17, hard_fraction=0.2, hard_floor=0.6,
                                                forward_fraction=0.5, progress_weight=2.0)}


def anchor_allowed(env: ManagerBasedRlEnv) -> torch.Tensor:
    """哪些环境在平地上（逐环境布尔）：没装坡的环境全算平地。"""
    slopes = getattr(env, "_miniduck_slopes", None)
    if slopes is None:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return slopes.angle_deg < FLAT_BELOW_DEG


def pin_relative(env: ManagerBasedRlEnv, angle_deg: float, downhill_from_heading_rad: float) -> None:
    """评测用：所有环境同一坡度，下坡方向按各自机器人此刻的朝向摆。

    下坡方向 = 朝向 + `downhill_from_heading_rad`：0 是正对下坡走（下坡），π 是上坡，
    +π/2 是下坡在左手边（横坡）。

    Args:
        env: mjlab 环境。
        angle_deg: 坡度（度）。
        downhill_from_heading_rad: 下坡方向相对机器人朝向的角度（弧度，逆时针为正）。
    """
    slopes = slopes_of(env)
    q = env.scene["robot"].data.root_link_quat_w
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    ids = torch.arange(env.num_envs, device=slopes.gravity.device)
    slopes.set(ids, torch.full_like(yaw, angle_deg), yaw + downhill_from_heading_rad)
