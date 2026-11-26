"""起身在弱舵机下训：critic 看舵机力气，奖励记「从背上翻下来」的进度。

原来的起身（standup-full）卡在第一步：从仰躺翻到侧面。仰躺开局里，舵机力气 ×1.0 / ×0.8 / ×0.7 时 6 秒内离开仰躺的只有 100% / 62% / 12%；
一旦翻到侧面，×0.7 也几乎全能站起。训练时舵机电压在环境间随机（`env.Skill.vin_range_v`），
弱的环境要自己找到不靠满力气的翻法。这里给起身任务加两样（`configure`）：

1. **critic 多看一维舵机力气**（供电 / 标称）。同一个仰躺姿势，强舵机翻得过去、弱舵机翻不过去，
   两者的价值差很多；critic 看不到力气就只能估平均值，优势里混进「这一只舵机强不强」的运气，
   弱舵机偶尔翻成功的那几次也分不出是动作好还是运气好。只给 critic、不给 actor 的特权信息，
   是非对称 actor-critic 的做法（Pinto 等，Asymmetric Actor Critic，RSS 2018）；真机上没有这个量，actor 照旧看不到。
2. **「离开仰躺」的进度奖励**。势函数 Φ = min(0, g_x + 0.5) / 0.5（g 是机体系单位重力：平躺在背上 −1，
   背面转离朝下超过 60° 即 0；站着后仰不到 30° 也是 0），每步奖励 Φ 的增量（按秒计）。上游起身奖励在仰、侧、趴之间
   没有梯度（直立奖励 cos(倾斜) 在三种躺姿都是 0，上游注释自己也写了「仰躺到趴之间的奖励地形是平的」），
   翻到一半又落回背上的尝试拿不到任何信号。势函数的增量就是 Ng、Harada、Russell（ICML 1999）的势函数塑形
   在 γ=1 时的形式：一整段的和只取决于起点与终点（翻下来记 +1、翻回去记 −1，来回晃得 0），
   不会被刷分，也不偏向翻向哪一边、是侧翻还是直接坐起。
"""
from __future__ import annotations

import torch
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers import ObservationTermCfg, RewardTermCfg

# 机体系重力的 x 分量高过它，就算离开了仰躺（背面不再在朝下 60° 以内）；同 getup_supine 的判据。
SUPINE_EXIT_GX = -0.5
# 进度奖励的权重：从平躺在背上到翻离仰躺一共记 2 分。起身任务站稳以后每秒的奖励约 10 分，
# 一次完整的翻身值约 0.2 秒的站立；翻到一半（Φ 走一半）记 1 分，够让 PPO 分出哪些动作翻得更远，
# 又远小于翻过去以后站起来的回报，不会让「翻到侧面就停」比站起来划算。
SUPINE_EXIT_WEIGHT = 2.0


def servo_vin(env: ManagerBasedRlEnv) -> torch.Tensor:
    """各环境舵机的供电电压（V），形状 (N,)；全身 14 个舵机是同一组 BAM 执行器、同一个电压。"""
    vins = [a.vin_tensor[:, 0] for a in env.scene["robot"]._actuators if getattr(a, "vin_tensor", None) is not None]
    if len(vins) != 1:
        raise RuntimeError(f"机器人上带逐环境供电的 BAM 执行器组应当正好一组，实际 {len(vins)} 组")
    return vins[0]


def servo_strength(env: ManagerBasedRlEnv, nominal_v: float) -> torch.Tensor:
    """critic 观测：舵机力气相对标称的偏离（供电 / 标称 − 1，满力气 0，×0.65 是 −0.35），形状 (N, 1)。

    BAM 里堵转力矩与空载转速都正比于供电，这一个数就是力气的倍数。
    """
    return (servo_vin(env) / nominal_v - 1.0).unsqueeze(1)


def supine_potential(gravity_b: torch.Tensor) -> torch.Tensor:
    """势函数 Φ：平躺在背上 −1，背面转离朝下超过 60°（或站着后仰不到 30°）为 0。

    Args:
        gravity_b: 机体系重力方向，形状 (N, 3)，不必归一化。

    Returns:
        形状 (N,)。
    """
    gx = gravity_b[:, 0] / gravity_b.norm(dim=1).clamp_min(1e-6)
    return torch.clamp(gx - SUPINE_EXIT_GX, max=0.0) / (1.0 + SUPINE_EXIT_GX)


class SupineExitProgress:
    """奖励项：这一步势函数 Φ 涨了多少（按秒计）；每个回合的第一步记 0。"""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        del cfg
        self._prev = torch.zeros(env.num_envs, device=env.device)
        # 刚重置的环境没有「上一步」：开局的姿势是摆出来的，不是翻出来的
        self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        phi = supine_potential(env.scene["robot"].data.projected_gravity_b)
        reward = torch.where(self._fresh, torch.zeros_like(phi), (phi - self._prev) / env.step_dt)
        self._prev = phi
        self._fresh = torch.zeros_like(self._fresh)
        return reward

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._fresh[env_ids] = True


def configure(cfg, nominal_v: float) -> None:
    """给起身任务的环境配置加上两样（文件头）：critic 的舵机力气观测、离开仰躺的进度奖励。

    Args:
        cfg: mjlab 的环境配置（原地改）。
        nominal_v: 硬件版本声明的标称供电（V），力气的分母。

    Raises:
        RuntimeError: 配置里已经有同名的项（上游加了同名的东西，要先核对是不是同一个意思）。
    """
    critic = cfg.observations["critic"].terms
    if "servo_strength" in critic or "supine_exit" in cfg.rewards:
        raise RuntimeError("起身任务里已经有 servo_strength / supine_exit 这一项 —— 上游加了同名的东西，先核对")
    critic["servo_strength"] = ObservationTermCfg(func=servo_strength, params={"nominal_v": nominal_v})
    cfg.rewards["supine_exit"] = RewardTermCfg(func=SupineExitProgress, weight=SUPINE_EXIT_WEIGHT)
