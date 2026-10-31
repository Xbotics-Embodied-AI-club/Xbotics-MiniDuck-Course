"""电机：把上游任务里的舵机执行器换成硬件版本声明的那种，环境建好后再核对确实换上了。

对外三个函数：`build_actuators`（换）、`check_actuators`（核对）、`describe`（一句话说明）。
共通的部分在这里：固件刚度、板子供电、固件限流对每种电机都一样处理；
厂家相关的部分在 `feetech.py`（用哪个 BAM 电机模型、参数从哪来、怎么认出建好的执行器），
按电机声明里的 `vendor` 分派。加一种新电机：在 `sim_model/servos/<厂家>/<型号>/` 放声明；
厂家是新的，再在这里加一个同样两个函数（`configure`、`built`）的模块并登记进 `_VENDORS`。
"""
from __future__ import annotations

import dataclasses
from types import ModuleType

from bam.mjlab import BamActuator
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuatorCfg

from hardware import Hardware

from . import feetech

_VENDORS: dict[str, ModuleType] = {"feetech": feetech}
_CURRENT_LIMITED: dict[type, type] = {}


def _vendor(hardware: Hardware) -> ModuleType:
    vendor = hardware.servo.vendor
    if vendor not in _VENDORS:
        raise RuntimeError(f"电机 {hardware.servo.key} 的厂家 {vendor!r} 没有执行器实现，已有：{', '.join(_VENDORS)}")
    return _VENDORS[vendor]


def current_limited(cfg_class: type) -> type:
    """给某个 BAM 执行器配置类派生一个「带固件限流」的子类。

    这一版 BAM 的 mjlab 配置没有限流字段，但电机模型本身支持（`max_current`：按占空比约束
    把 I = (duty·V − k_t·ω)/R 夹在 ±max_current 内）。电机模型在执行器构造时加载，
    所以在 `build()` 构造完立刻设上 —— 限流就成了配置的一部分，不管哪个训练器建环境都带着。

    Args:
        cfg_class: 执行器配置类。

    Returns:
        多一个 `max_current_a` 字段的子类（同一个父类只派生一次）。
    """
    if cfg_class not in _CURRENT_LIMITED:
        @dataclasses.dataclass(kw_only=True)
        class CurrentLimited(cfg_class):
            max_current_a: float

            def build(self, *args, **kwargs):
                actuator = super().build(*args, **kwargs)
                actuator._bam_model.actuator.max_current = self.max_current_a
                return actuator

        CurrentLimited.__name__ = CurrentLimited.__qualname__ = f"CurrentLimited{cfg_class.__name__}"
        _CURRENT_LIMITED[cfg_class] = CurrentLimited
    return _CURRENT_LIMITED[cfg_class]


def build_actuators(actuators, hardware: Hardware) -> tuple:
    """把上游的舵机执行器配置换成这个硬件版本的电机与供电。

    上游各任务都是同一类「摩擦随机化 BAM」配置；延迟、摩擦随机化这些字段照上游，
    换的是电机模型（厂家模块决定）、固件刚度、供电与固件限流。每次都构造新对象：
    上游配置是模块级共享的，而且参数文件路径在构造时才解析。

    Args:
        actuators: 上游实体配置里的执行器配置元组。
        hardware: 硬件版本。

    Returns:
        新的执行器配置元组。

    Raises:
        RuntimeError: 上游执行器不是 BAM 舵机（换不上），或厂家模块发现上游变了。
    """
    act_spec, supply = hardware.servo.actuator, hardware.supply
    sag = supply["load_sag_v_per_nm"]
    vendor = _vendor(hardware)
    out = []
    for act in actuators:
        if not isinstance(act, FrictionDRBamActuatorCfg):
            raise RuntimeError(f"执行器 {type(act).__name__} 不是上游的摩擦随机化 BAM 舵机，舵机换不上")
        fields = {f.name: getattr(act, f.name) for f in dataclasses.fields(act) if f.init}
        fields.update(
            kp_fw=float(act_spec["kp_fw"]), vin_range=tuple(supply["vin_range_v"]),
            vin_drop_gain_range=tuple(sag) if sag is not None else None,
            # 下限只在有负载压降时才起作用：没有压降时为空，免得留一个比电压范围还高的下限。
            vin_min=supply["sag_floor_v"])
        cfg_class = vendor.configure(act, hardware.servo, fields)
        if act_spec["max_current_a"] is not None:
            cfg_class = current_limited(cfg_class)
            fields["max_current_a"] = float(act_spec["max_current_a"])
        out.append(cfg_class(**fields))
    return tuple(out)


def check_actuators(env: ManagerBasedRlEnv, hardware: Hardware) -> None:
    """核对建好的环境里，每个舵机执行器就是声明的那种：电机模型、刚度、电压、限流。

    Args:
        env: 建好的 mjlab 环境。
        hardware: 硬件版本。

    Raises:
        RuntimeError: 机器人上没有 BAM 舵机，或任何一项与声明不符。
    """
    act_spec = hardware.servo.actuator
    low, high = hardware.supply["vin_range_v"]
    vendor = _vendor(hardware)
    actuators = [a for a in env.scene["robot"]._actuators if isinstance(a, BamActuator)]
    if not actuators:
        raise RuntimeError("机器人上没有 BAM 舵机执行器，舵机核对不了")
    for act in actuators:
        if not vendor.built(act, hardware.servo):
            raise RuntimeError(f"执行器是 {type(act).__name__}，与声明的电机 {hardware.servo.key} 不符 —— 舵机没换上")
        bam = act._bam_model.actuator
        if abs(float(bam.kp) - float(act_spec["kp_fw"])) > 1e-9:
            raise RuntimeError(f"舵机固件刚度是 {bam.kp}，声明的是 {act_spec['kp_fw']} —— 刚度没换上")
        vin = act.vin_tensor
        if float(vin.min()) < low - 1e-6 or float(vin.max()) > high + 1e-6:
            raise RuntimeError(f"舵机电压 {float(vin.min()):.2f}–{float(vin.max()):.2f} V "
                               f"不在声明的 {low}–{high} V 里 —— 供电没换上")
        if bam.max_current != act_spec["max_current_a"]:
            raise RuntimeError(f"舵机限流是 {bam.max_current}，声明的是 {act_spec['max_current_a']} —— 限流没换上")


def describe(hardware: Hardware) -> str:
    """一句话说明这一版的电机与供电，给建环境时的日志用。

    Args:
        hardware: 硬件版本。

    Returns:
        如「飞特 HD-1910（HD1910M）（kp 5，供电 5.9–6.1 V，不限流）」。
    """
    act_spec = hardware.servo.actuator
    low, high = hardware.supply["vin_range_v"]
    limit = act_spec["max_current_a"]
    return (f"{hardware.servo.description}（kp {act_spec['kp_fw']:.0f}，供电 {low:.1f}–{high:.1f} V，"
            f"{f'限流 {limit} A' if limit is not None else '不限流'}）")
