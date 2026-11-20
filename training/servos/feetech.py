"""飞特舵机：BAM 的飞特控制器 + 声明里的参数文件，并补上 mjlab 下缺的那个状态量。

刚度、供电、限流由 `servos` 统一换；这里只管飞特特有的两件事：电机模型换成参数文件
（`servo.json` 的 `params_json`，相对电机目录），以及下面这个初始化修正。

BAM 的飞特控制器（STS3215Actuator）用 `q_target_smooth` 记上一时刻的目标角、限制目标每步的变化，
但它只在读辨识日志（`load_log()`）时才创建这个量，mjlab 训练不走那条路，第一次 reset 就报
`AttributeError: 'STS3215Actuator' object has no attribute 'q_target_smooth'`。

这里在初始化时建好它，并在每个环境 reset 后的第一次计算时，从该环境当前的关节目标重新起步，
只动被重置的那些环境。做法照 fanhao375/microduck-replica 的 actuator/feetech_bam.py（Apache-2.0）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from mjlab.actuator.actuator import ActuatorCmd
from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator, FrictionDRBamActuatorCfg


class FeetechBamActuator(FrictionDRBamActuator):
    """带目标限速状态初始化的飞特 BAM 执行器。"""

    def initialize(self, mj_model, model, data, device) -> None:
        super().initialize(mj_model, model, data, device)
        self._bam_model.actuator.q_target_smooth = torch.zeros_like(self._prev_motor_torque)
        self._goal_reset_pending = torch.ones((data.nworld, 1), dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._goal_reset_pending.fill_(True)
        else:
            self._goal_reset_pending[env_ids] = True

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        actuator = self._bam_model.actuator
        # 用张量掩码而不是逐环境判断：不让 GPU 同步，也不动其他环境的目标历史
        actuator.q_target_smooth = torch.where(self._goal_reset_pending, cmd.pos,
                                               actuator.q_target_smooth)
        self._goal_reset_pending.zero_()
        return super().compute(cmd)


@dataclass(kw_only=True)
class FeetechBamActuatorCfg(FrictionDRBamActuatorCfg):
    """`FeetechBamActuator` 的配置（字段与上游的摩擦随机化 BAM 配置相同）。"""

    def build(self, entity, target_ids, target_names) -> FeetechBamActuator:
        return FeetechBamActuator(self, entity, target_ids, target_names)


def configure(act, servo, fields: dict) -> type:
    """把电机模型换成声明里的飞特参数文件。

    Args:
        act: 上游的执行器配置（飞特这边不看它自带的电机模型）。
        servo: 电机（`hardware.Servo`）。
        fields: 要构造的配置字段，这里换掉电机模型来源。

    Returns:
        配置类。
    """
    del act
    fields.update(motor_name=None, model=None, json_path=str(servo.dir / servo.actuator["params_json"]))
    return FeetechBamActuatorCfg


def built(actuator, servo) -> bool:
    """建好的执行器是不是飞特执行器、用的是不是声明的那份参数文件。"""
    return (isinstance(actuator, FeetechBamActuator)
            and Path(actuator.cfg.json_path) == servo.dir / servo.actuator["params_json"])
