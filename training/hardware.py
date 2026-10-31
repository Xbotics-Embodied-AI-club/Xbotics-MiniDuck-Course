"""硬件版本：训练、评测、导出只认这一个参数，其余都从声明里查。

一个硬件版本 = `sim_model/hardware/<版本>.json`，版本号是 `<日期>-<舵机>`（如 `2026-10-24-hd1910`），
它把三样东西组合起来：

- 机体结构 `sim_model/structure/<日期>/`：MJCF 变体、网格、质量属性表；
- 电机 `sim_model/servos/<厂家>/<型号>/servo.json`：执行器模型、固件刚度、固件限流；
- 供电：板子给舵机的电压，写在硬件版本文件里（它是板子的属性，不是舵机的）。

策略按同一个版本号存（`policies/<版本>/<动作>/`），所以每份策略都能对回它训练时的仿真模型。
训练入口用 `MINIDUCK_HARDWARE` 选版本；评测与导出从权重的训练设置里读，不看环境变量。

硬件更新时：结构件变了 → 加 `structure/<日期>/`；换了舵机 → 加 `servos/<厂家>/<型号>/`；
再加一个 `hardware/<日期>-<舵机>.json` 把它们组起来。已经训过策略的版本不再改动。

本文件不依赖仿真器：测试、导出核对这些不要 GPU 的步骤也用它。
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path

from paths import POLICY_DIR, SIM_MODEL, file_sha256

HARDWARE_DIR = SIM_MODEL / "hardware"
STRUCTURE_DIR = SIM_MODEL / "structure"
SERVO_DIR = SIM_MODEL / "servos"
ID_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}-[a-z0-9]+")



@dataclasses.dataclass(frozen=True)
class Servo:
    """一种电机：`sim_model/servos/<厂家>/<型号>/servo.json`。

    Attributes:
        key: `<厂家>/<型号>`，硬件版本文件里就这么引用它。
        spec: 声明内容。
    """

    key: str
    spec: dict

    @property
    def dir(self) -> Path:
        """这种电机的目录（参数文件按相对它的路径声明）。"""
        return SERVO_DIR / self.key

    @property
    def file(self) -> Path:
        """声明文件。"""
        return self.dir / "servo.json"

    @property
    def vendor(self) -> str:
        """厂家（`feetech`），执行器按它分派到 `training/servos/<厂家>.py`。"""
        return self.spec["vendor"]

    @property
    def description(self) -> str:
        """给人看的名字。"""
        return self.spec["description"]

    @property
    def actuator(self) -> dict:
        """执行器声明：模型、固件刚度 `kp_fw`、固件限流 `max_current_a`。"""
        return self.spec["actuator"]


@dataclasses.dataclass(frozen=True)
class Hardware:
    """一个硬件版本。

    Attributes:
        id: 版本号 `<日期>-<舵机>`。
        spec: `sim_model/hardware/<版本号>.json` 的内容。
        servo: 这一版用的电机。
    """

    id: str
    spec: dict
    servo: Servo

    @property
    def file(self) -> Path:
        """声明文件。"""
        return HARDWARE_DIR / f"{self.id}.json"

    @property
    def description(self) -> str:
        """给人看的一句话。"""
        return self.spec["description"]

    @property
    def structure_dir(self) -> Path:
        """机体结构目录 `sim_model/structure/<日期>/`。"""
        return STRUCTURE_DIR / self.spec["structure"]

    @property
    def model_dir(self) -> Path:
        """MJCF 变体与网格所在目录。"""
        return self.structure_dir / "microduck"

    @property
    def mass_table(self) -> Path:
        """躯干质量属性表（SolidWorks 报告原样数字），训练开始时拿它核对仿真里的模型。"""
        return self.structure_dir / "mass" / "trunk_base.json"

    @property
    def supply(self) -> dict:
        """供电：电压范围、负载压降、压降下限。"""
        return self.spec["supply"]

    @property
    def policies_dir(self) -> Path:
        """这一版的策略目录 `policies/<版本号>/`。"""
        return POLICY_DIR / self.id

    def training_record(self, robot_xml: str) -> dict:
        """训练设置里记下的这一版硬件：版本号、声明原文与各文件哈希。

        导出与评测靠 `hardware` 找回同一版；哈希把权重钉到训练时那份文件上，事后改了文件对得出来。

        Args:
            robot_xml: 这次训练用的 MJCF 文件名（在 `model_dir` 里）。

        Returns:
            要并进训练设置的字段。
        """
        record = {
            "hardware": self.id, "hardware_spec": self.spec,
            "hardware_spec_sha256": file_sha256(self.file),
            "servo": self.servo.key, "servo_spec": self.servo.spec,
            "servo_spec_sha256": file_sha256(self.servo.file),
            "robot_xml": robot_xml, "robot_xml_sha256": file_sha256(self.model_dir / robot_xml),
            "mass_table_sha256": file_sha256(self.mass_table),
        }
        params = self.servo.actuator.get("params_json")
        if params:
            record["servo_params_sha256"] = file_sha256(self.servo.dir / params)
        return record


def known() -> list[str]:
    """全部硬件版本号，按日期排。"""
    return sorted(p.stem for p in HARDWARE_DIR.glob("*.json"))


def load(hardware_id: str) -> Hardware:
    """读一个硬件版本，并核对它引用的结构与电机都在。

    Args:
        hardware_id: 版本号。

    Returns:
        硬件版本。

    Raises:
        RuntimeError: 没有这个版本，版本号不合 `<日期>-<舵机>`，或它引用的结构 / 电机不存在。
    """
    path = HARDWARE_DIR / f"{hardware_id}.json"
    if not path.is_file():
        raise RuntimeError(f"没有硬件版本 {hardware_id!r}，可选：{', '.join(known())}")
    if not ID_PATTERN.fullmatch(hardware_id):
        raise RuntimeError(f"硬件版本号 {hardware_id!r} 不合 <日期>-<舵机>（如 2026-10-24-hd1910）")
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec["id"] != hardware_id:
        raise RuntimeError(f"{path.name} 里写的版本号是 {spec['id']!r}，与文件名不符")
    servo_file = SERVO_DIR / spec["servo"] / "servo.json"
    if not servo_file.is_file():
        raise RuntimeError(f"硬件版本 {hardware_id} 引用的电机 {spec['servo']!r} 没有声明（{servo_file}）")
    hardware = Hardware(hardware_id, spec, Servo(spec["servo"], json.loads(servo_file.read_text(encoding="utf-8"))))
    if not hardware.mass_table.is_file():
        raise RuntimeError(f"硬件版本 {hardware_id} 引用的结构 {spec['structure']!r} 不完整（缺 {hardware.mass_table}）")
    return hardware


def from_env() -> Hardware:
    """训练入口用：从 `MINIDUCK_HARDWARE` 取这一次用哪个硬件版本（必填，没有默认值）。

    Returns:
        硬件版本。

    Raises:
        RuntimeError: 没给，或没有这个版本。
    """
    hardware_id = os.environ.get("MINIDUCK_HARDWARE", "").strip()
    if not hardware_id:
        raise RuntimeError(f"没有给 MINIDUCK_HARDWARE（硬件版本），可选：{', '.join(known())}")
    return load(hardware_id)


def of_training_settings(settings: dict) -> Hardware:
    """一份权重是在哪个硬件版本上训的：评测、导出都必须用同一版，不看环境变量。

    Args:
        settings: 权重里的训练设置。

    Returns:
        硬件版本。

    Raises:
        RuntimeError: 设置里没有硬件信息，说不清是在哪一版上训的。
    """
    if "hardware" in settings:
        return load(settings["hardware"])
    raise RuntimeError("这份权重的训练设置里没有硬件信息，说不清它是在哪个硬件版本上训的")
