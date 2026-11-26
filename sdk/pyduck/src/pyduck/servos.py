"""舵机：状态快照、掉扭矩诊断、实时刻度与零点标定的薄封装。"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._proto import P
from .errors import DuckCommandError

if TYPE_CHECKING:
    from . import Duck


@dataclass(frozen=True)
class ServoSnapshot:
    id: int
    name: str
    position: float          # rad
    target: float            # rad


@dataclass(frozen=True)
class ServoDiagnostics:
    id: int
    name: str
    voltage_v: float | None
    current_ma: int
    peak_ma: int
    temperature_c: int | None
    fault: int
    fault_tags: str
    fault_cn: str
    fault_count: int


@dataclass(frozen=True)
class Diagnostics:
    uptime_ms: int
    servo_count: int
    fault_total: int
    voltage_min_v: float
    voltage_max_v: float
    servos: dict[int, ServoDiagnostics]
    voltage_history: list[float]


@dataclass(frozen=True)
class CalibrationState:
    state: int
    last_id: int
    mask: int
    present: tuple[int, ...]
    zero_raw: tuple[int, ...]
    fault: tuple[int, ...]

    @property
    def state_name(self) -> str:
        return P.CALIB_STATE_NAMES.get(self.state, f"UNKNOWN({self.state})")

    @property
    def calibrated_count(self) -> int:
        return bin(self.mask & 0xFFFF).count("1")


class Servos:
    """``duck.servos``。"""

    def __init__(self, duck: Duck) -> None:
        self._duck = duck

    # ---- SSTA ---------------------------------------------------------
    def snapshot(self) -> dict[int, ServoSnapshot]:
        """当前关节位置与目标（rad），键是舵机 ID（1..15）。"""
        status = self._duck.state.get()
        return {
            sid: ServoSnapshot(sid, P.JOINT_NAMES.get(sid, "?"),
                               status.joint[sid], status.target[sid])
            for sid in range(1, min(len(status.joint), len(status.target)))
        }

    # ---- DSTA ---------------------------------------------------------
    def diagnostics(self, timeout: float | None = None) -> Diagnostics:
        """舵机掉扭矩病因快照（电压 / 电流 / 峰值 / 温度 / 故障位 / 电压历史）。"""
        seq = self._duck.next_seq()
        pkt = struct.pack(P.DIAG_REQ_FMT, P.DIAG_GET, P.VERSION, seq)
        data = self._duck._request(P.DIAG_STA, pkt, timeout)
        if len(data) != struct.calcsize(P.DIAG_FMT):
            raise DuckCommandError(f"DSTA 长度 {len(data)}B 不对")
        fields = struct.unpack(P.DIAG_FMT, data)
        uptime_ms, count, hist_count, fault_total = fields[3], fields[4], fields[5], fields[6]
        base = 10
        servos: dict[int, ServoDiagnostics] = {}
        for i in range(min(count, 16)):
            sid, fault, volt, cur, peak, fault_count, temp = fields[base + i * 7:base + i * 7 + 7]
            servos[sid] = ServoDiagnostics(
                id=sid,
                name=P.JOINT_NAMES.get(sid, "?"),
                voltage_v=None if volt == 0xFFFF else volt / 10.0,
                current_ma=cur,
                peak_ma=peak,
                temperature_c=None if temp == 0xFF else temp,
                fault=fault,
                fault_tags=P.fault_tags(fault),
                fault_cn=P.fault_cn(fault),
                fault_count=fault_count,
            )
        history = fields[base + 16 * 7:]
        return Diagnostics(
            uptime_ms=uptime_ms,
            servo_count=count,
            fault_total=fault_total,
            voltage_min_v=fields[8] / 10.0,
            voltage_max_v=fields[9] / 10.0,
            servos=servos,
            voltage_history=[3.0 + value / 10.0 for value in history[:hist_count]],
        )

    # ---- CCMD → CPOS --------------------------------------------------
    def _calib(self, cmd: int, joint: int = 0, milli_rad: int = 0,
               timeout: float | None = None) -> CalibrationState:
        seq = self._duck.next_seq()
        pkt = struct.pack(P.CCMD_FMT, P.CALIB_CMD, P.VERSION, seq, cmd, joint, milli_rad)
        data = self._duck._request(P.CALIB_POS, pkt, timeout)
        if len(data) != struct.calcsize(P.CPOS_FMT):
            raise DuckCommandError(f"CPOS 长度 {len(data)}B 不对")
        fields = struct.unpack(P.CPOS_FMT, data)
        return CalibrationState(
            state=fields[3],
            last_id=fields[4],
            mask=fields[5],
            present=fields[6:22],
            zero_raw=fields[22:38],
            fault=fields[38:54],
        )

    def raw_positions(self, timeout: float | None = None) -> dict[int, int | None]:
        """实时位置（原始刻度），键是舵机 ID；``-1`` 未应答、``-2`` 总线被闭环占着都返回 None。"""
        state = self._calib(P.CMD_QUERY, timeout=timeout)
        return {
            sid: (None if state.present[sid] < 0 else state.present[sid])
            for sid in range(1, len(state.present))
        }

    # ---- 标定 ---------------------------------------------------------
    def enter_passive(self) -> CalibrationState:
        """进入被动标定：全关节卸扭矩，等人手摆。"""
        return self._calib(P.CMD_ENTER_PASSIVE)

    def jog(self, joint_id: int, milli_rad: int) -> CalibrationState:
        """被动标定中把单个关节转一点再转回（核对方向）。"""
        if not 1 <= joint_id <= 15:
            raise ValueError("joint_id 必须是 1..15")
        return self._calib(P.CMD_JOG, joint=joint_id, milli_rad=int(milli_rad))

    def record_zero(self, joint_id: int) -> CalibrationState:
        """把该关节当前 raw 记为零点（仅 RAM，需 :meth:`apply_exit` 落盘）。"""
        if not 1 <= joint_id <= 15:
            raise ValueError("joint_id 必须是 1..15")
        return self._calib(P.CMD_RECORD, joint=joint_id)

    def apply_exit(self) -> CalibrationState:
        """整表写 NVS 并退出标定（之后机器人保持卸力）。"""
        return self._calib(P.CMD_APPLY_EXIT)

    def quit(self) -> CalibrationState:
        """放弃本次 RAM 改动并退出标定。"""
        return self._calib(P.CMD_EXIT_DISCARD)

    # ---- 转发 motion --------------------------------------------------
    def limp(self):
        return self._duck.motion.limp()

    def stand(self):
        return self._duck.motion.stand()
