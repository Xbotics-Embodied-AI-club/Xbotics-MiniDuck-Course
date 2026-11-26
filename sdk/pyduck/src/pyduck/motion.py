"""运动：DCMD 运动命令、drive() 平滑会话、一次性动作、卸力 / 启动。

固件的运动命令 500 ms 没新包就归零（安全站立）；坐姿请求是锁存的，要清掉得发一帧不带
``sit`` 位的命令。这些语义都写在 README 的「语义陷阱」里。
"""

from __future__ import annotations

import struct
import threading
import time
from typing import TYPE_CHECKING

from ._proto import P
from .state import Status

if TYPE_CHECKING:
    from . import Duck

SKILLS = {"ground-pick": 3, "ball-kick": 4, "roulade": 5}
SEND_HZ = 20.0
RAMP_SECONDS = 0.5
FLAG_SIT = 0x01
FLAG_MOUTH = 0x02


def _pack(duck: Duck, seq: int, command, *, skill: int = 0, flags: int = 0) -> bytes:
    twist = list(command[:3])
    head = list(command[3:7])
    body = list(command[7:13])
    return struct.pack(P.MOTION_FMT, P.MOTION, P.VERSION, seq, skill, flags, 0,
                       *twist, *head, *body)


class DriveSession:
    """``with duck.motion.drive() as d:`` 期间的 20 Hz 发送线程。

    改目标（``twist`` / ``head`` / ``body``）时，从当前已发送的值在 0.5 s 内线性过渡到新值；
    退出时先线性回零、再补几帧全零，保证固件收到明确的「站着」。
    """

    def __init__(self, duck: Duck) -> None:
        self._duck = duck
        self._lock = threading.RLock()
        self._target = [0.0] * 13
        self._current = [0.0] * 13
        self._from = [0.0] * 13
        self._ramp_start = time.monotonic()
        self._flags = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- 目标设置 -----------------------------------------------------
    def _set(self, start: int, values) -> None:
        with self._lock:
            now = time.monotonic()
            self._from = list(self._current)
            self._ramp_start = now
            target = list(self._target)
            for offset, value in enumerate(values):
                target[start + offset] = float(value)
            self._target = target

    def twist(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> DriveSession:
        """线速度 m/s、角速度 rad/s。"""
        self._set(0, (vx, vy, wz))
        return self

    def head(self, neck_pitch: float = 0.0, head_pitch: float = 0.0,
             head_yaw: float = 0.0, head_roll: float = 0.0) -> DriveSession:
        """颈 / 头的俯仰、偏航、横滚（rad）。"""
        self._set(3, (neck_pitch, head_pitch, head_yaw, head_roll))
        return self

    def body(self, x: float = 0.0, y: float = 0.0, z: float = 0.0,
             roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> DriveSession:
        """机身平移（m）与姿态（rad）。"""
        self._set(7, (x, y, z, roll, pitch, yaw))
        return self

    def mouth(self, open_: bool) -> DriveSession:
        """张嘴请求（锁存到下一帧不带这一位）。"""
        with self._lock:
            self._flags = (FLAG_SIT if self._flags & FLAG_SIT else 0) | (FLAG_MOUTH if open_ else 0)
        return self

    # ---- 线程 ---------------------------------------------------------
    def _step(self, now: float) -> list[float]:
        with self._lock:
            phase = min(1.0, max(0.0, (now - self._ramp_start) / RAMP_SECONDS))
            self._current = [a + (b - a) * phase for a, b in zip(self._from, self._target)]
            return list(self._current)

    def _run(self) -> None:
        period = 1.0 / SEND_HZ
        tick = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            command = self._step(now)
            with self._lock:
                flags = self._flags
            seq = self._duck.next_seq()
            try:
                self._duck.transport.send(_pack(self._duck, seq, command, flags=flags))
            except Exception:
                # 连接断了：会话结束由调用方决定怎么处理（退出时补零会再报一次）
                return
            tick += period
            delay = tick - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                tick = time.monotonic()

    def __enter__(self) -> DriveSession:
        with self._lock:
            self._current = list(self._target)
            self._from = list(self._target)
            self._ramp_start = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="pyduck-drive", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # 先线性减速到零，再停线程，最后补三帧全零（固件 500 ms 兜底归零，这里不等它）
        with self._lock:
            self._from = list(self._current)
            self._target = [0.0] * 13
            self._ramp_start = time.monotonic()
        time.sleep(RAMP_SECONDS + 1.0 / SEND_HZ)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        zero = [0.0] * 13
        for _ in range(3):
            try:
                self._duck.transport.send(_pack(self._duck, self._duck.next_seq(), zero))
            except Exception:
                break
            time.sleep(1.0 / SEND_HZ)


class Motion:
    """``duck.motion``。"""

    def __init__(self, duck: Duck) -> None:
        self._duck = duck

    # ---- 持续运动 -----------------------------------------------------
    def drive(self) -> DriveSession:
        """20 Hz 持续发送的上下文管理器；退出时自动回零。"""
        return DriveSession(self._duck)

    def _drive_for(self, seconds: float, **twist) -> None:
        with self.drive() as session:
            session.twist(**twist)
            time.sleep(seconds)

    def forward(self, seconds: float = 2.0, v: float | None = None) -> None:
        self._drive_for(seconds, vx=0.2 if v is None else v)

    def backward(self, seconds: float = 2.0, v: float | None = None) -> None:
        self._drive_for(seconds, vx=-(0.2 if v is None else v))

    def turn_left(self, seconds: float = 2.0, v: float | None = None) -> None:
        self._drive_for(seconds, wz=0.5 if v is None else v)

    def turn_right(self, seconds: float = 2.0, v: float | None = None) -> None:
        self._drive_for(seconds, wz=-(0.5 if v is None else v))

    # ---- 单帧 ---------------------------------------------------------
    def stop(self) -> None:
        """单发一帧全零运动命令。"""
        self._duck.transport.send(_pack(self._duck, self._duck.next_seq(), [0.0] * 13))

    def _send_for(self, command, *, flags: int = 0, skill: int = 0,
                  seconds: float = 0.3) -> None:
        period = 1.0 / SEND_HZ
        count = max(1, int(round(seconds * SEND_HZ)))
        for _ in range(count):
            try:
                self._duck.transport.send(
                    _pack(self._duck, self._duck.next_seq(), command, flags=flags, skill=skill))
            except Exception:
                return
            time.sleep(period)

    def sit(self) -> None:
        """请求坐下（坐姿锁存；要站起来用 :meth:`stand`）。"""
        self._send_for([0.0] * 13, flags=FLAG_SIT)

    def stand(self) -> Status:
        """从卸力 / 待机起身，或从坐姿解除锁存。

        固件里「站起来」只有 CCMD 9 这一条；它在卸力（LIMP）时走
        ``calibrationStandFromLimp`` 路径。闭环已经在跑（坐着、站着）时 CCMD 9 会被忽略，
        这时发一帧不带 sit 位的零命令，把坐姿锁存清掉。
        """
        status = self._duck.state.get()
        if status.state in (0, 7):               # IDLE / LIMP
            self._duck.servos._calib(P.CMD_STAND)
        else:                                     # 坐着：清 sit 锁存；站着：什么也不做
            self._send_for([0.0] * 13, flags=0, seconds=0.25)
        time.sleep(0.05)
        return self._duck.state.get()

    def limp(self) -> Status:
        """全身卸力（先扶住机器人）。"""
        self._duck.servos._calib(P.CMD_LIMP)
        time.sleep(0.05)
        return self._duck.state.get()

    # ---- 一次性动作 ---------------------------------------------------
    def skill(self, name: str) -> Status:
        """请求 core 动作（ground-pick / ball-kick / roulade）。

        固件认边沿，所以保持 300 ms 再归零；动作本身异步执行，进度看 SSTA 的 state。
        """
        if name not in SKILLS:
            raise ValueError(f"未知动作 {name!r}（可选：{', '.join(SKILLS)}）")
        self._send_for([0.0] * 13, skill=SKILLS[name], seconds=0.3)
        self._send_for([0.0] * 13, flags=0, skill=0, seconds=0.1)
        return self._duck.state.get()
