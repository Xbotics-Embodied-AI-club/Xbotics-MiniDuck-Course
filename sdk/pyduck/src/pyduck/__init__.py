"""pyduck —— MiniDuck 鸭子的 Python 上位机 SDK。

    import pyduck

    with pyduck.Duck.connect("192.168.4.1") as duck:
        print(duck.state.get().state_cn)
        with duck.motion.drive() as drive:
            drive.twist(vx=0.2)
            time.sleep(2)

协议常量与格式串全部来自 ``firmware/tools/protocol.py``（见 :mod:`pyduck._proto`），
这里只负责组包、收发与语义封装。
"""

from __future__ import annotations

import struct
import time
from collections.abc import Mapping
from typing import Any

from ._proto import P
from .errors import DuckCommandError, DuckConnectionError, DuckError, DuckTimeoutError
from .files import Files, files_writable
from .motion import DriveSession, Motion, SKILLS
from .policy import Policy, PolicyInfo, PolicyReply
from .records import Records
from .servos import CalibrationState, Diagnostics, ServoDiagnostics, ServoSnapshot, Servos
from .sounds import Sounds
from .state import Status, State
from .transport import Transport, UdpTransport, WsTransport, make_transport
from .tune import Tune
from .video import VideoStream
from .wifi import Wifi


class Duck:
    """一台 MiniDuck。用法见 :func:`connect` 与 README。"""

    def __init__(self, host: str, transport: Transport, *, timeout: float = 3.0,
                 web_ports: Mapping[str, int] | None = None) -> None:
        self.host = host
        self.transport = transport
        self.timeout = timeout
        self.web_ports = dict(P.WEB_PORTS)
        if web_ports:
            self.web_ports.update(web_ports)
        self._state: State | None = None
        self._motion: Motion | None = None
        self._servos: Servos | None = None
        self._tune: Tune | None = None
        self._files: Files | None = None
        self._records: Records | None = None
        self._sounds: Sounds | None = None
        self._wifi: Wifi | None = None
        self._policy: Policy | None = None
        self._video: VideoStream | None = None
        self._closed = False

    # ---- 连接 ---------------------------------------------------------
    @classmethod
    def connect(cls, host: str, transport: str | Transport = "ws", timeout: float = 3.0, *,
                ws_port: int | None = None, udp_port: int | None = None,
                web_ports: Mapping[str, int] | None = None,
                **transport_options: Any) -> Duck:
        """连接鸭子并给板子对时。

        Args:
            host: 板子的 IP（AP 模式默认 ``192.168.4.1``；没有 mDNS，要自己给）。
            transport: ``"ws"``（88 口，默认）或 ``"udp"``（5005 口），也可传现成的
                :class:`~pyduck.transport.Transport`。
            timeout: WebSocket 握手 / 请求的默认超时。
            ws_port / udp_port / web_ports: 只在测试或端口被占用时覆盖默认端口。
            **transport_options: 传给传输层（如 ``reconnect_limit``、``request_timeout``）。
        """
        if isinstance(transport, Transport):
            channel = transport
        else:
            port = ws_port if transport == "ws" else udp_port
            channel = make_transport(transport, host, timeout=timeout, port=port,
                                     **transport_options)
        duck = cls(host, channel, timeout=timeout, web_ports=web_ports)
        try:
            duck.sync_time()
        except Exception:
            channel.close()
            raise
        return duck

    def sync_time(self) -> None:
        """发一条 TIME（无回复），让事件日志能把开机毫秒换算成钟点。"""
        seq = self.next_seq()
        pkt = struct.pack(P.TIME_FMT, P.CLOCK_SET, P.VERSION, seq, int(time.time()))
        self.transport.send(pkt)

    # ---- 内部收发 -----------------------------------------------------
    def next_seq(self) -> int:
        return self.transport.next_seq()

    def _request(self, reply_magic: bytes, pkt: bytes, timeout: float | None = None) -> bytes:
        return self.transport.request(reply_magic, pkt, self.timeout if timeout is None else timeout)

    # ---- 懒属性 -------------------------------------------------------
    @property
    def state(self) -> State:
        if self._state is None:
            self._state = State(self)
        return self._state

    @property
    def motion(self) -> Motion:
        if self._motion is None:
            self._motion = Motion(self)
        return self._motion

    @property
    def servos(self) -> Servos:
        if self._servos is None:
            self._servos = Servos(self)
        return self._servos

    @property
    def tune(self) -> Tune:
        if self._tune is None:
            self._tune = Tune(self)
        return self._tune

    @property
    def files(self) -> Files:
        if self._files is None:
            self._files = Files(self)
        return self._files

    @property
    def records(self) -> Records:
        if self._records is None:
            self._records = Records(self)
        return self._records

    @property
    def sounds(self) -> Sounds:
        if self._sounds is None:
            self._sounds = Sounds(self)
        return self._sounds

    @property
    def wifi(self) -> Wifi:
        if self._wifi is None:
            self._wifi = Wifi(self)
        return self._wifi

    @property
    def policy(self) -> Policy:
        if self._policy is None:
            self._policy = Policy(self)
        return self._policy

    @property
    def video(self) -> VideoStream:
        if self._video is None:
            self._video = VideoStream()
        return self._video

    # ---- 收尾 ---------------------------------------------------------
    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.transport.close()

    def __enter__(self) -> Duck:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "Duck",
    "DuckError",
    "DuckTimeoutError",
    "DuckConnectionError",
    "DuckCommandError",
    "Status",
    "State",
    "Motion",
    "DriveSession",
    "Servos",
    "ServoSnapshot",
    "ServoDiagnostics",
    "Diagnostics",
    "CalibrationState",
    "Tune",
    "Files",
    "files_writable",
    "Records",
    "Sounds",
    "Wifi",
    "Policy",
    "PolicyInfo",
    "PolicyReply",
    "VideoStream",
    "SKILLS",
    "P",
    "Transport",
    "WsTransport",
    "UdpTransport",
    "make_transport",
]
