"""传输层：WebSocket（默认，88 口）与 UDP（5005 口），同一套二进制包格式。

两条通道对上层是同一个接口：

- ``send(pkt)``：只发不等（对时 TIME、运动 DCMD、TCMD 下发这种没有回复的包）；
- ``request(reply_magic, pkt, timeout)``：发出去并等 ``magic + seq`` 都对上的回复。

后台泵线程负责两件事：按 ``(magic, seq)`` 把收到的包交给等待者；以及让 WebSocket 的
控制帧（ping/pong）继续被处理——websocket-client 只有在 ``recv()`` 循环里才会自动回 pong。
"""

from __future__ import annotations

import socket
import struct
import threading
import time

import websocket

from ._proto import P
from .errors import DuckConnectionError, DuckError, DuckTimeoutError

_SEQ_OFFSET = 6          # 所有请求/回复包头都是 <4sHH：magic、version、seq
_MIN_UDP_GAP = 0.020     # 固件每个主循环只解析一个入向包，UDP 连发要留间隔


class _Waiter:
    __slots__ = ("event", "reply", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.reply: bytes | None = None
        self.error: DuckError | None = None


class Transport:
    """收发与请求表的公共部分。子类实现 ``send`` 与 ``_close_socket``。"""

    def __init__(self, *, request_timeout: float = 2.0) -> None:
        self._waiters: dict[tuple[bytes, int], _Waiter] = {}
        self._lock = threading.RLock()
        self._send_lock = threading.RLock()
        self._seq = 0
        self._closed = False
        self.request_timeout = request_timeout

    # ---- 序号与请求表 -------------------------------------------------
    def next_seq(self) -> int:
        with self._lock:
            self._seq = (self._seq + 1) & 0xFFFF
            return self._seq

    def _register(self, magic: bytes, seq: int) -> _Waiter:
        waiter = _Waiter()
        with self._lock:
            if self._closed:
                raise DuckConnectionError("连接已关闭")
            self._waiters[(magic, seq)] = waiter
        return waiter

    def _unregister(self, magic: bytes, seq: int) -> None:
        with self._lock:
            self._waiters.pop((magic, seq), None)

    def _dispatch(self, data: bytes) -> bool:
        if len(data) < _SEQ_OFFSET + 2:
            return False
        seq = struct.unpack_from("<H", data, _SEQ_OFFSET)[0]
        key = (data[:4], seq)
        with self._lock:
            waiter = self._waiters.pop(key, None)
        if waiter is None:
            return False
        waiter.reply = data
        waiter.event.set()
        return True

    def _fail_waiters(self, error: DuckError) -> None:
        with self._lock:
            waiters = list(self._waiters.values())
            self._waiters.clear()
        for waiter in waiters:
            waiter.error = error
            waiter.event.set()

    # ---- 上层接口 -----------------------------------------------------
    def send(self, pkt: bytes) -> None:      # pragma: no cover - 子类实现
        raise NotImplementedError

    def request(self, reply_magic: bytes, pkt: bytes, timeout: float | None = None) -> bytes:
        """发出 ``pkt`` 并等 ``reply_magic`` 的回复。

        Args:
            reply_magic: 期望的**回复** magic（如 STAT 的回复是 SSTA）。
            pkt: 已带序号、完整的请求包。
            timeout: 秒；``None`` 用传输层的默认值。

        Raises:
            DuckTimeoutError: 期限内没等到回复。
            DuckConnectionError: 连接已断且重连失败。
        """
        if len(pkt) < _SEQ_OFFSET + 2:
            raise ValueError("请求包太短，包头应为 <4sHH")
        seq = struct.unpack_from("<H", pkt, _SEQ_OFFSET)[0]
        waiter = self._register(reply_magic, seq)
        try:
            self.send(pkt)
        except Exception:
            self._unregister(reply_magic, seq)
            raise
        wait = self.request_timeout if timeout is None else timeout
        if not waiter.event.wait(wait):
            self._unregister(reply_magic, seq)
            raise DuckTimeoutError(
                f"{wait:.1f} 秒内没收到 {reply_magic.decode('ascii', 'replace')} 回复"
                f"（seq {seq}）"
            )
        if waiter.error is not None:
            raise waiter.error
        assert waiter.reply is not None
        return waiter.reply

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._fail_waiters(DuckConnectionError("连接已关闭"))
        self._close_socket()

    def _close_socket(self) -> None:          # pragma: no cover - 子类实现
        raise NotImplementedError

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed


class WsTransport(Transport):
    """WebSocket 88 口（与各网页共用的那条）。"""

    def __init__(self, host: str, port: int | None = None, *, timeout: float = 3.0,
                 reconnect_interval: float = 2.0, reconnect_limit: int = 5,
                 request_timeout: float = 2.0) -> None:
        super().__init__(request_timeout=request_timeout)
        self.host = host
        self.port = P.WS_PORT if port is None else port
        self._url = f"ws://{host}:{self.port}/"
        self._reconnect_interval = reconnect_interval
        self._reconnect_limit = reconnect_limit
        self._ws: websocket.WebSocket | None = None
        self._stop = threading.Event()
        self._failure: DuckConnectionError | None = None
        self._connect(timeout)
        self._pump = threading.Thread(target=self._pump_loop, name="pyduck-ws", daemon=True)
        self._pump.start()

    # ---- 连接管理 -----------------------------------------------------
    def _connect(self, timeout: float) -> None:
        try:
            ws = websocket.create_connection(self._url, timeout=timeout, enable_multithread=True)
        except Exception as exc:
            raise DuckConnectionError(f"连不上 {self._url}：{exc}") from exc
        ws.settimeout(0.25)
        with self._lock:
            self._ws = ws

    def _reconnect(self) -> bool:
        for attempt in range(1, self._reconnect_limit + 1):
            if self._stop.is_set():
                return False
            self._stop.wait(self._reconnect_interval)
            if self._stop.is_set():
                return False
            try:
                self._connect(self._reconnect_interval)
                return True
            except DuckConnectionError:
                continue
        self._failure = DuckConnectionError(
            f"与 {self._url} 的 WebSocket 断开，重连 {self._reconnect_limit} 次都失败"
        )
        self._fail_waiters(self._failure)
        return False

    # ---- 泵 -----------------------------------------------------------
    def _pump_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                ws = self._ws
            if ws is None:
                if not self._reconnect():
                    return
                continue
            try:
                data = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as exc:      # 断线、协议错误：下面按断线处理
                with self._lock:
                    self._ws = None
                self._fail_waiters(DuckConnectionError(f"WebSocket 断开：{exc}"))
                continue
            if isinstance(data, str):     # 网页文本帧与命令通道无关
                continue
            self._dispatch(bytes(data))

    # ---- 收发 ---------------------------------------------------------
    def send(self, pkt: bytes) -> None:
        if self._closed:
            raise DuckConnectionError("连接已关闭")
        if self._failure is not None:
            raise self._failure
        with self._send_lock:
            with self._lock:
                ws = self._ws
            if ws is None:
                raise DuckConnectionError("WebSocket 还没连上（正在重连）")
            try:
                ws.send_binary(pkt)
            except Exception as exc:
                raise DuckConnectionError(f"WebSocket 发送失败：{exc}") from exc

    def _close_socket(self) -> None:
        self._stop.set()
        with self._lock:
            ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        pump = getattr(self, "_pump", None)
        if pump is not None and pump.is_alive():
            pump.join(timeout=2.0)


class UdpTransport(Transport):
    """UDP 5005 口（tools/ 下脚本用的那条）。"""

    def __init__(self, host: str, port: int | None = None, *, timeout: float = 3.0,
                 request_timeout: float = 2.0, min_gap: float = _MIN_UDP_GAP) -> None:
        super().__init__(request_timeout=request_timeout)
        self.host = host
        self.port = P.UDP_PORT if port is None else port
        self._min_gap = min_gap
        self._last_send = 0.0
        self._stop = threading.Event()
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.bind(("", 0))
            self._sock.settimeout(0.25)
        except OSError as exc:
            raise DuckConnectionError(f"UDP 套接字建立失败：{exc}") from exc
        self._pump = threading.Thread(target=self._pump_loop, name="pyduck-udp", daemon=True)
        self._pump.start()

    def _pump_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data, _addr = self._sock.recvfrom(2048)
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                continue
            self._dispatch(data)

    def send(self, pkt: bytes) -> None:
        if self._closed:
            raise DuckConnectionError("连接已关闭")
        with self._send_lock:
            # 固件每圈只解析一个入向包：连发要隔开，否则前面的包被后面的覆盖
            wait = self._min_gap - (time.monotonic() - self._last_send)
            if wait > 0:
                time.sleep(wait)
            try:
                self._sock.sendto(pkt, (self.host, self.port))
            except OSError as exc:
                raise DuckConnectionError(f"UDP 发送失败：{exc}") from exc
            self._last_send = time.monotonic()

    def _close_socket(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass
        pump = getattr(self, "_pump", None)
        if pump is not None and pump.is_alive():
            pump.join(timeout=2.0)


def make_transport(kind: str, host: str, *, timeout: float = 3.0,
                   port: int | None = None, **kwargs) -> Transport:
    """按名字建传输；``kind`` 为 ``"ws"`` 或 ``"udp"``。"""
    if kind == "ws":
        return WsTransport(host, port, timeout=timeout, **kwargs)
    if kind == "udp":
        return UdpTransport(host, port, timeout=timeout, **kwargs)
    raise ValueError(f"未知传输 {kind!r}（可选 ws / udp）")


__all__ = ["Transport", "WsTransport", "UdpTransport", "make_transport",
           "DuckTimeoutError", "DuckConnectionError"]
