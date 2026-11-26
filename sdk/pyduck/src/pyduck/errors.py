"""pyduck 的异常树。"""

from __future__ import annotations


class DuckError(Exception):
    """pyduck 所有错误的基类。"""


class DuckTimeoutError(DuckError):
    """等不到设备回复或状态没有在期限内变化。"""


class DuckConnectionError(DuckError):
    """连接建立失败，或断线后重连次数用尽。"""


class DuckCommandError(DuckError):
    """设备明确拒绝了这条命令（``reason`` 是固件给出的中文原因）。"""

    def __init__(self, reason: str, *, status_code: int | None = None):
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code
