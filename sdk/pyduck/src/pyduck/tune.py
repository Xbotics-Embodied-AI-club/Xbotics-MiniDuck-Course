"""调参：TGET 查询全量、TCMD 下发（无 ACK，2 s 后固件自动存 NVM）。"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

from ._proto import P
from .errors import DuckCommandError

if TYPE_CHECKING:
    from . import Duck


class Tune:
    """``duck.tune``。"""

    def __init__(self, duck: Duck) -> None:
        self._duck = duck

    def get(self, timeout: float | None = None) -> dict[str, float]:
        """TGET：返回全部参数（键名见 :class:`~pyduck._proto.P.TUNE_KEYS`）。"""
        seq = self._duck.next_seq()
        pkt = struct.pack(P.TUNE_FMT, P.TUNE_GET, P.VERSION, seq, 0, b"\0\0\0",
                          *([0] * 12), *([0.0] * 12))
        data = self._duck._request(P.TUNE_SET, pkt, timeout)
        if len(data) != struct.calcsize(P.TUNE_FMT):
            raise DuckCommandError(f"TCMD 长度 {len(data)}B 不对")
        fields = struct.unpack(P.TUNE_FMT, data)
        count = fields[3]
        keys = fields[5:5 + 12]
        values = fields[5 + 12:5 + 24]
        result: dict[str, float] = {}
        for i in range(min(count, 12)):
            key = keys[i]
            name = P.TUNE_KEYS[key] if key < len(P.TUNE_KEYS) else f"key{key}"
            result[name] = values[i]
        return result

    def set(self, **kwargs: float) -> None:
        """TCMD 下发（没有回复，改动立即生效）。

        Raises:
            ValueError: 键名不在 TUNE_KEYS 里，或一次超过 12 项。
        """
        unknown = [name for name in kwargs if name not in P.TUNE_KEYS]
        if unknown:
            raise ValueError(f"未知参数键 {', '.join(unknown)}（可选：{', '.join(P.TUNE_KEYS)}）")
        pairs = [(P.TUNE_KEYS.index(name), float(value)) for name, value in kwargs.items()]
        if len(pairs) > 12:
            raise ValueError("一次最多下发 12 项")
        keys = [0] * 12
        values = [0.0] * 12
        for i, (key, value) in enumerate(pairs):
            keys[i] = key
            values[i] = value
        seq = self._duck.next_seq()
        pkt = struct.pack(P.TUNE_FMT, P.TUNE_SET, P.VERSION, seq, len(pairs), b"\0\0\0",
                          *keys, *values)
        self._duck.transport.send(pkt)
