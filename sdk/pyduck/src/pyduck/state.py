"""状态（STAT → SSTA）、等待条件与事件日志（EGET → ELOG）。"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ._proto import P
from .errors import DuckCommandError, DuckTimeoutError

if TYPE_CHECKING:
    from . import Duck


@dataclass(frozen=True)
class Status:
    """一帧 SSTA。旧固件（224B）的 ``tof_*`` / ``head_*`` 是 ``None``。

    ``gravity``（身体）是躯干系单位向量；``head_gravity`` 是头部芯片系**未归一化**原始值，
    模长≈1000 mg（1 g）；``head_gyro`` 单位 rad/s。坑 #76。
    """

    uptime_ms: int
    firmware: int
    state: int
    policy: int
    servo_kind: int
    servo_count: int
    flags: int
    wifi_stations: int
    battery_mv: int
    loop_hz10: int
    gravity: tuple[float, float, float]
    gyro: tuple[float, float, float]
    joint: tuple[float, ...]
    target: tuple[float, ...]
    command: tuple[float, float, float]
    hardware_id: str
    policy_source: tuple[int, ...]
    sounds_loaded: int
    record_file_bytes: int
    record_motion_s: int
    boot_seq: int
    tof_mm: int | None = None
    tof_valid: bool | None = None
    head_gravity: tuple[float, float, float] | None = None
    head_gyro: tuple[float, float, float] | None = None
    received_at: float = field(default=0.0, compare=False)

    @property
    def state_name(self) -> str:
        """英文状态名（固件 RobotState），如 ``STANDING``。"""
        return P.ROBOT_STATES.get(self.state, f"UNKNOWN({self.state})")

    @property
    def state_cn(self) -> str:
        """中文状态名，如「站立」。"""
        return P.ROBOT_STATES_CN.get(self.state, f"未知({self.state})")

    @property
    def firmware_text(self) -> str:
        return P.firmware_text(self.firmware)

    @property
    def loop_hz(self) -> float:
        return self.loop_hz10 / 10.0

    @property
    def battery_v(self) -> float | None:
        return self.battery_mv / 1000.0 if self.battery_mv else None

    @property
    def flag_names(self) -> set[str]:
        return {name for name, bit in P.STATUS_FLAGS.items() if self.flags & bit}

    @property
    def writable(self) -> bool:
        """固件自己报的「现在能写 flash」（SSTA 的 writable 位）。"""
        return bool(self.flags & P.STATUS_FLAGS["writable"])

    @property
    def joint_by_id(self) -> dict[int, float]:
        """关节位置（rad），键是舵机 ID（1..15）。"""
        return {sid: self.joint[sid] for sid in range(1, len(self.joint))}

    @property
    def target_by_id(self) -> dict[int, float]:
        """关节目标（rad），键是舵机 ID（1..15）。"""
        return {sid: self.target[sid] for sid in range(1, len(self.target))}

    @property
    def tof_distance_mm(self) -> int | None:
        return self.tof_mm if self.tof_valid else None

    @classmethod
    def from_frame(cls, data: bytes) -> Status:
        if len(data) not in (P.STATUS_V1_LEN, P.STATUS_V2_LEN):
            raise DuckCommandError(
                f"SSTA 长度 {len(data)}B 不对（应为 {P.STATUS_V1_LEN} 或 {P.STATUS_V2_LEN}B）"
            )
        raw = P.parse_ssta(data)
        if raw["tof_mm"] is None:
            # 旧固件没有这一组字段：和 tof_mm / head_* 一样报 None，而不是笼统的 False
            raw["tof_valid"] = None
        return cls(**raw, received_at=time.time())


def _event_dict(record: tuple) -> dict[str, Any]:
    seq, uptime_ms, kind, a, b, c, d, _check = record
    return {
        "seq": seq,
        "uptime_ms": uptime_ms,
        "kind": kind,
        "kind_name": P.EVENT_KINDS.get(kind, f"UNKNOWN({kind})"),
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "text": P.event_text(kind, a, b, c, d),
    }


class State:
    """``duck.state``。"""

    def __init__(self, duck: Duck) -> None:
        self._duck = duck

    # ---- 单帧 ---------------------------------------------------------
    def get(self, timeout: float | None = None) -> Status:
        """问一次 STAT，返回解析好的 :class:`Status`。"""
        seq = self._duck.next_seq()
        pkt = struct.pack(P.STAT_REQ_FMT, P.STATUS_GET, P.VERSION, seq)
        data = self._duck._request(P.STATUS_REPLY, pkt, timeout)
        if len(data) < P.STATUS_V1_LEN:
            raise DuckCommandError(f"SSTA 长度 {len(data)}B 不对")
        return Status.from_frame(data)

    def watch(self, interval: float = 0.5, timeout: float | None = None):
        """按 ``interval`` 秒轮询，逐帧 yield :class:`Status`。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        while deadline is None or time.monotonic() < deadline:
            yield self.get()
            time.sleep(interval)

    def wait_for(self, pred, timeout: float = 10.0, interval: float = 0.1) -> Status:
        """轮询到 ``pred(status)`` 为真，返回那一帧；超时抛 :class:`DuckTimeoutError`。"""
        deadline = time.monotonic() + timeout
        last: Status | None = None
        while time.monotonic() < deadline:
            last = self.get()
            if pred(last):
                return last
            time.sleep(interval)
        state = f"（最后状态 {last.state_cn}）" if last is not None else ""
        raise DuckTimeoutError(f"{timeout:.1f} 秒内条件没有满足{state}")

    # ---- 事件日志 -----------------------------------------------------
    def _event_page(self, before_seq: int, timeout: float | None) -> tuple[int, int, int, list[tuple]]:
        seq = self._duck.next_seq()
        pkt = struct.pack(P.EGET_FMT, P.EVENT_GET, P.VERSION, seq, before_seq)
        data = self._duck._request(P.EVENT_LOG, pkt, timeout)
        head_len = struct.calcsize(P.ELOG_HEAD_FMT)
        rec_len = struct.calcsize(P.EVENT_FMT)
        if len(data) < head_len:
            raise DuckCommandError(f"ELOG 长度 {len(data)}B 不对")
        magic, _version, _seq, uptime_ms, newest_seq, count, flags, _reserved = \
            struct.unpack_from(P.ELOG_HEAD_FMT, data)
        if magic != P.EVENT_LOG:
            raise DuckCommandError(f"ELOG magic 不对：{magic!r}")
        if count > P.EVENT_MAX_RECORDS or len(data) < head_len + count * rec_len:
            raise DuckCommandError(f"ELOG 记录数 {count} 与包长 {len(data)}B 不符")
        records = [struct.unpack_from(P.EVENT_FMT, data, head_len + i * rec_len)
                   for i in range(count)]
        return uptime_ms, newest_seq, flags, records

    def _newest_seq(self, timeout: float | None = None) -> int:
        _uptime, newest, _flags, _records = self._event_page(0, timeout)
        return newest

    def _fetch_new(self, after_seq: int, timeout: float | None) -> tuple[int, list[tuple]]:
        """取 ``seq > after_seq`` 的全部新记录（必要时向旧翻页）。"""
        records: list[tuple] = []
        before = 0
        newest = 0
        while True:
            _uptime, newest, _flags, page = self._event_page(before, timeout)
            records.extend(record for record in page if record[0] > after_seq)
            if not page or len(page) < P.EVENT_MAX_RECORDS or page[-1][0] <= after_seq:
                break
            before = page[-1][0]
        records.sort(key=lambda record: record[0])
        return newest, records

    def _backfill(self, since: int | None, timeout: float | None,
                  limit: int | None = None) -> tuple[list[tuple], int, int]:
        """从新往旧翻页，收集 ``seq > since`` 的记录（按 seq 升序）。

        ``limit`` 给定后凑够即停（events() 的初始回溯用它防止拖回全部历史）。
        """
        records: list[tuple] = []
        before = 0
        newest = 0
        while True:
            _uptime, newest, _flags, page = self._event_page(before, timeout)
            if not page:
                break
            for record in page:
                if since is None or record[0] > since:
                    records.append(record)
            if limit is not None and len(records) >= limit:
                break
            oldest = page[-1][0]
            if since is not None and oldest <= since:
                break
            if len(page) < P.EVENT_MAX_RECORDS:
                break
            before = oldest
        records.sort(key=lambda record: record[0])
        return records, newest, before

    def history(self, limit: int = 64) -> list[dict]:
        """取最近的 ``limit`` 条事件（seq 升序）。只向后翻页到凑够为止，不会拖回全部历史。

        一次性查看用这个；持续跟随用 :meth:`events`——那是无限生成器，别对它 ``list()``。
        """
        if limit <= 0:
            return []
        records, _newest, _before = self._backfill(None, self._duck.timeout, limit)
        return [_event_dict(record) for record in records[-limit:]]

    def events(self, since: int | None = None, interval: float = 0.5, max_backfill: int | None = 32):
        """增量跟随事件日志（无限生成器，不要对它 ``list()``；取固定条数用 :meth:`history`）。

        第一次先回溯至多 ``max_backfill`` 条现有记录（``None`` 不限制，历史很长时慎用）按从旧到新
        yield 出来，之后每 ``interval`` 秒查一次最新一页，只 yield 新出现的。事件环形缓冲被覆盖导致
        seq 跳号时，跳过的条数放在下一条事件的 ``missed_before`` 字段里。
        """
        records, newest, _before = self._backfill(since, self._duck.timeout, max_backfill)
        if max_backfill is not None:
            records = records[-max_backfill:] if max_backfill > 0 else []
        last = since if since is not None else newest
        for record in records:
            last = max(last, record[0])
            yield _event_dict(record)
        missed = 0
        while True:
            time.sleep(interval)
            newest, fresh = self._fetch_new(last, self._duck.timeout)
            if newest < last:          # 事件日志被擦过（恢复出厂）：从头再来
                last = newest
                continue
            if fresh:
                gap = newest - last - len(fresh)
                if gap > 0:
                    missed += gap
                for record in fresh:
                    last = record[0]
                    event = _event_dict(record)
                    if missed:
                        event["missed_before"] = missed
                        missed = 0
                    yield event
            elif newest > last:
                missed += newest - last
                last = newest

    def wait_event(self, kind: int | str, timeout: float = 30.0, interval: float = 0.2) -> dict:
        """等到一条指定种类的新事件（``kind`` 可为编号或 ``"POLICY_RUN"`` 这样的名字）。

        从调用这一刻之后出现的记录里找，已经发生的不算。超时抛 :class:`DuckTimeoutError`。
        """
        if isinstance(kind, str):
            wanted = next((code for code, name in P.EVENT_KINDS.items() if name == kind), None)
            if wanted is None:
                raise ValueError(f"未知事件名 {kind!r}")
            kind = wanted
        last = self._newest_seq()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            newest, fresh = self._fetch_new(last, self._duck.timeout)
            for record in fresh:
                last = record[0]
                if record[2] == kind:
                    return _event_dict(record)
            if newest > last:
                last = newest
            time.sleep(interval)
        raise DuckTimeoutError(f"{timeout:.1f} 秒内没等到事件 {P.EVENT_KINDS.get(kind, kind)}")
