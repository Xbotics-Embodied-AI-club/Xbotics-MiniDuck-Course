"""自定义策略：列表、上传、热载、运行、停止（PCMD → PRSP）。

训练 / 导出约定与训练侧一致：观测 61 维、动作 14 维；文件由 ``tools/assets.py policy`` 生成，
名字 2..15 个字符，只允许 ``[A-Za-z0-9_-]``。上传前在本地把 128 字节文件头核一遍，
不合格就不发到板上。
"""

from __future__ import annotations

import re
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from . import _http
from ._proto import P
from .errors import DuckCommandError, DuckError

if TYPE_CHECKING:
    from . import Duck
    from .state import Status

POLICY_MAGIC = b"MDPL"                 # lib/policy/src/policy_int8_blob.h
POLICY_BLOB_VERSION = 1
POLICY_HEADER_BYTES = 128
POLICY_OBS_DIM = 61                    # 训练侧观测 61 维
POLICY_ACT_DIM = 14                    # 训练侧动作 14 维
POLICY_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{2,15}$")
POLICY_SOURCES = {0: "none", 1: "uploaded", 2: "factory"}


@dataclass(frozen=True)
class PolicyInfo:
    idx: int
    status: int
    obs_dim: int
    act_dim: int
    source: int
    name: str
    running: bool = False

    @property
    def status_name(self) -> str:
        return P.POLICY_ENTRY_STATUS.get(self.status, f"UNKNOWN({self.status})")

    @property
    def source_name(self) -> str:
        return POLICY_SOURCES.get(self.source, f"unknown({self.source})")


@dataclass(frozen=True)
class PolicyReply:
    result: int
    running_idx: int
    running_remain_ms: int
    entries: list[PolicyInfo] = field(default_factory=list)

    @property
    def result_text(self) -> str:
        return P.PCMD_RESULTS.get(self.result, f"未知结果 {self.result}")


class Policy:
    """``duck.policy``。"""

    def __init__(self, duck: Duck) -> None:
        self._duck = duck

    # ---- 收发 ---------------------------------------------------------
    def _command(self, cmd: int, *, idx: int = 0, duration_ms: int = 0,
                 scale: float = 1.0, name: str = "", timeout: float | None = None) -> PolicyReply:
        encoded = name.encode("ascii", "strict") if name else b""
        if len(encoded) > P.POLICY_NAME_MAX:
            raise ValueError(f"策略名最长 {P.POLICY_NAME_MAX} 个字符")
        payload = encoded.ljust(16, b"\0")
        seq = self._duck.next_seq()
        pkt = struct.pack(P.PCMD_FMT, P.POLICY_CMD, P.VERSION, seq, cmd, idx, 0,
                          duration_ms, scale, payload)
        data = self._duck._request(P.POLICY_REPLY, pkt, timeout)
        expected = struct.calcsize(P.PRSP_FMT)
        if len(data) != expected:
            raise DuckCommandError(f"PRSP 长度 {len(data)}B 不对（应为 {expected}B）")
        fields = struct.unpack(P.PRSP_FMT, data)
        result, count, running_idx, remain = fields[3], fields[4], fields[5], fields[6]
        entries: list[PolicyInfo] = []
        for i in range(min(count, P.POLICY_MAX_TOTAL)):
            base = 7 + i * 7
            idx_, status, obs_dim, act_dim, source, _reserved, raw_name = fields[base:base + 7]
            entries.append(PolicyInfo(
                idx=idx_,
                status=status,
                obs_dim=obs_dim,
                act_dim=act_dim,
                source=source,
                name=raw_name.rstrip(b"\0").decode("ascii", "replace"),
                running=(status == 2 and idx_ == running_idx),
            ))
        return PolicyReply(result=result, running_idx=running_idx,
                           running_remain_ms=remain, entries=entries)

    def _checked(self, reply: PolicyReply) -> PolicyReply:
        if reply.result != 0:
            raise DuckCommandError(reply.result_text)
        return reply

    # ---- 列表 ---------------------------------------------------------
    def list(self) -> list[PolicyInfo]:
        """PCMD LIST：注册表里全部条目（最多 12 个）。"""
        return self._checked(self._command(P.PCMD_CMDS["LIST"])).entries

    # ---- 上传 ---------------------------------------------------------
    def _precheck(self, blob: bytes, filename: str) -> None:
        if len(blob) < POLICY_HEADER_BYTES:
            raise DuckCommandError(f"{filename}：比策略文件头（128B）还短")
        (magic, version, header_bytes, payload_len, _crc, hardware, slot_raw, _name,
         obs_dim, act_dim) = struct.unpack_from(P.POLICY_HEADER_FMT, blob)[:10]
        if magic != POLICY_MAGIC:
            raise DuckCommandError(f"{filename}：不是策略文件（没有 MDPL 标记）")
        if version != POLICY_BLOB_VERSION or header_bytes != POLICY_HEADER_BYTES:
            raise DuckCommandError(f"{filename}：格式版本 {version}/头长 {header_bytes} 不认识")
        hardware_id = hardware.rstrip(b"\0").decode("ascii", "replace")
        if not hardware_id:
            raise DuckCommandError(f"{filename}：文件头里没有硬件版本")
        slot = slot_raw.rstrip(b"\0").decode("ascii", "replace")
        if slot != Path(filename).stem:
            raise DuckCommandError(
                f"{filename}：文件头里的模型名是 {slot!r}，文件名是 {Path(filename).stem!r}，"
                f"两者必须一致"
            )
        if not POLICY_NAME_RE.match(slot):
            raise DuckCommandError(f"{filename}：策略名 {slot!r} 不合规（2..15 个 [A-Za-z0-9_-]）")
        if obs_dim != POLICY_OBS_DIM or act_dim != POLICY_ACT_DIM:
            raise DuckCommandError(
                f"{filename}：观测/动作维度 {obs_dim}/{act_dim}，本固件要求 "
                f"{POLICY_OBS_DIM}/{POLICY_ACT_DIM}"
            )
        if payload_len % 16 or len(blob) < POLICY_HEADER_BYTES + payload_len:
            raise DuckCommandError(f"{filename}：数据区长度 {payload_len}B 不对（文件被截断？）")

    def upload(self, path: str | Path) -> dict:
        """上传 ``.mdp``；先在本地核对文件头，再 POST 到文件页（83 口）。"""
        local = Path(path)
        blob = local.read_bytes()
        self._precheck(blob, local.name)
        body = _http.post_json(
            self._duck, "files", P.FS_UPLOAD_PATH,
            data=blob, headers={P.FS_UPLOAD_NAME_HEADER: local.name}, timeout=120.0,
        )
        return body

    # ---- 载入 / 运行 --------------------------------------------------
    def load(self, name: str) -> PolicyReply:
        """PCMD LOAD：把 ``/policy/<name>.mdp`` 热载进注册表。"""
        if not POLICY_NAME_RE.match(name):
            raise ValueError(f"策略名 {name!r} 不合规（2..15 个 [A-Za-z0-9_-]）")
        return self._checked(self._command(P.PCMD_CMDS["LOAD"], name=name))

    def run(self, name_or_idx: str | int, seconds: float | None = None,
            scale: float = 1.0) -> Status:
        """运行一个已注册策略，并阻塞到它离开 SKILL 状态。

        Args:
            name_or_idx: 策略名或注册表下标。
            seconds: 必填，1..60 秒（固件单次运行上限 60 s）。
            scale: 动作缩放，0.05..1.5。

        Raises:
            DuckCommandError: 固件拒绝（未站稳 / 没找到 / 参数越限等）。
            ValueError: 参数不合法。
        """
        if seconds is None:
            raise ValueError("seconds 必填（1..60 秒）")
        if not 1.0 <= float(seconds) <= P.POLICY_RUN_MS_MAX / 1000.0:
            raise ValueError(f"seconds 必须在 1..{P.POLICY_RUN_MS_MAX // 1000} 秒之间")
        if not P.POLICY_RUN_SCALE_MIN <= scale <= P.POLICY_RUN_SCALE_MAX:
            raise ValueError(
                f"scale 必须在 {P.POLICY_RUN_SCALE_MIN}..{P.POLICY_RUN_SCALE_MAX} 之间")
        idx = name_or_idx if isinstance(name_or_idx, int) else 0
        if isinstance(name_or_idx, int) and not 0 <= name_or_idx < P.POLICY_MAX_TOTAL:
            raise ValueError(f"策略下标必须在 0..{P.POLICY_MAX_TOTAL - 1} 之间")
        name = "" if isinstance(name_or_idx, int) else str(name_or_idx)
        if name and not POLICY_NAME_RE.match(name):
            raise ValueError(f"策略名 {name!r} 不合规（2..15 个 [A-Za-z0-9_-]）")
        self._checked(self._command(
            P.PCMD_CMDS["RUN"], idx=idx, duration_ms=int(float(seconds) * 1000),
            scale=float(scale), name=name,
        ))
        return self._wait_until_skill_over(seconds)

    def _wait_until_skill_over(self, seconds: float) -> Status:
        deadline = time.monotonic() + float(seconds) + 5.0
        started = time.monotonic()
        seen_skill = False
        last = self._duck.state.get()
        while time.monotonic() < deadline:
            try:
                status = self._duck.state.get(timeout=0.5)
            except DuckError:
                time.sleep(0.1)
                continue
            last = status
            if status.state == 11:                 # SKILL
                seen_skill = True
            elif seen_skill:
                return status
            elif time.monotonic() - started > 2.0:
                # 固件接受了 RUN 却一直没进 SKILL（例如已被别的命令取消）：不必等满时限
                return status
            time.sleep(0.1)
        return last

    def stop(self) -> PolicyReply:
        """PCMD STOP：停止当前运行的策略。"""
        return self._checked(self._command(P.PCMD_CMDS["STOP"]))
