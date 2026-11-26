"""板上文件（83 口）：列表、上传、删除、恢复出厂、重启。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import _http
from ._proto import P
from .errors import DuckCommandError, DuckError, DuckTimeoutError

if TYPE_CHECKING:
    from . import Duck
    from .state import Status

_NOT_WRITABLE = "机器人未卸力，请先卸力（motion.limp()）再写文件"


def files_writable(status: Status) -> bool:
    """按 SSTA 预判文件区能不能写。

    固件判据是 ``robotFlashQuiet() && state != FALLEN``：待机、卸力为真；标定 / 舵机编号
    不正在动也为真。站立、走路、坐着、起身中都在驱动舵机，不能写。
    """
    if status.writable:
        return True
    return status.state in (0, 7, 8, 10)      # IDLE / LIMP / CALIBRATING / SERVO_SETUP


class Files:
    """``duck.files``。"""

    def __init__(self, duck: Duck) -> None:
        self._duck = duck

    def list(self, timeout: float = 10.0) -> dict[str, Any]:
        """GET /fs：文件列表、用量、可写状态、策略槽与叫声统计。"""
        return _http.get_json(self._duck, "files", P.FS_LIST_PATH, timeout=timeout)

    def upload(self, name: str, data_or_path: bytes | str | Path,
               precheck: bool = True) -> dict[str, Any]:
        """上传一个文件（``.mdp`` → /policy，``.wav`` → /audio）。

        Args:
            name: 板上文件名（含扩展名）。
            data_or_path: 文件内容（bytes）或本地路径。
            precheck: 先问一次 SSTA，按 ``robotFilesWritable`` 预判；不能写就直接拒绝，
                不让请求打到板上。
        """
        if precheck:
            status = self._duck.state.get()
            if not files_writable(status):
                raise DuckCommandError(_NOT_WRITABLE)
        data = data_or_path.read_bytes() if isinstance(data_or_path, (str, Path)) else data_or_path
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("data_or_path 必须是 bytes 或路径")
        return _http.post_json(
            self._duck, "files", P.FS_UPLOAD_PATH,
            data=bytes(data), headers={P.FS_UPLOAD_NAME_HEADER: name}, timeout=120.0,
        )

    def delete(self, path: str) -> dict[str, Any]:
        """删除一个上传的文件或旧的上电记录（出厂文件删不掉）。"""
        return _http.post_json(self._duck, "files", P.FS_DELETE_PATH,
                               params={"path": path}, timeout=15.0)

    def restore_factory(self) -> dict[str, Any]:
        """删掉全部上传的策略与叫声并重启（恢复出厂）。"""
        return _http.post_json(self._duck, "files", P.FS_RESTORE_PATH, timeout=30.0)

    def reboot(self, confirm: bool = True, timeout: float = 30.0) -> Status | dict[str, Any]:
        """POST /reboot；``confirm`` 时轮询 SSTA 的 ``boot_seq`` 变大。

        Raises:
            DuckCommandError: 固件拒绝重启（例如机器人没卸力）。
            DuckTimeoutError: ``timeout`` 秒内没看到 ``boot_seq`` 变化。
        """
        before = self._duck.state.get().boot_seq
        reply = _http.post_json(self._duck, "files", P.FS_REBOOT_PATH, timeout=15.0)
        if not confirm:
            return reply
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                status = self._duck.state.get(timeout=1.0)
            except DuckError:
                time.sleep(0.2)
                continue
            if status.boot_seq != before:
                return status
            time.sleep(0.2)
        raise DuckTimeoutError(f"重启后 {timeout:.0f} 秒内没看到 bootSeq 变化")
