"""上电记录（调试页 84 口）：会话列表与下载。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import _http
from ._proto import P
from .errors import DuckCommandError

if TYPE_CHECKING:
    from . import Duck


class Records:
    """``duck.records``。"""

    def __init__(self, duck: Duck) -> None:
        self._duck = duck

    def sessions(self, timeout: float = 10.0) -> dict[str, Any]:
        """GET /records：本次与历史开机记录。"""
        return _http.get_json(self._duck, "debug", "/records", timeout=timeout)

    def download(self, session: str | int = "current", save_path: str | Path | None = None,
                 timeout: float = 60.0) -> int:
        """把一份上电记录流式写到 ``save_path``，返回字节数。

        ``session="current"`` 是本次开机；数字是之前某次开机的序号。空文件抛
        :class:`DuckCommandError`（板子没有这份记录）。
        """
        if save_path is None:
            raise ValueError("save_path 必填（记录要落到本地文件）")
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        response = _http.get_stream(self._duck, "debug", P.RECORD_PATH,
                                    params={"session": str(session)}, timeout=timeout)
        try:
            with target.open("wb") as handle:
                for chunk in response.iter_content(65536):
                    if chunk:
                        handle.write(chunk)
                        written += len(chunk)
        finally:
            response.close()
        if written == 0:
            target.unlink(missing_ok=True)
            raise DuckCommandError(f"没有这份上电记录（session={session}）")
        return written
