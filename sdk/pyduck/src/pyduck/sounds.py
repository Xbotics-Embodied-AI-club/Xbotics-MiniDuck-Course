"""叫声（85 口）：状态与播放。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import _http

if TYPE_CHECKING:
    from . import Duck


class Sounds:
    """``duck.sounds``。"""

    def __init__(self, duck: Duck) -> None:
        self._duck = duck

    def status(self, timeout: float = 10.0) -> dict[str, Any]:
        """GET /status：音频是否就绪、已载入几段、可用的标签。"""
        return _http.get_json(self._duck, "sound", "/status", timeout=timeout)

    def play(self, tag: str, timeout: float = 10.0) -> dict[str, Any]:
        """POST /play：请求播放一个叫声标签（alarm / greet / inquire / peck / chirp / coo / wheee）。"""
        return _http.post_json(self._duck, "sound", "/play",
                               params={"tag": tag}, timeout=timeout)
