"""WiFi 配网（文件页 83 口的 /wifi）：状态与保存凭据。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import _http
from ._proto import P

if TYPE_CHECKING:
    from . import Duck


class Wifi:
    """``duck.wifi``。"""

    def __init__(self, duck: Duck) -> None:
        self._duck = duck

    def status(self, timeout: float = 10.0) -> dict[str, Any]:
        """GET /wifi：热点名、已保存的 SSID、连接状态、IP、RSSI、可写状态。"""
        return _http.get_json(self._duck, "files", P.WIFI_CONFIG_PATH, timeout=timeout)

    def configure(self, ssid: str, password: str, timeout: float = 15.0) -> dict[str, Any]:
        """POST /wifi：保存手机热点凭据并开始连接（口令只写不读）。

        写 NVS 与调参同一档：机器人没卸力时固件会 400 拒绝。
        """
        return _http.post_json(self._duck, "files", P.WIFI_CONFIG_PATH,
                               params={"ssid": ssid, "pass": password}, timeout=timeout)
