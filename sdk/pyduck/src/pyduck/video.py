"""视频流（预留）：OV2640 还未接入，v1 只留接口。"""

from __future__ import annotations

_NOT_READY = "OV2640 未接入；规划见 docs/camera-ov2640-feasibility.md"


class VideoStream:
    """MJPEG 视频流的接口约定。

    规划中的分辨率只有 ``"qvga"``（320×240）与 ``"vga"``（640×480），帧率 1..25。
    当前固件没有摄像头通路，构造即抛 :class:`NotImplementedError`。
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(_NOT_READY)

    def open(self, resolution: str = "qvga", fps: int = 25) -> None:
        """打开视频流。"""
        raise NotImplementedError(_NOT_READY)

    def read_frame(self) -> bytes:
        """取一帧 MJPEG（JPEG 字节，SOI…EOI）。"""
        raise NotImplementedError(_NOT_READY)

    def close(self) -> None:
        """关闭视频流。"""
        raise NotImplementedError(_NOT_READY)
