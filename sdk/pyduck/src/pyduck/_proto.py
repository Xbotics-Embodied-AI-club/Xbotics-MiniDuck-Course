"""加载固件协议定义（`tools/protocol.py`）。

固件协议、板上文件格式、事件编码的**唯一来源**是固件仓库的 ``firmware/tools/protocol.py``。
SDK 里不再抄任何格式串或常量，只从这里取，避免两头各写一份、错位了却解出一堆看似正常的数。

解析顺序：

1. 环境变量 ``PYDUCK_PROTOCOL`` 显式指定；
2. 从本文件向上找 ``firmware/tools/protocol.py``（仓库内开发：sdk/pyduck 与 firmware/ 同仓）；
3. 都不在（独立分发安装、只带了 pyduck 包）时用内置快照
   ``pyduck._protocol_snapshot``——它是 protocol.py 的拷贝，发布时同步刷新；
   仓库里的 ``tests/test_pyduck.py`` 会核对快照与正本一致。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_MODULE_NAME = "pyduck._firmware_protocol"


def _repo_protocol_candidates() -> list[Path]:
    """从本文件往上每一级找 ``firmware/tools/protocol.py``。"""
    here = Path(__file__).resolve()
    return [parent / "firmware" / "tools" / "protocol.py" for parent in here.parents]


def protocol_path() -> Path | None:
    """正本 ``tools/protocol.py`` 的位置；独立分发（找不到正本）时为 ``None``。"""
    override = os.environ.get("PYDUCK_PROTOCOL")
    if override:
        return Path(override).expanduser()
    for candidate in _repo_protocol_candidates():
        if candidate.is_file():
            return candidate
    return None


def load_protocol(path: str | Path | None = None):
    """加载并返回 ``protocol`` 模块（正本优先，独立分发用内置快照）。

    Raises:
        ImportError: 指定的文件不存在或无法作为 Python 模块加载，消息里带完整路径。
    """
    target = Path(path).expanduser() if path is not None else protocol_path()
    if target is None:
        from . import _protocol_snapshot
        return _protocol_snapshot
    if not target.is_file():
        raise ImportError(
            f"找不到固件协议定义 {target}。pyduck 需要和固件仓库在一起（能找到 "
            f"firmware/tools/protocol.py），或用环境变量 PYDUCK_PROTOCOL 指向它；"
            f"独立分发安装则使用内置协议快照。"
        )
    cached = sys.modules.get(_MODULE_NAME)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, target)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载固件协议定义 {target}（模块加载器不可用）")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_MODULE_NAME, None)
        raise
    return module


P = load_protocol()
PROTOCOL_FILE = protocol_path()
