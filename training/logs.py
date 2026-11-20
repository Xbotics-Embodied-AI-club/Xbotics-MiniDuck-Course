"""屏幕日志的唯一配置处：各模块用 `logging.getLogger(__name__)`，入口用 `run_cli`。

屏幕日志给看终端的人读：一眼知道起没起来、哪里不对、要不要动手。级别按后果定 ——
没人需要动手的不高于 INFO；结果还在出但降级了是 WARNING；这件事没做成是 ERROR；
程序要退出是 CRITICAL。逐步的数一律 DEBUG，诊断数据落文件不上屏。

第三方库打到屏幕上的东西也归这里管：能用它自己的开关关到 WARNING 的就关，
关不掉的写明是谁打的、为什么可以不看。目前关不掉的只有一处：

- 开头的三行 `[mdp] Patch … active`：上游 microduck_rl 的 mdp 模块在被导入时用 print 报
  它给 mjlab 打了哪几个补丁。`import mjlab` 时就按 entry point 自动导入了它，早于本仓任何
  代码，没有开关。说明性质，每次启动一样，可以不看。
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from collections.abc import Callable

FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
# 调到 DEBUG 才看得见的东西：mjlab 建环境时那十几张管理器表格（见 env.py）、逐轮的数。
LEVEL_ENV = "MINIDUCK_LOG_LEVEL"

log = logging.getLogger(__name__)


def setup_logging() -> None:
    """配置根 logger，并把第三方库的无害输出关到 WARNING。

    Raises:
        ValueError: `MINIDUCK_LOG_LEVEL` 不是合法的级别名。
    """
    name = os.environ.get(LEVEL_ENV, "INFO").upper()
    levels = logging.getLevelNamesMapping()
    if name not in levels:
        raise ValueError(f"{LEVEL_ENV}={name!r} 不是日志级别（DEBUG / INFO / WARNING / ERROR）")
    level = levels[name]
    logging.basicConfig(level=level, format=FORMAT, datefmt="%H:%M:%S", force=True)
    # warnings.warn 的输出改走日志（logger 名 py.warnings，级别 WARNING），不再裸写 stderr。
    logging.captureWarnings(True)
    # Lightning 开训时报「有几张卡、用了几张」这类 INFO，与本仓的就位行重复。
    for noisy in ("lightning.pytorch", "lightning.fabric"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # 上游的执行器配置用 `^(?!passive_).*` 选关节，同一个正则也匹配到 7 个 site；
    # 上游自己训练时也报这一条，执行器只作用在关节上，不影响结果。
    warnings.filterwarnings("ignore", message=r"Actuator config matched \d+ joint", category=UserWarning)
    # Lightning 建议给 DataLoader 开多个 worker。本仓的数据是在 GPU 上、在本进程里由仿真
    # 现采的，多进程既用不上也接不住那块显存。
    warnings.filterwarnings("ignore", message=r".*'train_dataloader' does not have many workers")
    # torch 提示 TorchScript 版 ONNX 导出将来会退役。导出刻意走这条路：上游 mjlab 的导出用的
    # 就是它（dynamo=False），产物的图结构才与上游运行时吃的一致。
    warnings.filterwarnings("ignore", message=r"You are using the legacy TorchScript-based ONNX export",
                            category=DeprecationWarning)


def run_cli(main: Callable[[], None]) -> None:
    """命令行入口的统一外壳：先配日志，再跑；出错时记一条 CRITICAL（带调用栈）后以 1 退出。

    Args:
        main: 入口函数。
    """
    setup_logging()
    try:
        main()
    except Exception:
        log.critical("%s 没有跑完，退出", os.path.basename(sys.argv[0]), exc_info=True)
        raise SystemExit(1) from None
