"""本仓的路径与运行身份 —— 不依赖仿真器的那一小块。

单独一个文件，是为了让测试、导出校验这些不需要 GPU 的步骤也能拿到同一套路径，
而不必先 import 整个 mjlab。
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# 这只鸭子在仿真里的物理，按硬件版本组织（硬件版本 / 机体结构 / 电机），读法见 hardware.py。
SIM_MODEL = REPO_ROOT / "sim_model"
# 交付给真机的策略文件（ONNX + int8 头文件 + 清单），进 git，按硬件版本分：policies/<硬件版本>/<动作>/。
POLICY_DIR = REPO_ROOT / "policies"


def file_sha256(path: Path) -> str:
    """文件内容的 sha256，用来把一份权重钉到它训练时的那份文件上。

    Args:
        path: 文件路径。

    Returns:
        十六进制摘要。
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_name() -> str:
    """这次训练叫什么，同时是权重目录名与 W&B run 名。

    名字由调用方声明（`MINIDUCK_RUN_NAME`），程序不替它编一个：
    拿时间戳或计数器现拼的名字永远不会冲突、也永远说不清它是哪一次。

    Returns:
        声明的 run 名。

    Raises:
        RuntimeError: 没有声明 `MINIDUCK_RUN_NAME`。
    """
    name = os.environ.get("MINIDUCK_RUN_NAME", "").strip()
    if not name:
        raise RuntimeError("没有给 MINIDUCK_RUN_NAME（这次训练的名字，也是权重目录名）")
    return name


def checkpoint_dir(name: str) -> Path:
    """某次训练的权重目录。

    权重是大文件，落在共享盘上、不进 git；根目录由 `.envrc` 的 `MINIDUCK_CKPT_ROOT` 给出。

    Args:
        name: run 名。

    Returns:
        `<MINIDUCK_CKPT_ROOT>/<name>`。

    Raises:
        RuntimeError: 没有 `MINIDUCK_CKPT_ROOT`（没 source `.envrc`）。
    """
    root = os.environ.get("MINIDUCK_CKPT_ROOT", "").strip()
    if not root:
        raise RuntimeError("没有 MINIDUCK_CKPT_ROOT —— 先 `source .envrc`")
    return Path(root) / name


def result_dir(name: str) -> Path:
    """评测摘要与录像落在哪。

    默认是仓内的 `result/<run 名>/`（已 gitignore）；用 `MINIDUCK_RESULT_ROOT` 换根目录。

    Args:
        name: run 名。

    Returns:
        结果目录（已建好）。
    """
    override = os.environ.get("MINIDUCK_RESULT_ROOT", "").strip()
    base = Path(override) if override else REPO_ROOT / "result"
    out = base / name
    out.mkdir(parents=True, exist_ok=True)
    return out
