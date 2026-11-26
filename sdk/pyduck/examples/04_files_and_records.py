# -*- coding: utf-8 -*-
"""例程 04：板上文件与上电记录。

    python 04_files_and_records.py --ip 192.168.1.150

写 flash（上传/删除/恢复出厂/重启）只在卸力或待机时 allowed——机器人站着跑会被
固件拒绝并返回中文原因。本例程只做安全的读取与下载。
"""
import argparse
from pathlib import Path

from pyduck import Duck

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="192.168.4.1")
parser.add_argument("--out", default="boot_record.mdr")
args = parser.parse_args()

with Duck.connect(args.ip, timeout=4) as duck:
    fs = duck.files.list()
    print(f"文件区 {fs['used'] / 1024:.0f}/{fs['total'] / 1024:.0f} KiB | 可写={fs['writable']} {fs.get('why', '')}")
    for f in fs["files"]:
        print(f"  {f['path']:<40} {f['size']:>8} B  factory={f['factory']}")
    for pol in fs["policies"]:
        print(f"  核心策略 {pol['slot']:<14} {pol['source']}")

    # 上电记录：50 Hz 运动帧 + 健康帧 + 文本日志，事后整份下载解码
    out = duck.records.download("current", Path(args.out))
    print("已下载本次上电记录:", out)
    print("解码: uv run firmware/tools/boot_record.py", args.out)

    # 写操作的正确姿势（鸭子站着时会被 400 拒绝，这是保护不是故障）：
    # duck.files.upload("my-policy.mdp", Path("my-policy.mdp"))   # 或带预检的 duck.policy.upload()
    # duck.files.delete("/policy/my-policy.mdp")
    # duck.files.reboot(confirm=True)   # 轮询 boot_seq 确认真重启了，回复 200 不算数
