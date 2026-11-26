# -*- coding: utf-8 -*-
"""例程 03：自定义策略全链路——上传 → 热载 → 运行 → 事件核对。

    python 03_custom_policy.py --ip 192.168.1.150 --mdp path/to/my-policy.mdp

.mdp 用「uv run firmware/tools/assets.py policy policies/<硬件版本>/<动作目录>/」
从训练导出转换，动作目录名即策略名：2..15 个 [A-Za-z0-9_-]，不要用 6 个核心槽位名
（standup / velstand-flat / sitstand / ground-pick / ball-kick / roulade）。
"""
import argparse
import threading
import time
from pathlib import Path

from pyduck import Duck
from pyduck.errors import DuckCommandError

STATUS_NAMES = {0: "空", 1: "就绪", 2: "运行中"}

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="192.168.4.1")
parser.add_argument("--mdp", required=True, help="要上传的 .mdp 文件")
parser.add_argument("--seconds", type=float, default=8, help="运行时长 1..60 s")
args = parser.parse_args()

name = Path(args.mdp).stem

with Duck.connect(args.ip, timeout=4) as duck:
    # 写 flash 只在卸力/待机时 allowed：鸭子站着时这两步会被拒（保护，不是故障）。
    # 想重跑而策略已在板上，用 --mdp 指同一个文件、站着直接跑即可。
    try:
        duck.policy.upload(args.mdp)    # 先本地核对 128B 文件头，不合格不会上板
        print("已上传:", name)
        duck.policy.load(name)          # 运行时热载进注册表，不重启
        print("已热载:", name)
    except DuckCommandError as e:
        print(f"上传/热载被拒（{e}），尝试直接用板上已有的 {name}")
    for info in duck.policy.list():
        if info.status != 0:
            print(f"  [{info.idx}] {info.name:<16} {STATUS_NAMES[info.status]} source={info.source}")

    duck.motion.stand()
    duck.state.wait_for(lambda s: s.state_cn == "站立", timeout=15)

    # run() 阻塞到运行结束；中途发任意运动命令或 policy.stop() 会打断（设备端 reason=2/1）
    done = {}
    t = threading.Thread(target=lambda: done.setdefault("st", duck.policy.run(name, seconds=args.seconds)))
    t.start()
    time.sleep(2)
    mid = duck.state.get()
    print(f"运行中：state={mid.state_cn} policy槽={mid.policy}")
    t.join(timeout=args.seconds + 10)
    print("结束:", done["st"].state_cn)

    for e in duck.state.history(8):         # 有界取最近几条，别对 events() 做 list()
        if e["kind"] in (14, 15):
            print(" ", e["text"])           # POLICY_RUN / POLICY_STOP 及原因
