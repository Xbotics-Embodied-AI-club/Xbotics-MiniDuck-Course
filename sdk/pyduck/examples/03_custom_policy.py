# -*- coding: utf-8 -*-
"""例程 03：自定义策略全链路——上传 → 热载 → 运行 → 事件核对。

先用 --output-root result/mypick-export 导出 ground-pick，再在仓库根目录运行。
机器人须从待机或卸力开始：

    uv run python firmware/tools/assets.py policy result/mypick-export/2026-11-26-hd1910/ground-pick --as mypick
    uv run python sdk/pyduck/examples/03_custom_policy.py --mdp result/mypick-export/2026-11-26-hd1910/ground-pick/mypick.mdp

源动作目录与清单保持不变；例程上传 mypick.mdp，并按 mypick 热载、运行，不覆盖核心槽位。
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

from pyduck import Duck
from pyduck.errors import DuckError

STATUS_NAMES = {0: "空", 1: "就绪", 2: "运行中"}

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="192.168.4.1")
parser.add_argument("--mdp", required=True, help="要上传的 .mdp 文件")
parser.add_argument("--seconds", type=float, default=8, help="运行时长 1..60 s")
args = parser.parse_args()

name = Path(args.mdp).stem

with Duck.connect(args.ip, timeout=4) as duck:
    # 上传写入 flash，须从待机或卸力开始；失败时不能改跑板上同名的旧文件。
    try:
        duck.policy.upload(args.mdp)
        print("已上传:", name)
        duck.policy.load(name)
        print("已热载:", name)
    except (DuckError, OSError, ValueError) as e:
        raise SystemExit(f"上传或热载 {name} 失败：{e}") from e
    for info in duck.policy.list():
        if info.status != 0:
            print(f"  [{info.idx}] {info.name:<16} {STATUS_NAMES[info.status]} source={info.source}")

    duck.motion.stand()
    duck.state.wait_for(lambda s: s.state_name == "STANDING", timeout=15)
    time.sleep(1.1)  # 收到站立状态后，板端仍要求持续站立满一秒。

    # run() 阻塞到策略结束；用工作线程运行，主线程才能在中途读状态。
    with ThreadPoolExecutor(max_workers=1) as worker:
        run = worker.submit(duck.policy.run, name, seconds=args.seconds)
        _, running = wait((run,), timeout=2)
        if running:
            mid = duck.state.get()
            print(f"运行中采样：state={mid.state_cn} policy槽={mid.policy}")
        try:
            end = run.result()
        except (DuckError, ValueError) as e:
            raise SystemExit(f"运行 {name} 失败：{e}") from e
    print("结束:", end.state_cn)

    for e in duck.state.history(8):         # 有界取最近几条，别对 events() 做 list()
        if e["kind"] in (14, 15):
            print(" ", e["text"])           # POLICY_RUN / POLICY_STOP 及原因
