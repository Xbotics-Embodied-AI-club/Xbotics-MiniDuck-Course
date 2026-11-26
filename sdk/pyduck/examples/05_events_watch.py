# -*- coding: utf-8 -*-
"""例程 05：事件日志——历史、等待、有界跟随。

    python 05_events_watch.py --ip 192.168.1.150

演示三种用法：history(n) 取最近 n 条；wait_event() 等某类事件发生；
events() 跟随新事件（无限生成器，必须自己设界退出）。
"""
import argparse

from pyduck import Duck

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="192.168.4.1")
parser.add_argument("--watch", default="FALL", help="等待的事件名或编号，如 FALL / GETUP_DONE / 5")
args = parser.parse_args()

with Duck.connect(args.ip, timeout=4) as duck:
    print("最近 10 条事件：")
    for e in duck.state.history(10):
        print(" ", e["text"])

    kind = args.watch
    print(f"等一条 {kind}（60 s 超时）…… 等着的时候可以推一下鸭子")
    try:
        e = duck.state.wait_event(int(kind) if kind.isdigit() else kind, timeout=60)
        print("等到了:", e["text"])
    except Exception as ex:
        print(f"没等到（{type(ex).__name__}）：鸭子站得稳也算正常结果")

    print("跟随 5 条新事件后退出（events() 是无限生成器，必须设界）：")
    n = 0
    for e in duck.state.events(max_backfill=0):     # 0 = 不回溯历史，只跟新的
        print(" ", e["text"])
        n += 1
        if n >= 5:
            break
