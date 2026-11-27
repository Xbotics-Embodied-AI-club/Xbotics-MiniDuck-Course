# -*- coding: utf-8 -*-
"""例程 02：行走与 ToF 避障闭环。

    python 02_drive_and_avoid.py --ip 192.168.1.150

鸭子须先站稳。前进中 SDK 自己轮询 ToF，距障碍 200 mm 内停车——设备不会主动推
传感器，闭环要放在上位机侧做（小智协议同理：LLM 下参数、设备/上位机做闭环）。
"""
import argparse
import time

from pyduck import Duck

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="192.168.4.1")
parser.add_argument("--stop-mm", type=int, default=200)
args = parser.parse_args()

with Duck.connect(args.ip, timeout=4) as duck:
    duck.motion.stand()
    duck.state.wait_for(lambda s: s.state_cn == "站立", timeout=15)
    print("已站稳，前进到障碍", args.stop_mm, "mm 内停止")

    with duck.motion.drive() as d:          # 后台 20 Hz 持续发 DCMD
        d.twist(vx=0.20)                    # 0.2 m/s 直行
        while True:
            st = duck.state.get()
            if 0 <= st.tof_mm < args.stop_mm:
                break
            time.sleep(0.05)
    # 退出 drive()：0.5 s 匀减速 + 补全零帧，鸭子原地站住（不要急停，容易摔）
    print("停下，ToF =", duck.state.get().tof_mm, "mm")

    duck.motion.turn_left(seconds=2.25, v=1.0)  # 左转
    time.sleep(0.5)
    duck.motion.sit()
    time.sleep(3)
    duck.motion.stand()
    duck.state.wait_for(lambda s: s.state_cn == "站立", timeout=15)
    print("完成：鸭子保持站立。收摊请 duck.motion.limp()（舵机卸力，先扶好鸭子）")
