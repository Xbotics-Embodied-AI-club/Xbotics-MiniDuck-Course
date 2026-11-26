# -*- coding: utf-8 -*-
"""例程 01：只读整机状态与传感器。

    python 01_state_and_sensors.py --ip 192.168.1.150

鸭子站起来之后跑（卸力/待机时关节读数是全 NaN）。本例程纯读取，不产生任何运动。
"""
import argparse
import math
import time

from pyduck import Duck

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="192.168.4.1")
args = parser.parse_args()

with Duck.connect(args.ip, timeout=4) as duck:
    st = duck.state.get()
    print(f"硬件 {st.hardware_id} | 状态 {st.state_cn} | 闭环 {st.loop_hz:.1f} Hz | "
          f"电池 {st.battery_v:.2f} V | 舵机在线 {st.servo_count}/15")

    # ToF：头部正前方测距。-1 = 无目标/离线；60 mm 以内被近场门槛丢弃（原厂镜头盖假回波）
    for _ in range(3):
        st = duck.state.get()
        print(f"ToF {st.tof_mm} mm (valid={st.tof_valid})")
        time.sleep(1)

    # IMU：身体重力是躯干系单位向量（直立 ≈ (0,0,-1)）；
    # 头部是芯片系未归一化原始值，模长 ≈ 1000 mg（坑 #76），两者语义不同别混用
    st = duck.state.get()
    print("身体重力(躯干系)", tuple(round(v, 3) for v in st.gravity))
    if st.head_gravity:
        mag = math.sqrt(sum(v * v for v in st.head_gravity))
        print(f"头部重力(芯片系) 模长 {mag:.0f} mg")
    print("身体角速度 rad/s", tuple(round(v, 3) for v in st.gyro))
    print("头部角速度 rad/s", tuple(round(v, 3) for v in st.head_gyro))

    # 舵机：当前位置 / 下发目标（rad）与健康诊断（电压、电流、温度、故障位）
    snap = duck.servos.snapshot()
    for sid in sorted(snap):
        s = snap[sid]
        print(f"  {s.id:>2} {s.name:<6} pos={s.position:+.3f} rad  tgt={s.target:+.3f} rad")
    diag = duck.servos.diagnostics()
    print(f"故障总数 {diag.fault_total}，总线电压 {diag.voltage_min_v:.1f}~{diag.voltage_max_v:.1f} V")
