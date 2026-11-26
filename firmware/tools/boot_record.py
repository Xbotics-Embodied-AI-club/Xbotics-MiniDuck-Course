"""解码 MiniDuck 的上电记录（调试页 84 口下载的 .mdr，或 tools/assets.py record 下载的）

    uv run firmware/tools/boot_record.py miniduck-r0012.mdr            # → miniduck-r0012/ 下：日志、表格、概况、图
    uv run firmware/tools/boot_record.py miniduck-r0012.mdr --no-plot  # 不画图

出来的东西：
    log.txt       串口日志全文（从开机第一行起）+ 事件 + 上次复位前留在 RTC 里的尾巴 + 没存进来的空档
    health.csv    每秒一帧：状态、电池电压、各舵机电压 / 电流 / 温度 / 故障、内存、主循环最慢一圈
    motion.csv    每个控制拍一帧：状态、策略、电池快采、姿态、14 个关节的位置 / 速度 / 动作 / 目标、命令
                  （位置与目标都是相对默认站姿的角度，rad；动作是策略原始输出）
    summary.txt   一眼看完：时长、最低电压、最大电流、最高温度、摔倒 / 起身经过、舵机故障、空档
    power.png     电池与舵机电压、电流、温度、状态随时间（--no-plot 不出）
    falls.png     每次摔倒前后：重力 z 与几个腿关节的目标 / 实际（有运动帧时才出）

格式定义见同目录的 protocol.py。
"""

from __future__ import annotations

import argparse
import csv
import struct
import time
from pathlib import Path

from protocol import (
    EVENT_KINDS,
    JOINT_NAMES,
    POLICY_JOINT_IDS,
    RECORD_EVENT_FMT,
    RECORD_GAP_FMT,
    RECORD_HEADER_FMT,
    RECORD_HEALTH_FMT,
    RECORD_MOTION_FLAGS,
    RECORD_MOTION_FMT,
    RECORD_SESSION_FMT,
    RECORD_STREAMS,
    RECORD_TAIL_FMT,
    RECORD_TYPES,
    RESET_REASONS,
    ROBOT_STATES,
    ROBOT_STATES_CN,
    event_text,
    fault_cn,
    firmware_text,
)

HEADER = struct.Struct(RECORD_HEADER_FMT)
POLICY_SLOTS = ("standup", "velstand-flat", "sitstand", "ground-pick", "ball-kick", "roulade")
MISSING = -32768


def read_records(data: bytes) -> list[dict]:
    """拆成一条条记录；尾部不完整的那条丢掉（下载被打断时会有）。按（来自哪一路, 序号）去重。"""
    out, pos, seen = [], 0, set()
    while pos + HEADER.size <= len(data):
        rtype, stream, length, seq, t_ms = HEADER.unpack_from(data, pos)
        body = data[pos + HEADER.size:pos + HEADER.size + length]
        if len(body) < length or rtype not in RECORD_TYPES:
            break
        pos += HEADER.size + length
        key = (stream, seq, rtype)
        if RECORD_TYPES[rtype] != "GAP":
            if key in seen:
                continue
            seen.add(key)
        out.append({"type": RECORD_TYPES[rtype], "stream": stream, "seq": seq, "t": t_ms, "body": body})
    return out


def text_of(raw: bytes) -> str:
    return raw.rstrip(b"\0").decode("utf-8", errors="replace")


def decode(records: list[dict]) -> dict:
    sessions, texts, health, motion, events, tails, gaps = [], [], [], [], [], [], []
    for r in records:
        b, kind = r["body"], r["type"]
        if kind == "SESSION" and len(b) >= struct.calcsize(RECORD_SESSION_FMT):
            ver, reset, servo, _r, boot, fw, _r2, hw, ssid = struct.unpack_from(RECORD_SESSION_FMT, b)
            sessions.append({"t": r["t"], "format": ver, "reset": reset, "servo": servo, "boot": boot, "firmware": fw,
                             "hardware": text_of(hw), "ssid": text_of(ssid)})
        elif kind == "TEXT":
            texts.append((r["seq"], r["t"], b.decode("utf-8", errors="replace")))
        elif kind == "HEALTH" and len(b) >= struct.calcsize(RECORD_HEALTH_FMT):
            v = struct.unpack_from(RECORD_HEALTH_FMT, b)
            health.append({"seq": r["seq"], "t": r["t"], "state": v[0], "servos": v[1], "stations": v[2], "ws": v[3],
                           "bat": v[4], "bat_raw": v[5], "heap": v[6], "heap_min": v[7], "heap_big": v[8],
                           "psram": v[9], "loop_max_us": v[10], "ticks": v[11], "volt": v[12:28], "cur": v[28:44],
                           "temp": v[44:60], "fault": v[60:76], "fault_cnt": v[76:92]})
        elif kind == "MOTION" and len(b) >= struct.calcsize(RECORD_MOTION_FMT):
            v = struct.unpack_from(RECORD_MOTION_FMT, b)
            motion.append({"seq": r["seq"], "t": r["t"], "state": v[0], "policy": v[1], "flags": v[2], "bat": v[4],
                           "pass_us": v[5], "angvel": v[6:9], "gravity": v[9:12], "pos": v[12:26], "vel": v[26:40],
                           "action": v[40:54], "target": v[54:68], "command": v[68:75]})
        elif kind == "EVENT" and len(b) >= struct.calcsize(RECORD_EVENT_FMT):
            events.append((r["t"],) + struct.unpack_from(RECORD_EVENT_FMT, b))
        elif kind == "RTC_TAIL":
            n = len(b) // struct.calcsize(RECORD_TAIL_FMT)
            for i in range(n):
                tails.append(struct.unpack_from(RECORD_TAIL_FMT, b, i * struct.calcsize(RECORD_TAIL_FMT)))
        elif kind == "GAP" and len(b) >= struct.calcsize(RECORD_GAP_FMT):
            stream, _r, skipped = struct.unpack_from(RECORD_GAP_FMT, b)
            gaps.append((r["t"], RECORD_STREAMS.get(stream, stream), skipped, r["seq"]))
    texts.sort()
    health.sort(key=lambda h: h["seq"])
    motion.sort(key=lambda m: m["seq"])
    events.sort()
    return {"sessions": sessions, "texts": texts, "health": health, "motion": motion, "events": events,
            "tails": tails, "gaps": merge_gaps(gaps)}


def merge_gaps(gaps: list[tuple]) -> list[tuple]:
    """同一路首尾相接的空档并成一条：固件跳过一长串运动帧时，每圈到点记一条。

    空档的记录头序号是没存进来的第一条；接得上的条件是上一条的序号 + 条数 = 这一条的序号。
    """
    out, last = [], {}
    for t, stream, skipped, seq in gaps:
        prev = last.get(stream)
        if prev is not None and prev[3] + prev[2] == seq:
            prev[2] += skipped
            continue
        last[stream] = [t, stream, skipped, seq]
        out.append(last[stream])
    return [(t, stream, skipped) for t, stream, skipped, _ in out]


def clock_offset(events: list[tuple]) -> float | None:
    """对时事件（CLOCK：网页或电脑工具连上时发的 Unix 秒）→ 开机毫秒换钟点的偏移（秒）。"""
    kinds = {v: k for k, v in EVENT_KINDS.items()}
    for t, kind, _a, b, c, _d in reversed(events):
        if kind == kinds["CLOCK"]:
            return (((b & 0xFFFF) << 16) | (c & 0xFFFF)) - t / 1000.0
    return None


def stamp(t_ms: int, offset: float | None) -> str:
    rel = f"{t_ms // 60000:3d}:{t_ms // 1000 % 60:02d}.{t_ms % 1000:03d}"
    if offset is None:
        return rel
    return rel + " " + time.strftime("%H:%M:%S", time.localtime(offset + t_ms / 1000.0))


def write_log(d: dict, out: Path, offset: float | None) -> None:
    lines = []
    s = d["sessions"][-1] if d["sessions"] else None
    if s:
        lines.append(f"# 第 {s['boot']} 次上电 · 固件 {firmware_text(s['firmware'])}"
                     f" · 硬件 {s['hardware'] or '?'} · 热点 {s['ssid'] or '?'} · 复位原因 {RESET_REASONS.get(s['reset'], s['reset'])}")
    if d["tails"]:
        lines.append(f"# 上次复位前留在 RTC 里的最后 {len(d['tails']) / 2:.0f} 秒（每半秒一点）：")
        for t_ms, bat, servo_min, state, servos, temp, faults, cur, loop_us in d["tails"]:
            sv = "-" if servo_min == 0xFFFF else f"{servo_min / 10:.1f} V"
            lines.append(f"#   {t_ms / 1000:8.1f} s  电池 {bat / 1000:.2f} V  舵机最低 {sv}  {ROBOT_STATES_CN.get(state, state)}"
                         f"  在线 {servos}  最高温 {temp} °C  最大电流 {cur} mA  故障 {fault_cn(faults)}  最慢一圈 {loop_us} µs")
    items = [(t, 0, f"{stamp(t, offset)}  {text}") for _, t, text in d["texts"]]
    for t, kind, a, b, c, dd in d["events"]:
        items.append((t, 1, f"{stamp(t, offset)}  【事件】{event_text(kind, a, b, c, dd)}"))
    for t, stream, skipped in d["gaps"]:
        items.append((t, 2, f"{stamp(t, offset)}  【空档】{stream} 有 {skipped} 条没存进文件"
                            "（没来得及存就被内存覆盖，或运动帧只存每次软下来前后那几段）"))
    items.sort(key=lambda x: (x[0], x[1]))
    lines += [x[2] for x in items]
    (out / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_health(d: dict, out: Path) -> None:
    with (out / "health.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        ids = range(1, 16)
        w.writerow(["t_s", "state", "battery_V", "battery_raw_V", "servos_online", "wifi_stations", "ws_clients",
                    "heap_free_KB", "heap_min_KB", "heap_largest_KB", "psram_free_KB", "loop_max_us", "control_ticks"]
                   + [f"volt_{i}" for i in ids] + [f"cur_mA_{i}" for i in ids] + [f"temp_{i}" for i in ids]
                   + [f"fault_{i}" for i in ids] + [f"fault_count_{i}" for i in ids])
        for h in d["health"]:
            w.writerow([f"{h['t'] / 1000:.3f}", ROBOT_STATES_CN.get(h["state"], h["state"]), h["bat"] / 1000,
                        h["bat_raw"] / 1000, h["servos"], h["stations"], h["ws"], h["heap"], h["heap_min"],
                        h["heap_big"], h["psram"], h["loop_max_us"], h["ticks"]]
                       + ["" if h["volt"][i] == 0xFF else h["volt"][i] / 10 for i in ids]
                       + [h["cur"][i] for i in ids] + ["" if h["temp"][i] == 0xFF else h["temp"][i] for i in ids]
                       + [h["fault"][i] for i in ids] + [h["fault_cnt"][i] for i in ids])


def write_motion(d: dict, out: Path) -> None:
    def q(v, scale):
        return "" if v == MISSING else v / scale
    with (out / "motion.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        names = [f"{JOINT_NAMES[j]}({j})" for j in POLICY_JOINT_IDS]
        w.writerow(["t_s", "seq", "state", "policy", "flags", "battery_V", "pass_us", "angvel_x", "angvel_y", "angvel_z",
                    "gravity_x", "gravity_y", "gravity_z"] + [f"pos_{n}" for n in names] + [f"vel_{n}" for n in names]
                   + [f"action_{n}" for n in names] + [f"target_{n}" for n in names]
                   + ["cmd_vx", "cmd_vy", "cmd_wz", "cmd_neck", "cmd_head_pitch", "cmd_head_yaw", "cmd_head_roll"])
        for m in d["motion"]:
            flags = [k for k, bit in RECORD_MOTION_FLAGS.items() if m["flags"] & bit]
            w.writerow([f"{m['t'] / 1000:.3f}", m["seq"], ROBOT_STATES_CN.get(m["state"], m["state"]),
                        POLICY_SLOTS[m["policy"]] if m["policy"] < len(POLICY_SLOTS) else "", "|".join(flags),
                        m["bat"] / 1000 if m["bat"] else "", m["pass_us"]]
                       + [q(v, 1000) for v in m["angvel"]] + [q(v, 10000) for v in m["gravity"]]
                       + [q(v, 1000) for v in m["pos"]] + [q(v, 1000) for v in m["vel"]]
                       + [q(v, 1000) for v in m["action"]] + [q(v, 1000) for v in m["target"]]
                       + [q(v, 1000) for v in m["command"]])


def falls(d: dict) -> list[int]:
    kinds = {v: k for k, v in EVENT_KINDS.items()}
    return [t for t, kind, *_ in d["events"] if kind == kinds["FALL"]]


def write_summary(d: dict, out: Path, offset: float | None) -> str:
    lines = []
    h, m = d["health"], d["motion"]
    span = max([x["t"] for x in h] + [x["t"] for x in m] + [t for _, t, _ in d["texts"]] + [0]) / 1000
    lines.append(f"记录时长：开机后 {span / 60:.1f} 分钟；日志 {len(d['texts'])} 行，健康 {len(h)} 帧，运动 {len(m)} 帧"
                 f"（{len(m) / 50:.0f} 秒），事件 {len(d['events'])} 条")
    if m:
        lines.append(f"运动帧覆盖：开机后 {m[0]['t'] / 1000:.1f}–{m[-1]['t'] / 1000:.1f} s"
                     + ("（中间有断档）" if m[-1]["seq"] - m[0]["seq"] + 1 > len(m) else ""))
    bats = [x["bat"] for x in h if x["bat"]] + [x["bat"] for x in m if x["bat"]]
    if bats:
        low = min(bats)
        when = next((x["t"] for x in m + h if x["bat"] == low), 0)
        lines.append(f"电池电压：最低 {low / 1000:.2f} V（开机后 {when / 1000:.1f} s），最高 {max(bats) / 1000:.2f} V")
    else:
        lines.append("电池电压：没有读数（只插了 USB，或 ADC 一直被 WiFi 占着）")
    volts = [(x["volt"][i], x["t"], i) for x in h for i in range(1, 16) if x["volt"][i] != 0xFF]
    if volts:
        v, t, i = min(volts)
        lines.append(f"舵机输入电压：最低 {v / 10:.1f} V（{i} 号 {JOINT_NAMES.get(i, '')}，开机后 {t / 1000:.1f} s）")
    curs = [(abs(x["cur"][i]), x["t"], i) for x in h for i in range(1, 16)]
    if curs and max(curs)[0] > 0:
        c, t, i = max(curs)
        lines.append(f"舵机电流：最大 {c} mA（{i} 号 {JOINT_NAMES.get(i, '')}，开机后 {t / 1000:.1f} s）")
    temps = [(x["temp"][i], x["t"], i) for x in h for i in range(1, 16) if x["temp"][i] != 0xFF]
    if temps:
        c, t, i = max(temps)
        lines.append(f"舵机温度：最高 {c} °C（{i} 号 {JOINT_NAMES.get(i, '')}，开机后 {t / 1000:.1f} s）")
    loops = [x["loop_max_us"] for x in h]
    if loops:
        # 只数舵机正被驱动的那几秒：不驱动时（软瘫、卸力）固件才写 flash，那时慢几十毫秒是有意的（固件 PITFALLS #44）
        driven = {k for k, v in ROBOT_STATES.items()
                  if v in ("STANDING_UP", "STANDING", "WALKING", "SITTING", "GETTING_UP", "CHECKING")}
        late = sum(1 for x in h if x["state"] in driven and x["loop_max_us"] > 20000)
        lines.append(f"主循环最慢一圈：{max(loops) / 1000:.1f} ms（从拍头算到做完，一拍 20 ms）；"
                     f"舵机被驱动时有 {late} 秒里误过拍")
    kinds = {v: k for k, v in EVENT_KINDS.items()}
    notable = {kinds[k] for k in ("FALL", "GETUP_START", "GETUP_DONE", "GETUP_BLOCKED", "GETUP_REST", "SERVO_FAULT",
                                  "SERVO_WAITING", "SERVO_CLEARED", "STANDUP", "BOOT")}
    ev = [(t, kind, a, b, c, dd) for t, kind, a, b, c, dd in d["events"] if kind in notable]
    if ev:
        lines.append("经过：")
        lines += [f"  {stamp(t, offset)}  {event_text(kind, a, b, c, dd)}" for t, kind, a, b, c, dd in ev]
    if d["tails"]:
        lines.append(f"上次复位前的尾巴：{len(d['tails']) / 2:.0f} 秒（见 log.txt 开头）")
    if d["gaps"]:
        lines.append("空档：" + "；".join(f"{s} 少 {n} 条（开机后 {t / 1000:.0f} s）" for t, s, n in d["gaps"]))
    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text, encoding="utf-8")
    return text


def plot(d: dict, out: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    made = []
    h, m = d["health"], d["motion"]
    if h or m:
        fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
        if m:
            ax[0].plot([x["t"] / 1000 for x in m if x["bat"]], [x["bat"] / 1000 for x in m if x["bat"]], lw=0.6,
                       label="电池（每拍快采）", color="#9fb3c8")
        if h:
            th = [x["t"] / 1000 for x in h]
            ax[0].plot(th, [x["bat"] / 1000 if x["bat"] else None for x in h], lw=1.5, label="电池（滤波）", color="#1f5f94")
            vmin = [min((x["volt"][i] for i in range(1, 16) if x["volt"][i] != 0xFF), default=None) for x in h]
            ax[0].plot(th, [v / 10 if v else None for v in vmin], lw=1.2, label="舵机输入电压（最低）", color="#d9822b")
            ax[1].plot(th, [max(abs(c) for c in x["cur"][1:16]) for x in h], lw=1.2, color="#c0392b")
            ax[2].plot(th, [max((t for t in x["temp"][1:16] if t != 0xFF), default=None) for x in h], lw=1.2,
                       color="#8e44ad")
            ax[3].step(th, [x["state"] for x in h], where="post", color="#2c3e50")
        for t in falls(d):
            for a in ax:
                a.axvline(t / 1000, color="#c0392b", lw=0.8, ls="--")
        ax[0].set_ylabel("电压 V")
        ax[0].legend(loc="lower left", fontsize=8)
        ax[1].set_ylabel("最大电流 mA")
        ax[2].set_ylabel("最高温度 °C")
        ax[3].set_ylabel("状态")
        ax[3].set_yticks(list(ROBOT_STATES_CN))
        ax[3].set_yticklabels([ROBOT_STATES_CN[k] for k in ROBOT_STATES_CN], fontsize=7)
        ax[3].set_xlabel("开机后 秒（红虚线 = 摔倒）")
        fig.tight_layout()
        p = out / "power.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        made.append(p)
    windows = [t for t in falls(d) if m and m[0]["t"] <= t <= m[-1]["t"] + 1]
    if windows:
        legs = [2, 3, 11, 12]   # 运动帧关节序（POLICY_JOINT_IDS）里：左髋俯仰、左膝、右髋俯仰、右膝
        fig, ax = plt.subplots(len(windows), 1, figsize=(12, 3.2 * len(windows)), squeeze=False)
        for k, t0 in enumerate(windows):
            seg = [x for x in m if t0 - 5000 <= x["t"] <= t0 + 15000]
            a = ax[k][0]
            ts = [(x["t"] - t0) / 1000 for x in seg]
            a.plot(ts, [x["gravity"][2] / 10000 if x["gravity"][2] != MISSING else None for x in seg], color="k", lw=1.5,
                   label="重力 z（直立 -1）")
            for j in legs:
                c = a.plot(ts, [x["pos"][j] / 1000 if x["pos"][j] != MISSING else None for x in seg], lw=1,
                           label=f"{JOINT_NAMES[POLICY_JOINT_IDS[j]]} 实际")[0].get_color()
                a.plot(ts, [x["target"][j] / 1000 if x["target"][j] != MISSING else None for x in seg], lw=1, ls="--",
                       color=c)
            a.axvline(0, color="#c0392b", lw=0.8)
            a.set_title(f"开机后 {t0 / 1000:.1f} s 摔倒前后（实线实际、虚线目标，相对默认站姿，rad）", fontsize=10)
            a.legend(loc="upper right", fontsize=7, ncol=3)
        ax[-1][0].set_xlabel("相对摔倒时刻 秒")
        fig.tight_layout()
        p = out / "falls.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        made.append(p)
    return made


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("record", type=Path)
    p.add_argument("-o", "--out", type=Path, help="输出目录（默认与记录同名）")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()
    data = args.record.read_bytes()
    d = decode(read_records(data))
    out = args.out or args.record.with_suffix("")
    out.mkdir(parents=True, exist_ok=True)
    offset = clock_offset(d["events"])
    write_log(d, out, offset)
    write_health(d, out)
    write_motion(d, out)
    print(write_summary(d, out, offset), end="")
    made = [] if args.no_plot else plot(d, out)
    print(f"→ {out}：log.txt、health.csv、motion.csv、summary.txt" + "".join(f"、{x.name}" for x in made))


if __name__ == "__main__":
    main()
