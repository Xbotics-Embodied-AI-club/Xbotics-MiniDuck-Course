"""固件 WiFi 协议与板上文件格式的电脑端定义：包格式、magic、命令编码、故障位、关节名、状态、上电记录格式。

与固件 lib/comm/src/wifi/WifiCommand.h、lib/servo/src/JointFault.h、src/hardware/hardware_profile.h、
src/events/events.h、src/state/robot_state.h、src/record/boot_record.h、lib/policy/src/policy_int8_blob.h
一一对应；电脑端工具和 pyduck 从这里读取同一套定义。
"""

import struct
import time

DEFAULT_IP = "192.168.4.1"     # 固件 SoftAP 的地址（热点名 Xbotics-MiniDuck-XXXX，XXXX = 热点 MAC 后 4 位，开机串口打出全名）
UDP_PORT = 5005
WS_PORT = 88                   # 各网页共用的 WebSocket（二进制包与 UDP 同一套格式）
VERSION = 1

# 网页：一个功能一个端口（固件 lib/comm/src/web/WebUi.h）
WEB_PORTS = {"remote": 80, "servo": 81, "calib": 82, "files": 83, "debug": 84, "sound": 85}

# WiFi 配网（固件 include/system_config.h 与 lib/comm/src/wifi/WifiLinkPolicy.h；文件页 83 口）
WIFI_CONFIG_PATH = "/wifi"                 # GET 看配网状态、POST 保存凭据
WIFI_NVS_NAMESPACE = "duckwifi"            # 凭据单独一个 NVS 命名空间（改名 = 所有机器人忘掉配过的热点）
WIFI_NVS_SSID_KEY = "ssid"
WIFI_NVS_PASS_KEY = "pass"
WIFI_SSID_MAX = 32                         # 802.11 SSID 最多 32 字节
WIFI_PASS_MIN = 8                          # WPA2 口令最少 8 位；0 = 开放网络
WIFI_PASS_MAX = 63                         # WPA2 口令最多 63 个可打印 ASCII 字符
WIFI_STA_CONNECT_TIMEOUT_MS = 15000        # 一次连接尝试多久算失败（AP 一直可用）
WIFI_STA_RETRY_MS = 30000                  # 失败后隔多久再试
WIFI_PUSH_MS = 500                         # 主循环把 WiFi 快照推给网页任务的周期

# 板上文件（策略、叫声、上电记录）：HTTP（文件页 83 口与 tools/assets.py 共用；固件 src/assets/asset_api.cpp）
FS_LIST_PATH = "/fs"            # GET → JSON：files、used、total、writable、why、policies、sounds
FS_UPLOAD_PATH = "/fs/upload"   # POST，请求头 X-Name = 文件名，请求体 = 文件原样；.mdp → /policy/，.wav → /audio/
FS_UPLOAD_NAME_HEADER = "X-Name"  # 文件名走请求头：板上 WebServer 收完请求体才解析网址参数
FS_DELETE_PATH = "/fs/delete"   # POST ?path=/policy/sitstand.mdp（只删上传的文件与上电记录，出厂文件删不掉）
FS_RESTORE_PATH = "/fs/restore" # POST：删掉全部上传的策略与叫声，回复后重启（恢复出厂）
FS_REBOOT_PATH = "/reboot"      # POST，回复后重启（新文件开机才载入）
RECORD_PATH = "/record"         # GET ?session=current|<序号>（调试页 84 口）：上电记录，格式见 boot_record.py

# 策略文件头（128 字节；固件 lib/policy/src/policy_int8_blob.h）
POLICY_HEADER_FMT = "<4sHHII24s16s16siiiifffiI20s"

# 6 个核心策略槽（固件 system_config.h PolicyIdx；自定义策略不占这些名字，也不许覆盖语义）
POLICY_CORE_SLOTS = ("standup", "velstand-flat", "sitstand", "ground-pick", "ball-kick", "roulade")
POLICY_MAX_TOTAL = 12          # 注册表总容量 = 6 核心 + 6 自定义（固件 POLICY_MAX_TOTAL）
POLICY_NAME_MAX = 15           # 自定义策略名最长 15 字符（注册表条目里固定 16B 含 NUL）
POLICY_RUN_MS_MAX = 60000      # PCMD RUN 单次最长 60 s（断连兜底）
POLICY_RUN_SCALE_MIN = 0.05    # RUN scale 限幅，同调参 walk_scale
POLICY_RUN_SCALE_MAX = 1.5

# 运动命令：64B，flags bit0 = 坐姿请求、bit1 = 张嘴请求；13 维命令 = twist(3) + head(4) + body(6)
MOTION = b"DCMD"
MOTION_FMT = "<4sHHBBH13f"

# 调参：72B，键表 + 值表（下发 TCMD；查询发 TGET，回复是全量 TCMD）
TUNE_SET = b"TCMD"
TUNE_GET = b"TGET"
TUNE_FMT = "<4sHHB3s12B12f"
TUNE_KEYS = [   # 与固件 WifiTuneKey 顺序一致
    "walk_scale", "stand_scale", "sitstand_scale", "stand_threshold",
    "head_lowpass", "legs_lowpass", "fall_gz_max", "walk_p_gain", "stand_p_gain",
]

# 诊断：DGET 请求 8B → DSTA 回复 244B（每舵机 10B × 16 + 64 点电压历史）
DIAG_GET = b"DGET"
DIAG_STA = b"DSTA"
DIAG_REQ_FMT = "<4sHH"
DIAG_FMT = "<4sHHI4BHH" + "BBHhhBB" * 16 + "64B"

# 标定：CCMD 请求 12B（末字段 = 点动增量，毫弧度，带符号）→ CPOS 回复 156B（按舵机 ID 下标）
CALIB_CMD = b"CCMD"
CALIB_POS = b"CPOS"
CCMD_FMT = "<4sHHBBh"
CPOS_FMT = "<4sHHBBH16i16i16B"
CMD_EXIT_DISCARD = 0
CMD_ENTER_PASSIVE = 1
CMD_ENTER_CHECK = 2
CMD_QUERY = 3
CMD_RECORD = 4
CMD_APPLY_EXIT = 5
CMD_RESET_TABLE = 6
CMD_JOG = 7
CMD_LIMP = 8          # 卸力（网页里只在遥控页 80 口）
CMD_STAND = 9         # 启动（从卸力或待机起身到站立姿态）
CALIB_STATE_NAMES = {0: "IDLE(正常)", 1: "PASSIVE(被动标定)", 2: "CHECK(零位检查)", 3: "LIMP(卸力)",
                     4: "SETUP(舵机编号)"}

# 舵机种类配置：SCFG 请求与回复同为 12B；kind 0 = 只查询 / 没配置过
SERVO_CFG = b"SCFG"
SCFG_FMT = "<4sHHBBBB"
SERVO_KINDS = {"hd1910": 1, "xl330": 2}   # 与固件 ServoKind 一致

# 事件日志：EGET 请求 12B → ELOG 回复 532B（32 条 × 16B，新的在前）；对时 TIME 12B（无回复）
EVENT_GET = b"EGET"
EVENT_LOG = b"ELOG"
CLOCK_SET = b"TIME"
EGET_FMT = "<4sHHI"
ELOG_HEAD_FMT = "<4sHHIIBBH"
EVENT_FMT = "<IIBBhhBB"          # 一条记录：seq、开机毫秒、种类、a、b、c、d、校验（lib/log/src/eventlog/EventRing.h）
EVENT_MAX_RECORDS = 32
ELOG_FMT = ELOG_HEAD_FMT + "16s" * EVENT_MAX_RECORDS
TIME_FMT = "<4sHHI"

# 状态：STAT 请求 8B → SSTA 回复（固件 WifiStatusPacket）。
# v1 = 224B（SSTA_FMT）；v2 = 256B（SSTA_FMT_V2），尾部追加 ToF 与头部 IMU。
# version 字段保持 1，新字段靠帧长区分：收到 224B 说明固件旧，tof/head_* 解析为 None。
STATUS_GET = b"STAT"
STATUS_REPLY = b"SSTA"
STAT_REQ_FMT = "<4sHH"
SSTA_FMT = "<4sHHIHBBBBBBHH3f3f16f16f3f24s3BBIHH"                       # 224B（v1）
SSTA_V2_APPEND_FMT = "hBx3f3f4x"   # 追加 32B：@224 distanceMm(i16, -1=无目标) @226 tofFlags(bit0 有效)
SSTA_FMT_V2 = SSTA_FMT + SSTA_V2_APPEND_FMT                             # @228 headGravity[3]：芯片系、未归一化，
                                                                        # 模长≈1000 mg（与 body gravity 单位向量不同，坑 #76）
STATUS_V1_LEN = struct.calcsize(SSTA_FMT)                               # @240 headGyro[3] rad/s（芯片系）
STATUS_V2_LEN = struct.calcsize(SSTA_FMT_V2)
STATUS_FLAGS = {"policies_ok": 0x01, "body_imu": 0x02, "head_imu": 0x04, "voice": 0x08, "writable": 0x10,
                "battery": 0x20, "direction_ok": 0x40, "all_servos": 0x80}

# 策略控制：PCMD 请求 36B → PRSP 回复 288B（WS 88 与 UDP 5005 同收发路径）。
# RUN 按名（name 非空）或按 idx 运行一个已注册策略；STOP 停止当前运行；LIST 查全表；LOAD 运行时热载 /policy/<name>.mdp。
# 运行语义同 DCMD.skill：要求站稳 ≥1s 且 standup 激活，期间命令观测锁零，结束/停止/摔倒/新运动指令回 standup。
POLICY_CMD = b"PCMD"
POLICY_REPLY = b"PRSP"
PCMD_FMT = "<4sHHBBHIf16s"        # 36B：cmd@8 idx@9 reserved@10 durationMs@12 scale@16 name@20
PRSP_FMT = "<4sHHBBBxI" + "BBBBBB16s" * POLICY_MAX_TOTAL + "8x"   # 288B
PCMD_CMDS = {"RUN": 1, "STOP": 2, "LIST": 3, "LOAD": 4}
PCMD_RESULTS = {0: "完成", 1: "未找到策略", 2: "机器人未站稳", 3: "名字非法",
                4: "注册表已满", 5: "载入失败（文件校验不过）", 6: "参数越限"}
POLICY_ENTRY_STATUS = {0: "空", 1: "就绪", 2: "运行中"}
POLICY_STOP_REASONS = {0: "时长到正常结束", 1: "被 STOP 停止", 2: "被运动指令取消", 3: "摔倒中断"}

# 机器人状态（固件 src/state/robot_state.h 的 RobotState）
ROBOT_STATES = {0: "IDLE", 1: "STANDING_UP", 2: "STANDING", 3: "WALKING", 4: "SITTING", 5: "FALLEN",
                6: "GETTING_UP", 7: "LIMP", 8: "CALIBRATING", 9: "CHECKING", 10: "SERVO_SETUP", 11: "SKILL"}
ROBOT_STATES_CN = {0: "待机", 1: "起身中", 2: "站立", 3: "行走", 4: "坐姿", 5: "倒地", 6: "起身中",
                   7: "已卸力", 8: "标定中", 9: "零位检查中", 10: "舵机编号中", 11: "做动作"}

# 舵机编号：SSET 请求 16B → SREP 回复 64B（固件 WifiSetupRequest / WifiSetupPacket）
SETUP_SET = b"SSET"
SETUP_REPLY = b"SREP"
SSET_FMT = "<4sHHBBBBhH"
SREP_FMT = "<4sHHBBBBBBHiHhBBBB32s4s"
SETUP_CMDS = {"QUERY": 0, "ENTER": 1, "EXIT": 2, "SCAN": 3, "READ": 4, "SET_ID": 5, "CENTER": 6, "WIGGLE": 7,
              "RELEASE": 8}
SETUP_RESULTS = {0: "完成", 1: "未进入编号模式", 2: "机器人未卸力", 3: "舵机无响应", 4: "该 ID 已被占用",
                 5: "写入校验失败", 6: "参数错误", 7: "上一操作未完成"}

# 上电记录（固件 src/record/boot_record.h）：一条条记录 = 12 字节头 + 数据
RECORD_HEADER_FMT = "<BBHII"   # 种类、来自哪一路、数据长度、这一路里的序号、开机毫秒
RECORD_TYPES = {1: "SESSION", 2: "TEXT", 3: "HEALTH", 4: "MOTION", 5: "EVENT", 6: "RTC_TAIL", 7: "GAP"}
RECORD_STREAMS = {0: "text", 1: "health", 2: "motion"}
RECORD_FORMAT_VERSION = 1
RECORD_SESSION_FMT = "<BBBBIHH24s24s"
RECORD_HEALTH_FMT = "<BBBBHHHHHHHH16B16h16B16B16B"
RECORD_MOTION_FMT = "<BBBBHH3h3h14h14h14h14h7h"
RECORD_EVENT_FMT = "<BBhhB"
RECORD_TAIL_FMT = "<IHHBBBBhH"
RECORD_GAP_FMT = "<B3sI"
RECORD_MOTION_FLAGS = {"sit": 0x01, "mouth": 0x02, "voice": 0x04, "read_fail": 0x08}
POLICY_JOINT_IDS = (10, 9, 8, 7, 6, 11, 12, 13, 14, 5, 4, 3, 2, 1)   # 运动帧里关节的顺序（system_config.h）

# 事件种类（与固件 src/events/events.h 的 EventKind 编码一致）
EVENT_KINDS = {
    1: "BOOT", 2: "CLOCK", 3: "CONTROL_START", 4: "CONTROL_HALT", 5: "FALL", 6: "GETUP_START",
    7: "GETUP_DONE", 8: "GETUP_BLOCKED", 9: "SERVO_FAULT", 10: "SERVO_WAITING", 11: "SERVO_CLEARED",
    12: "GETUP_REST", 13: "STANDUP", 14: "POLICY_RUN", 15: "POLICY_STOP",
}
# 倒地原因（固件 src/control/control.h 的 FallReason）
FALL_REASONS = {1: "倾斜过大", 2: "舵机保护", 3: "关节读数丢失", 4: "舵机无响应", 5: "起身超时",
                6: "起身后未恢复直立", 7: "动作结束时还倒着"}
# 复位原因（ESP-IDF esp_reset_reason_t）
RESET_REASONS = {0: "未知", 1: "上电", 2: "外部复位", 3: "软件重启", 4: "程序崩溃", 5: "中断看门狗",
                 6: "任务看门狗", 7: "看门狗", 8: "深睡唤醒", 9: "欠压", 10: "SDIO", 11: "USB",
                 12: "JTAG", 13: "eFuse", 14: "电源毛刺", 15: "CPU 锁死"}

# 故障位（JointFault，两种舵机统一）
JOINT_FAULTS = ((0x01, "VOLT"), (0x02, "HEAT"), (0x04, "OLOAD"), (0x08, "POWER"),
                (0x10, "ENC"), (0x40, "NOREPLY"), (0x80, "TQLOST"))

# 舵机 ID → 关节（与固件 include/system_config.h 的 ID 映射一致）
JOINT_NAMES = {
    1: "右踝", 2: "右膝", 3: "右髋俯仰", 4: "右髋横滚", 5: "右髋偏航",
    6: "左踝", 7: "左膝", 8: "左髋俯仰", 9: "左髋横滚", 10: "左髋偏航",
    11: "颈俯仰", 12: "头俯仰", 13: "头偏航", 14: "头横滚", 15: "嘴",
}


FAULT_CN = {"VOLT": "电压越限", "HEAT": "过热", "OLOAD": "过载", "POWER": "过流/供电不足", "ENC": "编码器",
            "NOREPLY": "无响应", "TQLOST": "扭矩关闭"}


def parse_ssta(data: bytes) -> dict:
    """SSTA 帧 → 字段 dict；224B 旧固件 tof/head_* 为 None，256B 新固件带 ToF 与头部 IMU。"""
    base = struct.unpack(SSTA_FMT, data[:STATUS_V1_LEN])
    d = {
        "uptime_ms": base[3], "firmware": base[4], "state": base[5], "policy": base[6],
        "servo_kind": base[7], "servo_count": base[8], "flags": base[9], "wifi_stations": base[10],
        "battery_mv": base[11], "loop_hz10": base[12],
        "gravity": base[13:16], "gyro": base[16:19],
        "joint": base[19:35], "target": base[35:51], "command": base[51:54],
        "hardware_id": base[54].rstrip(b"\0").decode("ascii", "replace"),
        "policy_source": base[55:58], "sounds_loaded": base[58],
        "record_file_bytes": base[59], "record_motion_s": base[60], "boot_seq": base[61],
        "tof_mm": None, "tof_valid": False, "head_gravity": None, "head_gyro": None,
    }
    if len(data) >= STATUS_V2_LEN:
        v2 = struct.unpack(SSTA_V2_APPEND_FMT, data[STATUS_V1_LEN:STATUS_V2_LEN])
        d["tof_mm"] = v2[0]
        d["tof_valid"] = bool(v2[1] & 0x01)
        d["head_gravity"] = v2[2:5]
        d["head_gyro"] = v2[5:8]
    return d


def fault_tags(bits: int) -> str:
    """故障位 → 可读标记（"VOLT|TQLOST"）；没有故障返回 "-"。"""
    return "|".join(name for bit, name in JOINT_FAULTS if bits & bit) or "-"


def fault_cn(bits: int) -> str:
    """故障位 → 中文（"过热、扭矩关闭"）。"""
    return "、".join(FAULT_CN[name] for bit, name in JOINT_FAULTS if bits & bit) or "无"


def _servo(sid: int) -> str:
    return f"{sid} 号舵机（{JOINT_NAMES.get(sid, '?')}）"


def _temp(t: int) -> str:
    return "温度无读数" if t < 0 else f"{t} °C"


def firmware_text(code: int) -> str:
    """固件版本编码（主×1000 + 次×10 + 修订）→ 点分文本；旧 0.x 编码保持不变。"""
    return f"{code // 1000}.{code // 10 % 100}.{code % 10}"


def event_text(kind: int, a: int, b: int, c: int, d: int) -> str:
    """一条事件 → 一句中文（参数含义见固件 src/events/events.h）。"""
    name = EVENT_KINDS.get(kind)
    kinds = {v: k for k, v in SERVO_KINDS.items()}
    if name == "BOOT":
        return f"开机（{RESET_REASONS.get(a, a)}），固件 {firmware_text(b)}，舵机 {kinds.get(d, '?')}"
    if name == "CLOCK":
        unix = ((b & 0xFFFF) << 16) | (c & 0xFFFF)
        return "对时 " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(unix))
    if name == "CONTROL_START":
        return "闭环启动"
    if name == "CONTROL_HALT":
        return "闭环停止"
    if name == "FALL":
        return f"倒地：{FALL_REASONS.get(a, a)}（重力 z {b / 1000:+.2f}）"
    if name == "GETUP_START":
        return f"起身开始（第 {a} 次）"
    if name == "GETUP_DONE":
        return f"{'起身完成' if a == 1 else '起身超时'}（重力 z {b / 1000:+.2f}，{c / 100:.1f} s）"
    if name == "GETUP_BLOCKED":
        return "起身中止：舵机配置写入或读取失败"
    if name == "SERVO_FAULT":
        return f"{_servo(a)}故障：{fault_cn(d)}（{_temp(b)}，{c} mA）"
    if name == "SERVO_WAITING":
        return f"{_servo(a)}未恢复：{fault_cn(d)}（{_temp(b)}，已等待 {c} s）"
    if name == "SERVO_CLEARED":
        return f"{_servo(a)}故障清除（{c} s）"
    if name == "GETUP_REST":
        return f"连续 {a} 次起身失败，卸力冷却 {c} s"
    if name == "STANDUP":
        return "启动完成" if a == 1 else "启动中止（见串口日志）"
    if name == "POLICY_RUN":
        return f"自定义策略运行：#{a}（{b} s）"
    if name == "POLICY_STOP":
        return f"自定义策略 #{a} 结束：{POLICY_STOP_REASONS.get(b, b)}"
    return f"未知事件 {kind}（a={a} b={b} c={c} d={d}）"


def format_events(records, uptime_now_ms: int, now_unix: float) -> list[str]:
    """一批事件记录（EVENT_FMT 解出来的元组，任意顺序）→ 按开机分段、按时间排好的中文行。

    钟点：本次开机按「现在的钟点 − 现在的开机毫秒」换算；更早的开机靠那次开机里的对时记录，没有就只写开机后多久。
    """
    recs = sorted(records, key=lambda r: r[0])
    sessions = [[]]
    for r in recs:
        if EVENT_KINDS.get(r[2]) == "BOOT" and sessions[-1]:
            sessions.append([])
        sessions[-1].append(r)
    lines = []
    for k, sess in enumerate(sessions):
        offset = None
        for r in sess:
            if EVENT_KINDS.get(r[2]) == "CLOCK":
                offset = (((r[4] & 0xFFFF) << 16) | (r[5] & 0xFFFF)) - r[1] / 1000.0
        current = k == len(sessions) - 1
        if current:
            offset = now_unix - uptime_now_ms / 1000.0
        ago = len(sessions) - 1 - k
        head = "本次开机" if current else f"往前第 {ago} 次开机"
        if EVENT_KINDS.get(sess[0][2]) != "BOOT":
            head += "（开机那条已被覆盖）"
        lines.append(f"—— {head} ——")
        for seq, ms, kind, a, b, c, d, _check in sess:
            when = (time.strftime("%m-%d %H:%M:%S", time.localtime(offset + ms / 1000.0))
                    if offset is not None else f"开机后 {ms // 60000:d}:{ms // 1000 % 60:02d}")
            lines.append(f"#{seq:<6} {when}  {event_text(kind, a, b, c, d)}")
    return lines
