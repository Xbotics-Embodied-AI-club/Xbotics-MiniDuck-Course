# pyduck

MiniDuck 鸭子的 Python 上位机 SDK：运动、状态、IMU / ToF、舵机、调参、文件、上电记录、叫声、配网与自定义策略。
包里的协议常量与格式串**全部**取自固件仓库的 `firmware/tools/protocol.py`，SDK 不另抄一份。

## 安装

```bash
uv sync  # 在学员仓根目录执行，同时安装 pyduck 和课程依赖
```

依赖：`websocket-client>=1.6`、`requests>=2.31`，Python ≥ 3.10。

**协议来源**：SDK 不内置抄写协议常量——仓库内运行时自动向上寻找
`firmware/tools/protocol.py` 读取正本；**独立分发安装**（旁边没有固件仓库）使用包内置的
协议快照 `pyduck/_protocol_snapshot.py`。也可以用环境变量
`PYDUCK_PROTOCOL=/path/to/protocol.py` 显式指定正本位置。

## Quickstart

先连机器人的热点 `Xbotics-MiniDuck-XXXX`（开机串口会打出全名），板子地址默认 `192.168.4.1`
（**没有 mDNS**，要自己给 IP）。

```python
import time
import pyduck

with pyduck.Duck.connect("192.168.4.1") as duck:      # 默认 WebSocket 88 口
    st = duck.state.get()
    print(st.state_cn, st.battery_v, st.loop_hz)

    with duck.motion.drive() as d:                    # 20 Hz 持续发送
        d.twist(vx=0.2)
        time.sleep(2)                                 # 退出时自动匀减速回零
    duck.motion.turn_left(seconds=1)

    duck.motion.sit()
    time.sleep(1)
    duck.motion.stand()

    print(duck.tune.get())
    duck.tune.set(walk_scale=0.6)

    for event in duck.state.events():                 # 增量跟随事件日志（无限生成器，别 list()）
        print(event["text"])
    recent = duck.state.history(20)                   # 取最近 20 条用 history()

# UDP（tools/ 下脚本用的通道）
duck = pyduck.Duck.connect("192.168.4.1", transport="udp")
duck.close()
```

> ⚠️ `events()` 是**无限跟随生成器**（内部 `while True`），对它 `list()` 会永远卡住；
> 一次性取历史用 `history(n)`。`events()` 首次回溯默认只取 32 条（`max_backfill`）。

### 常用入口

| 属性 | 作用 |
| --- | --- |
| `duck.state` | `get()`、`watch()`、`wait_for()`、`events()`、`history()`、`wait_event()` |
| `duck.motion` | `drive()`、`forward/backward/turn_left/turn_right`、`sit/stand/limp/stop`、`skill()` |
| `duck.servos` | `snapshot()`、`diagnostics()`、`raw_positions()`、标定命令 |
| `duck.tune` | `get()` / `set()` |
| `duck.files` | 列表、上传、删除、恢复出厂、重启 |
| `duck.records` | 上电记录列表与下载 |
| `duck.sounds` / `duck.wifi` | 叫声状态与播放 / 配网 |
| `duck.policy` | 策略列表、上传、热载、运行、停止 |
| `duck.video` | 预留（当前抛 `NotImplementedError`） |

## 例程

`examples/` 下五个可直接跑的例程（连接参数都是 `--ip`）：

| 例程 | 内容 |
| --- | --- |
| `01_state_and_sensors.py` | 纯只读：整机状态、ToF/IMU 采样、舵机快照与诊断 |
| `02_drive_and_avoid.py` | `drive()` 行走 + ToF 闭环避障、坐/站 |
| `03_custom_policy.py` | 自定义策略：上传→热载→运行→事件核对（`--mdp`） |
| `04_files_and_records.py` | 文件列表、上电记录下载（写操作姿势看注释） |
| `05_events_watch.py` | 事件历史、`wait_event` 等待、有界跟随 |

```bash
python sdk/pyduck/examples/01_state_and_sensors.py --ip 192.168.1.150
python sdk/pyduck/examples/03_custom_policy.py --ip 192.168.1.150 --mdp my-policy.mdp
```

## 语义陷阱

这些是固件真实行为，不是 SDK 的毛病：

- **DCMD 没有 ACK。** 运动命令是“后到覆盖先到 + 500 ms 没新包就归零”，发出去不代表执行了。
  要看结果请查 SSTA。
- **twist 500 ms 新鲜度，但坐姿是锁存的。** 停发运动包后速度自动归零；坐姿请求不会自己消失，
  要站起来得发一帧不带 `sit` 位的命令（`motion.stand()` 会做）。
- **一次性动作与策略要求站稳 ≥1 秒且 standup 已激活。** 否则固件返回“机器人未站稳”（策略）或
  直接忽略（skill）。动作期间观测锁零，结束 / 停止 / 摔倒 / 新运动指令都会回到 standup。
- **`drive()` 退出要匀减速。** 会话退出时在 0.5 s 内线性回零、再补几帧全零；不要在退出后立刻
  发新的高速命令，给固件一个明确的“站着”。
- **写文件要 `robotFilesWritable`。** 待机、卸力（或标定 / 舵机编号没在动）才写 flash；站立、
  走路、坐着、起身中写 flash 会让关节顿一下。`files.upload()` 会先查 SSTA 拒绝，不要绕过。
- **重启要靠 `bootSeq` 确认。** `POST /reboot` 的回复只表示“收到了”，固件回复后还会再查一次
  能不能重启；`files.reboot()` 轮询 SSTA 的 `boot_seq` 变大才算真的重启了。
- **事件日志从新往旧翻页，连上先发 TIME。** `EGET` 的 `beforeSeq` 只能往前翻；事件里的开机毫秒
  要靠 `TIME` 对时才能换算成钟点（`Duck.connect()` 会自动发一次）。环形缓冲被覆盖时会跳号，
  `events()` 把跳过的条数放在下一条事件的 `missed_before` 里。
- **没有 mDNS。** 热点地址是固定的 `192.168.4.1`（或配网后的路由器地址），SDK 不做服务发现。
- **同一时刻只解析一个入向包。** 走 UDP 时 SDK 强制发送间隔 ≥20 ms；WebSocket 不需要，但也不要
  在一个事件循环里无间隔地灌包。

## 自定义策略：训练与导出约定

- **维度契约与训练侧一致：观测 61 维、动作 14 维。** 文件头里 `obs_dim=61`、`act_dim=14`，
  固件不匹配直接拒载。
- **导出**：训练产物经 `training/export/export_policy.py` 得到 `*_int8.h`，再在仓库根运行
  `uv run firmware/tools/assets.py policy policies/<硬件版本>/<动作目录>/`，同目录生成
  `<模型>.mdp`。转换时硬件版本取自目录名，模型名取自动作目录。
- **名字**：2..15 个字符，只允许 `[A-Za-z0-9_-]`；文件头里的模型名必须与文件名主干一致。
  6 个核心槽位（`standup`、`velstand-flat`、`sitstand`、`ground-pick`、`ball-kick`、`roulade`）
  不要占用。
- **上传**：`duck.policy.upload("velstand-flat.mdp")` 先在本地核对 128 字节文件头（magic、版本、
  硬件版本、模型名、维度），不合格不发板；通过后 POST 到文件页 83 口。上传后 `policy.load(name)`
  热载，或重启让开机流程载入。
- **运行**：`duck.policy.run("my-policy", seconds=10, scale=1.0)`。固件单次最长 60 s；
  `run()` 在固件接受后阻塞轮询 SSTA，直到状态离开 `SKILL`（或超时），中途可以被
  `duck.policy.stop()` 或任意运动命令打断。

## 已知限制（v1）

- **没有推送。** 状态与事件都是轮询；实时性受 20 Hz 发送与会话轮询间隔限制。
- **没有鉴权。** 固件是开放热点 + 明文 HTTP/WS，只适合局域网联调。
- **explore 漫游不在 v1。** 避障 / 自主漫游接口没有封装。
- **视频未接入。** `duck.video` 构造即抛 `NotImplementedError`。

## 运行例程

先连接小鸭子的热点。在学员仓根目录运行 `uv run python sdk/pyduck/examples/01_state_and_sensors.py` 读状态与传感器；
其余例程见 [examples/](examples/)。
