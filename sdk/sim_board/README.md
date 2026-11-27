# 用 pyduck 连本机仿真

本机仿真板用学员仓的 MuJoCo 模型和 `2026-11-26-hd1910` 出厂策略运行。它只监听 `127.0.0.1`，不改真机。ToF 读数来自脸部传感器向挡板打出的射线；没有回波时读数为 `-1`。

## 开始

在学员仓根目录，用 Python 3.12 同步一遍锁定环境：`uv sync --locked`。启动仿真板（不用填模型路径）：

```sh
sudo "$(uv run --locked python -c 'import sys; print(sys.executable)')" -m sdk.sim_board.server
```

上面的 `sudo` 只用于 Linux 默认不让普通用户绑定低端口的情况：SDK 默认用 WebSocket 88 读状态、发动作，用 HTTP 83 传策略。环境同步仍在普通用户下执行。系统允许普通用户监听这两个端口时，直接用 `uv run --locked python -m sdk.sim_board.server`。也可不改系统设置，启动时加 `--ws-port 18088 --files-port 18083`，自己的代码用 `Duck.connect("127.0.0.1", ws_port=18088, web_ports={"files": 18083})`；课件和例程 02 使用默认端口。

看到 `READY` 后，另开终端，从学员仓根目录运行：

```sh
uv run --locked python sdk/pyduck/examples/02_drive_and_avoid.py --ip 127.0.0.1
```

然后用课件 1.2 的四行读状态；刚启动还没站起时，状态是「待机」。

```python
import pyduck
with pyduck.Duck.connect("127.0.0.1") as duck:
    st = duck.state.get()
    print(st.state_cn, st.tof_mm, st.loop_hz)
```

## 接口范围

支持状态、事件、站立/坐下/卸力、运动与出厂一次性动作，以及 `.mdp` 文件的列表、上传、热载、运行、停止和删除。只有待机或卸力时才可写文件；策略上传保存在内存，进程退出后清空。挡板场景不含球，踢球只执行动作。电压是仿真标称值；不提供 UDP、真机外设诊断或固件录像下载。
