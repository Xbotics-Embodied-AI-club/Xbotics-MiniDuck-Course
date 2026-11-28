# 仿真器

仓库里的仿真模型和你手上的小鸭子是同一个硬件版本 {{hardware}}，出厂策略可以直接在电脑上运行。

<div class="duck-hero">
  <div>
    <p class="duck-kicker">MUJOCO · 仿真</p>
    <p class="duck-lead">在电脑里先跑一遍出厂动作</p>
    <p>不用连接真机。看策略走路只需 CPU；课程里训练策略时再用显卡。</p>
  </div>
  <img src="_static/deck/simulator.webp" width="800" height="600" alt="现场课仿真渲染：小鸭子站在网格地面上">
</div>

## 安装

需要 [uv](https://docs.astral.sh/uv/)。第一次安装要下载不少依赖，需要几分钟。

```{code-block} bash
:class: duck-wrap-code

git clone https://github.com/Xbotics-Embodied-AI-club/Xbotics-MiniDuck-Course.git
cd Xbotics-MiniDuck-Course
uv sync
```

## 看出厂策略走路

```{code-block} bash
:class: duck-wrap-code

uv run python training/export/viewer.py --skill velstand-flat
```

<figure class="duck-figure">
  <video class="duck-motion" controls muted playsinline preload="none" width="540" height="540" poster="_static/deck/walk.webp" src="_static/deck/walk.mp4" aria-label="小鸭子仿真行走动画"></video>
  <figcaption>先看动作效果，再运行上面的命令；实际窗口的按键见下表。</figcaption>
</figure>

弹出 MuJoCo 窗口后，在运行这条命令的终端里按键控制：

| 按键 | 动作 |
| --- | --- |
| 长按 `W` | 持续向前走，松开后停下 |
| `S` | 后退 |
| `A`、`D` | 转向 |
| `J`、`L` | 横向平移 |
| 空格 | 清零运动指令，停止行走 |
| `R` | 重置仿真 |
| `Q` | 关闭查看器并返回终端 |

主线 M04 与选修 S03–S05 训练的是平地直走。用查看器看这些课程策略时，用 `W` 观察前进步态；后退和转向不是这两个训练任务的目标。

## 看其他动作

坐立策略用第一个指令值区分坐下和站立：`--vx 1` 表示坐下，不带这个参数表示站立。

```{code-block} bash
:class: duck-wrap-code

# 从趴下的姿势起身
uv run python training/export/viewer.py --skill standup --start prone

# 坐下
uv run python training/export/viewer.py --skill sitstand --vx 1

# 保持站立
uv run python training/export/viewer.py --skill sitstand
```

## 用 Python 连接真机

在学员仓根目录运行 `uv sync`，会一并安装随仓提供的 `pyduck`。它可以读取小鸭子的状态、控制动作和管理板上文件。

先让电脑连上小鸭子的热点，再从仓库根目录运行只读例程：

```{code-block} bash
:class: duck-wrap-code

uv run python sdk/pyduck/examples/01_state_and_sensors.py --ip 192.168.4.1
```

其他用法见 [pyduck 文档与例程](https://github.com/Xbotics-Embodied-AI-club/Xbotics-MiniDuck-Course/tree/main/sdk/pyduck)。

## 用 Python 连接本机仿真板

没有真机也能运行 SDK 例程。仿真板只监听本机地址 `127.0.0.1`，用同一硬件版本的仿真模型和出厂策略响应 `pyduck`。

在学员仓根目录用 Python 3.12 同步锁定环境，然后启动仿真板：

```{code-block} bash
:class: duck-wrap-code

uv sync --locked
uv run --locked python -m sdk.sim_board.server
```

如果 Linux 不允许普通用户监听默认端口（WebSocket 88、文件服务 83），同步依赖仍用普通用户，只把启动命令换成：

```{code-block} bash
:class: duck-wrap-code

sudo "$(uv run --locked python -c 'import sys; print(sys.executable)')" -m sdk.sim_board.server
```

看到 `READY` 后，另开终端回到学员仓根目录，运行例程 02。仿真鸭起身后会向前走，遇到挡板时转向约 90°。

```{code-block} bash
:class: duck-wrap-code

uv run --locked python sdk/pyduck/examples/02_drive_and_avoid.py --ip 127.0.0.1
```

## 打包与运行自定义策略

先按[本机仿真板文档](https://github.com/Xbotics-Embodied-AI-club/Xbotics-MiniDuck-Course/tree/main/sdk/sim_board#运行自己训练的捡取策略)导出自己的策略。以导出到 `result/mypick-export` 的 `ground-pick` 为例：

```{code-block} bash
:class: duck-wrap-code

uv run --locked python firmware/tools/assets.py policy \
  result/mypick-export/2026-11-26-hd1910/ground-pick --as mypick
```

`--as mypick` 把板端名称写进 `mypick.mdp`。名称用 2–15 个英文字母、数字、下划线或连字符，避开出厂策略名。源目录和导出清单保持不变；不要直接改 `.mdp` 的文件名。

让仿真板从待机或卸力开始。例程 03 会上传 `mypick.mdp`，按 `mypick` 热载，等小鸭子站稳再运行。如果刚跑过例程 02，可以先重启仿真板回到待机。

```{code-block} bash
:class: duck-wrap-code

uv run --locked python sdk/pyduck/examples/03_custom_policy.py --ip 127.0.0.1 \
  --mdp result/mypick-export/2026-11-26-hd1910/ground-pick/mypick.mdp
```

仿真板退出后，上传的策略随进程清空。
