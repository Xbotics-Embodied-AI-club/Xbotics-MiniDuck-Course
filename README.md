# 小鸭子带你学强化学习

**主页与使用手册：<https://xbotics-embodied-ai-club.github.io/Xbotics-MiniDuck-Course/>**

MiniDuck 是一台高约 25 厘米、质量约 0.78 千克的双足机器人，行走、抗扰动和跌倒恢复都由神经网络控制。
拿到小鸭子后，照主页的手册上电、连接，就能让它走起来。

| 当前版本 | |
| --- | --- |
| 硬件版本 | `2026-10-24-hd1910`（飞特 HD-1910，V0.2 头板 + 身体板） |
| 固件 | 0.8.1 |

完整声明见 [versions.toml](versions.toml)，每次更新改了什么、你要做什么见 [CHANGELOG.md](CHANGELOG.md)。
编译好的固件在 [firmware/0.8.1/](firmware/0.8.1)，烧写工具在[这里下载](https://github.com/Xbotics-Embodied-AI-club/Xbotics-MiniDuck-Course/releases/download/v20261031.01/flash_download_tool_3.9.11.zip)。

## 仓库里有什么

| 路径 | 内容 |
| --- | --- |
| `sim_model/` | 这只鸭子的仿真模型：结构与质量、舵机、供电，按硬件版本组织 |
| `training/` | 正式代码：仿真环境、PPO 训练、评测、策略查看器；`uv sync` 时装进环境。把策略导出到真机的工具收在 `training/export_tools/` |
| `notebooks/` | 课上带着写的代码，一期一个（`ep03_corridor.ipynb` 是第 3 期）；需要时调用 `training/`，`training/` 不依赖它 |
| `policies/` | 出厂策略，与仿真模型按同一个硬件版本对应；`*.mdp` 是传给机器人的文件，仿真器也直接跑它 |
| `firmware/0.8.1/` | 编译好的固件：六个 `.bin` 文件和一份烧写地址，不用自己编译 |
| `firmware/tools/` | 电脑端工具：把策略和音频转成板上格式、上传到机器人 |
| `firmware/lib/policy/` | 策略推理的 C 代码，导出策略时在电脑上编译核对 |

## 装仿真环境

需要 [uv](https://docs.astral.sh/uv/)。训练要一台带 NVIDIA 显卡的 Linux 电脑。

```bash
git clone https://github.com/Xbotics-Embodied-AI-club/Xbotics-MiniDuck-Course.git
cd Xbotics-MiniDuck-Course
uv sync
uv run python training/run_policy_viewer.py --skill velstand-flat   # 在仿真器里看出厂策略走路
```

## 署名

作者：锟酱 · Xbotics 具身智能社区。机器人的结构模型与训练任务来自 [Pollen Robotics 的 microduck_rl](https://github.com/pollen-robotics/microduck_rl)，
我们换了电池与电路板，按实称修正了质量，并换用飞特 HD-1910 舵机。

