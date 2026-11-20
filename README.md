# 小鸭子带你学强化学习

**主页与使用手册：<https://xbotics-embodied-ai-club.github.io/Xbotics-MiniDuck-Course/>**

MiniDuck 是一台高约 25 厘米、质量约 0.79 千克的双足机器人，行走、抗扰动和跌倒恢复都由神经网络控制。
拿到小鸭子后，照主页的手册上电、连接，就能让它走起来。

| 当前版本 | |
| --- | --- |
| 硬件版本 | `2026-11-07-hd1910`（飞特 HD-1910，V0.2 头板 + 身体板） |
| 固件 | 0.9.1 |
| 出厂策略 | 起身 · 走路 · 坐站 · 捡东西 · 踢球 · 翻滚 |

完整声明见 [versions.toml](versions.toml)，每次更新改了什么、你要做什么见 [CHANGELOG.md](CHANGELOG.md)。
编译好的固件在 [firmware/0.9.1/](firmware/0.9.1)，烧写工具在[这里下载](https://github.com/Xbotics-Embodied-AI-club/Xbotics-MiniDuck-Course/releases/download/v20261120.01/flash_download_tool_3.9.11.zip)。

## 仓库里有什么

| 路径 | 内容 |
| --- | --- |
| `sim_model/` | 这只鸭子的仿真模型：结构与质量、舵机、供电，按硬件版本组织 |
| `training/` | 正式代码，`uv sync` 时装进环境，在任何目录都能导入：仿真环境 `env.py`、硬件版本 `hardware.py`（notebook 里的 `import hardware` 就是它）、策略查看器、评测，以及第 7 期才用的正式训练器 `train_ppo.py` 与 `model.py`。把策略导出到真机的工具收在 `training/export_tools/` |
| `notebooks/` | 课上带着写的代码，文件名前缀与期数对应（`ep03_sarsa_dqn.ipynb` 是第 3 期）。环境从 `training/` 导入；策略网络、采样和训练步骤在第 4–6 期带着手写，先不用看 `model.py` 和 `train_ppo.py` |
| `policies/` | 出厂策略，与仿真模型按同一个硬件版本对应：`standup`（起身，舵机出力弱时也翻得过身）、`velstand-flat`（走路，在坡上训过）、`sitstand`（坐站）、`ground-pick`（捡东西）、`ball-kick`（踢球）、`roulade`（翻滚）；`*.mdp` 是传给机器人的文件，仿真器也直接跑它 |
| `firmware/0.9.1/` | 编译好的固件：六个 `.bin` 文件和一份烧写地址，不用自己编译 |
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
