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

## 课程与动手代码

主线 9 讲人人按顺序学，选修 5 讲给想深入推导的学员。动手代码一讲一个文件夹，文件夹名就是讲次。

| 讲次 | 视频 | 动手代码 |
| --- | --- | --- |
| 主线 01 上手 MiniDuck | [B 站](https://www.bilibili.com/video/BV1rFxe6hEdr) | — |
| 主线 02 强化学习的数学表达 | [B 站](https://www.bilibili.com/video/BV1xExx6jEGW) | [`notebooks/M02-rl-math/`](notebooks/M02-rl-math) |
| 主线 03 一集讲清 PPO | — | — |
| 主线 04 跑通一次训练 | — | [`notebooks/M04-train/`](notebooks/M04-train) |
| 主线 05 改奖励，训自己的版本 | — | [`notebooks/M05-reward/`](notebooks/M05-reward) |
| 主线 06 仿真建模与真机对齐 | — | — |
| 选修 S01 价值函数与贝尔曼方程 | [B 站](https://www.bilibili.com/video/BV1es4G6ZExV)（与 S02 同一集） | — |
| 选修 S02 时序差分与 DQN | [B 站](https://www.bilibili.com/video/BV1es4G6ZExV)（与 S01 同一集） | [`notebooks/S02-td-dqn/`](notebooks/S02-td-dqn) |
| 选修 S03 策略梯度与 REINFORCE | — | [`notebooks/S03-policy-gradient/`](notebooks/S03-policy-gradient) |
| 选修 S04 Actor-Critic 与广义优势估计 | — | [`notebooks/S04-actor-critic/`](notebooks/S04-actor-critic) |
| 选修 S05 从 TRPO 到 PPO | — | [`notebooks/S05-trpo-ppo/`](notebooks/S05-trpo-ppo) |

装配动画和整套课程表在[手册](https://xbotics-embodied-ai-club.github.io/Xbotics-MiniDuck-Course/)里。

## 仓库里有什么

| 路径 | 内容 |
| --- | --- |
| `sim_model/` | 这只鸭子的仿真模型：结构与质量、舵机、供电，按硬件版本组织 |
| `training/` | 训练代码，`uv sync` 时作为 `training` 包装进环境，在任何目录都能导入（`from training.sim import DuckEnv`）。按一只鸭子从仿真到真机的顺序分成几块：`sim/` 仿真环境与硬件版本，`ppo/` 网络与 PPO（`trainer.fit(PPO(net), PPOData(env, net))`），`resume/` 从已有策略接着训，`evaluate.py` 评测与录像，`export/` 导出到真机与策略查看器，`train_ppo.py` 一条命令训完一个动作 |
| `notebooks/` | 课上带着写的代码，一讲一个文件夹，文件夹名就是讲次（`M02-rl-math/` 是主线 02，`S03-policy-gradient/` 是选修 S03）。主线 04、05 用 `training` 里的模块拼一个训练；选修 S03–S05 把 `training/ppo` 一版一版手写出来（v1 REINFORCE → v2 A2C → v3 PPO） |
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
uv run python training/export/viewer.py --skill velstand-flat   # 在仿真器里看出厂策略走路
```
