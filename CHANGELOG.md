# 更新日志

版本号是发布日期加当天序号。每个版本在 [Releases](https://github.com/Xbotics-Embodied-AI-club/Xbotics-MiniDuck-Course/releases) 有同名发布，发布后不再改动。

## v20261102.04

第 3 期 notebook 的 DQN 固定了随机动作的种子，每次运行结果相同，与视频一致。硬件版本、固件、策略都不变。

**你要做什么**

- 电脑：`git pull`，不用重新 `uv sync`。
- 鸭子：不用动。

## v20261102.03

第 3 期的 notebook 换成 `notebooks/ep03_sarsa_dqn.ipynb`：在 gym 的悬崖行走上写 SARSA 与 Q 学习，在倒立摆上写 DQN；加入 gymnasium 依赖。硬件版本、固件、策略都不变。

**你要做什么**

- 电脑：`git pull` 后运行 `uv sync`。
- 鸭子：不用动。

## v20261102.02

README 写清 `training/` 与 `notebooks/` 的关系。代码、硬件版本、固件、策略都不变。

**你要做什么**

- 电脑：`git pull`，不用重新 `uv sync`。
- 鸭子：不用动。

## v20261102.01

加入 `notebooks/`，放第 2、3 期课上写的代码，文件名前缀与期数对应。`training/` 里导出策略的工具收进 `training/export_tools/`，`training/` 随 `uv sync` 装进环境，在任何目录都能 `import`。硬件版本、固件、策略都不变。

**你要做什么**

- 电脑：`git pull` 后运行 `uv sync`。导出策略的命令改成 `uv run python training/export_tools/export_policy.py`。
- 鸭子：不用动。

## v20261101.01

仓库里直接放编译好的固件 `firmware/0.8.1/`，固件本身仍是 0.8.1，文件与之前的固件包逐字节相同。`policies/` 里只留传给机器人的 `.mdp` 和清单，仿真器直接跑 `.mdp`。硬件版本、策略都不变。

**你要做什么**

- 电脑：`git pull`，不用重新 `uv sync`。导出过自己策略的，先把 `policies/` 里改动的文件移到别处，再 `git pull`。
- 鸭子：不用动。

## v20261031.03

`uv.lock` 钉住 JupyterLab 与 matplotlib 的版本。硬件版本、固件、策略都不变。

**你要做什么**

- 电脑：`git pull` 后运行 `uv sync`。
- 鸭子：不用动。

## v20261031.02

加入 JupyterLab 与 matplotlib，用来在 notebook 里动手。硬件版本、固件、策略都不变。

**你要做什么**

- 电脑：`git pull` 后运行 `uv sync`。
- 鸭子：不用动。

## v20261031.01

初始版本。

| 内容 | 版本 |
| --- | --- |
| 硬件版本 | `2026-10-24-hd1910`（飞特 HD-1910，V0.2 头板 + 身体板） |
| 固件 | 0.8.1 |
| 出厂策略 | standup · velstand-flat · sitstand |

**你要做什么**

- 鸭子：固件是 0.7.6 到 0.8.0 时，按手册「固件与策略更新」只烧 `0x10000_firmware.bin`；更早的固件六个文件全烧；已经是 0.8.1 的不用动。
- 电脑：第一次 `git clone` 后运行 `uv sync`。
