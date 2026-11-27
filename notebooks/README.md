# 课上的动手代码

一讲一个文件夹，文件夹名就是讲次。仿真、网络、评测都从 `training` 包导入（`uv sync` 时已装进环境），notebook 里只写这一讲要讲的东西。
训练都是 Lightning 的一行 `trainer.fit(model, data)`：data 一半每轮用当前网络在仿真里采样、换成训练样本，model 一半算一批数据的损失。

| 文件夹 | 这一讲写什么 | 用到仓库里的 |
| --- | --- | --- |
| `M02-rl-math/` | 建环境、走几步、算回报 | `training.sim` |
| `M04-train/` | 用仓库里的 PPO 从零训练平地直走：建环境、建网络、`PPOData` 与 `PPO`、写 Trainer、`fit`、评测和录像 | `training.sim`、`training.ppo`、`training.evaluate` |
| `M05-reward/` | 两组从同一份出厂走路检查点开始，在 M04 的 `course-walk` 续训 300 轮，只切换 `air_time` 权重 | 同上 |
| `S02-td-dqn/` | SARSA、Q 学习、DQN（gym 的悬崖行走与倒立摆） | — |
| `S03-policy-gradient/` | v1 REINFORCE：回报 `reward_to_go`；data 一半 `ReinforceData`、model 一半 `Reinforce` | `RolloutData`、`PolicyModule`（两半的骨架）、`ActorCritic`、`evaluate` / `record_video` |
| `S04-actor-critic/` | v2 A2C：手写广义优势估计；`A2CData`、`A2C`（加价值损失） | 同上 |
| `S05-trpo-ppo/` | v3 PPO：切 4 个小批用 5 遍、裁剪的目标、按 KL 调学习率；最后三版放在同一个评测里比 | 同上，加 `compute_gae` |

选修三讲各自独立：每一版的两半都直接继承同一对骨架，版本之间的差别就是这两半里改的那几行。
S04、S05 最后比较时会读上一讲存下的权重，所以按 S03 → S04 → S05 的顺序跑。

主线 M04、M05 用 `course-walk`：先练平地向前走，再加入轻推和少量站立指令。S03–S05 共用 `course-walk-elective`，前 500 轮沿用同一设置，之后重置时从站姿附近的小范围实际关节角开始，练不同初态下的起步。两种任务都用同一硬件档案，不训练后退、原地转圈、摆头或跌倒起身。

主线 04 的完整训练（6000 轮）从仓库根执行：

```bash
source .envrc
MINIDUCK_HARDWARE=2026-11-26-hd1910 MINIDUCK_SKILL=course-walk MINIDUCK_RUN_NAME=m04-walk uv run python -m training.train_ppo
```

主线 05 从同一份出厂走路训练检查点开始。请从[配套课程版本的 Release 附件](https://github.com/Xbotics-Embodied-AI-club/Xbotics-MiniDuck-Course/releases)下载 `m05_factory_walk.pt`，放在 `notebooks/M05-reward/` 再运行；notebook 的第一格会校验文件 SHA-256。`policies/` 下的板端 `.mdp` 只供推理，不能代替训练检查点。
