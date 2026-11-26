# 课上的动手代码

一讲一个文件夹，文件夹名就是讲次。仿真、网络、评测都从 `training` 包导入（`uv sync` 时已装进环境），notebook 里只写这一讲要讲的东西。
训练都是 Lightning 的一行 `trainer.fit(model, data)`：data 一半每轮用当前网络在仿真里采样、换成训练样本，model 一半算一批数据的损失。

| 文件夹 | 这一讲写什么 | 用到仓库里的 |
| --- | --- | --- |
| `M02-rl-math/` | 建环境、走几步、算回报 | `training.sim` |
| `M04-train/` | 用仓库里的 PPO 拼一个训练：建环境、建网络、`PPOData` 与 `PPO`、写 Trainer、`fit`、评测和录像 | `training.sim`、`training.ppo`、`training.evaluate` |
| `M05-reward/` | 改一项奖励权重（`SKILLS["walk"].with_reward_weights(air_time=0.0)`），照主线 04 拼起来训，和主线 04 的结果比 | 同上 |
| `S02-td-dqn/` | SARSA、Q 学习、DQN（gym 的悬崖行走与倒立摆） | — |
| `S03-policy-gradient/` | v1 REINFORCE：回报 `reward_to_go`；data 一半 `ReinforceData`、model 一半 `Reinforce` | `RolloutData`、`PolicyModule`（两半的骨架）、`ActorCritic`、`evaluate` / `record_video` |
| `S04-actor-critic/` | v2 A2C：手写广义优势估计，和 `training/ppo/advantage.py` 逐位比对；`A2CData`、`A2C`（加价值损失） | 同上，加 `compute_gae` |
| `S05-trpo-ppo/` | v3 PPO：切 4 个小批用 5 遍、裁剪的目标、按 KL 调学习率；最后和 `training/ppo/ppo.py` 逐位比对，三版放在同一个评测里比 | 同上 |

选修三讲各自独立：每一版的两半都直接继承同一对骨架，版本之间的差别就是这两半里改的那几行。
S04、S05 最后比较时会读上一讲存下的权重，所以按 S03 → S04 → S05 的顺序跑。
