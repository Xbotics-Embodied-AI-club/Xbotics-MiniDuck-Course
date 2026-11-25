# 课上的动手代码

一讲一个文件夹，文件夹名就是讲次。环境、网络、评测都从 `training/` 导入（`uv sync` 时已装进环境），notebook 里只写这一讲要讲的东西。

| 文件夹 | 这一讲写什么 | 用到仓库里的 |
| --- | --- | --- |
| `M02-rl-math/` | 建环境、走几步、算回报 | `env.DuckEnv`、`hardware` |
| `S02-td-dqn/` | SARSA、Q 学习、DQN（gym 的悬崖行走与倒立摆） | — |
| `S03-policy-gradient/` | 回报 `reward_to_go`；`Reinforce`：把一轮数据换成样本、策略梯度的损失 | `onpolicy`（采样、每轮重采、Lightning 骨架）、`model.ActorCritic`、`rollout.evaluate` / `record_video` |
| `S04-actor-critic/` | `A2C(Reinforce)`：优势换成 critic 给的 GAE，损失加价值损失 | 上面这些，加 `model.compute_gae` |
| `S05-trpo-ppo/` | `PPO(A2C)`：小批反复用、裁剪的目标、按 KL 调学习率 | 同上 |

S04、S05 是把上一讲的 notebook 另存一份接着写，所以前面几格就是上一讲的代码。
三讲一层层继承下来的 PPO，与正式训练器 `training/train_ppo.py` 是同一套算法；正式训练器另有记录、存盘、课程表与行为克隆。
