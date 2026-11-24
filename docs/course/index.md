# 课程表

《小鸭子带你学强化学习》用这台小鸭子，从强化学习的基本概念一路讲到训练出会走路的策略、再部署到真机上。每一讲是一集视频，动手的部分在仓库的 `notebooks/` 里。

| 讲次 | 主题 | |
| --- | --- | --- |
| 01 | 上手 MiniDuck | [观看](01-miniduck.md) |
| 02 | 强化学习的数学表达：智能体、环境与回报 | [观看](02-rl-problem.md) |
| 03 | 价值函数与贝尔曼方程 | [观看](03-value-bellman.md) |
| 04 | 时序差分与 DQN | 制作中 |
| 05 | 策略梯度与 REINFORCE | 制作中 |
| 06 | Actor-Critic 与广义优势估计 | 制作中 |
| 07 | 从 TRPO 到 PPO | 制作中 |
| 08 | 用 PPO 训练行走 | 制作中 |
| 09 | 其他任务与奖励函数 | 制作中 |
| 10 | 仿真建模与真机对齐 | 制作中 |
| 11 | 域随机化与多动作训练 | 制作中 |
| 12 | 模型量化原理 | 制作中 |
| 13 | 量化验证与真机部署 | 制作中 |

## 打开 notebook

先照[仿真器](../simulator.md)装好环境，然后在仓库目录里：

```bash
uv run jupyter lab notebooks/
```

```{toctree}
:hidden:

01-miniduck
02-rl-problem
03-value-bellman
```
