# 课程表

《小鸭子带你学强化学习》分两条线。主线 9 讲是实操，按顺序学：从强化学习的基本概念讲起，亲手训练、量化，最后把自己训的策略部署到小鸭子真机上。选修 5 讲讲强化学习的推导，给想深入的同学。每一讲是一集视频，动手的部分在仓库的 `notebooks/` 里，一讲一个文件夹，文件夹名就是讲次（如 `M06-sim-align/`）。

<figure class="duck-figure">
  <img src="../_static/deck/sim-to-real.webp" width="1000" height="563" alt="现场课并排展示小鸭子在仿真和真机上的动作">
  <figcaption>现场课 · 从仿真训练，到真机复现。</figcaption>
</figure>

<h2>先看这几讲</h2>

::::{container} duck-grid duck-grid-three
:::{container} duck-card
[主线 01 · 上手 MiniDuck](01-miniduck.md)

连热点、用手机遥控，认识机身里的舵机和传感器。
:::
:::{container} duck-card
[主线 02 · 强化学习的数学表达](02-rl-problem.md)

从智能体、环境和回报讲起，动手跑一次仿真。
:::
:::{container} duck-card
[选修 S1–S2 · 价值函数与 DQN](03-value-bellman.md)

先用三格走廊推导，再到悬崖行走和倒立摆上练习。
:::
::::

## 主线

| 讲次 | 主题 | |
| --- | --- | --- |
| 01 | 上手 MiniDuck | [观看](01-miniduck.md) |
| 02 | 强化学习的数学表达：智能体、环境与回报 | [观看](02-rl-problem.md) |
| 03 | 一集讲清 PPO：沿着回报往上爬 | 制作中 |
| 04 | 跑通一次训练 | 制作中 |
| 05 | 改奖励，训自己的版本 | 制作中 |
| 06 | 仿真建模与真机对齐 | 制作中 |
| 07 | 域随机化与多动作训练 | 制作中 |
| 08 | 模型量化与真机部署 | 制作中 |
| 09 | 编辑自己的动作 | 制作中 |

## 选修

| 讲次 | 主题 | |
| --- | --- | --- |
| S1 | 价值函数与贝尔曼方程 | [观看](03-value-bellman.md)（视频前半） |
| S2 | 时序差分与 DQN | [观看](03-value-bellman.md)（视频后半） |
| S3 | 策略梯度与 REINFORCE | 制作中 |
| S4 | Actor-Critic 与广义优势估计 | 制作中 |
| S5 | 从 TRPO 到 PPO | 制作中 |

## 打开 notebook

先照[仿真器](../simulator.md)装好环境，然后在仓库目录里：

```bash
uv run jupyter lab notebooks/
```

