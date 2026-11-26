"""网络与 PPO：`trainer.fit(PPO(net), PPOData(env, net))`。

- `network.py`：策略与价值网络 `ActorCritic`。
- `data.py`：data 一半的骨架 `RolloutData`：每轮用当前网络在仿真里采一段（`sample_rollout`），换成训练样本、切成批。
- `module.py`：model 一半的骨架 `PolicyModule`：一批数据的损失，Adam 更新，actor 与 critic 分开裁梯度。
- `advantage.py`：广义优势估计 `compute_gae`。
- `ppo.py`：PPO 的两半 `PPOData`、`PPO`。
- `distill.py`：行为克隆（出厂的走路与起身在 PPO 之后各加一步克隆）。
"""
from training.ppo.advantage import compute_gae, normalize
from training.ppo.data import RolloutData, flat, sample_rollout
from training.ppo.module import PolicyModule
from training.ppo.network import ActorCritic
from training.ppo.ppo import PPO, PPOData

__all__ = ["PPO", "ActorCritic", "PPOData", "PolicyModule", "RolloutData", "compute_gae", "flat", "normalize",
           "sample_rollout"]
