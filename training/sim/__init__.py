"""仿真：一个动作的并行仿真环境（`DuckEnv`、`SKILLS`），硬件版本（`hardware`），舵机模型（`servos`），
坡地（`terrain`、`upright`）与起身（`getup`）的环境改动。"""
from training.sim.env import SKILLS, DuckEnv

__all__ = ["SKILLS", "DuckEnv"]
