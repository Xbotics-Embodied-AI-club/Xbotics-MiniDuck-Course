"""从部署用的 ONNX 取回 actor，作为续训的起点。

上游发布的策略（如 alpha_stand.onnx）只有 ONNX，没有训练 checkpoint。rsl_rl 导出时把观测归一化
烘进了图里：`actions = MLP((obs − mean) / (std + eps))`，四层全连接、中间三个 ELU。
我们的模型是同一个结构、同一个 eps（`model.NORMALIZER_EPS`），所以能逐层搬回来：
    mean → actor_normalizer.mean；除数 − eps = std → actor_normalizer.var = std²（源训练里始终不变的观测维
    方差是 0，导出的除数正好等于 eps，反推成 std = 0，与源模型一致）；
    四个 Gemm 的权重与偏置 → actor 的四个线性层。
critic 与动作标准差 ONNX 里没有，原样留着，由调用方决定怎么处理。
搬完用 onnxruntime 对同一批观测核对一遍输出，对不上就报错 —— 不靠「结构看起来一样」。
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper

from model import NORMALIZER_EPS, ActorCritic

EXPECTED_OPS = ["Sub", "Div", "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"]
NUM_CHECK_OBS = 256
CHECK_TOLERANCE = 1e-4
ORT_LOG_WARNING = 2
# 除数与 eps 的比较容差：eps 写进 ONNX 时是 float32（0.01 → 0.0099999998），比 eps "小" 那一点只是舍入
DIVISOR_EPS_TOLERANCE = 1e-7


def load_actor_from_onnx(model: ActorCritic, path: Path, normalizer_count: int) -> None:
    """把 rsl_rl 风格 ONNX 里的 actor 与观测归一化装进 `model`。

    Args:
        model: 要装入的模型（只动 actor 与 actor 归一化）。
        path: ONNX 路径。
        normalizer_count: 归一化统计量当作已经看过多少个样本。在线归一化按
            「新批量 / 累计样本数」的比例更新，计数从 0 起的话第一批数据就把搬来的统计量整个冲掉；
            取源训练实际看过的样本量（迭代数 × 每轮步数 × 环境数），续训时更新幅度就与源训练末尾一致。

    Raises:
        RuntimeError: 图的算子序列不是「归一化 + 四层全连接」，形状与模型不符，
            Gemm 不是 y = x·Wᵀ + b 的标准形式，或搬完后输出与 onnxruntime 对不上。
    """
    graph = onnx.load(str(path)).graph
    ops = [node.op_type for node in graph.node]
    if ops != EXPECTED_OPS:
        raise RuntimeError(f"{path.name} 的算子序列是 {ops}，不是 rsl_rl 导出的「归一化 + 四层全连接」"
                           f"（{EXPECTED_OPS}），搬不过来")
    # to_array 给的是只读视图；复制一份，torch 才能安全地从它建张量。
    initializers = {init.name: numpy_helper.to_array(init).copy() for init in graph.initializer}
    sub, div, *rest = graph.node
    gemms = [node for node in rest if node.op_type == "Gemm"]
    for gemm in gemms:
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in gemm.attribute}
        if attrs.get("transB") != 1 or attrs.get("alpha", 1.0) != 1.0 or attrs.get("beta", 1.0) != 1.0:
            raise RuntimeError(f"{path.name} 的 {gemm.name} 属性 {attrs} 不是 y = x·Wᵀ + b")

    mean = initializers[sub.input[1]].reshape(-1)
    divisor = initializers[div.input[1]].reshape(-1)
    # 源训练里始终不变的观测维（例：坐站任务用不上的侧移 / 转向 / 身体姿态命令）方差就是 0，除数正好等于 eps；
    # 只有明显比 eps 还小才说明除数不是「std + eps」，那时搬不了
    if np.any(divisor < NORMALIZER_EPS - DIVISOR_EPS_TOLERANCE):
        raise RuntimeError(f"{path.name} 的归一化除数有比 eps 还小的分量，不是「std + eps」的形式，反推不出标准差")
    std = np.clip(divisor.astype(np.float64) - NORMALIZER_EPS, 0.0, None).astype(np.float32)
    linears = [model.actor[i] for i in range(0, len(model.actor), 2)]
    pairs = [(initializers[g.input[1]], initializers[g.input[2]]) for g in gemms]
    expected = [(tuple(layer.weight.shape), tuple(layer.bias.shape)) for layer in linears]
    found = [(w.shape, b.shape) for w, b in pairs]
    if mean.shape != tuple(model.actor_normalizer.mean.shape) or found != expected:
        raise RuntimeError(f"{path.name} 的形状（归一化 {mean.shape}，各层 {found}）与模型"
                           f"（归一化 {tuple(model.actor_normalizer.mean.shape)}，各层 {expected}）不符")

    with torch.no_grad():
        normalizer = model.actor_normalizer
        normalizer.mean.copy_(torch.from_numpy(mean))
        normalizer.var.copy_(torch.from_numpy(std ** 2))
        normalizer.count.fill_(normalizer_count)
        for layer, (weight, bias) in zip(linears, pairs, strict=True):
            layer.weight.copy_(torch.from_numpy(weight))
            layer.bias.copy_(torch.from_numpy(bias))
    check_matches_onnx(model, path, mean, std)


def check_matches_onnx(model: ActorCritic, path: Path, mean: np.ndarray, std: np.ndarray) -> None:
    """在源训练的观测分布附近（均值 ± 3 个标准差）抽一批观测，核对两边输出。

    Args:
        model: 已装好 actor 的模型。
        path: ONNX 路径。
        mean: 观测均值。
        std: 观测标准差。

    Raises:
        RuntimeError: 最大误差超过 CHECK_TOLERANCE。
    """
    rng = np.random.default_rng(0)
    obs = (mean + std * rng.uniform(-3.0, 3.0, size=(NUM_CHECK_OBS, mean.size))).astype(np.float32)
    options = ort.SessionOptions()
    options.log_severity_level = ORT_LOG_WARNING
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    reference = np.concatenate([session.run(None, {session.get_inputs()[0].name: row[None]})[0]
                                for row in obs])
    # 在 CPU 上的副本里算：训练入口开了 TF32，GPU 上的矩阵乘只有约 1e-3 的精度，
    # 核对的是搬运本身，不该把这份精度损失算进来。
    with torch.no_grad():
        ours = copy.deepcopy(model).cpu().act_inference(torch.from_numpy(obs)).numpy()
    error = float(np.abs(ours - reference).max())
    if error > CHECK_TOLERANCE:
        raise RuntimeError(f"从 {path.name} 搬来的 actor 与 ONNX 输出最大差 {error:.2e}，"
                           f"超过 {CHECK_TOLERANCE:.0e}")
