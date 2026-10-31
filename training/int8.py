"""把策略（走路、站起……）量化成 int8，给 ESP32-S3 上的 esp-nn 跑。

量化口径照 TFLite / esp-nn 的 int8 约定，每一处都要与单片机上的整数运算**逐位一致**：

- 权重：每个输出通道对称量化（零点 0），用 esp-nn 的逐通道全连接内核
  `esp_nn_fully_connected_per_ch_s8`；
- 激活：整张量非对称量化（scale + 零点）；
- 偏置：int32，标度是「输入标度 × 该通道权重标度」；
- 重量化：累加和乘一个定点乘数再移位（`esp_nn_multiply_by_quantized_mult` 的 ANSI 版，
  见 `multiply_by_quantized_mult`）。esp-nn 的 S3 汇编版声明与 ANSI 版逐位一致；
- 层间 ELU：上一层的 int8 输出只有 256 种取值，所以「反量化 → ELU → 按下一层的输入标度
  再量化」整个预先算成一张 256 项的 int8 查表，单片机上不做浮点；
- 输入：原始观测先按训练时的归一化统计量在浮点里标准化，再量化成 int8。
  这一步在单片机上是浮点，编译时要 `-ffp-contract=off`，否则乘加被合成一条指令、
  舍入与这里的 float32 参考不一致。

本文件是纯 numpy，不依赖仿真器：量化、整数参考前向、C 头文件生成都在这里。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

QMIN, QMAX = -128, 127
# firmware/lib/policy/src/policy_int8.h 里的静态上限（POLICY_INT8_MAX_WIDTH / _MAX_LAYERS），两边要一致。
C_MAX_WIDTH = 512
C_MAX_LAYERS = 8
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1
# esp-nn S3 的 s8 快路径要求行长是 16 的倍数；第一层输入 61 维补零到 64。
ROW_ALIGN = 16
# 标定时激活范围取哪段分位数：两端各砍掉万分之一，免得几个离群值把整段分辨率吃掉。
CALIB_PERCENTILES = (0.01, 99.99)


def quantize_multiplier(real: float) -> tuple[int, int]:
    """把一个正实数乘数拆成 int32 定点乘数与移位：real ≈ mult × 2^(shift − 31)。

    Args:
        real: 正实数乘数。

    Returns:
        (mult, shift)。mult 落在 [2^30, 2^31)；shift 为正表示左移。

    Raises:
        ValueError: 乘数不是正数。
    """
    if not real > 0.0:
        raise ValueError(f"重量化乘数必须为正，得到 {real}")
    frac, exp = math.frexp(real)
    mult = math.floor(frac * (1 << 31) + 0.5)
    if mult == 1 << 31:
        mult //= 2
        exp += 1
    return int(mult), int(exp)


def sat_round_doubling_high_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """esp-nn 的 `esp_nn_sat_round_doubling_high_mul`：(a·b·2 + 舍入) 的高 32 位。

    Args:
        a: int64 数组，取值在 int32 范围内。
        b: int64 数组（可广播），取值在 int32 范围内。

    Returns:
        int64 数组，取值在 int32 范围内。
    """
    overflow = (a == b) & (a == INT32_MIN)
    nudge = np.where((a < 0) ^ (b < 0), 1 - (1 << 30), 1 << 30)
    product = a * b + nudge
    # esp_nn_pick_sat_high32_of64：负数先加 2^31−1 再算术右移，等价于向零取整。
    to_add = (product >> 63) & ((1 << 31) - 1)
    result = (product + to_add) >> 31
    return np.where(overflow, INT32_MAX, result)


def divide_by_power_of_two(value: np.ndarray, exponent: np.ndarray) -> np.ndarray:
    """esp-nn 的 `esp_nn_div_by_power_of_two`：右移，四舍五入（.5 远离零）。

    Args:
        value: int64 数组。
        exponent: 非负的移位量（可广播）。

    Returns:
        int64 数组。
    """
    mask = (np.int64(1) << exponent) - 1
    remainder = value & mask
    result = value >> exponent
    threshold = (mask >> 1) + (result < 0)
    return result + (remainder > threshold)


def multiply_by_quantized_mult(x: np.ndarray, mult: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """esp-nn 的 `esp_nn_multiply_by_quantized_mult`（ANSI 版）。

    Args:
        x: int64 累加和。
        mult: 定点乘数（可广播）。
        shift: 移位（可广播），正为左移。

    Returns:
        重量化后的 int64 数组。
    """
    left = np.maximum(shift, 0)
    right = np.maximum(-shift, 0)
    return divide_by_power_of_two(sat_round_doubling_high_mul(x << left, mult), right)


def activation_params(lo: float, hi: float) -> tuple[float, int]:
    """给一段激活范围定 int8 的标度与零点。范围总会被撑到包含 0（0 必须能被精确表示）。

    Args:
        lo: 范围下界。
        hi: 范围上界。

    Returns:
        (scale, zero_point)。
    """
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    scale = (hi - lo) / (QMAX - QMIN) if hi > lo else 1.0
    zero = int(np.clip(round(QMIN - lo / scale), QMIN, QMAX))
    return scale, zero


def elu(x: np.ndarray) -> np.ndarray:
    """ELU(α=1)，与 `torch.nn.ELU()` 相同。

    Args:
        x: 输入。

    Returns:
        x>0 时为 x，否则 e^x − 1。
    """
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


@dataclass
class Int8Layer:
    """一层全连接的 int8 参数，字段与 esp-nn 逐通道全连接的入参一一对应。"""

    weight: np.ndarray     # int8 [out, in_padded]
    bias: np.ndarray       # int32 [out]
    mult: np.ndarray       # int32 [out]
    shift: np.ndarray      # int32 [out]
    in_scale: float
    in_zero: int
    out_scale: float
    out_zero: int
    # 隐藏层：本层 int8 输出 → ELU → 下一层 int8 输入 的查表（下标是 q + 128）；最后一层为 None。
    lut: np.ndarray | None


@dataclass
class Int8Policy:
    """整个 actor 的 int8 版本：浮点归一化 + 输入量化 + 四层全连接。"""

    obs_mean: np.ndarray      # float32 [61]
    obs_inv_std: np.ndarray   # float32 [61]，= 1 / (sqrt(var) + eps)
    in_inv_scale: np.float32  # 输入量化：q = floor(x · in_inv_scale + (in_zero + 0.5))
    layers: list[Int8Layer]

    @property
    def obs_dim(self) -> int:
        """观测维度。"""
        return int(self.obs_mean.shape[0])

    def quantize_input(self, obs: np.ndarray) -> np.ndarray:
        """原始观测 → 第一层的 int8 输入（已补零到对齐长度）。

        全程 float32，与单片机上的写法同序：先减均值、再乘倒数标准差、再乘倒数标度加零点。

        Args:
            obs: 形状 (N, 61) 的原始观测。

        Returns:
            形状 (N, in_padded) 的 int64 数组（取值在 int8 范围内）。
        """
        x = (obs.astype(np.float32) - self.obs_mean) * self.obs_inv_std
        t = x * self.in_inv_scale + np.float32(self.layers[0].in_zero + 0.5)
        q = np.clip(np.floor(t), QMIN, QMAX).astype(np.int64)
        padded = np.zeros((q.shape[0], self.layers[0].weight.shape[1]), dtype=np.int64)
        padded[:, : q.shape[1]] = q
        return padded

    def forward_q(self, obs: np.ndarray) -> np.ndarray:
        """整数参考前向，返回最后一层的 int8 输出。单片机上应当逐位得到同一个结果。

        Args:
            obs: 形状 (N, 61) 的原始观测。

        Returns:
            形状 (N, 14) 的 int64 数组（取值在 int8 范围内）。
        """
        q = self.quantize_input(obs)
        for layer in self.layers:
            acc = q @ layer.weight.astype(np.int64).T + layer.bias.astype(np.int64)
            acc = acc + (-layer.in_zero) * layer.weight.astype(np.int64).sum(axis=1)
            if acc.min() < INT32_MIN or acc.max() > INT32_MAX:
                raise OverflowError("int32 累加溢出 —— 单片机上会静默回绕")
            out = multiply_by_quantized_mult(acc, layer.mult.astype(np.int64),
                                             layer.shift.astype(np.int64))
            q = np.clip(out + layer.out_zero, QMIN, QMAX)
            if layer.lut is not None:
                q = layer.lut.astype(np.int64)[q - QMIN]
        return q

    def forward(self, obs: np.ndarray) -> np.ndarray:
        """int8 前向，输出反量化成浮点动作（与 ONNX 的 `actions` 同义）。

        Args:
            obs: 形状 (N, 61) 的原始观测。

        Returns:
            形状 (N, 14) 的 float32 动作。
        """
        last = self.layers[-1]
        return (np.float32(last.out_scale) * (self.forward_q(obs) - last.out_zero)).astype(np.float32)


def float_forward(weights, biases, x):
    """浮点前向，顺带返回每层的激活前输出（标定用）。

    Args:
        weights: 各层权重 [out, in]。
        biases: 各层偏置。
        x: 已归一化的输入 (N, in)。

    Returns:
        (最终输出, 各层激活前输出的列表)。
    """
    pre = []
    for i, (w, b) in enumerate(zip(weights, biases, strict=True)):
        x = x @ w.T + b
        pre.append(x)
        if i < len(weights) - 1:
            x = elu(x)
    return x, pre


def quantize_actor(weights, biases, obs_mean, obs_var, normalizer_eps, calib_obs) -> Int8Policy:
    """把浮点 actor 量化成 `Int8Policy`。

    Args:
        weights: 各层权重 [out, in]（float）。
        biases: 各层偏置。
        obs_mean: 观测归一化的均值。
        obs_var: 观测归一化的方差。
        normalizer_eps: 归一化分母里的 eps。
        calib_obs: 标定用的原始观测 (N, 61)，应来自策略在仿真里真实走路时的分布。

    Returns:
        量化好的策略。
    """
    mean = np.asarray(obs_mean, dtype=np.float32)
    inv_std = (np.float32(1.0) / (np.sqrt(np.asarray(obs_var, dtype=np.float32))
                                  + np.float32(normalizer_eps))).astype(np.float32)
    x = (np.asarray(calib_obs, dtype=np.float32) - mean) * inv_std
    _, pre = float_forward(weights, biases, x.astype(np.float64))

    lo, hi = np.percentile(x, CALIB_PERCENTILES)
    in_scale, in_zero = activation_params(lo, hi)
    layers: list[Int8Layer] = []
    for i, (w, b) in enumerate(zip(weights, biases, strict=True)):
        w = np.asarray(w, dtype=np.float64)
        out_scale, out_zero = activation_params(*np.percentile(pre[i], CALIB_PERCENTILES))
        w_scale = np.maximum(np.abs(w).max(axis=1), 1e-12) / QMAX
        q_w = np.clip(np.round(w / w_scale[:, None]), -QMAX, QMAX).astype(np.int8)
        padded_in = -(-w.shape[1] // ROW_ALIGN) * ROW_ALIGN
        weight = np.zeros((w.shape[0], padded_in), dtype=np.int8)
        weight[:, : w.shape[1]] = q_w
        bias = np.round(np.asarray(b, dtype=np.float64) / (in_scale * w_scale)).astype(np.int64)
        if bias.min() < INT32_MIN or bias.max() > INT32_MAX:
            raise OverflowError(f"第 {i} 层偏置超出 int32")
        mults, shifts = zip(*(quantize_multiplier(in_scale * s / out_scale) for s in w_scale),
                            strict=True)
        if max(shifts) > 0:
            # 正移位在 C 里是 `x * (1 << shift)`，int32 会溢出；按本网络的量级不该出现。
            raise ValueError(f"第 {i} 层出现正移位 {max(shifts)}，重量化乘数 ≥ 0.5")
        lut = None
        next_scale, next_zero = in_scale, in_zero
        if i < len(weights) - 1:
            grid = out_scale * (np.arange(QMIN, QMAX + 1) - out_zero)
            act = elu(grid)
            next_scale, next_zero = activation_params(act.min(), act.max())
            lut = np.clip(np.floor(act / next_scale + next_zero + 0.5), QMIN, QMAX).astype(np.int8)
        layers.append(Int8Layer(weight=weight, bias=bias.astype(np.int32),
                                mult=np.asarray(mults, dtype=np.int32),
                                shift=np.asarray(shifts, dtype=np.int32),
                                in_scale=in_scale, in_zero=in_zero,
                                out_scale=out_scale, out_zero=out_zero, lut=lut))
        in_scale, in_zero = next_scale, next_zero
    return Int8Policy(obs_mean=mean, obs_inv_std=inv_std,
                      in_inv_scale=np.float32(1.0 / layers[0].in_scale), layers=layers)


def save_npz(policy: Int8Policy, path) -> None:
    """把 int8 策略存成 npz，Python 侧直接读回、不必重新量化。

    Args:
        policy: int8 策略。
        path: 输出路径。
    """
    arrays = {"obs_mean": policy.obs_mean, "obs_inv_std": policy.obs_inv_std,
              "in_inv_scale": np.float32(policy.in_inv_scale), "num_layers": len(policy.layers)}
    for i, layer in enumerate(policy.layers):
        for field in ("weight", "bias", "mult", "shift"):
            arrays[f"l{i}_{field}"] = getattr(layer, field)
        arrays[f"l{i}_scales"] = np.array([layer.in_scale, layer.out_scale])
        arrays[f"l{i}_zeros"] = np.array([layer.in_zero, layer.out_zero])
        if layer.lut is not None:
            arrays[f"l{i}_lut"] = layer.lut
    np.savez(path, **arrays)


def load_npz(path) -> Int8Policy:
    """读回 `save_npz` 存的 int8 策略。

    Args:
        path: npz 路径。

    Returns:
        int8 策略。
    """
    data = np.load(path)
    num_layers = int(data["num_layers"])
    layers = []
    for i in range(num_layers):
        in_scale, out_scale = (float(v) for v in data[f"l{i}_scales"])
        in_zero, out_zero = (int(v) for v in data[f"l{i}_zeros"])
        layers.append(Int8Layer(
            weight=data[f"l{i}_weight"], bias=data[f"l{i}_bias"], mult=data[f"l{i}_mult"],
            shift=data[f"l{i}_shift"], in_scale=in_scale, in_zero=in_zero,
            out_scale=out_scale, out_zero=out_zero,
            # 隐藏层必有 ELU 查表、最后一层必没有：缺了就是文件坏了，当场 KeyError。
            lut=data[f"l{i}_lut"] if i < num_layers - 1 else None))
    return Int8Policy(obs_mean=data["obs_mean"], obs_inv_std=data["obs_inv_std"],
                      in_inv_scale=np.float32(data["in_inv_scale"]), layers=layers)


def _c_float(value: float) -> str:
    """float32 常量的 C 写法。

    9 位有效数字足以让 float32 原样往返；整数值要补小数点 ——
    `50f` 在 C 里是非法后缀，`50.0f` 才是浮点常量。
    """
    text = f"{float(value):.9g}"
    if not any(ch in text for ch in ".e"):
        text += ".0"
    return text + "f"


def _c_array(ctype: str, name: str, values, per_line: int = 32, aligned: bool = False) -> str:
    flat = np.asarray(values).reshape(-1)
    if np.issubdtype(flat.dtype, np.floating):
        items = [_c_float(v) for v in flat]
    else:
        items = [str(int(v)) for v in flat]
    rows = [", ".join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
    attr = " __attribute__((aligned(16)))" if aligned else ""
    return (f"static const {ctype} {name}[{len(items)}]{attr} = {{\n    "
            + ",\n    ".join(rows) + "\n};\n")


def write_c_header(policy: Int8Policy, path, test_obs: np.ndarray, provenance: str,
                   name: str) -> None:
    """生成给 C 用的头文件：全部量化参数 + 测试向量，打包成一个 `policy_int8_t <name>_policy`。

    推理代码是各动作共用的 `firmware/lib/policy/src/`，每个动作的头文件只带数据。
    测试向量是一批原始观测与它们在整数参考前向下的 int8 输出 —— 单片机上
    （以及主机上的 `training/hostcheck/`）跑同一批输入，必须逐位相等。

    Args:
        policy: int8 策略。
        path: 输出路径。
        test_obs: 测试向量的原始观测 (K, 61)。
        provenance: 写进文件头注释的来源说明（哪份权重、哪次训练）。
        name: C 里的名字前缀（数组与策略实例都以它开头），须是合法的 C 标识符。

    Raises:
        ValueError: 名字不是 C 标识符，或层数、层宽超出 C 库的静态上限。
    """
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"{name!r} 不是合法的 C 标识符（动作名里的连字符要换成下划线）")
    if len(policy.layers) > C_MAX_LAYERS:
        raise ValueError(f"{len(policy.layers)} 层超过 C 库的上限 {C_MAX_LAYERS}")
    widest = max(max(layer.weight.shape) for layer in policy.layers)
    if widest > C_MAX_WIDTH:
        raise ValueError(f"最宽一层 {widest} 超过 C 库的中间缓冲 {C_MAX_WIDTH}")
    expected = policy.forward_q(test_obs)
    parts = [
        "// 由 training/export_policy.py 生成，不要手改。\n",
        "// " + provenance.replace("\n", "\n// ") + "\n",
        "// 推理代码见 firmware/lib/policy/src/。浮点部分须以 -ffp-contract=off 编译。\n",
        "#pragma once\n#include <stdint.h>\n\n#include \"policy_int8.h\"\n\n",
        _c_array("float", f"{name}_obs_mean", policy.obs_mean),
        _c_array("float", f"{name}_obs_inv_std", policy.obs_inv_std),
    ]
    dims = []
    for i, layer in enumerate(policy.layers):
        out_dim, in_dim = layer.weight.shape
        dims.append(f"    {{{in_dim}, {out_dim}, {-layer.in_zero}, {layer.out_zero}}},")
        parts += [
            _c_array("int8_t", f"{name}_l{i}_weight", layer.weight, aligned=True),
            _c_array("int32_t", f"{name}_l{i}_bias", layer.bias),
            _c_array("int32_t", f"{name}_l{i}_mult", layer.mult),
            _c_array("int32_t", f"{name}_l{i}_shift", layer.shift),
        ]
        if layer.lut is not None:
            parts.append(_c_array("int8_t", f"{name}_l{i}_lut", layer.lut))
    layer_ids = range(len(policy.layers))

    def table(ctype: str, field: str, table_name: str) -> str:
        items = ", ".join(f"{name}_l{i}_{field}" for i in layer_ids)
        return (f"static const {ctype} *const {name}_{table_name}[{len(policy.layers)}] = "
                f"{{{items}}};\n")

    luts = ", ".join(f"{name}_l{i}_lut" if policy.layers[i].lut is not None else "0"
                     for i in layer_ids)
    last = policy.layers[-1]
    parts += [
        "// 每层：{补齐后的输入长度, 输出通道数, input_offset, output_offset}\n",
        f"static const int32_t {name}_layer_dims[{len(policy.layers)}][4] = {{\n",
        "\n".join(dims), "\n};\n",
        table("int8_t", "weight", "weights"), table("int32_t", "bias", "biases"),
        table("int32_t", "mult", "mults"), table("int32_t", "shift", "shifts"),
        f"static const int8_t *const {name}_luts[{len(policy.layers)}] = {{{luts}}};\n\n",
        _c_array("float", f"{name}_test_obs", test_obs.astype(np.float32), per_line=8),
        _c_array("int8_t", f"{name}_test_out_q", expected),
        f"\nstatic const policy_int8_t {name}_policy = {{\n",
        f"    .name = \"{name}\",\n",
        f"    .obs_dim = {policy.obs_dim},\n",
        f"    .act_dim = {last.weight.shape[0]},\n",
        f"    .num_layers = {len(policy.layers)},\n",
        f"    .layer_dims = {name}_layer_dims,\n",
        f"    .weights = {name}_weights,\n",
        f"    .bias = {name}_biases,\n",
        f"    .mult = {name}_mults,\n",
        f"    .shift = {name}_shifts,\n",
        f"    .lut = {name}_luts,\n",
        f"    .obs_mean = {name}_obs_mean,\n",
        f"    .obs_inv_std = {name}_obs_inv_std,\n",
        f"    .in_inv_scale = {_c_float(float(policy.in_inv_scale))},\n",
        f"    .in_zero_plus_half = {policy.layers[0].in_zero + 0.5:.1f}f,\n",
        f"    .out_scale = {_c_float(last.out_scale)},\n",
        f"    .out_zero = {last.out_zero},\n",
        f"    .num_tests = {test_obs.shape[0]},\n",
        f"    .test_obs = {name}_test_obs,\n",
        f"    .test_out_q = {name}_test_out_q,\n",
        "};\n",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(parts))
