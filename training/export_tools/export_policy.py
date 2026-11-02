"""把一份训好的权重导出成真机用的文件，并证明导出没有把策略弄坏。

产出都在 `policies/<硬件版本>/<动作>/`（进 git；硬件版本取权重训练时记的那一版，与 `sim_model/hardware/` 一一对应）。
下表以走路为例；`<名>` 是动作名把连字符换成下划线
（velstand-flat → velstand_flat），用作文件名与 C 里的前缀：

| 文件 | 给谁 |
|---|---|
| `walk.onnx` | 上游运行时 / 任何 ONNX 推理：归一化烘进图里，输入 `obs` [1,61]，输出 `actions` [1,14]，opset 18，附元数据 —— 与上游 `scripts/export.py` 的产物同一个约定 |
| `walk_int8.h` | ESP32-S3 固件：int8 权重、逐通道重量化参数、层间 ELU 查表、测试向量，打包成 `walk_policy`（推理代码共用 `firmware/lib/policy/`） |
| `walk_int8.npz` | Python：同一份 int8 参数，读回即可评测，不必重新量化 |
| `MANIFEST.json` | 来源（哪次训练、哪份权重、哪份物理模型）、各文件摘要、三种形态的评测对照 |

证明分三层，任何一层不过都退非零、不写清单：
1. ONNX 与 PyTorch 在标定观测上逐项相差不超过 1e-5；
2. C 推理代码 + esp-nn 内核在主机上跑测试向量，与 Python 整数参考逐位相同；
3. 浮点、ONNX、int8 三种形态在仿真里各闭环跑一个完整回合，评测摘要并排写进清单
   （通用口径；各动作专门的成功率测法在实验目录，不在这里）。

跑法（训练机上，仓根目录）：
    MINIDUCK_SKILL=<动作> MINIDUCK_RUN_NAME=<run> uv run python training/export_tools/export_policy.py [model_6000.pt]
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

from env import DuckEnv, Skill, skill_from_env
from hardware import Hardware
from export_tools.int8 import load_npz, quantize_actor, save_npz, write_c_header
from logs import run_cli
from model import NORMALIZER_EPS, DeployedActor
from paths import REPO_ROOT, checkpoint_dir, run_name
from rollout import (
    collect_observations,
    evaluate,
    latest_checkpoint,
    load_checkpoint,
    trained_hardware,
)

NUM_TEST_VECTORS = 32
ONNX_TOLERANCE = 1e-5
# onnxruntime 会话日志级别：2 = WARNING。默认会把图优化的细节打到屏幕上。
ORT_LOG_WARNING = 2

log = logging.getLogger(__name__)


def ort_session(path: Path) -> ort.InferenceSession:
    """CPU 上的 onnxruntime 会话，只在出问题时说话。

    Args:
        path: ONNX 路径。

    Returns:
        推理会话。
    """
    options = ort.SessionOptions()
    options.log_severity_level = ORT_LOG_WARNING
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def sha256(path: Path) -> str:
    """文件内容的 sha256。

    Args:
        path: 文件路径。

    Returns:
        十六进制摘要。
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_onnx(actor: DeployedActor, skill: Skill, hardware: Hardware, path: Path,
                extra_metadata: dict) -> dict:
    """导出 ONNX 并附上元数据。

    导出器参数与 mjlab 的 `export_policy_to_onnx` 相同（旧版导出路径、opset 18、固定 batch 1）；
    元数据用 mjlab 的 `get_base_metadata` 现取（关节名、默认姿态、动作缩放、观测项名等），
    再加上本仓的来源字段。

    Args:
        actor: 归一化 + actor 网络（CPU 上）。
        skill: 动作声明（元数据从这个任务的环境里取）。
        hardware: 硬件版本（与权重训练时的一致）。
        path: 输出路径。
        extra_metadata: 追加的元数据。

    Returns:
        写进文件的全部元数据。
    """
    actor.eval()
    torch.onnx.export(actor, (torch.zeros(1, actor.normalizer.mean.shape[0]),), str(path),
                      export_params=True, opset_version=18, input_names=["obs"],
                      output_names=["actions"], dynamic_axes={}, dynamo=False)
    env = DuckEnv(skill, hardware, num_envs=1)
    try:
        metadata = {**get_base_metadata(env.unwrapped, run_path=extra_metadata["run"]),
                    **extra_metadata}
    finally:
        env.close()
    attach_metadata_to_onnx(str(path), metadata)
    onnx.checker.check_model(onnx.load(str(path)))
    return metadata


def onnx_policy(path: Path):
    """把 ONNX 包成「观测 → 动作」函数。模型是固定 batch 1，逐个环境跑。

    Args:
        path: ONNX 路径。

    Returns:
        接收 (N, 61) 张量、返回同设备 (N, 14) 张量的函数。
    """
    session = ort_session(path)

    def run(obs: torch.Tensor) -> torch.Tensor:
        rows = obs.detach().cpu().numpy().astype(np.float32)
        out = np.concatenate([session.run(["actions"], {"obs": r[None]})[0] for r in rows])
        return torch.from_numpy(out).to(obs.device)

    return run


def int8_policy(path: Path):
    """把 int8 参数包成「观测 → 动作」函数（numpy 整数参考前向）。

    Args:
        path: npz 路径。

    Returns:
        接收 (N, 61) 张量、返回同设备 (N, 14) 张量的函数。
    """
    policy = load_npz(path)

    def run(obs: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(policy.forward(obs.detach().cpu().numpy())).to(obs.device)

    return run


def pick_test_vectors(obs: np.ndarray, count: int) -> np.ndarray:
    """从标定观测里等间隔取一批当测试向量，再补两条极端输入（全零、超范围）。

    Args:
        obs: 标定观测。
        count: 从里面取多少条。

    Returns:
        测试向量 (count + 2, 61)。
    """
    picked = obs[np.linspace(0, len(obs) - 1, count).astype(int)]
    extremes = np.stack([np.zeros(obs.shape[1]), np.full(obs.shape[1], 50.0)])
    return np.concatenate([picked, extremes]).astype(np.float32)


def main():
    """导出、校验、写清单。

    Raises:
        RuntimeError: ONNX 与 PyTorch 不一致，或主机上的 C 自检没过（此时不写清单）。
    """
    skill = skill_from_env()
    c_name = skill.name.replace("-", "_")
    name = run_name()
    ckpt_path = checkpoint_dir(name) / sys.argv[1] if len(sys.argv) > 1 else latest_checkpoint(name)
    model, ckpt = load_checkpoint(ckpt_path, "cpu")
    settings = ckpt["training_settings"]
    hw = trained_hardware(ckpt)
    out_dir = hw.policies_dir / skill.name
    counter = int(ckpt["common_step_counter"])
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path, header_path, npz_path = (out_dir / f"{c_name}.onnx", out_dir / f"{c_name}_int8.h",
                                        out_dir / f"{c_name}_int8.npz")
    provenance = {
        "skill": skill.name, "task": skill.task_id, "hardware": hw.id, "servo": hw.servo.key,
        "run": name, "checkpoint": ckpt_path.name, "iteration": int(ckpt["iteration"]),
        "robot_xml_sha256": settings["robot_xml_sha256"],
        "mass_table_sha256": settings["mass_table_sha256"],
        # 缺这一项（None）= 在电池盒不碰撞的模型上训的（env.OUR_COLLISION_GEOMS 之前）
        "collision_geom_renames": settings.get("collision_geom_renames"),
        "hardware_spec": settings.get("hardware_spec"),
        "servo_spec": settings.get("servo_spec", {"supply": settings.get("servo_supply")}),
        "upstream_commit": settings["upstream_source"]["vcs_info"]["commit_id"],
        # 从已发布策略续训的才有这一项（哪个 ONNX、它在源训练里的第几轮）。
        "init_from_policy": settings.get("init_from_policy"),
    }

    log.info("导出 %s / %s（第 %d 轮）到 policies/%s/%s/", name, ckpt_path.name, ckpt["iteration"],
             hw.id, skill.name)

    # ① ONNX
    actor = DeployedActor(model).cpu()
    metadata = export_onnx(actor, skill, hw, onnx_path,
                           {k: str(v) for k, v in provenance.items()})

    # 标定观测：GPU 上的浮点策略在仿真里真实跑这个动作时遇到的分布。
    gpu_model, _ = load_checkpoint(ckpt_path, "cuda:0")
    calib = collect_observations(gpu_model.act_inference, skill, hw, counter)
    with torch.no_grad():
        torch_out = actor(torch.from_numpy(calib[:4096])).numpy()
    session = ort_session(onnx_path)
    onnx_out = np.concatenate([session.run(["actions"], {"obs": r[None]})[0] for r in calib[:4096]])
    onnx_err = float(np.abs(onnx_out - torch_out).max())
    if onnx_err > ONNX_TOLERANCE:
        raise RuntimeError(f"ONNX 与 PyTorch 不一致：最大差 {onnx_err:.2e}（门槛 {ONNX_TOLERANCE:.0e}），"
                           f"{onnx_path.name} 不可用")
    log.info("写好 %s：与 PyTorch 在 %d 条标定观测上最大差 %.1e", onnx_path.name, len(onnx_out), onnx_err)

    # ② int8
    linears = [m for m in model.actor if isinstance(m, torch.nn.Linear)]
    policy = quantize_actor([m.weight.detach().numpy() for m in linears],
                            [m.bias.detach().numpy() for m in linears],
                            model.actor_normalizer.mean.numpy(), model.actor_normalizer.var.numpy(),
                            NORMALIZER_EPS, calib)
    save_npz(policy, npz_path)
    write_c_header(policy, header_path, pick_test_vectors(calib, NUM_TEST_VECTORS),
                   json.dumps(provenance, ensure_ascii=False), c_name)
    int8_out = policy.forward(calib[:4096])
    int8_err = np.abs(int8_out - torch_out)
    log.info("写好 %s 与 %s：开环误差均值 %.4f rad，99 分位 %.4f rad",
             header_path.name, npz_path.name, int8_err.mean(), np.percentile(int8_err, 99))
    hostcheck = subprocess.run(["bash", str(REPO_ROOT / "training/export_tools/hostcheck/run.sh"), str(out_dir),
                                c_name], capture_output=True, text=True, check=False)
    if hostcheck.returncode != 0:
        raise RuntimeError(f"主机上的 C 自检没过，{header_path.name} 不可用：\n"
                           f"{hostcheck.stdout}\n{hostcheck.stderr}")
    log.info("主机 C 自检通过：%s", hostcheck.stdout.strip())

    # ③ 三种形态各闭环跑一个回合
    evals = {
        "fp32_torch": evaluate(gpu_model.act_inference, skill, hw, counter),
        "onnx": evaluate(onnx_policy(onnx_path), skill, hw, counter),
        "int8": evaluate(int8_policy(npz_path), skill, hw, counter),
    }
    for form, result in evals.items():
        log.info("闭环 %s：%d 个环境 × %d 步，被判失败 %d 次，每步平均奖励 %.3f",
                 form, result["num_envs"], result["steps"], result["falls"],
                 result["mean_step_reward"])

    manifest = {
        **provenance,
        "files": {p.name: sha256(p) for p in (onnx_path, header_path, npz_path)},
        "onnx": {"opset": 18, "input": "obs [1,61]", "output": "actions [1,14]",
                 "max_abs_diff_vs_torch": onnx_err,
                 "metadata": {k: metadata[k] for k in ("joint_names", "observation_names",
                                                       "command_names", "action_scale",
                                                       "default_joint_pos")}},
        "int8": {
            "kernel": "esp-nn esp_nn_fully_connected_per_ch_s8 @ d116020",
            "weight_bytes": int(sum(layer.weight.size for layer in policy.layers)),
            "open_loop_abs_err_vs_fp32": {"mean": float(int8_err.mean()),
                                          "p99": float(np.percentile(int8_err, 99)),
                                          "max": float(int8_err.max())},
            "hostcheck": hostcheck.stdout.strip(),
            "test_vectors": NUM_TEST_VECTORS + 2,
        },
        "closed_loop_eval": evals,
    }
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    log.info("清单写到 MANIFEST.json，导出完成")


if __name__ == "__main__":
    run_cli(main)
