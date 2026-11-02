"""Lightweight native MuJoCo viewer for MiniDuck policies (ONNX or the on-board .mdp file).

Windows PowerShell examples, from the repository root:
    uv run python training/run_policy_viewer.py --skill velstand-flat
    uv run python training/run_policy_viewer.py --skill velstand-flat --start prone
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import logging
import os
import queue
import select
import sys
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
from bam.model import load_model
from bam.mujoco import MujocoController

import hardware
from hardware import Hardware
from export_tools.int8 import policy_from_mdp
from logs import run_cli
from paths import POLICY_DIR, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "firmware/tools"))
from assets import POLICY_HEADER_BYTES, read_policy_blob  # noqa: E402

DEFAULT_HARDWARE = "2026-10-24-hd1910"
CONTROL_HZ = 50
DECIMATION = 4
PHYSICS_DT = 1.0 / (CONTROL_HZ * DECIMATION)
OBS_DIM = 61
ACTION_DIM = 14
SKILL_SCENES = {
    "walk": "scene_walk.xml",
    "standup": "scene.xml",
    "sitstand": "scene.xml",
    "velstand-flat": "scene_allcollisions.xml",
    "velstand-rough-backlash": "scene_backlash.xml",
}
ROBOT_XML_BY_SKILL = {
    "walk": "robot_walk.xml",
    "standup": "robot_groundcontact.xml",
    "sitstand": "robot_groundcontact.xml",
    "velstand-flat": "robot_allcollisions.xml",
    "velstand-rough-backlash": "robot_allcollisions_backlash.xml",
}
LOCOMOTION_SKILLS = {"walk", "velstand-flat", "velstand-rough-backlash"}
STIFF_SOLREF = (-5.0e4, -2.0e2)
STIFF_SOLIMP = (0.99, 0.9999, 0.001, 0.5, 2.0)
HELD_KEY_VK = {
    "w": 0x57, "s": 0x53, "a": 0x41, "d": 0x44,
    "j": 0x4A, "l": 0x4C, "space": 0x20,
    "q": 0x51, "r": 0x52,
}
POSIX_INITIAL_HOLD_SECONDS = 0.65
POSIX_REPEAT_HOLD_SECONDS = 0.18
POSIX_MOTION_KEYS = {"w", "s", "a", "d", "j", "l", "space"}

log = logging.getLogger(__name__)


class TerminalKeys:
    """Read held keys on Windows and approximate them from POSIX key repeat."""

    def __init__(self):
        self.keys: queue.Queue[str] = queue.Queue()
        self.stopping = threading.Event()
        self._lock = threading.Lock()
        self._held: set[str] = set()
        self._held_until: dict[str, float] = {}
        self._terminal_fd: int | None = None
        self._terminal_settings = None
        self._reader_thread: threading.Thread | None = None
        self.platform = "windows" if os.name == "nt" else "posix"
        self.enabled = self.platform == "windows" or (
            self.platform == "posix" and sys.stdin.isatty()
        )

    def __enter__(self):
        if not self.enabled:
            log.warning("stdin 不是交互式终端，终端键盘控制不可用")
            return self
        if self.platform == "posix":
            import termios
            import tty

            self._terminal_fd = sys.stdin.fileno()
            self._terminal_settings = termios.tcgetattr(self._terminal_fd)
            tty.setcbreak(self._terminal_fd)
            target = self._read_posix
            log.info("POSIX 终端按键使用自动重复近似按住状态；松开后最多约 %.2f 秒停止",
                     POSIX_INITIAL_HOLD_SECONDS)
        else:
            target = self._read_windows
        self._reader_thread = threading.Thread(target=target, daemon=True)
        self._reader_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        self.stopping.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=0.2)
        if self._terminal_fd is not None and self._terminal_settings is not None:
            import termios

            termios.tcsetattr(self._terminal_fd, termios.TCSADRAIN, self._terminal_settings)

    def _read_windows(self):
        get_async_key_state = ctypes.windll.user32.GetAsyncKeyState
        get_async_key_state.argtypes = [ctypes.c_int]
        get_async_key_state.restype = ctypes.c_short
        previous: set[str] = set()
        while not self.stopping.is_set():
            held = {
                name for name, virtual_key in HELD_KEY_VK.items()
                if get_async_key_state(virtual_key) & 0x8000
            }
            with self._lock:
                self._held = held
            for key in ("r", "q"):
                if key in held and key not in previous:
                    self.keys.put(key)
            previous = held
            time.sleep(0.01)

    def _read_posix(self):
        import os as posix_os

        assert self._terminal_fd is not None
        while not self.stopping.is_set():
            ready, _, _ = select.select([self._terminal_fd], [], [], 0.03)
            if not ready:
                continue
            for value in posix_os.read(self._terminal_fd, 32):
                key = chr(value).lower()
                if key in POSIX_MOTION_KEYS or key in {"q", "r"}:
                    self._record_posix_key(key, time.monotonic())

    def _record_posix_key(self, key: str, now: float):
        with self._lock:
            was_held = self._held_until.get(key, 0.0) > now
            timeout = POSIX_REPEAT_HOLD_SECONDS if was_held else POSIX_INITIAL_HOLD_SECONDS
            self._held_until[key] = now + timeout
        if key in {"q", "r"} and not was_held:
            self.keys.put(key)

    def held(self) -> set[str]:
        with self._lock:
            if self.platform == "posix":
                now = time.monotonic()
                self._held_until = {
                    key: deadline for key, deadline in self._held_until.items()
                    if deadline > now
                }
                return set(self._held_until)
            return set(self._held)


class Policy:
    """CPU ONNX policy with metadata emitted by training/export_tools/export_policy.py."""

    def __init__(self, path: Path):
        options = ort.SessionOptions()
        options.log_severity_level = 2
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "obs" or inputs[0].shape[-1] != OBS_DIM:
            raise RuntimeError(f"ONNX 输入必须是 obs[1,{OBS_DIM}]，实际为 {[(x.name, x.shape) for x in inputs]}")
        if len(outputs) != 1 or outputs[0].name != "actions" or outputs[0].shape[-1] != ACTION_DIM:
            raise RuntimeError(f"ONNX 输出必须是 actions[1,{ACTION_DIM}]，实际为 {[(x.name, x.shape) for x in outputs]}")

        self.metadata = self.session.get_modelmeta().custom_metadata_map
        self.skill = self.metadata.get("skill", "")
        self.hardware = self.metadata.get("hardware", "")
        self.joint_names = self.metadata.get("joint_names", "").split(",")
        self.action_scale = float(self.metadata.get("action_scale", "1.0"))
        self.default_pose = np.fromstring(
            self.metadata.get("default_joint_pos", ""), sep=",", dtype=np.float64
        )
        manifest_path = path.parent / "MANIFEST.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            metadata = manifest.get("onnx", {}).get("metadata", {})
            if metadata.get("default_joint_pos"):
                self.default_pose = np.asarray(metadata["default_joint_pos"], dtype=np.float64)
        if len(self.joint_names) != ACTION_DIM or self.default_pose.shape != (ACTION_DIM,):
            raise RuntimeError("ONNX 缺少有效的 14 维关节顺序/default_joint_pos 元数据")

    def infer(self, observation: np.ndarray) -> np.ndarray:
        batch = observation.astype(np.float32, copy=False)[None, :]
        return self.session.run(["actions"], {"obs": batch})[0][0]


class MdpPolicy:
    """传给机器人的那份 .mdp 在 CPU 上跑 int8 前向；关节顺序与默认姿态读同目录的 MANIFEST.json。"""

    def __init__(self, path: Path):
        blob = path.read_bytes()
        info = read_policy_blob(blob)
        self.int8, _test_obs, _test_out_q = policy_from_mdp(info, blob[POLICY_HEADER_BYTES:])
        manifest = json.loads((path.parent / "MANIFEST.json").read_text(encoding="utf-8"))
        onnx_metadata = manifest["onnx"]["metadata"]
        self.metadata = {"robot_xml_sha256": manifest["robot_xml_sha256"]}
        self.skill = info["slot"]
        self.hardware = info["hardware"]
        self.joint_names = onnx_metadata["joint_names"]
        self.action_scale = float(onnx_metadata["action_scale"])
        self.default_pose = np.asarray(onnx_metadata["default_joint_pos"], dtype=np.float64)

    def infer(self, observation: np.ndarray) -> np.ndarray:
        return self.int8.forward(observation[None, :])[0]


def load_policy(path: Path) -> Policy | MdpPolicy:
    return MdpPolicy(path) if path.suffix == ".mdp" else Policy(path)


def resolve_policy_path(skill: str, hw: Hardware, requested: Path | None) -> Path:
    if requested is None:
        # 刚导出的 ONNX 优先；只有出厂 .mdp 时用 .mdp。
        policy_dir = POLICY_DIR / hw.id / skill
        path = policy_dir / f"{skill.replace('-', '_')}.onnx"
        if not path.is_file():
            path = policy_dir / f"{skill}.mdp"
    else:
        path = requested.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到策略文件：{path}")
    return path


def load_bam_model(hw: Hardware):
    actuator = hw.servo.actuator
    parameter_file = actuator.get("params_json")
    if parameter_file:
        bam_model = load_model(json_file=str(hw.servo.dir / parameter_file))
    else:
        bam_model = load_model(
            motor_name=actuator["bundled_motor"], model=actuator["bundled_model"]
        )
    bam_model.actuator.kp = float(actuator["kp_fw"])
    bam_model.actuator.vin = sum(hw.supply["vin_range_v"]) / 2.0
    bam_model.actuator.max_current = actuator["max_current_a"]
    return bam_model


def build_simulation(hw: Hardware, skill: str, policy: Policy):
    """Build the repository MJCF and replace position actuators with BAM motors."""
    spec = mujoco.MjSpec.from_file(str(hw.model_dir / SKILL_SCENES[skill]))
    actuator_by_joint: dict[str, str] = {}
    for actuator in spec.actuators:
        target = actuator.target
        joint_name = target.name if hasattr(target, "name") else str(target)
        if joint_name.startswith("passive_"):
            continue
        actuator.set_to_motor()
        actuator.forcelimited = True
        actuator.ctrllimited = False
        actuator.gear = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        actuator_by_joint[joint_name] = actuator.name
        for joint in spec.joints:
            if joint.name == joint_name:
                joint.damping = np.zeros((3, 1))
                joint.frictionloss = 0.0
                joint.solref_friction = STIFF_SOLREF
                joint.solimp_friction = STIFF_SOLIMP
                break

    if set(policy.joint_names) != set(actuator_by_joint):
        raise RuntimeError(
            f"策略与模型关节不匹配：policy={policy.joint_names}, model={sorted(actuator_by_joint)}"
        )
    model = spec.compile()
    model.opt.timestep = PHYSICS_DT
    data = mujoco.MjData(model)
    bam_model = load_bam_model(hw)
    force_limit = max(hw.supply["vin_range_v"]) * bam_model.kt.value / bam_model.R.value
    actuator_ids = [model.actuator(actuator_by_joint[name]).id for name in policy.joint_names]
    model.actuator_forcelimited[actuator_ids] = 1
    model.actuator_forcerange[actuator_ids] = (-force_limit, force_limit)

    voltage_drop = hw.supply["load_sag_v_per_nm"]
    if voltage_drop is not None:
        voltage_drop = sum(voltage_drop) / 2.0
    controller = MujocoController(
        bam_model,
        [actuator_by_joint[name] for name in policy.joint_names],
        model,
        data,
        vin_drop_gain=voltage_drop,
        vin_min=hw.supply["sag_floor_v"],
    )
    if hw.servo.vendor == "feetech":
        bam_model.actuator.q_target_smooth = np.zeros(ACTION_DIM, dtype=np.float64)

    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                 for name in policy.joint_names]
    imu_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    if min(*joint_ids, imu_id, trunk_id, freejoint_id) < 0:
        raise RuntimeError("MJCF 缺少策略关节、IMU 或躯干自由关节")
    return (model, data, bam_model, controller, model.jnt_qposadr[joint_ids],
            model.jnt_dofadr[joint_ids], model.sensor_adr[imu_id], trunk_id,
            model.jnt_qposadr[freejoint_id])


class Observer:
    def __init__(self, model, data, bam_model, controller, policy: Policy,
                 standing_policy: Policy | None, qpos_indices, qvel_indices,
                 imu_adr: int, trunk_id: int, free_qpos_adr: int,
                 command: np.ndarray, threshold: float, start: str):
        self.model, self.data = model, data
        self.bam_model, self.controller = bam_model, controller
        self.policy, self.standing_policy = policy, standing_policy
        self.qpos_indices, self.qvel_indices = qpos_indices, qvel_indices
        self.imu_adr, self.trunk_id = imu_adr, trunk_id
        self.free_qpos_adr = free_qpos_adr
        self.command, self.threshold, self.start = command, threshold, start
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        root = self.free_qpos_adr
        if self.start == "prone":
            self.data.qpos[root:root + 7] = [0.0, 0.0, 0.09, 0.0, 1.0, 0.0, 0.0]
        else:
            self.data.qpos[root:root + 7] = [0.0, 0.0, 0.125, 1.0, 0.0, 0.0, 0.0]
        self.data.qpos[self.qpos_indices] = self.policy.default_pose
        self.controller.reset(self.data.qpos)
        self.controller.last_ts = self.data.time
        self.controller.q_target[:] = self.policy.default_pose
        if self.policy.hardware.endswith("hd1910"):
            self.bam_model.actuator.q_target_smooth = self.policy.default_pose.copy()
        self.last_action.fill(0.0)
        mujoco.mj_forward(self.model, self.data)

    def active_policy(self) -> Policy:
        if self.standing_policy and np.linalg.norm(self.command) <= self.threshold:
            return self.standing_policy
        return self.policy

    def observation(self) -> np.ndarray:
        rotation = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, self.data.xquat[self.trunk_id])
        gravity = rotation.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        angular_velocity = self.data.sensordata[self.imu_adr:self.imu_adr + 3]
        relative_position = self.data.qpos[self.qpos_indices] - self.policy.default_pose
        joint_velocity = self.data.qvel[self.qvel_indices]
        command = np.zeros(13, dtype=np.float32)
        if self.active_policy() is self.policy:
            command[:3] = self.command
        observation = np.concatenate((angular_velocity, gravity, relative_position,
                                      joint_velocity, self.last_action, command)).astype(np.float32)
        if observation.shape != (OBS_DIM,):
            raise RuntimeError(f"策略观测维度 {observation.shape}，期望 {(OBS_DIM,)}")
        return observation

    def step(self):
        active = self.active_policy()
        observation = self.observation()
        action = active.infer(observation)
        self.controller.q_target[:] = self.policy.default_pose + action * self.policy.action_scale
        self.last_action[:] = action
        for _ in range(DECIMATION):
            self.controller.update()
            mujoco.mj_step(self.model, self.data)
        return observation, action, active


def set_command_from_held_keys(command: np.ndarray, held: set[str]):
    if "space" in held:
        command.fill(0.0)
        return
    command[0] = 0.3 * (int("w" in held) - int("s" in held))
    command[2] = 1.5 * (int("a" in held) - int("d" in held))
    command[1] = 0.2 * (int("j" in held) - int("l" in held))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", choices=tuple(SKILL_SCENES), default="velstand-flat")
    parser.add_argument("--hardware", choices=hardware.known(), default=DEFAULT_HARDWARE)
    parser.add_argument("--policy", type=Path, help="ONNX 或 .mdp 路径；默认按硬件和技能查找")
    parser.add_argument("--standing", type=Path, help="可选 standup 策略（ONNX 或 .mdp），低速时自动切换")
    parser.add_argument("--stand-threshold", type=float, default=0.05)
    parser.add_argument("--vx", type=float, default=None)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--start", choices=("default", "prone"), default="default")
    parser.add_argument("--save-csv", type=Path)
    args = parser.parse_args()
    if args.standing and args.skill not in LOCOMOTION_SKILLS:
        parser.error("--standing 只能用于行走技能")

    hw = hardware.load(args.hardware)
    policy_path = resolve_policy_path(args.skill, hw, args.policy)
    policy = load_policy(policy_path)
    if policy.skill and policy.skill != args.skill:
        parser.error(f"策略技能是 {policy.skill}，当前选择 {args.skill}")
    if policy.hardware and policy.hardware != hw.id:
        parser.error(f"策略硬件是 {policy.hardware}，当前选择 {hw.id}")
    expected_xml_hash = policy.metadata.get("robot_xml_sha256")
    if expected_xml_hash:
        xml_path = hw.model_dir / ROBOT_XML_BY_SKILL[args.skill]
        actual_xml_hash = hashlib.sha256(xml_path.read_bytes()).hexdigest()
        if actual_xml_hash != expected_xml_hash:
            log.warning("当前 MJCF 与策略训练时记录的文件不同：%s", xml_path.name)
    standing_policy = load_policy(args.standing.expanduser().resolve()) if args.standing else None
    if standing_policy:
        if standing_policy.skill and standing_policy.skill != "standup":
            parser.error("--standing 文件不是 standup 策略")
        if standing_policy.hardware and standing_policy.hardware != hw.id:
            parser.error("standup 策略硬件与 --hardware 不匹配")
        if (standing_policy.joint_names != policy.joint_names
                or not np.allclose(standing_policy.default_pose, policy.default_pose, atol=1e-3)):
            parser.error("两份策略的关节顺序或默认姿态不一致")

    model, data, bam_model, controller, qpos_indices, qvel_indices, imu_adr, trunk_id, free_adr = (
        build_simulation(hw, args.skill, policy)
    )
    vx = args.vx if args.vx is not None else 0.0
    command = np.asarray([vx, args.vy, args.wz], dtype=np.float32)
    observer = Observer(model, data, bam_model, controller, policy, standing_policy,
                        qpos_indices, qvel_indices, imu_adr, trunk_id, free_adr,
                        command, args.stand_threshold, args.start)
    log.info("直接 MuJoCo + BAM；CPU 推理 %s；硬件 %s；控制 %d Hz，%d 子步",
             policy_path.name, hw.id, CONTROL_HZ, DECIMATION)

    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_stream = args.save_csv.open("w", newline="", encoding="utf-8") if args.save_csv else None
    csv_writer = None
    if csv_stream:
        csv_writer = csv.writer(csv_stream)
        csv_writer.writerow(["step", "policy", "vx", "vy", "wz"]
                            + [f"obs_{i:02d}" for i in range(OBS_DIM)]
                            + [f"action_{i:02d}" for i in range(ACTION_DIM)])
    try:
        with TerminalKeys() as keys, mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = trunk_id
            viewer.cam.distance = 0.9
            viewer.cam.elevation = -15.0
            viewer.cam.azimuth = 150.0
            viewer.cam.lookat[:] = data.xpos[trunk_id]
            viewer.sync()
            step_count = 0
            window_start = time.perf_counter()
            window_steps = 0
            log.info("按住 W/S 前后、A/D 转向、J/L 横移；松开即停，空格紧急停止，R 重置，Q 退出")
            keyboard_active = False
            while viewer.is_running():
                tick = time.perf_counter()
                quit_requested = False
                while True:
                    try:
                        key = keys.keys.get_nowait()
                    except queue.Empty:
                        break
                    if key == "q":
                        quit_requested = True
                        break
                    if key == "r":
                        observer.reset()
                if quit_requested:
                    break
                held = keys.held()
                if held.intersection({"w", "s", "a", "d", "j", "l", "space"}):
                    keyboard_active = True
                if keyboard_active:
                    set_command_from_held_keys(command, held)

                with viewer.lock():
                    observation, applied_action, active_policy = observer.step()
                viewer.sync()
                step_count += 1
                window_steps += 1
                if csv_writer:
                    csv_writer.writerow([step_count, active_policy.skill, *command,
                                         *observation.tolist(), *applied_action.tolist()])
                remaining = 1.0 / CONTROL_HZ - (time.perf_counter() - tick)
                if remaining > 0:
                    time.sleep(remaining)
                now = time.perf_counter()
                if now - window_start >= 2.0:
                    log.info("步数 %d；实际频率 %.1f Hz；指令 [%.2f, %.2f, %.2f]",
                             step_count, window_steps / (now - window_start), *command)
                    window_start, window_steps = now, 0
    finally:
        if csv_stream:
            csv_stream.close()
            log.info("动作记录写到 %s", args.save_csv)


if __name__ == "__main__":
    run_cli(main)
