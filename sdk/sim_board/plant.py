"""Single MiniDuck BAM plant backed by the viewer's MuJoCo model and INT8 policies."""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
from training.export.viewer import (
    ACTION_DIM,
    DECIMATION,
    OBS_DIM,
    SKILL_SCENES,
    MdpPolicy,
    Observer,
    build_simulation,
    hardware,
    policy_from_mdp,
    read_policy_blob,
)

WALK = "velstand-flat"
STAND = "standup"
SIT = "sitstand"
OBSTACLE_GROUP = 5
TOF_MAX_M = 2.0
TOF_NEAR_M = 0.06
WALL_HALF_SIZE = (0.018, 0.30, 0.155)
WALL_COLOR = (0.65, 0.75, 0.84, 1.0)
DEFAULT_WALL_X = 0.60


class Plant(Observer):
    """Use the viewer's pose, joint order, observation and servo model in a board scene."""

    def __init__(self, policy_root: Path, hardware_id: str, wall_x: float, slots: tuple[str, ...]):
        self.hw = hardware.load(hardware_id)
        self.paths = {name: policy_root / name / f"{name}.mdp" for name in slots}
        self.actors = {name: MdpPolicy(path) for name, path in self.paths.items()}
        base = self.actors[WALK]
        for name, actor in self.actors.items():
            if actor.hardware != self.hw.id or actor.joint_names != base.joint_names:
                raise ValueError(f"{name} does not match the simulated robot's hardware or joint order")
        spec = mujoco.MjSpec.from_file(str(self.hw.model_dir / SKILL_SCENES[WALK]))
        spec.worldbody.add_geom(
            name="tof_wall", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(wall_x + WALL_HALF_SIZE[0], 0, WALL_HALF_SIZE[2]), size=WALL_HALF_SIZE,
            rgba=WALL_COLOR, group=OBSTACLE_GROUP,
        )
        spec.texture("groundplane").rgb1[:] = (0.93, 0.95, 0.97)
        spec.texture("groundplane").rgb2[:] = (0.80, 0.85, 0.90)
        spec.texture("groundplane").markrgb[:] = (0.74, 0.80, 0.86)
        spec.material("groundplane").reflectance = 0
        for texture in spec.textures:
            if texture.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                texture.rgb1[:] = texture.rgb2[:] = (1, 1, 1)
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 720
        model, data, motor, controller, *indexes = build_simulation(self.hw, WALK, base, spec=spec)
        super().__init__(model, data, motor, controller, base, None, *indexes,
                         np.zeros(3, dtype=np.float32), 0.05, "default")
        self.tof_site = data.site("tof")
        self.wall_mask = np.zeros(6, dtype=np.uint8)
        self.wall_mask[OBSTACLE_GROUP] = 1
        self.ray_geom = np.empty(1, dtype=np.int32)
        self.zero_command = np.zeros(13, dtype=np.float32)
        self.obs_command = self.zero_command.copy()
        self.active_name = WALK
        self.last_observation = self.observation()

    def ray_distance(self) -> float:
        """Cast from the face along the ToF site's forward axis, in metres."""
        direction = self.tof_site.xmat.reshape(3, 3)[:, 0].copy()
        return mujoco.mj_ray(self.model, self.data, self.tof_site.xpos, direction,
                             self.wall_mask, True, -1, self.ray_geom)

    def range_mm(self) -> int:
        distance = self.ray_distance()
        return round(1000 * distance) if TOF_NEAR_M <= distance <= TOF_MAX_M else -1

    def advance(self, name: str | None, command: np.ndarray, phase: float | None, scale: float) -> None:
        """Advance a 20 ms control tick using the selected registered actor."""
        self.obs_command[:] = command
        if phase is not None:
            self.obs_command[:2] = math.cos(phase), math.sin(phase)
        if name is None:
            self.data.ctrl.fill(0.0)
        else:
            # Firmware shares accumulated action state when switching only between stand and walk.
            if self.active_name != name and {self.active_name, name} != {STAND, WALK}:
                self.last_action.fill(0.0)
            self.active_name = name
            self.policy = self.actors[name]
            obs = self.observation()
            obs[-len(self.obs_command):] = self.obs_command
            action = self.policy.infer(obs)
            self.controller.q_target[:] = self.policy.default_pose + action * scale
            self.last_action[:] = action
            self.last_observation = obs
        for _ in range(DECIMATION):
            if name is not None:
                self.controller.update()
            mujoco.mj_step(self.model, self.data)

    def reset_pose(self) -> None:
        """Reinitialize with the walking scene's pose, regardless of the last skill."""
        self.active_name = WALK
        self.policy = self.actors[WALK]
        self.obs_command.fill(0)
        self.reset()
        self.last_observation = self.observation()

    def switch_on(self) -> None:
        """Resume servo control without resetting pose or simulation time."""
        self.controller.reset(self.data.qpos)
        self.controller.last_ts = self.data.time
        self.controller.q_target[:] = self.data.qpos[self.qpos_indices]
        self.bam_model.actuator.q_target_smooth = self.controller.q_target.copy()
        self.last_action.fill(0.0)

    def load_uploaded(self, name: str, blob: bytes) -> None:
        """Load real INT8 weights against this board's fixed home pose and joint order."""
        info = read_policy_blob(blob)
        if (info["hardware"], info["slot"], info["obs_dim"], info["act_dim"]) != (
            self.hw.id, name, OBS_DIM, ACTION_DIM,
        ):
            raise ValueError("策略与仿真板的硬件、名称或维度不符")
        actor = object.__new__(MdpPolicy)
        actor.int8, inputs, outputs = policy_from_mdp(info, blob[128:])
        if not len(inputs) or not np.array_equal(actor.int8.forward_q(inputs), outputs):
            raise ValueError("INT8 策略自检不通过")
        actor.hardware, actor.skill = self.hw.id, name
        actor.default_pose = self.actors[WALK].default_pose
        actor.joint_names = self.actors[WALK].joint_names
        actor.action_scale = 1.0
        self.actors[name] = actor
