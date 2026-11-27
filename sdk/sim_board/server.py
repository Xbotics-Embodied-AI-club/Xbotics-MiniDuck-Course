"""Local MiniDuck firmware-protocol server backed by the course's MuJoCo plant."""
from __future__ import annotations

import argparse
import asyncio
import logging
import math
import re
import struct
import time
from collections import deque
from pathlib import Path

import numpy as np
from aiohttp import WSMsgType, web
from firmware.tools.protocol import (
    CALIB_CMD,
    CALIB_POS,
    CCMD_FMT,
    CLOCK_SET,
    CMD_LIMP,
    CMD_QUERY,
    CMD_STAND,
    CPOS_FMT,
    EGET_FMT,
    ELOG_FMT,
    EVENT_FMT,
    EVENT_GET,
    EVENT_KINDS,
    EVENT_LOG,
    EVENT_MAX_RECORDS,
    FS_DELETE_PATH,
    FS_LIST_PATH,
    FS_REBOOT_PATH,
    FS_RESTORE_PATH,
    FS_UPLOAD_NAME_HEADER,
    FS_UPLOAD_PATH,
    MOTION,
    MOTION_FMT,
    PCMD_CMDS,
    PCMD_FMT,
    POLICY_CMD,
    POLICY_CORE_SLOTS,
    POLICY_JOINT_IDS,
    POLICY_MAX_TOTAL,
    POLICY_REPLY,
    POLICY_RUN_MS_MAX,
    POLICY_RUN_SCALE_MAX,
    POLICY_RUN_SCALE_MIN,
    PRSP_FMT,
    ROBOT_STATES,
    SSTA_FMT_V2,
    STATUS_FLAGS,
    STATUS_GET,
    STATUS_REPLY,
    VERSION,
    WEB_PORTS,
    WS_PORT,
)
from training.export.viewer import CONTROL_HZ, read_policy_blob
from training.paths import POLICY_DIR

from .plant import DEFAULT_WALL_X, SIT, STAND, WALK, WALL_HALF_SIZE, Plant

HOST = "127.0.0.1"
FIRMWARE = 105
HARDWARE_ID = "2026-11-26-hd1910"
STATES = {name: code for code, name in ROBOT_STATES.items()}
EVENTS = {name: code for code, name in EVENT_KINDS.items()}
SKILL_SECONDS = (4.0, 2.0, 5.0)
SKILL_FIRST = len(POLICY_CORE_SLOTS) - len(SKILL_SECONDS)
MEMORY_BYTES = 0x880000
log = logging.getLogger(__name__)


class Board:
    """One simulated board; all I/O handlers and the control loop share an event loop."""

    def __init__(self, plant: Plant, wall_x: float):
        self.plant = plant
        self.wall_x = wall_x
        self.command = plant.zero_command.copy()
        self.posture = "IDLE"
        self.active: str | None = None
        self.sit = False
        self.skill_request = 0
        self.running: tuple[str, float, float, float, bool] | None = None
        self.motion_at = self.stand_since = self.transition_until = 0.0
        self.boot_seq = 1
        self.boot_at = time.monotonic()
        self.ticks: deque[float] = deque(maxlen=CONTROL_HZ + 1)
        self.events: deque[tuple[int, ...]] = deque(maxlen=256)
        self.files: dict[str, bytes] = {}
        self.slots = dict(enumerate(POLICY_CORE_SLOTS))
        self.trace: list[tuple] = []
        self.fall_ticks = 0
        self.event("BOOT", b=FIRMWARE, d=self.boot_seq)

    @property
    def writable(self) -> bool:
        return self.posture in {"IDLE", "LIMP"}

    def reboot(self) -> None:
        self.plant.reset_pose()
        self.posture, self.active, self.running = "IDLE", None, None
        self.command.fill(0)
        self.sit = False
        self.skill_request = self.fall_ticks = 0
        self.motion_at = self.stand_since = self.transition_until = 0.0
        self.boot_seq += 1
        self.boot_at = time.monotonic()
        self.ticks.clear()
        self.events.clear()
        self.event("BOOT", b=FIRMWARE, d=self.boot_seq)

    def uptime_ms(self) -> int:
        return int(1000 * (time.monotonic() - self.boot_at))

    def event(self, kind: str, a: int = 0, b: int = 0, c: int = 0, d: int = 0) -> None:
        index = self.events[-1][0] + 1 if self.events else 1
        self.events.append((index, self.uptime_ms(), EVENTS[kind], a, b, c, d, 0))

    def standing_ready(self, now: float) -> bool:
        return self.posture == "STANDING" and now - self.stand_since >= 1.0

    def stop_policy(self, reason: int) -> None:
        if self.running is not None:
            name, _, _, _, builtin = self.running
            if not builtin:
                self.event("POLICY_STOP", a=self.slot(name), b=reason)
            self.running = None

    def slot(self, name: str | None) -> int:
        return next((i for i, current in self.slots.items() if current == name), 0)

    def motion(self, payload: tuple, now: float) -> None:
        skill, flags = payload[3:5]
        command = payload[6:]
        if not all(map(math.isfinite, command)) or self.writable:
            return
        new_skill = skill and skill != self.skill_request
        self.skill_request = skill
        self.sit = bool(flags & 1)
        self.command[:] = command
        self.motion_at = now
        if self.running and not self.running[-1] and (self.sit or any(command)):
            self.stop_policy(2)
        if (new_skill and self.running is None and SKILL_FIRST <= skill < len(POLICY_CORE_SLOTS)
                and self.standing_ready(now)):
            name = POLICY_CORE_SLOTS[skill]
            seconds = SKILL_SECONDS[skill - SKILL_FIRST]
            self.running = (name, now, seconds, self.plant.actors[name].action_scale, True)

    def calibrate(self, cmd: int, seq: int, now: float) -> bytes:
        if cmd == CMD_STAND and self.writable:
            self.plant.switch_on()
            self.posture, self.transition_until = "STANDING_UP", now + 1.2
            self.event("CONTROL_START")
        elif cmd == CMD_LIMP:
            self.stop_policy(1)
            self.command.fill(0)
            self.sit = False
            self.posture = "LIMP"
            self.event("CONTROL_HALT")
        elif cmd not in (CMD_STAND, CMD_QUERY):
            raise ValueError("仿真板不提供舵机标定")
        return struct.pack(CPOS_FMT, CALIB_POS, VERSION, seq, 3 if self.writable else 0,
                           0, 0, *([-1] * 16), *([0] * 32))

    def tick(self, now: float) -> None:
        plant = self.plant
        if self.posture == "IDLE":
            return
        if now - self.motion_at >= 0.5:
            self.command.fill(0)
        if self.running and now >= self.running[1] + self.running[2]:
            self.stop_policy(0)
        old = self.posture
        name, command, phase, scale = STAND, plant.zero_command, None, 1.0
        if old == "LIMP" or (old == "FALLEN" and now < self.transition_until):
            name = None
        elif old == "FALLEN":
            plant.switch_on()
            self.posture, self.transition_until = "GETTING_UP", now + 1.2
            self.event("GETUP_START")
        elif old in {"STANDING_UP", "GETTING_UP"} and now < self.transition_until:
            if self.active == SIT:
                name = SIT
        elif self.running:
            name, started, seconds, scale, builtin = self.running
            self.posture = "SKILL"
            if builtin and name == "ground-pick":
                phase = math.tau * min(1.0, (now - started) / seconds)
        elif self.sit:
            name = SIT
            command = plant.obs_command
            command.fill(0)
            command[0] = 1
            self.posture = "SITTING"
        elif old == "SITTING":
            name = SIT
            self.posture, self.transition_until = "STANDING_UP", now + 1.2
        elif np.linalg.norm(self.command[:3]) > 0.05:
            name, command = WALK, self.command
            self.posture = "WALKING"
        else:
            self.posture = "STANDING"
            if old != "STANDING":
                self.stand_since = now
                self.event("GETUP_DONE" if old == "GETTING_UP" else "STANDUP")
        plant.advance(name, command, phase, scale)
        self.active = name
        self.ticks.append(now)
        if self.posture in {"STANDING", "WALKING", "SKILL"} and name != "roulade":
            self.fall_ticks = self.fall_ticks + 1 if plant.last_observation[5] > -0.5 else 0
            if self.fall_ticks >= CONTROL_HZ // 2:
                self.stop_policy(3)
                self.posture, self.transition_until = "FALLEN", now + 1.0
                self.event("FALL", a=1, b=round(1000 * float(plant.last_observation[5])))
                self.fall_ticks = 0
        else:
            self.fall_ticks = 0

    def status(self, seq: int) -> bytes:
        p = self.plant
        positions, targets = [float("nan")] * 16, [float("nan")] * 16
        if self.active is not None:
            for sid, pos, tgt in zip(POLICY_JOINT_IDS, p.data.qpos[p.qpos_indices], p.controller.q_target, strict=True):
                positions[sid], targets[sid] = pos, tgt
        flag = STATUS_FLAGS["body_imu"] | STATUS_FLAGS["battery"]
        if self.writable:
            flag |= STATUS_FLAGS["writable"]
        tof = p.range_mm()
        hz10 = 0 if not self.active or len(self.ticks) < 2 else round(10 * (len(self.ticks) - 1) / (self.ticks[-1] - self.ticks[0]))
        obs = p.last_observation
        return struct.pack(
            SSTA_FMT_V2, STATUS_REPLY, VERSION, seq, self.uptime_ms(), FIRMWARE,
            STATES[self.posture], self.slot(self.active), 1, len(POLICY_JOINT_IDS), flag, 1,
            round(1000 * p.bam_model.actuator.vin), hz10, *obs[3:6], *obs[:3],
            *positions, *targets, *p.obs_command[:3], p.hw.id.encode(), 2, 2, 2, 0, 0, 0,
            self.boot_seq, tof, int(tof >= 0), *([float("nan")] * 6),
        )

    def event_page(self, seq: int, before: int) -> bytes:
        page = [r for r in reversed(self.events) if not before or r[0] < before][:EVENT_MAX_RECORDS]
        records = [struct.pack(EVENT_FMT, *r) for r in page]
        records += [bytes(struct.calcsize(EVENT_FMT))] * (EVENT_MAX_RECORDS - len(records))
        return struct.pack(ELOG_FMT, EVENT_LOG, VERSION, seq, self.uptime_ms(), self.events[-1][0],
                           len(page), 0, 0, *records)

    def policy_command(self, data: bytes, seq: int, now: float) -> bytes:
        _, _, _, cmd, idx, _, duration, scale, raw_name = struct.unpack(PCMD_FMT, data)
        name = raw_name.rstrip(b"\0").decode("ascii")
        result = 0
        if cmd == PCMD_CMDS["LOAD"]:
            if name not in self.files or not self.writable:
                result = 5
            else:
                slot = self.slot(name) or next(
                    (i for i in range(len(POLICY_CORE_SLOTS), POLICY_MAX_TOTAL) if i not in self.slots), None,
                )
                if slot is None:
                    result = 4
                else:
                    try:
                        self.plant.load_uploaded(name, self.files[name])
                    except (ValueError, OverflowError, struct.error):
                        result = 5
                    else:
                        self.slots[slot] = name
        elif cmd == PCMD_CMDS["RUN"]:
            name = name or self.slots.get(idx, "")
            if name not in self.slots.values():
                result = 1
            elif not self.standing_ready(now) or self.running is not None:
                result = 2
            elif not (1 <= duration <= POLICY_RUN_MS_MAX and POLICY_RUN_SCALE_MIN <= scale <= POLICY_RUN_SCALE_MAX):
                result = 6
            else:
                self.running = (name, now, duration / 1000, scale, False)
                self.event("POLICY_RUN", a=self.slot(name), b=round(duration / 1000), c=round(100 * scale))
        elif cmd == PCMD_CMDS["STOP"]:
            if self.running and not self.running[-1]:
                self.stop_policy(1)
        elif cmd != PCMD_CMDS["LIST"]:
            result = 6
        active = self.slot(self.running[0]) if self.running else 255
        remain = max(0, round(1000 * (self.running[1] + self.running[2] - now))) if self.running else 0
        entries = [field for i, slot_name in sorted(self.slots.items())
                   for field in (i, 2 if i == active else 1, 61, 14, 2 if i < len(POLICY_CORE_SLOTS) else 1,
                                 0, slot_name.encode())]
        entries.extend([0, 0, 0, 0, 0, 0, b""] * (POLICY_MAX_TOTAL - len(self.slots)))
        return struct.pack(PRSP_FMT, POLICY_REPLY, VERSION, seq, result, len(self.slots), active, remain, *entries)

    def packet(self, data: bytes) -> bytes | None:
        magic, version, seq = struct.unpack_from("<4sHH", data)
        if version != VERSION:
            raise ValueError("协议版本不匹配")
        now = time.monotonic()
        if magic == STATUS_GET:
            return self.status(seq)
        if magic == MOTION:
            self.motion(struct.unpack(MOTION_FMT, data), now)
        elif magic == CALIB_CMD:
            return self.calibrate(struct.unpack(CCMD_FMT, data)[3], seq, now)
        elif magic == POLICY_CMD:
            return self.policy_command(data, seq, now)
        elif magic == EVENT_GET:
            return self.event_page(seq, struct.unpack(EGET_FMT, data)[3])
        elif magic == CLOCK_SET:
            self.event("CLOCK")
        else:
            raise ValueError(f"仿真板不支持报文 {magic!r}")
        return None

    def record(self, now: float) -> None:
        p = self.plant
        tof = p.range_mm()
        origin = p.tof_site.xpos.copy()
        forward = p.tof_site.xmat.reshape(3, 3)[:, 0]
        hit = origin + forward * (tof / 1000) if tof >= 0 else np.full(3, np.nan)
        root = p.free_qpos_adr
        qpos = p.data.qpos.copy()
        self.trace.append((now, qpos, qpos[root:root + 7], qpos[p.qpos_indices],
                           p.data.qvel[:2].copy(), tof, origin, hit, self.posture, self.active or "",
                           p.obs_command[:3].copy()))

    def save_trace(self, path: Path, start: float) -> None:
        clock, qpos, root, joints, velocity, tof, origin, hit, posture, active, command = zip(*self.trace, strict=True)
        with path.open("xb") as output:
            np.savez_compressed(output, t=np.asarray(clock) - start, clock_origin=start, qpos=qpos, root=root, joints=joints,
                                joint_ids=POLICY_JOINT_IDS, velocity_xy=velocity, tof=tof, tof_origin=origin,
                                tof_hit=hit, posture=posture, policy=active, command=command,
                                wall_x=self.wall_x, wall_half_size=WALL_HALF_SIZE)


def file_routes(board: Board) -> web.Application:
    """Expose the .mdp file interface, retaining uploads in memory for this process."""
    app = web.Application(client_max_size=MEMORY_BYTES)

    async def listing(_request: web.Request) -> web.Response:
        factory = [{"path": f"/factory/policy/{board.plant.hw.id}/{p.name}",
                    "size": p.stat().st_size, "factory": True, "deletable": False}
                   for p in board.plant.paths.values()]
        uploaded = [{"path": f"/policy/{name}.mdp", "size": len(blob), "factory": False,
                     "deletable": True} for name, blob in board.files.items()]
        policies = [{"slot": name, "source": "factory" if i < len(POLICY_CORE_SLOTS) else "uploaded"}
                    for i, name in board.slots.items()]
        return web.json_response({"total": MEMORY_BYTES, "used": sum(f["size"] for f in factory + uploaded),
                                  "writable": board.writable, "files": factory + uploaded,
                                  "policies": policies, "hardware": board.plant.hw.id, "firmware": "0.10.5"})

    async def write_file(request: web.Request) -> web.Response:
        blob = await request.read() if request.path == FS_UPLOAD_PATH else b""
        if not board.writable:
            return web.json_response({"ok": False, "message": "须在待机或卸力时写文件"}, status=400)
        try:
            if request.path == FS_UPLOAD_PATH:
                filename = request.headers.get(FS_UPLOAD_NAME_HEADER, "")
                if not re.fullmatch(r"[A-Za-z0-9_-]{2,15}\.mdp", filename):
                    raise ValueError("只支持合法命名的自定义 .mdp")
                name = filename.removesuffix(".mdp")
                if name in POLICY_CORE_SLOTS:
                    raise ValueError("不能覆盖核心槽位")
                header = read_policy_blob(blob)
                if (header["hardware"], header["slot"], header["obs_dim"], header["act_dim"]) != (
                    board.plant.hw.id, name, 61, 14,
                ):
                    raise ValueError("硬件、名称或维度不匹配")
                board.files[name] = blob
                return web.json_response({"ok": True, "path": f"/policy/{filename}"})
            if request.path == FS_DELETE_PATH:
                path = request.query.get("path", "")
                name = path.removeprefix("/policy/").removesuffix(".mdp")
                if path != f"/policy/{name}.mdp" or name not in board.files:
                    raise ValueError("没有这个上传文件")
                del board.files[name]
                if name in board.slots.values():
                    del board.slots[board.slot(name)]
                board.plant.actors.pop(name, None)
            else:
                if request.path == FS_RESTORE_PATH:
                    board.files.clear()
                    board.slots = dict(enumerate(POLICY_CORE_SLOTS))
                    board.plant.actors = {n: board.plant.actors[n] for n in POLICY_CORE_SLOTS}
                board.reboot()
            return web.json_response({"ok": True})
        except (ValueError, struct.error) as exc:
            return web.json_response({"ok": False, "message": str(exc)}, status=400)

    app.router.add_get(FS_LIST_PATH, listing)
    for route in (FS_UPLOAD_PATH, FS_DELETE_PATH, FS_REBOOT_PATH, FS_RESTORE_PATH):
        app.router.add_post(route, write_file)
    return app


async def serve(args: argparse.Namespace) -> None:
    board = Board(Plant(args.policies, args.policies.name, args.wall_x, POLICY_CORE_SLOTS), args.wall_x)

    async def websocket(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=512)
        await ws.prepare(request)
        async for msg in ws:
            if msg.type != WSMsgType.BINARY:
                continue
            try:
                answer = board.packet(msg.data)
            except (struct.error, ValueError) as exc:
                await ws.close(code=1002, message=str(exc).encode()[:120])
                break
            if answer is not None:
                await ws.send_bytes(answer)
        return ws

    control = web.Application()
    control.router.add_get("/", websocket)
    runners = [web.AppRunner(control), web.AppRunner(file_routes(board))]
    start = time.monotonic()
    try:
        for runner, port in zip(runners, (args.ws_port, args.files_port), strict=True):
            await runner.setup()
            await web.TCPSite(runner, HOST, port).start()
        start = time.monotonic()
        log.info("READY ws://%s:%d HTTP %d; %s", HOST, args.ws_port, args.files_port, board.plant.hw.id)
        next_tick = start
        while args.seconds is None or next_tick - start < args.seconds:
            await asyncio.sleep(max(0, next_tick - time.monotonic()))
            now = time.monotonic()
            board.tick(now)
            if args.trace:
                board.record(now)
            next_tick += 1 / CONTROL_HZ
            if next_tick <= now:
                next_tick = now + 1 / CONTROL_HZ
    finally:
        for runner in runners:
            await runner.cleanup()
        if args.trace and board.trace:
            board.save_trace(args.trace, start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(policies=POLICY_DIR / HARDWARE_ID)
    parser.add_argument("--wall-x", type=float, default=DEFAULT_WALL_X, help="front face of obstacle, in metres")
    parser.add_argument("--ws-port", type=int, default=WS_PORT)
    parser.add_argument("--files-port", type=int, default=WEB_PORTS["files"])
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--trace", type=Path, help="write actual timed MuJoCo states for recording")
    args = parser.parse_args()
    if args.trace and (args.seconds is None or args.trace.exists()):
        parser.error("--trace requires --seconds and a new file")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
