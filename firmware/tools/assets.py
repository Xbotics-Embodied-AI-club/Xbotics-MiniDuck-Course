"""MiniDuck 板上文件：把策略、叫声转成固件认的格式与文件名，经热点传到板子上（与网页「文件」页 83 口同一个接口）

    uv run firmware/tools/assets.py policy policies/2026-10-24-hd1910/*                # 导出的策略 → 同目录 <模型>.mdp
    uv run firmware/tools/assets.py audio 我的叫声.mp3 --as greet_3 [--adpcm]             # WAV/MP3/FLAC/OGG → greet_3.wav（当前目录）
    uv run firmware/tools/assets.py upload velstand-flat.mdp greet_3.wav --reboot         # 先连机器人的热点
    uv run firmware/tools/assets.py ls | rm /audio/greet_3.wav | restore | reboot | check 文件 ...
    uv run firmware/tools/assets.py record -o 本次上电.mdr [--session 3]                  # 下载上电记录（84 口）

板上的文件：
    /factory/policy/<硬件版本>/<模型>.mdp   出厂策略，烧写固件包时写入，网页删改不了
    /factory/audio/<叫法>_<变体号>.wav       出厂叫声
    /policy/<模型>.mdp                       上传的策略：开机优先用，载不进来（坏了、硬件版本不对、自检不过）就退回出厂那份
    /audio/<叫法>_<变体号>.wav               上传的叫声：同名替换出厂那段，新变体号是加一段
    /log/r<序号>.mdr                         上电记录
模型 standup / velstand-flat / sitstand；叫法见 AUDIO_LABELS，变体号 0–9（播放时在已有的变体里随机挑）。
「恢复出厂」= 删掉 /policy 与 /audio 下的全部文件、重启。

格式按片上容量定（实测数在固件 README）：策略是 int8 二进制（一份 214 KB，比 ONNX 小 3.7 倍）；叫声是 16 kHz 单声道
WAV，16 位 PCM，或 --adpcm 的 4 位 IMA ADPCM（再小 4 倍，信噪比约 26 dB）。上传传文件原样，板子校验通过才替换。

写 flash 时板子两个核一起停：舵机正被驱动时（闭环、站起来、回中检查、点动）板子拒收，先在遥控页（80 口）点「卸力」，
传完重启生效。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import time
import wave
import zlib
from pathlib import Path

from protocol import (
    DEFAULT_IP,
    FS_DELETE_PATH,
    FS_LIST_PATH,
    FS_REBOOT_PATH,
    FS_RESTORE_PATH,
    FS_UPLOAD_NAME_HEADER,
    FS_UPLOAD_PATH,
    POLICY_HEADER_FMT,
    RECORD_PATH,
    WEB_PORTS,
)

FIRMWARE = Path(__file__).resolve().parent.parent
POLICIES = FIRMWARE.parent / "policies"

# 策略文件（与 lib/policy/src/policy_int8_blob.h 一致）
POLICY_MAGIC = b"MDPL"
POLICY_VERSION = 1
POLICY_HEADER_BYTES = 128
POLICY_SLOTS = ("standup", "velstand-flat", "sitstand")   # 固件的三个模型，文件名 = 模型名
POLICY_EXT = ".mdp"

# 板上布局（与 lib/assets/src/asset_layout.h 一致）
FACTORY_DIR = "/factory"
POLICY_DIR = "/policy"
AUDIO_DIR = "/audio"
LOG_DIR = "/log"
AUDIO_EXT = ".wav"
LOG_EXT = ".mdr"
AUDIO_RATE = 16000
AUDIO_LABELS = ("alarm", "greet", "inquire", "peck", "chirp", "coo", "wheee_start", "wheee_loop", "wheee_end")
AUDIO_MAX_VARIANTS = 10
AUDIO_NAME_RE = re.compile(r"^(" + "|".join(AUDIO_LABELS) + r")_(\d)$")
NAME_MAX = 32                                          # 文件名最长（含结尾 NUL）

# WAV（与 lib/voice/src/playback/wav_file.h、ima_adpcm.h 一致）
WAV_FORMAT_PCM = 0x0001
WAV_FORMAT_IMA_ADPCM = 0x0011
ADPCM_BLOCK_ALIGN = 256                                # 一块 505 个采样
ADPCM_MAX_BLOCK_ALIGN = 1024
IMA_STEP = (7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66, 73, 80, 88,
            97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658,
            724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327, 3660,
            4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899, 15289, 16818,
            18500, 20350, 22385, 24623, 27086, 29794, 32767)
IMA_INDEX_ADJUST = (-1, -1, -1, -1, 2, 4, 6, 8)

# 导出的 _int8.h 里的数组与结构体字段
_ARRAY_RE = re.compile(r"static const (int8_t|int32_t|float) (\w+)\[(\d+)\](?:\[(\d+)\])?"
                       r"(?: __attribute__\(\(aligned\(16\)\)\))? = \{(.*?)\};", re.S)
_FIELD_RE = re.compile(r"^\s*\.(\w+) = ([^,\n]+),", re.M)


def _align16(n: int) -> int:
    return (n + 15) & ~15


# ---------------------------------------------------------------------------------------------------------
# 策略：导出的 _int8.h → .mdp
# ---------------------------------------------------------------------------------------------------------

def parse_policy_header(text: str) -> dict:
    """读 training/export_policy.py 生成的 _int8.h：各数组的数值与 policy_int8_t 的标量字段。

    Returns:
        {"name", "obs_dim", "act_dim", "num_layers", "num_tests", "in_inv_scale", "in_zero_plus_half",
         "out_scale", "out_zero", "arrays": {后缀: (C 类型, 数值列表)}}，数组按去掉 "<name>_" 前缀的后缀存。
    """
    fields = {k: v.strip() for k, v in _FIELD_RE.findall(text)}
    name = fields["name"].strip('"')
    arrays = {}
    for ctype, sym, n0, n1, body in _ARRAY_RE.findall(text):
        if not sym.startswith(name + "_"):
            raise ValueError(f"数组 {sym} 不是策略 {name} 的")
        cast = float if ctype == "float" else int
        flat = body.replace("{", " ").replace("}", " ")   # 二维数组（layer_dims）按行加了花括号
        values = [cast(v.strip().rstrip("f")) for v in flat.split(",") if v.strip()]
        expect = int(n0) * (int(n1) if n1 else 1)
        if len(values) != expect:
            raise ValueError(f"数组 {sym} 声明 {expect} 个数，实际 {len(values)} 个")
        arrays[sym[len(name) + 1:]] = (ctype, values)
    out = {"name": name, "arrays": arrays}
    for key in ("obs_dim", "act_dim", "num_layers", "num_tests", "out_zero"):
        out[key] = int(fields[key])
    for key in ("in_inv_scale", "in_zero_plus_half", "out_scale"):
        out[key] = float(fields[key].rstrip("f"))
    return out


def build_policy_blob(parsed: dict, hardware_id: str, slot: str, source_sha256: bytes) -> bytes:
    """按 policy_int8_blob.h 的格式拼出一份策略文件。"""
    a = parsed["arrays"]
    layers = parsed["num_layers"]
    obs, act, tests = parsed["obs_dim"], parsed["act_dim"], parsed["num_tests"]
    dims = a["layer_dims"][1]
    payload = bytearray()

    def put(code: str, values) -> None:
        payload.extend(struct.pack("<" + code * len(values), *values))
        payload.extend(b"\0" * (_align16(len(payload)) - len(payload)))

    put("i", dims)
    put("f", a["obs_mean"][1])
    put("f", a["obs_inv_std"][1])
    lut_mask = 0
    for layer in range(layers):
        in_len, channels = dims[layer * 4], dims[layer * 4 + 1]
        weight = a[f"l{layer}_weight"][1]
        if len(weight) != in_len * channels:
            raise ValueError(f"第 {layer} 层权重 {len(weight)} 个，layer_dims 说是 {in_len}×{channels}")
        put("b", weight)
        for part in ("bias", "mult", "shift"):
            put("i", a[f"l{layer}_{part}"][1])
        if f"l{layer}_lut" in a:
            put("b", a[f"l{layer}_lut"][1])
            lut_mask |= 1 << layer
    put("f", a["test_obs"][1])
    put("b", a["test_out_q"][1])
    if len(a["test_obs"][1]) != tests * obs or len(a["test_out_q"][1]) != tests * act:
        raise ValueError("测试向量的个数与 num_tests 对不上")
    header = struct.pack(POLICY_HEADER_FMT, POLICY_MAGIC, POLICY_VERSION, POLICY_HEADER_BYTES, len(payload),
                         zlib.crc32(payload), hardware_id.encode(), slot.encode(), parsed["name"].encode(),
                         obs, act, layers, tests, parsed["in_inv_scale"], parsed["in_zero_plus_half"],
                         parsed["out_scale"], parsed["out_zero"], lut_mask, source_sha256[:20])
    assert len(header) == POLICY_HEADER_BYTES
    return header + bytes(payload)


def read_policy_blob(blob: bytes) -> dict:
    """读一份策略文件的头并核对数据区校验（板上解析的同一套规则的电脑端版本，check 与测试用）。"""
    if len(blob) < POLICY_HEADER_BYTES:
        raise ValueError("比文件头还短")
    (magic, version, header_bytes, payload_len, crc, hw, slot, name, obs, act, layers, tests,
     in_inv_scale, in_zero_plus_half, out_scale, out_zero, lut_mask, src) = struct.unpack_from(POLICY_HEADER_FMT, blob)
    if magic != POLICY_MAGIC:
        raise ValueError("不是策略文件（没有 MDPL 标记）")
    if version != POLICY_VERSION or header_bytes != POLICY_HEADER_BYTES:
        raise ValueError(f"格式版本 {version} 不认识（本工具是 {POLICY_VERSION}）")
    payload = blob[POLICY_HEADER_BYTES:POLICY_HEADER_BYTES + payload_len]
    if len(payload) != payload_len or payload_len % 16:
        raise ValueError("数据区长度不对（文件被截断？）")
    if zlib.crc32(payload) != crc:
        raise ValueError("数据区校验不对（文件坏了）")
    return {"hardware": hw.rstrip(b"\0").decode(), "slot": slot.rstrip(b"\0").decode(),
            "name": name.rstrip(b"\0").decode(), "obs_dim": obs, "act_dim": act, "num_layers": layers,
            "num_tests": tests, "in_inv_scale": in_inv_scale, "in_zero_plus_half": in_zero_plus_half,
            "out_scale": out_scale, "out_zero": out_zero, "lut_mask": lut_mask, "source_sha256": src.hex(),
            "payload_bytes": payload_len}


def _policy_source(policy_dir: Path) -> Path:
    headers = sorted(policy_dir.glob("*_int8.h"))
    if len(headers) != 1:
        raise SystemExit(f"{policy_dir}：要正好一个 *_int8.h（training/export_policy.py 的产物），找到 {len(headers)} 个")
    return headers[0]


def convert_policy(policy_dir: Path) -> Path:
    """policies/<硬件版本>/<模型>/ 里导出的 _int8.h → 同目录 <模型>.mdp。硬件版本取目录名，并与 MANIFEST 核对。"""
    policy_dir = policy_dir.resolve()
    slot, hardware_id = policy_dir.name, policy_dir.parent.name
    if slot not in POLICY_SLOTS:
        raise SystemExit(f"{policy_dir}：固件只有 {'、'.join(POLICY_SLOTS)} 三个模型，{slot} 不进固件")
    manifest = json.loads((policy_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("skill") != slot:
        raise SystemExit(f"{policy_dir}：MANIFEST 里的动作是 {manifest.get('skill')}，目录名是 {slot}")
    if "hardware" in manifest and manifest["hardware"] != hardware_id:
        raise SystemExit(f"{policy_dir}：MANIFEST 里的硬件版本是 {manifest['hardware']}，目录名是 {hardware_id}")
    source = _policy_source(policy_dir).read_bytes()
    blob = build_policy_blob(parse_policy_header(source.decode("utf-8")), hardware_id, slot,
                             hashlib.sha256(source).digest())
    read_policy_blob(blob)   # 写之前按读的规则核一遍
    out = policy_dir / f"{slot}{POLICY_EXT}"
    out.write_bytes(blob)
    return out


def policy_file_is_current(mdp: Path) -> bool:
    """这份 .mdp 是不是由同目录现在的 _int8.h 转出来的（头里记着源文件 SHA-256 的前 20 字节）。"""
    source = _policy_source(mdp.parent).read_bytes()
    return read_policy_blob(mdp.read_bytes())["source_sha256"] == hashlib.sha256(source).digest()[:20].hex()


# ---------------------------------------------------------------------------------------------------------
# 叫声：任意音频 → 16 kHz 单声道 WAV（16 位 PCM，或 4 位 IMA ADPCM）
# ---------------------------------------------------------------------------------------------------------

def check_audio_name(stem: str) -> None:
    """板上只认 <叫法>_<变体号>，叫法见 AUDIO_LABELS。"""
    if not AUDIO_NAME_RE.match(stem):
        raise SystemExit(f"叫声名 {stem!r} 不合规：要 <叫法>_<变体号 0-9>，叫法 {'/'.join(AUDIO_LABELS)}")


def parse_wav(data: bytes) -> dict:
    """解析 RIFF/WAVE 头（与固件 wav_parse 同一套规则：按块找 fmt、fact、data，块按偶数对齐）。"""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("不是 RIFF/WAVE 文件")
    info, fact, pos = None, 0, 12
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], int.from_bytes(data[pos + 4:pos + 8], "little")
        body = pos + 8
        if size > len(data) - body:
            raise ValueError("文件被截断")
        if cid == b"fmt ":
            f = data[body:body + size]
            info = {"format": int.from_bytes(f[0:2], "little"), "channels": int.from_bytes(f[2:4], "little"),
                    "rate": int.from_bytes(f[4:8], "little"), "block_align": int.from_bytes(f[12:14], "little"),
                    "bits": int.from_bytes(f[14:16], "little"),
                    "samples_per_block": int.from_bytes(f[18:20], "little") if size >= 20 else 0}
        elif cid == b"fact" and size >= 4:
            fact = int.from_bytes(data[body:body + 4], "little")
        elif cid == b"data":
            if info is None:
                raise ValueError("data 之前没有 fmt 块")
            info["data_offset"], info["data_bytes"] = body, size
            if info["format"] == WAV_FORMAT_IMA_ADPCM:
                blocks = -(-size // info["block_align"])
                upper = blocks * ((info["block_align"] - 4) * 2 + 1)
                info["samples"] = fact if 0 < fact <= upper else upper
            else:
                info["samples"] = size // max(1, info["bits"] // 8 * info["channels"])
            return info
        pos = body + size + (size & 1)
    raise ValueError("没有 data 块")


def board_audio_problem(info: dict) -> str | None:
    """不是板上能播的格式时返回原因（与固件 wav_check_board_format 一致），能播返回 None。"""
    if info["rate"] != AUDIO_RATE or info["channels"] != 1:
        return f"要 16 kHz 单声道（这份是 {info['rate']} Hz / {info['channels']} 声道）"
    if info["format"] == WAV_FORMAT_PCM:
        return None if info["bits"] == 16 else f"PCM 要 16 位（这份是 {info['bits']} 位）"
    if info["format"] == WAV_FORMAT_IMA_ADPCM:
        ba = info["block_align"]
        if info["bits"] != 4 or not 36 <= ba <= ADPCM_MAX_BLOCK_ALIGN or info["samples_per_block"] != (ba - 4) * 2 + 1:
            return "IMA ADPCM 的块参数不对"
        return None
    return f"格式码 {info['format']:#06x} 不能播（只认 16 位 PCM 与 4 位 IMA ADPCM）"


def _riff(chunks: list[tuple[bytes, bytes]]) -> bytes:
    body = b"WAVE"
    for cid, data in chunks:
        body += cid + len(data).to_bytes(4, "little") + data + (b"\0" if len(data) & 1 else b"")
    return b"RIFF" + len(body).to_bytes(4, "little") + body


def _u16(v: int) -> bytes:
    return v.to_bytes(2, "little")


def _u32(v: int) -> bytes:
    return v.to_bytes(4, "little")


def pcm16_wav(samples: bytes) -> bytes:
    """16 kHz 单声道 16 位 PCM 的 WAV（samples 是小端 int16）。"""
    fmt = _u16(WAV_FORMAT_PCM) + _u16(1) + _u32(AUDIO_RATE) + _u32(AUDIO_RATE * 2) + _u16(2) + _u16(16)
    return _riff([(b"fmt ", fmt), (b"data", samples)])


def ima_adpcm_encode(pcm: list[int], block_align: int = ADPCM_BLOCK_ALIGN) -> bytes:
    """单声道 IMA ADPCM 编码（微软块格式；重建公式与固件 ima_adpcm.c 逐位相同，编码端跟着解码端的预测走）。"""
    per_block = (block_align - 4) * 2 + 1
    out = bytearray()
    for start in range(0, len(pcm), per_block):
        block = pcm[start:start + per_block]
        pred, index = block[0], 0
        if len(block) > 1:   # 初始步长按开头的跳变挑，免得块头几个采样追不上
            while index < 88 and IMA_STEP[index] < abs(block[1] - block[0]):
                index += 1
        out += (pred & 0xFFFF).to_bytes(2, "little") + bytes((index, 0))
        codes = []
        for s in block[1:]:
            step, diff, code = IMA_STEP[index], s - pred, 0
            if diff < 0:
                code, diff = 8, -diff
            if diff >= step:
                code |= 4
                diff -= step
            if diff >= step >> 1:
                code |= 2
                diff -= step >> 1
            if diff >= step >> 2:
                code |= 1
            pred, index = _ima_step(pred, index, code)
            codes.append(code)
        if len(codes) % 2:
            codes.append(0)
        out += bytes(codes[i] | (codes[i + 1] << 4) for i in range(0, len(codes), 2))
    return bytes(out)


def _ima_step(pred: int, index: int, code: int) -> tuple[int, int]:
    step = IMA_STEP[index]
    diff = step >> 3
    if code & 4:
        diff += step
    if code & 2:
        diff += step >> 1
    if code & 1:
        diff += step >> 2
    pred = max(-32768, min(32767, pred - diff if code & 8 else pred + diff))
    return pred, max(0, min(88, index + IMA_INDEX_ADJUST[code & 7]))


def ima_adpcm_decode(data: bytes, block_align: int, samples: int) -> list[int]:
    """IMA ADPCM 解码（与固件 ima_adpcm_decode_block 同一算法；check 与测试用）。"""
    out = []
    for off in range(0, len(data), block_align):
        block = data[off:off + block_align]
        pred, index = int.from_bytes(block[0:2], "little", signed=True), block[2]
        out.append(pred)
        for byte in block[4:]:
            for code in (byte & 0x0F, byte >> 4):
                pred, index = _ima_step(pred, index, code)
                out.append(pred)
    return out[:samples]


def adpcm_wav(pcm: list[int], block_align: int = ADPCM_BLOCK_ALIGN) -> bytes:
    """16 kHz 单声道 4 位 IMA ADPCM 的 WAV（fmt 带 samples_per_block 扩展，fact 记总采样数）。"""
    per_block = (block_align - 4) * 2 + 1
    fmt = (_u16(WAV_FORMAT_IMA_ADPCM) + _u16(1) + _u32(AUDIO_RATE) + _u32(AUDIO_RATE * block_align // per_block)
           + _u16(block_align) + _u16(4) + _u16(2) + _u16(per_block))
    return _riff([(b"fmt ", fmt), (b"fact", _u32(len(pcm))), (b"data", ima_adpcm_encode(pcm, block_align))])


def load_audio_as_pcm16(src: Path) -> list[int]:
    """WAV / MP3 / FLAC / OGG → 16 kHz 单声道 int16 采样（已经是板上 16 位格式的直接读，否则 soxr 高质量重采样、混单声道）。"""
    raw = src.read_bytes()
    try:
        info = parse_wav(raw)
        if info["format"] == WAV_FORMAT_PCM and board_audio_problem(info) is None:
            data = raw[info["data_offset"]:info["data_offset"] + info["data_bytes"]]
            return list(struct.unpack(f"<{len(data) // 2}h", data))
    except ValueError:
        pass   # 不是 WAV（MP3、FLAC…）或格式不同，走下面的通用路径
    import numpy as np
    import soundfile
    import soxr

    try:
        data, rate = soundfile.read(str(src), dtype="float32", always_2d=True)
    except soundfile.LibsndfileError as e:
        raise SystemExit(f"{src} 读不了（{e}）：能转的是 WAV、MP3、FLAC、OGG；m4a / AAC 先用别的软件转成 WAV 或 MP3") from None
    mono = data.mean(axis=1)
    if rate != AUDIO_RATE:
        mono = soxr.resample(mono, rate, AUDIO_RATE, quality="VHQ")
    return np.clip(np.round(mono * 32767.0), -32768, 32767).astype(np.int16).tolist()


def convert_audio(src: Path, dst: Path, adpcm: bool) -> None:
    pcm = load_audio_as_pcm16(src)
    if not pcm:
        raise SystemExit(f"{src} 里没有采样")
    dst.write_bytes(adpcm_wav(pcm) if adpcm else pcm16_wav(struct.pack(f"<{len(pcm)}h", *pcm)))


# ---------------------------------------------------------------------------------------------------------
# 出厂文件（pio run -t buildfs / uploadfs 之前由 tools/pio_extra_assets.py 调）
# ---------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------
# 上传与板上文件管理（文件页 83 口；上电记录在调试页 84 口）
# ---------------------------------------------------------------------------------------------------------

def upload_name(local: Path) -> str:
    """本地文件 → 上传时的文件名（板上按扩展名分目录）；名字或格式不对就报错，不发出去。"""
    if len(local.name) >= NAME_MAX:
        raise SystemExit(f"{local.name}：文件名太长（最多 {NAME_MAX - 1} 个字符）")
    if local.suffix == POLICY_EXT:
        info = read_policy_blob(local.read_bytes())
        if local.stem != info["slot"]:
            raise SystemExit(f"{local.name}：头里的模型是 {info['slot']}，板上按文件名找，改名为 {info['slot']}{POLICY_EXT}")
        return local.name
    if local.suffix == AUDIO_EXT:
        check_audio_name(local.stem)
        problem = board_audio_problem(parse_wav(local.read_bytes()))
        if problem:
            raise SystemExit(f"{local.name}：{problem}；先 uv run firmware/tools/assets.py audio {local} --as {local.stem}")
        return local.name
    raise SystemExit(f"{local.name}：板上只收 {POLICY_EXT}（策略）和 {AUDIO_EXT}（叫声）")


def _url(ip: str, page: str, path: str) -> str:
    return f"http://{ip}:{WEB_PORTS[page]}{path}"


def _reply(r) -> dict:
    try:
        return r.json()
    except ValueError:
        return {"ok": False, "message": f"HTTP {r.status_code}: {r.text[:200]}"}


def upload(files: list[Path], ip: str) -> None:
    import requests

    for local in files:
        name = upload_name(local)
        with local.open("rb") as f:
            r = requests.post(_url(ip, "files", FS_UPLOAD_PATH), data=f, timeout=120,
                              headers={"Content-Type": "application/octet-stream", FS_UPLOAD_NAME_HEADER: name})
        reply = _reply(r)
        if r.status_code != 200 or not reply.get("ok"):
            raise SystemExit(f"{local.name} 没传上去：{reply.get('message', r.status_code)}")
        print(f"  {reply.get('path', name)}  {local.stat().st_size // 1024} KB  {reply.get('message', '')}")


def board_state(ip: str, timeout: float = 10) -> dict:
    import requests

    return requests.get(_url(ip, "files", FS_LIST_PATH), timeout=timeout).json()


def print_board(state: dict) -> None:
    print(f"文件区：用了 {state['used'] // 1024} / {state['total'] // 1024} KB；"
          f"{'可以写入' if state['writable'] else '现在不能写入：' + state['why']}")
    for f in state["files"]:
        print(f"  {f['path']:40s} {f['size'] // 1024:6d} KB")
    for slot in state.get("policies", []):
        print(f"  策略 {slot['slot']:14s} {slot['note']}")
    if "sounds" in state:
        print(f"  叫声 {state['sounds']['loaded']} 段（跳过 {state['sounds']['skipped']} 个文件）")


def post(ip: str, page: str, route: str, **data) -> dict:
    import requests

    reply = _reply(requests.post(_url(ip, page, route), params=data, timeout=15))
    print(reply.get("message", ""))
    return reply


def newest_boot_record(state: dict) -> int:
    """文件列表里最新那份上电记录的序号：每次开机新起一份，重启过它就变大。"""
    return max((f.get("record", 0) for f in state.get("files", []) if f["path"].startswith(f"{LOG_DIR}/")), default=0)


def restart_and_confirm(ip: str, route: str, down_within: float = 8.0, back_within: float = 60.0) -> bool:
    """「重启」或「恢复出厂」，再看板子是不是真的重启了。

    回复之后固件还会再查一次（存记录前后各一次），那时机器人动起来了就不重启、在文件列表里带 notice 说明；
    所以光看回复不够，重启过的证据是文件列表里多了一份新的上电记录。热点断开十几秒，电脑自己重连上热点之后才答；
    一直等不回来不算失败（板子多半在重启），提示自己再看。
    """
    import requests

    before = newest_boot_record(board_state(ip))
    if not post(ip, "files", route).get("ok"):
        return False
    start = time.monotonic()
    went_away = False
    while time.monotonic() - start < down_within + back_within:
        time.sleep(1)
        try:
            state = board_state(ip, timeout=2)
        except requests.RequestException:
            went_away = True
            continue
        if newest_boot_record(state) > before:
            print("已重启：开机载入了新文件")
            return True
        if state.get("notice"):
            print(f"板子没重启：{state['notice']}")
            return False
        if not went_away and time.monotonic() - start > down_within:
            print(f"板子没重启（{down_within:.0f} 秒内一直连得上）：到文件页（83 口）看提示")
            return False
    print("板子在重启：热点回来后重新连上再看（uv run firmware/tools/assets.py ls）")
    return True


def download_record(ip: str, session: str, out: Path) -> int:
    """下载上电记录（调试页 84 口）：current = 本次上电（文件里已存的 + 内存里还没存的），数字 = 之前某次上电。"""
    import requests

    with requests.get(_url(ip, "debug", RECORD_PATH), params={"session": session}, stream=True, timeout=60) as r:
        if r.status_code != 200:
            raise SystemExit(f"下载失败：HTTP {r.status_code} {r.text[:200]}")
        n = 0
        with out.open("wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
                n += len(chunk)
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("policy", help="导出的策略目录 → 同目录 <模型>.mdp")
    sp.add_argument("dirs", nargs="+", type=Path)
    sa = sub.add_parser("audio", help="任意音频 → <名>.wav（16 kHz 单声道；默认 16 位 PCM）")
    sa.add_argument("src", type=Path)
    sa.add_argument("--as", dest="name", required=True, help="<叫法>_<变体号>，如 greet_3")
    sa.add_argument("--adpcm", action="store_true", help="4 位 IMA ADPCM：小 4 倍，信噪比约 26 dB")
    sa.add_argument("-o", "--out-dir", type=Path, default=Path("."))
    su = sub.add_parser("upload", help="传文件到板上（.mdp → /policy，.wav → /audio）")
    su.add_argument("files", nargs="+", type=Path)
    su.add_argument("--ip", default=DEFAULT_IP)
    su.add_argument("--reboot", action="store_true", help="传完重启，开机载入新文件")
    sl = sub.add_parser("ls", help="板上的文件、开机载入了什么")
    sl.add_argument("--ip", default=DEFAULT_IP)
    sr = sub.add_parser("rm", help="删板上一个上传的文件或上电记录（出厂文件删不掉）")
    sr.add_argument("path")
    sr.add_argument("--ip", default=DEFAULT_IP)
    so = sub.add_parser("restore", help="恢复出厂：删掉全部上传的策略与叫声，重启")
    so.add_argument("--ip", default=DEFAULT_IP)
    sb = sub.add_parser("reboot", help="重启板子（新文件开机才载入）")
    sb.add_argument("--ip", default=DEFAULT_IP)
    sc = sub.add_parser("check", help="本地核对文件格式与命名")
    sc.add_argument("files", nargs="+", type=Path)
    sd = sub.add_parser("record", help="下载上电记录")
    sd.add_argument("-o", "--out", type=Path, required=True)
    sd.add_argument("--session", default="current", help="current（本次上电）或之前某次的序号（ls 里的 /log/r<序号>.mdr）")
    sd.add_argument("--ip", default=DEFAULT_IP)
    args = p.parse_args()

    if args.cmd == "policy":
        for d in args.dirs:
            if d.name not in POLICY_SLOTS:
                print(f"跳过 {d}（固件里没有这个模型）")
                continue
            out = convert_policy(d)
            info = read_policy_blob(out.read_bytes())
            print(f"{out}：{info['hardware']} / {info['slot']}，{info['num_layers']} 层，"
                  f"{info['num_tests']} 个测试向量，{out.stat().st_size // 1024} KB")
    elif args.cmd == "audio":
        check_audio_name(args.name)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        dst = args.out_dir / f"{args.name}{AUDIO_EXT}"
        convert_audio(args.src, dst, args.adpcm)
        info = parse_wav(dst.read_bytes())
        print(f"{dst}：{'IMA ADPCM 4 位' if args.adpcm else 'PCM 16 位'}，{info['samples'] / AUDIO_RATE:.2f} 秒，"
              f"{dst.stat().st_size // 1024} KB")
    elif args.cmd == "upload":
        names = [upload_name(f) for f in args.files]   # 先全部核对，有一个不对就一个都不传
        state = board_state(args.ip)
        if not state["writable"]:
            raise SystemExit(f"板子现在不能写入：{state['why']}")
        upload(args.files, args.ip)
        print(f"传了 {len(names)} 个文件" + ("" if args.reboot else "；重启后生效"))
        if args.reboot and not restart_and_confirm(args.ip, FS_REBOOT_PATH):
            sys.exit(1)
    elif args.cmd == "ls":
        print_board(board_state(args.ip))
    elif args.cmd == "rm":
        if not post(args.ip, "files", FS_DELETE_PATH, path=args.path).get("ok"):
            sys.exit(1)
    elif args.cmd == "restore":
        if not restart_and_confirm(args.ip, FS_RESTORE_PATH):
            sys.exit(1)
    elif args.cmd == "reboot":
        if not restart_and_confirm(args.ip, FS_REBOOT_PATH):
            sys.exit(1)
    elif args.cmd == "check":
        bad = 0
        for f in args.files:
            try:
                name = upload_name(f)
                if f.suffix == POLICY_EXT:
                    info = read_policy_blob(f.read_bytes())
                    print(f"{f}: ok → {POLICY_DIR}/{name}（{info['hardware']} / {info['slot']}）")
                else:
                    info = parse_wav(f.read_bytes())
                    kind = "ADPCM" if info["format"] == WAV_FORMAT_IMA_ADPCM else "PCM"
                    print(f"{f}: ok → {AUDIO_DIR}/{name}（{kind}，{info['samples'] / AUDIO_RATE:.2f} 秒）")
            except (SystemExit, ValueError, wave.Error) as e:
                print(f"{f}: {e}")
                bad += 1
        sys.exit(1 if bad else 0)
    elif args.cmd == "record":
        n = download_record(args.ip, args.session, args.out)
        print(f"{args.out}：{n // 1024} KB")


if __name__ == "__main__":
    main()
