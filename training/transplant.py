"""把关节更多的策略移植到锁死了一部分关节的硬件版本上：切掉锁死关节的那几维，其余权重原样搬过来。

用在头部锁死（`locked_joints`）的硬件版本上从原来的 14 关节走路权重起训：从零训会学成跪着走，而原来的权重已经会站着走。

观测按项（term）对齐：两边同名的项照搬；逐关节的项（关节角、关节速度、上一帧动作）按关节名挑出
留下的关节；目标里没有的项（头部指令）和锁死关节的那几维被切掉。切掉的输入在锁死后是定值
（关节停在站姿 → 相对站姿的角度 0、速度 0；上一帧动作与头部指令都是 0），它们经过归一化、乘上第一层
权重的贡献是一个常数，并进第一层的偏置 —— 移植后的网络在开局时与原策略「头停在站姿、不给头部指令」
时逐位相同。actor 最后一层与动作标准差按关节名挑出留下的关节。

本文件不碰仿真：布局（各观测项的名字与维数、关节名）由调用方从环境里取（`layout_of`）。
"""
from __future__ import annotations

import dataclasses

import torch

from model import NORMALIZER_EPS, ActorCritic

# 逐关节的观测项：维数 = 关节数，顺序 = 关节顺序。
JOINT_TERMS = ("joint_pos", "joint_vel", "actions")


@dataclasses.dataclass(frozen=True)
class Layout:
    """一个环境的观测与动作布局。

    Attributes:
        actor: actor 观测的各项 (名字, 维数)，按拼接顺序。
        critic: critic 观测的各项。
        joints: 动作对应的关节名，按动作顺序。
    """

    actor: tuple[tuple[str, int], ...]
    critic: tuple[tuple[str, int], ...]
    joints: tuple[str, ...]


def layout_of(env) -> Layout:
    """从建好的环境里取布局。

    Args:
        env: `env.DuckEnv`。

    Returns:
        布局。
    """
    om = env.unwrapped.observation_manager
    groups = {g: tuple(zip(om.active_terms[g], (int(d[0]) for d in om.group_obs_term_dim[g]), strict=True))
              for g in ("actor", "critic")}
    joints = tuple(j for j in env.unwrapped.scene["robot"].joint_names if not j.startswith("passive_"))
    if len(joints) != env.action_dim:
        raise RuntimeError(f"动作 {env.action_dim} 维，关节 {len(joints)} 个，对不上")
    return Layout(groups["actor"], groups["critic"], joints)


def _columns(source_terms, target_terms, source_joints, target_joints) -> tuple[list[int], list[int]]:
    """目标观测每一维对应源观测的哪一维；以及源观测里被切掉的那些维。

    Raises:
        RuntimeError: 目标里有源没有的项，或同名项维数对不上。
    """
    offsets, start = {}, 0
    for name, dim in source_terms:
        offsets[name] = (start, dim)
        start += dim
    kept = []
    for name, dim in target_terms:
        if name not in offsets:
            raise RuntimeError(f"目标观测里的 {name!r} 在源策略里没有，移植不了")
        begin, source_dim = offsets[name]
        if name in JOINT_TERMS:
            if source_dim != len(source_joints) or dim != len(target_joints):
                raise RuntimeError(f"{name} 维数（源 {source_dim} / 目标 {dim}）与关节数对不上")
            kept += [begin + source_joints.index(j) for j in target_joints]
        elif dim == source_dim:
            kept += list(range(begin, begin + dim))
        else:
            raise RuntimeError(f"{name} 源 {source_dim} 维、目标 {dim} 维，不是逐关节的项，移植不了")
    dropped = sorted(set(range(start)) - set(kept))
    return kept, dropped


def _transplant_input(state, prefix, normalizer, kept, dropped):
    """切第一层输入与归一化统计量，切掉的输入按定值 0 并进偏置。"""
    mean, var = state[f"{normalizer}.mean"], state[f"{normalizer}.var"]
    weight, bias = state[f"{prefix}.0.weight"], state[f"{prefix}.0.bias"]
    fixed = (0.0 - mean[dropped]) / (torch.sqrt(var[dropped]) + NORMALIZER_EPS)
    out = {f"{normalizer}.mean": mean[kept].clone(), f"{normalizer}.var": var[kept].clone(),
           f"{normalizer}.count": state[f"{normalizer}.count"].clone(),
           f"{prefix}.0.weight": weight[:, kept].clone(),
           f"{prefix}.0.bias": bias + weight[:, dropped] @ fixed}
    return out


def transplant(source_state: dict, source: Layout, target: Layout) -> dict:
    """把源策略的权重切成目标布局的形状。

    Args:
        source_state: 源 `ActorCritic` 的 state_dict。
        source: 源布局。
        target: 目标布局（关节须是源关节的子集）。

    Returns:
        目标 `ActorCritic` 的 state_dict。

    Raises:
        RuntimeError: 目标里有源没有的关节或观测项。
    """
    missing = sorted(set(target.joints) - set(source.joints))
    if missing:
        raise RuntimeError(f"目标关节 {missing} 在源策略里没有")
    state = dict(source_state)
    actor_kept, actor_dropped = _columns(source.actor, target.actor, list(source.joints), list(target.joints))
    critic_kept, critic_dropped = _columns(source.critic, target.critic, list(source.joints), list(target.joints))
    state.update(_transplant_input(source_state, "actor", "actor_normalizer", actor_kept, actor_dropped))
    state.update(_transplant_input(source_state, "critic", "critic_normalizer", critic_kept, critic_dropped))
    rows = [source.joints.index(j) for j in target.joints]
    last = max(int(k.split(".")[1]) for k in source_state if k.startswith("actor.") and k.endswith(".weight"))
    state[f"actor.{last}.weight"] = source_state[f"actor.{last}.weight"][rows].clone()
    state[f"actor.{last}.bias"] = source_state[f"actor.{last}.bias"][rows].clone()
    state["std"] = source_state["std"][rows].clone()
    probe = ActorCritic(len(actor_kept), len(critic_kept), len(rows))
    probe.load_state_dict(state)   # 形状对不上就在这里报错
    return state
