"""SAPIEN 3 ground-truth I/O used by the RoboTwin adapter.

All SAPIEN imports are lazy so that this module can be imported (and the rest
of SimuGuard unit-tested) without the simulator installed.

Lessons carried over from TwinGuard:
* velocities and mass live on PhysX components, not on ``sapien.Entity``
  (reading them from the entity silently returned zeros before 2026-08-31);
* ``qacc`` is derived by PhysX and is not restorable, so it is excluded from
  the public state used for restore verification;
* drive targets and generalized forces (``qf``) are part of the command for
  the next step and must be captured with every control record.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ...core.types import BodyInfo, BodyKind, BodyRole, BodyState, ContactPair, ContactPoint


def _physx():
    import sapien.physx as physx

    return physx


def vec(value: Any, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != size:
        raise ValueError(f"expected {size} values, got {array.size}")
    return array


def entity_key(entity: Any) -> Any:
    """Stable identity for an entity within one scene."""

    for attr in ("per_scene_id", "global_id"):
        value = getattr(entity, attr, None)
        if value is not None and not callable(value):
            return (attr, int(value))
    return ("pyid", id(entity))


def component_key(component: Any) -> Any:
    entity = getattr(component, "entity", None)
    if entity is None:
        return ("pyid", id(component))
    return entity_key(entity)


def unique_ids(names: Iterable[str], prefix: str) -> list[str]:
    counts: Counter[str] = Counter()
    ids = []
    for name in names:
        base = f"{prefix}:{name or 'unnamed'}"
        ids.append(base if counts[base] == 0 else f"{base}#{counts[base]}")
        counts[base] += 1
    return ids


# ---------------------------------------------------------------------------- inventory
@dataclass
class BodyHandle:
    info: BodyInfo
    entity: Any
    component: Any | None  # PhysX rigid component (dynamic/static/link)


@dataclass
class ArticulationHandle:
    art_id: str
    articulation: Any
    is_robot: bool


def enumerate_scene(
    scene: Any,
    *,
    robot_articulations: set[Any],
    role_by_entity: dict[Any, BodyRole],
    scene_name_tokens: tuple[str, ...] = ("table", "wall", "ground"),
) -> tuple[dict[str, BodyHandle], dict[str, ArticulationHandle]]:
    physx = _physx()
    bodies: dict[str, BodyHandle] = {}

    actors = list(scene.get_all_actors() or [])
    for body_id, entity in zip(unique_ids((e.get_name() for e in actors), "actor"), actors):
        dynamic = entity.find_component_by_type(physx.PhysxRigidDynamicComponent)
        static = entity.find_component_by_type(physx.PhysxRigidStaticComponent) if dynamic is None else None
        component = dynamic or static
        if component is None:
            continue  # cameras, lights, markers without collision
        if dynamic is not None:
            kind = BodyKind.KINEMATIC if bool(dynamic.get_kinematic()) else BodyKind.DYNAMIC
            mass = float(dynamic.get_mass())
        else:
            kind, mass = BodyKind.STATIC, None
        name = entity.get_name()
        role = role_by_entity.get(entity_key(entity))
        if role is None:
            if kind == BodyKind.STATIC or any(token in name.lower() for token in scene_name_tokens):
                role = BodyRole.SCENE
            else:
                role = BodyRole.OBJECT
        bodies[body_id] = BodyHandle(
            BodyInfo(
                body_id=body_id,
                name=name,
                kind=kind,
                role=role,
                mass=mass,
                collision_shape_count=len(component.get_collision_shapes() or []),
            ),
            entity,
            component,
        )

    articulations: dict[str, ArticulationHandle] = {}
    arts = list(scene.get_all_articulations() or [])
    for art_id, art in zip(unique_ids((a.get_name() for a in arts), "art"), arts):
        is_robot = articulation_key(art) in robot_articulations
        articulations[art_id] = ArticulationHandle(art_id, art, is_robot)
        links = list(art.get_links() or [])
        link_ids = unique_ids((link.get_name() for link in links), f"link:{art_id.split(':', 1)[1]}")
        for body_id, link in zip(link_ids, links):
            bodies[body_id] = BodyHandle(
                BodyInfo(
                    body_id=body_id,
                    name=link.get_name(),
                    kind=BodyKind.LINK,
                    role=BodyRole.ROBOT if is_robot else role_by_entity.get(component_key(link), BodyRole.OBJECT),
                    mass=float(link.get_mass()),
                    articulation=art_id,
                    collision_shape_count=len(link.get_collision_shapes() or []),
                ),
                link.entity,
                link,
            )
    return bodies, articulations


def articulation_key(articulation: Any) -> Any:
    """Identity of an articulation via its root link entity (wrapper-independent)."""

    root = articulation.get_root() if hasattr(articulation, "get_root") else articulation.get_links()[0]
    return component_key(root)


# ---------------------------------------------------------------------------- states & contacts
def read_state(handle: BodyHandle) -> BodyState:
    component = handle.component
    pose = component.get_pose() if handle.info.kind == BodyKind.LINK else handle.entity.get_pose()
    if handle.info.kind in (BodyKind.STATIC,):
        v = w = np.zeros(3)
    else:
        v = vec(component.get_linear_velocity(), 3)
        w = vec(component.get_angular_velocity(), 3)
    return BodyState(handle.info.body_id, vec(pose.p, 3), vec(pose.q, 4), v, w)


def read_contacts(scene: Any, id_by_key: dict[Any, str]) -> list[ContactPair]:
    """One :class:`ContactPair` per *body* pair.

    SAPIEN reports one ``PhysxContact`` per collision-*shape* pair; a CoACD
    decomposed can against a decomposed basket yields dozens of entries per
    substep.  Points are merged per unordered body pair and impulses are
    oriented so that ``total_impulse`` acts on ``body_a``.
    """

    merged: dict[tuple[str, str], ContactPair] = {}
    for contact in scene.get_contacts() or []:
        body0, body1 = contact.bodies[0], contact.bodies[1]
        a = id_by_key.get(component_key(body0)) or f"unknown:{_entity_name(body0)}"
        b = id_by_key.get(component_key(body1)) or f"unknown:{_entity_name(body1)}"
        key = (a, b) if a <= b else (b, a)
        sign = 1.0 if key[0] == a else -1.0
        pair = merged.get(key)
        if pair is None:
            pair = merged[key] = ContactPair(key[0], key[1], [], shape_pairs=0)
        pair.shape_pairs += 1
        for point in contact.points:
            pair.points.append(
                ContactPoint(
                    position=vec(point.position, 3),
                    normal=sign * vec(point.normal, 3),
                    impulse=sign * vec(point.impulse, 3),
                    separation=float(point.separation),
                )
            )
    return list(merged.values())


def _entity_name(component: Any) -> str:
    entity = getattr(component, "entity", None)
    try:
        return entity.get_name() if entity is not None else "?"
    except Exception:  # noqa: BLE001
        return "?"


# ---------------------------------------------------------------------------- control
def capture_articulation_control(articulation: Any) -> dict[str, Any]:
    joints = []
    for index, joint in enumerate(articulation.get_active_joints() or []):
        joints.append(
            {
                "index": index,
                "name": joint.get_name(),
                "drive_target": np.asarray(joint.get_drive_target(), dtype=float).reshape(-1).tolist(),
                "drive_velocity_target": np.asarray(joint.get_drive_velocity_target(), dtype=float).reshape(-1).tolist(),
            }
        )
    return {"qf": np.asarray(articulation.get_qf(), dtype=float).reshape(-1).tolist(), "joints": joints}


def apply_articulation_control(articulation: Any, payload: dict[str, Any], art_id: str) -> list[str]:
    problems: list[str] = []
    qf = np.asarray(payload.get("qf", []), dtype=float).reshape(-1)
    if qf.size:
        if qf.size != int(articulation.get_dof()):
            problems.append(f"{art_id}:qf:{qf.size}!={articulation.get_dof()}")
        else:
            articulation.set_qf(qf)
    joints = list(articulation.get_active_joints() or [])
    for item in payload.get("joints", []):
        index = int(item["index"])
        if index >= len(joints) or joints[index].get_name() != item["name"]:
            problems.append(f"{art_id}:joint[{index}]")
            continue
        joints[index].set_drive_target(_scalar_or_array(item["drive_target"]))
        joints[index].set_drive_velocity_target(_scalar_or_array(item["drive_velocity_target"]))
    return problems


def _scalar_or_array(values: Any) -> Any:
    array = np.asarray(values, dtype=float).reshape(-1)
    return float(array[0]) if array.size == 1 else array


# ---------------------------------------------------------------------------- public state
def _pose_dict(pose: Any) -> dict[str, list[float]]:
    return {"p": vec(pose.p, 3).tolist(), "q": vec(pose.q, 4).tolist()}


def _sapien_pose(data: dict[str, Any]) -> Any:
    import sapien

    return sapien.Pose(np.asarray(data["p"], dtype=float), np.asarray(data["q"], dtype=float))


def capture_public_state(bodies: dict[str, BodyHandle], articulations: dict[str, ArticulationHandle]) -> dict[str, Any]:
    actors = {}
    for body_id, handle in bodies.items():
        if handle.info.kind not in (BodyKind.DYNAMIC, BodyKind.KINEMATIC):
            continue
        component = handle.component
        actors[body_id] = {
            "pose": _pose_dict(handle.entity.get_pose()),
            "linear_velocity": vec(component.get_linear_velocity(), 3).tolist(),
            "angular_velocity": vec(component.get_angular_velocity(), 3).tolist(),
        }
    arts = {}
    for art_id, handle in articulations.items():
        art = handle.articulation
        arts[art_id] = {
            "root_pose": _pose_dict(art.get_root_pose()),
            "root_linear_velocity": vec(art.get_root_linear_velocity(), 3).tolist(),
            "root_angular_velocity": vec(art.get_root_angular_velocity(), 3).tolist(),
            "qpos": np.asarray(art.get_qpos(), dtype=float).reshape(-1).tolist(),
            "qvel": np.asarray(art.get_qvel(), dtype=float).reshape(-1).tolist(),
        }
    return {"actors": actors, "articulations": arts}


def restore_public_state(
    state: dict[str, Any],
    bodies: dict[str, BodyHandle],
    articulations: dict[str, ArticulationHandle],
) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    mismatched: list[str] = []
    for body_id, data in state.get("actors", {}).items():
        handle = bodies.get(body_id)
        if handle is None or handle.info.kind not in (BodyKind.DYNAMIC, BodyKind.KINEMATIC):
            missing.append(body_id)
            continue
        handle.entity.set_pose(_sapien_pose(data["pose"]))
        if handle.info.kind == BodyKind.DYNAMIC:
            handle.component.set_linear_velocity(np.asarray(data["linear_velocity"], dtype=float))
            handle.component.set_angular_velocity(np.asarray(data["angular_velocity"], dtype=float))
    for art_id, data in state.get("articulations", {}).items():
        handle = articulations.get(art_id)
        if handle is None:
            missing.append(art_id)
            continue
        art = handle.articulation
        art.set_root_pose(_sapien_pose(data["root_pose"]))
        art.set_root_linear_velocity(np.asarray(data["root_linear_velocity"], dtype=float))
        art.set_root_angular_velocity(np.asarray(data["root_angular_velocity"], dtype=float))
        for field in ("qpos", "qvel"):
            values = np.asarray(data[field], dtype=float)
            if values.size != int(art.get_dof()):
                mismatched.append(f"{art_id}:{field}")
                continue
            getattr(art, f"set_{field}")(values)
    return missing, mismatched


# ---------------------------------------------------------------------------- native state
def physx_system(scene: Any) -> Any:
    getter = getattr(scene, "get_physx_system", None)
    return getter() if callable(getter) else getattr(scene, "physx_system")


def pack_native(scene: Any) -> bytes | None:
    system = physx_system(scene)
    pack = getattr(system, "pack", None)
    return bytes(pack()) if callable(pack) else None


def unpack_native(scene: Any, data: bytes) -> None:
    system = physx_system(scene)
    system.unpack(data)


# ---------------------------------------------------------------------------- interventions
def solver_iterations(bodies: dict[str, BodyHandle], articulations: dict[str, ArticulationHandle]) -> dict[str, list[int]]:
    dyn = [h.component for h in bodies.values() if h.info.kind == BodyKind.DYNAMIC]
    arts = [h.articulation for h in articulations.values()]
    return {
        "rigid_position": sorted({int(c.get_solver_position_iterations()) for c in dyn}),
        "rigid_velocity": sorted({int(c.get_solver_velocity_iterations()) for c in dyn}),
        "articulation_position": sorted({int(a.get_solver_position_iterations()) for a in arts}),
        "articulation_velocity": sorted({int(a.get_solver_velocity_iterations()) for a in arts}),
    }


def set_solver_iterations(
    bodies: dict[str, BodyHandle],
    articulations: dict[str, ArticulationHandle],
    *,
    position: int,
    velocity: int,
) -> None:
    for handle in bodies.values():
        if handle.info.kind == BodyKind.DYNAMIC:
            handle.component.set_solver_position_iterations(int(position))
            handle.component.set_solver_velocity_iterations(int(velocity))
    for handle in articulations.values():
        handle.articulation.set_solver_position_iterations(int(position))
        handle.articulation.set_solver_velocity_iterations(int(velocity))
