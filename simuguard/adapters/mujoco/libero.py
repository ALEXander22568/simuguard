"""LIBERO (BDDL task suites on robosuite) on the MuJoCo adapter.

LIBERO builds every task from a BDDL file on robosuite: movable objects (one free body each) in
``env.objects_dict``, fixtures (cabinets, stoves, microwaves, racks; welded to the world or
articulated) in ``env.fixtures_dict``, named regions in ``env.object_sites_dict`` (a region knows
the object or fixture it belongs to) and the goal as predicates in
``env.parsed_problem["goal_state"]``, e.g. ``[["on", "akita_black_bowl_1", "plate_1"]]`` or
``[["in", "white_yellow_mug_1", "microwave_1_heating_region"], ["close", "microwave_1"]]``.

Roles come from the goal: the first argument of a binary predicate is a TARGET (the object the
policy has to move) and the object or fixture owning the second argument is a CONTAINER, together
with every body below it (drawers, doors, burner plates).  Table regions leave the table in the
scene: making the table a container would make every object resting on it a "risky" pair.  All
other free objects are OBJECTs, still watched by the detectors; robot, gripper and mount bodies are
ROBOT (robosuite >= 1.5 names the mount ``fixed_mount0_*``).

Episodes are rebuilt the way LIBERO evaluations run them: ``OffScreenRenderEnv`` on the task's BDDL
file, ``env.seed(seed)`` (``np.random.seed``: fixture placement on reset draws from the global
numpy generator, the benchmark's init states only hold qpos/qvel), ``env.reset()`` (a hard reset:
a new model and ``MjSim``), ``env.set_init_state(init_states[i])``.  The same four calls with the
same arguments in a fresh process give the same model and state, which is what exact replay needs;
:func:`model_fingerprint` checks the model half of that.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import mujoco
import numpy as np

from .adapter import DEFAULT_ROBOT_PREFIXES, MujocoAdapter, TaskRoles, attach_robosuite

ROBOT_PREFIXES = DEFAULT_ROBOT_PREFIXES + ("fixed_mount",)
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
BODY = mujoco.mjtObj.mjOBJ_BODY


def robosuite_env(env: Any) -> Any:
    """The robosuite env inside a LIBERO ``ControlEnv`` / ``OffScreenRenderEnv`` (or the env itself)."""
    inner = getattr(env, "env", None)
    return inner if inner is not None and hasattr(inner, "sim") else env


def goal_predicates(goal: Any) -> list[list[str]]:
    """Flatten a parsed BDDL goal into atomic predicates ``[name, arg, ...]`` (lower-case names)."""
    out: list[list[str]] = []
    if not isinstance(goal, (list, tuple)) or not goal:
        return out
    head = goal[0]
    if isinstance(head, str):
        name = head.lower()
        if name in ("and", "or", "not"):
            for sub in goal[1:]:
                out.extend(goal_predicates(sub))
        else:
            out.append([name, *[str(a) for a in goal[1:]]])
        return out
    for sub in goal:  # a list of predicates (LIBERO's implicit "and")
        out.extend(goal_predicates(sub))
    return out


def _subtree(model: mujoco.MjModel, root: int) -> list[int]:
    out = []
    for b in range(model.nbody):
        p = b
        while p > 0:
            if p == root:
                out.append(b)
                break
            p = int(model.body_parentid[p])
    return out


def libero_roles(env: Any) -> tuple[TaskRoles, dict[str, Any]]:
    """TaskRoles from the BDDL goal, plus a description of how each role was chosen."""
    rs = robosuite_env(env)
    model = rs.sim.model._model
    objects = {name: obj.root_body for name, obj in rs.objects_dict.items()}
    fixtures = {name: obj.root_body for name, obj in getattr(rs, "fixtures_dict", {}).items()}
    sites = {name: getattr(site, "parent_name", None) for name, site in getattr(rs, "object_sites_dict", {}).items()}
    goal = (getattr(rs, "parsed_problem", None) or {}).get("goal_state", [])

    def owner(arg: str) -> str | None:
        if arg in objects or arg in fixtures:
            return arg
        parent = sites.get(arg)
        return parent if (parent in objects or parent in fixtures) else None

    targets: set[str] = set()
    containers: set[str] = set()
    for predicate in goal_predicates(goal):
        args = predicate[1:]
        if len(args) < 2:
            continue  # open / close / turnon: an articulated fixture, nothing is carried
        first, second = owner(args[0]), owner(args[1])
        if first in objects:
            targets.add(first)
        if second is not None and second != first:
            containers.add(second)
    containers -= targets

    target_bodies = {objects[n] for n in targets}
    container_bodies: set[str] = set()
    for name in sorted(containers):
        root_body = objects.get(name) or fixtures.get(name)
        root = mujoco.mj_name2id(model, BODY, root_body) if root_body else -1
        if root < 0:
            continue
        container_bodies.update(mujoco.mj_id2name(model, BODY, b) for b in _subtree(model, root))
    container_bodies -= target_bodies
    roles = TaskRoles(targets=target_bodies, containers=container_bodies, robot_prefixes=ROBOT_PREFIXES)
    problem = getattr(rs, "parsed_problem", None) or {}
    language = problem.get("language_instruction") or problem.get("language") or ""
    info = {
        "goal_state": goal,
        "target_objects": sorted(targets),
        "container_objects": sorted(containers),
        "target_bodies": sorted(target_bodies),
        "container_bodies": sorted(container_bodies),
        "objects": objects,
        "fixtures": fixtures,
        "obj_of_interest": list(getattr(rs, "obj_of_interest", []) or []),
        "language": " ".join(language) if isinstance(language, (list, tuple)) else str(language),
    }
    return roles, info


def model_fingerprint(model: mujoco.MjModel) -> dict[str, Any]:
    """SHA-256 of the compiled model plus the exact body poses LIBERO randomises (fixture placement).

    JSON keeps float64 exactly (repr round trip), so :func:`restore_placement` can write them back.
    """
    buf = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buf)
    return {
        "sha256": hashlib.sha256(buf.tobytes()).hexdigest(),
        "body_pos": model.body_pos.tolist(),
        "body_quat": model.body_quat.tolist(),
    }


def restore_placement(model: mujoco.MjModel, fingerprint: dict[str, Any]) -> float:
    """Write recorded body poses back into a rebuilt model; returns the largest change (m or quat units).

    Needed where the rebuild does not reproduce fixture placement: robosuite >= 1.5 samples placements
    from a per-env ``np.random.default_rng(seed=None)`` that LIBERO's ``env.seed`` does not reach.
    """
    pos = np.asarray(fingerprint["body_pos"], dtype=np.float64).reshape(model.body_pos.shape)
    quat = np.asarray(fingerprint["body_quat"], dtype=np.float64).reshape(model.body_quat.shape)
    change = float(max(np.abs(model.body_pos - pos).max(), np.abs(model.body_quat - quat).max()))
    model.body_pos[:] = pos
    model.body_quat[:] = quat
    return change


def libero_task(suite_name: str, task_id: int) -> tuple[Any, str, Any]:
    """(task, bddl path, init states) from LIBERO's benchmark registry."""
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    return task, bddl, suite.get_task_init_states(task_id)


def make_env(suite_name: str, task_id: int, init_index: int, seed: int, *, camera_size: int = 256,
             **env_kwargs: Any) -> tuple[Any, dict, dict[str, Any]]:
    """Build a LIBERO episode: env, first observation, episode description."""
    from libero.libero.envs import OffScreenRenderEnv

    task, bddl, init_states = libero_task(suite_name, task_id)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=camera_size, camera_widths=camera_size, **env_kwargs)
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(init_states[init_index % len(init_states)])
    meta = {"suite": suite_name, "task_id": int(task_id), "task": task.name, "language": task.language,
            "init_index": int(init_index), "seed": int(seed), "camera_size": int(camera_size),
            "bddl": os.path.relpath(bddl, os.path.dirname(os.path.dirname(bddl))), "n_init_states": len(init_states)}
    return env, obs, meta


def attach_libero(env: Any, *, task_name: str = "") -> tuple[MujocoAdapter, dict[str, Any]]:
    """MuJoCo adapter on a LIBERO env with goal-derived roles; attach again after every reset."""
    import robosuite

    rs = robosuite_env(env)
    roles, info = libero_roles(env)
    adapter = attach_robosuite(rs, roles=roles, task_name=task_name or type(rs).__name__)
    info.update({
        "robosuite": robosuite.__version__, "mujoco": mujoco.__version__,
        "lite_physics": bool(getattr(rs, "lite_physics", False)),
        "hooked_method": "step2" if getattr(rs, "lite_physics", False) else "step",
        "control_freq": float(rs.control_freq), "timestep": float(adapter.model.opt.timestep),
        "substeps_per_step": int(round(1.0 / (float(rs.control_freq) * float(adapter.model.opt.timestep)))),
    })
    fingerprint = model_fingerprint(adapter.model)
    base = adapter._ground_truth_fn

    def ground_truth() -> dict[str, Any]:
        payload = dict(base()) if base is not None else {}
        payload.update({"libero": info, "model_fingerprint": fingerprint})
        return payload

    adapter._ground_truth_fn = ground_truth
    return adapter, info
