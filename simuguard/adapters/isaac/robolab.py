"""NVIDIA RoboLab (Isaac Lab 2.2, batched GPU envs) glue: recording, exact replay, depenetration caps.

RoboLab runs ``num_envs`` copies of one task in one PhysX GPU scene.  Nothing in RoboLab is
modified; :class:`BatchRecorder` wraps a live ``RobolabEnv`` at runtime:

* every PhysX tensor-API write (``ArticulationView`` / ``RigidBodyView`` ``set_*`` and
  ``apply_*``, ``SimulationView.update_articulations_kinematic`` ...) is logged with the physics
  step it precedes (:class:`~.robolab_io.WriteLog`), which makes the whole batch replayable;
* ``env.sim.step`` is wrapped: before each physics step the per-env monitors capture actuation,
  after it one batched read gives every body state and contact of all envs;
* ``env._reset_idx`` is wrapped to log RoboLab's resets per env: the initial reset, terminations
  (success or timeout, after which RoboLab freezes the env) and the silent "physics artifact"
  resets of envs that terminate within two policy steps;
* each env gets a :class:`RobolabEnvAdapter` (a SimAdapter slice) and its own SubstepMonitor and
  EpisodeRecorder, i.e. one standard SimuGuard episode record per env.  An artifact reset ends
  that env's segment (marked discarded) and opens a new one.

Layout below ``<out>/<task>/``::

    batch/install.json          bodies, views, physics settings, caps
    batch/initial_states.npz    all body states and robot joints when the recorder was installed
    batch/run_<k>_writes.npz    every tensor-API write of run k (replay input)
    batch/run_<k>_robot.npz     robot joint positions/velocities per physics step (float32)
    batch/run_<k>.json          step range, resets, segments, timing, storage
    run_<k>/env_<e>/seg_<j>/    SimuGuard episode record (manifest, states.npz, controls.npz,
                                contacts.npz, trace.jsonl.gz, events.jsonl, summary.json)
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ...core.adapter import AdapterCapabilities, HookHandle, SimAdapter
from ...core.snapshot import RestoreReport, Snapshot
from ...core.snapshot_types import ControlRecord
from ...core.types import BodyInfo, BodyKind, BodyRole, BodyState, ContactPair
from .robolab_io import (
    CommandBuffer,
    ContactLog,
    DepenetrationCap,
    WriteLog,
    body_id_from_relative,
    cap_transitions,
    contact_pairs_from_data,
    split_env_path,
)

XYZW_TO_WXYZ = [3, 0, 1, 2]
LOGGED_PREFIXES = ("set_", "apply_")
SIM_VIEW_METHODS = ("update_articulations_kinematic", "clear_forces", "flush")
SKIP_METHODS = ("set_subspace_roots",)


# ---------------------------------------------------------------------------- scene inventory
def _env_pattern(path: str) -> str:
    parts = split_env_path(path)
    return f"/World/envs/env_*/{parts[1]}" if parts else path


def _kinematic(path: str) -> bool:
    import omni.usd

    prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
    attr = prim.GetAttribute("physics:kinematicEnabled") if prim and prim.IsValid() else None
    return bool(attr.Get()) if attr and attr.HasValue() else False


def task_roles(env: Any) -> tuple[dict[str, BodyRole], dict[str, Any]]:
    """Roles from RoboLab's termination terms: ``object``-like params -> TARGET, ``container``-like -> CONTAINER."""

    roles: dict[str, BodyRole] = {}
    terms: dict[str, Any] = {}
    names = set(env.scene.rigid_objects.keys()) | set(env.scene.articulations.keys())
    manager = env.termination_manager
    for term in manager.active_terms:
        params = dict(getattr(manager.get_term_cfg(term), "params", {}) or {})
        clean = {}
        for key, value in params.items():
            values = value if isinstance(value, (list, tuple)) else [value]
            strings = [v for v in values if isinstance(v, str)]
            if strings:
                clean[key] = strings
            for v in strings:
                if v not in names:
                    continue
                if "container" in key or "receptacle" in key:
                    roles.setdefault(v, BodyRole.CONTAINER)
                elif "object" in key or "target" in key:
                    roles.setdefault(v, BodyRole.TARGET)
        terms[term] = clean
    return roles, terms


class BatchScene:
    """Body inventory of all envs and the batched readers (one rigid view, articulation link views)."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.num_envs = int(env.num_envs)
        self.dt = float(env.physics_dt)
        self.sim_view = env.sim.physics_sim_view
        self.roles, self.termination_params = task_roles(env)
        self.objects: list[str] = list(env.scene.rigid_objects.keys())
        self.object_paths: dict[str, list[str]] = {}
        self.object_kind: dict[str, BodyKind] = {}
        paths: list[str] = []
        for name in self.objects:
            view = env.scene.rigid_objects[name].root_physx_view
            self.object_paths[name] = [str(p) for p in view.prim_paths]
            self.object_kind[name] = BodyKind.KINEMATIC if _kinematic(self.object_paths[name][0]) else BodyKind.DYNAMIC
            paths += self.object_paths[name]
        self.rigid_view = self.sim_view.create_rigid_body_view(paths) if paths else None
        self.rigid_rows: dict[tuple[int, str], int] = {}
        if self.rigid_view is not None:
            for row, path in enumerate(self.rigid_view.prim_paths):
                env_id, rel = split_env_path(str(path))
                self.rigid_rows[(env_id, body_id_from_relative(rel))] = row
        self.articulations: dict[str, Any] = {name: art.root_physx_view for name, art in env.scene.articulations.items()}
        self.links: dict[str, list[str]] = {}
        self.link_paths: dict[str, list[list[str]]] = {}
        for name, view in self.articulations.items():
            self.link_paths[name] = [[str(p) for p in row] for row in view.link_paths]
            self.links[name] = [f"{name}/{p.rsplit('/', 1)[-1]}" for p in self.link_paths[name][0]]
        self.robot = "robot" if "robot" in self.articulations else next(iter(self.articulations), None)
        self.dof_names = list(self.articulations[self.robot].dof_paths[0]) if self.robot else []

    def body_infos(self, env_id: int) -> dict[str, BodyInfo]:
        out: dict[str, BodyInfo] = {}
        for name in self.objects:
            role = self.roles.get(name) or (BodyRole.SCENE if "table" in name.lower() else BodyRole.OBJECT)
            out[name] = BodyInfo(name, name, self.object_kind[name], role,
                                 metadata={"env": env_id, "path": self.object_paths[name][env_id]})
        for art, ids in self.links.items():
            is_robot = art == self.robot
            role = BodyRole.ROBOT if is_robot else self.roles.get(art, BodyRole.OBJECT)
            for body_id, path in zip(ids, self.link_paths[art][env_id]):
                out[body_id] = BodyInfo(body_id, body_id.rsplit("/", 1)[-1], BodyKind.LINK, role, articulation=art,
                                        metadata={"env": env_id, "path": path})
        return out

    def body_path(self, env_id: int, body_id: str) -> str | None:
        if body_id in self.object_paths:
            return self.object_paths[body_id][env_id]
        art, _, link = body_id.partition("/")
        if art in self.link_paths:
            for path in self.link_paths[art][env_id]:
                if path.rsplit("/", 1)[-1] == link:
                    return path
        return None

    def read(self) -> dict[str, np.ndarray]:
        """All states, float64, quaternions wxyz: objects (E, O, 13); links per articulation (E, L, 13)."""

        out: dict[str, np.ndarray] = {}
        if self.rigid_view is not None:
            pose = self.rigid_view.get_transforms().cpu().numpy().astype(np.float64)
            vel = self.rigid_view.get_velocities().cpu().numpy().astype(np.float64)
            rows = np.concatenate([pose[:, :3], pose[:, 3:7][:, XYZW_TO_WXYZ], vel], axis=1)
            table = np.zeros((self.num_envs, len(self.objects), 13))
            for (env_id, name), row in self.rigid_rows.items():
                table[env_id, self.objects.index(name)] = rows[row]
            out["objects"] = table
        for name, view in self.articulations.items():
            pose = view.get_link_transforms().cpu().numpy().astype(np.float64)
            vel = view.get_link_velocities().cpu().numpy().astype(np.float64)
            out[f"links:{name}"] = np.concatenate([pose[..., :3], pose[..., 3:7][..., XYZW_TO_WXYZ], vel], axis=-1)
        return out

    def read_robot_joints(self) -> tuple[np.ndarray, np.ndarray]:
        view = self.articulations[self.robot]
        return view.get_dof_positions().cpu().numpy(), view.get_dof_velocities().cpu().numpy()

    def states_for(self, cache: dict[str, np.ndarray], env_id: int, ids: Iterable[str] | None = None) -> dict[str, BodyState]:
        out: dict[str, BodyState] = {}
        wanted = None if ids is None else set(ids)
        if "objects" in cache:
            for i, name in enumerate(self.objects):
                if wanted is None or name in wanted:
                    r = cache["objects"][env_id, i]
                    out[name] = BodyState(name, r[0:3], r[3:7], r[7:10], r[10:13])
        for art, ids_ in self.links.items():
            arr = cache[f"links:{art}"][env_id]
            for i, body_id in enumerate(ids_):
                if wanted is None or body_id in wanted:
                    r = arr[i]
                    out[body_id] = BodyState(body_id, r[0:3], r[3:7], r[7:10], r[10:13])
        return out


class BatchContacts:
    """One PhysX rigid contact view per scene object, filtered against the other objects and all links."""

    def __init__(self, scene: BatchScene, patches_per_pair: int = 64) -> None:
        self.scene = scene
        self.overflow_steps = 0
        self.overflow_points = 0
        self.views: list[tuple[str, list[str], list[int], Any]] = []
        self.errors: dict[str, str] = {}
        link_bodies = [(body_id, _env_pattern(path)) for art in scene.links
                       for body_id, path in zip(scene.links[art], scene.link_paths[art][0])]
        objects = scene.objects
        for i, name in enumerate(objects):
            others = [(o, _env_pattern(scene.object_paths[o][0])) for o in objects if o != name]
            filters = others + link_bodies
            skip = [k for k, (o, _) in enumerate(others) if objects.index(o) < i]
            try:
                view = scene.sim_view.create_rigid_contact_view(
                    _env_pattern(scene.object_paths[name][0]),
                    filter_patterns=[p for _, p in filters],
                    max_contact_data_count=scene.num_envs * len(filters) * patches_per_pair,
                )
                self.views.append((name, [b for b, _ in filters], skip, view))
            except Exception as exc:  # noqa: BLE001
                self.errors[name] = f"{type(exc).__name__}: {exc}"

    def read(self) -> dict[int, list[ContactPair]]:
        out: dict[int, list[ContactPair]] = {}
        dt = self.scene.dt
        for name, filters, skip, view in self.views:
            forces, points, normals, seps, counts, starts = view.get_contact_data(dt)
            counts = counts.cpu().numpy()
            if not counts.any():
                continue
            lost = int(counts.sum()) - int(forces.shape[0])
            if lost > 0:
                self.overflow_steps += 1
                self.overflow_points += lost
            pairs = contact_pairs_from_data(
                name, filters, forces.cpu().numpy(), points.cpu().numpy(), normals.cpu().numpy(),
                seps.cpu().numpy(), counts, starts.cpu().numpy(), dt, skip_filters=skip,
            )
            for env_id, items in pairs.items():
                out.setdefault(env_id, []).extend(items)
        return out


# ---------------------------------------------------------------------------- tensor-API write capture
class WriteCapture:
    """Class-level wrappers on the PhysX tensor views that log every state-changing call."""

    def __init__(self, on_write: Callable[[str, str, Any, Any, int | None], None]) -> None:
        self.on_write = on_write
        self.names: dict[int, str] = {}
        self.unknown: dict[str, list[str]] = {}
        self._originals: list[tuple[type, str, Any]] = []
        self.paused = 0

    def name(self, view: Any) -> str:
        key = self.names.get(id(view))
        if key is None:
            key = f"view{len(self.unknown)}"
            self.unknown[key] = [str(p) for p in getattr(view, "prim_paths", [])]
            self.names[id(view)] = key
        return key

    def install(self) -> None:
        from omni.physics.tensors.impl import api

        for cls in (api.ArticulationView, api.RigidBodyView):
            for attr in dir(cls):
                if attr.startswith(LOGGED_PREFIXES) and attr not in SKIP_METHODS and callable(getattr(cls, attr)):
                    self._wrap(cls, attr)
        for attr in SIM_VIEW_METHODS:
            if hasattr(api.SimulationView, attr):
                self._wrap(api.SimulationView, attr)

    def _wrap(self, cls: type, attr: str) -> None:
        original = getattr(cls, attr)
        capture = self

        def wrapper(view: Any, *args: Any, **kwargs: Any) -> Any:
            if not capture.paused:
                data = kwargs.get("data", args[0] if args else None)
                indices = kwargs.get("indices", args[1] if len(args) > 1 else None)
                count = getattr(view, "count", None)
                if attr.startswith("apply_"):
                    capture.on_write(capture.name(view), attr, None, None, count)
                else:
                    capture.on_write(capture.name(view), attr, data, indices, count)
            return original(view, *args, **kwargs)

        wrapper.__wrapped__ = original
        setattr(cls, attr, wrapper)
        self._originals.append((cls, attr, original))

    def remove(self) -> None:
        for cls, attr, original in reversed(self._originals):
            setattr(cls, attr, original)
        self._originals.clear()


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


# ---------------------------------------------------------------------------- depenetration caps
class CapController:
    """Writes ``physxRigidBody:maxDepenetrationVelocity`` per body/env and restores the authored value.

    ``via="usd"`` writes the USD attribute, ``via="usdrt"`` writes the Fabric (usdrt) attribute.
    Caps active from step 1 are best applied with :func:`prebuild_caps` before the scene is simulated.
    """

    ATTR = "physxRigidBody:maxDepenetrationVelocity"

    def __init__(self, scene: BatchScene, caps: list[DepenetrationCap], via: str = "usd") -> None:
        self.scene = scene
        self.caps = list(caps)
        self.via = via
        self.original: dict[str, float | None] = {}
        self.log: list[dict[str, Any]] = []

    def targets(self, cap: DepenetrationCap) -> list[str]:
        scene = self.scene
        envs = range(scene.num_envs) if cap.envs is None else cap.envs
        bodies = cap.bodies
        if bodies is None:
            bodies = [o for o in scene.objects if scene.object_kind[o] == BodyKind.DYNAMIC]
        paths = []
        for env_id in envs:
            for body_id in bodies:
                path = scene.body_path(env_id, body_id)
                if path is None:
                    raise KeyError(f"unknown body {body_id!r} (have {scene.objects + sum(scene.links.values(), [])})")
                paths.append(path)
        return paths

    def _get(self, path: str) -> float | None:
        import omni.usd

        prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
        attr = prim.GetAttribute(self.ATTR)
        return float(attr.Get()) if attr and attr.HasValue() else None

    def _set(self, paths: list[str], value: float | None) -> None:
        if self.via == "usdrt":
            import omni.usd
            import usdrt

            stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
            for path in paths:
                stage.GetPrimAtPath(path).GetAttribute(self.ATTR).Set(float(self.original[path] if value is None else value))
            return
        import omni.usd
        from pxr import PhysxSchema

        stage = omni.usd.get_context().get_stage()
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            target = self.original[path] if value is None else value
            api = PhysxSchema.PhysxRigidBodyAPI(prim)
            if target is None:
                api.GetMaxDepenetrationVelocityAttr().Clear()
            else:
                api.GetMaxDepenetrationVelocityAttr().Set(float(target))

    def before_step(self, substep: int) -> None:
        for cap, on in cap_transitions(self.caps, substep):
            paths = self.targets(cap)
            for path in paths:
                self.original.setdefault(path, self._get(path))
            self._set(paths, cap.value if on else None)
            self.log.append({"substep": substep, "on": on, "cap": cap.to_dict(), "bodies": len(paths), "via": self.via})


def free_scene_bodies(stage: Any, env_id: int) -> list[str]:
    """Dynamic (non-kinematic) rigid bodies directly under ``/World/envs/env_<e>/scene``."""

    from pxr import UsdPhysics

    out = []
    for prim in stage.GetPrimAtPath(f"/World/envs/env_{env_id}/scene").GetChildren():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            attr = prim.GetAttribute("physics:kinematicEnabled")
            if not (attr and attr.HasValue() and attr.Get()):
                out.append(prim.GetName())
    return out


def body_prim(stage: Any, env_id: int, body_id: str) -> Any:
    """The rigid-body prim of a body id: ``banana`` -> ``/World/envs/env_<e>/scene/banana``; a robot link
    ``robot/<link>`` -> the rigid body named ``<link>`` anywhere under ``/World/envs/env_<e>/robot``."""

    from pxr import Usd, UsdPhysics

    if not body_id.startswith("robot/"):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/scene/{body_id}")
        if not prim.IsValid():
            raise ValueError(f"no body {body_id!r} in env {env_id}")
        return prim
    link = body_id.split("/", 1)[1]
    root = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/robot")
    for prim in Usd.PrimRange(root):
        if prim.GetName() == link and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim
    raise ValueError(f"no robot link {link!r} in env {env_id}")


def prebuild_caps(caps: list[DepenetrationCap], num_envs: int) -> Callable[[], None]:
    """Write the caps that start at step 1 into USD right before the scene is first simulated.

    Patches ``isaaclab.sim.SimulationContext.reset`` (the first reset is when PhysX parses the
    stage).  Bodies are given as ``scene/<name>`` relative paths here: ``banana`` ->
    ``/World/envs/env_<e>/scene/banana``; robot links ``robot/<link>`` (see ``body_prim``).  Returns an undo
    function.
    """

    import isaaclab.sim as isim

    original = isim.SimulationContext.reset
    state = {"done": False, "paths": []}

    def reset(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not state["done"]:
            import omni.usd
            from pxr import PhysxSchema

            stage = omni.usd.get_context().get_stage()
            for cap in caps:
                if cap.start > 1:
                    continue
                for env_id in range(num_envs) if cap.envs is None else cap.envs:
                    for body_id in cap.bodies or free_scene_bodies(stage, env_id):
                        prim = body_prim(stage, env_id, body_id)
                        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateMaxDepenetrationVelocityAttr().Set(float(cap.value))
                        state["paths"].append(str(prim.GetPath()))
            state["done"] = True
        return original(self, *args, **kwargs)

    isim.SimulationContext.reset = reset

    def undo() -> None:
        isim.SimulationContext.reset = original

    undo.state = state  # type: ignore[attr-defined]
    return undo


# ---------------------------------------------------------------------------- per-env SimAdapter slice
class RobolabEnvAdapter(SimAdapter):
    """One env of the batch as a SimAdapter; all reads come from the batch cache of the last step."""

    name = "robolab"

    def __init__(self, hub: "BatchRecorder", env_id: int) -> None:
        self.hub = hub
        self.env_id = env_id
        self._bodies = hub.scene.body_infos(env_id)

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            native_state_snapshot=False,
            notes=["batched GPU env: replay is batch-level (write log of all envs), see adapters/isaac/robolab.py"],
        )

    def timestep(self) -> float:
        return self.hub.scene.dt

    def bodies(self, refresh: bool = False) -> dict[str, BodyInfo]:
        return self._bodies

    def read_states(self, body_ids: Iterable[str] | None = None) -> dict[str, BodyState]:
        return self.hub.scene.states_for(self.hub.cache, self.env_id, body_ids)

    def read_contacts(self) -> list[ContactPair]:
        return self.hub.contacts.get(self.env_id, [])

    def control_step(self) -> int:
        return int(self.hub.episode_length[self.env_id])

    def task_ground_truth(self) -> dict[str, Any]:
        return self.hub.ground_truth(self.env_id)

    def install_substep_hook(self, after: Callable[[], None], before: Callable[[], None] | None = None) -> HookHandle:
        self.hub.hooks[self.env_id] = (before, after)
        return HookHandle(lambda: self.hub.hooks.pop(self.env_id, None), "robolab_batch_step")

    def capture_control(self) -> ControlRecord:
        return ControlRecord(self.hub.substep + 1, self.control_step(), self.hub.commands.payload(self.env_id))

    def apply_control(self, control: ControlRecord) -> None:
        raise NotImplementedError("RoboLab replay is batch-level: replay_batch()")

    def capture_snapshot(self, substep: int, *, include_native: bool = True) -> Snapshot:
        states = {k: v.to_dict(None) for k, v in sorted(self.read_states().items())}
        return Snapshot(substep, self.control_step(), {"bodies": states}, self.capture_control())

    def restore_snapshot(self, snapshot: Snapshot, *, method: str = "auto") -> RestoreReport:
        raise NotImplementedError("RoboLab replay is batch-level: replay_batch()")

    def step_physics(self) -> None:
        raise NotImplementedError("RoboLab replay is batch-level: replay_batch()")


# ---------------------------------------------------------------------------- recorder
class BatchRecorder:
    """Records every env of a RobolabEnv across RoboLab's runs (see module docstring)."""

    def __init__(
        self,
        env: Any,
        out_dir: str | Path,
        *,
        task_name: str,
        detector_cfg: dict[str, Any],
        monitor_cfg: dict[str, Any] | None = None,
        caps: list[DepenetrationCap] | None = None,
        cap_via: str = "usd",
        record: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.env = env
        self.out = Path(out_dir)
        self.task_name = task_name
        self.detector_cfg = detector_cfg
        self.monitor_cfg = dict(monitor_cfg or {})
        self.record = record
        self.metadata = dict(metadata or {})
        self.scene = BatchScene(env)
        self.contacts_reader = BatchContacts(self.scene) if record else None
        self.caps = CapController(self.scene, caps or [], cap_via)
        self.writes = WriteLog()
        self.commands = CommandBuffer(self.scene.num_envs)
        self.capture = WriteCapture(self._on_write)
        self.substep = 0  # physics steps done since install
        self.physx_steps = 0
        self.cache: dict[str, np.ndarray] = {}
        self.contacts: dict[int, list[ContactPair]] = {}
        self.episode_length = np.zeros(self.scene.num_envs, dtype=np.int64)
        self.hooks: dict[int, tuple[Callable[[], None] | None, Callable[[], None]]] = {}
        self.run: dict[str, Any] | None = None
        self.overhead_s = 0.0
        self.profile = dict.fromkeys(("before_s", "read_states_s", "read_contacts_s", "monitors_s", "contact_log_s"), 0.0)
        self._sim_step = None
        self._reset_idx = None
        self._physx_sub = None
        self._next_write = 0  # runs are contiguous: run k starts where run k-1 ended
        self._next_substep = 1

    # ------------------------------------------------------------------ install / remove
    def install(self) -> "BatchRecorder":
        env = self.env
        batch = self.out / "batch"
        batch.mkdir(parents=True, exist_ok=True)
        for name, view in self.scene.articulations.items():
            self.capture.names[id(view)] = f"articulation:{name}"
        for name in self.scene.objects:
            self.capture.names[id(env.scene.rigid_objects[name].root_physx_view)] = f"rigid:{name}"
        self.capture.names[id(self.scene.sim_view)] = "sim_view"
        self.capture.install()
        self._sim_step = env.sim.step
        env.sim.step = self._step
        self._reset_idx = env._reset_idx
        env._reset_idx = self._reset_wrapper
        import omni.physx

        self._physx_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(self._count_physx_step)
        self.cache = self.scene.read()
        q, qd = self.scene.read_robot_joints()
        np.savez_compressed(batch / "initial_states.npz", objects=self.cache.get("objects", np.zeros(0)),
                            robot_links=self.cache.get(f"links:{self.scene.robot}", np.zeros(0)), dof_pos=q, dof_vel=qd)
        info = {
            "task": self.task_name,
            "num_envs": self.scene.num_envs,
            "physics_dt": self.scene.dt,
            "decimation": int(env.cfg.decimation),
            "objects": {o: {"kind": self.scene.object_kind[o].value, "paths": self.scene.object_paths[o]} for o in self.scene.objects},
            "articulations": {a: {"links": self.scene.links[a]} for a in self.scene.links},
            "robot": self.scene.robot,
            "dof_names": self.scene.dof_names,
            "roles": {k: v.value for k, v in self.scene.roles.items()},
            "termination_params": self.scene.termination_params,
            "contact_views": [v[0] for v in self.contacts_reader.views] if self.contacts_reader else [],
            "contact_view_errors": self.contacts_reader.errors if self.contacts_reader else {},
            "caps": [c.to_dict() for c in self.caps.caps],
            "cap_via": self.caps.via,
            "physics": physics_summary(env),
            "metadata": self.metadata,
        }
        (batch / "install.json").write_text(json.dumps(info, indent=2, default=str))
        return self

    def remove(self) -> None:
        self.capture.remove()
        if self._sim_step is not None:
            self.env.sim.step = self._sim_step
            self.env._reset_idx = self._reset_idx
        self._physx_sub = None

    def _count_physx_step(self, _dt: float) -> None:
        self.physx_steps += 1

    # ------------------------------------------------------------------ write log
    def _on_write(self, target: str, method: str, data: Any, indices: Any, count: int | None) -> None:
        t0 = time.perf_counter()
        idx = _to_numpy(indices)
        arr = _to_numpy(data)
        if arr is not None and idx is not None and arr.ndim and (arr.shape[0] == count or arr.shape[0] != idx.size):
            arr = arr[idx.astype(np.int64)]
        self.writes.append(self.substep + 1, target, method, idx, arr)
        if target == f"articulation:{self.scene.robot}":
            self.commands.update(self.writes.writes[-1])
        self.overhead_s += time.perf_counter() - t0

    # ------------------------------------------------------------------ step / reset wrappers
    def _step(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        self.caps.before_step(self.substep + 1)
        if self.run is not None and self.run["started_substep"] is None:
            self._open_all_segments()
        for before, _ in list(self.hooks.values()):
            if before is not None:
                before()
        t1 = time.perf_counter()
        result = self._sim_step(*args, **kwargs)
        t2 = time.perf_counter()
        self.substep += 1
        prof = self.profile
        prof["before_s"] += t1 - t0
        if self.record and self.hooks:
            self.cache = self.scene.read()
            t3 = time.perf_counter()
            self.contacts = self.contacts_reader.read() if self.contacts_reader else {}
            self.episode_length = self.env.episode_length_buf.cpu().numpy()
            t4 = time.perf_counter()
            for env_id, (_, after) in list(self.hooks.items()):
                after()
            t5 = time.perf_counter()
            for env_id in self.hooks:
                local = self.substep - self.segment_info[env_id]["start_substep"] + 1
                self.segment_contacts[env_id].append(local, self.contacts.get(env_id, []))
            t6 = time.perf_counter()
            prof["read_states_s"] += t3 - t2
            prof["read_contacts_s"] += t4 - t3
            prof["monitors_s"] += t5 - t4
            prof["contact_log_s"] += t6 - t5
        if self.run is not None:
            q, qd = self.scene.read_robot_joints()
            self.run["robot_q"].append(q)
            self.run["robot_qd"].append(qd)
            self.run["physics_s"] += t2 - t1
        self.overhead_s += (t1 - t0) + (time.perf_counter() - t2)
        return result

    def _reset_wrapper(self, env_ids: Any) -> Any:
        t0 = time.perf_counter()
        env = self.env
        ids = [int(i) for i in env_ids.tolist()]
        stepped = bool(env._has_stepped)
        frozen_before = env._frozen_envs.cpu().numpy().copy()
        lengths = env.episode_length_buf.cpu().numpy().copy()
        self.overhead_s += time.perf_counter() - t0
        result = self._reset_idx(env_ids)
        t1 = time.perf_counter()
        frozen_after = env._frozen_envs.cpu().numpy()
        for eid in ids:
            if not stepped:
                kind = "initial"
            elif frozen_before[eid]:
                kind = "frozen"
            elif frozen_after[eid]:
                kind = "terminated"
            else:
                kind = "artifact" if lengths[eid] <= 2 else "reset"
            event = {"substep": self.substep, "env": eid, "kind": kind, "episode_length": int(lengths[eid])}
            if kind == "terminated":
                event["success"] = bool(env._env_results.get(eid))
            if self.run is not None:
                self.run["resets"].append(event)
                if kind in ("terminated", "artifact", "reset") and eid in self.monitors:
                    self._close_segment(eid, kind, event)
                    if kind != "terminated":
                        self._open_segment(eid)
        self.overhead_s += time.perf_counter() - t1
        return result

    # ------------------------------------------------------------------ runs and segments
    def begin_run(self, run_idx: int) -> None:
        self.run = {
            "run": int(run_idx),
            "write_start": self._next_write,
            "first_substep": self._next_substep,
            "started_substep": None,
            "resets": [],
            "segments": [],
            "robot_q": [],
            "robot_qd": [],
            "physics_s": 0.0,
            "wall_start": time.time(),
            "overhead_start": self.overhead_s,
            "profile_start": dict(self.profile),
            "physx_start": self.physx_steps,
            "overflow_start": (self.contacts_reader.overflow_steps, self.contacts_reader.overflow_points)
            if self.contacts_reader else None,
        }
        self.monitors: dict[int, Any] = {}
        self.segment_info: dict[int, dict[str, Any]] = {}
        self.segment_contacts: dict[int, ContactLog] = {}
        self.segment_counter: dict[int, int] = {}

    def _open_all_segments(self) -> None:
        self.run["started_substep"] = self.substep + 1
        if self.record:
            for env_id in range(self.scene.num_envs):
                self._open_segment(env_id)

    def _open_segment(self, env_id: int) -> None:
        if not self.record:
            return
        from ...core.monitor import MonitorConfig, SubstepMonitor
        from ...core.recorder import EpisodeRecorder
        from ...presets import default_detectors

        k = self.segment_counter.get(env_id, 0)
        self.segment_counter[env_id] = k + 1
        seg_dir = self.out / f"run_{self.run['run']}" / f"env_{env_id}" / f"seg_{k}"
        cfg = MonitorConfig.from_dict({**default_monitor_config(), **self.monitor_cfg})
        meta = {
            **self.metadata,
            "task": self.task_name,
            "run": self.run["run"],
            "env": env_id,
            "segment": k,
            "batch_substep_offset": self.substep,  # local substep s = batch substep offset + s
            "num_envs": self.scene.num_envs,
        }
        monitor = SubstepMonitor(
            RobolabEnvAdapter(self, env_id),
            default_detectors(self.detector_cfg),
            episode_id=f"{self.task_name}:run{self.run['run']}:env{env_id}:seg{k}",
            config=cfg,
            recorder=EpisodeRecorder(seg_dir, trace_decimation=cfg.trace_decimation),
            metadata=meta,
        )
        self.monitors[env_id] = monitor
        self.segment_contacts[env_id] = ContactLog()
        self.segment_info[env_id] = {"env": env_id, "segment": k, "dir": str(seg_dir), "start_substep": self.substep + 1}
        self.cache = self.scene.read()
        self.episode_length = self.env.episode_length_buf.cpu().numpy()
        monitor.attach()

    def _close_segment(self, env_id: int, kind: str, event: dict[str, Any] | None) -> None:
        monitor = self.monitors.pop(env_id)
        info = self.segment_info.pop(env_id)
        contacts = self.segment_contacts.pop(env_id)
        summary = monitor.finalize()
        seg_dir = Path(info["dir"])
        contacts.save(seg_dir / "contacts.npz")
        info.update({
            "end_substep": self.substep,
            "end": kind,
            "discarded": kind in ("artifact", "reset"),
            "success": event.get("success") if event else None,
            "episode_length": event.get("episode_length") if event else None,
            "confirmed": summary.get("confirmed_count"),
            "flags": summary.get("flag_count"),
            "events_by_detector": summary.get("event_counts_by_detector"),
            "monitor_errors": summary.get("error_count"),
            "bytes": _dir_bytes(seg_dir),
        })
        self.run["segments"].append(info)

    def end_run(self, results: Any = None, timing: dict[str, Any] | None = None) -> dict[str, Any]:
        run = self.run
        for env_id in list(self.monitors):
            self._close_segment(env_id, "run_end", None)
        batch = self.out / "batch"
        writes = self.writes.since(run["write_start"])
        self._next_write = len(self.writes)
        self._next_substep = self.substep + 1
        k = run["run"]
        meta = {"task": self.task_name, "run": k, "num_envs": self.scene.num_envs, "first_substep": run["first_substep"],
                "last_substep": self.substep, "unknown_views": self.capture.unknown}
        if self.record:
            WriteLog.save(writes, batch / f"run_{k}_writes.npz", meta)
            np.savez_compressed(batch / f"run_{k}_robot.npz",
                                first_substep=np.int64(run["started_substep"] or run["first_substep"]),
                                dof_pos=np.asarray(run["robot_q"]), dof_vel=np.asarray(run["robot_qd"]),
                                dof_names=np.array(json.dumps(self.scene.dof_names)))
        wall = time.time() - run["wall_start"]
        steps = self.substep - run["first_substep"] + 1
        report = {
            **{key: run[key] for key in ("run", "first_substep", "started_substep", "resets", "segments")},
            "last_substep": self.substep,
            "physics_steps": steps,
            "physx_step_events": self.physx_steps - run["physx_start"],
            "writes": len(writes),
            "write_methods": _count(w.method for w in writes),
            "unknown_views": self.capture.unknown,
            "apply_force_calls": sum(1 for w in writes if w.method.startswith("apply_")),
            "contact_overflow": {"steps": self.contacts_reader.overflow_steps - run["overflow_start"][0],
                                 "points": self.contacts_reader.overflow_points - run["overflow_start"][1]}
            if self.contacts_reader else None,
            "caps_log": self.caps.log,
            "timing": {
                "wall_s": wall,
                "physics_s": run["physics_s"],
                "recorder_overhead_s": self.overhead_s - run["overhead_start"],
                "recorder_overhead_per_substep_ms": 1000.0 * (self.overhead_s - run["overhead_start"]) / max(steps, 1),
                "recorder_profile_s": {k: v - run["profile_start"][k] for k, v in self.profile.items()},
                "robolab": timing,
            },
            "results": results,
        }
        report["storage_bytes"] = {
            "segments": sum(s["bytes"] for s in run["segments"]),
            "batch": sum(_file_bytes(batch / f) for f in (f"run_{k}_writes.npz", f"run_{k}_robot.npz")),
        }
        (batch / f"run_{k}.json").write_text(json.dumps(report, indent=2, default=str))
        self.run = None
        return report

    def ground_truth(self, env_id: int) -> dict[str, Any]:
        return {
            "task": self.task_name,
            "env": env_id,
            "num_envs": self.scene.num_envs,
            "roles": {k: v.value for k, v in self.scene.roles.items()},
            "termination_params": self.scene.termination_params,
            "contact_views": [v[0] for v in self.contacts_reader.views] if self.contacts_reader else [],
            "caps": [c.to_dict() for c in self.caps.caps if c.envs is None or env_id in c.envs],
        }


def default_monitor_config() -> dict[str, Any]:
    """Full per-substep record per env; bundles off (replay is batch-level), only the initial snapshot."""

    return {
        "track_roles": ["target", "container", "object", "scene", "robot"],
        "state_log_roles": ["target", "container", "object", "scene"],
        "snapshot_interval_substeps": 10**9,
        "snapshot_capacity": 2,
        "bundle_statuses": [],
        "frame_buffer_substeps": 1,
        "control_log_maxlen": None,
        "trace_decimation": 8,
    }


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def _file_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _count(items: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return out


def physics_summary(env: Any) -> dict[str, Any]:
    import carb
    import omni.usd

    cfg = env.cfg.sim
    out: dict[str, Any] = {
        "dt": float(cfg.dt),
        "render_interval": int(cfg.render_interval),
        "device": str(cfg.device),
        "use_fabric": bool(cfg.use_fabric),
        "replicate_physics": bool(env.cfg.scene.replicate_physics),
        "physx_cfg": {k: getattr(cfg.physx, k) for k in dir(cfg.physx) if not k.startswith("_") and not callable(getattr(cfg.physx, k))},
    }
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(cfg.physics_prim_path)
    if prim and prim.IsValid():
        out["scene_attrs"] = {a.GetName(): _plain(a.Get()) for a in prim.GetAttributes() if a.GetName().startswith("physxScene:")}
    bodies = {}
    for name, obj in env.scene.rigid_objects.items():
        bodies[name] = _physx_attrs(stage.GetPrimAtPath(str(obj.root_physx_view.prim_paths[0])))
    for name, art in env.scene.articulations.items():
        bodies[name] = _physx_attrs(stage.GetPrimAtPath(str(art.root_physx_view.prim_paths[0])))
    out["env0_body_attrs"] = bodies
    settings = carb.settings.get_settings()
    out["carb"] = {k: settings.get(k) for k in ("/physics/disableContactProcessing", "/physics/fabricEnabled",
                                                 "/physics/suppressReadback", "/physics/cudaDevice")}
    return out


def _physx_attrs(prim: Any) -> dict[str, Any]:
    if not prim or not prim.IsValid():
        return {}
    keep = ("physxRigidBody:", "physxArticulation:", "physics:kinematicEnabled", "physics:rigidBodyEnabled")
    return {a.GetName(): _plain(a.Get()) for a in prim.GetAttributes() if a.GetName().startswith(keep)}


def _plain(value: Any) -> Any:
    return value if isinstance(value, (bool, int, float, str, type(None))) else str(value)


# ---------------------------------------------------------------------------- replay
def replay_batch(
    env: Any,
    record_dir: str | Path,
    *,
    runs: list[int],
    caps: list[DepenetrationCap] | None = None,
    cap_via: str = "usd",
    render: bool = False,
    out_dir: str | Path | None = None,
    save_states: bool = False,
    save_contacts: bool = False,
    on_step: Callable[[int, dict[str, np.ndarray]], None] | None = None,
) -> dict[str, Any]:
    """Re-apply the recorded writes of ``runs`` (from run 0 on) step by step in a fresh env.

    The env must be built like the recorded one (task, seed, num_envs) and not stepped yet.
    Compares the initial state, then per physics step every recorded segment's states.npz
    (objects of each env) and the robot joints of every run.  After every physics step the
    scene buffers are updated as ``env.step`` does, and at every policy-step boundary RoboLab's
    ``success`` termination term is evaluated, so each segment gets a replayed outcome next to
    the recorded one.  ``save_states`` / ``save_contacts`` write the replayed object states and
    per-env contacts of the checked runs to ``out_dir``.
    """

    record_dir = Path(record_dir)
    out_dir = Path(out_dir) if out_dir is not None else None
    batch = record_dir / "batch"
    scene = BatchScene(env)
    controller = CapController(scene, caps or [], cap_via)
    contacts_reader = BatchContacts(scene) if save_contacts else None
    views = {f"articulation:{n}": v for n, v in scene.articulations.items()}
    views.update({f"rigid:{n}": env.scene.rigid_objects[n].root_physx_view for n in scene.objects})
    views["sim_view"] = scene.sim_view
    success_term = _success_term(env)
    decimation = int(env.cfg.decimation)
    report: dict[str, Any] = {"runs": {}, "caps": [c.to_dict() for c in controller.caps], "cap_via": cap_via, "render": render,
                              "success_term": success_term is not None}

    initial = np.load(batch / "initial_states.npz")
    cache = scene.read()
    q, qd = scene.read_robot_joints()
    from .robolab_io import compare_state_arrays

    report["initial"] = {
        "objects": compare_state_arrays(initial["objects"], cache["objects"]),
        "robot_links": compare_state_arrays(initial["robot_links"], cache[f"links:{scene.robot}"]),
        "dof_pos": compare_state_arrays(initial["dof_pos"], q),
    }
    substep = 0
    last = max(runs)
    for k in range(last + 1):
        writes, meta = WriteLog.load(batch / f"run_{k}_writes.npz")
        run_info = json.loads((batch / f"run_{k}.json").read_text())
        by_step: dict[int, list[Any]] = {}
        for w in writes:
            by_step.setdefault(w.substep, []).append(w)
        check = k in runs
        refs = _segment_refs(run_info, scene, record_dir) if check else []
        robot = dict(np.load(batch / f"run_{k}_robot.npz")) if check else None
        started = run_info.get("started_substep") or meta["first_substep"]
        stats = {"objects": _Diff(), "dof_pos": _Diff(), "first_divergence": None}
        saved: list[np.ndarray] = []
        saved_steps: list[int] = []
        contact_logs = {e: ContactLog() for e in range(scene.num_envs)} if (check and contacts_reader) else {}
        t0 = time.time()
        for s in range(meta["first_substep"], meta["last_substep"] + 2):
            for w in by_step.get(s, []):
                _apply_write(views, w, scene)
            if s > meta["last_substep"]:
                break
            controller.before_step(s)
            env.sim.step(render=False)
            if render and s % int(env.cfg.sim.render_interval) == 0:
                env.sim.render()
            env.scene.update(dt=scene.dt)
            substep = s
            if not check:
                continue
            cache = scene.read()
            if on_step is not None:
                on_step(s, cache)
            if save_states:
                saved.append(cache["objects"])
                saved_steps.append(s)
            if contacts_reader is not None:
                for env_id, pairs in contacts_reader.read().items():
                    contact_logs[env_id].append(s, pairs)
            boundary = s >= started and (s - started + 1) % decimation == 0
            success = _success_now(env, success_term) if (boundary and success_term is not None) else None
            for ref in refs:
                # RoboLab ends an episode on success only after two policy steps (earlier ends are reset as artifacts)
                if success is not None and ref["start"] + 2 * decimation <= s <= ref["end"] \
                        and ref["replayed_success_substep"] is None and success[ref["env"]]:
                    ref["replayed_success_substep"] = s
                local = s - ref["offset"]
                row = ref["index"].get(local)
                if row is None:
                    continue
                got = cache["objects"][ref["env"]][ref["columns"]]
                want = ref["states"][row]
                if stats["objects"].add(want, got):
                    if ref["first_divergence"] is None:
                        ref["first_divergence"] = {"substep": s, **_divergence_detail(want, got, ref["body_ids"])}
                    if stats["first_divergence"] is None:
                        stats["first_divergence"] = {"substep": s, "env": ref["env"], "segment": ref["segment"],
                                                     "max_abs_diff": float(np.max(np.abs(want - got)))}
            if robot is not None:
                i = s - int(robot["first_substep"])
                if 0 <= i < len(robot["dof_pos"]):
                    stats["dof_pos"].add(robot["dof_pos"][i], scene.read_robot_joints()[0])
        if check:
            report["runs"][k] = {
                "substeps": [meta["first_substep"], meta["last_substep"]],
                "objects": stats["objects"].to_dict(),
                "dof_pos": stats["dof_pos"].to_dict(),
                "first_divergence": stats["first_divergence"],
                "segments_compared": len(refs),
                "segments": [_segment_outcome(ref) for ref in refs],
                "wall_s": time.time() - t0,
            }
            if out_dir is not None and save_states:
                np.savez_compressed(out_dir / f"run_{k}_replayed_objects.npz", substeps=np.asarray(saved_steps),
                                    objects=np.asarray(saved), body_ids=np.array(json.dumps(scene.objects)))
            for env_id, log in contact_logs.items():
                if out_dir is not None and len(log):
                    log.save(out_dir / f"run_{k}_env_{env_id}_replayed_contacts.npz")
    report["caps_log"] = controller.log
    report["last_substep"] = substep
    report["physics"] = physics_summary(env)
    if contacts_reader is not None:
        report["contact_overflow"] = {"steps": contacts_reader.overflow_steps, "points": contacts_reader.overflow_points}
    return report


def _success_term(env: Any) -> Any:
    manager = getattr(env, "termination_manager", None)
    if manager is None or "success" not in manager.active_terms:
        return None
    return manager.get_term_cfg("success")


def _success_now(env: Any, term: Any) -> np.ndarray:
    value = term.func(env, **term.params)
    return np.asarray(value.cpu().numpy() if hasattr(value, "cpu") else value, dtype=bool).reshape(-1)


def _divergence_detail(want: np.ndarray, got: np.ndarray, body_ids: list[str]) -> dict[str, Any]:
    diff = np.abs(np.asarray(want, dtype=np.float64) - np.asarray(got, dtype=np.float64))
    groups = {"pos": slice(0, 3), "quat": slice(3, 7), "lin_vel": slice(7, 10), "ang_vel": slice(10, 13)}
    bodies = {}
    for i, body in enumerate(body_ids):
        row = {g: float(diff[i, sl].max()) for g, sl in groups.items() if diff[i, sl].max() > 0}
        if row:
            bodies[body] = row
    return {"max_abs_diff": float(diff.max()), "bodies": bodies}


def _segment_outcome(ref: dict[str, Any]) -> dict[str, Any]:
    replayed = ref["replayed_success_substep"] is not None
    return {
        "env": ref["env"],
        "segment": ref["segment"],
        "substeps": [ref["start"], ref["end"]],
        "end": ref["end_kind"],
        "discarded": ref["discarded"],
        "recorded_success": ref["recorded_success"],
        "recorded_confirmed": ref["confirmed"],
        "recorded_flags": ref["flags"],
        "replayed_success": replayed,
        "replayed_success_substep": ref["replayed_success_substep"],
        "outcome_flipped": None if ref["recorded_success"] is None or ref["discarded"] else replayed != bool(ref["recorded_success"]),
        "first_divergence": ref["first_divergence"],
    }


class _Diff:
    def __init__(self) -> None:
        self.values = 0
        self.differing = 0
        self.max_abs = 0.0

    def add(self, want: np.ndarray, got: np.ndarray) -> bool:
        want = np.asarray(want, dtype=np.float64)
        got = np.asarray(got, dtype=np.float64)
        same = (want == got) | (np.isnan(want) & np.isnan(got))
        n = int(np.size(same) - np.count_nonzero(same))
        self.values += int(np.size(same))
        self.differing += n
        if n:
            self.max_abs = max(self.max_abs, float(np.nanmax(np.abs(want - got))))
        return n > 0

    def to_dict(self) -> dict[str, Any]:
        return {"values": self.values, "differing_values": self.differing, "max_abs_diff": self.max_abs,
                "bit_exact": self.differing == 0 and self.values > 0}


def _segment_refs(run_info: dict[str, Any], scene: BatchScene, record_dir: Path) -> list[dict[str, Any]]:
    from .robolab_io import segment_dir

    refs = []
    for seg in run_info["segments"]:
        path = segment_dir(record_dir, seg) / "states.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            body_ids = json.loads(str(data["body_ids"])) if "body_ids" in data.files else None
            substeps = data["substeps"]
            states = data["states"]
        if body_ids is None:
            continue
        refs.append({
            "env": seg["env"],
            "segment": seg["segment"],
            "offset": seg["start_substep"] - 1,
            "start": seg["start_substep"],
            "end": seg["end_substep"],
            "end_kind": seg.get("end"),
            "discarded": seg.get("discarded"),
            "recorded_success": seg.get("success"),
            "confirmed": seg.get("confirmed"),
            "flags": seg.get("flags"),
            "body_ids": body_ids,
            "columns": [scene.objects.index(b) for b in body_ids],
            "states": states,
            "index": {int(s): i for i, s in enumerate(substeps.tolist())},
            "first_divergence": None,
            "replayed_success_substep": None,
        })
    return refs


def _apply_write(views: dict[str, Any], write: Any, scene: BatchScene) -> None:
    import torch

    view = views.get(write.target)
    if view is None:
        raise KeyError(f"write to unknown view {write.target!r}: not replayable")
    fn = getattr(view, write.method)
    if write.method in ("update_articulations_kinematic", "clear_forces", "flush"):
        fn()
        return
    if write.method.startswith("apply_"):
        raise RuntimeError(f"recorded {write.method} without data: not replayable")
    device = scene.env.device
    count = int(view.count)
    data = write.data
    full = np.zeros((count,) + data.shape[1:], dtype=data.dtype)
    idx = write.indices.astype(np.int64)
    full[idx] = data
    if write.method in _CPU_SETTERS:
        fn(torch.from_numpy(full), torch.from_numpy(write.indices.copy()))
    else:
        fn(torch.from_numpy(full).to(device), torch.from_numpy(write.indices.copy()).to(device))


# Isaac Lab passes these property buffers on the CPU (the tensor API reads them on the host).
_CPU_SETTERS = {
    "set_masses", "set_coms", "set_inertias", "set_material_properties", "set_contact_offsets", "set_rest_offsets",
    "set_disable_gravities", "set_disable_simulations", "set_dof_limits", "set_dof_stiffnesses", "set_dof_dampings",
    "set_dof_max_forces", "set_dof_max_velocities", "set_dof_armatures", "set_dof_friction_coefficients",
    "set_dof_friction_properties", "set_fixed_tendon_properties", "set_spatial_tendon_properties",
}
