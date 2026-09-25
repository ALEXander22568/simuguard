"""SimuGuard adapter for NVIDIA Isaac Sim / Isaac Lab (PhysX 5, CPU or GPU pipeline).

Works on any Isaac Sim stage; :mod:`.robodojo` builds the body list for RoboDojo.

Ground truth
    Poses and centre-of-mass velocities of every tracked rigid body and articulation link
    come from one PhysX tensor view (``omni.physics.tensors`` rigid-body view), read right
    after each physics step.  Contacts come from the omni.physx contact report
    (``get_contact_report``): one header per shape pair with per-point position, normal,
    separation and the normal impulse of that step.  PhysX only reports pairs in which at
    least one actor carries ``PhysxContactReportAPI``; :func:`enable_contact_reporting`
    adds it (threshold 0) and must run before PhysX parses the prims, otherwise the change
    re-creates the actor.  Pairs without a watched body are never reported, which is what
    the detectors need (object-object, object-robot, object-scene).

Actuation
    Articulation drives are captured right before every physics step through tensor
    articulation views: position targets, velocity targets and joint actuation forces,
    i.e. exactly what Isaac Lab's ``write_data_to_sim`` handed to PhysX for that step.
    Anything written to body poses/velocities or joint states between two steps (task code
    teleporting an object) is detected by comparing the pre-step state with the state the
    previous step left and is recorded as well; replay writes it back.

Replay
    Isaac Sim exposes no solver-state serialization for the PhysX scene (contact caches,
    warm-start impulses, GPU buffers), so there is no native snapshot.  The public snapshot
    (all body poses and velocities, joint positions/velocities/targets) restores in place
    but not bit-exactly in contact.  Exact replay, where the platform is deterministic,
    comes from rebuilding the scene the same way and replaying the recorded per-substep
    actuation from the episode start (restore method ``none``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from ...core.adapter import AdapterCapabilities, HookHandle, SimAdapter, SubstepCallback
from ...core.snapshot import ControlRecord, RestoreReport, Snapshot, compare_states
from ...core.types import BodyInfo, BodyKind, BodyRole, BodyState, ContactPair
from .physx_io import PathResolver, changed_entries, merge_contact_records, states_from_arrays

WATCHED_ROLES = (BodyRole.TARGET, BodyRole.CONTAINER, BodyRole.OBJECT)


@dataclass
class BodySpec:
    """One body the adapter should know about."""

    body_id: str
    path: str  # rigid-body / link prim path; for STATIC bodies the root prim of the collider group
    kind: BodyKind
    role: BodyRole
    articulation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArticulationSpec:
    art_id: str
    root_path: str  # prim carrying ArticulationRootAPI
    is_robot: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.array(value.numpy(), copy=True)
    return np.array(value, copy=True)


class IsaacAdapter(SimAdapter):
    """Ground truth, substep hook, actuation capture and public snapshots for an Isaac Sim stage."""

    name = "isaac"

    def __init__(
        self,
        sim_context: Any,
        bodies: list[BodySpec],
        articulations: list[ArticulationSpec],
        *,
        task_name: str = "",
        control_step_fn: Callable[[], int] | None = None,
        ground_truth_fn: Callable[[], dict[str, Any]] | None = None,
        contact_reporting: bool = True,
        watched_contacts_only: bool = True,
        detect_between_step_writes: bool = True,
        physics_step_counter: bool = True,
    ) -> None:
        self.sim = sim_context
        self.task_name = task_name
        self._control_step_fn = control_step_fn
        self._ground_truth_fn = ground_truth_fn
        self.contact_reporting = contact_reporting
        self.watched_contacts_only = watched_contacts_only
        self.detect_between_step_writes = detect_between_step_writes
        self._specs = {spec.body_id: spec for spec in bodies}
        self._arts = {spec.art_id: spec for spec in articulations}
        self._hook: HookHandle | None = None
        self._raw_step: Callable[..., Any] | None = None
        self._internal_step = 0
        self._post_step: dict[str, np.ndarray] | None = None
        self._between_step_writes = 0
        self._physics_steps_seen = 0
        self._hooked_steps = 0
        self._step_subscription: Any = None
        self._timing = {
            "read_states_s": 0.0, "read_contacts_s": 0.0, "capture_control_s": 0.0,
            "hook_before_s": 0.0, "physics_step_s": 0.0, "hook_after_s": 0.0,
        }
        self._rb_cache: tuple[np.ndarray, np.ndarray] | None = None  # last read, valid until a step or write
        self._contact_stats = {"reads": 0, "headers": 0, "points": 0}
        self._build_views()
        if physics_step_counter:
            self._subscribe_step_counter()
        self._bodies = self._build_inventory()
        watched = {b for b, info in self._bodies.items() if info.role in WATCHED_ROLES}
        self._watched = watched
        self.resolver = self._build_resolver()

    # ------------------------------------------------------------------ views
    @property
    def sim_view(self) -> Any:
        from isaacsim.core.simulation_manager import SimulationManager

        view = SimulationManager.get_physics_sim_view()
        if view is None:
            raise RuntimeError("physics simulation view is not initialised (is the simulation playing?)")
        return view

    def _build_views(self) -> None:
        sim_view = self.sim_view
        # one rigid-body view over every dynamic/kinematic body and articulation link
        self._rb_ids: list[str] = []
        paths = [s.path for s in self._specs.values() if s.kind != BodyKind.STATIC]
        self._rb_view = None
        if paths:
            view = sim_view.create_rigid_body_view(paths)
            order = list(view.prim_paths)
            by_path = {s.path: s.body_id for s in self._specs.values() if s.kind != BodyKind.STATIC}
            missing = sorted(set(paths) - set(order))
            if missing:
                raise RuntimeError(f"rigid-body view did not resolve {len(missing)} bodies, e.g. {missing[:3]}")
            self._rb_view = view
            self._rb_ids = [by_path[p] for p in order]
        self._rb_row = {body_id: i for i, body_id in enumerate(self._rb_ids)}
        self._free_rows = np.array(
            [i for i, b in enumerate(self._rb_ids) if self._specs[b].kind in (BodyKind.DYNAMIC, BodyKind.KINEMATIC)],
            dtype=np.int64,
        )
        # one articulation view per articulation (robots and articulated objects)
        self._art_views: dict[str, Any] = {}
        self._art_dofs: dict[str, list[str]] = {}
        for art_id, spec in self._arts.items():
            view = sim_view.create_articulation_view(spec.root_path)
            if view.count != 1:
                raise RuntimeError(f"articulation view for {spec.root_path} resolved {view.count} articulations")
            self._art_views[art_id] = view
            self._art_dofs[art_id] = list(view.shared_metatype.dof_names)
        self._index0 = self._index_tensor([0])

    def _index_tensor(self, rows: list[int]) -> Any:
        import torch

        return torch.tensor(rows, dtype=torch.int32, device=self._device())

    def _device(self) -> str:
        device = getattr(self.sim, "device", None) or "cpu"
        return str(device)

    def _tensor(self, array: np.ndarray) -> Any:
        import torch

        return torch.tensor(np.asarray(array), dtype=torch.float32, device=self._device())

    def _subscribe_step_counter(self) -> None:
        """Count every PhysX step, to prove that the hook saw all of them."""

        try:
            import omni.physx

            iface = omni.physx.get_physx_interface()

            def on_step(_dt: float) -> None:
                self._physics_steps_seen += 1

            self._step_subscription = iface.subscribe_physics_step_events(on_step)
        except Exception:  # noqa: BLE001 - diagnostics only
            self._step_subscription = None

    # ------------------------------------------------------------------ inventory
    def _build_inventory(self) -> dict[str, BodyInfo]:
        masses: dict[str, float] = {}
        if self._rb_view is not None:
            try:
                m = _to_numpy(self._rb_view.get_masses()).reshape(len(self._rb_ids), -1)[:, 0]
                masses = {b: float(m[i]) for i, b in enumerate(self._rb_ids)}
            except Exception:  # noqa: BLE001
                masses = {}
        out: dict[str, BodyInfo] = {}
        for body_id, spec in self._specs.items():
            out[body_id] = BodyInfo(
                body_id=body_id,
                name=spec.path.rsplit("/", 1)[-1],
                kind=spec.kind,
                role=spec.role,
                mass=masses.get(body_id),
                articulation=spec.articulation,
                collision_shape_count=int(spec.metadata.get("collision_shapes", 0)),
                metadata={"path": spec.path, **spec.metadata},
            )
        return out

    def _build_resolver(self) -> PathResolver:
        exact = {s.path: s.body_id for s in self._specs.values() if s.kind != BodyKind.STATIC}
        prefixes = {s.path: s.body_id for s in self._specs.values() if s.kind == BodyKind.STATIC}

        def fallback(path: str) -> str:
            parts = [p for p in path.split("/") if p]
            return "static:" + "/".join(parts[-2:]) if parts else "static:?"

        return PathResolver(exact=exact, prefixes=prefixes, fallback=fallback)

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            substep_hook=True,
            contacts=self.contact_reporting,
            public_state_snapshot=True,
            native_state_snapshot=False,
            control_capture=True,
            notes=[
                "exact: poses/CoM velocities from a PhysX rigid-body tensor view read after every step",
                "exact: actuation = articulation drive position/velocity targets + joint actuation forces "
                "read from PhysX right before every step, plus any between-step state writes",
                "contacts: omni.physx contact report (normal impulse per point); only pairs where one body "
                "carries PhysxContactReportAPI are reported; friction impulses are not included",
                "approximate: public snapshot restore (PhysX contact caches / warm start not restorable); "
                "no native snapshot exists in Isaac Sim",
                "exact replay only by rebuilding the scene identically and replaying recorded actuation "
                "from the episode start (restore method 'none'), if the PhysX pipeline is deterministic",
            ],
        )

    def timestep(self) -> float:
        return float(self.sim.get_physics_dt())

    def bodies(self, refresh: bool = False) -> dict[str, BodyInfo]:
        return dict(self._bodies)

    def body_ids_with_role(self, role: BodyRole) -> list[str]:
        return sorted(b for b, info in self._bodies.items() if info.role == role)

    def control_step(self) -> int:
        return int(self._control_step_fn()) if self._control_step_fn else self._internal_step

    def task_ground_truth(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task_name": self.task_name,
            "targets": self.body_ids_with_role(BodyRole.TARGET),
            "containers": self.body_ids_with_role(BodyRole.CONTAINER),
            "timestep": self.timestep(),
        }
        if self._ground_truth_fn is not None:
            try:
                payload.update(self._ground_truth_fn())
            except Exception as exc:  # noqa: BLE001
                payload["ground_truth_error"] = f"{type(exc).__name__}: {exc}"
        return payload

    # ------------------------------------------------------------------ ground truth
    def _read_rb(self, fresh: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Poses (xyzw) and velocities of the rigid-body view; cached until the next step or write."""
        if self._rb_view is None:
            return np.zeros((0, 7)), np.zeros((0, 6))
        if fresh or self._rb_cache is None:
            poses = _to_numpy(self._rb_view.get_transforms()).reshape(-1, 7).astype(np.float64)
            vels = _to_numpy(self._rb_view.get_velocities()).reshape(-1, 6).astype(np.float64)
            self._rb_cache = (poses, vels)
        poses, vels = self._rb_cache
        return poses.copy(), vels.copy()

    def _invalidate(self) -> None:
        self._rb_cache = None

    def read_states(self, body_ids: Iterable[str] | None = None) -> dict[str, BodyState]:
        started = time.perf_counter()
        poses, vels = self._read_rb()
        ids = self._rb_ids if body_ids is None else [b for b in body_ids if b in self._rb_row]
        rows = [self._rb_row[b] for b in ids]
        out = states_from_arrays(ids, poses, vels, rows)
        self._timing["read_states_s"] += time.perf_counter() - started
        return out

    def read_contacts(self) -> list[ContactPair]:
        if not self.contact_reporting:
            return []
        started = time.perf_counter()
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools

        headers, data = get_physx_simulation_interface().get_contact_report()
        self._timing["contact_report_call_s"] = self._timing.get("contact_report_call_s", 0.0) + time.perf_counter() - started
        self._contact_stats["headers"] += len(headers)
        self._contact_stats["points"] += len(data)
        self._contact_stats["reads"] += 1

        def decode(key: Any) -> str:
            return str(PhysicsSchemaTools.intToSdfPath(key))

        def resolve(key: Any) -> str:
            return self.resolver.resolve(key, decode)

        keep = None
        if self.watched_contacts_only:
            watched = self._watched
            keep = lambda a, b: a in watched or b in watched  # noqa: E731
        pairs = merge_contact_records(headers, data, resolve, keep=keep)
        self._timing["read_contacts_s"] += time.perf_counter() - started
        return pairs

    # ------------------------------------------------------------------ stepping
    def install_substep_hook(self, after: SubstepCallback, before: SubstepCallback | None = None) -> HookHandle:
        if self._hook is not None and not self._hook.removed:
            raise RuntimeError("a SimuGuard substep hook is already installed")
        sim = self.sim
        original = sim.step
        self._raw_step = original

        def hooked_step(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            if before is not None:
                before()
            t1 = time.perf_counter()
            result = original(*args, **kwargs)
            t2 = time.perf_counter()
            self._invalidate()
            self._internal_step += 1
            self._hooked_steps += 1
            if self.detect_between_step_writes:
                self.mark_post_step()
            after()
            t3 = time.perf_counter()
            self._timing["hook_before_s"] += t1 - t0
            self._timing["physics_step_s"] += t2 - t1
            self._timing["hook_after_s"] += t3 - t2
            return result

        if self.detect_between_step_writes:
            self.mark_post_step()
        sim.step = hooked_step  # instance attribute shadows the class method

        def remove() -> None:
            try:
                del sim.step
            except AttributeError:
                pass

        self._hook = HookHandle(remove_fn=remove, mechanism="simulation_context_step_instance_attribute")
        return self._hook

    def step_physics(self) -> None:
        """One raw physics step (no rendering), bypassing the hook: used by replay."""

        step = self._raw_step or type(self.sim).step.__get__(self.sim)
        step(render=False)
        self._invalidate()
        self._internal_step += 1
        if self.detect_between_step_writes:
            self.mark_post_step()

    def hook_statistics(self) -> dict[str, Any]:
        return {
            "hooked_steps": self._hooked_steps,
            "physics_steps_seen": self._physics_steps_seen if self._step_subscription is not None else None,
            "between_step_writes": self._between_step_writes,
            "unknown_contact_actors": dict(list(self.resolver.unknown.items())[:50]),
            "contact_report": dict(self._contact_stats),
            "timing_s": dict(self._timing),
        }

    # ------------------------------------------------------------------ state vectors
    def _articulation_state(self, art_id: str) -> dict[str, np.ndarray]:
        view = self._art_views[art_id]
        return {
            "q": _to_numpy(view.get_dof_positions()).reshape(-1),
            "qd": _to_numpy(view.get_dof_velocities()).reshape(-1),
            "root_pose": _to_numpy(view.get_root_transforms()).reshape(-1),
            "root_vel": _to_numpy(view.get_root_velocities()).reshape(-1),
        }

    def _state_vectors(self, fresh: bool = False) -> dict[str, np.ndarray]:
        poses, vels = self._read_rb(fresh=fresh)
        state = {
            "free_pose": poses[self._free_rows].reshape(-1) if len(self._free_rows) else np.zeros(0),
            "free_vel": vels[self._free_rows].reshape(-1) if len(self._free_rows) else np.zeros(0),
        }
        for art_id in self._art_views:
            for key, value in self._articulation_state(art_id).items():
                state[f"{art_id}|{key}"] = value
        return state

    def mark_post_step(self) -> None:
        """Remember the state a physics step left, to catch writes made between steps."""

        self._post_step = self._state_vectors()

    # ------------------------------------------------------------------ actuation
    def capture_control(self) -> ControlRecord:
        started = time.perf_counter()
        payload: dict[str, Any] = {}
        for art_id, view in self._art_views.items():
            payload[art_id] = {
                "pos_target": _to_numpy(view.get_dof_position_targets()).reshape(-1).astype(np.float64),
                "vel_target": _to_numpy(view.get_dof_velocity_targets()).reshape(-1).astype(np.float64),
                "effort": _to_numpy(view.get_dof_actuation_forces()).reshape(-1).astype(np.float64),
            }
        if self.detect_between_step_writes and self._post_step is not None:
            now = self._state_vectors(fresh=True)  # task code may have moved things since the last read
            writes: dict[str, Any] = {}
            for key, value in now.items():
                then = self._post_step.get(key)
                if then is None:
                    continue
                idx = changed_entries(value, then)
                if idx.size:
                    writes[key] = {"idx": idx.astype(np.int64), "val": value[idx].astype(np.float64)}
            if writes:
                payload["__writes__"] = writes
                self._between_step_writes += 1
        self._timing["capture_control_s"] += time.perf_counter() - started
        return ControlRecord(substep=-1, control_step=self.control_step(), payload=payload)

    def apply_control(self, control: ControlRecord) -> None:
        payload = control.payload
        writes = payload.get("__writes__")
        if writes:
            self._apply_writes(writes)
        for art_id, data in payload.items():
            if art_id == "__writes__":
                continue
            view = self._art_views.get(art_id)
            if view is None:
                raise RuntimeError(f"control topology mismatch: missing articulation {art_id}")
            n = len(self._art_dofs[art_id])
            for key, setter in (
                ("pos_target", view.set_dof_position_targets),
                ("vel_target", view.set_dof_velocity_targets),
                ("effort", view.set_dof_actuation_forces),
            ):
                values = np.asarray(data[key], dtype=np.float32).reshape(1, -1)
                if values.shape[1] != n:
                    raise RuntimeError(f"control topology mismatch: {art_id}.{key} has {values.shape[1]} dofs, view {n}")
                setter(self._tensor(values), self._index0)

    def _apply_writes(self, writes: dict[str, Any]) -> None:
        current = self._state_vectors(fresh=True)
        for key, change in writes.items():
            if key not in current:
                raise RuntimeError(f"between-step write for unknown state vector {key}")
            vector = current[key].copy()
            vector[np.asarray(change["idx"], dtype=np.int64)] = np.asarray(change["val"], dtype=np.float64)
            self._write_state_vector(key, vector)

    def _write_state_vector(self, key: str, vector: np.ndarray) -> None:
        if key in ("free_pose", "free_vel"):
            if self._rb_view is None or not len(self._free_rows):
                return
            rows = self._index_tensor(self._free_rows.tolist())
            full = _to_numpy(self._rb_view.get_transforms() if key == "free_pose" else self._rb_view.get_velocities())
            width = 7 if key == "free_pose" else 6
            full = full.reshape(-1, width)
            full[self._free_rows] = vector.reshape(-1, width)
            data = self._tensor(full)
            self._invalidate()
            if key == "free_pose":
                self._rb_view.set_transforms(data, rows)
            else:
                self._rb_view.set_velocities(data, rows)
            return
        self._invalidate()
        art_id, field_name = key.split("|", 1)
        view = self._art_views[art_id]
        setter = {
            "q": view.set_dof_positions,
            "qd": view.set_dof_velocities,
            "root_pose": view.set_root_transforms,
            "root_vel": view.set_root_velocities,
        }[field_name]
        setter(self._tensor(vector.reshape(1, -1)), self._index0)

    # ------------------------------------------------------------------ snapshots
    def capture_public_state(self) -> dict[str, Any]:
        poses, vels = self._read_rb(fresh=True)
        bodies = {}
        for i, body_id in enumerate(self._rb_ids):
            if self._specs[body_id].kind in (BodyKind.DYNAMIC, BodyKind.KINEMATIC):
                bodies[body_id] = {"pose_xyzw": poses[i].tolist(), "vel": vels[i].tolist()}
        arts = {}
        for art_id, view in self._art_views.items():
            state = self._articulation_state(art_id)
            arts[art_id] = {
                "root_pose_xyzw": state["root_pose"].tolist(),
                "root_vel": state["root_vel"].tolist(),
                "q": state["q"].tolist(),
                "qd": state["qd"].tolist(),
                "pos_target": _to_numpy(view.get_dof_position_targets()).reshape(-1).tolist(),
                "vel_target": _to_numpy(view.get_dof_velocity_targets()).reshape(-1).tolist(),
            }
        return {"bodies": bodies, "articulations": arts}

    def capture_snapshot(self, substep: int, *, include_native: bool = True) -> Snapshot:
        control = self.capture_control()
        control.substep = int(substep)
        control.payload.pop("__writes__", None)
        return Snapshot(
            substep=int(substep),
            control_step=self.control_step(),
            public_state=self.capture_public_state(),
            control=control,
            native_state=None,
            native_format=None,
            metadata={"task_name": self.task_name, "timestep": self.timestep()},
        )

    def restore_snapshot(self, snapshot: Snapshot, *, method: str = "auto") -> RestoreReport:
        if method == "auto":
            method = "public"
        report = RestoreReport(method=method, substep=snapshot.substep)
        if method == "public":
            report.missing, report.mismatched = self._restore_public(snapshot.public_state)
        elif method != "none":
            raise ValueError(f"unknown restore method for Isaac Sim: {method} (no native snapshots)")
        self.apply_control(snapshot.control)
        self._internal_step = snapshot.substep
        if self.detect_between_step_writes:
            self.mark_post_step()
        report.public_state_error = compare_states(snapshot.public_state, self.capture_public_state())
        return report

    def _restore_public(self, state: dict[str, Any]) -> tuple[list[str], list[str]]:
        missing: list[str] = []
        mismatched: list[str] = []
        if self._rb_view is not None and len(self._free_rows):
            poses, vels = self._read_rb(fresh=True)
            for body_id, data in state.get("bodies", {}).items():
                row = self._rb_row.get(body_id)
                if row is None:
                    missing.append(body_id)
                    continue
                poses[row] = np.asarray(data["pose_xyzw"], dtype=np.float64)
                vels[row] = np.asarray(data["vel"], dtype=np.float64)
            rows = self._index_tensor(self._free_rows.tolist())
            self._rb_view.set_transforms(self._tensor(poses), rows)
            self._rb_view.set_velocities(self._tensor(vels), rows)
            self._invalidate()
        for art_id, data in state.get("articulations", {}).items():
            view = self._art_views.get(art_id)
            if view is None:
                missing.append(art_id)
                continue
            n = len(self._art_dofs[art_id])
            if len(data["q"]) != n:
                mismatched.append(art_id)
                continue
            view.set_root_transforms(self._tensor(np.asarray(data["root_pose_xyzw"]).reshape(1, -1)), self._index0)
            view.set_root_velocities(self._tensor(np.asarray(data["root_vel"]).reshape(1, -1)), self._index0)
            view.set_dof_positions(self._tensor(np.asarray(data["q"]).reshape(1, -1)), self._index0)
            view.set_dof_velocities(self._tensor(np.asarray(data["qd"]).reshape(1, -1)), self._index0)
            view.set_dof_position_targets(self._tensor(np.asarray(data["pos_target"]).reshape(1, -1)), self._index0)
            view.set_dof_velocity_targets(self._tensor(np.asarray(data["vel_target"]).reshape(1, -1)), self._index0)
        self._invalidate()
        return missing, mismatched

    def close(self) -> None:
        if self._hook is not None:
            self._hook.remove()
        self._step_subscription = None

    # ------------------------------------------------------------------ interventions
    def set_max_depenetration_velocity(self, value: float, body_ids: Iterable[str] | None = None) -> list[str]:
        """Set ``physxRigidBody:maxDepenetrationVelocity`` on live free bodies (USD property change).

        The bodies must already carry ``PhysxRigidBodyAPI`` (applying the schema to a simulated prim
        would re-create its actor): RoboDojo objects get it at creation with
        ``install_contact_reporting(physx_rigid_body_api=True)``.  Returns the bodies changed.
        """

        import omni.usd
        from pxr import PhysxSchema

        stage = omni.usd.get_context().get_stage()
        ids = [b for b, s in self._specs.items() if s.kind == BodyKind.DYNAMIC] if body_ids is None else list(body_ids)
        changed = []
        for body_id in ids:
            prim = stage.GetPrimAtPath(self._specs[body_id].path)
            if prim and prim.IsValid() and prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI(prim).GetMaxDepenetrationVelocityAttr().Set(float(value))
                changed.append(body_id)
        return changed

    # ------------------------------------------------------------------ diagnostics
    def set_body_velocity(self, body_id: str, linear: Iterable[float], angular: Iterable[float] = (0.0, 0.0, 0.0)) -> None:
        """Overwrite one free body's velocity between steps (positive controls and tests only)."""

        row = self._rb_row[body_id]
        vels = self._read_rb(fresh=True)[1]
        vels[row, :3] = np.asarray(list(linear), dtype=np.float64)
        vels[row, 3:] = np.asarray(list(angular), dtype=np.float64)
        self._invalidate()
        self._rb_view.set_velocities(self._tensor(vels), self._index_tensor([row]))

    def net_contact_force_reader(self, body_ids: Iterable[str]) -> Callable[[], dict[str, np.ndarray]]:
        """PhysX's own net contact force per body (rigid contact view), to cross-check read_contacts.

        Uses ``create_rigid_contact_view(...).get_net_contact_forces(dt)``, a different PhysX code path
        than the contact report.  Bodies must carry PhysxContactReportAPI.
        """

        ids = [b for b in body_ids if b in self._specs and self._specs[b].kind != BodyKind.STATIC]
        view = self.sim_view.create_rigid_contact_view([self._specs[b].path for b in ids], filter_patterns=[],
                                                       max_contact_data_count=0)
        try:
            count = view.sensor_count
        except AttributeError:  # the view has no backend when the bodies lack PhysxContactReportAPI
            count = None
        if not count:
            raise RuntimeError("rigid contact view unavailable (the bodies need PhysxContactReportAPI)")
        order = list(view.sensor_paths) if hasattr(view, "sensor_paths") else [self._specs[b].path for b in ids]
        by_path = {self._specs[b].path: b for b in ids}
        dt = self.timestep()

        def read() -> dict[str, np.ndarray]:
            forces = _to_numpy(view.get_net_contact_forces(dt)).reshape(-1, 3)
            return {by_path[p]: forces[i].astype(np.float64) for i, p in enumerate(order) if p in by_path}

        return read

    # ------------------------------------------------------------------ properties for reports
    def physx_body_properties(self, body_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        """Authored PhysX rigid-body properties (depenetration cap, solver iterations...)."""

        import omni.usd
        from pxr import PhysxSchema, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        out: dict[str, dict[str, Any]] = {}
        for body_id in body_ids:
            spec = self._specs.get(body_id)
            if spec is None:
                continue
            prim = stage.GetPrimAtPath(spec.path)
            if not prim or not prim.IsValid():
                continue
            props: dict[str, Any] = {}
            if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                api = PhysxSchema.PhysxRigidBodyAPI(prim)
                for name, attr in (
                    ("max_depenetration_velocity", api.GetMaxDepenetrationVelocityAttr()),
                    ("solver_position_iterations", api.GetSolverPositionIterationCountAttr()),
                    ("solver_velocity_iterations", api.GetSolverVelocityIterationCountAttr()),
                    ("sleep_threshold", api.GetSleepThresholdAttr()),
                    ("enable_ccd", api.GetEnableCCDAttr()),
                    ("max_linear_velocity", api.GetMaxLinearVelocityAttr()),
                ):
                    value = attr.Get() if attr and attr.HasAuthoredValue() else None
                    props[name] = None if value is None else (float(value) if not isinstance(value, bool) else value)
            else:
                props["physx_rigid_body_api"] = False
            mass_api = UsdPhysics.MassAPI(prim) if prim.HasAPI(UsdPhysics.MassAPI) else None
            if mass_api is not None:
                attr = mass_api.GetMassAttr()
                props["authored_mass"] = float(attr.Get()) if attr and attr.HasAuthoredValue() else None
            out[body_id] = props
        return out


def enable_contact_reporting(prim_paths: Iterable[str], threshold: float = 0.0) -> list[str]:
    """Add ``PhysxContactReportAPI`` to every rigid body at or below the given prims.

    Call it before PhysX parses the prims (object creation), not on a running body:
    adding an API schema to a simulated prim makes omni.physx re-create the actor.
    Also turns omni.physx contact processing on (Isaac Lab switches it off by default).
    Returns the rigid-body paths that now report contacts.
    """

    import carb
    import omni.usd
    from pxr import PhysxSchema, Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
    enabled: list[str] = []
    for root_path in prim_paths:
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            continue
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsInstanceProxy():
                continue
            api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            api.CreateThresholdAttr().Set(float(threshold))
            enabled.append(str(prim.GetPath()))
    return enabled


def scan_rigid_bodies(root_path: str) -> dict[str, list[dict[str, Any]]]:
    """USD scan below ``root_path``: rigid bodies, articulation roots and static collider prims."""

    import omni.usd
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    result: dict[str, list[dict[str, Any]]] = {"rigid": [], "articulation_roots": [], "colliders": []}
    if not root or not root.IsValid():
        return result
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            result["articulation_roots"].append({"path": path})
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api = UsdPhysics.RigidBodyAPI(prim)
            enabled = api.GetRigidBodyEnabledAttr().Get()
            kinematic = api.GetKinematicEnabledAttr().Get()
            result["rigid"].append(
                {"path": path, "enabled": True if enabled is None else bool(enabled), "kinematic": bool(kinematic)}
            )
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            result["colliders"].append({"path": path})
    return result
