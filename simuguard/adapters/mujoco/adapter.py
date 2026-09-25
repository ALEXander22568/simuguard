"""SimuGuard adapter for MuJoCo (raw ``mujoco.MjModel`` / ``mujoco.MjData``).

Works on any MuJoCo scene; :func:`attach_robosuite` wires it into a robosuite / RoboCasa environment.

Ground truth
    Body poses from ``data.xpos`` / ``data.xquat`` (wxyz), velocities from ``mj_objectVelocity``
    shifted to the body's centre of mass.  Contacts come from ``data.contact`` only: equality
    constraints such as welds are not contacts and are never reported (their constraint forces
    are what a naive read of ``efc_force`` mistakes for contact forces).  Each active contact
    contributes one point with its penetration (``dist < 0``), normal, and the impulse of its
    force from ``mj_contactForce`` over the timestep.

Replay
    MuJoCo exposes its whole integration state (``mjSTATE_INTEGRATION``: time, qpos, qvel,
    actuator activations, warm-start accelerations, plugin state, mocap poses, user data, ctrl,
    applied forces, equality activation).  A snapshot stores it natively, so restoring it and
    re-applying the recorded actuation reproduces the continuation exactly, unlike PhysX, whose
    contact caches cannot be restored.  The actuation captured before every step is ``ctrl`` plus
    whatever applied forces, mocap poses or equality switches are in use, plus every ``qpos`` /
    ``qvel`` / ``act`` entry that code outside the physics step wrote since the previous step
    (benchmarks do this: RoboCasa's toaster-oven fixture sets its door joint between control
    steps).  Replay writes those entries back before stepping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import mujoco
import numpy as np

from ...core.adapter import AdapterCapabilities, HookHandle, SimAdapter, SubstepCallback
from ...core.snapshot import ControlRecord, RestoreReport, Snapshot
from ...core.types import BodyInfo, BodyKind, BodyRole, BodyState, ContactPair, ContactPoint

INTEGRATION = mujoco.mjtState.mjSTATE_INTEGRATION
DEFAULT_ROBOT_PREFIXES = ("robot", "gripper", "mobilebase", "mount", "base0", "left_eef", "right_eef")


@dataclass
class TaskRoles:
    """Which bodies the detectors watch: explicit names beat the defaults."""

    targets: set[str] = field(default_factory=set)       # body names
    containers: set[str] = field(default_factory=set)    # body names (a receptacle's bodies)
    robot_prefixes: tuple[str, ...] = DEFAULT_ROBOT_PREFIXES
    free_bodies_are_targets: bool = False                # no explicit targets: watch every free body


def _body_name(model: mujoco.MjModel, b: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"


def _body_kind(model: mujoco.MjModel, b: int) -> BodyKind:
    if b == 0:
        return BodyKind.STATIC
    if model.body_mocapid[b] >= 0:
        return BodyKind.KINEMATIC
    jnt0, njnt = int(model.body_jntadr[b]), int(model.body_jntnum[b])
    types = {int(model.jnt_type[j]) for j in range(jnt0, jnt0 + njnt)} if njnt > 0 else set()
    if int(mujoco.mjtJoint.mjJNT_FREE) in types:
        return BodyKind.DYNAMIC
    # a body moves if it or any ancestor has a joint
    p = b
    while p > 0:
        if model.body_jntnum[p] > 0 or model.body_mocapid[p] >= 0:
            return BodyKind.LINK
        p = int(model.body_parentid[p])
    return BodyKind.STATIC


class MujocoAdapter(SimAdapter):
    name = "mujoco"

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        roles: TaskRoles | None = None,
        task_name: str = "",
        stepper: Callable[[], None] | None = None,
        control_step_fn: Callable[[], int] | None = None,
        ground_truth_fn: Callable[[], dict[str, Any]] | None = None,
        watched_contacts_only: bool = True,
    ) -> None:
        self.model = model
        self.data = data
        self.roles = roles or TaskRoles(free_bodies_are_targets=True)
        self.task_name = task_name
        self._stepper = stepper or (lambda: mujoco.mj_step(self.model, self.data))
        self._raw_stepper = self._stepper
        self._control_step_fn = control_step_fn
        self._ground_truth_fn = ground_truth_fn
        self._hook: HookHandle | None = None
        self._internal_step = 0
        self._post_step: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._bodies: dict[str, BodyInfo] = {}
        self._index: dict[str, int] = {}
        self._id_by_index: dict[int, str] = {}
        self._vel = np.zeros(6)
        self._force = np.zeros(6)
        # contacts touching none of the target / container / object bodies (robot self-contact,
        # wheels on the floor, fixtures on fixtures) are skipped: no detector reads them
        self.watched_contacts_only = watched_contacts_only
        self._watch = np.zeros(model.nbody, dtype=bool)
        self.bodies(refresh=True)

    # ------------------------------------------------------------------ inventory
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            substep_hook=True, contacts=True, public_state_snapshot=True, native_state_snapshot=True,
            control_capture=True,
            notes=["contacts from data.contact only (equality constraints excluded)",
                   "native snapshot = mj_getState(mjSTATE_INTEGRATION), exact restore"],
        )

    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    def _role(self, name: str, kind: BodyKind) -> BodyRole:
        r = self.roles
        if name in r.targets:
            return BodyRole.TARGET
        if name in r.containers:
            return BodyRole.CONTAINER
        if any(name.startswith(p) for p in r.robot_prefixes):
            return BodyRole.ROBOT
        if kind == BodyKind.DYNAMIC:
            return BodyRole.TARGET if (r.free_bodies_are_targets and not r.targets) else BodyRole.OBJECT
        if kind == BodyKind.STATIC:
            return BodyRole.SCENE
        return BodyRole.OTHER

    def bodies(self, refresh: bool = False) -> dict[str, BodyInfo]:
        if refresh or not self._bodies:
            m = self.model
            geoms_per_body = np.bincount(
                m.geom_bodyid[(m.geom_contype != 0) | (m.geom_conaffinity != 0)], minlength=m.nbody)
            self._bodies, self._index, self._id_by_index = {}, {}, {}
            for b in range(m.nbody):
                name = "world" if b == 0 else _body_name(m, b)
                kind = _body_kind(m, b)
                body_id = f"body:{name}"
                self._bodies[body_id] = BodyInfo(
                    body_id=body_id, name=name, kind=kind, role=self._role(name, kind),
                    mass=float(m.body_mass[b]), articulation=None,
                    collision_shape_count=int(geoms_per_body[b]),
                    metadata={"body_index": b, "root": _body_name(m, int(m.body_rootid[b])) if b else "world"},
                )
                self._index[body_id] = b
                self._id_by_index[b] = body_id
            watched = (BodyRole.TARGET, BodyRole.CONTAINER, BodyRole.OBJECT)
            self._watch[:] = [self._bodies[self._id_by_index[b]].role in watched for b in range(m.nbody)]
        return dict(self._bodies)

    def body_ids_with_role(self, role: BodyRole) -> list[str]:
        return sorted(b for b, info in self._bodies.items() if info.role == role)

    # ------------------------------------------------------------------ ground truth
    def read_states(self, body_ids: Iterable[str] | None = None) -> dict[str, BodyState]:
        m, d = self.model, self.data
        ids = self._bodies.keys() if body_ids is None else body_ids
        out: dict[str, BodyState] = {}
        for body_id in ids:
            b = self._index.get(body_id)
            if b is None:
                continue
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, self._vel, 0)
            ang = self._vel[:3].copy()
            lin = self._vel[3:].copy() + np.cross(ang, d.xipos[b] - d.xpos[b])  # body origin -> centre of mass
            out[body_id] = BodyState(
                body_id=body_id, position=d.xpos[b].copy(), quaternion=d.xquat[b].copy(),
                linear_velocity=lin, angular_velocity=ang,
            )
        return out

    def read_contacts(self) -> list[ContactPair]:
        m, d = self.model, self.data
        n = int(d.ncon)
        if n == 0:
            return []
        con = d.contact
        geom = np.asarray(con.geom).reshape(n, 2)
        bodies = m.geom_bodyid[geom]
        keep = np.asarray(con.efc_address).reshape(n) >= 0  # excluded / inactive: no constraint, no force
        if self.watched_contacts_only:
            keep &= self._watch[bodies[:, 0]] | self._watch[bodies[:, 1]]
        index = np.flatnonzero(keep)
        if index.size == 0:
            return []
        dt = self.timestep()
        dist = np.asarray(con.dist).reshape(n)
        pos = np.asarray(con.pos).reshape(n, 3)
        frames = np.asarray(con.frame).reshape(n, 3, 3)       # rows: normal, tangent1, tangent2
        pairs: dict[tuple[str, str], ContactPair] = {}
        for i in index:
            ida, idb = self._id_by_index[int(bodies[i, 0])], self._id_by_index[int(bodies[i, 1])]
            mujoco.mj_contactForce(m, d, int(i), self._force)  # force in the contact frame
            world_force = frames[i].T @ self._force[:3]
            normal = frames[i, 0].copy()
            impulse = world_force * dt
            key, flip = ((ida, idb), False) if ida <= idb else ((idb, ida), True)
            if flip:  # keep the normal and impulse oriented from body_a to body_b
                normal, impulse = -normal, -impulse
            point = ContactPoint(position=pos[i].copy(), normal=normal, impulse=impulse, separation=float(dist[i]))
            pair = pairs.get(key)
            if pair is None:
                pairs[key] = ContactPair(body_a=key[0], body_b=key[1], points=[point], shape_pairs=1)
            else:
                pair.points.append(point)
        return list(pairs.values())

    def control_step(self) -> int:
        return int(self._control_step_fn()) if self._control_step_fn else self._internal_step

    def task_ground_truth(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task_name": self.task_name,
            "targets": self.body_ids_with_role(BodyRole.TARGET),
            "containers": self.body_ids_with_role(BodyRole.CONTAINER),
            "timestep": self.timestep(),
            "solref_default": [float(x) for x in self.model.opt.o_solref],
            "integrator": int(self.model.opt.integrator),
            "cone": int(self.model.opt.cone),
        }
        if self._ground_truth_fn is not None:
            try:
                payload.update(self._ground_truth_fn())
            except Exception as exc:  # noqa: BLE001
                payload["ground_truth_error"] = f"{type(exc).__name__}: {exc}"
        return payload

    # ------------------------------------------------------------------ stepping
    def install_substep_hook(self, after: SubstepCallback, before: SubstepCallback | None = None) -> HookHandle:
        if self._hook is not None and not self._hook.removed:
            raise RuntimeError("a SimuGuard substep hook is already installed")
        raw = self._raw_stepper

        def hooked() -> None:
            if before is not None:
                before()
            raw()
            self._internal_step += 1
            self.mark_post_step()
            after()

        self.mark_post_step()
        self._stepper = hooked

        def remove() -> None:
            self._stepper = raw

        self._hook = HookHandle(remove_fn=remove, mechanism="adapter_stepper")
        return self._hook

    def step(self) -> None:
        """Advance one physics step through the hook (use this in hand-written loops)."""
        self._stepper()

    def step_physics(self) -> None:
        """One raw physics step, bypassing the hook (used by replay)."""
        self._raw_stepper()
        self._internal_step += 1
        self.mark_post_step()

    def mark_post_step(self) -> None:
        """Remember the state a physics step left, to catch writes made between steps."""
        d = self.data
        self._post_step = (d.qpos.copy(), d.qvel.copy(), d.act.copy())

    # ------------------------------------------------------------------ actuation & snapshots
    def capture_control(self) -> ControlRecord:
        d = self.data
        payload: dict[str, Any] = {"ctrl": d.ctrl.copy()}
        if np.any(d.qfrc_applied):
            payload["qfrc_applied"] = d.qfrc_applied.copy()
        if np.any(d.xfrc_applied):
            rows = np.flatnonzero(np.any(d.xfrc_applied != 0, axis=1))
            payload["xfrc_rows"] = rows.astype(np.int64)
            payload["xfrc_values"] = d.xfrc_applied[rows].reshape(-1).copy()
        if self.model.nmocap:
            payload["mocap_pos"] = d.mocap_pos.reshape(-1).copy()
            payload["mocap_quat"] = d.mocap_quat.reshape(-1).copy()
        if self.model.neq:
            payload["eq_active"] = d.eq_active.astype(np.int64).copy()
        if self._post_step is not None:  # state written outside the physics step since the last one
            for key, now, then in zip(("qpos", "qvel", "act"), (d.qpos, d.qvel, d.act), self._post_step):
                changed = np.flatnonzero(now != then)
                if changed.size:
                    payload[f"{key}_idx"] = changed.astype(np.int64)
                    payload[f"{key}_val"] = now[changed].copy()
        return ControlRecord(substep=-1, control_step=self.control_step(), payload=payload)

    def apply_control(self, control: ControlRecord) -> None:
        d, p = self.data, control.payload
        d.ctrl[:] = np.asarray(p["ctrl"], dtype=np.float64)
        d.qfrc_applied[:] = np.asarray(p["qfrc_applied"], dtype=np.float64) if "qfrc_applied" in p else 0.0
        d.xfrc_applied[:] = 0.0
        if "xfrc_rows" in p:
            rows = np.asarray(p["xfrc_rows"], dtype=np.int64)
            d.xfrc_applied[rows] = np.asarray(p["xfrc_values"], dtype=np.float64).reshape(len(rows), 6)
        if "mocap_pos" in p:
            d.mocap_pos[:] = np.asarray(p["mocap_pos"], dtype=np.float64).reshape(-1, 3)
            d.mocap_quat[:] = np.asarray(p["mocap_quat"], dtype=np.float64).reshape(-1, 4)
        if "eq_active" in p:
            d.eq_active[:] = np.asarray(p["eq_active"]).astype(d.eq_active.dtype)
        for key, target in (("qpos", d.qpos), ("qvel", d.qvel), ("act", d.act)):
            if f"{key}_idx" in p:
                target[np.asarray(p[f"{key}_idx"], dtype=np.int64)] = np.asarray(p[f"{key}_val"], dtype=np.float64)

    def native_state(self) -> np.ndarray:
        size = mujoco.mj_stateSize(self.model, INTEGRATION)
        state = np.empty(size, dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, state, INTEGRATION)
        return state

    def set_native_state(self, state: np.ndarray) -> None:
        state = np.array(state, dtype=np.float64)  # writeable copy: mj_setState rejects read-only buffers
        mujoco.mj_setState(self.model, self.data, state, INTEGRATION)
        mujoco.mj_forward(self.model, self.data)             # derived quantities (poses, contacts)
        mujoco.mj_setState(self.model, self.data, state, INTEGRATION)  # forward must not leak into the state

    def capture_public_state(self) -> dict[str, Any]:
        d = self.data
        return {"qpos": d.qpos.tolist(), "qvel": d.qvel.tolist(), "act": d.act.tolist(), "time": float(d.time)}

    def capture_snapshot(self, substep: int, *, include_native: bool = True) -> Snapshot:
        control = self.capture_control()
        control.substep = int(substep)
        native = self.native_state().tobytes() if include_native else None
        return Snapshot(
            substep=int(substep), control_step=self.control_step(), public_state=self.capture_public_state(),
            control=control, native_state=native, native_format="mujoco_state_integration" if native else None,
            metadata={"task_name": self.task_name, "timestep": self.timestep(), "mujoco": mujoco.__version__},
        )

    def restore_snapshot(self, snapshot: Snapshot, *, method: str = "auto") -> RestoreReport:
        if method == "auto":
            method = "native" if snapshot.native_state is not None else "public"
        report = RestoreReport(method=method, substep=snapshot.substep)
        if method == "native":
            self.set_native_state(np.frombuffer(snapshot.native_state, dtype=np.float64))
        elif method == "public":
            ps = snapshot.public_state
            self.data.qpos[:] = ps["qpos"]
            self.data.qvel[:] = ps["qvel"]
            if len(ps.get("act", [])):
                self.data.act[:] = ps["act"]
            self.data.time = ps["time"]
            mujoco.mj_forward(self.model, self.data)
        elif method != "none":
            raise ValueError(f"unknown restore method: {method}")
        self.apply_control(snapshot.control)
        self._internal_step = snapshot.substep
        self.mark_post_step()
        return report


def attach_robosuite(env: Any, *, roles: TaskRoles | None = None, task_name: str = "") -> MujocoAdapter:
    """Adapter bound to a robosuite / RoboCasa env whose physics step becomes the hooked step.

    robosuite runs, per physics substep, ``sim.step1(); _pre_action(); sim.step2()`` when
    ``lite_physics`` is on (RoboCasa's default) and ``sim.forward(); _pre_action(); sim.step()``
    otherwise.  The hook wraps ``step2`` or ``step`` on the ``sim`` instance, so its ``before``
    callback sees the actuation the controller just wrote for that substep.  Replay steps with
    ``mj_step``, which equals ``step1`` + ``step2`` because the actuation only enters ``step2``.
    Attach again after every reset: a hard reset builds a new ``sim``.
    """
    sim = env.sim
    model, data = sim.model._model, sim.data._data
    method = "step2" if getattr(env, "lite_physics", False) else "step"
    original = getattr(sim, method)
    adapter = MujocoAdapter(
        model, data, roles=roles, task_name=task_name or type(env).__name__,
        control_step_fn=lambda: int(getattr(env, "timestep", 0)),
        ground_truth_fn=lambda: {"success": bool(env._check_success())} if hasattr(env, "_check_success") else {},
    )

    def install_on_sim(after: SubstepCallback, before: SubstepCallback | None = None) -> HookHandle:
        if adapter._hook is not None and not adapter._hook.removed:
            raise RuntimeError("a SimuGuard substep hook is already installed")

        def hooked(*args: Any, **kwargs: Any) -> Any:
            if before is not None:
                before()
            out = original(*args, **kwargs)
            adapter._internal_step += 1
            adapter.mark_post_step()
            after()
            return out

        adapter.mark_post_step()
        setattr(sim, method, hooked)

        def remove() -> None:
            try:
                delattr(sim, method)  # the class method is visible again
            except AttributeError:
                pass

        adapter._hook = HookHandle(remove_fn=remove, mechanism=f"robosuite_sim_{method}_instance_attribute")
        return adapter._hook

    adapter.install_substep_hook = install_on_sim  # type: ignore[method-assign]
    return adapter
