"""RoboTwin 2.0 (SAPIEN 3) adapter."""

from __future__ import annotations

from typing import Any, Iterable

from ...core.adapter import AdapterCapabilities, HookHandle, SimAdapter, SubstepCallback
from ...core.snapshot import ControlRecord, RestoreReport, Snapshot, compare_states
from ...core.types import BodyInfo, BodyRole, BodyState, ContactPair
from . import sapien_io as io
from .tasks import ContainmentGate, TaskSpec, box_corners, get_task_spec, model_bounds


class RoboTwinAdapter(SimAdapter):
    """Ground truth, substep hook and snapshots for a RoboTwin ``Base_Task`` env.

    Construct after ``env.setup_demo(...)`` (the scene and task objects must exist).
    """

    name = "robotwin"

    def __init__(self, env: Any, *, task_spec: TaskSpec | None = None, task_name: str | None = None) -> None:
        self.env = env
        self.task_name = task_name or str(getattr(env, "task_name", "") or type(env).__name__)
        self.spec = task_spec or get_task_spec(self.task_name)
        self._bodies: dict[str, io.BodyHandle] = {}
        self._articulations: dict[str, io.ArticulationHandle] = {}
        self._id_by_key: dict[Any, str] = {}
        self._role_warnings: list[str] = []
        self._hook: HookHandle | None = None
        self._raw_step: Any | None = None
        self.bodies(refresh=True)

    # ------------------------------------------------------------------ scene access
    @property
    def scene(self) -> Any:
        scene = self.env.scene
        return getattr(scene, "_simuguard_wrapped_scene", scene)

    def capabilities(self) -> AdapterCapabilities:
        native = callable(getattr(io.physx_system(self.scene), "pack", None))
        return AdapterCapabilities(
            substep_hook=True,
            contacts=True,
            public_state_snapshot=True,
            native_state_snapshot=native,
            control_capture=True,
            notes=[
                "PhysX contact warm-start caches are not part of the public state; use native pack for fidelity.",
                "Snapshots taken inside a TOPP sub-trajectory restore physics only, not RoboTwin's Python planner state.",
            ],
        )

    def timestep(self) -> float:
        return float(self.scene.get_timestep())

    def control_step(self) -> int:
        return int(getattr(self.env, "take_action_cnt", 0) or 0)

    # ------------------------------------------------------------------ inventory
    def _role_by_entity(self) -> dict[Any, BodyRole]:
        roles: dict[Any, BodyRole] = {}
        self._role_warnings = []
        for attrs, role in ((self.spec.container_attrs, BodyRole.CONTAINER), (self.spec.target_attrs, BodyRole.TARGET)):
            for attr in attrs:
                wrapper = getattr(self.env, attr, None)
                entity = getattr(wrapper, "actor", wrapper)
                if entity is None:
                    self._role_warnings.append(f"task attribute '{attr}' not found for role {role.value}")
                    continue
                if hasattr(entity, "get_links"):  # articulated object
                    for link in entity.get_links():
                        roles[io.component_key(link)] = role
                else:
                    roles[io.entity_key(entity)] = role
        return roles

    def _robot_articulation_keys(self) -> set[Any]:
        robot = getattr(self.env, "robot", None)
        keys = set()
        if robot is None:
            return keys
        for value in vars(robot).values():
            if hasattr(value, "get_links") and hasattr(value, "get_qpos"):
                try:
                    keys.add(io.articulation_key(value))
                except Exception:  # noqa: BLE001
                    continue
        return keys

    def bodies(self, refresh: bool = False) -> dict[str, BodyInfo]:
        if refresh or not self._bodies:
            self._bodies, self._articulations = io.enumerate_scene(
                self.scene,
                robot_articulations=self._robot_articulation_keys(),
                role_by_entity=self._role_by_entity(),
            )
            self._id_by_key = {io.component_key(h.component): body_id for body_id, h in self._bodies.items()}
        return {body_id: handle.info for body_id, handle in self._bodies.items()}

    def body_ids_with_role(self, role: BodyRole) -> list[str]:
        return sorted(b for b, h in self._bodies.items() if h.info.role == role)

    def read_states(self, body_ids: Iterable[str] | None = None) -> dict[str, BodyState]:
        ids = self._bodies.keys() if body_ids is None else body_ids
        return {body_id: io.read_state(self._bodies[body_id]) for body_id in ids if body_id in self._bodies}

    def read_contacts(self) -> list[ContactPair]:
        return io.read_contacts(self.scene, self._id_by_key)

    def task_ground_truth(self) -> dict[str, Any]:
        env = self.env
        payload: dict[str, Any] = {
            "task_name": self.task_name,
            "task_spec": {
                "target_attrs": list(self.spec.target_attrs),
                "container_attrs": list(self.spec.container_attrs),
                "containment": list(self.spec.containment) if self.spec.containment else None,
                "notes": self.spec.notes,
            },
            "role_warnings": list(self._role_warnings),
            "take_action_cnt": getattr(env, "take_action_cnt", None),
            "step_lim": getattr(env, "step_lim", None),
            "eval_success": getattr(env, "eval_success", None),
            "plan_success": getattr(env, "plan_success", None),
            "targets": self.body_ids_with_role(BodyRole.TARGET),
            "containers": self.body_ids_with_role(BodyRole.CONTAINER),
        }
        for attr in self.spec.ground_truth_attrs:
            value = getattr(env, attr, None)
            if hasattr(value, "item") and callable(value.item) and getattr(value, "size", 1) == 1:
                value = value.item()
            payload[attr] = value if isinstance(value, (int, float, str, bool, type(None))) else str(value)
        try:
            payload["check_success"] = bool(env.check_success())
        except Exception as exc:  # noqa: BLE001
            payload["check_success_error"] = f"{type(exc).__name__}: {exc}"
        return payload

    def containment_gate(self) -> ContainmentGate | None:
        """Gate for :class:`ContactEjectionDetector` built from asset model_data bounds."""

        if self.spec.containment is None:
            return None
        target_attr, container_attr = self.spec.containment
        target_wrapper = getattr(self.env, target_attr)
        container_wrapper = getattr(self.env, container_attr)
        target_ids = self.body_ids_with_role(BodyRole.TARGET)
        container_ids = self.body_ids_with_role(BodyRole.CONTAINER)
        if len(target_ids) != 1 or len(container_ids) != 1:
            raise ValueError(f"containment gate needs exactly one target/container, got {target_ids}/{container_ids}")
        t_low, t_high = model_bounds(target_wrapper.config)
        c_low, c_high = model_bounds(container_wrapper.config)
        return ContainmentGate(
            target_body=target_ids[0],
            container_body=container_ids[0],
            target_corners_local=box_corners(t_low, t_high),
            container_low=c_low,
            container_high=c_high,
            tolerance_m=self.spec.containment_tolerance_m,
        )

    # ------------------------------------------------------------------ substep hook
    def install_substep_hook(self, after: SubstepCallback, before: SubstepCallback | None = None) -> HookHandle:
        if self._hook is not None and not self._hook.removed:
            raise RuntimeError("a SimuGuard substep hook is already installed")
        scene = self.scene
        original = scene.step
        self._raw_step = original

        def hooked_step(*args: Any, **kwargs: Any) -> Any:
            if before is not None:
                before()
            result = original(*args, **kwargs)
            after()
            return result

        try:
            scene.step = hooked_step  # instance attribute shadows the class method
            if scene.step is not hooked_step:
                raise AttributeError("instance attribute did not take effect")

            def remove() -> None:
                try:
                    del scene.step
                except AttributeError:
                    pass
                self._raw_step = None

            self._hook = HookHandle(remove_fn=remove, mechanism="instance_attribute")
        except (AttributeError, TypeError):
            proxy = _SceneProxy(scene, hooked_step)
            self.env.scene = proxy

            def remove_proxy() -> None:
                if self.env.scene is proxy:
                    self.env.scene = scene
                self._raw_step = None

            self._hook = HookHandle(remove_fn=remove_proxy, mechanism="scene_proxy")
        return self._hook

    def step_physics(self) -> None:
        (self._raw_step or self.scene.step)()

    # ------------------------------------------------------------------ control & snapshots
    def capture_control(self) -> ControlRecord:
        payload = {
            art_id: io.capture_articulation_control(handle.articulation)
            for art_id, handle in self._articulations.items()
        }
        return ControlRecord(substep=-1, control_step=self.control_step(), payload=payload)

    def apply_control(self, control: ControlRecord) -> None:
        problems = []
        for art_id, data in control.payload.items():
            handle = self._articulations.get(art_id)
            if handle is None:
                problems.append(f"missing:{art_id}")
                continue
            problems.extend(io.apply_articulation_control(handle.articulation, data, art_id))
        if problems:
            raise RuntimeError(f"control topology mismatch: {problems}")

    def capture_public_state(self) -> dict[str, Any]:
        return io.capture_public_state(self._bodies, self._articulations)

    def capture_snapshot(self, substep: int, *, include_native: bool = True) -> Snapshot:
        native = io.pack_native(self.scene) if include_native else None
        control = self.capture_control()
        control.substep = int(substep)
        return Snapshot(
            substep=int(substep),
            control_step=self.control_step(),
            public_state=self.capture_public_state(),
            control=control,
            native_state=native,
            native_format="physx_cpu_pack" if native is not None else None,
            metadata={"task_name": self.task_name, "timestep": self.timestep()},
        )

    def restore_snapshot(self, snapshot: Snapshot, *, method: str = "auto") -> RestoreReport:
        if method == "auto":
            method = "native" if snapshot.native_state is not None else "public"
        report = RestoreReport(method=method, substep=snapshot.substep)
        if method == "none":
            pass  # caller rebuilt the environment deterministically; only verify and re-apply control
        elif method == "native":
            if snapshot.native_state is None:
                raise ValueError("snapshot has no native state")
            io.unpack_native(self.scene, snapshot.native_state)
        elif method == "public":
            report.missing, report.mismatched = io.restore_public_state(snapshot.public_state, self._bodies, self._articulations)
        elif method == "native+public":
            if snapshot.native_state is None:
                raise ValueError("snapshot has no native state")
            io.unpack_native(self.scene, snapshot.native_state)
            report.missing, report.mismatched = io.restore_public_state(snapshot.public_state, self._bodies, self._articulations)
        else:
            raise ValueError(f"unknown restore method: {method}")
        self.apply_control(snapshot.control)
        report.public_state_error = compare_states(snapshot.public_state, self.capture_public_state())
        return report

    # ------------------------------------------------------------------ interventions
    def solver_iterations(self) -> dict[str, list[int]]:
        return io.solver_iterations(self._bodies, self._articulations)

    def set_solver_iterations(self, *, position: int, velocity: int) -> None:
        io.set_solver_iterations(self._bodies, self._articulations, position=position, velocity=velocity)

    def set_mass(self, body_id: str, mass: float) -> None:
        handle = self._bodies[body_id]
        handle.component.set_mass(float(mass))


class _SceneProxy:
    """Fallback when the scene object does not accept instance attributes."""

    def __init__(self, scene: Any, step: Any) -> None:
        object.__setattr__(self, "_simuguard_wrapped_scene", scene)
        object.__setattr__(self, "step", step)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._simuguard_wrapped_scene, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._simuguard_wrapped_scene, name, value)
