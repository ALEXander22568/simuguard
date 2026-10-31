"""Deterministic toy simulator implementing the SimAdapter contract.

World: a static table, a container resting at z=0, a free target (point mass)
and a kinematic robot link.  The target falls under gravity, rests on the
container floor (z=0) with restitution, and receives a velocity ``kick`` from
the per-substep command while touching the container -- a stand-in for a
solver impulse artefact.  Pickled state doubles as the "native" snapshot.
"""

from __future__ import annotations

import copy
import pickle
from typing import Any, Iterable

import numpy as np

from simuguard.core.adapter import AdapterCapabilities, HookHandle, SimAdapter
from simuguard.core.snapshot import ControlRecord, RestoreReport, Snapshot, compare_states
from simuguard.core.types import BodyInfo, BodyKind, BodyRole, BodyState, ContactPair, ContactPoint

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


class ToyAdapter(SimAdapter):
    name = "toy"

    def __init__(self, dt: float = 0.004, restitution: float = 0.0, target_mass: float = 0.01) -> None:
        self.dt = dt
        self.restitution = restitution
        self.target_mass = target_mass
        self.state = {
            "target_p": np.array([0.0, 0.0, 0.05]),
            "target_v": np.zeros(3),
            "robot_p": np.array([0.3, 0.0, 0.2]),
            "robot_v": np.zeros(3),
        }
        self.command = {"kick": [0.0, 0.0, 0.0], "robot_v": [0.0, 0.0, 0.0]}
        self.actions = 0
        self._callbacks: list[Any] = []
        self._last_contacts: list[ContactPair] = []
        self._infos = {
            "actor:table": BodyInfo("actor:table", "table", BodyKind.STATIC, BodyRole.SCENE),
            "actor:basket": BodyInfo("actor:basket", "basket", BodyKind.DYNAMIC, BodyRole.CONTAINER, mass=0.5),
            "actor:can": BodyInfo("actor:can", "can", BodyKind.DYNAMIC, BodyRole.TARGET, mass=target_mass),
            "link:robot:gripper": BodyInfo("link:robot:gripper", "gripper", BodyKind.LINK, BodyRole.ROBOT, mass=0.2, articulation="art:robot"),
        }

    # ---- physics -------------------------------------------------------------
    def step_physics(self) -> None:
        dt = self.dt
        s = self.state
        s["robot_v"] = np.asarray(self.command["robot_v"], dtype=float)
        s["robot_p"] = s["robot_p"] + s["robot_v"] * dt
        v_before = s["target_v"].copy()
        touching = s["target_p"][2] <= 1e-9
        if touching:
            s["target_v"] = s["target_v"] + np.asarray(self.command["kick"], dtype=float)
        s["target_v"] = s["target_v"] + np.array([0.0, 0.0, -9.81]) * dt
        s["target_p"] = s["target_p"] + s["target_v"] * dt
        contacts: list[ContactPair] = []
        if s["target_p"][2] <= 0.0:
            s["target_p"][2] = 0.0
            if s["target_v"][2] < 0:
                s["target_v"][2] = -self.restitution * s["target_v"][2]
        if s["target_p"][2] <= 1e-9 or touching:
            dv = s["target_v"] - v_before
            impulse = self.target_mass * (dv - np.array([0.0, 0.0, -9.81]) * dt)
            contacts.append(
                ContactPair(
                    "actor:can",
                    "actor:basket",
                    [ContactPoint(s["target_p"].copy(), np.array([0.0, 0.0, 1.0]), impulse, -0.0001)],
                )
            )
        self._last_contacts = contacts

    def step(self) -> None:
        """Benchmark-facing step: physics + hooks (what an env would call)."""

        for before, _ in list(self._callbacks):
            if before is not None:
                before()
        self.step_physics()
        for _, after in list(self._callbacks):
            after()

    # ---- SimAdapter ------------------------------------------------------------
    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(native_state_snapshot=True)

    def timestep(self) -> float:
        return self.dt

    def bodies(self, refresh: bool = False) -> dict[str, BodyInfo]:
        return dict(self._infos)

    def read_states(self, body_ids: Iterable[str] | None = None) -> dict[str, BodyState]:
        s = self.state
        all_states = {
            "actor:table": BodyState("actor:table", np.array([0.0, 0.0, -0.05]), IDENTITY, np.zeros(3), np.zeros(3)),
            "actor:basket": BodyState("actor:basket", np.zeros(3), IDENTITY, np.zeros(3), np.zeros(3)),
            "actor:can": BodyState("actor:can", s["target_p"].copy(), IDENTITY, s["target_v"].copy(), np.zeros(3)),
            "link:robot:gripper": BodyState("link:robot:gripper", s["robot_p"].copy(), IDENTITY, s["robot_v"].copy(), np.zeros(3)),
        }
        if body_ids is None:
            return all_states
        return {k: all_states[k] for k in body_ids if k in all_states}

    def read_contacts(self) -> list[ContactPair]:
        return list(self._last_contacts)

    def control_step(self) -> int:
        return self.actions

    def install_substep_hook(self, after: Any, before: Any = None) -> HookHandle:
        entry = (before, after)
        self._callbacks.append(entry)
        return HookHandle(remove_fn=lambda: self._callbacks.remove(entry), mechanism="toy_callback")

    def capture_control(self) -> ControlRecord:
        return ControlRecord(-1, self.actions, copy.deepcopy(self.command))

    def apply_control(self, control: ControlRecord) -> None:
        self.command = copy.deepcopy(control.payload)

    def _public(self) -> dict[str, Any]:
        return {key: np.asarray(value).tolist() for key, value in self.state.items()}

    def capture_snapshot(self, substep: int, *, include_native: bool = True) -> Snapshot:
        control = self.capture_control()
        control.substep = substep
        return Snapshot(
            substep=substep,
            control_step=self.actions,
            public_state=self._public(),
            control=control,
            native_state=pickle.dumps({k: v.copy() for k, v in self.state.items()}) if include_native else None,
            native_format="toy_pickle" if include_native else None,
        )

    def restore_snapshot(self, snapshot: Snapshot, *, method: str = "auto") -> RestoreReport:
        if method in ("auto", "native") and snapshot.native_state is not None:
            self.state = pickle.loads(snapshot.native_state)
            method = "native"
        else:
            self.state = {k: np.asarray(v, dtype=float) for k, v in snapshot.public_state.items()}
            method = "public"
        self.apply_control(snapshot.control)
        report = RestoreReport(method=method, substep=snapshot.substep)
        report.public_state_error = compare_states(snapshot.public_state, self._public())
        return report


def run_kick_episode(adapter: ToyAdapter, *, settle: int = 60, kick_at: int = 80, total: int = 200, kick=(0.0, 0.0, 3.0)) -> None:
    """Let the target settle, apply one kick substep, then continue."""

    for i in range(1, total + 1):
        adapter.command = {"kick": list(kick) if i == kick_at else [0.0, 0.0, 0.0], "robot_v": [0.0, 0.0, 0.0]}
        if i % 10 == 0:
            adapter.actions += 1
        adapter.step()
