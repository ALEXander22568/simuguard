"""Simulator adapter contract.

An adapter is the only component that touches simulator objects.  It exposes
ground truth (bodies, states, contacts), a post-substep hook, and snapshot
capture/restore.  Everything else in SimuGuard is written against this
interface, so a new benchmark needs one adapter and no core changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .snapshot import ControlRecord, RestoreReport, Snapshot
from .types import BodyInfo, BodyState, ContactPair, SubstepFrame


SubstepCallback = Callable[[], None]


@dataclass
class HookHandle:
    """Returned by :meth:`SimAdapter.install_substep_hook`; call ``remove()``."""

    remove_fn: Callable[[], None]
    mechanism: str
    removed: bool = False

    def remove(self) -> None:
        if not self.removed:
            self.remove_fn()
            self.removed = True


@dataclass
class AdapterCapabilities:
    substep_hook: bool = True
    contacts: bool = True
    public_state_snapshot: bool = True
    native_state_snapshot: bool = False  # e.g. PhysX pack/unpack
    control_capture: bool = True
    notes: list[str] = field(default_factory=list)


class SimAdapter(ABC):
    """Minimal interface every simulator/benchmark adapter must implement."""

    name: str = "abstract"

    # ---- ground truth ------------------------------------------------------
    @abstractmethod
    def capabilities(self) -> AdapterCapabilities: ...

    @abstractmethod
    def timestep(self) -> float: ...

    @abstractmethod
    def bodies(self, refresh: bool = False) -> dict[str, BodyInfo]:
        """Body inventory keyed by stable ``body_id``."""

    @abstractmethod
    def read_states(self, body_ids: Iterable[str] | None = None) -> dict[str, BodyState]: ...

    @abstractmethod
    def read_contacts(self) -> list[ContactPair]: ...

    @abstractmethod
    def control_step(self) -> int:
        """Benchmark policy-action counter (not the physics substep)."""

    def task_ground_truth(self) -> dict[str, Any]:
        """Task-level ground truth (success predicate inputs, object ids...)."""

        return {}

    def read_frame(
        self,
        substep: int,
        body_ids: Iterable[str] | None = None,
        with_contacts: bool = True,
    ) -> SubstepFrame:
        dt = self.timestep()
        return SubstepFrame(
            substep=int(substep),
            sim_time=float(substep) * dt,
            control_step=int(self.control_step()),
            timestep=dt,
            states=self.read_states(body_ids),
            contacts=self.read_contacts() if with_contacts else [],
        )

    # ---- instrumentation ---------------------------------------------------
    @abstractmethod
    def install_substep_hook(self, after: SubstepCallback, before: SubstepCallback | None = None) -> HookHandle:
        """Call ``before()`` immediately before and ``after()`` immediately after every physics substep.

        ``before`` is where actuation must be captured: benchmarks may set
        state-dependent commands (e.g. RoboTwin's gravity-compensation ``qf``)
        right before stepping, which cannot be recovered after the step.
        """

    # ---- snapshots ---------------------------------------------------------
    @abstractmethod
    def capture_control(self) -> ControlRecord: ...

    @abstractmethod
    def apply_control(self, control: ControlRecord) -> None: ...

    @abstractmethod
    def capture_snapshot(self, substep: int, *, include_native: bool = True) -> Snapshot: ...

    @abstractmethod
    def restore_snapshot(self, snapshot: Snapshot, *, method: str = "auto") -> RestoreReport: ...

    @abstractmethod
    def step_physics(self) -> None:
        """Advance exactly one physics substep (used by replay)."""
