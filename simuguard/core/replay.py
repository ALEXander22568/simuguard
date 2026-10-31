"""Deterministic replay of a bundle and fidelity measurement.

A replay is only evidence about the *source* event if, under unchanged
settings, it reproduces the reference trajectory (fidelity) and the detector
fires again (event reproduction).  Interventions are applied by the caller
between ``restore`` and the first replayed substep via ``before_replay``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from .adapter import SimAdapter
from .detectors.base import Detector, DetectorContext
from .events import Event, EventStatus
from .snapshot import ReplayBundle
from .statelog import StateLog
from .types import SubstepFrame


@dataclass
class ReplayResult:
    restore: dict[str, Any]
    substeps: int
    body_ids: list[str]
    max_position_error_m: dict[str, float]
    first_substep_over_tolerance: dict[str, int | None]
    position_tolerance_m: float
    events: list[dict[str, Any]] = field(default_factory=list)
    trajectory: dict[int, dict[str, list[float]]] | None = None
    max_speed_mps: dict[str, float] = field(default_factory=dict)
    state_log: StateLog | None = None  # full-precision replayed states, for windowed metrics

    @property
    def overall_max_error_m(self) -> float:
        return max(self.max_position_error_m.values(), default=0.0)

    @property
    def within_tolerance(self) -> bool:
        return all(value is None for value in self.first_substep_over_tolerance.values())

    @property
    def confirmed_events(self) -> list[dict[str, Any]]:
        return [e for e in self.events if e["status"] == EventStatus.CONFIRMED.value]

    def to_dict(self) -> dict[str, Any]:
        return {
            "restore": self.restore,
            "substeps": self.substeps,
            "body_ids": self.body_ids,
            "position_tolerance_m": self.position_tolerance_m,
            "overall_max_position_error_m": self.overall_max_error_m,
            "within_tolerance": self.within_tolerance,
            "max_position_error_m": self.max_position_error_m,
            "first_substep_over_tolerance": self.first_substep_over_tolerance,
            "max_speed_mps": self.max_speed_mps,
            "confirmed_event_count": len(self.confirmed_events),
            "events": self.events,
        }


def replay_bundle(
    adapter: SimAdapter,
    bundle: ReplayBundle,
    *,
    method: str = "auto",
    body_ids: Iterable[str] | None = None,
    detectors: Iterable[Detector] = (),
    position_tolerance_m: float = 1.0e-3,
    before_replay: Callable[[SimAdapter], None] | None = None,
    episode_id: str = "replay",
    keep_trajectory: bool = False,
    keep_state_log: bool = False,
) -> ReplayResult:
    restore = adapter.restore_snapshot(bundle.snapshot, method=method)
    if before_replay is not None:
        before_replay(adapter)

    reference = {int(frame["substep"]): frame for frame in bundle.reference_frames}
    if body_ids is None:
        ids: set[str] = set()
        for frame in bundle.reference_frames:
            ids.update(frame.get("states", {}).keys())
        body_ids = sorted(ids)
    body_ids = list(body_ids)

    detectors = list(detectors)
    context = DetectorContext(episode_id=episode_id, timestep=adapter.timestep(), bodies=adapter.bodies())
    for detector in detectors:
        detector.reset(context)

    max_error = {body_id: 0.0 for body_id in body_ids}
    max_speed = {body_id: 0.0 for body_id in body_ids}
    first_over: dict[str, int | None] = {body_id: None for body_id in body_ids}
    events: dict[str, Event] = {}
    frame: SubstepFrame | None = None
    tracked = sorted(set(body_ids) | {b for b in context.free_bodies()})
    trajectory: dict[int, dict[str, list[float]]] | None = {} if keep_trajectory else None
    state_log = StateLog(body_ids) if keep_state_log else None

    for record in bundle.controls:
        adapter.apply_control(record)
        adapter.step_physics()
        frame = adapter.read_frame(record.substep, tracked, with_contacts=bool(detectors))
        if state_log is not None:
            state_log.append(record.substep, frame.states)
        if trajectory is not None:
            trajectory[record.substep] = {b: frame.states[b].position.tolist() for b in body_ids if b in frame.states}
        for body_id in body_ids:
            state = frame.states.get(body_id)
            if state is not None:
                max_speed[body_id] = max(max_speed[body_id], state.speed)
        ref = reference.get(record.substep)
        if ref is not None:
            for body_id in body_ids:
                expected = ref.get("states", {}).get(body_id)
                actual = frame.states.get(body_id)
                if expected is None or actual is None:
                    continue
                error = float(np.linalg.norm(np.asarray(actual.position) - np.asarray(expected["p"], dtype=float)))
                max_error[body_id] = max(max_error[body_id], error)
                if error > position_tolerance_m and first_over[body_id] is None:
                    first_over[body_id] = record.substep
        for detector in detectors:
            for event in detector.observe(frame, context):
                events[event.event_id] = event
        context.previous = frame

    for detector in detectors:
        for event in detector.finalize(frame, context):
            events[event.event_id] = event

    return ReplayResult(
        restore=restore.to_dict(),
        substeps=len(bundle.controls),
        body_ids=body_ids,
        max_position_error_m=max_error,
        first_substep_over_tolerance=first_over,
        position_tolerance_m=position_tolerance_m,
        events=[event.to_dict() for event in events.values()],
        trajectory=trajectory,
        max_speed_mps=max_speed,
        state_log=state_log,
    )
