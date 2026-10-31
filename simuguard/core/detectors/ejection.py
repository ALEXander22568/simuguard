"""Contact-triggered ejection detector.

Port of the TwinGuard ``PhysicsSubstepMonitor`` confirmation logic (the version
used for the 84-episode cavity review), rewritten against the simulator-agnostic
data model and role-based body selection.

Lifecycle per free body::

    onset (CANDIDATE)  : recent contact within ``contact_window_ms`` AND
                         speed >= min_speed AND per-substep |dv| >= min_delta_v
    confirmation       : displacement since onset >= min_flight_distance AND
                         ( free flight >= min_free_flight_ms                -> ballistic_free_flight
                         | risky contact AND max|dv| >= violent_delta_v AND
                           max speed >= violent_speed                     -> violent_risky_contact_displacement )
    window close       : after ``post_window_ms``; unconfirmed -> REJECTED

"Risky" contact is a TARGET<->CONTAINER pair.  The second confirmation path
exists because an object ejected by concave-geometry artefacts can keep a
persistent contact manifold with the container while it flies away.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from ..events import Event, EventStatus
from ..types import BodyRole, ContactPair, SubstepFrame
from .base import Detector, DetectorContext

# gate(frame, body_id, context) -> {"active": bool, ...}
GateFn = Callable[[SubstepFrame, str, DetectorContext], dict[str, Any]]


@dataclass
class EjectionConfig:
    contact_window_ms: float = 40.0
    post_window_ms: float = 240.0
    min_speed_mps: float = 0.50
    min_delta_v_mps: float = 0.35
    min_flight_distance_m: float = 0.04
    min_free_flight_ms: float = 16.0
    violent_delta_v_mps: float = 0.75
    violent_speed_mps: float = 1.00
    monitored_roles: tuple[str, ...] = ("target", "container", "object")
    risky_pair_roles: tuple[str, str] = ("target", "container")
    # The two per-substep |dv| thresholds hold for this timestep; on another timestep they are
    # scaled by dt / reference, i.e. applied as accelerations.  None: use them as given.
    delta_v_reference_timestep_s: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EjectionConfig":
        data = dict(data or {})
        for key in ("monitored_roles", "risky_pair_roles"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)


@dataclass
class _Evidence:
    substep: int
    partners: tuple[str, ...]
    max_force_n: float
    max_impulse_ns: float
    max_penetration_m: float
    risky: bool


@dataclass
class _Candidate:
    event_id: str
    body_id: str
    onset_substep: int
    deadline_substep: int
    control_step: int
    onset_position: np.ndarray
    onset_speed: float
    onset_delta_v: float
    evidence: _Evidence
    gate_at_onset: dict[str, Any] = field(default_factory=dict)
    max_distance: float = 0.0
    max_height_gain: float = 0.0
    max_speed: float = 0.0
    max_delta_v: float = 0.0
    free_flight: int = 0
    max_free_flight: int = 0
    confirmed_at: int | None = None
    reasons: list[str] = field(default_factory=list)


class ContactEjectionDetector(Detector):
    name = "contact_ejection"

    def __init__(self, config: EjectionConfig | None = None, gate: GateFn | None = None) -> None:
        self.cfg = config or EjectionConfig()
        self.gate = gate
        self._reset_state()

    def config(self) -> dict[str, Any]:
        payload = asdict(self.cfg)
        payload["gate"] = getattr(self.gate, "__name__", type(self.gate).__name__) if self.gate else None
        return payload

    def _reset_state(self) -> None:
        self._contact_window = 1
        self._post_window = 1
        self._min_free_flight = 1
        self._min_delta_v = self.cfg.min_delta_v_mps
        self._violent_delta_v = self.cfg.violent_delta_v_mps
        self._prev_velocity: dict[str, np.ndarray] = {}
        self._history: dict[str, deque[_Evidence]] = defaultdict(deque)
        self._pending: dict[str, _Candidate] = {}
        self._cooldown_until: dict[str, int] = {}

    def reset(self, context: DetectorContext) -> None:
        self._reset_state()
        dt = context.timestep
        self._contact_window = max(1, math.ceil(self.cfg.contact_window_ms / (1000.0 * dt)))
        self._post_window = max(1, math.ceil(self.cfg.post_window_ms / (1000.0 * dt)))
        self._min_free_flight = max(1, math.ceil(self.cfg.min_free_flight_ms / (1000.0 * dt)))
        scale = dt / self.cfg.delta_v_reference_timestep_s if self.cfg.delta_v_reference_timestep_s else 1.0
        self._min_delta_v = self.cfg.min_delta_v_mps * scale
        self._violent_delta_v = self.cfg.violent_delta_v_mps * scale

    # ------------------------------------------------------------------
    def _is_risky(self, pair: ContactPair, context: DetectorContext) -> bool:
        wanted = {BodyRole(role) for role in self.cfg.risky_pair_roles}
        return {context.role(pair.body_a), context.role(pair.body_b)} == wanted

    def _evidence(self, frame: SubstepFrame, body_id: str, contacts: list[ContactPair], context: DetectorContext) -> _Evidence | None:
        active = [pair for pair in contacts if pair.point_count > 0]
        if not active:
            return None
        return _Evidence(
            substep=frame.substep,
            partners=tuple(sorted({pair.other(body_id) for pair in active})),
            max_force_n=max(pair.force_estimate(frame.timestep) for pair in active),
            max_impulse_ns=max(pair.impulse_magnitude for pair in active),
            max_penetration_m=max(pair.max_penetration for pair in active),
            risky=any(self._is_risky(pair, context) for pair in active),
        )

    def observe(self, frame: SubstepFrame, context: DetectorContext) -> list[Event]:
        events: list[Event] = []
        roles = tuple(BodyRole(role) for role in self.cfg.monitored_roles)
        contacts_by_body: dict[str, list[ContactPair]] = defaultdict(list)
        for pair in frame.contacts:
            contacts_by_body[pair.body_a].append(pair)
            contacts_by_body[pair.body_b].append(pair)

        for body_id in context.free_bodies(roles):
            state = frame.states.get(body_id)
            if state is None or not state.is_finite():
                continue
            velocity = np.asarray(state.linear_velocity, dtype=float)
            delta_v = float(np.linalg.norm(velocity - self._prev_velocity.get(body_id, velocity)))
            self._prev_velocity[body_id] = velocity.copy()
            speed = float(np.linalg.norm(velocity))
            contacts = contacts_by_body.get(body_id, [])

            history = self._history[body_id]
            evidence = self._evidence(frame, body_id, contacts, context)
            if evidence is not None:
                history.append(evidence)
            while history and frame.substep - history[0].substep > self._contact_window:
                history.popleft()

            candidate = self._pending.get(body_id)
            if candidate is not None:
                events.extend(self._update(candidate, frame, state.position, speed, delta_v, bool(evidence), context))
                continue
            if frame.substep < self._cooldown_until.get(body_id, -1):
                continue
            if not history or speed < self.cfg.min_speed_mps or delta_v < self._min_delta_v:
                continue

            recent = max(history, key=lambda item: item.max_force_n)
            gate = self.gate(frame, body_id, context) if self.gate else {}
            candidate = _Candidate(
                event_id=f"{context.episode_id}:{self.name}:{body_id}:{frame.substep}",
                body_id=body_id,
                onset_substep=frame.substep,
                deadline_substep=frame.substep + self._post_window,
                control_step=frame.control_step,
                onset_position=np.asarray(state.position, dtype=float).copy(),
                onset_speed=speed,
                onset_delta_v=delta_v,
                evidence=recent,
                gate_at_onset=gate,
                max_speed=speed,
                max_delta_v=delta_v,
            )
            self._pending[body_id] = candidate
            events.append(self._event(candidate, frame, EventStatus.CANDIDATE))
        return events

    def _update(
        self,
        candidate: _Candidate,
        frame: SubstepFrame,
        position: np.ndarray,
        speed: float,
        delta_v: float,
        in_contact: bool,
        context: DetectorContext,
    ) -> list[Event]:
        events: list[Event] = []
        position = np.asarray(position, dtype=float)
        candidate.max_distance = max(candidate.max_distance, float(np.linalg.norm(position - candidate.onset_position)))
        candidate.max_height_gain = max(candidate.max_height_gain, float(position[2] - candidate.onset_position[2]))
        candidate.max_speed = max(candidate.max_speed, speed)
        candidate.max_delta_v = max(candidate.max_delta_v, delta_v)
        if in_contact:
            candidate.free_flight = 0
        else:
            candidate.free_flight += 1
            candidate.max_free_flight = max(candidate.max_free_flight, candidate.free_flight)

        if candidate.confirmed_at is None and candidate.max_distance >= self.cfg.min_flight_distance_m:
            reasons: list[str] = []
            if candidate.max_free_flight >= self._min_free_flight:
                reasons.append("ballistic_free_flight")
            if (
                candidate.evidence.risky
                and candidate.max_delta_v >= self._violent_delta_v
                and candidate.max_speed >= self.cfg.violent_speed_mps
            ):
                reasons.append("violent_risky_contact_displacement")
            if reasons:
                candidate.confirmed_at = frame.substep
                candidate.reasons = reasons
                events.append(self._event(candidate, frame, EventStatus.CONFIRMED))

        if frame.substep >= candidate.deadline_substep:
            events.extend(self._close(candidate, frame))
        return events

    def _close(self, candidate: _Candidate, frame: SubstepFrame | None) -> list[Event]:
        self._pending.pop(candidate.body_id, None)
        substep = frame.substep if frame is not None else candidate.deadline_substep
        self._cooldown_until[candidate.body_id] = substep + self._contact_window
        if candidate.confirmed_at is None:
            return [self._event(candidate, frame, EventStatus.REJECTED)]
        return []

    def finalize(self, frame: SubstepFrame | None, context: DetectorContext) -> list[Event]:
        events: list[Event] = []
        for candidate in list(self._pending.values()):
            events.extend(self._close(candidate, frame))
        return events

    def _event(self, candidate: _Candidate, frame: SubstepFrame | None, status: EventStatus) -> Event:
        ev = candidate.evidence
        return Event(
            event_id=candidate.event_id,
            detector=self.name,
            kind="contact_triggered_ejection",
            status=status,
            onset_substep=candidate.onset_substep,
            substep=frame.substep if frame is not None else candidate.deadline_substep,
            control_step=candidate.control_step,
            bodies=(candidate.body_id, *ev.partners),
            metrics={
                "onset_speed_mps": candidate.onset_speed,
                "onset_delta_v_mps": candidate.onset_delta_v,
                "max_speed_mps": candidate.max_speed,
                "max_delta_v_mps": candidate.max_delta_v,
                "max_distance_m": candidate.max_distance,
                "max_height_gain_m": candidate.max_height_gain,
                "max_free_flight_substeps": candidate.max_free_flight,
                "contact_force_estimate_n": ev.max_force_n,
                "contact_impulse_ns": ev.max_impulse_ns,
                "contact_penetration_m": ev.max_penetration_m,
                "risky_contact": ev.risky,
                "confirmed_substep": candidate.confirmed_at,
                "gate_at_onset": candidate.gate_at_onset,
            },
            reasons=list(candidate.reasons),
        )
