"""Single-substep signal detectors (flags with per-key cooldown)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..events import Event, EventStatus
from ..types import BodyRole, SubstepFrame, pair_key
from .base import Detector, DetectorContext


class _CooldownFlags(Detector):
    cooldown_ms: float = 100.0

    def reset(self, context: DetectorContext) -> None:
        self._cooldown = max(1, math.ceil(self.cooldown_ms / (1000.0 * context.timestep)))
        self._until: dict[str, int] = {}

    def _allowed(self, key: str, substep: int) -> bool:
        if substep < self._until.get(key, -1):
            return False
        self._until[key] = substep + self._cooldown
        return True

    def _flag(self, context: DetectorContext, frame: SubstepFrame, kind: str, key: str, bodies: tuple[str, ...], metrics: dict[str, Any]) -> Event:
        return Event(
            event_id=f"{context.episode_id}:{self.name}:{key}:{frame.substep}",
            detector=self.name,
            kind=kind,
            status=EventStatus.FLAG,
            onset_substep=frame.substep,
            substep=frame.substep,
            control_step=frame.control_step,
            bodies=bodies,
            metrics=metrics,
        )


@dataclass
class NonFiniteConfig:
    max_linear_speed_mps: float = 100.0


class NonFiniteStateDetector(_CooldownFlags):
    """NaN/Inf state or physically absurd speed on any tracked body (once per body)."""

    name = "non_finite_state"

    def __init__(self, config: NonFiniteConfig | None = None) -> None:
        self.cfg = config or NonFiniteConfig()

    def config(self) -> dict[str, Any]:
        return asdict(self.cfg)

    def reset(self, context: DetectorContext) -> None:
        super().reset(context)
        self._seen: set[str] = set()

    def observe(self, frame: SubstepFrame, context: DetectorContext) -> list[Event]:
        events = []
        for body_id, state in frame.states.items():
            if body_id in self._seen:
                continue
            finite = state.is_finite()
            speed = state.speed if finite else float("inf")
            if finite and speed <= self.cfg.max_linear_speed_mps:
                continue
            self._seen.add(body_id)
            events.append(
                self._flag(
                    context, frame, "non_finite_or_absurd_state", body_id, (body_id,),
                    {"finite": finite, "speed_mps": speed if finite else None},
                )
            )
        return events


@dataclass
class PenetrationConfig:
    max_penetration_m: float = 0.005
    ignore_robot_robot: bool = True
    cooldown_ms: float = 100.0


class PenetrationDetector(_CooldownFlags):
    name = "deep_penetration"

    def __init__(self, config: PenetrationConfig | None = None) -> None:
        self.cfg = config or PenetrationConfig()
        self.cooldown_ms = self.cfg.cooldown_ms

    def config(self) -> dict[str, Any]:
        return asdict(self.cfg)

    def observe(self, frame: SubstepFrame, context: DetectorContext) -> list[Event]:
        events = []
        for pair in frame.contacts:
            depth = pair.max_penetration
            if depth <= self.cfg.max_penetration_m:
                continue
            roles = (context.role(pair.body_a), context.role(pair.body_b))
            if self.cfg.ignore_robot_robot and roles == (BodyRole.ROBOT, BodyRole.ROBOT):
                continue
            key = pair_key(pair.body_a, pair.body_b)
            if self._allowed(key, frame.substep):
                events.append(
                    self._flag(
                        context, frame, "deep_penetration", key, (pair.body_a, pair.body_b),
                        {"penetration_m": depth, "points": pair.point_count, "roles": [r.value for r in roles]},
                    )
                )
        return events


@dataclass
class ImpulseSpikeConfig:
    max_force_estimate_n: float = 200.0
    metric: str = "net"  # "net" = |sum impulse|/dt ; "abs" = sum |impulse|/dt
    require_free_body: bool = True
    cooldown_ms: float = 100.0


class ImpulseSpikeDetector(_CooldownFlags):
    """Contact force estimate (|impulse|/dt) above a bound on a pair with a free body."""

    name = "impulse_spike"

    def __init__(self, config: ImpulseSpikeConfig | None = None) -> None:
        self.cfg = config or ImpulseSpikeConfig()
        self.cooldown_ms = self.cfg.cooldown_ms

    def config(self) -> dict[str, Any]:
        return asdict(self.cfg)

    def observe(self, frame: SubstepFrame, context: DetectorContext) -> list[Event]:
        events = []
        free = set(context.free_bodies())
        for pair in frame.contacts:
            if self.cfg.require_free_body and not ({pair.body_a, pair.body_b} & free):
                continue
            force = pair.force_estimate(frame.timestep) if self.cfg.metric == "net" else pair.force_abs_estimate(frame.timestep)
            if force <= self.cfg.max_force_estimate_n:
                continue
            key = pair_key(pair.body_a, pair.body_b)
            if self._allowed(key, frame.substep):
                events.append(
                    self._flag(
                        context, frame, "impulse_spike", key, (pair.body_a, pair.body_b),
                        {"force_estimate_n": force, "metric": self.cfg.metric, "impulse_ns": pair.impulse_magnitude, "impulse_abs_sum_ns": pair.impulse_abs_sum},
                    )
                )
        return events


@dataclass
class ActuationBoundConfig:
    """Physical-plausibility bound (experimental, P0-b).

    A free object that was just in contact with non-robot geometry should not
    move much faster than the fastest robot link could have pushed it.
    ``speed > ratio * max_robot_link_speed + margin`` is therefore evidence of
    solver energy injection rather than policy-driven motion.
    """

    ratio: float = 3.0
    margin_mps: float = 0.5
    contact_window_ms: float = 40.0
    cooldown_ms: float = 200.0


class ActuationBoundDetector(_CooldownFlags):
    name = "actuation_bound"

    def __init__(self, config: ActuationBoundConfig | None = None) -> None:
        self.cfg = config or ActuationBoundConfig()
        self.cooldown_ms = self.cfg.cooldown_ms

    def config(self) -> dict[str, Any]:
        return asdict(self.cfg)

    def reset(self, context: DetectorContext) -> None:
        super().reset(context)
        self._window = max(1, math.ceil(self.cfg.contact_window_ms / (1000.0 * context.timestep)))
        self._last_env_contact: dict[str, int] = {}
        self._robot_speed_history: list[tuple[int, float]] = []

    def observe(self, frame: SubstepFrame, context: DetectorContext) -> list[Event]:
        robot_ids = [b for b, info in context.bodies.items() if info.role == BodyRole.ROBOT]
        robot_speed = max((frame.states[b].speed for b in robot_ids if b in frame.states), default=0.0)
        self._robot_speed_history.append((frame.substep, robot_speed))
        self._robot_speed_history = [(s, v) for s, v in self._robot_speed_history if frame.substep - s <= self._window]
        bound_speed = max(v for _, v in self._robot_speed_history)

        for pair in frame.contacts:
            for body_id in (pair.body_a, pair.body_b):
                if context.role(pair.other(body_id)) != BodyRole.ROBOT:
                    self._last_env_contact[body_id] = frame.substep

        events = []
        for body_id in context.free_bodies((BodyRole.TARGET, BodyRole.OBJECT)):
            state = frame.states.get(body_id)
            if state is None or not state.is_finite():
                continue
            last = self._last_env_contact.get(body_id)
            if last is None or frame.substep - last > self._window:
                continue
            limit = self.cfg.ratio * bound_speed + self.cfg.margin_mps
            if state.speed > limit and self._allowed(body_id, frame.substep):
                events.append(
                    self._flag(
                        context, frame, "speed_exceeds_actuation_bound", body_id, (body_id,),
                        {
                            "speed_mps": state.speed,
                            "max_robot_link_speed_mps": bound_speed,
                            "limit_mps": limit,
                            "kinetic_energy_j": _kinetic_energy(context, body_id, np.asarray(state.linear_velocity)),
                        },
                    )
                )
        return events


def _kinetic_energy(context: DetectorContext, body_id: str, velocity: np.ndarray) -> float | None:
    info = context.bodies.get(body_id)
    if info is None or info.mass is None:
        return None
    return 0.5 * float(info.mass) * float(velocity @ velocity)
