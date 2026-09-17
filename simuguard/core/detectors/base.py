"""Detector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..events import Event
from ..types import BodyInfo, BodyKind, BodyRole, SubstepFrame


@dataclass
class DetectorContext:
    episode_id: str
    timestep: float
    bodies: dict[str, BodyInfo]
    previous: SubstepFrame | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def role(self, body_id: str) -> BodyRole:
        info = self.bodies.get(body_id)
        return info.role if info is not None else BodyRole.OTHER

    def free_bodies(self, roles: tuple[BodyRole, ...] = (BodyRole.TARGET, BodyRole.CONTAINER, BodyRole.OBJECT)) -> list[str]:
        """Dynamic (non-kinematic, non-static, non-robot) bodies with the given roles."""

        return [
            body_id
            for body_id, info in self.bodies.items()
            if info.kind == BodyKind.DYNAMIC and info.role in roles
        ]


class Detector(ABC):
    """Stateful per-episode detector.

    ``observe`` is called once per substep and returns *new or updated* events.
    Detectors must be pure observers: they never touch the simulator.
    """

    name: str = "detector"

    def reset(self, context: DetectorContext) -> None:
        """Called when the monitor attaches to a new episode."""

    @abstractmethod
    def observe(self, frame: SubstepFrame, context: DetectorContext) -> list[Event]: ...

    def finalize(self, frame: SubstepFrame | None, context: DetectorContext) -> list[Event]:
        """Close any open windows at episode end."""

        return []

    def config(self) -> dict[str, Any]:
        return {}
