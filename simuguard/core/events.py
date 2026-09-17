"""Anomaly events emitted by detectors and routed to human verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventStatus(str, Enum):
    FLAG = "flag"  # single-substep observation, no lifecycle
    CANDIDATE = "candidate"  # onset seen, confirmation window open
    CONFIRMED = "confirmed"  # detector criteria met
    REJECTED = "rejected"  # window closed without confirmation


class ReviewStatus(str, Enum):
    PENDING = "pending"
    NOT_REQUIRED = "not_required"


@dataclass
class Event:
    event_id: str
    detector: str
    kind: str
    status: EventStatus
    onset_substep: int
    substep: int
    control_step: int
    bodies: tuple[str, ...]
    metrics: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    review: ReviewStatus = ReviewStatus.PENDING

    @property
    def is_positive(self) -> bool:
        return self.status in (EventStatus.CONFIRMED, EventStatus.FLAG)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "detector": self.detector,
            "kind": self.kind,
            "status": self.status.value,
            "onset_substep": self.onset_substep,
            "substep": self.substep,
            "control_step": self.control_step,
            "bodies": list(self.bodies),
            "metrics": _jsonable(self.metrics),
            "reasons": list(self.reasons),
            "review": self.review.value,
        }


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    return value
