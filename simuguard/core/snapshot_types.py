"""Plain record types shared by snapshots and control logs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ControlRecord:
    """Actuation state in effect for one substep (adapter-specific payload)."""

    substep: int
    control_step: int
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"substep": self.substep, "control_step": self.control_step, "payload": self.payload}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ControlRecord":
        return cls(int(data["substep"]), int(data["control_step"]), data["payload"])
