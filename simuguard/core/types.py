"""Simulator-agnostic ground-truth data model.

Every record here is plain numpy/python so the core (monitor, detectors,
snapshots, recorder) never imports a simulator.  Adapters translate native
simulator objects into these types.

Conventions
-----------
* Units are SI (m, s, kg, N, N*s).
* Quaternions are ``wxyz`` (SAPIEN convention).
* ``substep`` counts physics steps since the monitor attached; ``control_step``
  is the benchmark's policy-action counter at that substep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class BodyKind(str, Enum):
    DYNAMIC = "dynamic"
    KINEMATIC = "kinematic"
    STATIC = "static"
    LINK = "articulation_link"


class BodyRole(str, Enum):
    """Semantic role used by detectors; assigned by the adapter/task spec."""

    TARGET = "target"  # manipulated object whose fate defines task success
    CONTAINER = "container"  # receptacle the target is placed into / onto
    OBJECT = "object"  # any other free rigid object
    ROBOT = "robot"  # robot links (grippers, arms, base)
    SCENE = "scene"  # table, walls, ground and other fixed geometry
    OTHER = "other"


@dataclass(frozen=True)
class BodyInfo:
    """Static (per-episode) metadata about one simulated body."""

    body_id: str
    name: str
    kind: BodyKind
    role: BodyRole
    mass: float | None = None
    articulation: str | None = None
    collision_shape_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_id": self.body_id,
            "name": self.name,
            "kind": self.kind.value,
            "role": self.role.value,
            "mass": self.mass,
            "articulation": self.articulation,
            "collision_shape_count": self.collision_shape_count,
            "metadata": self.metadata,
        }


@dataclass
class BodyState:
    """Ground-truth kinematic state of one body at one substep."""

    body_id: str
    position: np.ndarray  # (3,) world position of the body frame
    quaternion: np.ndarray  # (4,) wxyz
    linear_velocity: np.ndarray  # (3,) centre-of-mass linear velocity
    angular_velocity: np.ndarray  # (3,)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.linear_velocity))

    def is_finite(self) -> bool:
        return bool(
            np.all(np.isfinite(self.position))
            and np.all(np.isfinite(self.quaternion))
            and np.all(np.isfinite(self.linear_velocity))
            and np.all(np.isfinite(self.angular_velocity))
        )

    def to_dict(self, decimals: int | None = 7) -> dict[str, Any]:
        return {
            "p": _rounded(self.position, decimals),
            "q": _rounded(self.quaternion, decimals),
            "v": _rounded(self.linear_velocity, decimals),
            "w": _rounded(self.angular_velocity, decimals),
        }


@dataclass
class ContactPoint:
    position: np.ndarray  # (3,)
    normal: np.ndarray  # (3,)
    impulse: np.ndarray  # (3,) impulse applied during the substep, N*s
    separation: float  # < 0 means penetration


@dataclass
class ContactPair:
    """All contact points between two bodies reported for one substep."""

    body_a: str
    body_b: str
    points: list[ContactPoint]
    shape_pairs: int = 1  # simulator shape-pair manifolds merged into this body pair

    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def total_impulse(self) -> np.ndarray:
        if not self.points:
            return np.zeros(3)
        return np.sum([point.impulse for point in self.points], axis=0)

    @property
    def impulse_magnitude(self) -> float:
        return float(np.linalg.norm(self.total_impulse))

    @property
    def impulse_abs_sum(self) -> float:
        """Sum of per-point impulse magnitudes (does not cancel across hulls, e.g. a squeezing grasp)."""

        return float(sum(np.linalg.norm(point.impulse) for point in self.points))

    @property
    def max_penetration(self) -> float:
        if not self.points:
            return 0.0
        return float(max(0.0, -min(point.separation for point in self.points)))

    def force_estimate(self, timestep: float) -> float:
        """Mean net contact force over the substep, |sum impulse| / dt."""

        return self.impulse_magnitude / max(float(timestep), 1.0e-12)

    def force_abs_estimate(self, timestep: float) -> float:
        """Sum of per-point force magnitudes over the substep, sum |impulse| / dt."""

        return self.impulse_abs_sum / max(float(timestep), 1.0e-12)

    def involves(self, body_id: str) -> bool:
        return body_id in (self.body_a, self.body_b)

    def other(self, body_id: str) -> str:
        return self.body_b if self.body_a == body_id else self.body_a

    def to_dict(self, timestep: float) -> dict[str, Any]:
        return {
            "a": self.body_a,
            "b": self.body_b,
            "points": self.point_count,
            "shape_pairs": self.shape_pairs,
            "impulse": round(self.impulse_magnitude, 9),
            "force_est": round(self.force_estimate(timestep), 6),
            "force_abs_est": round(self.force_abs_estimate(timestep), 6),
            "penetration": round(self.max_penetration, 9),
        }


@dataclass
class SubstepFrame:
    """Everything observed immediately after one physics substep."""

    substep: int
    sim_time: float
    control_step: int
    timestep: float
    states: dict[str, BodyState]
    contacts: list[ContactPair]
    extras: dict[str, Any] = field(default_factory=dict)

    def contacts_of(self, body_id: str) -> list[ContactPair]:
        return [pair for pair in self.contacts if pair.involves(body_id)]

    def to_dict(self, body_ids: set[str] | None = None, decimals: int | None = 7) -> dict[str, Any]:
        """``decimals=None`` keeps full float precision (required for replay references)."""

        states = self.states if body_ids is None else {k: v for k, v in self.states.items() if k in body_ids}
        contacts = (
            self.contacts
            if body_ids is None
            else [pair for pair in self.contacts if pair.body_a in body_ids or pair.body_b in body_ids]
        )
        payload = {
            "substep": self.substep,
            "t": round(self.sim_time, 9),
            "control_step": self.control_step,
            "states": {key: value.to_dict(decimals) for key, value in sorted(states.items())},
            "contacts": [pair.to_dict(self.timestep) for pair in contacts],
        }
        if self.extras:
            payload["extras"] = self.extras
        return payload


def _rounded(values: np.ndarray, decimals: int | None) -> list[float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    return (array if decimals is None else np.round(array, decimals)).tolist()


def pair_key(body_a: str, body_b: str) -> str:
    return "|".join(sorted((body_a, body_b)))
