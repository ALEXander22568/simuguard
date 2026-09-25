"""Simulator-free helpers for the Isaac Sim adapter.

Everything that does not need ``omni`` / ``pxr`` lives here so it can be unit
tested on any machine: tensor layout conversion, actor-path resolution for
contact reports, and merging of PhysX contact-report records into
:class:`ContactPair` objects.

PhysX conventions used below
----------------------------
* Physics tensor views return poses as ``[x, y, z, qx, qy, qz, qw]`` and
  velocities as ``[vx, vy, vz, wx, wy, wz]`` of the centre of mass.
* A contact-report header names two actors; each of its points carries the
  normal along which actor0 must move to separate from actor1 and the impulse
  applied to actor0 during the step (normal impulse only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from ...core.types import BodyState, ContactPair, ContactPoint


def pose_xyzw_to_wxyz(poses: np.ndarray) -> np.ndarray:
    """(N, 7) ``[p, qx, qy, qz, qw]`` -> (N, 7) ``[p, qw, qx, qy, qz]``."""

    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 7)
    return np.concatenate([poses[:, :3], poses[:, 6:7], poses[:, 3:6]], axis=1)


def pose_wxyz_to_xyzw(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 7)
    return np.concatenate([poses[:, :3], poses[:, 4:7], poses[:, 3:4]], axis=1)


def states_from_arrays(
    body_ids: list[str],
    poses_xyzw: np.ndarray,
    velocities: np.ndarray,
    rows: Iterable[int] | None = None,
) -> dict[str, BodyState]:
    """BodyState per body from one tensor-view read (``rows`` selects view rows)."""

    poses = pose_xyzw_to_wxyz(poses_xyzw)
    vel = np.asarray(velocities, dtype=np.float64).reshape(-1, 6)
    out: dict[str, BodyState] = {}
    for body_id, row in zip(body_ids, range(len(body_ids)) if rows is None else rows):
        out[body_id] = BodyState(
            body_id=body_id,
            position=poses[row, :3].copy(),
            quaternion=poses[row, 3:7].copy(),
            linear_velocity=vel[row, :3].copy(),
            angular_velocity=vel[row, 3:6].copy(),
        )
    return out


class PathResolver:
    """Map PhysX actor prim paths to SimuGuard body ids.

    Exact paths are registered for rigid bodies and articulation links; a static
    object made of several collider prims is registered once by its root path and
    matched by prefix.  Unknown paths fall back to ``fallback(path)`` (and are
    remembered, so each path is resolved once).
    """

    def __init__(
        self,
        exact: dict[str, str] | None = None,
        prefixes: dict[str, str] | None = None,
        fallback: Callable[[str], str] | None = None,
    ) -> None:
        self.exact = dict(exact or {})
        self.prefixes = sorted((dict(prefixes or {})).items(), key=lambda item: -len(item[0]))
        self.fallback = fallback or (lambda path: f"unknown:{path}")
        self._cache: dict[Any, str] = {}
        self.unknown: dict[str, str] = {}

    def resolve_path(self, path: str) -> str:
        body_id = self.exact.get(path)
        if body_id is not None:
            return body_id
        # a collider below a registered rigid body reports the body as its actor, but be tolerant
        for prefix, prefix_id in self.prefixes:
            if path == prefix or path.startswith(prefix + "/"):
                return prefix_id
        for exact_path, exact_id in self.exact.items():
            if path.startswith(exact_path + "/"):
                return exact_id
        body_id = self.fallback(path)
        self.unknown[path] = body_id
        return body_id

    def resolve(self, key: Any, decode: Callable[[Any], str]) -> str:
        """Resolve an encoded path (e.g. the int64 of a contact header), caching per key."""

        body_id = self._cache.get(key)
        if body_id is None:
            body_id = self.resolve_path(decode(key))
            self._cache[key] = body_id
        return body_id


@dataclass
class RawContactHeader:
    """Minimal view of an omni.physx contact-report header (for tests and adapters)."""

    actor0: Any
    actor1: Any
    offset: int
    count: int


def merge_contact_records(
    headers: Iterable[Any],
    points: list[Any],
    resolve: Callable[[Any], str],
    *,
    keep: Callable[[str, str], bool] | None = None,
    header_fields: tuple[str, str, str, str] = ("actor0", "actor1", "contact_data_offset", "num_contact_data"),
) -> list[ContactPair]:
    """Merge contact-report headers (one per shape pair) into one ContactPair per body pair.

    ``points[i]`` must expose ``position``, ``normal``, ``impulse`` (3-vectors) and
    ``separation``.  Impulses and normals are oriented so that ``total_impulse`` acts
    on ``body_a`` (the lexicographically smaller id), matching the other adapters.
    Headers without points (contact lost) are skipped.
    """

    f_a0, f_a1, f_off, f_n = header_fields
    merged: dict[tuple[str, str], ContactPair] = {}
    for header in headers:
        count = int(getattr(header, f_n))
        if count <= 0:
            continue
        a = resolve(getattr(header, f_a0))
        b = resolve(getattr(header, f_a1))
        if a == b:
            continue  # two colliders of the same body (e.g. a static group): not a body pair
        if keep is not None and not keep(a, b):
            continue
        key, sign = ((a, b), 1.0) if a <= b else ((b, a), -1.0)
        pair = merged.get(key)
        if pair is None:
            pair = merged[key] = ContactPair(key[0], key[1], [], shape_pairs=0)
        pair.shape_pairs += 1
        start = int(getattr(header, f_off))
        # one pass over the pybind structs, one array per header (per-point numpy ops were the cost)
        rows = np.array([_point_row(p) for p in points[start : start + count]], dtype=float).reshape(-1, 10)
        if sign < 0.0:
            rows[:, 3:9] *= -1.0
        pair.points.extend(ContactPoint(r[0:3], r[3:6], r[6:9], float(r[9])) for r in rows)
    return list(merged.values())


def _xyz(value: Any) -> tuple[float, float, float]:
    try:
        return value.x, value.y, value.z
    except AttributeError:
        x, y, z = np.asarray(value, dtype=float).reshape(3)
        return x, y, z


def _point_row(point: Any) -> tuple[float, ...]:
    """position(3) normal(3) impulse(3) separation of one contact-report point."""

    return (*_xyz(point.position), *_xyz(point.normal), *_xyz(point.impulse), point.separation)


def changed_entries(now: np.ndarray, then: np.ndarray) -> np.ndarray:
    """Flat indices where two equally shaped arrays differ (NaN == NaN)."""

    now = np.asarray(now).reshape(-1)
    then = np.asarray(then).reshape(-1)
    if now.shape != then.shape:
        return np.arange(now.size)
    differ = ~((now == then) | (np.isnan(now) & np.isnan(then)))
    return np.flatnonzero(differ)
