"""Simulator-free parts of the RoboLab (Isaac Lab, batched GPU envs) integration.

* :class:`WriteLog`: every call Isaac Lab makes into the PhysX tensor API between two
  physics steps (joint targets and efforts, root poses/velocities written by resets,
  kinematic updates), tagged with the physics step it precedes.  Replaying the log in a
  freshly built env of the same task, seed and ``num_envs`` reproduces the whole batch.
* :class:`CommandBuffer`: the actuation in effect per env, rebuilt from the write log.
* :func:`contact_pairs_from_data`: PhysX rigid-contact-view buffers -> ContactPairs per env.
* :class:`DepenetrationCap` / :func:`parse_cap`: max depenetration velocity intervention
  (value, bodies, envs, substep window).
* :func:`ejection_config`: detector thresholds for RoboLab's 1/120 s step.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from ...core.types import ContactPair, ContactPoint

WRITE_LOG_SCHEMA = "simuguard-robolab-write-log-v1"
CONTACT_LOG_SCHEMA = "simuguard-robolab-contact-log-v1"
ENV_PATH = re.compile(r"^/World/envs/env_(\d+)/(.+)$")
DETECTOR_REFERENCE_TIMESTEP_S = 0.004  # thresholds on velocity change are given for a 4 ms step

# view methods whose calls change the simulation (Isaac Lab write_* / reset paths)
ACTUATION_METHODS = ("set_dof_position_targets", "set_dof_velocity_targets", "set_dof_actuation_forces")


def split_env_path(path: str) -> tuple[int, str] | None:
    """``/World/envs/env_3/scene/banana`` -> ``(3, "scene/banana")``."""

    match = ENV_PATH.match(path)
    return (int(match.group(1)), match.group(2)) if match else None


def segment_dir(record_dir: Path, seg: dict[str, Any]) -> Path:
    """A recorded segment's directory under ``record_dir`` (``run_k/env_e/seg_s``), wherever the batch was recorded or copied to."""

    return Path(record_dir).joinpath(*Path(seg["dir"]).parts[-3:])


def body_id_from_relative(relative: str) -> str:
    """``scene/banana`` -> ``banana``; robot links keep their articulation prefix (``robot/panda_hand``)."""

    return relative[len("scene/"):] if relative.startswith("scene/") else relative


def ejection_config(base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Detector config for RoboLab: defaults unchanged, |dv| thresholds referenced to 4 ms."""

    cfg = json.loads(json.dumps(base or {}))
    cfg.setdefault("ejection", {}).setdefault("delta_v_reference_timestep_s", DETECTOR_REFERENCE_TIMESTEP_S)
    return cfg


def effective_ejection_thresholds(ejection: dict[str, Any], dt: float) -> dict[str, float]:
    """The per-substep values the ejection detector applies at timestep ``dt``."""

    import math

    from ...core.detectors.ejection import EjectionConfig

    cfg = EjectionConfig.from_dict(ejection)
    scale = dt / cfg.delta_v_reference_timestep_s if cfg.delta_v_reference_timestep_s else 1.0
    return {
        "timestep_s": dt,
        "scale": scale,
        "min_delta_v_mps": cfg.min_delta_v_mps * scale,
        "violent_delta_v_mps": cfg.violent_delta_v_mps * scale,
        "contact_window_substeps": max(1, math.ceil(cfg.contact_window_ms / (1000.0 * dt))),
        "post_window_substeps": max(1, math.ceil(cfg.post_window_ms / (1000.0 * dt))),
        "min_free_flight_substeps": max(1, math.ceil(cfg.min_free_flight_ms / (1000.0 * dt))),
    }


# ---------------------------------------------------------------------------- write log
@dataclass
class Write:
    substep: int  # the physics step this write precedes (1-based, counted from install)
    target: str  # view key, e.g. "articulation:robot", "rigid:banana", "sim_view"
    method: str
    indices: np.ndarray | None  # view rows written (env ids for per-env views), original dtype
    data: np.ndarray | None  # rows of the written buffer at ``indices``, original dtype


class WriteLog:
    """Ordered tensor-API writes; saved per run as one npz."""

    def __init__(self) -> None:
        self.writes: list[Write] = []

    def __len__(self) -> int:
        return len(self.writes)

    def append(self, substep: int, target: str, method: str, indices: Any = None, data: Any = None) -> None:
        idx = None if indices is None else np.asarray(indices).reshape(-1).copy()
        arr = None if data is None else np.array(data, copy=True)
        if arr is not None and idx is not None and arr.shape[0] != idx.size:
            arr = arr[idx]
        self.writes.append(Write(int(substep), target, method, idx, arr))

    def since(self, position: int) -> list[Write]:
        return self.writes[position:]

    def by_substep(self) -> dict[int, list[Write]]:
        out: dict[int, list[Write]] = {}
        for write in self.writes:
            out.setdefault(write.substep, []).append(write)
        return out

    def substeps(self) -> tuple[int, int] | None:
        if not self.writes:
            return None
        return self.writes[0].substep, self.writes[-1].substep

    @staticmethod
    def save(writes: list[Write], path: str | Path, meta: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        names = sorted({w.target for w in writes} | {w.method for w in writes})
        code = {n: i for i, n in enumerate(names)}
        blobs: dict[str, list[np.ndarray]] = {}
        offsets: dict[str, int] = {}
        table = np.zeros((len(writes), 10), dtype=np.int64)
        dtypes: list[str] = []
        idx_parts: list[np.ndarray] = []
        idx_offset = 0
        for i, w in enumerate(writes):
            n_idx = -1 if w.indices is None else w.indices.size
            if w.indices is not None:
                idx_parts.append(w.indices.astype(np.int64))
            dtype = "" if w.data is None else w.data.dtype.str
            idx_dtype = "" if w.indices is None else w.indices.dtype.str
            for name in (dtype, idx_dtype):
                if name not in dtypes:
                    dtypes.append(name)
            if w.data is not None:
                flat = np.ascontiguousarray(w.data).reshape(-1)
                start = offsets.get(dtype, 0)
                blobs.setdefault(dtype, []).append(flat)
                offsets[dtype] = start + flat.size
                rows = w.data.shape[0] if w.data.ndim else 1
                width = flat.size // max(rows, 1)
            else:
                start, rows, width = 0, -1, 0
            table[i] = (w.substep, code[w.target], code[w.method], idx_offset, n_idx, dtypes.index(idx_dtype),
                        dtypes.index(dtype), start, rows, width)
            idx_offset += max(n_idx, 0)
        arrays = {f"data_{dtypes.index(d)}": np.concatenate(blobs[d]) for d in dtypes if d in blobs}
        np.savez_compressed(
            path,
            schema=np.array(WRITE_LOG_SCHEMA),
            names=np.array(json.dumps(names)),
            dtypes=np.array(json.dumps(dtypes)),
            meta=np.array(json.dumps(meta or {})),
            shapes=np.array(json.dumps([None if w.data is None else list(w.data.shape[1:]) for w in writes])),
            table=table,
            indices=np.concatenate(idx_parts) if idx_parts else np.zeros(0, dtype=np.int64),
            **arrays,
        )
        return path

    @staticmethod
    def load(path: str | Path) -> tuple[list[Write], dict[str, Any]]:
        with np.load(path, allow_pickle=False) as data:
            if str(data["schema"]) != WRITE_LOG_SCHEMA:
                raise ValueError(f"not a RoboLab write log: {path}")
            names = json.loads(str(data["names"]))
            dtypes = json.loads(str(data["dtypes"]))
            meta = json.loads(str(data["meta"]))
            shapes = json.loads(str(data["shapes"]))
            table = data["table"]
            indices = data["indices"]
            blobs = {i: data[f"data_{i}"] for i in range(len(dtypes)) if f"data_{i}" in data}
        writes: list[Write] = []
        for k, (substep, target, method, i0, n_idx, it_i, dt_i, start, rows, width) in enumerate(table.tolist()):
            idx = None if n_idx < 0 else indices[i0 : i0 + n_idx].astype(np.dtype(dtypes[it_i]))
            arr = None
            if rows >= 0:
                arr = blobs[dt_i][start : start + rows * width].astype(np.dtype(dtypes[dt_i]), copy=True).reshape(rows, *shapes[k])
            writes.append(Write(substep, names[target], names[method], idx, arr))
        return writes, meta


def iter_env_rows(write: Write) -> Iterator[tuple[int, np.ndarray]]:
    """(env id, written row) for a per-env view write."""

    if write.indices is None or write.data is None:
        return
    for row, env_id in enumerate(write.indices.tolist()):
        yield int(env_id), write.data[row]


class CommandBuffer:
    """Latest actuation command per env and method, fed from tensor-API writes."""

    def __init__(self, num_envs: int, methods: Iterable[str] = ACTUATION_METHODS) -> None:
        self.num_envs = int(num_envs)
        self.methods = tuple(methods)
        self._rows: dict[str, list[np.ndarray | None]] = {m: [None] * self.num_envs for m in self.methods}

    def update(self, write: Write) -> None:
        rows = self._rows.get(write.method)
        if rows is None:
            return
        for env_id, row in iter_env_rows(write):
            rows[env_id] = np.array(row, copy=True)

    def payload(self, env_id: int) -> dict[str, list[float] | None]:
        out: dict[str, list[float] | None] = {}
        for method in self.methods:
            row = self._rows[method][env_id]
            out[method[len("set_dof_"):]] = None if row is None else [float(x) for x in row]
        return out


# ---------------------------------------------------------------------------- contacts
class ArrayContactPair(ContactPair):
    """A ContactPair backed by (n, 3) arrays: aggregates are vectorized, points are built on demand."""

    def __init__(self, body_a: str, body_b: str, positions: np.ndarray, normals: np.ndarray, impulses: np.ndarray,
                 separations: np.ndarray) -> None:
        self.body_a = body_a
        self.body_b = body_b
        self.positions = positions
        self.normals = normals
        self.impulses = impulses
        self.separations = separations
        self.shape_pairs = len(separations)

    @property
    def points(self) -> list[ContactPoint]:
        return [ContactPoint(self.positions[k].copy(), self.normals[k].copy(), self.impulses[k].copy(), float(self.separations[k]))
                for k in range(len(self.separations))]

    @property
    def point_count(self) -> int:
        return len(self.separations)

    @property
    def total_impulse(self) -> np.ndarray:
        return self.impulses.sum(axis=0) if len(self.separations) else np.zeros(3)

    @property
    def impulse_abs_sum(self) -> float:
        return float(np.linalg.norm(self.impulses, axis=1).sum())

    @property
    def max_penetration(self) -> float:
        return float(max(0.0, -self.separations.min())) if len(self.separations) else 0.0


def contact_pairs_from_data(
    sensor_body: str,
    filter_bodies: list[str],
    forces: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    separations: np.ndarray,
    counts: np.ndarray,
    starts: np.ndarray,
    dt: float,
    skip_filters: Iterable[int] = (),
) -> dict[int, list[ContactPair]]:
    """ContactPairs per env from one rigid contact view (one sensor body per env, row = env id).

    Each data row is one contact patch with a normal force.  The impulse of the step,
    force * dt along the reported normal, is assigned to ``body_a`` with the same sign
    convention as the other adapters (flipped when the sensor body is ``body_b``).
    ``skip_filters`` drops filter columns already reported from the other side.
    """

    counts = np.asarray(counts).reshape(len(counts), -1)
    starts = np.asarray(starts).reshape(counts.shape)
    forces = np.asarray(forces, dtype=float).reshape(-1)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    separations = np.asarray(separations, dtype=float).reshape(-1)
    skip = set(skip_filters)
    out: dict[int, list[ContactPair]] = {}
    envs, cols = np.nonzero(counts)
    for env_id, col in zip(envs.tolist(), cols.tolist()):
        if col in skip:
            continue
        other = filter_bodies[col]
        if other == sensor_body:
            continue
        start = int(starts[env_id, col])
        n = min(int(counts[env_id, col]), max(forces.size - start, 0))
        if n == 0:
            continue
        sl = slice(start, start + n)
        sign = 1.0 if sensor_body <= other else -1.0
        a, b = (sensor_body, other) if sign > 0 else (other, sensor_body)
        normal = normals[sl] * sign
        impulse = normal * (forces[sl] * dt)[:, None]
        out.setdefault(env_id, []).append(ArrayContactPair(a, b, points[sl], normal, impulse, separations[sl]))
    return out


class ContactLog:
    """Per-point contacts of one env, appended per substep, saved as float32 npz (not a replay input)."""

    def __init__(self) -> None:
        self.names: list[str] = []
        self._code: dict[str, int] = {}
        self._rows: list[np.ndarray] = []

    def _id(self, name: str) -> int:
        if name not in self._code:
            self._code[name] = len(self.names)
            self.names.append(name)
        return self._code[name]

    def append(self, substep: int, pairs: list[ContactPair]) -> None:
        for pair in pairs:
            if isinstance(pair, ArrayContactPair):
                pos, nrm, imp, sep = pair.positions, pair.normals, pair.impulses, pair.separations
            else:
                pts = pair.points
                pos = [p.position for p in pts]
                nrm = [p.normal for p in pts]
                imp = [p.impulse for p in pts]
                sep = [p.separation for p in pts]
            if not len(sep):
                continue
            rows = np.empty((len(sep), 13), dtype=np.float32)
            rows[:, 0] = substep
            rows[:, 1] = self._id(pair.body_a)
            rows[:, 2] = self._id(pair.body_b)
            rows[:, 3:6] = pos
            rows[:, 6:9] = nrm
            rows[:, 9:12] = imp
            rows[:, 12] = sep
            self._rows.append(rows)

    def __len__(self) -> int:
        return sum(len(r) for r in self._rows)

    def save(self, path: str | Path) -> Path:
        rows = np.concatenate(self._rows) if self._rows else np.zeros((0, 13), dtype=np.float32)
        np.savez_compressed(
            path,
            schema=np.array(CONTACT_LOG_SCHEMA),
            names=np.array(json.dumps(self.names)),
            columns=np.array(json.dumps(["substep", "body_a", "body_b", "px", "py", "pz", "nx", "ny", "nz", "jx", "jy", "jz", "separation"])),
            rows=rows,
        )
        return Path(path)


# ---------------------------------------------------------------------------- intervention
@dataclass(frozen=True)
class DepenetrationCap:
    """Cap PhysX ``maxDepenetrationVelocity`` (m/s) on selected bodies for a substep window.

    ``bodies``: body ids (``banana``, ``robot/panda_link7``) or None for every free (dynamic,
    non-robot) object; robot links only when listed.  ``envs``: env ids or None for all.  The cap is on from the physics step
    ``start`` (1-based, counted like the recorded substeps) until right before step ``end``,
    where the recorded values are restored; ``end=None`` keeps it on.
    """

    value: float
    bodies: tuple[str, ...] | None = None
    envs: tuple[int, ...] | None = None
    start: int = 1
    end: int | None = None

    def active(self, substep: int) -> bool:
        return substep >= self.start and (self.end is None or substep < self.end)

    def selects(self, env_id: int, body_id: str) -> bool:
        return (self.envs is None or env_id in self.envs) and (self.bodies is None or body_id in self.bodies)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_cap(spec: str) -> DepenetrationCap:
    """``"1.0"`` or ``"1.0,bodies=banana+bowl,envs=0+3,from=1200,to=1500"``."""

    head, *items = [s.strip() for s in spec.split(",") if s.strip()]
    kwargs: dict[str, Any] = {"value": float(head)}
    for item in items:
        key, _, value = item.partition("=")
        if key == "bodies":
            kwargs["bodies"] = tuple(v for v in value.split("+") if v)
        elif key == "envs":
            kwargs["envs"] = tuple(int(v) for v in value.split("+") if v)
        elif key == "from":
            kwargs["start"] = int(value)
        elif key == "to":
            kwargs["end"] = int(value)
        else:
            raise ValueError(f"unknown cap field {key!r} in {spec!r}")
    cap = DepenetrationCap(**kwargs)
    if cap.end is not None and cap.end <= cap.start:
        raise ValueError(f"empty cap window in {spec!r}")
    return cap


def cap_transitions(caps: list[DepenetrationCap], substep: int) -> list[tuple[DepenetrationCap, bool]]:
    """Caps switching on (True) or off (False) right before physics step ``substep``."""

    out: list[tuple[DepenetrationCap, bool]] = []
    for cap in caps:
        if substep == cap.start:
            out.append((cap, True))
        if cap.end is not None and substep == cap.end:
            out.append((cap, False))
    return out


# ---------------------------------------------------------------------------- provenance
def sha256_file(path: str | Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tree_sha256(root: str | Path, pattern: str = "**/*.py") -> str:
    """One hash over the relative paths and bytes of the matching files, in sorted order."""
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(root.glob(pattern)):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def server_info(path: str | Path | None) -> dict[str, Any] | None:
    """The policy server description at ``path`` (written when the server was started), plus whether its pid lives."""
    if path is None:
        return None
    info = json.loads(Path(path).read_text())
    try:
        os.kill(info["pid"], 0)
        info["pid_alive"] = True
    except ProcessLookupError:
        info["pid_alive"] = False
    except PermissionError:
        info["pid_alive"] = True
    return info


def freeze_text(distributions: Iterable[tuple[str, str]]) -> str:
    """``pip freeze``-style text, one ``name==version`` per line, sorted case-insensitively."""

    lines = sorted({f"{name}=={version}" for name, version in distributions}, key=str.lower)
    return "\n".join(lines) + "\n"


def compare_state_arrays(recorded: np.ndarray, replayed: np.ndarray) -> dict[str, Any]:
    """Max abs difference and bit equality of two equally shaped float arrays."""

    recorded = np.asarray(recorded, dtype=np.float64)
    replayed = np.asarray(replayed, dtype=np.float64)
    if recorded.shape != replayed.shape:
        return {"shape_recorded": list(recorded.shape), "shape_replayed": list(replayed.shape), "bit_exact": False}
    diff = np.abs(recorded - replayed)
    differ = np.flatnonzero(~((recorded == replayed) | (np.isnan(recorded) & np.isnan(replayed))).reshape(-1))
    return {
        "values": int(recorded.size),
        "max_abs_diff": float(np.nanmax(diff)) if diff.size else 0.0,
        "differing_values": int(differ.size),
        "bit_exact": bool(differ.size == 0),
    }
