"""Simulator state snapshots, control logs and replay bundles.

Design (motivated by the TwinGuard replay-fidelity failures): re-running a
policy from the same env seed does not reproduce a contact event, so replay
must start from state saved *inside the same rollout*.  The monitor keeps

* a :class:`SnapshotRing` of periodic full snapshots (public state + optional
  native solver state such as a PhysX pack), and
* a :class:`ControlLog` of the actuation commands applied at every substep.

When a detector fires, :func:`build_replay_bundle` combines the latest
snapshot before the trigger with the controls that followed it.  Replaying
that bundle under the original settings must reproduce the event before any
intervention (decomposition, solver, mass...) is interpreted.
"""

from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .controls import ControlLog
from .snapshot_types import ControlRecord

SNAPSHOT_SCHEMA = "simuguard-snapshot-v1"
BUNDLE_SCHEMA = "simuguard-replay-bundle-v1"


def json_sha256(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass
class Snapshot:
    substep: int
    control_step: int
    public_state: dict[str, Any]
    control: ControlRecord
    native_state: bytes | None = None
    native_format: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SNAPSHOT_SCHEMA

    @property
    def public_sha256(self) -> str:
        return json_sha256(self.public_state)

    @property
    def native_sha256(self) -> str | None:
        return hashlib.sha256(self.native_state).hexdigest() if self.native_state is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "substep": self.substep,
            "control_step": self.control_step,
            "public_state": self.public_state,
            "public_sha256": self.public_sha256,
            "control": self.control.to_dict(),
            "native_format": self.native_format,
            "native_sha256": self.native_sha256,
            "native_state_b64": (
                base64.b64encode(self.native_state).decode("ascii") if self.native_state is not None else None
            ),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Snapshot":
        if data.get("schema_version") != SNAPSHOT_SCHEMA:
            raise ValueError(f"unsupported snapshot schema: {data.get('schema_version')!r}")
        native = data.get("native_state_b64")
        snapshot = cls(
            substep=int(data["substep"]),
            control_step=int(data["control_step"]),
            public_state=data["public_state"],
            control=ControlRecord.from_dict(data["control"]),
            native_state=base64.b64decode(native) if native else None,
            native_format=data.get("native_format"),
            metadata=data.get("metadata", {}),
        )
        if data.get("public_sha256") and snapshot.public_sha256 != data["public_sha256"]:
            raise ValueError("snapshot public_state hash mismatch")
        if data.get("native_sha256") and snapshot.native_sha256 != data["native_sha256"]:
            raise ValueError("snapshot native_state hash mismatch")
        return snapshot

    def save(self, path: str | Path) -> Path:
        return _write_json_gz(Path(path), self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "Snapshot":
        return cls.from_dict(_read_json_gz(Path(path)))


@dataclass
class RestoreReport:
    method: str
    substep: int
    missing: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)
    public_state_error: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return not self.missing and not self.mismatched

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "substep": self.substep,
            "ok": self.ok,
            "missing": self.missing,
            "mismatched": self.mismatched,
            "public_state_error": self.public_state_error,
        }


class SnapshotRing:
    """Periodic snapshots with bounded memory and pinning on triggers."""

    def __init__(self, capacity: int = 16, interval_substeps: int = 100) -> None:
        if capacity < 1 or interval_substeps < 1:
            raise ValueError("capacity and interval_substeps must be >= 1")
        self.capacity = int(capacity)
        self.interval = int(interval_substeps)
        self._ring: "OrderedDict[int, Snapshot]" = OrderedDict()
        self._pinned: dict[int, Snapshot] = {}

    def due(self, substep: int) -> bool:
        return substep % self.interval == 0

    def add(self, snapshot: Snapshot) -> None:
        self._ring[snapshot.substep] = snapshot
        while len(self._ring) > self.capacity:
            self._ring.popitem(last=False)

    def maybe_capture(self, substep: int, capture: Callable[[], Snapshot]) -> Snapshot | None:
        if not self.due(substep):
            return None
        snapshot = capture()
        self.add(snapshot)
        return snapshot

    def latest_at_or_before(self, substep: int) -> Snapshot | None:
        candidates = [s for key, s in {**self._ring, **self._pinned}.items() if key <= substep]
        return max(candidates, key=lambda s: s.substep) if candidates else None

    def pin_before(self, substep: int) -> Snapshot | None:
        snapshot = self.latest_at_or_before(substep)
        if snapshot is not None:
            self._pinned[snapshot.substep] = snapshot
        return snapshot

    def __len__(self) -> int:
        return len(set(self._ring) | set(self._pinned))

    def substeps(self) -> list[int]:
        return sorted(set(self._ring) | set(self._pinned))


@dataclass
class ReplayBundle:
    """Snapshot + subsequent controls + reference trace for one trigger."""

    snapshot: Snapshot
    controls: list[ControlRecord]
    reference_frames: list[dict[str, Any]]
    trigger: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = BUNDLE_SCHEMA

    @property
    def start_substep(self) -> int:
        return self.snapshot.substep

    @property
    def end_substep(self) -> int:
        return self.controls[-1].substep if self.controls else self.snapshot.substep

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trigger": self.trigger,
            "metadata": self.metadata,
            "snapshot": self.snapshot.to_dict(),
            "controls_encoding": "control_log_npz_b64",
            "controls": base64.b64encode(ControlLog.from_records(self.controls).to_bytes()).decode("ascii"),
            "reference_frames": self.reference_frames,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplayBundle":
        if data.get("schema_version") != BUNDLE_SCHEMA:
            raise ValueError(f"unsupported bundle schema: {data.get('schema_version')!r}")
        return cls(
            snapshot=Snapshot.from_dict(data["snapshot"]),
            controls=_decode_controls(data),
            reference_frames=data.get("reference_frames", []),
            trigger=data.get("trigger", {}),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: str | Path) -> Path:
        return _write_json_gz(Path(path), self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBundle":
        return cls.from_dict(_read_json_gz(Path(path)))


def build_replay_bundle(
    ring: SnapshotRing,
    controls: ControlLog,
    *,
    trigger_substep: int,
    end_substep: int,
    reference_frames: list[dict[str, Any]] | None = None,
    trigger: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ReplayBundle:
    snapshot = ring.latest_at_or_before(trigger_substep)
    if snapshot is None:
        raise LookupError(f"no snapshot at or before substep {trigger_substep}")
    records = controls.between(snapshot.substep, end_substep)
    return ReplayBundle(
        snapshot=snapshot,
        controls=records,
        reference_frames=list(reference_frames or []),
        trigger=dict(trigger or {"substep": trigger_substep}),
        metadata=dict(metadata or {}),
    )


def _decode_controls(data: dict[str, Any]) -> list[ControlRecord]:
    if data.get("controls_encoding") == "control_log_npz_b64":
        return list(ControlLog.from_bytes(base64.b64decode(data["controls"])).records())
    return [ControlRecord.from_dict(item) for item in data["controls"]]  # legacy JSON list


def compare_states(expected: Any, actual: Any, *, ignore_keys: tuple[str, ...] = ()) -> dict[str, Any]:
    """Recursive numeric diff of two JSON-like public states."""

    numeric: list[tuple[str, float]] = []
    structural: list[dict[str, Any]] = []

    def visit(left: Any, right: Any, path: str) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                if key in ignore_keys:
                    continue
                child = f"{path}.{key}" if path else str(key)
                if key not in left or key not in right:
                    structural.append({"path": child, "kind": "missing_key"})
                else:
                    visit(left[key], right[key], child)
            return
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            try:
                a = np.asarray(left, dtype=float)
                b = np.asarray(right, dtype=float)
            except (TypeError, ValueError):
                if list(left) != list(right):
                    structural.append({"path": path, "kind": "value_mismatch"})
                return
            if a.shape != b.shape:
                structural.append({"path": path, "kind": "shape_mismatch"})
                return
            if a.size:
                numeric.append((path, float(np.max(np.abs(a - b)))))
            return
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            numeric.append((path, abs(float(left) - float(right))))
            return
        if left != right:
            structural.append({"path": path, "kind": "value_mismatch"})

    visit(expected, actual, "")
    nonzero = sorted((item for item in numeric if item[1] > 0.0), key=lambda item: -item[1])
    return {
        "max_abs_error": nonzero[0][1] if nonzero else 0.0,
        "nonzero_fields": len(nonzero),
        "compared_fields": len(numeric),
        "structural_differences": structural,
        "largest": [{"path": path, "max_abs_error": err} for path, err in nonzero[:10]],
    }


def deepcopy_state(state: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(state)


def _write_json_gz(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=4) as handle:
        json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
    temporary.replace(path)
    return path


def _read_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)
