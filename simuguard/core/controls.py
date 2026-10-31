"""Compact per-substep control log.

Public behaviour is identical to a list of :class:`ControlRecord` (``append``,
``between``, ``records``); storage is not.  Each payload is split once into a
*template* (structure, joint names, indices) and a flat numeric vector.  The
template is stored once per distinct structure; vectors are stored as float32
when every value round-trips exactly, otherwise the whole log is promoted to
float64.  Exact replay therefore never depends on the storage precision.

Measured on RoboTwin aloha-agilex (38 joints): 4,032 JSON bytes per substep
-> 114 numbers (456 bytes as float32) before compression.
"""

from __future__ import annotations

import io
import json
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .snapshot_types import ControlRecord

CONTROL_LOG_SCHEMA = "simuguard-control-log-v1"
_SEGMENT = "__simuguard_segment__"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _split(payload: Any, values: list[float]) -> Any:
    """Return a hashable-JSON template; append numeric leaves to ``values``."""

    if isinstance(payload, dict):
        return {str(k): _split(v, values) for k, v in payload.items()}
    if isinstance(payload, (list, tuple, np.ndarray)):
        items = payload.tolist() if isinstance(payload, np.ndarray) else list(payload)
        if items and all(_is_number(x) for x in items):
            kind = "int" if all(isinstance(x, (int, np.integer)) for x in items) else "float"
            values.extend(float(x) for x in items)
            return {_SEGMENT: [len(items), kind]}
        return [_split(item, values) for item in items]
    if isinstance(payload, (float, np.floating)):
        values.append(float(payload))
        return {_SEGMENT: [0, "float"]}  # length 0 marks a scalar
    if isinstance(payload, np.integer):
        return int(payload)
    return payload  # str, int, bool, None: structural constants


def _join(template: Any, values: np.ndarray, cursor: list[int]) -> Any:
    if isinstance(template, dict):
        if _SEGMENT in template and len(template) == 1:
            length, kind = template[_SEGMENT]
            start = cursor[0]
            if length == 0:
                cursor[0] += 1
                return float(values[start])
            cursor[0] += length
            chunk = values[start : start + length]
            return [int(round(v)) for v in chunk] if kind == "int" else [float(v) for v in chunk]
        return {k: _join(v, values, cursor) for k, v in template.items()}
    if isinstance(template, list):
        return [_join(item, values, cursor) for item in template]
    return template


class ControlLog:
    """Append-only (optionally bounded) control history with compact storage."""

    def __init__(self, maxlen: int | None = None) -> None:
        self.maxlen = maxlen
        self._templates: list[Any] = []
        self._template_index: dict[str, int] = {}
        self._substeps: list[int] = []
        self._control_steps: list[int] = []
        self._schema_ids: list[int] = []
        self._rows: list[np.ndarray] = []
        self.dtype = np.float32

    # ------------------------------------------------------------------ writing
    def append(self, record: ControlRecord) -> None:
        if self._substeps and record.substep <= self._substeps[-1]:
            raise ValueError("control records must have strictly increasing substeps")
        values: list[float] = []
        template = _split(record.payload, values)
        key = json.dumps(template, sort_keys=True, separators=(",", ":"))
        schema_id = self._template_index.get(key)
        if schema_id is None:
            schema_id = len(self._templates)
            self._templates.append(template)
            self._template_index[key] = schema_id
        row64 = np.asarray(values, dtype=np.float64)
        if self.dtype == np.float32:
            row32 = row64.astype(np.float32)
            if np.array_equal(row32.astype(np.float64), row64, equal_nan=True):
                row = row32
            else:
                self._promote()
                row = row64
        else:
            row = row64
        self._substeps.append(int(record.substep))
        self._control_steps.append(int(record.control_step))
        self._schema_ids.append(schema_id)
        self._rows.append(row)
        if self.maxlen is not None and len(self._rows) > self.maxlen:
            drop = len(self._rows) - self.maxlen
            del self._substeps[:drop], self._control_steps[:drop], self._schema_ids[:drop], self._rows[:drop]

    def _promote(self) -> None:
        self.dtype = np.float64
        self._rows = [row.astype(np.float64) for row in self._rows]

    # ------------------------------------------------------------------ reading
    def _record(self, i: int) -> ControlRecord:
        payload = _join(self._templates[self._schema_ids[i]], self._rows[i], [0])
        return ControlRecord(self._substeps[i], self._control_steps[i], payload)

    def records(self) -> Iterator[ControlRecord]:
        for i in range(len(self._rows)):
            yield self._record(i)

    def between(self, after_substep: int, until_substep: int) -> list[ControlRecord]:
        """Controls for substeps in ``(after_substep, until_substep]``; raises on gaps."""

        lo = bisect_right(self._substeps, after_substep)
        hi = bisect_right(self._substeps, until_substep)
        expected = list(range(after_substep + 1, until_substep + 1))
        if self._substeps[lo:hi] != expected:
            raise LookupError(
                f"control log does not cover substeps ({after_substep}, {until_substep}]; "
                f"oldest={self.oldest_substep}, newest={self.newest_substep}"
            )
        return [self._record(i) for i in range(lo, hi)]

    def get(self, substep: int) -> ControlRecord | None:
        i = bisect_left(self._substeps, substep)
        return self._record(i) if i < len(self._substeps) and self._substeps[i] == substep else None

    @property
    def oldest_substep(self) -> int | None:
        return self._substeps[0] if self._substeps else None

    @property
    def newest_substep(self) -> int | None:
        return self._substeps[-1] if self._substeps else None

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def nbytes(self) -> int:
        return int(sum(row.nbytes for row in self._rows))

    # ------------------------------------------------------------------ persistence
    def to_bytes(self) -> bytes:
        buffer = io.BytesIO()
        widths = np.array([row.size for row in self._rows], dtype=np.int32)
        flat = np.concatenate(self._rows).astype(self.dtype) if self._rows else np.zeros(0, dtype=self.dtype)
        np.savez_compressed(
            buffer,
            schema=np.array(CONTROL_LOG_SCHEMA),
            templates=np.array(json.dumps(self._templates, separators=(",", ":"))),
            substeps=np.asarray(self._substeps, dtype=np.int64),
            control_steps=np.asarray(self._control_steps, dtype=np.int64),
            schema_ids=np.asarray(self._schema_ids, dtype=np.int32),
            widths=widths,
            values=flat,
        )
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "ControlLog":
        with np.load(io.BytesIO(data), allow_pickle=False) as npz:
            if str(npz["schema"]) != CONTROL_LOG_SCHEMA:
                raise ValueError(f"unsupported control log schema: {npz['schema']}")
            log = cls()
            log._templates = json.loads(str(npz["templates"]))
            log._template_index = {
                json.dumps(t, sort_keys=True, separators=(",", ":")): i for i, t in enumerate(log._templates)
            }
            log._substeps = npz["substeps"].tolist()
            log._control_steps = npz["control_steps"].tolist()
            log._schema_ids = npz["schema_ids"].tolist()
            values = npz["values"]
            log.dtype = np.float64 if values.dtype == np.float64 else np.float32
            offsets = np.concatenate([[0], np.cumsum(npz["widths"])]).astype(np.int64)
            log._rows = [values[offsets[i] : offsets[i + 1]].copy() for i in range(len(log._substeps))]
        return log

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(self.to_bytes())
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ControlLog":
        return cls.from_bytes(Path(path).read_bytes())

    @classmethod
    def from_records(cls, records: list[ControlRecord]) -> "ControlLog":
        log = cls()
        for record in records:
            log.append(record)
        return log
