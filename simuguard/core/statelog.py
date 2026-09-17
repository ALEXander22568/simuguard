"""Full-precision per-substep state log for a fixed set of bodies.

Stores position(3) quaternion(4) linear velocity(3) angular velocity(3) as
float64 for every substep.  It is the reference for exact-replay checks and a
compact analysis trace (the JSON trace is rounded and meant for humans).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from .types import BodyState

STATE_LOG_SCHEMA = "simuguard-state-log-v1"
FIELDS = ("px", "py", "pz", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz")
WIDTH = len(FIELDS)


class StateLog:
    def __init__(self, body_ids: list[str]) -> None:
        self.body_ids = list(body_ids)
        self._index = {body_id: i for i, body_id in enumerate(self.body_ids)}
        self._substeps: list[int] = []
        self._rows: list[np.ndarray] = []

    def append(self, substep: int, states: dict[str, BodyState]) -> None:
        row = np.full((len(self.body_ids), WIDTH), np.nan, dtype=np.float64)
        for body_id, i in self._index.items():
            state = states.get(body_id)
            if state is not None:
                row[i] = np.concatenate(
                    [state.position, state.quaternion, state.linear_velocity, state.angular_velocity]
                )
        self._substeps.append(int(substep))
        self._rows.append(row)

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def substeps(self) -> np.ndarray:
        return np.asarray(self._substeps, dtype=np.int64)

    def array(self) -> np.ndarray:
        """Shape (substeps, bodies, 13)."""

        if not self._rows:
            return np.zeros((0, len(self.body_ids), WIDTH))
        return np.stack(self._rows)

    def positions(self, body_id: str) -> np.ndarray:
        return self.array()[:, self._index[body_id], 0:3]

    def to_bytes(self) -> bytes:
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            schema=np.array(STATE_LOG_SCHEMA),
            body_ids=np.array(json.dumps(self.body_ids)),
            fields=np.array(json.dumps(FIELDS)),
            substeps=self.substeps,
            states=self.array(),
        )
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "StateLog":
        with np.load(io.BytesIO(data), allow_pickle=False) as npz:
            if str(npz["schema"]) != STATE_LOG_SCHEMA:
                raise ValueError(f"unsupported state log schema: {npz['schema']}")
            log = cls(json.loads(str(npz["body_ids"])))
            log._substeps = npz["substeps"].tolist()
            log._rows = list(npz["states"])
        return log

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(self.to_bytes())
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "StateLog":
        return cls.from_bytes(Path(path).read_bytes())
