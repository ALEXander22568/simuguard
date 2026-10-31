"""On-disk episode artefacts.

Layout (one directory per monitored episode)::

    <episode_dir>/
      manifest.json         bodies, detector configs, adapter capabilities, provenance
      trace.jsonl.gz        decimated substep frames (tracked non-robot bodies + their contacts)
      events.jsonl          every event status update, append-only
      bundles/<name>.json.gz replay bundles for triggering events
      controls.npz          compact per-substep actuation (exact replay input)
      states.npz            float64 per-substep state of non-robot tracked bodies (replay reference)
      initial_snapshot.json.gz state at attach (verifies a rebuilt env)
      snapshots.npz         optional: every native snapshot held at the end (MonitorConfig.save_snapshots)
      summary.json          final counts and the latest status of each event
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any

from .controls import ControlLog
from .events import Event
from .snapshot import ReplayBundle, Snapshot
from .statelog import StateLog
from .types import SubstepFrame


class EpisodeRecorder:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        trace_decimation: int = 1,
        trace_body_ids: set[str] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.trace_decimation = max(0, int(trace_decimation))
        self.trace_body_ids = trace_body_ids
        self._trace = None
        self._events = None

    def begin(self, manifest: dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.output_dir / "manifest.json", manifest)
        if self.trace_decimation > 0:
            self._trace = gzip.open(self.output_dir / "trace.jsonl.gz", "wt", encoding="utf-8", compresslevel=4)
        self._events = open(self.output_dir / "events.jsonl", "a", encoding="utf-8")

    def frame(self, frame: SubstepFrame) -> None:
        if self._trace is None or frame.substep % self.trace_decimation != 0:
            return
        self._trace.write(json.dumps(frame.to_dict(self.trace_body_ids), separators=(",", ":")) + "\n")

    def event(self, event: Event) -> None:
        if self._events is not None:
            self._events.write(json.dumps(event.to_dict(), separators=(",", ":")) + "\n")
            self._events.flush()

    def bundle(self, event_id: str, bundle: ReplayBundle) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", event_id)[-150:]
        return bundle.save(self.output_dir / "bundles" / f"{safe}.json.gz")

    def episode_logs(
        self,
        *,
        controls: ControlLog | None,
        states: StateLog | None,
        initial_snapshot: Snapshot | None,
    ) -> None:
        if controls is not None and len(controls):
            controls.save(self.output_dir / "controls.npz")
        if states is not None and len(states):
            states.save(self.output_dir / "states.npz")
        if initial_snapshot is not None:
            initial_snapshot.save(self.output_dir / "initial_snapshot.json.gz")

    def snapshot_archive(self, snapshots: list[Snapshot]) -> None:
        """All native snapshots of the episode in one file: substeps, control steps, raw state bytes."""
        if not snapshots:
            return
        import numpy as np

        blobs = [np.frombuffer(s.native_state, dtype=np.uint8) for s in snapshots]
        size = max(len(b) for b in blobs)
        states = np.zeros((len(blobs), size), dtype=np.uint8)
        for i, b in enumerate(blobs):
            states[i, : len(b)] = b
        np.savez_compressed(
            self.output_dir / "snapshots.npz",
            substeps=np.array([s.substep for s in snapshots], dtype=np.int64),
            control_steps=np.array([s.control_step for s in snapshots], dtype=np.int64),
            lengths=np.array([len(b) for b in blobs], dtype=np.int64),
            states=states,
            native_format=np.array(snapshots[0].native_format or ""),
        )

    def end(self, summary: dict[str, Any]) -> None:
        for handle in (self._trace, self._events):
            if handle is not None:
                handle.close()
        self._trace = None
        self._events = None
        _write_json(self.output_dir / "summary.json", summary)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)
