"""On-disk episode artefacts.

Layout (one directory per monitored episode)::

    <episode_dir>/
      manifest.json         bodies, detector configs, adapter capabilities, provenance
      trace.jsonl.gz        decimated substep frames (tracked non-robot bodies + their contacts)
      events.jsonl          every event status update, append-only
      bundles/<name>.json.gz replay bundles for triggering events
      summary.json          final counts and the latest status of each event
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any

from .events import Event
from .snapshot import ReplayBundle
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
