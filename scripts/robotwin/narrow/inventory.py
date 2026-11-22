#!/usr/bin/env python3
"""List every scored policy episode under some run roots, with its physics-invalid status.

Reads only small files (summary.json, manifest.json, events_carrier.json / events_gravity.json)
and checks which recording files exist.  Uses the same rule as ``attribute_failures.py`` to find
the first physics-invalid onset (carrier stage if it ran, else gravity stage).  Plain Python 3, no
numpy, so it runs with any interpreter on a host.

Writes one JSON line per policy segment::

    python scripts/robotwin/narrow/inventory.py --host 4090-A --label campaign/rep1 \
        --root runs/campaign_x/default/rep1 [--root ...] > inventory_4090-A.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def first_invalid(segment: Path) -> tuple[str | None, dict | None, int]:
    """(stage file used, first physics-invalid event, number of physics-invalid events)."""

    for name in ("events_carrier.json", "events_gravity.json"):
        path = segment / name
        if path.is_file():
            events = [e for e in json.loads(path.read_text()).get("events", []) if e.get("verdict") == "physics_invalid"]
            if not events:
                return name, None, 0
            first = min(events, key=lambda e: int(e["onset_substep"]))
            return name, first, len(events)
    return None, None, 0


def segments_under(root: Path) -> list[Path]:
    found = sorted(root.glob("simuguard/segments/*/summary.json")) or sorted(root.glob("**/simuguard/segments/*/summary.json"))
    if not found:
        found = sorted(root.glob("segments/*/summary.json"))
    return [p.parent for p in found]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--root", action="append", required=True, help="run root (repeatable)")
    args = ap.parse_args()
    for root in args.root:
        root_path = Path(root).resolve()
        if not root_path.is_dir():
            print(json.dumps({"host": args.host, "root": str(root_path), "error": "missing root"}))
            continue
        for segment in segments_under(root_path):
            try:
                summary = json.loads((segment / "summary.json").read_text())
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({"host": args.host, "segment": str(segment), "error": f"summary: {exc}"}))
                continue
            meta = summary.get("metadata") or {}
            if meta.get("phase") != "policy":
                continue
            manifest_meta = {}
            if (segment / "manifest.json").is_file():
                manifest_meta = (json.loads((segment / "manifest.json").read_text()).get("metadata") or {})
            outcome = meta.get("outcome") or {}
            stage, event, n_invalid = first_invalid(segment)
            sim_intervention = meta.get("sim_intervention", manifest_meta.get("sim_intervention"))
            print(json.dumps({
                "host": args.host,
                "root": str(root_path),
                "segment": str(segment),
                "task": meta.get("task") or manifest_meta.get("task"),
                "seed": meta.get("seed", manifest_meta.get("seed")),
                "success": bool(outcome.get("eval_success")),
                "confirmed_events": int(summary.get("confirmed_count", 0)),
                "invalid_stage": stage,
                "n_invalid": n_invalid,
                "first_invalid_onset": int(event["onset_substep"]) if event else None,
                "first_invalid_body": (event.get("body") or (event.get("bodies") or [None])[0]) if event else None,
                "first_invalid_event_id": event.get("event_id") if event else None,
                "sim_intervention": sim_intervention,
                "files": {name: (segment / name).is_file() for name in
                          ("controls.npz", "states.npz", "trace.jsonl.gz", "manifest.json", "events_gravity.json",
                           "events_carrier.json")},
            }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
