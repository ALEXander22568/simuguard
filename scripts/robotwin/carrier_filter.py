#!/usr/bin/env python3
"""Third stage after gravity_filter.py: is a physics-invalid event a push the robot transmitted
through another body?

An object in a container that the robot pushes (RoboCasa: a basket shoved by the mobile base)
moves fast without touching the robot and without falling, so contact ejection and the gravity
stage both accept it.  Carriers are dynamic bodies that touched the object and a robot link in
the window before onset.  The event is explained when the object's peak speed after onset
(0.1 s) is below ``--min-ratio`` times the fastest carrier's speed before onset: a body thrown
in the same solver step as the object is not moving before it, a robot-driven carrier is.

Reads trace.jsonl.gz, manifest.json and events_gravity.json of every segment; writes
events_carrier.json next to them and prints, per run, events and segments that keep the label.
Windows are in seconds, so the stage works on any time step.

Usage::

    python scripts/robotwin/carrier_filter.py --runs runs/crosstask7_all36/* runs/campaign_x/default/rep*
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def classify(segment: Path, before_s: float, after_s: float, min_ratio: float, v_floor: float) -> dict | None:
    gravity = segment / "events_gravity.json"
    if not gravity.is_file():
        return None
    gv = json.loads(gravity.read_text())
    events = [e for e in gv["events"] if e.get("verdict") == "physics_invalid"]
    out = {"segment": segment.name, "phase": gv.get("phase"), "outcome": gv.get("outcome"), "events": []}
    if events:
        manifest = json.loads((segment / "manifest.json").read_text())
        kind = {b: i["kind"] for b, i in manifest["bodies"].items()}
        role = {b: i["role"] for b, i in manifest["bodies"].items()}
        dt = float(manifest.get("timestep_s") or 0.004)
        before, after = max(1, round(before_s / dt)), max(1, round(after_s / dt))
        onsets = [(int(e["onset_substep"]), e["bodies"][0], e.get("event_id")) for e in events]
        lo, hi = min(o for o, _, _ in onsets) - before, max(o for o, _, _ in onsets) + after
        frames: dict[int, dict] = {}
        with gzip.open(segment / "trace.jsonl.gz", "rt", encoding="utf-8") as fh:
            for line in fh:
                frame = json.loads(line)
                if frame["substep"] < lo:
                    continue
                if frame["substep"] > hi:
                    break
                frames[frame["substep"]] = frame

        def speed(s: int, body: str) -> float:
            state = frames.get(s, {}).get("states", {}).get(body)
            return float(np.linalg.norm(state["v"])) if state else 0.0

        for onset, body, event_id in onsets:
            window = range(onset - before, onset)
            partners = set()
            for s in window:
                for c in frames.get(s, {}).get("contacts", []):
                    if body in (c["a"], c["b"]):
                        other = c["b"] if c["a"] == body else c["a"]
                        if kind.get(other) == "dynamic":
                            partners.add(other)
            carriers = set()
            for s in window:
                for c in frames.get(s, {}).get("contacts", []):
                    for p in partners:
                        if p in (c["a"], c["b"]) and role.get(c["b"] if c["a"] == p else c["a"]) == "robot":
                            carriers.add(p)
            v_obj = max((speed(s, body) for s in range(onset, onset + after + 1)), default=0.0)
            v_carrier = max((speed(s, p) for s in window for p in carriers), default=0.0)
            ratio = v_obj / max(v_carrier, v_floor)
            out["events"].append({
                "event_id": event_id, "onset_substep": onset, "body": body, "carriers": sorted(carriers),
                "peak_speed_mps": round(v_obj, 4), "carrier_speed_before_mps": round(v_carrier, 4),
                "speed_over_carrier": round(ratio, 3),
                "verdict": "physics_invalid" if ratio >= min_ratio else "carrier_explained",
            })
    (segment / "events_carrier.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run directories holding [simuguard/]segments/*")
    ap.add_argument("--before", type=float, default=0.04, help="s before onset searched for carriers")
    ap.add_argument("--after", type=float, default=0.1, help="s after onset searched for the peak speed")
    ap.add_argument("--min-ratio", type=float, default=1.5)
    ap.add_argument("--v-floor", type=float, default=0.05, help="m/s")
    args = ap.parse_args()
    for run in map(Path, args.runs):
        segments = sorted((run / "simuguard" / "segments").glob("*")) or sorted((run / "segments").glob("*"))
        counts = {}
        for segment in segments:
            res = classify(segment, args.before, args.after, args.min_ratio, args.v_floor)
            if not res or not res["events"]:
                continue
            kept = [e for e in res["events"] if e["verdict"] == "physics_invalid"]
            c = counts.setdefault(res.get("phase") or "?", [0, 0, 0, 0])
            c[0] += len(res["events"])
            c[1] += len(kept)
            c[2] += 1
            c[3] += bool(kept)
        print(f"{run}: " + "; ".join(f"{phase}: events {a} -> {b}, segments {c} -> {d}" for phase, (a, b, c, d) in sorted(counts.items()))
              + " (physics-invalid before -> after the carrier stage)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
