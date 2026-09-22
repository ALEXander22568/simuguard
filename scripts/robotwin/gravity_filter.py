#!/usr/bin/env python3
"""Keep only the events whose speed gravity cannot explain.

The contact-ejection detector flags fast motion of a free object near a container.
An object that is *dropped* into a bin moves fast too, for a legitimate reason: it
fell.  This filter separates the two without any task-specific gate.  For each
confirmed event it takes the object's peak speed in the short window right after the
onset (default 0.1 s, before any later fall can add speed), measures how far the
object had fallen from its highest point in the preceding window to that peak, and
compares the peak speed with the free-fall speed for that drop, sqrt(2 g dz).

    ratio = v_peak / max(sqrt(2 g dz), v_floor)

A dropped object has ratio <= 1 (it never moves faster than gravity allows); an
object that received contact energy the robot did not supply has ratio well above
1.  Events with ratio above ``--min-ratio`` (default 1.5) are kept as
physics-invalid; the rest are relabelled ``gravity_explained``.

The filter reads the recorded state log, so it runs on finished runs without a
simulator.  It writes ``events_gravity.json`` next to each ``summary.json`` and a
per-run table.

Usage::

    python scripts/robotwin/gravity_filter.py --runs runs/crosstask_x/* runs/campaign_x/default/rep*
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simuguard.core import StateLog  # noqa: E402

G = 9.81


def classify_segment(segment: Path, before: int, after: int, min_ratio: float, v_floor: float) -> dict | None:
    summary_path = segment / "summary.json"
    states_path = segment / "states.npz"
    if not summary_path.is_file() or not states_path.is_file():
        return None
    summary = json.loads(summary_path.read_text())
    events = [e for e in summary.get("events", []) if e.get("status") == "confirmed"]
    meta = summary.get("metadata") or {}
    out = {"segment": segment.name, "phase": meta.get("phase"), "seed": meta.get("seed"),
           "outcome": meta.get("outcome"), "events": []}
    if not events:
        return out
    log = StateLog.load(states_path)
    ids, array, substeps = log.body_ids, log.array(), log.substeps
    for event in events:
        bodies = [b for b in event.get("bodies", []) if b in ids]
        record = {"event_id": event.get("event_id"), "onset_substep": event["onset_substep"],
                  "detector": event.get("detector"), "bodies": event.get("bodies")}
        if not bodies:
            record.update({"verdict": "unresolved", "reason": "no tracked body"})
            out["events"].append(record)
            continue
        index = ids.index(bodies[0])
        onset = int(event["onset_substep"])
        window = (substeps >= onset - before) & (substeps <= onset + after)
        if window.sum() < 10:
            record.update({"verdict": "unresolved", "reason": "window outside state log"})
            out["events"].append(record)
            continue
        z = array[window, index, 2]
        speed = np.linalg.norm(array[window, index, 7:10], axis=1)
        local = substeps[window]
        after_onset = np.nonzero(local >= onset)[0]
        k = int(after_onset[np.argmax(speed[after_onset])])
        drop = float(max(np.max(z[: k + 1]) - z[k], 0.0))
        free_fall = float(np.sqrt(2.0 * G * drop))
        ratio = float(speed[k]) / max(free_fall, v_floor)
        record.update({
            "peak_speed_mps": round(float(speed[k]), 4),
            "drop_before_peak_m": round(drop, 4),
            "free_fall_speed_mps": round(free_fall, 4),
            "speed_over_free_fall": round(ratio, 3),
            "verdict": "physics_invalid" if ratio > min_ratio else "gravity_explained",
        })
        out["events"].append(record)
    (segment / "events_gravity.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="run directories holding simuguard/segments")
    parser.add_argument("--before", type=int, default=150)
    parser.add_argument("--after", type=int, default=25, help="substeps after onset searched for the peak (0.1 s)")
    parser.add_argument("--min-ratio", type=float, default=1.5)
    parser.add_argument("--v-floor", type=float, default=0.05, help="m/s; avoids dividing by ~0 for a tiny drop")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    table = []
    for run in args.runs:
        run_path = Path(run).resolve()
        segments = sorted((run_path / "simuguard" / "segments").glob("*"))
        if not segments:
            segments = sorted(run_path.glob("*/simuguard/segments/*"))
        counts = {"policy": {"episodes": 0, "with_flags": 0, "with_invalid": 0, "flags": 0, "invalid": 0, "success": 0,
                             "fail_with_invalid": 0},
                  "expert": {"episodes": 0, "with_flags": 0, "with_invalid": 0, "flags": 0, "invalid": 0,
                             "check_fail": 0, "check_fail_with_invalid": 0}}
        for segment in segments:
            result = classify_segment(segment, args.before, args.after, args.min_ratio, args.v_floor)
            if result is None or result["phase"] not in counts:
                continue
            bucket = counts[result["phase"]]
            invalid = [e for e in result["events"] if e.get("verdict") == "physics_invalid"]
            bucket["episodes"] += 1
            bucket["flags"] += len(result["events"])
            bucket["invalid"] += len(invalid)
            bucket["with_flags"] += bool(result["events"])
            bucket["with_invalid"] += bool(invalid)
            outcome = result.get("outcome") or {}
            if result["phase"] == "policy":
                ok = bool(outcome.get("eval_success"))
                bucket["success"] += ok
                bucket["fail_with_invalid"] += (not ok) and bool(invalid)
            else:
                failed = bool(outcome.get("plan_success")) and not outcome.get("check_success")
                bucket["check_fail"] += failed
                bucket["check_fail_with_invalid"] += failed and bool(invalid)
        table.append({"run": str(run_path), **counts})
        p, x = counts["policy"], counts["expert"]
        print(f"{run_path.name:28s} policy: {p['episodes']:3d} eps, {p['success']:3d} ok, flagged {p['with_flags']:3d} -> invalid {p['with_invalid']:3d} "
              f"({p['flags']:3d} -> {p['invalid']:3d} events), failures with invalid {p['fail_with_invalid']:2d} | "
              f"expert: {x['episodes']:3d} segs, check-fail {x['check_fail']:2d}, of which with invalid {x['check_fail_with_invalid']:2d}")
    if args.output:
        Path(args.output).write_text(json.dumps(table, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
