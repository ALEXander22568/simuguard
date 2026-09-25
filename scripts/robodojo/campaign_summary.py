#!/usr/bin/env python3
"""Summarise SimuGuard x RoboDojo runs: episodes, successes and the three event stages.

Stage 1 (detector)  confirmed contact-ejection events from the monitor (summary.json);
Stage 2 (gravity)   scripts/robotwin/gravity_filter.py: peak speed after onset vs the free-fall speed
                    of the preceding drop, windows 0.6 s before / 0.1 s after the onset;
Stage 3 (carrier)   scripts/robotwin/carrier_filter.py: the object's peak speed vs the speed of any
                    dynamic body that touched both it and a robot link just before the onset.

Windows are given in seconds and converted with each segment's physics step (RoboDojo: 4 ms, the same
step as RoboTwin, so the RoboTwin substep windows 150/25 apply unchanged).  Also reports replay fidelity
when the run did a rebuild-and-replay check (``--replay``).

Usage::

    python scripts/robodojo/campaign_summary.py RUN_ROOT [--out summary.json]

RUN_ROOT holds jobs (``<job>/<task>/episodes.jsonl`` + ``segments/``), e.g. runs/v1.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[1] / "robotwin"))

from carrier_filter import classify as carrier_classify  # noqa: E402
from gravity_filter import classify_segment  # noqa: E402


def load_episodes(root: Path) -> list[dict]:
    episodes = []
    for log in sorted(root.glob("*/*/episodes.jsonl")):
        job, task = log.parent.parent.name, log.parent.name
        for line in log.read_text().splitlines():
            if line.strip():
                ep = json.loads(line)
                ep["job"] = job
                seg = Path(ep.get("segment", ""))
                ep["segment_local"] = str(log.parent / "segments" / seg.name) if seg.name else ""
                episodes.append(ep)
    return episodes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--min-ratio", type=float, default=1.5)
    ap.add_argument("--v-floor", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    table: dict = defaultdict(lambda: defaultdict(int))
    details = []
    for ep in load_episodes(Path(args.root)):
        seg = Path(ep["segment_local"])
        key = (ep.get("policy"), ep["task"])
        row = table[key]
        row["episodes"] += 1
        if ep.get("error"):
            row["errors"] += 1
            details.append({**{k: ep.get(k) for k in ("job", "task", "layout", "error")}})
            continue
        row["success"] += bool(ep.get("success"))
        summary_path = seg / "summary.json"
        if not summary_path.is_file():
            row["no_monitor"] += 1
            continue
        summary = json.loads(summary_path.read_text())
        dt = float(summary.get("timestep_s") or 0.004)
        confirmed = [e for e in summary.get("events", []) if e.get("status") == "confirmed"]
        grav = classify_segment(seg, round(0.6 / dt), round(0.1 / dt), args.min_ratio, args.v_floor) or {"events": []}
        g_invalid = [e for e in grav["events"] if e.get("verdict") == "physics_invalid"]
        carr = carrier_classify(seg, 0.04, 0.1, args.min_ratio, args.v_floor) or {"events": []}
        c_invalid = [e for e in carr["events"] if e.get("verdict") == "physics_invalid"]
        row["stage1_detected_eps"] += bool(confirmed)
        row["stage2_gravity_eps"] += bool(g_invalid)
        row["stage3_carrier_eps"] += bool(c_invalid)
        row["stage1_events"] += len(confirmed)
        row["stage2_events"] += len(g_invalid)
        row["stage3_events"] += len(c_invalid)
        row["fail_with_invalid"] += bool(c_invalid) and not ep.get("success")
        replay = ep.get("replay") or {}
        if replay:
            row["replayed"] += 1
            row["replay_bit_exact"] += bool(replay.get("bit_exact"))
            row["replay_max_err_mm_x1000"] = max(row["replay_max_err_mm_x1000"], int(1e6 * replay.get("max_position_error_m", 0.0)))
        details.append({
            "job": ep["job"], "task": ep["task"], "layout": ep.get("layout"), "success": ep.get("success"),
            "substeps": ep.get("substeps"), "confirmed": len(confirmed), "gravity_invalid": len(g_invalid),
            "carrier_invalid": len(c_invalid),
            "events": [{k: e.get(k) for k in ("onset_substep", "body", "carriers", "peak_speed_mps", "speed_over_carrier", "verdict")}
                       for e in carr["events"]],
            "gravity": [{k: e.get(k) for k in ("onset_substep", "bodies", "peak_speed_mps", "speed_over_free_fall", "verdict")}
                        for e in grav["events"]],
            "replay": {k: replay.get(k) for k in ("bit_exact", "max_position_error_m", "max_free_body_position_error_m", "first_divergence")} if replay else None,
            "hook": {k: (ep.get("hook") or {}).get(k) for k in ("hooked_steps", "physics_steps_seen", "between_step_writes")},
        })

    print(f"{'policy':18s} {'task':28s} {'N':>3s} {'S':>3s} {'err':>3s} | {'det':>4s} {'grav':>4s} {'carr':>4s} "
          f"(episodes; events {'d/g/c':>8s}) | fail+invalid | replay exact")
    for (policy, task), row in sorted(table.items()):
        print(f"{str(policy):18s} {task:28s} {row['episodes']:3d} {row['success']:3d} {row['errors']:3d} | "
              f"{row['stage1_detected_eps']:4d} {row['stage2_gravity_eps']:4d} {row['stage3_carrier_eps']:4d} "
              f"({row['stage1_events']}/{row['stage2_events']}/{row['stage3_events']:<3d}) | "
              f"{row['fail_with_invalid']:12d} | {row['replay_bit_exact']}/{row['replayed']} "
              f"(max {row['replay_max_err_mm_x1000'] / 1000:.3f} mm)")
    if args.out:
        Path(args.out).write_text(json.dumps({"table": {f"{p}|{t}": r for (p, t), r in table.items()}, "episodes": details},
                                             indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
