#!/usr/bin/env python3
"""Summarise an XR-1 x RoboCasa campaign (xr1_campaign.sh): per contact setting and task,
episodes, successes, detected events, physics-invalid events (gravity stage), and how deep free
objects sank into other bodies.

The gravity stage is RoboTwin's (scripts/robotwin/gravity_filter.py) with its windows kept in
seconds (0.6 s before the onset, 0.1 s after), i.e. converted to this benchmark's 2 ms step.
Penetration comes from the recorded trace: for every free object, the deepest contact with a
non-robot body and with the robot over the whole episode.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "robotwin"))

from gravity_filter import classify_segment  # noqa: E402


def penetration_stats(segment: Path) -> dict:
    manifest = json.loads((segment / "manifest.json").read_text())
    roles = {b: info["role"] for b, info in manifest["bodies"].items()}
    free = {b for b, info in manifest["bodies"].items() if info["kind"] == "dynamic"}
    worst_env, worst_robot, deep_env = 0.0, 0.0, []
    with gzip.open(segment / "trace.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            frame = json.loads(line)
            for c in frame["contacts"]:
                a, b, depth = c["a"], c["b"], float(c["penetration"])
                if depth <= 0 or not (a in free or b in free):
                    continue
                robot = roles.get(a) == "robot" or roles.get(b) == "robot"
                if robot:
                    worst_robot = max(worst_robot, depth)
                else:
                    worst_env = max(worst_env, depth)
                    if depth >= 0.003:
                        deep_env.append((frame["substep"], a, b, depth))
    return {"max_pen_env_mm": 1000 * worst_env, "max_pen_robot_mm": 1000 * worst_robot,
            "substeps_pen_env_ge_3mm": len(deep_env),
            "deepest_env": sorted(deep_env, key=lambda x: -x[3])[:5]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("--penetration", action="store_true", help="also scan traces (slow)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.campaign)
    table: dict = defaultdict(lambda: defaultdict(list))
    for log in sorted(root.glob("*/*/episodes.jsonl")):
        contact, task = log.parent.parent.name, log.parent.name
        for line in log.read_text().splitlines():
            if not line.strip():
                continue
            ep = json.loads(line)
            seg = Path(ep["segment"])
            if not seg.is_dir():
                seg = log.parent / "segments" / seg.name
            summary = json.loads((seg / "summary.json").read_text())
            dt = float(summary.get("timestep_s") or 0.002)
            res = classify_segment(seg, round(0.6 / dt), round(0.1 / dt), 1.5, 0.05) or {"events": []}
            ep["invalid"] = [e for e in res["events"] if e.get("verdict") == "physics_invalid"]
            ep["gravity"] = res["events"]
            if args.penetration:
                ep.update(penetration_stats(seg))
            table[contact][task].append(ep)

    out = {}
    for contact in sorted(table):
        print(f"== {contact}")
        tot = defaultdict(int)
        for task in sorted(table[contact]):
            eps = sorted(table[contact][task], key=lambda e: e["episode"])
            n, s = len(eps), sum(e["success"] for e in eps)
            det = sum(bool(e["confirmed"]) for e in eps)
            inv = [e for e in eps if e["invalid"]]
            e_fail = sum(not e["success"] for e in inv)
            e_succ = sum(bool(e["success"]) for e in inv)
            err = sum(bool(e["error"]) for e in eps)
            line = (f"  {task:28s} N {n:2d}  S {s:2d}  detected {det:2d}  invalid {len(inv):2d} (fail {e_fail}, succ {e_succ})"
                    f"  errors {err}")
            if args.penetration:
                pe = [e["max_pen_env_mm"] for e in eps]
                line += f"  max pen vs env {max(pe):5.1f} mm (median {sorted(pe)[len(pe) // 2]:4.1f})"
            print(line)
            for k, v in (("N", n), ("S", s), ("detected", det), ("invalid", len(inv)), ("E_fail", e_fail), ("E_succ", e_succ)):
                tot[k] += v
            out.setdefault(contact, {})[task] = eps
        print(f"  {'TOTAL':28s} " + "  ".join(f"{k} {v}" for k, v in tot.items()))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
