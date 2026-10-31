#!/usr/bin/env python3
"""Summarise init_state_audit.py: per task and suite, how many official init states make the physics
throw objects before the policy acts (confirmed ejection -> gravity stage -> carrier stage), how deep
the objects start inside their supports, and which objects."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audit")
    ap.add_argument("--pen-mm", type=float, default=5.0, help="report objects starting deeper than this")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    for log in sorted(Path(args.audit).glob("*/t*/audit.jsonl")):
        rows += [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    by_suite: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_suite[r["suite"]][r["task_id"]].append(r)
    result: dict = {}
    for suite in [s for s in SUITES if s in by_suite]:
        tot = defaultdict(int)
        print(f"== {suite}")
        for task in sorted(by_suite[suite]):
            rs = by_suite[suite][task]
            n = len(rs)
            conf = sum(r["confirmed"] > 0 for r in rs)
            grav = sum(r["after_gravity"] > 0 for r in rs)
            inv = sum(r["physics_invalid"] > 0 for r in rs)
            deep: dict = defaultdict(float)
            fast: dict = defaultdict(float)
            for r in rs:
                for name, o in r["objects"].items():
                    deep[name] = max(deep[name], o["init_penetration_mm"])
                    fast[name] = max(fast[name], o["peak_speed_mps"])
            deep_objs = {k: (round(v, 1), round(fast[k], 2)) for k, v in sorted(deep.items(), key=lambda kv: -kv[1])
                         if v >= args.pen_mm}
            errors = sum(r["monitor_errors"] for r in rs)
            print(f"  t{task} {rs[0]['task'][:58]:58s} inits {n:2d}  confirmed {conf:2d}  gravity {grav:2d}"
                  f"  invalid {inv:2d}  deepest start {max(deep.values(), default=0):5.1f} mm  errors {errors}")
            if deep_objs:
                print(f"      objects starting >= {args.pen_mm} mm deep (mm, peak m/s): {deep_objs}")
            result.setdefault(suite, {})[task] = {"task": rs[0]["task"], "inits": n, "confirmed": conf,
                                                  "after_gravity": grav, "physics_invalid": inv,
                                                  "deep_objects": deep_objs, "monitor_errors": errors}
            for k, v in (("inits", n), ("confirmed", conf), ("after_gravity", grav), ("physics_invalid", inv),
                         ("tasks_with_invalid", int(inv > 0)), ("monitor_errors", errors)):
                tot[k] += v
        print(f"  TOTAL {dict(tot)}")
        result.setdefault("totals", {})[suite] = dict(tot)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
