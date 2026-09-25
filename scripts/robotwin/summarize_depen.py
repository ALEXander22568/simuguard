#!/usr/bin/env python3
"""Summarise exact replays of failed episodes under PhysX depenetration-velocity caps.

Reads the per-segment reports that attribute_failures.py writes with the conditions
``baseline depen_<v> ...`` and prints, per task, how many failures the cap turns into successes,
split by whether the recorded episode held a physics-invalid event.  The baseline condition
must reproduce the recorded episode bit for bit; segments where it does not are listed apart.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="attribute_failures.py output directories (one per task)")
    ap.add_argument("--invalid", help="JSON {report file stem: true} of segments with a physics-invalid event; "
                                      "default: recorded_confirmed_events > 0")
    ap.add_argument("--out")
    args = ap.parse_args()
    invalid = json.loads(Path(args.invalid).read_text()) if args.invalid else None
    report = {}
    for d in map(Path, args.dirs):
        rows = []
        for f in sorted(d.glob("*.json")):
            r = json.loads(f.read_text())
            conds = {c["name"]: c for c in r["conditions"]}
            base = conds.get("baseline", {})
            has_event = bool(invalid.get(f.stem)) if invalid is not None else r["recorded_confirmed_events"] > 0
            rows.append({"segment": r["segment"], "event": has_event,
                         "exact": bool(base.get("bit_identical")) and bool(base.get("reproduces_recorded_outcome")),
                         **{n: {"success": bool(c.get("success")), "events": c.get("confirmed_events"),
                                "peak": c.get("peak_target_speed_mps")} for n, c in conds.items()}})
        caps = sorted({n for row in rows for n in row if n.startswith("depen_")}, key=lambda n: -float(n.split("_")[1]))
        groups = defaultdict(list)
        for row in rows:
            groups["with event" if row["event"] else "event-free"].append(row)
        print(f"== {d.name}: {len(rows)} failures replayed, baseline exact {sum(r['exact'] for r in rows)}/{len(rows)}")
        for label, members in groups.items():
            parts = [f"{c}: {sum(m[c]['success'] for m in members if c in m)}/{sum(c in m for m in members)} succeed, "
                     f"{sum(bool(m[c]['events']) for m in members if c in m)} still with events" for c in caps]
            print(f"   {label:10s} n={len(members):2d}  " + " | ".join(parts))
        report[d.name] = rows
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
