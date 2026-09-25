#!/usr/bin/env python3
"""Per-task CSV from a campaign summary (campaign_summary.py --out) and an init-state audit
summary (audit_summary.py --out).  Usage: task_table.py SUMMARY.json AUDIT_SUMMARY.json OUT.csv"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main() -> int:
    summary = json.loads(Path(sys.argv[1]).read_text())
    audit = json.loads(Path(sys.argv[2]).read_text())
    rows = []
    for key, t in summary["tasks"].items():
        suite, task = key.split(":")
        a = audit.get(suite, {}).get(task) or audit.get(suite, {}).get(int(task)) or {}
        deep = a.get("deep_objects") or {}
        name = next((e["task"] for e in summary["episodes"] if e["suite"] == suite and e["task_id"] == int(task)), "")
        rows.append({
            "suite": suite, "task_id": int(task), "task": name,
            "episodes": t["episodes"], "successes": t["successes"],
            "detected_settle": t["detected_episodes_settle_phase"], "detected_policy": t["detected_episodes_policy_phase"],
            "invalid_settle": t["physics_invalid_episodes_settle_phase"], "invalid_policy": t["physics_invalid_episodes_policy_phase"],
            "policy_phase_flags": sum(t["policy_phase_flags_by_detector"].values()),
            "audit_init_states": a.get("inits"), "audit_invalid_init_states": a.get("physics_invalid"),
            "deepest_start_mm": max((v[0] for v in deep.values()), default=0.0),
        })
    rows.sort(key=lambda r: (("libero_spatial", "libero_object", "libero_goal", "libero_10").index(r["suite"]), r["task_id"]))
    with open(sys.argv[3], "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {sys.argv[3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
