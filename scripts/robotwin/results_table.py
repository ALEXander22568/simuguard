#!/usr/bin/env python3
"""One row per task, same columns for everyone: the unified results table.

    task | N | S | official | E_fail | E_succ | F+ | F± | S- | audited | upper

* N        scored episodes (the benchmark's own count; skipped seeds are not in it)
* S        official successes
* official S / N
* E_fail   failures with at least one physics-invalid event (after the gravity filter)
* E_succ   successes with at least one physics-invalid event
* F+       failures that become successes when the artifact is removed (verdict environment_caused)
* F±       failures that flip only under the mass counterfactual (environment_possible)
* S-       successes that fail once the artifact is removed (artifact_assisted_success)
* audited  (S - S- + F+) / N       <- the corrected score; the denominator never changes
* upper    (S - S- + F+ + F±) / N  <- upper bound if every "possible" case were environment-caused

Nothing is excluded from N: an episode the environment broke is counted the way the
policy would have scored without the artifact, not thrown away.

Usage::

    python scripts/robotwin/results_table.py --runs runs/tier1_x/* --attr runs/attr_* --out docs/results.csv

``--runs`` are task run directories (each with manifest.json and, after gravity_filter.py,
events_gravity.json in the segments).  ``--attr`` are attribution output directories whose
reports name the segment they audited.  Tasks without attribution get F+/F±/S- = 0 and the
row is marked ``attr: no``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

COLUMNS = ["task", "N", "S", "official", "E_fail", "E_succ", "F_plus", "F_pm", "S_minus",
           "audited", "upper", "attr", "run_dir"]


def task_row(run: Path, verdicts: dict[str, str]) -> dict | None:
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    task = manifest["task"]
    n = int(manifest["policy_episodes"])
    s = int(manifest["policy_success"])
    e_fail = e_succ = f_plus = f_pm = s_minus = 0
    audited_any = False
    for episode in manifest["episodes"]:
        policy = episode.get("policy")
        if not policy:
            continue
        segment = run / policy["segment"]
        invalid = False
        gravity = segment / "events_gravity.json"
        if gravity.is_file():
            invalid = any(e.get("verdict") == "physics_invalid" for e in json.loads(gravity.read_text())["events"])
        else:
            invalid = policy["confirmed_events"] > 0  # filter not run yet: raw detector count
        ok = bool(policy.get("eval_success"))
        e_fail += (not ok) and invalid
        e_succ += ok and invalid
        verdict = verdicts.get(str(segment.resolve()))
        if verdict:
            audited_any = True
            f_plus += verdict == "environment_caused"
            f_pm += verdict == "environment_possible"
            s_minus += verdict == "artifact_assisted_success"
    return {
        "task": task, "N": n, "S": s, "official": f"{s}/{n}",
        "E_fail": e_fail, "E_succ": e_succ, "F_plus": f_plus, "F_pm": f_pm, "S_minus": s_minus,
        "audited": f"{s - s_minus + f_plus}/{n}", "upper": f"{s - s_minus + f_plus + f_pm}/{n}",
        "attr": "yes" if audited_any else "no", "run_dir": str(run),
    }


def load_verdicts(attr_dirs: list[str]) -> dict[str, str]:
    verdicts: dict[str, str] = {}
    for root in attr_dirs:
        for report in Path(root).rglob("*.json"):
            if report.name == "summary.json":
                continue
            try:
                data = json.loads(report.read_text())
            except json.JSONDecodeError:
                continue
            if data.get("phase") == "policy" and "verdict" in data and "segment" in data:
                verdicts[str(Path(data["segment"]).resolve())] = data["verdict"]
    return verdicts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="task run directories")
    parser.add_argument("--attr", nargs="*", default=[], help="attribution output directories")
    parser.add_argument("--out", help="CSV to write (appends rows for new tasks, replaces existing ones)")
    args = parser.parse_args()

    verdicts = load_verdicts(args.attr)
    rows = [r for r in (task_row(Path(run), verdicts) for run in args.runs) if r]
    existing: dict[str, dict] = {}
    if args.out and Path(args.out).is_file():
        with open(args.out, newline="") as handle:
            existing = {r["task"]: r for r in csv.DictReader(handle)}
    for row in rows:
        existing[row["task"]] = row
    print("| " + " | ".join(COLUMNS[:-1]) + " |")
    print("|" + "---|" * (len(COLUMNS) - 1))
    for row in sorted(existing.values(), key=lambda r: r["task"]):
        print("| " + " | ".join(str(row[c]) for c in COLUMNS[:-1]) + " |")
    if args.out:
        with open(args.out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(sorted(existing.values(), key=lambda r: r["task"]))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
