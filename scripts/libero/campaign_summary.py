#!/usr/bin/env python3
"""Summarise a Pi0.5 x LIBERO campaign (pi05_campaign.sh) through SimuGuard's three stages.

  detector  confirmed contact-ejection events (the other detectors only raise flags);
  gravity   scripts/robotwin/gravity_filter.py: peak speed after onset over the free-fall speed of
            the drop before it; > 1.5 stays physics-invalid, else gravity-explained.  Windows kept
            in seconds (0.6 s before the onset, 0.1 s after), i.e. 300 / 50 substeps at 2 ms;
  carrier   scripts/robotwin/carrier_filter.py: an event is explained when a body the robot was
            touching carried the object (object speed < 1.5x the carrier's speed before onset).

Per suite (and with --per-task per task): episodes, successes, episodes with a detected event,
with a physics-invalid event after the gravity stage and after the carrier stage (split by the
episode's outcome), events during the settle steps (before the policy acts), flags per detector,
monitor errors.  Writes events_gravity.json / events_carrier.json into every segment with events
and, with --out, one JSON with every episode.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[1] / "robotwin"))

from carrier_filter import classify as carrier_classify  # noqa: E402
from gravity_filter import classify_segment  # noqa: E402

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")


def load_episodes(root: Path) -> list[dict]:
    out = []
    for log in sorted(root.glob("*/t*/episodes.jsonl")):
        for line in log.read_text().splitlines():
            if line.strip():
                ep = json.loads(line)
                seg = Path(ep["segment"])
                ep["segment"] = str(seg if seg.is_dir() else log.parent / "segments" / seg.name)
                out.append(ep)
    return out


def classify(ep: dict) -> dict:
    seg = Path(ep["segment"])
    summary = json.loads((seg / "summary.json").read_text())
    dt = float(summary.get("timestep_s") or 0.002)
    settle = int(ep.get("settle_substeps") or 0)
    flags = Counter(e["detector"] for e in summary["events"] if e.get("status") == "flag")
    gravity = classify_segment(seg, round(0.6 / dt), round(0.1 / dt), 1.5, 0.05) or {"events": []}
    carrier = carrier_classify(seg, 0.04, 0.1, 1.5, 0.05) if any(
        e.get("verdict") == "physics_invalid" for e in gravity["events"]) else {"events": []}
    confirmed = [e for e in summary["events"] if e.get("status") == "confirmed"]
    g_invalid = [e for e in gravity["events"] if e.get("verdict") == "physics_invalid"]
    c_invalid = [e for e in (carrier or {"events": []})["events"] if e.get("verdict") == "physics_invalid"]

    def split(events: list[dict]) -> tuple[int, int]:  # (settle phase, policy phase)
        n_settle = sum(int(e["onset_substep"]) <= settle for e in events)
        return n_settle, len(events) - n_settle

    return {
        **ep, "flags_by_detector": dict(flags),
        "detected": len(confirmed), "gravity_invalid": len(g_invalid), "carrier_invalid": len(c_invalid),
        "detected_split": split(confirmed), "gravity_split": split(g_invalid), "invalid_split": split(c_invalid),
        "events_detail": {"gravity": gravity["events"], "carrier": (carrier or {"events": []})["events"]},
    }


def table(rows: list[dict], label: str) -> dict:
    """Episode counts per stage; each stage also split into settle phase (before the policy acts) / policy phase."""
    n = len(rows)
    s = sum(bool(r["success"]) for r in rows)
    flags = Counter()
    for r in rows:
        flags.update(r["flags_by_detector"])
    out: dict = {"episodes": n, "successes": s, "success_rate": round(s / n, 3) if n else None}
    for stage, key in (("detected", "detected_split"), ("after_gravity", "gravity_split"), ("physics_invalid", "invalid_split")):
        eps = [r for r in rows if sum(r[key])]
        policy = [r for r in rows if r[key][1]]
        out[f"{stage}_episodes"] = len(eps)
        out[f"{stage}_events"] = sum(sum(r[key]) for r in rows)
        out[f"{stage}_episodes_settle_phase"] = sum(1 for r in rows if r[key][0])
        out[f"{stage}_episodes_policy_phase"] = len(policy)
        out[f"{stage}_policy_phase_failed"] = sum(not r["success"] for r in policy)
    out.update({
        "episodes_with_errors": sum(bool(r.get("error")) for r in rows),
        "monitor_errors": sum(int(r.get("monitor_errors") or 0) for r in rows),
        "flags_by_detector": dict(flags),
        "mean_steps": round(sum(r["steps"] for r in rows) / n, 1) if n else None,
        "mean_wall_s": round(sum(r["wall_s"] for r in rows) / n, 1) if n else None,
    })
    stage = lambda k: f"{out[k + '_episodes']:3d} ({out[k + '_episodes_settle_phase']:2d}/{out[k + '_episodes_policy_phase']:2d})"  # noqa: E731
    print(f"  {label:28s} N {n:3d}  S {s:3d} ({100 * s / max(n, 1):5.1f}%)  detected {stage('detected')}"
          f"  gravity {stage('after_gravity')}  carrier {stage('physics_invalid')}"
          f"  policy-phase invalid & failed {out['physics_invalid_policy_phase_failed']}"
          f"  errors {out['episodes_with_errors']}/{out['monitor_errors']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("--per-task", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = [classify(ep) for ep in load_episodes(Path(args.campaign))]
    by_suite: dict = defaultdict(list)
    for r in rows:
        by_suite[r["suite"]].append(r)
    result: dict = {"suites": {}, "tasks": {}}
    print("per suite: episodes, successes, episodes with events after the detector / gravity / carrier stage "
          "(settle phase / policy phase)")
    for suite in [s for s in SUITES if s in by_suite] + sorted(set(by_suite) - set(SUITES)):
        result["suites"][suite] = table(by_suite[suite], suite)
        if args.per_task:
            tasks: dict = defaultdict(list)
            for r in by_suite[suite]:
                tasks[r["task_id"]].append(r)
            for t in sorted(tasks):
                result["tasks"][f"{suite}:{t}"] = table(tasks[t], f"  t{t} {tasks[t][0]['task'][:22]}")
    result["total"] = table(rows, "TOTAL")
    for r in rows:
        for e in r["events_detail"]["carrier"] or []:
            if e.get("verdict") == "physics_invalid" and int(e["onset_substep"]) > int(r.get("settle_substeps") or 0):
                print(f"  policy-phase physics-invalid: {r['suite']} t{r['task_id']} ep{r['episode']} success={r['success']} "
                      f"{e['body']} onset {e['onset_substep']} peak {e['peak_speed_mps']} m/s "
                      f"carriers {e['carriers']} ratio {e['speed_over_carrier']}")
    if args.out:
        result["episodes"] = rows
        Path(args.out).write_text(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
