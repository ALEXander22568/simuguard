#!/usr/bin/env python3
"""Run the episode-level intervention protocol over many recorded segments.

Each segment (one episode) is one independent unit: baseline replay + one replay
per intervention.  Segments without confirmed events are skipped by default.

Usage::

    python scripts/robotwin/intervention_episode_batch.py --robotwin-root <RoboTwin> \\
        --run-dir <run>/simuguard --output-dir runs/interventions \\
        --interventions baseline solver_high mass_100g
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--run-dir", required=True, help="<run>/simuguard directory holding segments/")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--interventions", nargs="+", default=["baseline", "solver_high", "mass_100g"])
    parser.add_argument("--include-without-events", action="store_true")
    parser.add_argument("--phases", nargs="+", default=["policy", "expert"])
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    run = Path(args.run_dir).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    segments = []
    for segment in sorted((run / "segments").iterdir()):
        summary_path = segment / "summary.json"
        if not summary_path.is_file() or not (segment / "controls.npz").is_file():
            continue
        summary = json.loads(summary_path.read_text())
        phase = (summary.get("metadata") or {}).get("phase")
        confirmed = summary.get("confirmed_count", 0)
        if phase not in args.phases:
            continue
        if confirmed == 0 and not args.include_without_events:
            continue
        segments.append((segment, phase, confirmed, summary["metadata"].get("seed")))

    print(f"[batch] {len(segments)} segments with events", flush=True)
    rows = []
    for index, (segment, phase, confirmed, seed) in enumerate(segments, start=1):
        report = out / f"{segment.name}.json"
        print(f"[batch] {index}/{len(segments)} {segment.name} phase={phase} events={confirmed}", flush=True)
        proc = subprocess.run(
            [
                args.python, "-B", str(HERE.parent / "intervention_episode.py"),
                "--robotwin-root", args.robotwin_root, "--segment-dir", str(segment),
                "--report", str(report), "--interventions", *args.interventions,
            ],
            capture_output=True, text=True,
        )
        (out / f"{segment.name}.log").write_text(proc.stdout + proc.stderr)
        if not report.exists():
            rows.append({"segment": segment.name, "phase": phase, "seed": seed, "error": proc.returncode})
            continue
        data = json.loads(report.read_text())
        row = {
            "segment": segment.name,
            "phase": phase,
            "seed": seed,
            "substeps": data["substeps"],
            "recorded_events": data["live_confirmed_events"],
            "conditions": {
                c["name"]: {
                    "peak_speed_mps": c["episode_peak_target_speed_mps"],
                    "confirmed": c["confirmed_events"],
                    "reproduced": c["recorded_events_reproduced"],
                    "bit_identical": c.get("state_log_bit_identical"),
                }
                for c in data["conditions"]
            },
        }
        rows.append(row)
        print("   " + json.dumps(row["conditions"], default=str)[:300], flush=True)

    gated = [r for r in rows if r.get("conditions", {}).get("baseline", {}).get("bit_identical")]
    summary = {
        "segments": len(rows),
        "baseline_gate_passed": len(gated),
        "interventions": args.interventions,
        "per_condition": {},
        "rows": rows,
    }
    for name in args.interventions:
        peaks = [r["conditions"][name]["peak_speed_mps"] for r in gated if name in r["conditions"]]
        confirmed = [r["conditions"][name]["confirmed"] for r in gated if name in r["conditions"]]
        reproduced = [r["conditions"][name]["reproduced"] for r in gated if name in r["conditions"]]
        recorded = [r["recorded_events"] for r in gated if name in r["conditions"]]
        if peaks:
            summary["per_condition"][name] = {
                "episodes": len(peaks),
                "peak_speed_mps": {"median": st.median(peaks), "min": min(peaks), "max": max(peaks)},
                "episodes_with_confirmed_events": sum(1 for c in confirmed if c),
                "confirmed_events_total": sum(confirmed),
                "recorded_events_reproduced": f"{sum(reproduced)}/{sum(recorded)}",
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
