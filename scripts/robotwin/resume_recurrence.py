#!/usr/bin/env python3
"""How often does a recorded event recur when the policy re-decides from the same state?

Replay a recorded episode to a point *before* a confirmed event, hand control
back to the live policy, and repeat.  The pre-event state is identical in every
retry (bit-exact replay), so whatever differs afterwards comes from the policy's
own run-to-run variation - or from a simulation setting, if one is changed.

Reading the result:
* recurrence close to 1 -> at that state the event is essentially unavoidable:
  it belongs to the simulator/configuration, not to one unlucky action;
* recurrence well below 1 -> the event depends on the exact actions, and the
  policy can often steer around it.

Rolling back further (``--distances``) shows how far the policy must be rewound
to escape the event.

Usage::

    python scripts/robotwin/resume_recurrence.py --robotwin-root <RoboTwin> \\
        --segment-dir <seg> --event-index 0 --distances 100 250 1000 \\
        --repeats 5 --interventions baseline mass_100g --port <bridge> \\
        --output-dir runs/recurrence
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
    parser.add_argument("--segment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--port", required=True, help="LingBot bridge port (servers started separately)")
    parser.add_argument("--event-index", type=int, default=0, help="which recorded confirmed event to target")
    parser.add_argument("--distances", type=int, nargs="+", default=[250])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--interventions", nargs="+", default=["baseline"])
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    segment = Path(args.segment_dir).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((segment / "summary.json").read_text())
    events = [e for e in summary["events"] if e["status"] == "confirmed"]
    if not events:
        raise SystemExit(f"{segment} has no confirmed events")
    target_event = events[min(args.event_index, len(events) - 1)]
    onset = target_event["onset_substep"]

    report = {
        "segment": str(segment),
        "seed": summary["metadata"].get("seed"),
        "phase": summary["metadata"].get("phase"),
        "recorded_events": len(events),
        "target_event": {k: target_event[k] for k in ("event_id", "onset_substep", "control_step", "reasons")},
        "target_event_peak_speed_mps": target_event["metrics"]["max_speed_mps"],
        "steps_after_resume": args.steps,
        "runs": [],
    }

    for intervention in args.interventions:
        for distance in args.distances:
            resume = max(1, onset - distance)
            for repeat in range(1, args.repeats + 1):
                name = f"{intervention}_d{distance}_r{repeat}"
                run_dir = out / name
                print(f"[recurrence] {name} (resume at {resume}, onset {onset})", flush=True)
                proc = subprocess.run(
                    [
                        args.python, "-B", str(HERE.parent / "resume_from_replay.py"),
                        "--robotwin-root", args.robotwin_root, "--segment-dir", str(segment),
                        "--output-dir", str(run_dir), "--mode", "policy",
                        "--port", str(args.port), "--resume-substep", str(resume),
                        "--steps", str(args.steps), "--intervention", intervention,
                    ],
                    capture_output=True, text=True,
                )
                (out / f"{name}.log").write_text(proc.stdout + proc.stderr)
                report_path = run_dir / "report.json"
                if not report_path.exists():
                    report["runs"].append({"name": name, "intervention": intervention, "distance": distance,
                                           "repeat": repeat, "error": proc.returncode})
                    continue
                data = json.loads(report_path.read_text())
                report["runs"].append(
                    {
                        "name": name, "intervention": intervention, "distance": distance, "repeat": repeat,
                        "resume_substep": data["resume_substep"],
                        "prefix_bit_identical": data.get("prefix_bit_identical"),
                        "events_after_resume": data.get("confirmed_events_after_resume"),
                        "peak_speed_after_resume_mps": data.get("peak_target_speed_after_resume_mps"),
                        "success_after_resume": data.get("success_after_resume"),
                        "substeps_after_resume": data.get("substeps_after_resume"),
                        "actions": data.get("actions_executed"),
                    }
                )
                print("   " + json.dumps(report["runs"][-1], default=str)[:260], flush=True)
                _write(out, report)

    _write(out, report)
    print(json.dumps(report["summary"], indent=1, default=str))
    return 0


def _write(out: Path, report: dict) -> None:
    groups: dict[str, list[dict]] = {}
    for run in report["runs"]:
        if "events_after_resume" not in run:
            continue
        groups.setdefault(f"{run['intervention']}@{run['distance']}", []).append(run)
    summary = {}
    for key, runs in groups.items():
        recurred = [r for r in runs if (r["events_after_resume"] or 0) > 0]
        peaks = [r["peak_speed_after_resume_mps"] for r in runs if r["peak_speed_after_resume_mps"] is not None]
        summary[key] = {
            "retries": len(runs),
            "recurrence_rate": round(len(recurred) / len(runs), 3) if runs else None,
            "prefix_bit_identical_all": all(r.get("prefix_bit_identical") for r in runs),
            "peak_speed_mps": {"median": round(st.median(peaks), 3), "max": round(max(peaks), 3)} if peaks else None,
            "success_after_resume": sum(1 for r in runs if r.get("success_after_resume")),
        }
    report["summary"] = summary
    (out / "recurrence.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
