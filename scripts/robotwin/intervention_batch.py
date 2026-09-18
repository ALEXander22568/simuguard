#!/usr/bin/env python3
"""Run the intervention protocol over many replay bundles and summarise the result.

For every bundle: replay unchanged (gate), then once per intervention, and record
whether the detector still confirms the event and how fast the target still moves.

Usage::

    python scripts/robotwin/intervention_batch.py --robotwin-root <RoboTwin> \\
        --bundles <dir-or-glob> [...] --output-dir runs/interventions \\
        --interventions baseline solver_high mass_100g --limit 12
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()


def collect(patterns: list[str], limit: int | None) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.json.gz")))
        else:
            files.extend(sorted(Path(p) for p in glob.glob(pattern)))
    seen: dict[str, Path] = {}
    for f in files:
        seen.setdefault(str(f), f)
    ordered = list(seen.values())
    return ordered[:limit] if limit else ordered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--bundles", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--interventions", nargs="+", default=["baseline", "solver_high", "mass_100g"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    bundles = collect(args.bundles, args.limit)
    print(f"[batch] {len(bundles)} bundles", flush=True)

    results = []
    for index, bundle in enumerate(bundles):
        report = out / f"{bundle.stem.replace('.json', '')}.json"
        print(f"[batch] {index + 1}/{len(bundles)} {bundle.name}", flush=True)
        proc = subprocess.run(
            [
                args.python, "-B", str(HERE.parent / "replay_intervention.py"),
                "--robotwin-root", args.robotwin_root, "--bundle", str(bundle),
                "--report", str(report), "--interventions", *args.interventions,
            ],
            capture_output=True, text=True,
        )
        (out / f"{report.stem}.log").write_text(proc.stdout + proc.stderr)
        if not report.exists():
            results.append({"bundle": bundle.name, "error": proc.returncode})
            continue
        data = json.loads(report.read_text())
        row = {
            "bundle": bundle.name,
            "phase": data.get("phase"),
            "onset_substep": data["trigger_event"]["onset_substep"],
            "live_max_speed_mps": data["trigger_metrics"].get("max_speed_mps"),
            "gate_at_onset": (data["trigger_metrics"].get("gate_at_onset") or {}).get("active"),
            "conditions": {},
        }
        for condition in data["conditions"]:
            target_speeds = [v for k, v in condition["max_speed_mps"].items() if "basket" not in k]
            row["conditions"][condition["name"]] = {
                "confirmed": condition["confirmed_events"],
                "target_max_speed_mps": max(target_speeds) if target_speeds else None,
                "matches_live": condition["trajectory_matches_live"],
                "max_position_error_m": condition["max_position_error_m"],
            }
        results.append(row)

    summary = {"bundles": len(bundles), "interventions": args.interventions, "rows": results}
    valid = [r for r in results if "conditions" in r and r["conditions"].get("baseline", {}).get("matches_live")]
    summary["baseline_gate_passed"] = len(valid)
    for name in args.interventions:
        speeds = [r["conditions"][name]["target_max_speed_mps"] for r in valid if name in r["conditions"]]
        confirmed = [r["conditions"][name]["confirmed"] for r in valid if name in r["conditions"]]
        if speeds:
            summary[name] = {
                "events": len(speeds),
                "target_max_speed_mps": {
                    "median": statistics.median(speeds),
                    "mean": statistics.fmean(speeds),
                    "max": max(speeds),
                },
                "events_still_confirmed": sum(1 for c in confirmed if c),
                "confirmed_total": sum(confirmed),
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
