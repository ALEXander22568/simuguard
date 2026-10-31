#!/usr/bin/env python3
"""Run counterfactual_contact.py over every finished episode of a campaign (in parallel) and
tabulate: per moment, the pressed object's peak speed and displacement with the recorded
contact time constants and with them capped, and the confirmed ejections in each."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(segment: Path, out: Path, args: argparse.Namespace) -> Path | None:
    target = out / f"{segment.parent.parent.name}__{segment.name}.json"
    if target.exists():
        return target
    cmd = [sys.executable, str(HERE / "counterfactual_contact.py"), str(segment), "--out", str(target),
           "--moments", str(args.moments), "--min-depth", str(args.min_depth), "--tau", str(args.tau),
           "--window", str(args.window)]
    log = target.with_suffix(".log")
    with log.open("w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return target if target.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign", help="campaign root with default/<task>/segments/*")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--moments", type=int, default=3)
    ap.add_argument("--min-depth", type=float, default=0.003)
    ap.add_argument("--tau", type=float, default=0.004)
    ap.add_argument("--window", type=float, default=1.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    segments = sorted(p for p in Path(args.campaign).glob("default/*/segments/*") if (p / "summary.json").is_file()
                      and (p / "snapshots.npz").is_file())
    with ThreadPoolExecutor(args.workers) as pool:
        results = [r for r in pool.map(lambda s: run(s, out, args), segments) if r is not None]

    cap = f"tau_{1000 * args.tau:g}ms"
    rows = []
    for path in results:
        report = json.loads(path.read_text())
        for m in report.get("moments", []):
            a, b = m["pair"]
            free = set(m["recorded"]["peak_speed"])
            obj = next((x.split(":", 1)[1] for x in (a, b) if x.split(":", 1)[1] in free), None)
            if obj is None:
                continue
            rows.append({"task": report["task"], "seed": report["seed"], "substep": m["substep"],
                         "depth_mm": 1000 * m["depth_m"], "object": obj,
                         "partner": (b if a.endswith(obj) else a).split(":", 1)[1],
                         "v_recorded": m["recorded"]["peak_speed"][obj], "v_capped": m[cap]["peak_speed"][obj],
                         "disp_recorded": m["recorded"]["displacement"][obj], "disp_capped": m[cap]["displacement"][obj],
                         "events_recorded": len(m["recorded"]["confirmed"]), "events_capped": len(m[cap]["confirmed"]),
                         "robot_touches": m["recorded"]["robot_touches_pair"], "replay_err_m": report["replay_max_abs_pos_err_m"]})
    rows.sort(key=lambda r: -r["depth_mm"])
    print(f"{len(results)} episodes replayed, {len(rows)} moments (object pressed >= {1000 * args.min_depth:g} mm into a non-robot body)")
    for r in rows:
        print(f"  {r['task']:26s} seed {r['seed']:5d} {r['object']:>18s} into {r['partner'][:24]:24s} {r['depth_mm']:5.1f} mm  "
              f"v {r['v_recorded']:5.2f} -> {r['v_capped']:5.2f} m/s  disp {100 * r['disp_recorded']:5.1f} -> {100 * r['disp_capped']:5.1f} cm  "
              f"events {r['events_recorded']} -> {r['events_capped']}  replay err {r['replay_err_m']:.1e}")
    (out / "table.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
