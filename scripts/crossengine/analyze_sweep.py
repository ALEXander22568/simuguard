#!/usr/bin/env python3
"""Exit speed versus initial penetration, per engine and solver setting.

Reads the per-step results of sapien_side.py and mujoco_side.py on the same scenarios and
reports, for every (engine, setting):

  d0      penetration of the can into the basket at the first step (both engines measure the
          identical geometry at the identical pose, so d0 is shared)
  v_exit  the can's peak speed in the first 0.1 s (before gravity can add more than ~1 m/s)
  tau_eff d0 / v_exit, the time over which the solver removes the penetration

and how many poses end with the can thrown: v_exit >= 1 m/s and displaced >= 4 cm after 0.5 s.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

WINDOW_S = 0.1


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def metrics(run: dict) -> dict:
    dt = float(run.get("timestep") or 0.004)
    n = max(1, int(round(WINDOW_S / dt)))
    speed = np.asarray(run["speed"], dtype=float)
    pen = np.asarray(run["penetration"], dtype=float)
    return {"d0": float(pen[0]), "v_exit": float(np.nanmax(speed[:n])), "displacement": float(run["displacement"]),
            "finite": bool(np.isfinite(speed).all() and speed.max() < 1e3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sapien", required=True)
    ap.add_argument("--mujoco", required=True)
    ap.add_argument("--out", required=True, help="summary JSON")
    ap.add_argument("--plot", help="PDF/PNG scatter of v_exit against d0")
    args = ap.parse_args()

    runs = [("PhysX", r) for r in load(Path(args.sapien))] + [("MuJoCo", r) for r in load(Path(args.mujoco))]
    table: dict = defaultdict(dict)
    for engine, r in runs:
        table[(engine, r["variant"])][r["scenario"]] = metrics(r)
    # d0 from the default PhysX run (identical pose and geometry in every setting)
    reference = table.get(("PhysX", "default"), {})
    summary = {}
    for (engine, variant), rows in sorted(table.items()):
        names = sorted(set(rows) & set(reference))
        d0 = np.array([reference[n]["d0"] for n in names])
        v = np.array([rows[n]["v_exit"] for n in names])
        disp = np.array([rows[n]["displacement"] for n in names])
        finite = np.array([rows[n]["finite"] for n in names])
        deep = d0 >= 0.005
        thrown = (v >= 1.0) & (disp >= 0.04)
        tau = d0[deep] / np.maximum(v[deep], 1e-9)
        summary[f"{engine}/{variant}"] = {
            "poses": int(len(names)), "diverged": int((~finite).sum()),
            "thrown": int(thrown.sum()), "thrown_deep": int((thrown & deep).sum()), "deep_poses": int(deep.sum()),
            "v_exit_median": float(np.median(v)), "v_exit_p90": float(np.percentile(v, 90)),
            "tau_eff_ms_median_deep": float(1000 * np.median(tau)) if tau.size else None,
            "tau_eff_ms_iqr_deep": [float(1000 * np.percentile(tau, 25)), float(1000 * np.percentile(tau, 75))] if tau.size else None,
        }
        s = summary[f"{engine}/{variant}"]
        print(f"{engine:6s} {variant:26s} poses {s['poses']:3d}  thrown {s['thrown']:3d} ({s['thrown_deep']}/{s['deep_poses']} of d0>=5mm)  "
              f"v_exit median {s['v_exit_median']:6.2f} p90 {s['v_exit_p90']:7.2f} m/s  tau_eff(d0>=5mm) {s['tau_eff_ms_median_deep']} ms  diverged {s['diverged']}")
    Path(args.out).write_text(json.dumps(summary, indent=1))

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        styles = {("PhysX", "default"): ("#c0504d", "o", "PhysX, RoboTwin settings"),
                  ("PhysX", "depen_cap_1"): ("#e6a0a0", "s", "PhysX, depenetration capped at 1 m/s"),
                  ("MuJoCo", "default_4ms"): ("#4f81bd", "o", "MuJoCo, default soft contact (tau = 20 ms)"),
                  ("MuJoCo", "timeconst_8ms"): ("#9bbbe0", "^", "MuJoCo, tau = 8 ms")}
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for key, (color, marker, label) in styles.items():
            rows = table.get(key)
            if not rows:
                continue
            names = sorted(set(rows) & set(reference))
            d0 = np.array([reference[n]["d0"] for n in names]) * 1000
            v = np.array([rows[n]["v_exit"] for n in names])
            ax.scatter(d0, np.clip(v, 1e-2, 1e3), s=9, c=color, marker=marker, label=label, alpha=0.75, linewidths=0)
        d = np.linspace(0.5, 40, 50)
        for tau, text in ((0.004, "d / 4 ms (one step)"), (0.020, "d / 20 ms")):
            ax.plot(d, d / 1000 / tau, color="0.35", lw=1, ls="--")
            ax.text(d[-1], d[-1] / 1000 / tau, text, fontsize=7, ha="right", va="bottom", color="0.3")
        ax.set_yscale("log")
        ax.set_xlabel("initial penetration of the can into the basket (mm)")
        ax.set_ylabel("peak speed in the first 0.1 s (m/s)")
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=200)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
