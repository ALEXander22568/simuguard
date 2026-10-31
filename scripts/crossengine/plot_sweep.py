#!/usr/bin/env python3
"""Figure: exit speed of the can against its initial penetration into the basket, per engine
and solver setting (binned median and interquartile range), and the share of poses in which the
can is thrown.  Same geometry, masses, inertias and time step in both engines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

WINDOW_S = 0.1
SETTINGS = [  # key, label, colour, line style
    (("PhysX", "default"), "PhysX, RoboTwin settings", "#c96f6f", "-"),
    (("PhysX", "depen_cap_1"), "PhysX, depenetration ≤ 1 m/s", "#e7a9a2", "--"),
    (("MuJoCo", "timeconst_8ms"), "MuJoCo, τ = 8 ms", "#8fb3d9", "--"),
    (("MuJoCo", "default_4ms"), "MuJoCo, default (τ = 20 ms)", "#4f7cac", "-"),
]
BAR_EXTRA = [(("PhysX", "depen_cap_0.1"), "PhysX, depenetration ≤ 0.1 m/s", "#f2d0cb"),
             (("PhysX", "mass_100g"), "PhysX, can 100 g (RoboTwin: 10 g)", "#b08080"),
             (("PhysX", "solver_32_8"), "PhysX, 32/8 solver iterations (10/1)", "#8c5a5a")]


def load(path: Path, engine: str) -> dict:
    out: dict = {}
    for run in json.loads(path.read_text()):
        dt = float(run.get("timestep") or 0.004)
        n = max(1, int(round(WINDOW_S / dt)))
        speed = np.asarray(run["speed"], dtype=float)
        out.setdefault((engine, run["variant"]), {})[run["scenario"]] = {
            "d0": float(run["penetration"][0]), "v": float(np.nanmax(speed[:n])), "disp": float(run["displacement"])}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sapien", required=True)
    ap.add_argument("--mujoco", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = {**load(Path(args.sapien), "PhysX"), **load(Path(args.mujoco), "MuJoCo")}
    ref = data[("PhysX", "default")]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#555555", "axes.labelcolor": "#222222", "xtick.color": "#444444",
                         "ytick.color": "#444444", "pdf.fonttype": 42})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 2.9), gridspec_kw={"width_ratios": [1.45, 1]})
    edges = np.arange(5, 40, 5) / 1000
    centres = (edges[:-1] + edges[1:]) / 2 * 1000
    for key, label, colour, style in SETTINGS:
        rows = data[key]
        names = sorted(set(rows) & set(ref))
        d0 = np.array([ref[n]["d0"] for n in names])
        v = np.array([rows[n]["v"] for n in names])
        med, lo, hi = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = v[(d0 >= a) & (d0 < b)]
            med.append(np.median(sel) if sel.size >= 5 else np.nan)
            lo.append(np.percentile(sel, 25) if sel.size >= 5 else np.nan)
            hi.append(np.percentile(sel, 75) if sel.size >= 5 else np.nan)
        ax.fill_between(centres, lo, hi, color=colour, alpha=0.22, linewidth=0)
        ax.plot(centres, med, style, color=colour, lw=2, marker="o", ms=3.2, label=label)
    d = np.linspace(5, 37.5, 20)
    for tau, text, y_off in ((0.004, "d / 4 ms (one step)", 1.12), (0.020, "d / 20 ms", 1.12)):
        ax.plot(d, d / 1000 / tau, color="#9a9a9a", lw=0.9, ls=":")
        ax.text(d[-1], d[-1] / 1000 / tau * y_off, text, fontsize=6.5, ha="right", va="bottom", color="#6b6b6b")
    ax.set_yscale("log")
    ax.set_xlabel("initial penetration of the can into the basket (mm)")
    ax.set_ylabel("peak speed in the first 0.1 s (m/s)")
    ax.grid(axis="y", color="#e6e6e6", lw=0.6, which="major")
    ax.legend(fontsize=6.4, frameon=False, loc="lower right", handlelength=2.2)
    ax.text(-0.14, 1.02, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold")

    bars = [(k, l, c) for k, l, c, _ in SETTINGS] + BAR_EXTRA
    order = [0, 5, 6, 1, 4, 2, 3]
    bars = [bars[i] for i in order]
    shares, labels, colours = [], [], []
    for key, label, colour in bars:
        rows = data[key]
        names = sorted(set(rows) & set(ref))
        thrown = sum(rows[n]["v"] >= 1.0 and rows[n]["disp"] >= 0.04 for n in names)
        shares.append(100 * thrown / len(names))
        labels.append(f"{label}\n{thrown}/{len(names)}")
        colours.append(colour)
    y = np.arange(len(bars))[::-1]
    bx.barh(y, shares, color=colours, height=0.62, edgecolor="white", linewidth=1.5)
    for yi, s in zip(y, shares):
        bx.text(s + 2, yi, f"{s:.0f}%", va="center", fontsize=7, color="#333333")
    bx.set_yticks(y, labels, fontsize=6.4)
    bx.set_xlim(0, 110)
    bx.set_xlabel("poses with the can thrown (%)\n(≥ 1 m/s and ≥ 4 cm in 0.5 s)")
    bx.tick_params(axis="y", length=0)
    bx.text(-0.95, 1.02, "(b)", transform=bx.transAxes, fontsize=9, fontweight="bold")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(args.out, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
