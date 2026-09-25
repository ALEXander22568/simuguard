#!/usr/bin/env python3
"""Reproduce a confirmed event by exact replay, then repeat it under interventions.

Protocol (the causal core of SimuGuard):

1. ``baseline``  - rebuild the env with the episode's seed and replay the recorded
   actuation unchanged.  The recorded trajectory must be reproduced (fidelity gate)
   and the detector must fire again; otherwise the bundle proves nothing.
2. interventions - same replay, with one simulation setting changed before the first
   replayed substep.  An event that disappears under exactly one changed setting is
   evidence that the setting, not the policy, produced it.

Interventions are applied through the adapter, so the policy, the controls and the
scene are identical across conditions.

Usage::

    python scripts/robotwin/replay_intervention.py --robotwin-root <RoboTwin> \\
        --bundle <segment>/bundles/<event>.json.gz --report out.json \\
        --interventions baseline solver_high mass_100g mass_50g
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SIMUGUARD_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIMUGUARD_ROOT))

from simuguard.adapters.robotwin import RoboTwinAdapter, make_task_env  # noqa: E402
from simuguard.core import ReplayBundle, replay_bundle  # noqa: E402
from simuguard.core.types import BodyRole  # noqa: E402
from simuguard.presets import default_detectors  # noqa: E402


def build_intervention(name: str):
    """Return ``(description, fn(adapter))`` for one condition."""

    if name == "baseline":
        return "unchanged simulation settings (fidelity + reproduction gate)", None
    if name.startswith("solver_"):
        _, pos, vel = name.split("_") if name.count("_") == 2 else ("", name.split("_")[1], None)
        position = 32 if pos == "high" else int(pos)
        velocity = 8 if vel is None else int(vel)

        def apply(adapter: RoboTwinAdapter, position=position, velocity=velocity) -> None:
            adapter.set_solver_iterations(position=position, velocity=velocity)

        return f"solver iterations position={position} velocity={velocity}", apply
    if name.startswith("mass_"):
        grams = float(name.split("_")[1].rstrip("g"))

        def apply(adapter: RoboTwinAdapter, grams=grams) -> None:
            for body_id in adapter.body_ids_with_role(BodyRole.TARGET):
                adapter.set_mass(body_id, grams / 1000.0)

        return f"target mass = {grams} g", apply
    if name.startswith("depen_"):
        cap = float(name.split("_", 1)[1])

        def apply(adapter: RoboTwinAdapter, cap=cap) -> None:
            adapter.set_max_depenetration_velocity(cap)

        return f"PhysX max depenetration velocity = {cap} m/s on every dynamic body and link (default: unlimited)", apply
    raise SystemExit(f"unknown intervention: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--interventions", nargs="+", default=["baseline", "solver_high", "mass_100g"],
        help="baseline | solver_high | solver_<pos>_<vel> | mass_<grams>g | depen_<m/s>",
    )
    parser.add_argument("--position-tolerance-m", type=float, default=1e-4)
    args = parser.parse_args()

    bundle = ReplayBundle.load(args.bundle)
    meta = bundle.metadata
    task, seed = meta["task"], int(meta["seed"])
    trigger = bundle.trigger
    report = {
        "bundle": str(Path(args.bundle).resolve()),
        "task": task,
        "seed": seed,
        "phase": meta.get("phase"),
        "trigger_event": {k: trigger.get(k) for k in ("event_id", "detector", "kind", "onset_substep", "bodies", "reasons")},
        "trigger_metrics": (trigger.get("metrics") or {}),
        "replayed_substeps": len(bundle.controls),
        "conditions": [],
    }

    for name in args.interventions:
        description, apply = build_intervention(name)
        env, _ = make_task_env(args.robotwin_root, task, seed)
        try:
            adapter = RoboTwinAdapter(env)
            gate = None
            try:
                gate = adapter.containment_gate()
            except Exception:  # noqa: BLE001
                pass
            before = adapter.solver_iterations()
            result = replay_bundle(
                adapter,
                bundle,
                method="none",
                detectors=default_detectors(ejection_gate=gate),
                before_replay=apply,
                position_tolerance_m=args.position_tolerance_m,
                episode_id=f"intervention:{name}",
            )
            confirmed = result.confirmed_events
            target = adapter.body_ids_with_role(BodyRole.TARGET)
            report["conditions"].append(
                {
                    "name": name,
                    "description": description,
                    "solver_iterations_before": before,
                    "solver_iterations_after": adapter.solver_iterations(),
                    "target_mass_kg": {b: adapter.mass(b) for b in target},
                    "trajectory_matches_live": result.within_tolerance,
                    "max_position_error_m": result.overall_max_error_m,
                    "max_speed_mps": {b: v for b, v in result.max_speed_mps.items() if b in target or "basket" in b},
                    "confirmed_events": len(confirmed),
                    "confirmed_event_details": [
                        {k: e[k] for k in ("onset_substep", "reasons")} | {"max_speed_mps": e["metrics"]["max_speed_mps"]}
                        for e in confirmed[:5]
                    ],
                    "flag_counts": _count(result.events),
                }
            )
            print(json.dumps(report["conditions"][-1], default=str)[:600], flush=True)
        finally:
            try:
                env.close_env()
            except Exception:  # noqa: BLE001
                pass

    Path(args.report).write_text(json.dumps(report, indent=2, default=str))
    baseline = next((c for c in report["conditions"] if c["name"] == "baseline"), None)
    if baseline is not None and not (baseline["trajectory_matches_live"] and baseline["confirmed_events"]):
        print("BASELINE GATE FAILED: the event was not reproduced; interventions are not interpretable")
        return 1
    return 0


def _count(events: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        key = f"{event['detector']}:{event['status']}"
        counts[key] = counts.get(key, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
