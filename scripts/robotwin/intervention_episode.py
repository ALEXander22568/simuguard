#!/usr/bin/env python3
"""Episode-level intervention experiment (one replay per condition).

The independent unit is the *episode*, not the event: one episode-start replay
covers every event in it, so replaying once per bundle would count the same
replay many times (pseudo-replication) and report the prefix maximum as if it
belonged to each event.

Protocol per recorded segment:

1. ``baseline`` - rebuild the env with the episode's seed, replay the recorded
   actuation unchanged.  Gate: the replay must match ``states.npz`` exactly and
   the detector must reproduce the recorded events.
2. interventions - identical replay with exactly one simulation setting changed.

Reported per condition: peak target speed over the episode, peak speed inside
each recorded event's window, how many events the detector confirms, and which
recorded events still have a match.

Usage::

    python scripts/robotwin/intervention_episode.py --robotwin-root <RoboTwin> \\
        --segment-dir <run>/simuguard/segments/<seg> --report out.json \\
        --interventions baseline solver_high mass_100g
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SIMUGUARD_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIMUGUARD_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from replay_intervention import build_intervention  # noqa: E402
from simuguard.adapters.robotwin import RoboTwinAdapter, make_task_env  # noqa: E402
from simuguard.core import ControlLog, ReplayBundle, Snapshot, StateLog, replay_bundle  # noqa: E402
from simuguard.core.types import BodyRole  # noqa: E402
from simuguard.presets import default_detectors  # noqa: E402


def speed_series(log: StateLog, body_id: str) -> np.ndarray:
    array = log.array()
    index = log.body_ids.index(body_id)
    return np.linalg.norm(array[:, index, 7:10], axis=1)


def window_peak(log: StateLog, body_id: str, onset: int, before: int, after: int) -> float:
    substeps = log.substeps
    speeds = speed_series(log, body_id)
    mask = (substeps >= onset - before) & (substeps <= onset + after)
    return float(np.nanmax(speeds[mask])) if mask.any() else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--segment-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--interventions", nargs="+", default=["baseline", "solver_high", "mass_100g"])
    parser.add_argument("--window-before", type=int, default=50)
    parser.add_argument("--window-after", type=int, default=250)
    parser.add_argument("--match-tolerance", type=int, default=125, help="substeps; event matched to a recorded onset")
    args = parser.parse_args()

    segment = Path(args.segment_dir).resolve()
    manifest = json.loads((segment / "manifest.json").read_text())
    summary = json.loads((segment / "summary.json").read_text())
    meta = {**manifest["metadata"], **(summary.get("metadata") or {})}  # summary carries phase/outcome
    task, seed = meta["task"], int(meta["seed"])
    controls = ControlLog.load(segment / "controls.npz")
    live_states = StateLog.load(segment / "states.npz")
    initial = Snapshot.load(segment / "initial_snapshot.json.gz")
    live_events = [e for e in summary["events"] if e["status"] == "confirmed"]
    bundle = ReplayBundle(
        snapshot=initial,
        controls=controls.between(0, controls.newest_substep),
        reference_frames=[],
        trigger={"source": "episode"},
        metadata={"task": task, "seed": seed},
    )

    report = {
        "segment": str(segment),
        "task": task,
        "seed": seed,
        "phase": meta.get("phase"),
        "outcome": meta.get("outcome"),
        "substeps": summary["substeps"],
        "live_confirmed_events": len(live_events),
        "live_event_onsets": [e["onset_substep"] for e in live_events],
        "window": [args.window_before, args.window_after],
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
            targets = adapter.body_ids_with_role(BodyRole.TARGET)
            result = replay_bundle(
                adapter,
                bundle,
                method="none",
                body_ids=sorted(live_states.body_ids),
                detectors=default_detectors(ejection_gate=gate),
                before_replay=apply,
                position_tolerance_m=1e-4,
                episode_id=f"intervention:{name}",
                keep_state_log=True,
            )
            replayed = result.state_log
            target = targets[0]
            confirmed = result.confirmed_events
            onsets = sorted(e["onset_substep"] for e in confirmed)
            matched = [
                any(abs(o - live["onset_substep"]) <= args.match_tolerance for o in onsets) for live in live_events
            ]
            condition = {
                "name": name,
                "description": description,
                "target_body": target,
                "target_mass_kg": adapter.bodies()[target].mass,
                "solver_iterations": adapter.solver_iterations(),
                "episode_peak_target_speed_mps": float(np.nanmax(speed_series(replayed, target))),
                "confirmed_events": len(confirmed),
                "confirmed_onsets": onsets[:20],
                "recorded_events_reproduced": int(sum(matched)),
                "event_window_peak_speed_mps": [
                    round(window_peak(replayed, target, e["onset_substep"], args.window_before, args.window_after), 3)
                    for e in live_events
                ],
                "flag_counts": _count(result.events),
            }
            if name == "baseline":
                condition["matches_recorded_trajectory"] = result.within_tolerance
                condition["max_position_error_m"] = result.overall_max_error_m
                live = live_states.array()
                rep = replayed.array()[:, [replayed.body_ids.index(b) for b in live_states.body_ids], :]
                condition["state_log_bit_identical"] = bool(
                    live.shape == rep.shape and np.array_equal(live, rep, equal_nan=True)
                )
            report["conditions"].append(condition)
            print(json.dumps({k: condition[k] for k in list(condition)[:9]}, default=str)[:500], flush=True)
        finally:
            try:
                env.close_env()
            except Exception:  # noqa: BLE001
                pass

    Path(args.report).write_text(json.dumps(report, indent=2, default=str))
    baseline = next((c for c in report["conditions"] if c["name"] == "baseline"), None)
    if baseline and not (baseline.get("state_log_bit_identical") and baseline["recorded_events_reproduced"]):
        print("BASELINE GATE FAILED: replay did not reproduce the recording; interventions are not interpretable")
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
