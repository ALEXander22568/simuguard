#!/usr/bin/env python3
"""End-to-end check of the RoboTwin adapter on a real (unmodified) RoboTwin checkout.

1. Build ``envs.<task>`` with the official evaluator's argument loader.
2. Attach SubstepMonitor (ground truth, contacts, detectors, snapshots, control log).
3. Run the task's scripted expert ``play_once()`` (no policy server required).
4. Replay from in-episode snapshots with native (PhysX pack) and public restore
   and report trajectory fidelity against the recorded reference.

Usage (on the GPU node, simulator GPU only)::

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/path/to/SimuGuard \\
      python scripts/check_robotwin_integration.py --robotwin-root /path/to/RoboTwin \\
      --task place_can_basket --seed 100000 --output-dir runs/integration
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

SIMUGUARD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMUGUARD_ROOT))

from simuguard import __version__  # noqa: E402
from simuguard.adapters.robotwin import RoboTwinAdapter, make_task_env, robotwin_provenance  # noqa: E402
from simuguard.core import EpisodeRecorder, MonitorConfig, SubstepMonitor, build_replay_bundle, replay_bundle  # noqa: E402
from simuguard.core.types import BodyRole  # noqa: E402
from simuguard.presets import default_detectors  # noqa: E402


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--task", default="place_can_basket")
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--snapshot-interval", type=int, default=100)
    parser.add_argument("--replay-substeps", type=int, default=250)
    parser.add_argument("--replay-points", type=int, default=3)
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "simuguard_version": __version__,
        "simuguard_commit": git_commit(SIMUGUARD_ROOT),
        "task": args.task,
        "seed": args.seed,
        "robotwin": robotwin_provenance(args.robotwin_root),
    }

    t0 = time.time()
    env, _task_args = make_task_env(args.robotwin_root, args.task, args.seed)
    report["setup_wall_s"] = round(time.time() - t0, 2)
    try:
        adapter = RoboTwinAdapter(env)
        bodies = adapter.bodies()
        report["inventory"] = {
            "timestep_s": adapter.timestep(),
            "body_count": len(bodies),
            "by_role": dict(Counter(info.role.value for info in bodies.values())),
            "by_kind": dict(Counter(info.kind.value for info in bodies.values())),
            "targets": adapter.body_ids_with_role(BodyRole.TARGET),
            "containers": adapter.body_ids_with_role(BodyRole.CONTAINER),
            "free_objects": sorted(b for b, i in bodies.items() if i.role in (BodyRole.TARGET, BodyRole.CONTAINER, BodyRole.OBJECT)),
            "capabilities": adapter.capabilities().__dict__,
            "task_ground_truth": adapter.task_ground_truth(),
            "solver_iterations": adapter.solver_iterations(),
        }
        states = adapter.read_states(report["inventory"]["targets"] + report["inventory"]["containers"])
        report["inventory"]["initial_states"] = {k: v.to_dict(6) for k, v in states.items()}

        gate = adapter.containment_gate()
        monitor = SubstepMonitor(
            adapter,
            default_detectors(ejection_gate=gate),
            episode_id=f"{args.task}-seed{args.seed}-expert",
            config=MonitorConfig(
                snapshot_interval_substeps=args.snapshot_interval,
                snapshot_capacity=10_000,
                frame_buffer_substeps=200_000,
                control_log_maxlen=200_000,
            ),
            recorder=EpisodeRecorder(output / "episode"),
            metadata={"policy": "robotwin_expert_play_once", "seed": args.seed},
        )
        monitor.attach()
        report["hook_mechanism"] = monitor._hook.mechanism if monitor._hook else None

        t0 = time.time()
        play_error = None
        try:
            env.play_once()
        except Exception as exc:  # noqa: BLE001
            play_error = f"{type(exc).__name__}: {exc}"
        report["play_once_wall_s"] = round(time.time() - t0, 2)
        summary = monitor.finalize()
        report["expert"] = {
            "error": play_error,
            "plan_success": bool(getattr(env, "plan_success", False)),
            "check_success": bool(env.check_success()),
        }
        contact_pairs = Counter()
        max_target_contact_force = 0.0
        target = report["inventory"]["targets"][0] if report["inventory"]["targets"] else None
        for frame in monitor.frames:
            for pair in frame.contacts:
                if target and pair.involves(target):
                    contact_pairs[pair.other(target)] += 1
                    max_target_contact_force = max(max_target_contact_force, pair.force_estimate(frame.timestep))
        report["monitor"] = {
            "substeps": summary["substeps"],
            "simulated_time_s": summary["simulated_time_s"],
            "error_count": summary["error_count"],
            "errors": summary["errors"][:5],
            "event_counts_by_detector": summary["event_counts_by_detector"],
            "confirmed_count": summary["confirmed_count"],
            "snapshots_held": len(summary["snapshots_held"]),
            "target_contact_substeps_by_partner": dict(contact_pairs.most_common(10)),
            "target_max_contact_force_estimate_n": max_target_contact_force,
            "containment_gate_active_substeps": sum(
                1 for f in monitor.frames if target and gate and gate(f, target, monitor.context)["active"]
            ) if gate else None,
        }

        # ---- replay fidelity from in-episode snapshots -----------------------
        last = summary["substeps"]
        held = [s for s in summary["snapshots_held"] if 0 < s <= last - args.replay_substeps]
        picks = sorted({held[int(i * (len(held) - 1) / max(1, args.replay_points - 1))] for i in range(args.replay_points)}) if held else []
        robot_ids = set(adapter.body_ids_with_role(BodyRole.ROBOT))
        reference_ids = set(report["inventory"]["free_objects"]) | robot_ids
        replays = []
        for start in picks:
            end = start + args.replay_substeps
            frames = [f.to_dict(reference_ids, decimals=None) for f in monitor.frames if start < f.substep <= end]
            bundle = build_replay_bundle(monitor.snapshots, monitor.controls, trigger_substep=start, end_substep=end, reference_frames=frames)
            for method in ("native", "public"):
                result = replay_bundle(adapter, bundle, method=method, body_ids=sorted(reference_ids), position_tolerance_m=1e-4)
                replays.append(
                    {
                        "start_substep": start,
                        "substeps": args.replay_substeps,
                        "method": method,
                        "restore_ok": result.restore["ok"],
                        "restore_public_state_max_abs_error": result.restore["public_state_error"]["max_abs_error"],
                        "overall_max_position_error_m": result.overall_max_error_m,
                        "within_1e-4_m": result.within_tolerance,
                        "object_max_position_error_m": {k: v for k, v in result.max_position_error_m.items() if k not in robot_ids},
                        "robot_max_position_error_m": max((v for k, v in result.max_position_error_m.items() if k in robot_ids), default=None),
                        "first_divergent_bodies": sorted(
                            ((v, k) for k, v in result.first_substep_over_tolerance.items() if v is not None)
                        )[:5],
                    }
                )
        report["replay_fidelity"] = replays

        # ---- diagnostic A: is restore+replay itself deterministic? -------------
        if picks:
            start = picks[len(picks) // 2]
            end = start + args.replay_substeps
            frames = [f.to_dict(reference_ids, decimals=None) for f in monitor.frames if start < f.substep <= end]
            bundle = build_replay_bundle(monitor.snapshots, monitor.controls, trigger_substep=start, end_substep=end, reference_frames=frames)
            first = replay_bundle(adapter, bundle, method="native", body_ids=sorted(reference_ids), keep_trajectory=True)
            second = replay_bundle(adapter, bundle, method="native", body_ids=sorted(reference_ids), keep_trajectory=True)
            worst = 0.0
            for substep, positions in first.trajectory.items():
                for body_id, p in positions.items():
                    q = second.trajectory[substep][body_id]
                    worst = max(worst, max(abs(a - b) for a, b in zip(p, q)))
            report["diagnostic_replay_vs_replay"] = {"start_substep": start, "substeps": args.replay_substeps, "max_abs_position_difference_m": worst}

        # ---- diagnostic B: rebuild env with same seed, replay full control log --
        rebuild_end = min(last, (picks[len(picks) // 2] + args.replay_substeps) if picks else last)
        initial = monitor.snapshots.latest_at_or_before(0)
        frames = [f.to_dict(reference_ids, decimals=None) for f in monitor.frames if 0 < f.substep <= rebuild_end]
        full_bundle = build_replay_bundle(monitor.snapshots, monitor.controls, trigger_substep=0, end_substep=rebuild_end, reference_frames=frames)
        try:
            env.close_env()
        except Exception:  # noqa: BLE001
            pass
        env, _ = make_task_env(args.robotwin_root, args.task, args.seed)
        rebuilt = RoboTwinAdapter(env)
        rebuilt_result = replay_bundle(rebuilt, full_bundle, method="none", body_ids=sorted(reference_ids), position_tolerance_m=1e-4)
        report["diagnostic_rebuild_from_episode_start"] = {
            "initial_snapshot_substep": initial.substep if initial else None,
            "rebuilt_env_matches_initial_public_state_max_abs_error": rebuilt_result.restore["public_state_error"]["max_abs_error"],
            "replayed_substeps": rebuilt_result.substeps,
            "overall_max_position_error_m": rebuilt_result.overall_max_error_m,
            "within_1e-4_m": rebuilt_result.within_tolerance,
            "first_divergent_bodies": sorted(
                ((v, k) for k, v in rebuilt_result.first_substep_over_tolerance.items() if v is not None)
            )[:5],
            "object_max_position_error_m": {k: v for k, v in rebuilt_result.max_position_error_m.items() if k not in robot_ids},
        }
        status = 0 if (summary["error_count"] == 0 and play_error is None) else 1
    except Exception:  # noqa: BLE001
        report["fatal"] = traceback.format_exc()
        status = 2
    finally:
        try:
            env.close_env()
        except Exception:  # noqa: BLE001
            pass

    (output / "integration_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in ("hook_mechanism", "expert", "monitor", "replay_fidelity", "diagnostic_replay_vs_replay", "diagnostic_rebuild_from_episode_start", "fatal")}, indent=1, default=str)[:6000])
    return status


if __name__ == "__main__":
    raise SystemExit(main())
