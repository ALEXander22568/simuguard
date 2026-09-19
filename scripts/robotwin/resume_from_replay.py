#!/usr/bin/env python3
"""Replay a recorded episode up to a chosen substep, then hand control back.

Why: comparing two closed-loop evaluations from the episode start mixes the
effect of the changed setting with the policy's own run-to-run variation.
Replaying to a common state first removes that: every condition starts from the
*identical* pre-event state, and only one thing afterwards differs (the
simulation setting, or which controller drives the arm).

Modes after the resume point:
* ``hold``   - command the current joint targets (no policy needed).  Used to
  check that the environment is still drivable after a replay.
* ``policy`` - connect to a running LingBot-VA bridge and run the official
  observe/act loop.

Caveat for ``policy``: the policy is reset at the resume point, so it restarts
its own visual history from that frame; it does not continue the context it had
built before.  Rebuilding that context would require re-rendering and replaying
the observation history.

Usage::

    python scripts/robotwin/resume_from_replay.py --robotwin-root <RoboTwin> \\
        --segment-dir <run>/simuguard/segments/<seg> --resume-before-event 1 \\
        --mode policy --host 127.0.0.1 --port <bridge port> --steps 120 \\
        --intervention mass_100g --output-dir runs/resume_x
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SIMUGUARD_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIMUGUARD_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from replay_intervention import build_intervention  # noqa: E402
from simuguard.adapters.robotwin import RoboTwinAdapter, make_task_env  # noqa: E402
from simuguard.adapters.robotwin.env import official_eval_module  # noqa: E402
from simuguard.core import ControlLog, EpisodeRecorder, Snapshot, StateLog, SubstepMonitor  # noqa: E402
from simuguard.core.types import BodyRole  # noqa: E402
from simuguard.integrations.robotwin_eval import eval_monitor_config  # noqa: E402
from simuguard.presets import default_detectors  # noqa: E402


def replay_to(adapter: RoboTwinAdapter, controls: list, until_substep: int) -> int:
    applied = 0
    for record in controls:
        if record.substep > until_substep:
            break
        if record.substep == 0:
            continue
        adapter.apply_control(record)
        adapter.scene.step()  # hooked when a monitor is attached
        applied += 1
    return applied


def hold_actions(env, steps: int) -> int:
    left = list(env.robot.get_left_arm_jointState())
    right = list(env.robot.get_right_arm_jointState())
    action = np.asarray(left + right, dtype=float)
    done = 0
    for _ in range(steps):
        env.take_action(action, action_type="qpos")
        done += 1
        if env.eval_success:
            break
    return done


def policy_actions(module, env, args, instruction: str, steps: int, should_stop=None) -> dict:
    """Official observe/act loop.  ``should_stop()`` is checked after every action (e.g. abort on an event)."""

    usr_args = {
        "protocol": "ws", "host": args.host, "port": args.port,
        "policy_name": "LingBot_VA", "bench_name": "RoboTwin", "task_name": args.task or env.task_name,
        "evaluation_id": f"simuguard-resume-{int(time.time())}", "trial_id": f"resume-{args.resume_substep}",
        "action_case_id": "resume", "request_timeout_s": args.request_timeout_s,
        "xpolicylab_root": str(Path(args.robotwin_root).resolve() / "XPolicyLab"),
    }
    client = module.build_policy_client(usr_args)
    stats = {"actions": 0, "chunks": 0, "success": False, "stopped": False}
    try:
        module.prepare_policy_case(client, env.task_name, int(args.seed_for_policy), instruction, "joint")
        module.reset_policy(client)
        observation = env.get_obs()
        while stats["actions"] < steps and not module.is_episode_end(env):
            payload = module.robotwin_obs_to_xpolicylab(
                observation, instruction=instruction, env_idx=0, frequency=args.frequency, task_env=env
            )
            client.call(func_name="update_obs", obs=payload)
            chunk = module.normalize_action_chunk(client.call(func_name="get_action"))
            stats["chunks"] += 1
            if not chunk:
                break
            for index, action in enumerate(chunk):
                flat, action_type = module.xpolicylab_action_to_robotwin(
                    action, action_type="joint", current_observation=observation
                )
                env.take_action(flat, action_type=action_type)
                stats["actions"] += 1
                if env.eval_success:
                    stats["success"] = True
                    break
                if should_stop is not None and should_stop():
                    stats["stopped"] = True
                    break
                if module.is_episode_end(env) or stats["actions"] >= steps or index + 1 == len(chunk):
                    break
                observation = env.get_obs()
                client.call(func_name="update_obs", obs=module.robotwin_obs_to_xpolicylab(
                    observation, instruction=instruction, env_idx=0, frequency=args.frequency, task_env=env
                ))
            if stats["success"] or stats["stopped"]:
                break
            observation = env.get_obs()
    finally:
        module.close_policy_client(client)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--segment-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("hold", "policy"), default="hold")
    parser.add_argument("--resume-substep", type=int, default=None)
    parser.add_argument("--resume-before-event", type=int, default=None,
                        help="resume this many substeps before the first recorded event (see --margin)")
    parser.add_argument("--margin", type=int, default=250)
    parser.add_argument("--steps", type=int, default=120, help="policy actions after the resume point")
    parser.add_argument("--intervention", default="baseline")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=None)
    parser.add_argument("--frequency", type=int, default=30)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--task", default=None)
    args = parser.parse_args()

    segment = Path(args.segment_dir).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((segment / "manifest.json").read_text())
    summary = json.loads((segment / "summary.json").read_text())
    meta = {**manifest["metadata"], **(summary.get("metadata") or {})}
    task, seed = args.task or meta["task"], int(meta["seed"])
    args.seed_for_policy = seed
    controls = ControlLog.load(segment / "controls.npz")
    live_states = StateLog.load(segment / "states.npz")
    events = [e for e in summary["events"] if e["status"] == "confirmed"]

    if args.resume_substep is None:
        if not events:
            raise SystemExit("segment has no confirmed events; pass --resume-substep explicitly")
        index = max(1, args.resume_before_event or 1) - 1
        args.resume_substep = max(1, events[index]["onset_substep"] - args.margin)

    module = official_eval_module(args.robotwin_root)  # chdir + sys.path, like the official evaluator
    env, _task_args = make_task_env(args.robotwin_root, task, seed)
    report = {
        "segment": str(segment), "task": task, "seed": seed, "phase": meta.get("phase"),
        "recorded_substeps": summary["substeps"], "recorded_events": len(events),
        "resume_substep": args.resume_substep, "mode": args.mode, "intervention": args.intervention,
        "steps_requested": args.steps,
    }
    try:
        adapter = RoboTwinAdapter(env)
        gate = None
        try:
            gate = adapter.containment_gate()
        except Exception:  # noqa: BLE001
            pass
        monitor = SubstepMonitor(
            adapter, default_detectors(ejection_gate=gate), episode_id=f"resume:{segment.name}",
            config=eval_monitor_config({"bundle_mode": "snapshot"}),
            recorder=EpisodeRecorder(out / "episode"),
            metadata={"task": task, "seed": seed, "resume_substep": args.resume_substep,
                      "mode": args.mode, "intervention": args.intervention, "source_segment": segment.name},
        )
        monitor.attach()

        replayed = replay_to(adapter, controls.between(0, controls.newest_substep), args.resume_substep)
        report["replayed_substeps"] = replayed
        # fidelity of the replayed prefix
        live = live_states.array()
        mask = live_states.substeps <= args.resume_substep
        rep = monitor.state_log.array()
        cols = [monitor.state_log.body_ids.index(b) for b in live_states.body_ids]
        n = min(int(mask.sum()), rep.shape[0])
        report["prefix_bit_identical"] = bool(np.array_equal(live[mask][:n], rep[:n][:, cols, :], equal_nan=True))
        report["events_during_replay"] = monitor.summary()["confirmed_count"]

        if args.intervention != "baseline":
            _, apply = build_intervention(args.intervention)
            if apply is not None:
                apply(adapter)
        report["target_mass_kg"] = {b: adapter.bodies()[b].mass for b in adapter.body_ids_with_role(BodyRole.TARGET)}
        report["solver_iterations"] = adapter.solver_iterations()

        # hand control back to the benchmark's own action interface
        env.take_action_cnt = int(controls.get(args.resume_substep).control_step)
        env.eval_success = False
        report["take_action_cnt_at_resume"] = env.take_action_cnt
        started = time.time()
        if args.mode == "hold":
            report["actions_executed"] = hold_actions(env, args.steps)
        else:
            if not args.port:
                raise SystemExit("--port is required for --mode policy")
            instruction = summary.get("metadata", {}).get("outcome", {}).get("instruction") or task
            report["instruction"] = instruction
            report["policy"] = policy_actions(module, env, args, instruction, args.steps)
            report["actions_executed"] = report["policy"]["actions"]
        report["resume_wall_s"] = round(time.time() - started, 1)
        report["success_after_resume"] = bool(env.eval_success)
        report["check_success_after_resume"] = bool(env.check_success())

        final = monitor.finalize()
        report["substeps_total"] = final["substeps"]
        report["substeps_after_resume"] = final["substeps"] - replayed
        report["confirmed_events_total"] = final["confirmed_count"]
        report["confirmed_events_after_resume"] = sum(
            1 for e in final["events"] if e["status"] == "confirmed" and e["onset_substep"] > args.resume_substep
        )
        report["event_counts_by_detector"] = final["event_counts_by_detector"]
        target = adapter.body_ids_with_role(BodyRole.TARGET)[0]
        speeds = monitor.state_log.array()[:, monitor.state_log.body_ids.index(target), 7:10]
        after = monitor.state_log.substeps > args.resume_substep
        report["peak_target_speed_after_resume_mps"] = (
            float(np.nanmax(np.linalg.norm(speeds[after], axis=1))) if after.any() else None
        )
    finally:
        try:
            env.close_env()
        except Exception:  # noqa: BLE001
            pass

    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=1, default=str)[:1800])
    return 0 if report.get("prefix_bit_identical") else 1


if __name__ == "__main__":
    raise SystemExit(main())
