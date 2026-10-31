#!/usr/bin/env python3
"""Validate the official-evaluator integration without a policy server.

``live``   : drive an instrumented env with the official call sequence
             expert segment : setup_demo -> play_once -> close_env
             policy segment : setup_demo -> take_action loop (scripted qpos) -> close_env
             Checks lifecycle (phases, attach/finalize, hook removal, errors) and artifact sizes.
             ``--disable`` runs the identical sequence without monitors (wall-time baseline).

``replay`` : in a *fresh process*, rebuild one recorded segment's env and replay its
             control log three times:
               bare     : apply_control + scene.step()                        (timing baseline)
               off      : same + read all body states                          (reference)
               on       : same + full SubstepMonitor attached (hooked step)    (instrumented)
             Checks: rebuilt initial state == recorded; off == on bit-for-bit for ALL bodies;
             off == live states.npz bit-for-bit (cross-process exactness); monitor overhead.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SIMUGUARD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMUGUARD_ROOT))

from simuguard.adapters.robotwin import RoboTwinAdapter, make_task_env  # noqa: E402
from simuguard.adapters.robotwin.env import load_official_task_args, official_eval_module  # noqa: E402
from simuguard.core import ControlLog, EpisodeRecorder, Snapshot, StateLog, SubstepMonitor, compare_states  # noqa: E402
from simuguard.integrations.robotwin_eval import RoboTwinEvalInstrumentation, eval_monitor_config  # noqa: E402
from simuguard.presets import default_detectors  # noqa: E402


# ----------------------------------------------------------------------------- live
def scripted_policy_actions(env, n_actions: int) -> int:
    base_left = list(env.robot.get_left_arm_jointState())
    base_right = list(env.robot.get_right_arm_jointState())
    done = 0
    for k in range(n_actions):
        wave = 0.15 * np.sin(2 * np.pi * k / 30.0)
        left, right = list(base_left), list(base_right)
        left[1] += wave
        right[1] -= wave
        env.take_action(np.asarray(left + right, dtype=float), action_type="qpos")
        done += 1
        if env.eval_success:
            break
    return done


def run_live(args) -> int:
    root = Path(args.robotwin_root).resolve()
    out = Path(args.output_dir).resolve()
    module = official_eval_module(root)
    task_args = load_official_task_args(root, args.task)
    instrumentation = RoboTwinEvalInstrumentation(out, enabled=not args.disable, task_name=args.task)
    class_decorator = instrumentation.wrap_class_decorator(module.class_decorator)
    env = class_decorator(args.task)
    started = time.time()
    for episode, seed in enumerate(args.seeds):
        env.setup_demo(now_ep_num=episode, seed=seed, is_test=True, **task_args)
        env.play_once()
        env.close_env()
        env.setup_demo(now_ep_num=episode, seed=seed, is_test=True, **task_args)
        actions = scripted_policy_actions(env, args.policy_actions)
        env.close_env()
        print(f"seed {seed}: scripted policy actions={actions}", flush=True)
    wall = time.time() - started

    segments = instrumentation.segments
    checks = {
        "segment_count_ok": len(segments) == 2 * len(args.seeds),
        "phases_ok": [s["phase"] for s in segments] == ["expert", "policy"] * len(args.seeds),
        "instrumentation_errors": len(instrumentation.errors),
    }
    if not args.disable:
        checks.update(
            {
                "all_monitored": all(s["monitored"] for s in segments),
                "all_hooks_removed": all(s.get("monitor", {}).get("hook_removed") for s in segments),
                "monitor_errors": sum(s.get("monitor", {}).get("error_count", 1) for s in segments),
                "all_substeps_positive": all(s.get("monitor", {}).get("substeps", 0) > 0 for s in segments),
                "required_files_present": all(
                    all((Path(s["output_dir"]) / f).exists() for f in (
                        "manifest.json", "summary.json", "trace.jsonl.gz", "controls.npz", "states.npz", "initial_snapshot.json.gz"
                    ))
                    for s in segments
                ),
            }
        )
    instrumentation.write_run_summary({"wall_s": wall, "checks": checks, "mode": "live_validation", "seeds": args.seeds})
    print(json.dumps(checks, indent=1))
    return 0 if all(v in (True, 0) for v in checks.values()) else 1


# ----------------------------------------------------------------------------- replay
def _all_state_log(adapter: RoboTwinAdapter) -> StateLog:
    return StateLog(sorted(adapter.bodies()))


def _replay(env, adapter, controls, *, read_states: bool, monitor: SubstepMonitor | None = None):
    log = _all_state_log(adapter) if read_states else None
    step = env.scene.step  # hooked instance attribute when a monitor is attached
    started = time.perf_counter()
    for record in controls:
        adapter.apply_control(record)
        step()
        if log is not None:
            log.append(record.substep, adapter.read_states())
    elapsed = time.perf_counter() - started
    return log, elapsed


def _arrays_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


def run_replay(args) -> int:
    segment = Path(args.segment_dir).resolve()
    manifest = json.loads((segment / "manifest.json").read_text())
    meta = manifest["metadata"]
    task, seed, episode = meta["task"], int(meta["seed"]), int(meta.get("episode_index") or 0)
    controls_log = ControlLog.load(segment / "controls.npz")
    live_states = StateLog.load(segment / "states.npz")
    initial = Snapshot.load(segment / "initial_snapshot.json.gz")
    last = controls_log.newest_substep
    controls = controls_log.between(0, last)
    report: dict = {"segment": str(segment), "task": task, "seed": seed, "substeps": last,
                    "control_log_dtype": np.dtype(controls_log.dtype).name}

    def build():
        env, _ = make_task_env(args.robotwin_root, task, seed, episode_index=episode)
        return env, RoboTwinAdapter(env)

    # bare timing
    env, adapter = build()
    report["rebuilt_initial_public_state_max_abs_error"] = compare_states(
        initial.public_state, adapter.capture_public_state()
    )["max_abs_error"]
    _, report["bare_wall_s"] = _replay(env, adapter, controls, read_states=False)
    env.close_env()

    # off: reference trajectory of every body
    env, adapter = build()
    off_log, report["off_wall_s"] = _replay(env, adapter, controls, read_states=True)
    env.close_env()

    # on: full monitor attached, hooked scene.step
    env, adapter = build()
    with tempfile.TemporaryDirectory() as tmp:
        monitor = SubstepMonitor(
            adapter,
            default_detectors(ejection_gate=adapter.containment_gate()),
            episode_id="replay-on",
            config=eval_monitor_config(),
            recorder=EpisodeRecorder(Path(tmp) / "on"),
        )
        monitor.attach()
        on_log, report["on_wall_s"] = _replay(env, adapter, controls, read_states=True, monitor=monitor)
        summary = monitor.finalize()
    env.close_env()
    report["on_monitor"] = {k: summary[k] for k in ("substeps", "error_count", "confirmed_count", "flag_count")}

    off, on = off_log.array(), on_log.array()
    report["off_vs_on_bit_identical_all_bodies"] = _arrays_equal(off, on)
    report["off_vs_on_max_abs_diff"] = float(np.nanmax(np.abs(off - on))) if off.shape == on.shape else None
    report["bodies_compared_off_on"] = len(off_log.body_ids)

    index = {b: i for i, b in enumerate(off_log.body_ids)}
    live = live_states.array()
    columns = [index[b] for b in live_states.body_ids]
    rebuilt = off[:, columns, :]
    report["rebuilt_vs_live_bit_identical"] = _arrays_equal(rebuilt, live)
    report["rebuilt_vs_live_max_abs_diff"] = float(np.nanmax(np.abs(rebuilt - live))) if rebuilt.shape == live.shape else None
    report["bodies_compared_live"] = live_states.body_ids

    per = 1000.0 / max(last, 1)
    report["ms_per_substep"] = {
        "bare": report["bare_wall_s"] * per,
        "off_read_states": report["off_wall_s"] * per,
        "on_monitor_plus_read_states": report["on_wall_s"] * per,
        "monitor_overhead_vs_off": (report["on_wall_s"] - report["off_wall_s"]) * per,
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))
    ok = (
        report["rebuilt_initial_public_state_max_abs_error"] == 0.0
        and report["off_vs_on_bit_identical_all_bodies"]
        and report["rebuilt_vs_live_bit_identical"]
        and report["on_monitor"]["error_count"] == 0
    )
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    live = sub.add_parser("live")
    live.add_argument("--robotwin-root", required=True)
    live.add_argument("--task", default="place_can_basket")
    live.add_argument("--seeds", type=int, nargs="+", default=[100000, 100001])
    live.add_argument("--policy-actions", type=int, default=40)
    live.add_argument("--output-dir", required=True)
    live.add_argument("--disable", action="store_true")
    replay = sub.add_parser("replay")
    replay.add_argument("--robotwin-root", required=True)
    replay.add_argument("--segment-dir", required=True)
    replay.add_argument("--report", required=True)
    args = parser.parse_args()
    return run_live(args) if args.mode == "live" else run_replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
