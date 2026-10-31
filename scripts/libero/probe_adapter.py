#!/usr/bin/env python3
"""Smoke test of the MuJoCo adapter on one LIBERO task.

Checks, on a real LIBERO scene (any robosuite stack the venv provides):
  1. inventory: robosuite / MuJoCo versions, lite_physics and the hooked method, timestep and
     substeps per control step, solver settings, body kinds and the goal-derived roles;
  2. contacts: after the 15 settle steps LIBERO evaluations start with, the net contact force on
     every free object divided by its weight (a resting object reads ~1), deepest penetrations,
     and that the monitor's detectors ran without errors;
  3. overhead: env step time with and without the SimuGuard monitor (rendering included);
  4. exact replay in-process: native snapshot, N control steps of a scripted reach, restore,
     re-apply the recorded actuation, compare every free body; the same replay with the recorded
     warm starts dropped shows whether the benchmark changes them between steps;
  5. one agentview frame (upright) saved as PNG, to look at.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simuguard.adapters.mujoco.libero import attach_libero, make_env, robosuite_env  # noqa: E402
from simuguard.core.monitor import MonitorConfig, SubstepMonitor  # noqa: E402
from simuguard.core.snapshot import ControlRecord  # noqa: E402
from simuguard.core.types import BodyKind, BodyRole  # noqa: E402
from simuguard.presets import default_detectors  # noqa: E402

SETTLE = np.array([0.0] * 6 + [-1.0])


def scripted(step: int) -> np.ndarray:
    """A slow reach down and back with the gripper closing half way (touches nothing on most tasks)."""
    a = np.zeros(7)
    a[2] = -0.4 if (step // 20) % 2 == 0 else 0.4
    a[0] = 0.2 * np.sin(step / 7.0)
    a[6] = 1.0 if step >= 30 else -1.0
    return a


def net_contact_force(adapter, body_id: str) -> np.ndarray:
    total = np.zeros(3)
    for pair in adapter.read_contacts():
        if body_id not in (pair.body_a, pair.body_b):
            continue
        sign = 1.0 if pair.body_b == body_id else -1.0  # impulses are oriented from body_a to body_b
        for point in pair.points:
            total += sign * np.asarray(point.impulse)
    return total / adapter.timestep()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--init", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timing-steps", type=int, default=40)
    ap.add_argument("--replay-steps", type=int, default=60)
    ap.add_argument("--out", default="probe")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    env, obs, meta = make_env(args.suite, args.task, args.init, args.seed)
    build_s = time.time() - t0
    rs = robosuite_env(env)
    adapter, info = attach_libero(env, task_name=meta["task"])
    model, data = adapter.model, adapter.data
    report: dict = {"episode": meta, "build_s": round(build_s, 2), "libero": {k: v for k, v in info.items() if k != "goal_state"},
                    "goal_state": info["goal_state"],
                    "model": {"nbody": model.nbody, "ngeom": model.ngeom, "nv": model.nv, "nu": model.nu, "neq": model.neq,
                              "nmocap": model.nmocap, "integrator": int(model.opt.integrator), "cone": int(model.opt.cone),
                              "solver": int(model.opt.solver), "iterations": int(model.opt.iterations),
                              "impratio": float(model.opt.impratio)}}
    rows, counts = np.unique(np.round(model.geom_solref, 5), axis=0, return_counts=True)
    report["geom_solref"] = {str(r.tolist()): int(c) for r, c in zip(rows, counts)}
    rows, counts = np.unique(np.round(model.geom_solimp, 4), axis=0, return_counts=True)
    report["geom_solimp"] = {str(r.tolist()): int(c) for r, c in zip(rows, counts)}
    bodies = adapter.bodies()
    kinds: dict = {}
    for b in bodies.values():
        key = f"{b.kind.value}/{b.role.value}"
        kinds[key] = kinds.get(key, 0) + 1
    report["body_kinds_roles"] = kinds
    free = sorted(b for b, i in bodies.items() if i.kind == BodyKind.DYNAMIC)
    report["free_bodies"] = {b: {"role": bodies[b].role.value, "mass_kg": round(bodies[b].mass, 5)} for b in free}

    # 1) contacts at rest, detectors running: attach the monitor, settle like LIBERO evaluations do
    monitor = SubstepMonitor(adapter, default_detectors({"ejection": {"delta_v_reference_timestep_s": 0.004}}),
                             episode_id="probe", config=MonitorConfig(snapshot_interval_substeps=100, snapshot_capacity=64,
                                                                      save_episode_logs=False))
    monitor.attach()
    for _ in range(15):
        obs, *_ = env.step(SETTLE.tolist())
    rest = {}
    for b in free:
        mass = bodies[b].mass
        f = net_contact_force(adapter, b)
        speed = float(np.linalg.norm(adapter.read_states([b])[b].linear_velocity))
        rest[b] = {"net_force_z_over_weight": round(float(f[2] / (mass * 9.81)), 4),
                   "net_force_xy_over_weight": round(float(np.linalg.norm(f[:2]) / (mass * 9.81)), 4),
                   "speed_mps": round(speed, 6)}
    report["resting_contacts"] = rest
    pairs = adapter.read_contacts()
    deepest = sorted((min(p.separation for p in c.points), c.body_a, c.body_b, len(c.points)) for c in pairs)[:8]
    report["contacts"] = {"pairs_watched": len(pairs), "ncon_total": int(data.ncon),
                          "deepest_mm": [(round(1000 * d, 3), a, b, n) for d, a, b, n in deepest]}

    # 2) overhead: monitored vs plain env steps (same scripted actions)
    t0 = time.time()
    for k in range(args.timing_steps):
        env.step(scripted(k).tolist())
    monitored = (time.time() - t0) / args.timing_steps
    summary = monitor.finalize()
    report["monitor"] = {k: summary[k] for k in ("substeps", "tracked_bodies", "confirmed_count", "flag_count", "error_count",
                                                  "event_counts_by_detector")}
    report["monitor_flags"] = [{"detector": e["detector"], "kind": e["kind"], "bodies": e["bodies"], "substep": e["onset_substep"],
                                "metrics": {k: v for k, v in (e.get("metrics") or {}).items() if isinstance(v, (int, float))}}
                               for e in summary["events"] if e.get("status") in ("flag", "confirmed")][:6]
    report["monitor_errors"] = summary["errors"][:3]
    t0 = time.time()
    for k in range(args.timing_steps):
        env.step(scripted(k).tolist())
    plain = (time.time() - t0) / args.timing_steps
    # the campaign configuration: recorder (per-substep trace, state log, control log, events) + snapshots
    from simuguard.core.recorder import EpisodeRecorder

    full = SubstepMonitor(adapter, default_detectors({"ejection": {"delta_v_reference_timestep_s": 0.004}}),
                          episode_id="probe_full", recorder=EpisodeRecorder(out / "segment"),
                          config=MonitorConfig.from_dict({"snapshot_interval_substeps": 250, "snapshot_capacity": 2000,
                                                          "control_log_maxlen": None, "bundle_mode": "snapshot",
                                                          "save_snapshots": True}))
    full.attach()
    t0 = time.time()
    for k in range(args.timing_steps):
        env.step(scripted(k).tolist())
    recorded = (time.time() - t0) / args.timing_steps
    t0 = time.time()
    full.finalize()
    finalize_s = time.time() - t0
    per = info["substeps_per_step"]
    report["timing"] = {"env_step_s_plain": round(plain, 4), "env_step_s_monitored": round(monitored, 4),
                        "env_step_s_monitored_recorded": round(recorded, 4), "finalize_s": round(finalize_s, 3),
                        "monitor_ms_per_substep": round(1000 * (monitored - plain) / per, 3),
                        "monitor_and_recorder_ms_per_substep": round(1000 * (recorded - plain) / per, 3)}

    # 3) exact replay in-process (hook only: records actuation before and free-body states after each step)
    adapter.mark_post_step()  # the plain loop above ran without the hook
    snap = adapter.capture_snapshot(0)
    controls: list[ControlRecord] = []
    ref: list[dict] = []
    hook = adapter.install_substep_hook(lambda: ref.append(adapter.read_states(free)),
                                        before=lambda: controls.append(adapter.capture_control()))
    for k in range(args.replay_steps):
        env.step(scripted(k).tolist())
    hook.remove()
    keys = sorted({key for c in controls for key in c.payload})

    def replay(drop: tuple[str, ...] = ()) -> dict:
        adapter.restore_snapshot(snap, method="native")
        errs = []
        for i, c in enumerate(controls):
            payload = {k: v for k, v in c.payload.items() if k not in drop}
            adapter.apply_control(ControlRecord(c.substep, c.control_step, payload))
            adapter.step_physics()
            now = adapter.read_states(free)
            errs.append(max(float(np.abs(now[b].position - ref[i][b].position).max()) for b in free))
        first = next((i for i, e in enumerate(errs) if e > 0), None)
        return {"substeps": len(errs), "max_abs_pos_err_m": max(errs), "first_nonzero_substep": first}

    report["replay"] = {"control_payload_keys": keys,
                        "records_with_warmstart": sum("qacc_warmstart" in c.payload for c in controls),
                        "exact": replay(), "without_recorded_warmstart": replay(drop=("qacc_warmstart",))}

    # 4) a frame to look at (robosuite renders bottom-up; flip rows for an upright image)
    try:
        import imageio.v2 as imageio

        frame = np.concatenate([obs["agentview_image"][::-1], obs["robot0_eye_in_hand_image"][::-1]], axis=1)
        imageio.imwrite(out / "frame.png", frame)
        report["frame"] = {"path": str(out / "frame.png"), "mean": float(frame.mean()), "std": float(frame.std())}
    except Exception as exc:  # noqa: BLE001
        report["frame"] = {"error": f"{type(exc).__name__}: {exc}"}
    (out / "probe.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
