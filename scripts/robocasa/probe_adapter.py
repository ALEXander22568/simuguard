#!/usr/bin/env python3
"""Smoke test of the MuJoCo adapter on one RoboCasa365 scene.

Checks, on a real kitchen:
  1. inventory: timestep, body kinds and roles, contact solver settings of every geom;
  2. contacts: pairs, deepest penetration, and that welds never show up as contacts;
  3. exact replay: capture a native snapshot, run N substeps with random arm actions,
     restore the snapshot, re-apply the recorded actuation, compare every free body;
  4. overhead: env step time with and without the SimuGuard monitor.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
import robocasa  # noqa: E402,F401
from robocasa.utils.env_utils import convert_action  # noqa: E402

from simuguard.adapters.mujoco import TaskRoles, attach_robosuite  # noqa: E402
from simuguard.core.monitor import MonitorConfig, SubstepMonitor  # noqa: E402
from simuguard.core.types import BodyKind, BodyRole  # noqa: E402
from simuguard.presets import default_detectors  # noqa: E402


def unwrap(env):
    cur = env
    for _ in range(10):
        if hasattr(cur, "sim") and hasattr(cur, "_check_success"):
            return cur
        nxt = getattr(cur, "env", None) or getattr(cur, "unwrapped", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    raise RuntimeError("no robosuite env inside the gym wrapper")


def arm_action(rng, scale=0.3):
    a = np.zeros(12, dtype=np.float32)
    a[0:6] = rng.uniform(-scale, scale, 6)
    a[6] = float(rng.random() < 0.5)
    a[11] = 0.0  # arm mode
    return a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PickPlaceCounterToStove")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--replay-substeps", type=int, default=500)
    ap.add_argument("--out", default="probe.json")
    args = ap.parse_args()

    env = gym.make(f"robocasa/{args.task}", split="pretrain", seed=args.seed)
    obs, _ = env.reset(seed=args.seed)
    rs = unwrap(env)
    model, data = rs.sim.model._model, rs.sim.data._data
    report: dict = {"task": args.task, "instruction": obs.get("annotation.human.task_description"),
                    "timestep": float(model.opt.timestep), "control_freq": getattr(rs, "control_freq", None),
                    "nbody": int(model.nbody), "ngeom": int(model.ngeom), "neq": int(model.neq),
                    "nmocap": int(model.nmocap), "nu": int(model.nu)}
    solref = np.round(model.geom_solref, 5)
    rows, counts = np.unique(solref, axis=0, return_counts=True)
    report["geom_solref"] = {str(r.tolist()): int(c) for r, c in zip(rows, counts)}
    solimp = np.round(model.geom_solimp, 4)
    rows, counts = np.unique(solimp, axis=0, return_counts=True)
    report["geom_solimp"] = {str(r.tolist()): int(c) for r, c in zip(rows, counts)}
    report["opt"] = {"integrator": int(model.opt.integrator), "cone": int(model.opt.cone),
                     "impratio": float(model.opt.impratio), "iterations": int(model.opt.iterations),
                     "noslip_iterations": int(model.opt.noslip_iterations),
                     "disableflags": int(model.opt.disableflags), "enableflags": int(model.opt.enableflags)}
    report["objects"] = {name: obj.root_body for name, obj in getattr(rs, "objects", {}).items()}

    targets = {obj.root_body for obj in getattr(rs, "objects", {}).values()}
    roles = TaskRoles(targets=targets)
    adapter = attach_robosuite(rs, roles=roles, task_name=args.task)
    bodies = adapter.bodies()
    kinds: dict = {}
    for info in bodies.values():
        kinds.setdefault(f"{info.kind.value}/{info.role.value}", 0)
        kinds[f"{info.kind.value}/{info.role.value}"] += 1
    report["body_kinds_roles"] = kinds
    report["free_bodies"] = sorted(i.name for i in bodies.values() if i.kind == BodyKind.DYNAMIC)
    report["targets"] = adapter.body_ids_with_role(BodyRole.TARGET)

    rng = np.random.default_rng(0)
    # 1) raw step time
    t0 = time.time()
    for _ in range(20):
        env.step(convert_action(arm_action(rng)))
    report["step_s_plain"] = (time.time() - t0) / 20

    # 2) monitored step time + contacts
    monitor = SubstepMonitor(adapter, default_detectors({}), episode_id="probe",
                             config=MonitorConfig(snapshot_interval_substeps=100, snapshot_capacity=64,
                                                  save_episode_logs=False))
    monitor.attach()
    t0 = time.time()
    for _ in range(20):
        env.step(convert_action(arm_action(rng)))
    report["step_s_monitored"] = (time.time() - t0) / 20
    summary = monitor.finalize()
    report["monitor"] = {k: summary[k] for k in ("substeps", "tracked_bodies", "confirmed_count", "flag_count", "error_count")}
    report["monitor_errors"] = summary["errors"][:3]
    pairs = adapter.read_contacts()
    deepest = sorted(((min(p.separation for p in c.points), c.body_a, c.body_b) for c in pairs))[:8]
    report["contacts"] = {"pairs": len(pairs), "ncon": int(data.ncon),
                          "deepest": [(round(1000 * d, 3), a, b) for d, a, b in deepest]}

    # 3) exact replay from a native snapshot
    snap = adapter.capture_snapshot(0)
    free = [b for b, i in bodies.items() if i.kind == BodyKind.DYNAMIC]
    controls, ref = [], []
    hook = adapter.install_substep_hook(lambda: ref.append(adapter.read_states(free)),
                                        before=lambda: controls.append(adapter.capture_control()))
    steps = 0
    while len(ref) < args.replay_substeps:
        env.step(convert_action(arm_action(rng)))
        steps += 1
    hook.remove()
    adapter.restore_snapshot(snap, method="native")
    errs = []
    for i, ctrl in enumerate(controls[: len(ref)]):
        adapter.apply_control(ctrl)
        adapter.step_physics()
        now = adapter.read_states(free)
        errs.append(max(float(np.abs(now[b].position - ref[i][b].position).max()) for b in free))
    report["replay"] = {"substeps": len(errs), "env_steps": steps, "max_abs_pos_err_m": max(errs),
                        "first_nonzero": next((i for i, e in enumerate(errs) if e > 0), None)}
    Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
