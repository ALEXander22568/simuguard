#!/usr/bin/env python3
"""Where does a rebuilt-scene replay of a recorded RoboCasa episode first diverge?

Rebuilds the scene from the episode's seed, compares the fresh state with the recorded initial
snapshot component by component, restores the snapshot, replays the recorded actuation and
reports the first substep at which any logged object leaves the recorded trajectory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from xr1_rollout import BASE_SEED, install_persistent_renderer, roles_for, unwrap  # noqa: E402

from simuguard.adapters.mujoco import attach_robosuite  # noqa: E402
from simuguard.core.controls import ControlLog  # noqa: E402
from simuguard.core.snapshot import Snapshot  # noqa: E402
from simuguard.core.statelog import StateLog  # noqa: E402

COMPONENTS = [("TIME", 1), ("QPOS", "nq"), ("QVEL", "nv"), ("ACT", "na"), ("WARMSTART", "nv"), ("CTRL", "nu"),
              ("QFRC_APPLIED", "nv"), ("XFRC_APPLIED", "nbody6"), ("EQ_ACTIVE", "neq"), ("MOCAP_POS", "nmocap3"),
              ("MOCAP_QUAT", "nmocap4"), ("USERDATA", "nuserdata"), ("PLUGIN", "npluginstate")]


def split_state(model, state: np.ndarray) -> dict:
    sizes = {"nq": model.nq, "nv": model.nv, "na": model.na, "nu": model.nu, "nbody6": 6 * model.nbody,
             "neq": model.neq, "nmocap3": 3 * model.nmocap, "nmocap4": 4 * model.nmocap,
             "nuserdata": model.nuserdata, "npluginstate": model.npluginstate}
    out, i = {}, 0
    for name, size in COMPONENTS:
        n = size if isinstance(size, int) else sizes[size]
        out[name] = state[i:i + n]
        i += n
    assert i == len(state), (i, len(state))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("segment")
    ap.add_argument("--substeps", type=int, default=5000)
    args = ap.parse_args()
    seg = Path(args.segment)
    meta = json.loads((seg / "manifest.json").read_text())["metadata"]
    controls = ControlLog.load(seg / "controls.npz")
    initial = Snapshot.load(seg / "initial_snapshot.json.gz")
    ref = StateLog.load(seg / "states.npz")
    keys = sorted({k for r in list(controls.records())[:2000] for k in r.payload})
    print("control payload keys:", keys)

    import gymnasium as gym
    import robocasa  # noqa: F401

    install_persistent_renderer()
    env = gym.make(f"robocasa/{meta['task']}", split="pretrain", seed=BASE_SEED)
    env.reset(seed=int(meta["seed"]))
    rs = unwrap(env)
    roles, _ = roles_for(rs)
    adapter = attach_robosuite(rs, roles=roles, task_name=meta["task"])
    fresh = split_state(adapter.model, adapter.native_state())
    recorded = split_state(adapter.model, np.frombuffer(initial.native_state, dtype=np.float64))
    for name in fresh:
        if len(fresh[name]):
            d = np.abs(fresh[name] - recorded[name])
            print(f"  initial {name:12s} max |fresh - recorded| = {d.max():.3e}" + (f" at index {int(d.argmax())}" if d.max() > 0 else ""))
    adapter.restore_snapshot(initial, method="native")
    ids, arr = ref.body_ids, ref.array()
    row = {int(s): i for i, s in enumerate(ref.substeps)}
    first = None
    for record in controls.between(0, args.substeps):
        adapter.apply_control(record)
        adapter.step_physics()
        if record.substep in row:
            now = adapter.read_states(ids)
            errs = {b: float(np.abs(now[b].position - arr[row[record.substep], j, :3]).max()) for j, b in enumerate(ids)}
            worst = max(errs, key=errs.get)
            if errs[worst] > 1e-9 and first is None:
                first = record.substep
                print(f"first divergence at substep {first} (control step {record.control_step}): {worst} off by {errs[worst]:.3e} m")
            if record.substep % 500 == 0 or (first is not None and record.substep - first in (1, 10, 100)):
                print(f"  substep {record.substep:6d}: worst {worst} {errs[worst]:.3e} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
