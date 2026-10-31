#!/usr/bin/env python3
"""Re-create one substep of a recorded LIBERO episode exactly and print its contacts geom by geom.

Rebuilds the episode (``make_env``; recorded placement written back if the rebuild differs),
restores the last archived native snapshot before the substep, replays the recorded actuation up
to it and prints every contact of ``--body`` as MuJoCo holds it after that substep (the contacts
the monitor saw): geom names and types, penetration, normal force.  The state is checked against
the recording before printing.

Usage::

    $PY scripts/libero/replay_inspect.py SEGMENT 3469 --body white_yellow_mug_1_main
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from check_replay import load_snapshots  # noqa: E402

from simuguard.adapters.mujoco.libero import attach_libero, make_env, model_fingerprint, restore_placement  # noqa: E402
from simuguard.core.controls import ControlLog  # noqa: E402
from simuguard.core.statelog import StateLog  # noqa: E402

GEOM_TYPES = {int(v): k[6:].lower() for k, v in vars(mujoco.mjtGeom).items() if k.startswith("mjGEOM_")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("segment")
    ap.add_argument("substep", type=int)
    ap.add_argument("--body", required=True, help="body name, e.g. white_yellow_mug_1_main")
    args = ap.parse_args()
    seg = Path(args.segment)
    manifest = json.loads((seg / "manifest.json").read_text())
    meta, fp = manifest["metadata"], manifest["task_ground_truth"]["model_fingerprint"]
    env, _, _ = make_env(meta["suite"], meta["task_id"], meta["init_index"], meta["seed"], camera_size=meta["camera_size"])
    adapter, _ = attach_libero(env, task_name=meta["task"])
    m, d = adapter.model, adapter.data
    if model_fingerprint(m)["sha256"] != fp["sha256"]:
        restore_placement(m, fp)
    archive = load_snapshots(seg / "snapshots.npz")
    start = max(s for s in archive if s < args.substep)
    adapter.set_native_state(archive[start])
    adapter.mark_post_step()
    for record in ControlLog.load(seg / "controls.npz").between(start, args.substep):
        adapter.apply_control(record)
        adapter.step_physics()
    ref = StateLog.load(seg / "states.npz")
    row = list(ref.substeps).index(args.substep)
    now = adapter.read_states(ref.body_ids)
    err = max(float(np.abs(now[b].position - ref.array()[row, j, :3]).max()) for j, b in enumerate(ref.body_ids))
    print(f"replayed substeps {start + 1}..{args.substep} from the snapshot at {start}: max position error vs recording {err} m")
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, args.body)
    force = np.zeros(6)
    name = lambda obj, i: mujoco.mj_id2name(m, obj, i) or f"#{i}"  # noqa: E731
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom[0]), int(c.geom[1])
        b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
        if body not in (b1, b2):
            continue
        mujoco.mj_contactForce(m, d, i, force)
        other = g2 if b1 == body else g1
        try:  # MuJoCo's own signed distance of the two geoms, for comparison with the contact's
            true = f"{1000 * mujoco.mj_geomDistance(m, d, g1, g2, 1.0, None):.1f} mm"
        except Exception as exc:  # noqa: BLE001
            true = f"n/a ({type(exc).__name__})"
        print(f"  contact {i}: {name(mujoco.mjtObj.mjOBJ_GEOM, g1)} ({GEOM_TYPES[int(m.geom_type[g1])]}) / "
              f"{name(mujoco.mjtObj.mjOBJ_GEOM, g2)} ({GEOM_TYPES[int(m.geom_type[g2])]}), body "
              f"{name(mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[other]))}: dist {1000 * c.dist:.1f} mm "
              f"(mj_geomDistance {true}), normal force {force[0]:.1f} N, normal {np.round(c.frame[:3], 3)}, "
              f"pos {np.round(c.pos, 3)}, geom sizes {np.round(m.geom_size[g1], 3)} / {np.round(m.geom_size[g2], 3)}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
