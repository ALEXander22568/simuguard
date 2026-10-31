#!/usr/bin/env python3
"""Exact replay of a recorded LIBERO episode in a fresh process.

Rebuilds the episode the way it was recorded (``make_env``: suite, task, init state, seed), checks
the rebuilt model against the recorded fingerprint (LIBERO places fixtures on reset from the global
numpy generator, so a wrong seed shows up here; on robosuite >= 1.5 placement comes from an
unseeded per-env generator, and the recorded body poses are written back), checks the fresh state
against the recorded initial snapshot component by component, restores that snapshot, replays the
recorded actuation of the whole episode and compares

  * every logged body (targets, containers, other free objects) at every substep with states.npz;
  * the complete MuJoCo integration state (robot included) with every snapshot archived in
    snapshots.npz (one every 250 substeps).

``--drop-warmstart`` replays without the recorded solver warm starts, to measure what they are for.
Writes ``replay_check[_nowarm].json`` into the segment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simuguard.adapters.mujoco.libero import attach_libero, make_env, model_fingerprint, restore_placement  # noqa: E402
from simuguard.core.controls import ControlLog  # noqa: E402
from simuguard.core.snapshot import ControlRecord, Snapshot  # noqa: E402
from simuguard.core.statelog import StateLog  # noqa: E402

COMPONENTS = ("TIME", "QPOS", "QVEL", "ACT", "WARMSTART", "CTRL", "QFRC_APPLIED", "XFRC_APPLIED", "EQ_ACTIVE",
              "MOCAP_POS", "MOCAP_QUAT", "USERDATA", "PLUGIN")  # mjSTATE_INTEGRATION, in bit order


def split_state(model: mujoco.MjModel, state: np.ndarray) -> dict[str, np.ndarray]:
    out, i = {}, 0
    for name in COMPONENTS:
        n = mujoco.mj_stateSize(model, getattr(mujoco.mjtState, f"mjSTATE_{name}"))
        out[name] = state[i:i + n]
        i += n
    assert i == len(state), (i, len(state))
    return out


def load_snapshots(path: Path) -> dict[int, np.ndarray]:
    if not path.is_file():
        return {}
    with np.load(path) as npz:
        return {int(s): npz["states"][i, : npz["lengths"][i]].view(np.float64).copy() for i, s in enumerate(npz["substeps"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("segment")
    ap.add_argument("--drop-warmstart", action="store_true")
    args = ap.parse_args()
    seg = Path(args.segment)
    manifest = json.loads((seg / "manifest.json").read_text())
    meta = manifest["metadata"]
    recorded_fp = manifest["task_ground_truth"]["model_fingerprint"]
    controls = ControlLog.load(seg / "controls.npz")
    initial = Snapshot.load(seg / "initial_snapshot.json.gz")
    ref = StateLog.load(seg / "states.npz")
    archive = load_snapshots(seg / "snapshots.npz")
    result: dict = {"segment": str(seg), "drop_warmstart": args.drop_warmstart}

    t0 = time.time()
    env, _, _ = make_env(meta["suite"], meta["task_id"], meta["init_index"], meta["seed"], camera_size=meta["camera_size"])
    adapter, info = attach_libero(env, task_name=meta["task"])
    result["rebuild_s"] = round(time.time() - t0, 2)
    result["stack"] = {k: info[k] for k in ("robosuite", "mujoco", "lite_physics", "hooked_method")}
    fp = model_fingerprint(adapter.model)
    result["model_identical"] = fp["sha256"] == recorded_fp["sha256"]
    if not result["model_identical"]:  # e.g. robosuite >= 1.5: placement not reproduced by the seed
        result["placement_restored_max_change"] = restore_placement(adapter.model, recorded_fp)
        result["model_identical_after_restore"] = model_fingerprint(adapter.model)["sha256"] == recorded_fp["sha256"]

    fresh = split_state(adapter.model, adapter.native_state())
    recorded = split_state(adapter.model, np.frombuffer(initial.native_state, dtype=np.float64))
    result["initial_state_max_abs_diff"] = {k: float(np.abs(fresh[k] - recorded[k]).max()) for k in COMPONENTS if len(fresh[k])}

    adapter.restore_snapshot(initial, method="native")
    ids, arr = ref.body_ids, ref.array()
    row = {int(s): i for i, s in enumerate(ref.substeps)}
    last = int(controls.newest_substep)
    worst_pos, worst_body, first, snap_errs = 0.0, None, None, {}
    t0 = time.time()
    for record in controls.between(0, last):
        if args.drop_warmstart and "qacc_warmstart" in record.payload:
            record = ControlRecord(record.substep, record.control_step,
                                   {k: v for k, v in record.payload.items() if k != "qacc_warmstart"})
        adapter.apply_control(record)
        adapter.step_physics()
        s = record.substep
        if s in row:
            now = adapter.read_states(ids)
            for j, b in enumerate(ids):
                err = float(np.abs(now[b].position - arr[row[s], j, :3]).max())
                if err > worst_pos:
                    worst_pos, worst_body = err, b
                if err > 0 and first is None:
                    first = {"substep": s, "control_step": record.control_step, "body": b, "error_m": err}
        if s in archive:
            snap_errs[s] = float(np.abs(adapter.native_state() - archive[s]).max())
    result.update({
        "replay_s": round(time.time() - t0, 2), "substeps_replayed": last, "bodies_compared": len(ids),
        "max_abs_pos_err_m": worst_pos, "worst_body": worst_body, "first_divergence": first,
        "records_with_warmstart": sum("qacc_warmstart" in r.payload for r in controls.records()),
        "full_state_snapshots_compared": len(snap_errs),
        "full_state_max_abs_diff": max(snap_errs.values()) if snap_errs else None,
        "full_state_first_nonzero_substep": next((s for s in sorted(snap_errs) if snap_errs[s] > 0), None),
    })
    out = seg / ("replay_check_nowarm.json" if args.drop_warmstart else "replay_check.json")
    out.write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result, indent=1, default=str))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
