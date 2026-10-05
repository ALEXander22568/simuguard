"""Bit-exact replay up to the onset, with one value-preserving pose write on the can K steps before it
(set_pose(get_pose()), velocities rewritten unchanged).  Tells whether the onset contact needs narrowphase
state that a pose write clears, and how many steps that state takes to form.
usage: noop_reset.py --segment DIR --onset N --k K [--target can|basket|none] --out FILE.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
W = "/mnt/nvme0/shared/USER/simuguard"
sys.path.insert(0, W + "/SimuGuard"); sys.path.insert(0, W + "/SimuGuard/scripts/robotwin")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", required=True); ap.add_argument("--onset", type=int, required=True)
    ap.add_argument("--k", type=int, default=-1, help="pose write before step onset-k+1 runs, i.e. k steps of history remain; -1: none")
    ap.add_argument("--target", default="can"); ap.add_argument("--out", required=True)
    ap.add_argument("--after", type=int, default=25); ap.add_argument("--robotwin-root", default=W + "/RoboTwin")
    a = ap.parse_args()
    import sapien, sapien.physx as px
    from simuguard.adapters.robotwin import RoboTwinAdapter, make_task_env
    from simuguard.core import ControlLog, StateLog
    from simuguard.integrations.robotwin_eval import reapply_recorded_intervention
    seg = Path(a.segment)
    summary = json.loads((seg / "summary.json").read_text()); manifest = json.loads((seg / "manifest.json").read_text())
    meta = {**manifest["metadata"], **(summary.get("metadata") or {})}
    controls = ControlLog.load(seg / "controls.npz"); rec = StateLog.load(seg / "states.npz"); arr = rec.array(); row_of = {int(s): i for i, s in enumerate(rec.substeps)}
    env, _ = make_task_env(a.robotwin_root, meta["task"], int(meta["seed"]))
    adapter = RoboTwinAdapter(env); reapply_recorded_intervention(adapter, meta); scene = adapter.scene
    ents = {}
    for e in scene.entities:
        n = e.get_name()
        if "071" in n: ents["can"] = e
        if "110" in n: ents["basket"] = e
    def comp(e):
        for c in e.get_components():
            if isinstance(c, (px.PhysxRigidDynamicComponent, px.PhysxRigidStaticComponent)): return c
    can = comp(ents["can"])
    write_at = a.onset - a.k + 1 if a.k >= 0 else None      # before this step runs
    out = {"segment": str(seg), "onset": a.onset, "k": a.k, "target": a.target, "steps": {}}
    max_err_before = 0.0
    for record in controls.between(0, controls.newest_substep):
        s = record.substep
        if s == 0: continue
        if write_at is not None and s == write_at and a.target != "none":
            e = ents[a.target]; c = comp(e); p = e.get_pose(); e.set_pose(sapien.Pose(p.p, p.q))
            if isinstance(c, px.PhysxRigidDynamicComponent):
                v, w = np.array(c.get_linear_velocity()), np.array(c.get_angular_velocity()); c.set_linear_velocity(v); c.set_angular_velocity(w)
        adapter.apply_control(record); scene.step()
        if s < a.onset - 3:
            if s % 500 == 0 or s >= a.onset - 60:
                max_err_before = max(max_err_before, float(np.abs(np.array(ents["can"].get_pose().p) - arr[row_of[s]][0][0:3]).max()))
            continue
        deep = 0.0; npairs = 0
        for c in scene.get_contacts() or []:
            names = [b.entity.get_name() for b in c.bodies]
            if any("071" in n for n in names) and any("110" in n for n in names) and c.points:
                npairs += 1; deep = min(deep, min(float(p.separation) for p in c.points))
        err = float(np.abs(np.array(ents["can"].get_pose().p) - arr[row_of[s]][0][0:3]).max())
        out["steps"][str(s)] = {"can_speed": float(np.linalg.norm(can.get_linear_velocity())), "min_sep_mm": deep * 1e3, "pairs": npairs, "pos_err_vs_recording_m": err}
        if s >= a.onset + a.after: break
    out["max_pos_err_before_window_m"] = max_err_before
    win = out["steps"]; on = win[str(a.onset)]
    out["summary"] = {"k": a.k, "target": a.target, "onset_min_sep_mm": on["min_sep_mm"], "onset_can_speed": on["can_speed"],
                      "peak_speed_after": max(v["can_speed"] for v in win.values()), "deepest_in_window_mm": min(v["min_sep_mm"] for v in win.values()),
                      "pos_err_at_onset-1": win[str(a.onset - 1)]["pos_err_vs_recording_m"], "prefix_err": max_err_before}
    print(json.dumps(out["summary"]), flush=True)
    Path(a.out).write_text(json.dumps(out)); env.close_env()

if __name__ == "__main__":
    main()
