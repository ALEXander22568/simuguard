"""Open loop replay of one recorded Place Can Basket segment with whatever SAPIEN is first on PYTHONPATH.
Reports where the replay first leaves the recording, what happens around the recorded artifact onset, and the
whole episode's peak can speed, deep can-basket contacts and success.
usage: replay_check.py --segment DIR --onset N --out ABS.json"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
W = "/mnt/nvme0/shared/USER/simuguard"
sys.path.insert(0, W + "/SimuGuard"); sys.path.insert(0, W + "/SimuGuard/scripts/robotwin")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--segment", required=True); ap.add_argument("--onset", type=int, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--robotwin-root", default=W + "/RoboTwin"); a = ap.parse_args()
    import sapien, sapien.physx as px
    lib = sorted({l.split()[-1] for l in open("/proc/self/maps") if "libsapien.so" in l})
    from simuguard.adapters.robotwin import RoboTwinAdapter, make_task_env
    from simuguard.core import ControlLog, StateLog
    from simuguard.integrations.robotwin_eval import reapply_recorded_intervention
    seg = Path(a.segment)
    summary = json.loads((seg / "summary.json").read_text()); manifest = json.loads((seg / "manifest.json").read_text())
    meta = {**manifest["metadata"], **(summary.get("metadata") or {})}
    controls = ControlLog.load(seg / "controls.npz"); rec = StateLog.load(seg / "states.npz"); arr = rec.array(); row_of = {int(s): i for i, s in enumerate(rec.substeps)}
    env, _ = make_task_env(a.robotwin_root, meta["task"], int(meta["seed"]))
    adapter = RoboTwinAdapter(env); reapply_recorded_intervention(adapter, meta); scene = adapter.scene
    info = adapter.bodies(); ents = {}
    for e in scene.entities:
        n = e.get_name()
        if "071" in n: ents["can"] = e
        if "110" in n: ents["basket"] = e
    col = None
    for i, bid in enumerate(rec.body_ids):
        if "071" in str(bid) or "071" in str(getattr(info.get(bid), "name", "")): col = i
    assert col is not None, rec.body_ids
    def comp(e):
        for c in e.get_components():
            if isinstance(c, (px.PhysxRigidDynamicComponent, px.PhysxRigidStaticComponent)): return c
    can = comp(ents["can"])
    out = {"segment": str(seg), "onset": a.onset, "sapien": sapien.__file__, "libsapien": lib, "recorded_success": (meta.get("outcome") or {}).get("eval_success")}
    first_div = None; peak = (0.0, None); rec_peak = (0.0, None); deep_steps = []; win = {}; first_success = None; t0 = time.time(); n = 0; max_err = 0.0
    for record in controls.between(0, controls.newest_substep):
        s = record.substep
        if s == 0: continue
        adapter.apply_control(record); scene.step(); n += 1
        v = float(np.linalg.norm(can.get_linear_velocity()))
        r = arr[row_of[s]][col]; err = float(np.abs(np.array(ents["can"].get_pose().p) - r[0:3]).max()); max_err = max(max_err, err)
        rv = float(np.linalg.norm(r[7:10]))
        if rv > rec_peak[0]: rec_peak = (rv, s)
        if first_div is None and err > 1e-6: first_div = s
        if v > peak[0]: peak = (v, s)
        deep = 0.0; npairs = 0
        for c in scene.get_contacts() or []:
            names = [b.entity.get_name() for b in c.bodies]
            if any("071" in x for x in names) and any("110" in x for x in names) and c.points:
                npairs += 1; deep = min(deep, min(float(p.separation) for p in c.points))
        if deep < -0.005: deep_steps.append((s, round(deep * 1e3, 2)))
        if a.onset - 3 <= s <= a.onset + 25: win[str(s)] = {"can_speed": v, "recorded_can_speed": rv, "min_sep_mm": deep * 1e3, "pairs": npairs, "pos_err_m": err}
        if first_success is None and s % 25 == 0 and env.check_success(): first_success = s
    out.update({"substeps": n, "wall_s": round(time.time() - t0, 1), "first_step_pos_err_gt_1e-6": first_div, "max_pos_err_m": max_err,
                "onset_min_sep_mm": win[str(a.onset)]["min_sep_mm"], "onset_can_speed": win[str(a.onset)]["can_speed"], "onset_recorded_can_speed": win[str(a.onset)]["recorded_can_speed"],
                "window_peak_can_speed": max(w["can_speed"] for w in win.values()), "window_deepest_mm": min(w["min_sep_mm"] for w in win.values()),
                "episode_peak_can_speed": peak[0], "episode_peak_step": peak[1], "recorded_peak_can_speed": rec_peak[0], "recorded_peak_step": rec_peak[1],
                "steps_with_can_basket_sep_below_-5mm": len(deep_steps), "deep_steps_first10": deep_steps[:10],
                "first_success_substep_checked_every_25": first_success, "success_at_end": bool(env.check_success()), "window": win})
    Path(a.out).write_text(json.dumps(out))
    print(json.dumps({k: v for k, v in out.items() if k != "window"}), flush=True)
    env.close_env()

if __name__ == "__main__":
    main()
