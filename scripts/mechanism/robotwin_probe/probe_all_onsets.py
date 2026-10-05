"""Bit-exact replay of one segment; at every physics-invalid onset dump the contacts between the task's
free objects and containers (per collision shape pair) for the step before and the onset step, with the
world vertices of every convex piece at the pose the onset step starts from.
usage: probe_all_onsets.py --segment DIR --out ABS_FILE.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
W = "/mnt/nvme0/shared/USER/simuguard"
sys.path.insert(0, W + "/SimuGuard"); sys.path.insert(0, W + "/SimuGuard/scripts/robotwin")

def quat_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--segment", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--robotwin-root", default=W + "/RoboTwin"); a = ap.parse_args()
    seg = Path(a.segment)
    onsets = None
    for name in ("events_carrier.json", "events_gravity.json"):
        p = seg / name
        if p.is_file():
            ev = [e for e in json.loads(p.read_text()).get("events", []) if e.get("verdict") == "physics_invalid"]
            onsets = sorted({int(e["onset_substep"]) for e in ev}); events = ev; break
    if not onsets: print("no physics-invalid onset"); return
    import sapien, sapien.physx as px
    from simuguard.adapters.robotwin import RoboTwinAdapter, make_task_env
    from simuguard.core import ControlLog
    from simuguard.core.types import BodyRole
    from simuguard.integrations.robotwin_eval import reapply_recorded_intervention
    summary = json.loads((seg / "summary.json").read_text()); manifest = json.loads((seg / "manifest.json").read_text())
    meta = {**manifest["metadata"], **(summary.get("metadata") or {})}
    controls = ControlLog.load(seg / "controls.npz")
    env, _ = make_task_env(a.robotwin_root, meta["task"], int(meta["seed"]))
    adapter = RoboTwinAdapter(env); reapply_recorded_intervention(adapter, meta); scene = adapter.scene
    adapter.bodies()
    ids = adapter.body_ids_with_role(BodyRole.TARGET) + adapter.body_ids_with_role(BodyRole.CONTAINER)
    comps = {bid: adapter._bodies[bid].component for bid in ids}
    names = {bid: comps[bid].entity.get_name() for bid in ids}
    by_entity = {id(comps[bid].entity): bid for bid in ids}
    shapes = {bid: comps[bid].get_collision_shapes() for bid in ids}
    def key(s): return (np.asarray(s.get_vertices()).tobytes(), np.asarray(s.get_local_pose().p).tobytes(), np.asarray(s.get_local_pose().q).tobytes())
    smap = {bid: {key(s): i for i, s in enumerate(ss) if hasattr(s, "get_vertices")} for bid, ss in shapes.items()}
    def vertices():
        out = {}
        for bid in ids:
            T = comps[bid].entity.get_pose(); rows = []
            for s in shapes[bid]:
                if not hasattr(s, "get_vertices"): rows.append(None); continue
                P = T * s.get_local_pose(); V = np.asarray(s.get_vertices()) * np.asarray(s.get_scale())
                rows.append(((quat_R(P.q) @ V.T).T + np.asarray(P.p)).tolist())
            out[bid] = rows
        return out
    def contacts():
        rows = []
        for c in scene.get_contacts() or []:
            b = [by_entity.get(id(x.entity)) for x in c.bodies]
            if b[0] is None or b[1] is None or not c.points: continue
            sid = [smap[b[i]].get(key(c.shapes[i]), -1) if hasattr(c.shapes[i], "get_vertices") else -1 for i in (0, 1)]
            rows.append({"bodies": b, "shapes": sid, "points": [{"p": [float(x) for x in p.position], "n": [float(x) for x in p.normal], "sep": float(p.separation)} for p in c.points]})
        return rows
    def state():
        return {bid: {"p": [float(x) for x in comps[bid].entity.get_pose().p], "v": ([float(x) for x in comps[bid].get_linear_velocity()] if isinstance(comps[bid], px.PhysxRigidDynamicComponent) else None)} for bid in ids}
    want = set(onsets) | {o - 1 for o in onsets}; last = max(onsets)
    out = {"segment": str(seg), "task": meta["task"], "seed": meta["seed"], "onsets": onsets, "names": names, "pieces": {b: len(s) for b, s in shapes.items()},
           "events": [{k: e.get(k) for k in ("onset_substep", "body_id", "peak_speed", "peak_speed_mps", "verdict")} for e in events], "steps": {}}
    for record in controls.between(0, controls.newest_substep):
        s = record.substep
        if s == 0: continue
        adapter.apply_control(record); scene.step()
        if s in want:
            row = {"contacts": contacts(), "state": state()}
            if s + 1 in onsets: row["vertices"] = vertices()   # pose the onset step starts from
            out["steps"][str(s)] = row
        if s >= last: break
    Path(a.out).write_text(json.dumps(out)); print("wrote", a.out, "onsets", len(onsets), flush=True)
    env.close_env()

if __name__ == "__main__":
    main()
