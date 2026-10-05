"""Bit-exact replay of one recorded segment; around the first artifact onset dump every can-basket
contact per collision-shape pair (points, normals, separations) and the world-space vertices of every
convex piece, so the reported overlap can be compared with the geometry.
usage: probe_onset.py --segment DIR --onset N [--window 3] [--pcm-off] --out FILE.json
"""
import argparse, json, sys, time, gzip
from pathlib import Path
import numpy as np
sys.path.insert(0, "/mnt/nvme0/shared/USER/simuguard/SimuGuard")
sys.path.insert(0, "/mnt/nvme0/shared/USER/simuguard/SimuGuard/scripts/robotwin")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", required=True); ap.add_argument("--onset", type=int, required=True)
    ap.add_argument("--window", type=int, default=3); ap.add_argument("--pcm-off", action="store_true")
    ap.add_argument("--out", required=True); ap.add_argument("--full", action="store_true"); ap.add_argument("--vertices-every-step", action="store_true")
    ap.add_argument("--robotwin-root", default="/mnt/nvme0/shared/USER/simuguard/RoboTwin")
    a = ap.parse_args()
    import sapien, sapien.physx as px
    if a.pcm_off:
        orig = sapien.SceneConfig
        def patched(*args, **kw):
            c = orig(*args, **kw); c.enable_pcm = False; return c
        sapien.SceneConfig = patched
    from simuguard.adapters.robotwin import RoboTwinAdapter, make_task_env
    from simuguard.core import ControlLog, StateLog
    from simuguard.integrations.robotwin_eval import reapply_recorded_intervention
    seg = Path(a.segment)
    summary = json.loads((seg / "summary.json").read_text()); manifest = json.loads((seg / "manifest.json").read_text())
    meta = {**manifest["metadata"], **(summary.get("metadata") or {})}
    controls = ControlLog.load(seg / "controls.npz"); rec = StateLog.load(seg / "states.npz")
    env, _ = make_task_env(a.robotwin_root, meta["task"], int(meta["seed"]))
    adapter = RoboTwinAdapter(env); reapply_recorded_intervention(adapter, meta)
    # entities of interest
    ents = {}
    for e in adapter.scene.get_all_actors() if hasattr(adapter.scene, "get_all_actors") else adapter.scene.entities:
        n = e.get_name()
        if "071" in n or "can" in n.lower(): ents["can"] = e
        if "110" in n or "basket" in n.lower(): ents["basket"] = e
    print("entities:", {k: v.get_name() for k, v in ents.items()}, flush=True)
    def body_comp(e):
        for c in e.get_components():
            if isinstance(c, (px.PhysxRigidDynamicComponent, px.PhysxRigidStaticComponent)): return c
    shape_index = {}
    pieces = {}
    for key, e in ents.items():
        comp = body_comp(e); shapes = comp.get_collision_shapes()
        pieces[key] = shapes
        for i, s in enumerate(shapes): shape_index[(key, id(s))] = i
    vmap = {}
    for key, shapes in pieces.items():
        for i, t in enumerate(shapes):
            vmap[(np.asarray(t.get_vertices()).tobytes(), np.asarray(t.get_local_pose().p).tobytes())] = (key, i)
    def shape_id(s):
        return vmap.get((np.asarray(s.get_vertices()).tobytes(), np.asarray(s.get_local_pose().p).tobytes()), ("?", -1))
    def world_vertices(key):
        e = ents[key]; comp = body_comp(e); T = comp.get_pose() if hasattr(comp, "get_pose") else e.get_pose()
        out = []
        for i, s in enumerate(pieces[key]):
            lp = s.get_local_pose(); P = T * lp
            R = _quat_to_R(P.q); V = np.asarray(s.get_vertices()) * np.asarray(s.get_scale())
            out.append((R @ V.T).T + np.asarray(P.p))
        return [v.tolist() for v in out]
    def dump_contacts():
        rows = []
        for c in adapter.scene.get_contacts() or []:
            names = [b.entity.get_name() if hasattr(b, "entity") else str(b) for b in c.bodies]
            if not (any("071" in n for n in names) and any("110" in n for n in names)): continue
            sids = [shape_id(s) for s in c.shapes]
            pts = [{"p": [float(x) for x in p.position], "n": [float(x) for x in p.normal], "sep": float(p.separation), "imp": [float(x) for x in p.impulse]} for p in c.points]
            rows.append({"bodies": names, "shapes": sids, "points": pts})
        return rows
    out = {"segment": str(seg), "onset": a.onset, "pcm_off": a.pcm_off, "steps": {}}
    live = rec.array(); rec_ids = list(rec.body_ids)
    t0 = time.time(); n = 0
    lo, hi = a.onset - a.window, a.onset + a.window
    for record in controls.between(0, controls.newest_substep):
        if record.substep == 0: continue
        adapter.apply_control(record); adapter.scene.step(); n += 1
        s = record.substep
        if lo <= s <= hi:
            st = {}
            for key, e in ents.items():
                comp = body_comp(e); P = comp.get_pose(); 
                st[key] = {"p": [float(x) for x in P.p], "q": [float(x) for x in P.q], "v": [float(x) for x in comp.get_linear_velocity()]}
            row = {"state": st, "contacts": dump_contacts()}
            if s in (a.onset - 1, a.onset) or a.vertices_every_step: row["vertices"] = {k: world_vertices(k) for k in ents}
            # compare with the recording
            k = np.searchsorted(rec.substeps, s) if hasattr(rec, "substeps") else None
            out["steps"][str(s)] = row
            cv=float(np.linalg.norm(st["can"]["v"])); mp=max([-(p["sep"]) for c in row["contacts"] for p in c["points"]] + [0])*1000
            print("substep %d: can v=%.3f contacts=%d maxpen=%.1fmm" % (s, cv, len(row["contacts"]), mp), flush=True)
        if a.full:
            comp = body_comp(ents["can"]); v = float(np.linalg.norm(comp.get_linear_velocity()))
            mp = 0.0
            for c in adapter.scene.get_contacts() or []:
                names = [b.entity.get_name() if hasattr(b, "entity") else str(b) for b in c.bodies]
                if any("071" in n for n in names) and any("110" in n for n in names):
                    for p in c.points: mp = max(mp, -float(p.separation))
            if v > 1.0 or mp > 0.005:
                out.setdefault("flags", []).append({"substep": s, "can_speed": v, "maxpen_mm": mp * 1000})
                print("FLAG substep %d can_speed %.2f maxpen %.1f mm" % (s, v, mp * 1000), flush=True)
            if s % 5000 == 0: print("...", s, "success", env.check_success(), flush=True)
            continue
        if s >= hi: break
    out["replayed_substeps"] = n; out["wall_s"] = time.time() - t0; out["success"] = bool(env.check_success())
    # bit-exactness of the prefix (can+basket states) up to onset-1
    try:
        ids = [rec_ids.index(b) for b in rec_ids]
        out["note"] = "prefix check not computed here (monitor not attached); use attribute_failures baseline for that"
    except Exception: pass
    Path(a.out).write_text(json.dumps(out))
    print("wrote", a.out, "wall", round(out["wall_s"], 1), flush=True)
    env.close_env()

def _quat_to_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],[2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],[2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

if __name__ == "__main__":
    main()
