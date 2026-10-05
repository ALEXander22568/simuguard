"""Does the relative pose history of can and basket alone reproduce the onset contact?
The robot is not actuated; before each physics step the can and the basket are placed at their recorded
poses (end of the previous step), starting K steps before the onset, from a fresh pair (no cached manifold).
usage: pair_replay.py --segment DIR --onset N --out FILE.json [--ks 1 2 5 ...] [--pcm-off]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
W = "/mnt/nvme0/shared/USER/simuguard"
sys.path.insert(0, W + "/SimuGuard"); sys.path.insert(0, W + "/SimuGuard/scripts/robotwin")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", required=True); ap.add_argument("--onset", type=int, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5, 10, 25, 100, 500, 2000, 0])
    ap.add_argument("--pcm-off", action="store_true"); ap.add_argument("--robotwin-root", default=W + "/RoboTwin")
    a = ap.parse_args()
    import sapien, sapien.physx as px
    if a.pcm_off:
        orig = sapien.SceneConfig
        def patched(*args, **kw):
            c = orig(*args, **kw); c.enable_pcm = False; return c
        sapien.SceneConfig = patched
    from simuguard.adapters.robotwin import RoboTwinAdapter, make_task_env
    from simuguard.core import StateLog
    seg = Path(a.segment)
    summary = json.loads((seg / "summary.json").read_text()); manifest = json.loads((seg / "manifest.json").read_text())
    meta = {**manifest["metadata"], **(summary.get("metadata") or {})}
    rec = StateLog.load(seg / "states.npz"); arr = rec.array(); subs = rec.substeps; row_of = {int(s): i for i, s in enumerate(subs)}
    env, _ = make_task_env(a.robotwin_root, meta["task"], int(meta["seed"]))
    adapter = RoboTwinAdapter(env); scene = adapter.scene
    ents = {}
    for e in scene.entities:
        n = e.get_name()
        if "071" in n: ents["can"] = e
        if "110" in n: ents["basket"] = e
    def comp(e):
        for c in e.get_components():
            if isinstance(c, (px.PhysxRigidDynamicComponent, px.PhysxRigidStaticComponent)): return c
    ids = {}
    info = adapter.bodies()
    for bid in rec.body_ids:
        name = getattr(info.get(bid), "name", "") or bid
        for k in ("071", "110"):
            if k in str(name) or k in str(bid): ids["can" if k == "071" else "basket"] = rec.body_ids.index(bid)
    print("entities", {k: v.get_name() for k, v in ents.items()}, "state columns", ids, "recorded substeps", int(subs[0]), int(subs[-1]), flush=True)
    shapes = {k: comp(e).get_collision_shapes() for k, e in ents.items()}
    vmap = {}
    for key, ss in shapes.items():
        for i, t in enumerate(ss): vmap[(np.asarray(t.get_vertices()).tobytes(), np.asarray(t.get_local_pose().p).tobytes())] = (key, i)
    sid = lambda s: vmap.get((np.asarray(s.get_vertices()).tobytes(), np.asarray(s.get_local_pose().p).tobytes()), ("?", -1))
    def place(step):  # poses at the end of `step`
        r = arr[row_of[step]]
        for key in ("basket", "can"):
            if key not in ids: continue
            s = r[ids[key]]; c = comp(ents[key])
            ents[key].set_pose(sapien.Pose(s[0:3], s[3:7]))
            if isinstance(c, px.PhysxRigidDynamicComponent):
                c.set_linear_velocity(s[7:10]); c.set_angular_velocity(s[10:13])
    def contacts():
        rows = []
        for c in scene.get_contacts() or []:
            names = [b.entity.get_name() for b in c.bodies]
            if not (any("071" in n for n in names) and any("110" in n for n in names)): continue
            if not c.points: continue
            pair = tuple(sid(s) for s in c.shapes)
            rows.append({"pair": [list(p) for p in pair], "first": names[0], "sep": [float(p.separation) for p in c.points], "n": [float(x) for x in c.points[0].normal]})
        return rows
    def part():  # break the pair: any cached manifold is destroyed with the broadphase pair
        e = ents["can"]; p = e.get_pose(); e.set_pose(sapien.Pose([5.0, 5.0, 3.0], p.q))
        for _ in range(3):
            c = comp(e); c.set_linear_velocity([0, 0, 0]); c.set_angular_velocity([0, 0, 0]); e.set_pose(sapien.Pose([5.0, 5.0, 3.0], p.q)); scene.step()
    out = {"segment": str(seg), "onset": a.onset, "pcm_off": a.pcm_off, "runs": []}
    for K in a.ks:
        start = max(int(subs[0]) + 1, a.onset - K) if K > 0 else int(subs[0]) + 1
        part()
        first_deep = None; at_onset = None; npairs = 0
        for s in range(start, a.onset + 1):
            place(s - 1); scene.step()
            if s >= a.onset - 2 or first_deep is None:
                rows = contacts()
                deep = [(min(r["sep"]), r) for r in rows if min(r["sep"]) < -0.005]
                if deep and first_deep is None:
                    first_deep = {"substep": s, "sep_mm": min(d[0] for d in deep) * 1e3, "pair": min(deep, key=lambda d: d[0])[1]["pair"]}
                if s == a.onset:
                    npairs = len(rows)
                    m = min(((min(r["sep"]), r) for r in rows), key=lambda t: t[0], default=None)
                    at_onset = None if m is None else {"min_sep_mm": m[0] * 1e3, "pair": m[1]["pair"], "points": len(m[1]["sep"])}
        res = {"K": K, "start": start, "steps": a.onset - start + 1, "first_deep": first_deep, "at_onset": at_onset, "pairs_at_onset": npairs}
        out["runs"].append(res); print(json.dumps(res), flush=True)
    Path(a.out).write_text(json.dumps(out)); env.close_env()

if __name__ == "__main__":
    main()
