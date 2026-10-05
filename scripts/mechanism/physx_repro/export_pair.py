"""Bit-exact replay of one segment up to its onset; exports what a standalone PhysX program needs to feed
one can piece / basket piece pair through contact generation: hull vertices, scale, shape local pose,
centre of mass frame, the actor pose before every step, and the contacts SAPIEN reported for the pair.
usage: export_pair.py --segment DIR --onset N --can-piece I --basket-piece J --out DIR
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
W = "/mnt/nvme0/shared/USER/simuguard"
sys.path.insert(0, W + "/SimuGuard"); sys.path.insert(0, W + "/SimuGuard/scripts/robotwin")
f9 = lambda a: " ".join("%.9g" % float(x) for x in np.asarray(a).reshape(-1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segment", required=True); ap.add_argument("--onset", type=int, required=True)
    ap.add_argument("--can-piece", type=int, required=True); ap.add_argument("--basket-piece", type=int, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--after", type=int, default=2)
    ap.add_argument("--robotwin-root", default=W + "/RoboTwin")
    a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    import sapien, sapien.physx as px
    from simuguard.adapters.robotwin import RoboTwinAdapter, make_task_env
    from simuguard.core import ControlLog
    from simuguard.integrations.robotwin_eval import reapply_recorded_intervention
    seg = Path(a.segment)
    summary = json.loads((seg / "summary.json").read_text()); manifest = json.loads((seg / "manifest.json").read_text())
    meta = {**manifest["metadata"], **(summary.get("metadata") or {})}
    controls = ControlLog.load(seg / "controls.npz")
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
    pick = {"can": a.can_piece, "basket": a.basket_piece}
    shapes = {k: comp(e).get_collision_shapes() for k, e in ents.items()}
    vmap = {}
    for key, ss in shapes.items():
        for i, t in enumerate(ss): vmap[(np.asarray(t.get_vertices()).tobytes(), np.asarray(t.get_local_pose().p).tobytes())] = (key, i)
    sid = lambda s: vmap.get((np.asarray(s.get_vertices()).tobytes(), np.asarray(s.get_local_pose().p).tobytes()), ("?", -1))
    info = {"segment": str(seg), "onset": a.onset, "pieces": pick, "timestep": float(scene.get_timestep())}
    for key in ("can", "basket"):
        c = comp(ents[key]); s = shapes[key][pick[key]]
        V = np.asarray(s.get_vertices()); lp = s.get_local_pose()
        dyn = isinstance(c, px.PhysxRigidDynamicComponent)
        cm = c.get_cmass_local_pose() if dyn else sapien.Pose()
        info[key] = {"dynamic": dyn, "n_vertices": int(len(V)), "vertex_dtype": str(V.dtype), "scale": [float(x) for x in s.get_scale()],
                     "contact_offset": float(s.get_contact_offset()), "rest_offset": float(s.get_rest_offset()),
                     "n_shapes": len(shapes[key]), "mass": float(c.get_mass()) if dyn else None,
                     "shape_attrs": [n for n in dir(s) if not n.startswith("_")][:80]}
        with open(out / f"hull_{key}.txt", "w") as fh:
            fh.write("%d\n" % len(V))
            for v in V: fh.write(f9(v) + "\n")
            fh.write("scale " + f9(s.get_scale()) + "\n")
            fh.write("local_p " + f9(lp.p) + "\nlocal_q_wxyz " + f9(lp.q) + "\n")
            fh.write("cmass_p " + f9(cm.p) + "\ncmass_q_wxyz " + f9(cm.q) + "\n")
        try:
            T = np.asarray(s.get_triangles()); np.save(out / f"tri_{key}.npy", T); info[key]["n_triangles"] = int(len(T))
        except Exception as ex:  # noqa: BLE001
            info[key]["n_triangles"] = "n/a: %s" % ex
    poses = open(out / "poses.txt", "w"); contacts = open(out / "contacts.txt", "w")
    def write_pose(step):   # poses the narrowphase of `step` sees = end of step-1
        row = [str(step)]
        for key in ("can", "basket"):
            P = ents[key].get_pose(); row += [f9(P.p), f9(P.q)]
        poses.write(" ".join(row) + "\n")
    last = a.onset + a.after
    for record in controls.between(0, controls.newest_substep):
        s = record.substep
        if s == 0: continue
        write_pose(s)
        adapter.apply_control(record); scene.step()
        for c in scene.get_contacts() or []:
            names = [b.entity.get_name() for b in c.bodies]
            if not (any("071" in n for n in names) and any("110" in n for n in names)) or not c.points: continue
            ids = dict(sid(sh) for sh in c.shapes)
            if ids.get("can") == pick["can"] and ids.get("basket") == pick["basket"]:
                contacts.write("%d %s %d n %s sep %s pos %s\n" % (s, names[0], len(c.points), f9(c.points[0].normal),
                               f9([p.separation for p in c.points]), f9([p.position for p in c.points])))
        if s >= last: break
    write_pose(last + 1)
    poses.close(); contacts.close()
    v = comp(ents["can"]).get_linear_velocity()
    info["can_speed_end"] = float(np.linalg.norm(v))
    (out / "info.json").write_text(json.dumps(info, indent=1))
    print("wrote", out, "steps", last, "can speed at end", info["can_speed_end"], flush=True)
    env.close_env()

if __name__ == "__main__":
    main()
