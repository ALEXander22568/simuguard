"""Generalisation of export_pair.py to any two bodies and any convex piece pair, several pairs per replay.
Bit-exact replay of one segment up to the last requested onset; for every case of that segment (bowls_cases.json
entries: onset, b0, s0, b1, s1 = body ids of the adapter and indices into get_collision_shapes()) writes a directory
that build/repro reads unchanged: hull_can.txt = piece of the FIRST shape of the reported contact (shape 0),
hull_basket.txt = piece of the second, poses.txt = actor poses of the two bodies before every step,
contacts.txt = what SAPIEN reported for the pair (either order; the name of the first body is written).
usage: export_pair_generic.py --cases FILE --segment DIR --out-root ABS_DIR
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
W = "/mnt/nvme0/shared/USER/simuguard"
sys.path.insert(0, W + "/SimuGuard"); sys.path.insert(0, W + "/SimuGuard/scripts/robotwin")
f9 = lambda a: " ".join("%.9g" % float(x) for x in np.asarray(a).reshape(-1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True); ap.add_argument("--segment", required=True); ap.add_argument("--out-root", required=True)
    ap.add_argument("--after", type=int, default=2); ap.add_argument("--robotwin-root", default=W + "/RoboTwin")
    a = ap.parse_args()
    cases = [c for c in json.load(open(a.cases)) if c["segment"] == a.segment]
    assert cases, "no case for this segment"
    root = Path(a.out_root).resolve()
    import sapien, sapien.physx as px
    from simuguard.adapters.robotwin import RoboTwinAdapter, make_task_env
    from simuguard.core import ControlLog
    from simuguard.core.types import BodyRole
    from simuguard.integrations.robotwin_eval import reapply_recorded_intervention
    seg = Path(a.segment)
    summary = json.loads((seg / "summary.json").read_text()); manifest = json.loads((seg / "manifest.json").read_text())
    meta = {**manifest["metadata"], **(summary.get("metadata") or {})}
    controls = ControlLog.load(seg / "controls.npz")
    env, _ = make_task_env(a.robotwin_root, meta["task"], int(meta["seed"]))
    adapter = RoboTwinAdapter(env); reapply_recorded_intervention(adapter, meta); scene = adapter.scene
    adapter.bodies()
    ids = adapter.body_ids_with_role(BodyRole.TARGET) + adapter.body_ids_with_role(BodyRole.CONTAINER)
    comps = {b: adapter._bodies[b].component for b in ids}
    by_entity = {id(comps[b].entity): b for b in ids}
    shapes = {b: comps[b].get_collision_shapes() for b in ids}
    key = lambda s: (np.asarray(s.get_vertices()).tobytes(), np.asarray(s.get_local_pose().p).tobytes(), np.asarray(s.get_local_pose().q).tobytes())
    smap = {b: {key(s): i for i, s in enumerate(ss) if hasattr(s, "get_vertices")} for b, ss in shapes.items()}
    files = []
    for c in cases:
        out = root / c["tag"]; out.mkdir(parents=True, exist_ok=True)
        info = {**c, "timestep": float(scene.get_timestep())}
        for name, b, si in (("can", c["b0"], c["s0"]), ("basket", c["b1"], c["s1"])):
            comp = comps[b]; s = shapes[b][si]; V = np.asarray(s.get_vertices()); lp = s.get_local_pose()
            dyn = isinstance(comp, px.PhysxRigidDynamicComponent)
            info[name] = {"body": b, "piece": si, "dynamic": dyn, "n_vertices": int(len(V)), "scale": [float(x) for x in s.get_scale()],
                          "contact_offset": float(s.get_contact_offset()), "rest_offset": float(s.get_rest_offset()), "n_shapes": len(shapes[b]),
                          "mass": float(comp.get_mass()) if dyn else None}
            with open(out / f"hull_{name}.txt", "w") as fh:
                fh.write("%d\n" % len(V))
                for v in V: fh.write(f9(v) + "\n")
                fh.write("scale " + f9(s.get_scale()) + "\n"); fh.write("local_p " + f9(lp.p) + "\nlocal_q_wxyz " + f9(lp.q) + "\n")
        (out / "info.json").write_text(json.dumps(info, indent=1))
        files.append((c, open(out / "poses.txt", "w"), open(out / "contacts.txt", "w")))
    last = max(c["onset"] for c in cases) + a.after
    def write_pose(step):
        for c, poses, _ in files:
            if step > c["onset"] + a.after + 1: continue
            row = [str(step)]
            for b in (c["b0"], c["b1"]):
                P = comps[b].entity.get_pose(); row += [f9(P.p), f9(P.q)]
            poses.write(" ".join(row) + "\n")
    for record in controls.between(0, controls.newest_substep):
        s = record.substep
        if s == 0: continue
        write_pose(s)
        adapter.apply_control(record); scene.step()
        want = [f for f in files if abs(s - f[0]["onset"]) <= 400]
        if want:
            for ct in scene.get_contacts() or []:
                b = [by_entity.get(id(x.entity)) for x in ct.bodies]
                if b[0] is None or b[1] is None or not ct.points: continue
                sid = [smap[b[i]].get(key(ct.shapes[i]), -1) if hasattr(ct.shapes[i], "get_vertices") else -1 for i in (0, 1)]
                for c, _, contacts in want:
                    if (b[0], sid[0], b[1], sid[1]) == (c["b0"], c["s0"], c["b1"], c["s1"]) or (b[1], sid[1], b[0], sid[0]) == (c["b0"], c["s0"], c["b1"], c["s1"]):
                        contacts.write("%d %s %d n %s sep %s pos %s\n" % (s, b[0].replace(" ", "_"), len(ct.points), f9(ct.points[0].normal),
                                       f9([p.separation for p in ct.points]), f9([p.position for p in ct.points])))
        if s >= last: break
    write_pose(last + 1)
    for _, p, q in files: p.close(); q.close()
    print("wrote", len(cases), "cases under", root, "steps", last, flush=True)
    env.close_env()

if __name__ == "__main__":
    main()
