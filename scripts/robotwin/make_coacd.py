"""Pre-decompose the basket collision meshes with CoACD at a coarser threshold and write them as GLB scenes
(one mesh per convex piece), so RoboTwin's loader treats each piece as one convex hull, as it does for the shipped file."""
import sys, glob, os, numpy as np, trimesh, coacd, logging
coacd.set_log_level("error")
root = sys.argv[1]; th = float(sys.argv[2])
for f in sorted(glob.glob(os.path.join(root, "assets/objects/110_basket/collision/base?.glb"))):
    out = f.replace(".glb", f"_coacd{th}.glb")
    if os.path.exists(out): print("exists", out); continue
    sc = trimesh.load(f, force="scene"); geoms = list(sc.geometry.values())
    m = trimesh.util.concatenate(geoms)
    parts = coacd.run_coacd(coacd.Mesh(m.vertices, m.faces), threshold=th, seed=0)
    new = trimesh.Scene()
    for i, (v, fc) in enumerate(parts):
        new.add_geometry(trimesh.Trimesh(np.asarray(v), np.asarray(fc), process=False), node_name=f"piece{i}", geom_name=f"piece{i}")
    new.export(out)
    chk = trimesh.load(out, force="scene")
    print(os.path.basename(f), "shipped pieces", len(geoms), "-> coacd", len(parts), "reloaded", len(chk.geometry))
