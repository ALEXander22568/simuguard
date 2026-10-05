"""Stack Bowls Three: one case per onset whose onset step holds an object-object contact deeper than 5 mm
(the deepest one), from probe/mech/tasks/stack_bowls_three__*.json.  Writes cases.json."""
import json, glob, os, numpy as np
W = "/mnt/nvme0/shared/USER/simuguard"
R = W + "/runs/crosstask4_50seeds_copy/stack_bowls_three/simuguard/segments"
cases = []
for fn in sorted(glob.glob(W + "/probe/mech/tasks/stack_bowls_three__*.json")):
    d = json.load(open(fn)); seg = os.path.basename(fn).split("__")[1][:-5]
    for o in d["onsets"]:
        cur = d["steps"].get(str(o)); prev = d["steps"].get(str(o - 1))
        if not cur or not prev or "vertices" not in prev: continue
        deep = [(min(p["sep"] for p in c["points"]), c) for c in cur["contacts"]]
        deep = [x for x in deep if x[0] < -0.005]
        if not deep: continue
        sep, c = min(deep, key=lambda x: x[0])
        V = prev["vertices"]; A = np.asarray(V[c["bodies"][0]][c["shapes"][0]]); B = np.asarray(V[c["bodies"][1]][c["shapes"][1]])
        n = np.array(c["points"][0]["n"]); n /= np.linalg.norm(n)
        side = float((A @ n).mean() - (B @ n).mean())
        cases.append({"tag": "%s_o%d" % (seg.split("_seed")[0], o), "segment": R + "/" + seg, "onset": o, "b0": c["bodies"][0], "s0": c["shapes"][0],
                      "b1": c["bodies"][1], "s1": c["shapes"][1], "sep_mm": [round(p["sep"] * 1e3, 3) for p in c["points"]],
                      "normal": [float(x) for x in n], "centre_side": "normal" if side > 0 else "inverted"})
json.dump(cases, open(W + "/probe/physx_repro/bowls_cases.json", "w"), indent=1)
for c in cases: print(c["tag"], c["b0"], c["s0"], c["b1"], c["s1"], min(c["sep_mm"]), c["centre_side"])
print(len(cases), "cases in", len({c["segment"] for c in cases}), "segments")
