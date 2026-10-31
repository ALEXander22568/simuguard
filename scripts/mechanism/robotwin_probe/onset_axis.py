"""Along the reported contact normal, at the pose the onset step starts from: where do the two pieces sit,
and which plane distance equals the reported separation?"""
import json, glob, sys, numpy as np
rows = []
for fn in sorted(glob.glob(sys.argv[1] + "/rep*.json")):
    d = json.load(open(fn)); o = d["onset"]; S = d["steps"]
    if "vertices" not in S.get(str(o - 1), {}): continue
    cur, prev = S[str(o)], S[str(o - 1)]
    best = None
    for c in cur["contacts"]:
        for p in c["points"]:
            if best is None or p["sep"] < best[0]: best = (p["sep"], c)
    sep, c = best
    if sep > -0.005: continue
    idx = dict(map(tuple, c["shapes"]))
    C = np.asarray(prev["vertices"]["can"][idx["can"]]); B = np.asarray(prev["vertices"]["basket"][idx["basket"]])
    n = np.array(c["points"][0]["n"]); n /= np.linalg.norm(n)
    P = np.array([p["p"] for p in c["points"]]); seps = np.array([p["sep"] for p in c["points"]])
    cn, bn, pn = C @ n, B @ n, P @ n
    o0 = bn.min()
    rows.append(dict(ep=fn.split("/")[-1][5:-5], first=c["bodies"][0][:6], sep=sep * 1e3, nsame=float(np.ptp([np.dot(n, np.array(p["n"])) for p in c["points"]])),
                     c=(cn.min() - o0, cn.max() - o0), b=(0.0, bn.max() - o0), p=(pn.min() - o0, pn.max() - o0), seps=seps * 1e3))
print("positions along the reported normal, mm, origin = basket piece's lowest vertex along the normal")
print(f"{'episode':30s} body0   sep   | basket piece  | can piece       | contact pts     | point seps")
cand = {k: [] for k in ("c_min - b_max", "c_min - b_min", "c_max - b_max", "c_max - b_min", "-(c_min - b_min)", "-(c_max - b_min)", "-(b_max - c_min)", "b_min - c_max")}
for r in rows:
    c0, c1 = np.array(r["c"]) * 1e3; b1 = r["b"][1] * 1e3; p0, p1 = np.array(r["p"]) * 1e3
    print(f"{r['ep']:30s} {r['first']:6s} {r['sep']:6.1f} | [0, {b1:5.1f}]   | [{c0:6.1f},{c1:6.1f}] | [{p0:6.1f},{p1:6.1f}] | {np.array2string(np.sort(r['seps']), precision=1, floatmode='fixed')}")
    vals = {"c_min - b_max": c0 - b1, "c_min - b_min": c0, "c_max - b_max": c1 - b1, "c_max - b_min": c1, "-(c_min - b_min)": -c0, "-(c_max - b_min)": -c1, "-(b_max - c_min)": c0 - b1, "b_min - c_max": -c1}
    for k, v in vals.items(): cand[k].append(v - r["sep"])
print()
for k, v in cand.items():
    v = np.array(v); print(f"reported sep vs {k:18s}: median |diff| {np.median(np.abs(v)):5.1f} mm, within 1 mm in {(np.abs(v) < 1).sum()}/{len(v)}")
