"""Sign of the reported contact normal: on which side of the basket piece does the can piece sit along it?
Healthy contacts (the step before onset, all can-basket pairs) against the contact that starts each artifact."""
import json, glob, sys, numpy as np
def side(step_vertices, c):
    idx = dict(map(tuple, c["shapes"]))
    if "can" not in idx or "basket" not in idx or idx["can"] < 0 or idx["basket"] < 0: return None
    C = np.asarray(step_vertices["can"][idx["can"]]); B = np.asarray(step_vertices["basket"][idx["basket"]])
    n = np.array(c["points"][0]["n"]); n /= np.linalg.norm(n)
    cn, bn = C @ n, B @ n
    gap_pos = cn.min() - bn.max()      # > 0: can entirely on the +n side of the basket piece
    gap_neg = bn.min() - cn.max()      # > 0: can entirely on the -n side
    return gap_pos, gap_neg, float(np.mean(cn) - np.mean(bn)), cn, bn
H = {"touch": [], "spec": []}; A = []
for fn in sorted(glob.glob(sys.argv[1] + "/rep*.json")):
    d = json.load(open(fn)); o = d["onset"]; S = d["steps"]
    if "vertices" not in S.get(str(o - 1), {}): continue
    prev, cur = S[str(o - 1)], S[str(o)]
    if "vertices" in S.get(str(o - 2), {}) or True:
        # healthy: contacts reported at step o-1 use poses at the end of o-2; vertices exist only for o-1 and o,
        # the can moves < 1 mm per step before onset, so o-1 vertices are used (error < 1 mm)
        for c in prev["contacts"]:
            if not c["points"]: continue
            r = side(prev["vertices"], c)
            if r is None: continue
            s = min(p["sep"] for p in c["points"])
            if s < -0.005: continue
            H["touch" if s < 0.002 else "spec"].append((r[2], s, r[0], r[1]))
    best = min(((p["sep"], c) for c in cur["contacts"] for p in c["points"]), key=lambda t: t[0])
    if best[0] > -0.005: continue
    r = side(prev["vertices"], best[1]); P = np.array([p["p"] for p in best[1]["points"]]); n = np.array(best[1]["points"][0]["n"]); n /= np.linalg.norm(n)
    seps = np.array([p["sep"] for p in best[1]["points"]]); pn = P @ n
    inside = np.mean((seps >= r[3].min() - pn - 5e-4) & (seps <= r[3].max() - pn + 5e-4))
    A.append((r[2], best[0], r[0], r[1], inside, float(np.min(seps + pn - r[3].min())), fn.split("/")[-1]))
for k, v in H.items():
    v = np.array(v); print(f"healthy {k:5s} contacts (n={len(v)}): can piece centre on +normal side in {(v[:,0] > 0).mean()*100:.1f}%; median reported sep {np.median(v[:,1])*1e3:.1f} mm")
a = np.array([x[:6] for x in A])
print(f"artifact onset contacts (n={len(a)}): can piece centre on +normal side in {(a[:,0] > 0).mean()*100:.1f}%; on -normal side {(a[:,0] < 0).mean()*100:.1f}%")
print(f"  can piece entirely on the -normal side of the basket piece (separated along the reported axis): {(a[:,3] > 0).sum()}/{len(a)}, gap along it median {np.median(a[a[:,3] > 0, 3])*1e3:.1f} mm")
print(f"  reported sep of each point equals (a point of the can piece - the point on the basket piece) . normal: share of points consistent {a[:,4].mean()*100:.0f}%")
print(f"  deepest point: sep - (lowest can vertex - basket point) along normal: median {np.median(a[:,5])*1e3:.2f} mm, max |.| {np.abs(a[:,5]).max()*1e3:.2f} mm")
print("  exceptions (centre on + side):", [x[6][5:-5] for x in A if x[0] > 0])
