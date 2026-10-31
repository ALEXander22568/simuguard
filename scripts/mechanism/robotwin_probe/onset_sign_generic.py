"""Normal direction of the contact that starts each artifact, any task (output of probe_all_onsets.py).
PhysX convention seen on healthy contacts: the normal points from the second shape to the first.
For every onset: the deepest object-object contact of the onset step; side of the first shape's piece
relative to the second's along the reported normal; true distance between the two pieces; and whether
the reported separation equals (extreme vertex of the first piece - contact point) . normal."""
import json, glob, sys, collections, numpy as np
from scipy.optimize import minimize

def hull_dist(A, B):
    A = np.asarray(A); B = np.asarray(B); na, nb = len(A), len(B)
    f = lambda w: (lambda d: d @ d)(w[:na] @ A - w[na:] @ B)
    g = lambda w: (lambda d: np.concatenate([2 * A @ d, -2 * B @ d]))(w[:na] @ A - w[na:] @ B)
    cons = [{"type": "eq", "fun": lambda w: w[:na].sum() - 1}, {"type": "eq", "fun": lambda w: w[na:].sum() - 1}]
    i, j = divmod(np.argmin(((A[:, None] - B[None]) ** 2).sum(-1)), nb); best = None
    for w0 in (np.r_[np.eye(na)[i], np.eye(nb)[j]], np.r_[np.full(na, 1 / na), np.full(nb, 1 / nb)]):
        r = minimize(f, w0, jac=g, bounds=[(0, 1)] * (na + nb), constraints=cons, method="SLSQP", options={"maxiter": 400, "ftol": 1e-14})
        if best is None or r.fun < best.fun: best = r
    return float(np.sqrt(max(best.fun, 0)))

def geom(V, c):
    b0, b1 = c["bodies"]; s0, s1 = c["shapes"]
    if s0 < 0 or s1 < 0 or V[b0][s0] is None or V[b1][s1] is None: return None
    A = np.asarray(V[b0][s0]); B = np.asarray(V[b1][s1]); n = np.array(c["points"][0]["n"]); n /= np.linalg.norm(n)
    return A, B, n

per_task = collections.defaultdict(lambda: {"healthy": [0, 0], "onsets": [], "no_deep": 0})
for fn in sorted(glob.glob(sys.argv[1] + "/*.json")):
    d = json.load(open(fn)); T = per_task[d["task"]]
    for o in d["onsets"]:
        prev, cur = d["steps"].get(str(o - 1)), d["steps"].get(str(o))
        if not prev or not cur or "vertices" not in prev: continue
        V = prev["vertices"]
        for c in prev["contacts"]:
            if min(p["sep"] for p in c["points"]) < -0.005: continue
            g = geom(V, c)
            if g is None: continue
            A, B, n = g; T["healthy"][0] += 1; T["healthy"][1] += float((A @ n).mean() - (B @ n).mean()) > 0
        deep = [(min(p["sep"] for p in c["points"]), c) for c in cur["contacts"]]
        deep = [x for x in deep if x[0] < -0.005]
        if not deep: T["no_deep"] += 1; continue
        sep, c = min(deep, key=lambda x: x[0]); g = geom(V, c)
        if g is None: continue
        A, B, n = g; an, bn = A @ n, B @ n
        pts = np.array([p["p"] for p in c["points"]]) @ n; seps = np.array([p["sep"] for p in c["points"]])
        ident = float(np.min(np.abs(seps[np.argmin(seps)] - (an.min() - pts[np.argmin(seps)]))))
        T["onsets"].append(dict(ep=fn.split("__")[-1][:-5], o=o, sep=sep * 1e3, plus=float(an.mean() - bn.mean()) > 0, dist=hull_dist(A, B) * 1e3, ident=ident * 1e3,
                                pair=f"{d['names'][c['bodies'][0]][:10]}#{c['shapes'][0]}-{d['names'][c['bodies'][1]][:10]}#{c['shapes'][1]}", npt=len(seps)))
for task, T in per_task.items():
    R = T["onsets"]; h = T["healthy"]
    print(f"\n== {task}: onsets with a deep (< -5 mm) object-object contact {len(R)}, onsets without one {T['no_deep']}")
    print(f"   healthy contacts one step earlier: {h[0]}, first piece on the +normal side in {100 * h[1] / max(h[0], 1):.1f}%")
    if not R: continue
    inv = [r for r in R if not r["plus"]]
    print(f"   onset contacts with the first piece on the -normal side (inverted): {len(inv)}/{len(R)}")
    print(f"   true distance between the two pieces, median {np.median([r['dist'] for r in R]):.1f} mm (inverted ones {np.median([r['dist'] for r in inv]) if inv else float('nan'):.1f}); > 2 mm in {sum(r['dist'] > 2 for r in R)}/{len(R)}")
    print(f"   reported sep median {np.median([r['sep'] for r in R]):.1f} mm; identity |sep - (lowest vertex - point).n| < 0.5 mm in {sum(r['ident'] < 0.5 for r in R)}/{len(R)}")
    for r in R[:40]: print(f"     {r['ep'][:28]:28s} o={r['o']:6d} {r['pair']:30s} sep {r['sep']:7.1f} dist {r['dist']:5.1f} {'inverted' if not r['plus'] else 'normal  '} ident {r['ident']:.2f} npt {r['npt']}")
