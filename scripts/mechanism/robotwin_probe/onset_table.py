"""For each probed episode: the deepest can-basket contact at the onset step, the true distance between
the two convex pieces that report it (one step before and at onset), and what the same pair reported before."""
import json, glob, sys, numpy as np
from scipy.optimize import minimize

def hull_dist(A, B):
    A = np.asarray(A); B = np.asarray(B); na, nb = len(A), len(B)
    def f(w):
        d = w[:na] @ A - w[na:] @ B; return d @ d
    def g(w):
        d = w[:na] @ A - w[na:] @ B; return np.concatenate([2 * A @ d, -2 * B @ d])
    cons = [{"type": "eq", "fun": lambda w: w[:na].sum() - 1}, {"type": "eq", "fun": lambda w: w[na:].sum() - 1}]
    best = None
    ia = np.argmin(((A[:, None] - B[None]) ** 2).sum(-1)); i, j = divmod(ia, nb)
    for w0 in (np.r_[np.eye(na)[i], np.eye(nb)[j]], np.r_[np.full(na, 1 / na), np.full(nb, 1 / nb)]):
        r = minimize(f, w0, jac=g, bounds=[(0, 1)] * (na + nb), constraints=cons, method="SLSQP", options={"maxiter": 500, "ftol": 1e-14})
        if best is None or r.fun < best.fun: best = r
    w = best.x; pa, pb = w[:na] @ A, w[na:] @ B
    return float(np.sqrt(max(best.fun, 0))), pa, pb

def pair_points(step, pair):
    out = []
    for c in step["contacts"]:
        if tuple(map(tuple, c["shapes"])) == pair: out += c["points"]
    return out

rows = []
for fn in sorted(glob.glob(sys.argv[1] + "/*.json")):
    d = json.load(open(fn)); o = d["onset"]; S = d["steps"]
    if str(o) not in S or "vertices" not in S[str(o)] or "vertices" not in S.get(str(o - 1), {}): continue
    cur, prev = S[str(o)], S[str(o - 1)]
    deepest = min(((p["sep"], tuple(map(tuple, c["shapes"])), p) for c in cur["contacts"] for p in c["points"]), key=lambda t: t[0], default=None)
    if deepest is None: continue
    sep, pair, pt = deepest
    idx = dict(pair)  # {"can": i, "basket": j}
    if "can" not in idx or "basket" not in idx: continue
    res = {}
    for tag, st in (("prev", prev), ("cur", cur)):
        dist, pa, pb = hull_dist(st["vertices"]["can"][idx["can"]], st["vertices"]["basket"][idx["basket"]])
        res[tag] = (dist, pa, pb)
    prev_pts = pair_points(prev, pair); prev2 = pair_points(S.get(str(o - 2), {"contacts": []}), pair)
    dvec = res["cur"][2] - res["cur"][1]; dirn = dvec / (np.linalg.norm(dvec) + 1e-12)   # can -> basket closest direction
    n = np.array(pt["n"]); cosang = float(abs(n @ dirn))
    A = np.asarray(cur["vertices"]["basket"][idx["basket"]]); ext = np.sort(A.max(0) - A.min(0))
    C = np.asarray(cur["vertices"]["can"][idx["can"]]); extc = np.sort(C.max(0) - C.min(0))
    dv = np.linalg.norm(np.array(cur["state"]["can"]["v"]) - np.array(prev["state"]["can"]["v"]))
    dp = np.linalg.norm(np.array(cur["state"]["can"]["p"]) - np.array(prev["state"]["can"]["p"]))
    npairs_prev = len({tuple(map(tuple, c["shapes"])) for c in prev["contacts"]}); npairs_cur = len({tuple(map(tuple, c["shapes"])) for c in cur["contacts"]})
    rows.append(dict(ep=fn.split("/")[-1][:-5], pair=f"c{idx['can']}-b{idx['basket']}", sep_mm=sep * 1e3, true_prev=res["prev"][0] * 1e3, true_cur=res["cur"][0] * 1e3,
                     prev_rep=(min(p["sep"] for p in prev_pts) * 1e3 if prev_pts else None), prev2_rep=(min(p["sep"] for p in prev2) * 1e3 if prev2 else None),
                     npts=len(pair_points(cur, pair)), cos=cosang, bext=ext * 1e3, cext=extc * 1e3, dv=dv, dp_mm=dp * 1e3, np_prev=npairs_prev, np_cur=npairs_cur))
f = lambda x: "   -  " if x is None else f"{x:6.1f}"
print(f"{'episode':34s} {'pair':9s} {'rep.sep':>7s} {'true-1':>6s} {'true0':>6s} {'rep-1':>6s} {'rep-2':>6s} npt |cos| basket piece ext mm     can piece ext mm        dv m/s  dp mm pairs")
for r in rows:
    print(f"{r['ep']:34s} {r['pair']:9s} {r['sep_mm']:7.1f} {r['true_prev']:6.1f} {r['true_cur']:6.1f} {f(r['prev_rep'])} {f(r['prev2_rep'])} {r['npts']:3d} {r['cos']:.2f} "
          f"{np.array2string(r['bext'], precision=0, floatmode='fixed'):22s} {np.array2string(r['cext'], precision=0, floatmode='fixed'):22s} {r['dv']:6.2f} {r['dp_mm']:5.2f} {r['np_prev']}->{r['np_cur']}")
sep = np.array([r["sep_mm"] for r in rows]); tr = np.array([r["true_cur"] for r in rows]); trp = np.array([r["true_prev"] for r in rows])
print(f"\nn={len(rows)}  reported sep median {np.median(sep):.1f} mm; true distance at onset median {np.median(tr):.1f} (min {tr.min():.1f}, share > 0.5 mm: {(tr > 0.5).mean():.2f}); one step before median {np.median(trp):.1f}")
print(f"pair reported one step before: {sum(r['prev_rep'] is not None for r in rows)}/{len(rows)}; corr(-sep, true0) = {np.corrcoef(-sep, tr)[0, 1]:.2f}; median(-sep - true0) = {np.median(-sep - tr):.1f} mm")
