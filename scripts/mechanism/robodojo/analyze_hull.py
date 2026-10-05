"""Truth with the body's convex hull (the collision mesh's hull; PhysX uses a convex decomposition of it):
for each reported contact point p (a point on the static SDF collider's surface) the signed distance of p to the
body hull at the pose the step starts from (negative = the collider's surface point is inside the body), and the
exit distance along the reported normal. Also the history of that for the deepest point of the onset step."""
import json, sys, numpy as np
from scipy.spatial import ConvexHull
d = json.load(open(sys.argv[1])); bfrag, sfrag, onset = sys.argv[2], sys.argv[3], int(sys.argv[4])
g = [p for gg in d["geometry"].values() for p in gg["prims"] if bfrag in p["path"] and "collision" in p["path"] and p["type"] == "Mesh"][0]
P = np.array(g["points"]); H = ConvexHull(P); A, b = H.equations[:, :3], H.equations[:, 3]
print("body hull: %d of %d vertices, volume %.1f cm^3, mesh local bbox %s .. %s" % (len(H.vertices), len(P), H.volume * 1e6, P.min(0).round(4), P.max(0).round(4)))
def M_of(s):
    for pv in d["steps"][str(s)]["poses"].values():
        if g["path"] in pv["prims"]: return np.array(pv["prims"][g["path"]])
def to_local(x, M): return (np.atleast_2d(x) - M[3, :3]) @ np.linalg.inv(M[:3, :3])
def sd_local(xl): return (xl @ A.T + b).max(1)           # < 0 inside the hull; equals -depth to the nearest face when inside
def exit_along(xl, nl):                                   # travel along +nl / -nl until leaving the hull (only when inside)
    out = []
    for sgn in (1, -1):
        den = (A @ (sgn * nl)); num = -(A @ xl + b); m = den > 1e-12
        out.append(float((num[m] / den[m]).min()) if m.any() else np.inf)
    return out
def contacts(s):
    return [(np.array(p["p"]), np.array(p["n"]), p["sep"], float(np.linalg.norm(p["imp"])), c["actor0"]) for c in d["steps"][str(s)]["contacts"]
            if bfrag in c["actor0"] + c["actor1"] and sfrag in c["actor0"] + c["actor1"] for p in c["points"]]
steps = sorted(int(k) for k in d["steps"])
print("\nper step: reported separation vs truth at the reported points (pose the step starts from)")
for s in [x for x in steps if x >= onset - 8 and x <= onset + 1 and x - 1 in steps]:
    cs = contacts(s); M = M_of(s - 1)
    if not cs: print(s, "no contact"); continue
    X = np.array([c[0] for c in cs]); N = np.array([c[1] for c in cs]); S = np.array([c[2] for c in cs]); I = np.array([c[3] for c in cs])
    sd = sd_local(to_local(X, M)); R = M[:3, :3]
    print(f"{s}: {len(S)} pts | reported sep min {S.min()*1e3:7.2f} mm | hull signed distance at those points min {sd.min()*1e3:7.2f} mm, at the deepest reported point {sd[np.argmin(S)]*1e3:7.2f} mm | |sep - truth| median {np.median(np.abs(S - sd))*1e3:.2f} mm | impulse {I.sum():.4f}")
    if s == onset:
        for q in np.argsort(S)[:6]:
            xl = to_local(X[q], M)[0]; nl = N[q] @ np.linalg.inv(R)
            ex = exit_along(xl, nl / np.linalg.norm(nl)) if sd[q] < 0 else (None, None)
            print(f"      p {X[q].round(4)} n {N[q].round(3)} sep {S[q]*1e3:7.2f} | point inside body hull: {sd[q] < 0} (nearest face {sd[q]*1e3:6.2f} mm; exit along +n {ex[0] if ex[0] is None else round(ex[0]*1e3,2)} mm, along -n {ex[1] if ex[1] is None else round(ex[1]*1e3,2)} mm) imp {I[q]:.4f}")
cs = contacts(onset); q = int(np.argmin([c[2] for c in cs])); x0, n0 = cs[q][0], cs[q][1]
print(f"\nhistory of the onset's deepest collider point p = {x0.round(4)} (fixed in the world, the collider is static):")
print(" step  hull signed distance at the start pose (mm)   reported contacts within 3 mm of p (sep mm, normal)")
for s in [x for x in steps if onset - 14 <= x <= onset + 1 and x - 1 in steps]:
    M = M_of(s - 1); sd = sd_local(to_local(x0, M))[0]
    near = [(c[2], c[1]) for c in contacts(s) if np.linalg.norm(c[0] - x0) < 0.003]
    print(f" {s}  {sd*1e3:8.2f}   {'; '.join('%.2f %s' % (a*1e3, np.round(nn,2)) for a, nn in near[:3]) or '-'}")
# how far the body surface moved at that point in the last step: velocity of the body point coinciding with p
M0, M1 = M_of(onset - 2), M_of(onset - 1)
xl = to_local(x0, M1)[0]; moved = xl @ M1[:3, :3] + M1[3, :3] - (xl @ M0[:3, :3] + M0[3, :3])
print(f"\nmaterial point of the body at p moved {np.linalg.norm(moved)*1e3:.2f} mm in the step before the onset ({np.linalg.norm(moved)/0.004:.2f} m/s), component along the reported normal {moved @ n0 * 1e3:.2f} mm")
