"""Geometric truth for the contact points PhysX reports between a body and a static SDF collider.
For every reported point: signed distance of the point to the collider's triangle mesh (negative = inside),
computed independently (closest triangle + ray parity), against the reported separation.
Also the deepest vertex of the body's collision mesh inside the collider at the pose each step starts from.
usage: analyze_tubes.py DUMP.json BODY_PATH_FRAGMENT STATIC_PATH_FRAGMENT [steps...]
"""
import json, sys, numpy as np
from scipy.spatial import cKDTree

d = json.load(open(sys.argv[1])); bfrag, sfrag = sys.argv[2], sys.argv[3]
steps = [int(x) for x in sys.argv[4:]] or sorted(map(int, d["steps"]))

def find_mesh(frag, need="collision"):
    for g in d["geometry"].values():
        for p in g["prims"]:
            if frag in p["path"] and need in p["path"] and p["type"] == "Mesh" and "PhysicsCollisionAPI" in p["schemas"]:
                return p
def tris(p):
    idx = np.array(p["face_indices"]); cnt = np.array(p["face_counts"]); out = []; o = 0
    for c in cnt:
        for k in range(1, c - 1): out.append((idx[o], idx[o + k], idx[o + k + 1]))
        o += c
    return np.array(out)
def xf(P, M):
    M = np.array(M); return P @ M[:3, :3] + M[3, :3]

bm, sm = find_mesh(bfrag), find_mesh(sfrag)
print("body mesh:", bm["path"], len(bm["points"]), "pts, approximation", bm["attrs"].get("physics:approximation"))
print("static mesh:", sm["path"], len(sm["points"]), "pts, approximation", sm["attrs"].get("physics:approximation"))
SP = xf(np.array(sm["points"]), sm["world"]); ST = tris(sm); A, B, C = SP[ST[:, 0]], SP[ST[:, 1]], SP[ST[:, 2]]
print("static mesh world bbox min", SP.min(0).round(4), "max", SP.max(0).round(4), "triangles", len(ST))
ctree = cKDTree((A + B + C) / 3); edge = max(np.linalg.norm(B - A, axis=1).max(), np.linalg.norm(C - A, axis=1).max())

def pt_tri(P, a, b, c):
    # distance from points P (n,3) to triangles (n,3)x3 (Ericson)
    ab, ac, ap = b - a, c - a, P - a
    d1 = (ab * ap).sum(1); d2 = (ac * ap).sum(1)
    bp = P - b; d3 = (ab * bp).sum(1); d4 = (ac * bp).sum(1)
    cp = P - c; d5 = (ab * cp).sum(1); d6 = (ac * cp).sum(1)
    va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
    Q = np.empty_like(P)
    denom = va + vb + vc; denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    v = vb / denom; w = vc / denom; Q[:] = a + ab * v[:, None] + ac * w[:, None]
    m = (d1 <= 0) & (d2 <= 0); Q[m] = a[m]
    m2 = (d3 >= 0) & (d4 <= d3); Q[m2] = b[m2]
    m3 = (vc <= 0) & (d1 >= 0) & (d3 <= 0); t = d1 / np.where(np.abs(d1 - d3) < 1e-30, 1e-30, d1 - d3); Q[m3] = (a + ab * t[:, None])[m3]
    m4 = (d6 >= 0) & (d5 <= d6); Q[m4] = c[m4]
    m5 = (vb <= 0) & (d2 >= 0) & (d6 <= 0); t = d2 / np.where(np.abs(d2 - d6) < 1e-30, 1e-30, d2 - d6); Q[m5] = (a + ac * t[:, None])[m5]
    m6 = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0); t = (d4 - d3) / np.where(np.abs((d4 - d3) + (d5 - d6)) < 1e-30, 1e-30, (d4 - d3) + (d5 - d6)); Q[m6] = (b + (c - b) * t[:, None])[m6]
    return np.linalg.norm(P - Q, axis=1)
def udist(P, k=64):
    P = np.atleast_2d(P); k = min(k, len(ST)); _, nn = ctree.query(P, k=k)
    best = np.full(len(P), np.inf)
    for j in range(k):
        t = nn[:, j]; best = np.minimum(best, pt_tri(P, A[t], B[t], C[t]))
    return best
def inside(P):
    P = np.atleast_2d(P); votes = np.zeros(len(P), int)
    for dirn in (np.array([0.123, 0.456, 0.881]), np.array([-0.61, 0.31, 0.73]), np.array([0.2, -0.9, 0.38])):
        dirn = dirn / np.linalg.norm(dirn); e1, e2 = B - A, C - A; h = np.cross(dirn, e2); det = (e1 * h).sum(1)
        ok = np.abs(det) > 1e-14; inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
        cnt = np.zeros(len(P), int)
        for i in range(0, len(P), 256):
            s = P[i:i + 256, None, :] - A[None]; u = (s * h[None]).sum(2) * inv
            q = np.cross(s, e1[None]); v = (q * dirn).sum(2) * inv; t = (q * e2[None]).sum(2) * inv
            cnt[i:i + 256] = (ok[None] & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)).sum(1)
        votes += cnt % 2
    return votes >= 2
def sdist(P):
    u = udist(P); return np.where(inside(P), -u, u)

BPl = np.array(bm["points"])
for s in steps:
    st = d["steps"][str(s)]; prev = d["steps"].get(str(s - 1))
    cs = [c for c in st["contacts"] if (bfrag in c["actor0"] + c["actor1"]) and (sfrag in c["actor0"] + c["actor1"])]
    line = f"step {s}: "
    if prev:
        Mb = next(v for k, v in prev["poses"][[n for n in prev["poses"] if bfrag.split('/')[0] in d['roots'][n] or True][0]]["prims"].items() if bm["path"] == k) if False else None
    # body mesh world at the pose the step starts from (= pose after the previous step)
    def body_world(frame):
        for n, pv in frame["poses"].items():
            if bm["path"] in pv["prims"]: return pv["prims"][bm["path"]]
    if prev and body_world(prev):
        W = xf(BPl, body_world(prev)); sub = W[::7]; sd = sdist(sub); i = int(np.argmin(sd))
        near = W[np.linalg.norm(W - sub[i], axis=1) < 0.004]; sdn = sdist(near) if len(near) else sd[i:i + 1]
        line += f"deepest body mesh vertex inside the static mesh at the start pose: {min(sd.min(), sdn.min()) * 1e3:7.2f} mm ({(sd < -0.0005).sum()} of {len(sub)} sampled vertices deeper than 0.5 mm) | "
    if not cs: print(line + "no contact reported"); continue
    P = np.array([p["p"] for c in cs for p in c["points"]]); N = np.array([p["n"] for c in cs for p in c["points"]]); S = np.array([p["sep"] for c in cs for p in c["points"]])
    I = np.array([np.linalg.norm(p["imp"]) for c in cs for p in c["points"]])
    truth = sdist(P); truth_shift = sdist(P - N * S[:, None]); truth_shift2 = sdist(P + N * S[:, None])
    j = int(np.argmin(S))
    line += f"{len(S)} pts, reported sep min {S.min() * 1e3:7.2f} mm (median {np.median(S) * 1e3:.2f}), impulse sum {I.sum():.4f} N s"
    print(line)
    print(f"     at the reported points: signed distance to the static mesh min {truth.min() * 1e3:.2f} / median {np.median(truth) * 1e3:.2f} mm; "
          f"|reported sep - true signed distance|: median {np.median(np.abs(S - truth)) * 1e3:.2f} mm, max {np.abs(S - truth).max() * 1e3:.2f} mm")
    print(f"     deepest reported point: sep {S[j] * 1e3:.2f} mm, true signed distance at p {truth[j] * 1e3:.2f}, at p - n*sep {truth_shift[j] * 1e3:.2f}, at p + n*sep {truth_shift2[j] * 1e3:.2f} mm; normal {N[j].round(3)} impulse {I[j]:.4f}")
    if s in steps[-3:] or S.min() < -0.002:
        order = np.argsort(S)[:8]
        for q in order: print(f"        p {P[q].round(4)} n {N[q].round(3)} sep {S[q] * 1e3:7.2f} true {truth[q] * 1e3:7.2f} imp {I[q]:.4f}")
