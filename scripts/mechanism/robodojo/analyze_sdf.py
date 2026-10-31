"""Is the static collider's mesh a closed surface, and where does the body point that PhysX reports as deep
inside actually sit?  Generalised winding number (1 inside, 0 outside, fractional = open or self-intersecting mesh)."""
import json, sys, collections, numpy as np
d = json.load(open(sys.argv[1])); sfrag, onset = sys.argv[2], int(sys.argv[3]); bfrag = sys.argv[4]
sm = [p for gg in d["geometry"].values() for p in gg["prims"] if sfrag in p["path"] and "collision" in p["path"] and p["type"] == "Mesh"][0]
M = np.array(sm["world"]); SP = np.array(sm["points"]) @ M[:3, :3] + M[3, :3]
idx = np.array(sm["face_indices"]); cnt = np.array(sm["face_counts"]); T = []; o = 0
for c in cnt:
    for k in range(1, c - 1): T.append((idx[o], idx[o + k], idx[o + k + 1]))
    o += c
T = np.array(T)
# weld coincident vertices before the edge test
key = np.round(SP / 1e-6).astype(np.int64); _, inv = np.unique(key, axis=0, return_inverse=True); Tw = inv[T]
edges = collections.Counter(); directed = collections.Counter()
for a, b, c in Tw:
    for u, v in ((a, b), (b, c), (c, a)):
        edges[(min(u, v), max(u, v))] += 1; directed[(u, v)] += 1
hist = collections.Counter(edges.values())
print(f"static mesh: {len(SP)} points ({inv.max()+1} after welding), {len(cnt)} faces ({dict(collections.Counter(cnt.tolist()))} vertices per face), {len(T)} triangles")
print(f"edges by number of incident triangles: {dict(sorted(hist.items()))}  -> closed 2-manifold needs every edge shared by exactly 2; boundary edges: {hist.get(1, 0)}")
print(f"inconsistently oriented edge pairs: {sum(1 for (u, v), n in directed.items() if n > 1)}; degenerate triangles: {int((np.linalg.norm(np.cross(SP[T[:,1]]-SP[T[:,0]], SP[T[:,2]]-SP[T[:,0]]),axis=1) < 1e-12).sum())}")
A, B, C = SP[T[:, 0]], SP[T[:, 1]], SP[T[:, 2]]
def winding(q):
    a, b, c = A - q, B - q, C - q; la, lb, lc = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1), np.linalg.norm(c, axis=1)
    num = np.einsum("ij,ij->i", a, np.cross(b, c)); den = la * lb * lc + (a * b).sum(1) * lc + (b * c).sum(1) * la + (c * a).sum(1) * lb
    return float(np.arctan2(num, den).sum() / (2 * np.pi))
vol = float(np.einsum("ij,ij->i", A, np.cross(B, C)).sum() / 6)
print(f"signed volume {vol*1e6:.1f} cm^3; bbox {SP.min(0).round(4)} .. {SP.max(0).round(4)}; winding number at the bbox centre {winding((SP.min(0)+SP.max(0))/2):.3f}, 5 cm above {winding(SP.max(0)+[0,0,0.05]):.3f}")
st = d["steps"][str(onset)]
pts = sorted([(p["sep"], np.array(p["p"]), np.array(p["n"])) for c in st["contacts"] if bfrag in c["actor0"] + c["actor1"] and sfrag in c["actor0"] + c["actor1"] for p in c["points"]], key=lambda t: t[0])
print("\nonset contact points: p = reported position (on the collider surface), q = p + n*sep (where the body's point would be)")
for sep, p, n in pts[:3] + pts[-3:]:
    q = p + n * sep
    print(f"  sep {sep*1e3:7.2f} mm  p {p.round(4)} winding {winding(p + 1e-5*n):.2f}/{winding(p - 1e-5*n):.2f} (just off the surface on the +n / -n side)   q {q.round(4)} winding {winding(q):.3f}")
# scan the winding number along the reported normal through the deepest point
sep, p, n = pts[0]
print("\nwinding number along the line p + t n (t in mm) for the deepest point:", " ".join(f"{t:+d}:{winding(p + n*t*1e-3):.2f}" for t in (-14, -11, -8, -5, -2, -1, 1, 2, 5, 8, 11)))
