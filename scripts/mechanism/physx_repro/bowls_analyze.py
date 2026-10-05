"""For every exported Stack Bowls Three case: run the standalone PhysX program with debug prints over the whole pose
history, keep the block of the onset step, compare its output with what SAPIEN reported, and compute an independent
ground truth (do the two convex pieces intersect at the pose the onset step starts from, and how deep).
usage: bowls_analyze.py [CASE_TAG ...]   (default: all of bowls_cases.json that are exported)
"""
import json, subprocess, sys, re, itertools
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
D = Path("/mnt/nvme0/shared/USER/simuguard/probe/physx_repro"); B = D / "data/bowls"

def quat_R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def load_hull(fn):
    L = open(fn).read().split("\n"); n = int(L[0]); V = np.array([[float(x) for x in l.split()] for l in L[1:1 + n]])
    kv = {l.split()[0]: [float(x) for x in l.split()[1:]] for l in L[1 + n:] if l.strip()}
    return V * np.array(kv["scale"]), np.array(kv["local_p"]), np.array(kv["local_q_wxyz"])

def world(V, lp, lq, p, q):
    return (quat_R(q) @ ((quat_R(lq) @ V.T).T + lp).T).T + p

def sat(A, B):
    """Largest separation over face normals and edge-edge axes; > 0: disjoint (lower bound of the distance),
    < 0: intersecting, -value = minimum translation depth."""
    axes = []
    edges = []
    for V in (A, B):
        h = ConvexHull(V); axes += [e[:3] for e in h.equations]
        E = set()
        for s in h.simplices:
            for i, j in ((0, 1), (1, 2), (0, 2)): E.add((min(s[i], s[j]), max(s[i], s[j])))
        d = np.array([V[j] - V[i] for i, j in E]); d /= np.linalg.norm(d, axis=1)[:, None]
        # unique directions
        u = []
        for v in d:
            if not any(abs(abs(v @ w) - 1) < 1e-6 for w in u): u.append(v)
        edges.append(np.array(u))
    for a, b in itertools.product(edges[0], edges[1]):
        c = np.cross(a, b); n = np.linalg.norm(c)
        if n > 1e-8: axes.append(c / n)
    best = -1e9; bax = None
    for n in axes:
        n = np.asarray(n); a, b = A @ n, B @ n
        s = max(b.min() - a.max(), a.min() - b.max())
        if s > best: best, bax = s, n
    return best, bax

def run_case(tag):
    d = B / tag; info = json.loads((d / "info.json").read_text()); o = info["onset"]
    sap = [l.split() for l in open(d / "contacts.txt") if l.split() and int(l.split()[0]) == o]
    res = {"tag": tag, "onset": o, "centre_side": info["centre_side"]}
    sep_s = None
    if sap:
        t = sap[0]; npt = int(t[2]); n = [float(x) for x in t[4:7]]; seps = [float(x) for x in t[8:8 + npt]]
        res["sapien"] = {"first": t[1], "n": n, "sep_mm": [round(s * 1e3, 2) for s in seps]}; sep_s = (n, seps)
        order = 0 if t[1] == info["b0"].replace(" ", "_") else 1
    else:
        order = 0; res["sapien"] = None
    res["order"] = order
    def repro(start):
        p = subprocess.Popen([str(D / "build/repro"), str(d), str(order), "0.1", "0.02", str(start), "1"], stdout=subprocess.PIPE, text=True, cwd=str(D))
        block = []; keep = False; last_range = None
        for line in p.stdout:
            if line.startswith("--- step "):
                s = int(line.split()[2]); keep = (s == o)
            elif line.startswith("step ") and ": pair in range" in line and int(line.split()[1].rstrip(":")) <= o: last_range = int(line.split()[1].rstrip(":"))
            if keep: block.append(line.rstrip("\n"))
        p.wait()
        return block, last_range
    def matches(block):
        m = re.search(r"^step %d: ret \d+ contacts (\d+) manifoldPts \d+ n (\S+) (\S+) (\S+) sep(.*)$" % o, "\n".join(block), re.M)
        if not m or not sep_s: return False
        rn = [float(m.group(i)) for i in (2, 3, 4)]; rs = [float(x) for x in m.group(5).split()]
        return bool(len(rs) == len(sep_s[1]) and np.allclose(sorted(rs), sorted(sep_s[1]), atol=2e-5) and np.allclose(rn, sep_s[0], atol=2e-4))
    block, last_range = repro(1); res["history"] = "full"
    if not matches(block):
        # the program's cache can drift from the engine's over hundreds of steps; look for a shorter history that
        # ends in SAPIEN's onset contact (empty cache at the start step)
        for k in (0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 100, 200, 400):
            blk, lr = repro(o - k)
            if matches(blk): block, last_range = blk, lr; res["history"] = "from onset-%d (empty cache)" % k; break
        else:
            res["history"] = "full (no start step in onset-0..400 reproduces it)"
    res["in_range_since"] = last_range; res["block"] = block
    txt = "\n".join(block)
    g = lambda pat, cast=str, default=None: (cast(re.search(pat, txt).group(1)) if re.search(pat, txt) else default)
    res["regenerate"] = "REGENERATE" in txt
    res["pts_before"] = g(r"before refresh: pts=(\d+)", int); res["pts_after_refresh"] = g(r"after refresh: pts=(\d+)", int)
    res["warm"] = g(r"REGENERATE \(gjk\); warm=(\d+)", int)
    res["gjk_status"] = g(r"gjkPenetration status=(\d+)", int); res["gjk_penDep_mm"] = g(r"gjkPenetration status=.*penDep=(-?[\d.]+)", lambda x: round(float(x) * 1e3, 2))
    res["degenerate_exit"] = "NO PROGRESS" in txt
    res["epa_status"] = g(r"epaPenetration returned status=(\d+)", int); res["epa_penDep_mm"] = g(r"epaPenetration returned.*penDep=(-?[\d.]+)", lambda x: round(float(x) * 1e3, 2))
    res["doOverlapTest"] = g(r"doOverlapTest=(\d)", int); res["fullContactGen"] = g(r"fullContactGen=(\d)", int)
    res["sat"] = [l for l in block if l.startswith("[sat]")]
    res["dotA"] = g(r"signed dot\(A face normal, gjk normal in A\)=(-?[\d.]+)", float); res["dotB"] = g(r"signed dot\(B face normal, gjk normal\)=(-?[\d.]+)", float)
    res["reference"] = g(r"-> reference is (\S+)")
    m = re.search(r"^step %d: ret \d+ contacts (\d+) manifoldPts \d+ n (\S+) (\S+) (\S+) sep(.*)$" % o, txt, re.M)
    if m:
        rn = [float(m.group(i)) for i in (2, 3, 4)]; rs = [float(x) for x in m.group(5).split()]
        res["repro"] = {"n": rn, "sep_mm": [round(s * 1e3, 2) for s in rs]}
        if sep_s:
            res["match"] = bool(len(rs) == len(sep_s[1]) and np.allclose(sorted(rs), sorted(sep_s[1]), atol=2e-5) and np.allclose(rn, sep_s[0], atol=2e-4))
    else:
        res["repro"] = None; res["match"] = False if sep_s else None
    # ground truth at the pose the onset step starts from
    pose = None
    for l in open(d / "poses.txt"):
        if l.startswith(str(o) + " "): pose = [float(x) for x in l.split()[1:]]; break
    A = world(*load_hull(d / "hull_can.txt"), np.array(pose[0:3]), np.array(pose[3:7])); Bv = world(*load_hull(d / "hull_basket.txt"), np.array(pose[7:10]), np.array(pose[10:14]))
    s, ax = sat(A, Bv)
    res["true_sep_mm"] = round(s * 1e3, 2)     # > 0 gap (lower bound), < 0 penetration depth
    res["ext_A_mm"] = [round(x * 1e3, 1) for x in np.sort(A.max(0) - A.min(0))]; res["ext_B_mm"] = [round(x * 1e3, 1) for x in np.sort(Bv.max(0) - Bv.min(0))]
    return res

def main():
    cases = json.load(open(D / "bowls_cases.json"))
    tags = sys.argv[1:] or [c["tag"] for c in cases if (B / c["tag"] / "info.json").exists() and (B / c["tag"] / "poses.txt").stat().st_size > 0]
    out = []
    for t in tags:
        try:
            r = run_case(t); out.append(r)
            print(json.dumps({k: v for k, v in r.items() if k not in ("block",)}), flush=True)
        except Exception as ex:  # noqa: BLE001
            print("FAILED", t, repr(ex), flush=True)
    json.dump(out, open(D / "bowls_results.json", "w"), indent=1)

if __name__ == "__main__":
    main()
