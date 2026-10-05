"""Stack Bowls Three onsets: same pose sequences through clean PhysX 5.3.1 and through 5.3.1 with only
GuGJKSimplex.h from 5.4.0 (and through 5.9.0). Deepest separation the program outputs at the onset step."""
import json, subprocess, re
from pathlib import Path
P = Path("/mnt/nvme0/shared/USER/simuguard/probe")
R = json.load(open(P / "physx_repro/bowls_results.json"))
bins = {"5.3.1": P / "physx_repro_versions/105.1-physx-5.3.1/build/repro", "5.3.1+simplex fix": P / "physx_repro_versions/var_simplex_eps/build/repro", "5.9.0": P / "physx_repro_590/build/repro"}
def run(b, d, order, start, onset):
    out = subprocess.run([str(b), str(d), str(order), "0.1", "0.02", str(start)], capture_output=True, text=True, cwd=str(P / "physx_repro")).stdout
    m = [l for l in out.splitlines() if l.startswith(f"step {onset}:")]
    if not m: return None
    seps = [float(x) for x in m[0].split("sep")[1].split()]
    return min(seps) * 1e3
print("history values:", sorted({str(r["history"])[:40] for r in R}))
rows = []
for r in R:
    d = P / "physx_repro/data/bowls" / r["tag"]; h = str(r["history"]); o = r["onset"]
    m = re.search(r"(\d+)", h); start = 1 if h.startswith("full") or not m else o - int(m.group(1))
    res = {k: run(b, d, r["order"], start, o) for k, b in bins.items()}
    kind = "true penetration" if r.get("epa_status") is not None and (r.get("true_sep_mm") or 0) < -1 else ("not reproduced" if not r.get("match") and r["tag"].startswith("seg0043") else "false depth")
    rows.append((r["tag"], kind, r["sapien"]["sep_mm"][0] if r.get("sapien") else None, r.get("true_sep_mm"), res))
    f = lambda v: "  none " if v is None else f"{v:7.1f}"
    print(f"{r['tag']:18s} {kind:16s} sapien {f(min(r['sapien']['sep_mm']) if r.get('sapien') else None)} true {f(r.get('true_sep_mm'))} | 5.3.1 {f(res['5.3.1'])} | +fix {f(res['5.3.1+simplex fix'])} | 5.9.0 {f(res['5.9.0'])}   start {start}")
fd = [x for x in rows if x[1] == "false depth"]
deep = lambda v: v is not None and v < -5
print(f"\nfalse depth cases {len(fd)}: deep contact at onset on 5.3.1 in {sum(deep(x[4]['5.3.1']) for x in fd)}, with the simplex fix in {sum(deep(x[4]['5.3.1+simplex fix']) for x in fd)}, on 5.9.0 in {sum(deep(x[4]['5.9.0']) for x in fd)}")
tp = [x for x in rows if x[1] == "true penetration"]
print(f"true penetration cases {len(tp)}: deep on 5.3.1 {sum(deep(x[4]['5.3.1']) for x in tp)}, with fix {sum(deep(x[4]['5.3.1+simplex fix']) for x in tp)}, on 5.9.0 {sum(deep(x[4]['5.9.0']) for x in tp)}")
