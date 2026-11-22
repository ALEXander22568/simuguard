"""Per-episode wall time before/after a cutoff (UTC ISO), from episodes.jsonl in a run dir.
Usage: python3 ep_timing.py RUN_DIR CUTOFF_UTC [POLICY...]"""
import json, sys, glob, statistics as st
from collections import defaultdict
run, cut = sys.argv[1], sys.argv[2]
pols = sys.argv[3:] or None
rows = defaultdict(lambda: {"before": [], "after": []})
hourly = defaultdict(list)
for f in glob.glob(f"{run}/*/*/episodes.jsonl"):
    for line in open(f):
        try: d = json.loads(line)
        except Exception: continue
        if d.get("status") != "counted" or not d.get("episode_wall_s") or not d.get("actions"): continue
        if pols and d["policy"] not in pols: continue
        side = "after" if d["started_utc"] >= cut else "before"
        sps = d["episode_wall_s"] / d["actions"]
        rows[(d["policy"], d["task"])][side].append((d["episode_wall_s"], sps))
        hourly[d["started_utc"][:13]].append(sps)
def s(v):
    if not v: return "n=0"
    w = [x[0] for x in v]; p = [x[1] for x in v]
    return f"n={len(v):3d} wall med {st.median(w):6.0f}s  s/step med {st.median(p):.3f} mean {st.mean(p):.3f}"
print("policy/task".ljust(32), "BEFORE".ljust(52), "AFTER")
for k in sorted(rows):
    print(f"{k[0]}/{k[1]}".ljust(32), s(rows[k]["before"]).ljust(52), s(rows[k]["after"]))
print("\nhourly s/step (UTC start hour): median, n")
for h in sorted(hourly): print(h, f"{st.median(hourly[h]):.3f}", len(hourly[h]))
