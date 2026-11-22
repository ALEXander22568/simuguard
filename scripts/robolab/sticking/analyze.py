"""analyze.py: join the frozen sticking outputs (out/out_<box>_*.jsonl) with episodes.json and the review labels.

Writes results.json (all numbers), flagged.csv (one row per flagged episode) and prints the tables used in report.md.
"""
import csv
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RA = HERE.parent

eps = json.loads((RA / "episodes.json").read_text())
ep = {(x["box"], x["seg_dir"]): x for x in eps}
res = {}
errors = []
for f in sorted(glob.glob(str(HERE / "out" / "out_*_*.jsonl"))):
    box = Path(f).name.split("_")[1]
    for line in open(f):
        r = json.loads(line)
        if r.get("error"):
            errors.append((box, r["seg_dir"], r["error"]))
        res[(box, r["seg_dir"])] = r
missing = [k for k in ep if k not in res]
extra = [k for k in res if k not in ep]


def ej(x):
    return x["events"].get("contact_ejection", {}).get("confirmed", 0) > 0


rows = []
for k, x in ep.items():
    r = res.get(k)
    rows.append(dict(x=x, r=r, stick=bool(r and r.get("flag")), ej=ej(x)))

out = {"n_episodes": len(ep), "n_results": len(res), "missing": len(missing), "extra": len(extra),
       "errors": errors, "no_fj": sum(1 for r in res.values() if r.get("has_fj") is False)}
per = {}
for pol in ("pi05", "gr00t", "all"):
    R = [z for z in rows if pol == "all" or z["x"]["policy"] == pol]
    n = len(R); f = [z for z in R if z["stick"]]; nf = [z for z in R if not z["stick"]]
    both = [z for z in f if z["ej"]]
    ejn = sum(z["ej"] for z in R)
    tasks = Counter(z["x"]["task"] for z in f)
    per[pol] = dict(n=n, flag=len(f), rate=len(f) / n if n else None, ej_flag=ejn, both=len(both),
                    stick_only=len(f) - len(both),
                    succ_flag=sum(z["x"]["success_rl"] for z in f), succ_rate_flag=(sum(z["x"]["success_rl"] for z in f) / len(f)) if f else None,
                    succ_other=sum(z["x"]["success_rl"] for z in nf), succ_rate_other=sum(z["x"]["success_rl"] for z in nf) / len(nf),
                    n_tasks_flag=len(tasks), top_tasks=tasks.most_common(10),
                    fingers_open=sum(1 for z in f if any(w["flag"] and w["fj_max"] is not None and w["fj_max"] < 0.1 for w in z["r"]["windows"])))
out["per_policy"] = per

# review items
items = json.loads((RA / "review_items.json").read_text())
labels = json.loads((RA / "review" / "labels_user_20261110b.json").read_text())["labels"]
key5 = {(x["box"], x["policy"], x["task"], x["run"], x["env"]): k for k, x in ep.items()}
rev = []
for it in items:
    k = key5.get((it["box"], it["policy"], it["task"], it["run"], it["env"]))
    r = res.get(k) if k else None
    lab = labels.get(it["id"], {})
    rev.append(dict(id=it["id"], set=it["set"], box=it["box"], policy=it["policy"], task=it["task"], run=it["run"], env=it["env"],
                    found=r is not None, stick=bool(r and r.get("flag")), label=lab,
                    bodies=(r or {}).get("flag_bodies"), onset=(r or {}).get("onset")))
out["review"] = {s: dict(n=sum(1 for v in rev if v["set"] == s), found=sum(1 for v in rev if v["set"] == s and v["found"]),
                         fire=[v for v in rev if v["set"] == s and v["stick"]]) for s in ("R", "P")}
out["R002"] = [v for v in rev if v["id"] == "R002"][0]

# flagged list
flagged = []
for z in rows:
    if not z["stick"]:
        continue
    x, r = z["x"], z["r"]
    w = [w for w in r["windows"] if w["flag"]]
    w.sort(key=lambda a: a["start"])
    best = max(w, key=lambda a: a["dur_s"])
    flagged.append(dict(box=x["box"], policy=x["policy"], task=x["task"], run=x["run"], env=x["env"], success=x["success_rl"],
                        ejection_flag=z["ej"], bodies="|".join(r["flag_bodies"]), onset=r["onset"], n_flag_windows=len(w),
                        longest_s=best["dur_s"], longest_disp_m=best["disp_m"], longest_links="|".join(best["links"]),
                        longest_fj_max=best["fj_max"], longest_min_sep_m=best["min_sep_m"], seg_dir=x["seg_dir"]))
flagged.sort(key=lambda a: (a["box"], a["policy"], a["task"], a["run"], a["env"]))
with open(HERE / "flagged.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(flagged[0].keys()) if flagged else ["box"])
    wr.writeheader(); wr.writerows(flagged)
out["n_flagged"] = len(flagged)
# windows that came close (descriptive only): unflagged episodes with a window >= 2 s but disp < 0.10, or disp >= 0.10 but dur in [1,2)
near = Counter()
for z in rows:
    if z["stick"] or not z["r"]:
        continue
    for w in z["r"]["windows"]:
        if w["dur_s"] >= 2.0:
            near["dur>=2s_disp<0.10"] += 1; break
        if 1.0 <= w["dur_s"] < 2.0 and w["disp_m"] >= 0.10:
            near["1<=dur<2s_disp>=0.10"] += 1; break
out["near_miss_episodes"] = dict(near)
(HERE / "results.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k != "review"}, indent=1, default=str)[:6000])
print("REVIEW R fire:", [(v["id"], v["task"], v["bodies"], v["label"]) for v in out["review"]["R"]["fire"]])
print("REVIEW P fire:", [(v["id"], v["task"], v["bodies"], v["label"]) for v in out["review"]["P"]["fire"]])
print({s: (out["review"][s]["n"], out["review"][s]["found"], len(out["review"][s]["fire"])) for s in ("R", "P")})
