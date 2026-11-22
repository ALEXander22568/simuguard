import json, glob
M = "runs-mirror"
for pt in ["gr00t/BananaInBowlTask", "pi05/BananasInCrateTask", "pi05/BananasInBinOneMoreTask"]:
    pol, task = pt.split("/")
    for v in ["cap1000", "cap1"]:
        d = json.load(open(glob.glob(f"runs/cf/p0/{pt}/{v}/counterfactual_*_runs0.json")[0]))
        for s in d["runs"]["0"]["segments"]:
            env, seg = s["env"], s["segment"]
            ev = [json.loads(l) for l in open(f"{M}/formal_{pol}/{task}/{task}/run_0/env_{env}/seg_{seg}/events.jsonl")]
            conf = [(e.get("substep", e.get("step")), e["bodies"][:2]) for e in ev if e["status"] == "confirmed"]
            flags = [e.get("substep", e.get("step")) for e in ev]
            fd = s.get("first_divergence") or {}
            print(pt, v, "env", env, "flip", s["outcome_flipped"], "div", fd, "conf", conf[:2], "nev", len(ev), "first_ev", flags[:1])
