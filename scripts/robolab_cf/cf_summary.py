"""Summarise counterfactual replays as fixed in runs/cf/PREREG.md.

usage: cf_summary.py OUT_JSON ROOT [REF]
ROOT holds <policy>/<task>/<variant>/counterfactual_*.json (variant nocap, cap1000, cap3, cap1, cap1_<B>[+<P>]).
REF is "recording" (default) or a variant tag (e.g. cap1000); per-body variants compare with REF, or with
REF_<bodies> when that replay exists.
"""
import glob
import json
import math
import os
import sys
from pathlib import Path

MIRROR = Path("/path/to/simuguard-robolab/runs-mirror")


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(c - h, 0.0), 3) + 0.0, round(min(c + h, 1.0), 3)]


def load(variant_dir):
    files = glob.glob(os.path.join(variant_dir, "counterfactual_*_runs*.json")) + glob.glob(os.path.join(variant_dir, "replay_*_runs*.json"))
    return json.load(open(files[0])) if files else None


def confirmed_bodies(policy, task, run, seg):
    path = MIRROR / f"formal_{policy}" / task / task / f"run_{run}" / f"env_{seg['env']}" / f"seg_{seg['segment']}" / "events.jsonl"
    out = set()
    for line in path.read_text().splitlines():
        e = json.loads(line)
        if e["status"] == "confirmed":
            out.add(e["bodies"][0])
    return out


def klass(seg):
    if seg["recorded_confirmed"]:
        return "confirmed"
    return "flagged-only" if seg["recorded_flags"] else "no-event"


def tally(rows):
    out = {}
    for name in ("confirmed", "no-event", "flagged-only"):
        r = [x for x in rows if x["class"] == name]
        n = len(r)
        before = sum(x["before"] for x in r)
        after = sum(x["after"] for x in r)
        s = [x for x in r if x["before"]]
        f = [x for x in r if not x["before"]]
        sf = sum(1 for x in s if not x["after"])
        fs = sum(1 for x in f if x["after"])
        out[name] = {
            "n": n,
            "success_before": [before, n, wilson(before, n)],
            "success_after": [after, n, wilson(after, n)],
            "success_change": None if n == 0 else round((after - before) / n, 3),
            "flips_success_to_fail": [sf, len(s), wilson(sf, len(s))],
            "flips_fail_to_success": [fs, len(f), wilson(fs, len(f))],
        }
    return out


def main(out_json, root, ref="recording"):
    root = Path(root)
    result = {"ref": ref, "tasks": {}, "pooled": {}}
    pooled = {}
    for vdir in sorted(root.glob("*/*/*")):
        if not vdir.is_dir() or vdir.parent.name == "closed":
            continue
        policy, task, variant = vdir.parts[-3], vdir.parts[-2], vdir.name
        rep = load(vdir)
        if rep is None:
            continue
        bodies = variant[len("cap1_"):].split("+") if variant.startswith("cap1_") else None
        ref_rep = None
        if ref != "recording":
            alt = vdir.parent / (f"{ref}_{variant[len('cap1_'):]}" if bodies else ref)
            ref_rep = load(alt if alt.exists() else vdir.parent / ref)
        info = {"bit_exact": {}, "first_divergence": {}, "segments_compared": {}}
        rows = []
        for run, r in rep["runs"].items():
            info["bit_exact"][run] = {k: r[k]["bit_exact"] for k in ("objects", "dof_pos")}
            info["first_divergence"][run] = (r.get("first_divergence") or {}).get("substep")
            info["segments_compared"][run] = r["segments_compared"]
            ref_segs = {(s["env"], s["segment"]): s for s in ref_rep["runs"][run]["segments"]} if ref_rep else None
            for s in r["segments"]:
                if s["discarded"] or s["recorded_success"] is None:
                    continue
                c = klass(s)
                if bodies and c == "confirmed" and bodies[0] not in confirmed_bodies(policy, task, run, s):
                    continue
                before = bool(s["recorded_success"]) if ref_segs is None else bool(ref_segs[(s["env"], s["segment"])]["replayed_success"])
                rows.append({"run": run, "env": s["env"], "class": c, "before": before, "after": bool(s["replayed_success"])})
        info["tally"] = tally(rows)
        info["episodes"] = rows
        result["tasks"].setdefault(f"{policy}/{task}", {})[variant] = info
        group = "per-body B" if bodies and len(bodies) == 1 else ("per-body B+P" if bodies else variant)
        pooled.setdefault((policy, group), []).extend(rows)
        pooled.setdefault(("all", group), []).extend(rows)
    for (policy, group), rows in sorted(pooled.items()):
        result["pooled"].setdefault(group, {})[policy] = tally(rows)
    Path(out_json).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
