"""Divergence onset vs event time per env, for counterfactual replays.

usage: cf_onset.py OUT_JSON ROOT
For each <policy>/<task>/<variant> replay: per env, class, t0 = onset_substep of the confirmed event (for per-body
variants, the confirmed event whose first body is the capped B; else the first confirmed event), the first flag
onset, the divergence onset from the recording, onset - t0 in substeps and seconds, and the outcome flip.
"""
import glob
import json
import sys
from pathlib import Path

MIRROR = Path("/path/to/simuguard-robolab/runs-mirror")
DT = 1 / 120


def events(policy, task, run, seg):
    path = MIRROR / f"formal_{policy}" / task / task / f"run_{run}" / f"env_{seg['env']}" / f"seg_{seg['segment']}" / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def klass(seg):
    if seg["recorded_confirmed"]:
        return "confirmed"
    return "flagged-only" if seg["recorded_flags"] else "no-event"


def main(out_json, root):
    root = Path(root)
    out = {}
    for vdir in sorted(root.glob("*/*/*")):
        files = glob.glob(str(vdir / "counterfactual_*_runs*.json")) + glob.glob(str(vdir / "replay_*_runs*.json"))
        if not files or vdir.parent.name == "closed":
            continue
        policy, task, variant = vdir.parts[-3], vdir.parts[-2], vdir.name
        rep = json.load(open(files[0]))
        tag = variant.split("_", 1)
        bodies = tag[1].split("+") if len(tag) == 2 and tag[0] in ("cap1", "cap1000") else None
        rows = []
        for run, r in rep["runs"].items():
            for s in r["segments"]:
                if s["discarded"]:
                    continue
                ev = events(policy, task, run, s)
                conf = [e for e in ev if e["status"] == "confirmed"]
                if bodies:
                    conf_b = [e for e in conf if e["bodies"][0] == bodies[0]]
                else:
                    conf_b = conf
                t0 = min((e["onset_substep"] for e in conf_b), default=None)
                first_ev = min((e["onset_substep"] for e in ev), default=None)
                onset = (s.get("first_divergence") or {}).get("substep")
                ref = t0 if t0 is not None else first_ev
                rows.append({
                    "run": run, "env": s["env"], "class": klass(s),
                    "confirmed_on_capped_body": bool(conf_b) if bodies else None,
                    "t0": t0, "first_event_onset": first_ev, "divergence_onset": onset,
                    "onset_minus_t0": None if onset is None or ref is None else onset - ref,
                    "onset_minus_t0_s": None if onset is None or ref is None else round((onset - ref) * DT, 3),
                    "relative_to": "t0" if t0 is not None else ("first_event" if first_ev is not None else None),
                    "flipped": bool(s["outcome_flipped"]),
                })
        out.setdefault(f"{policy}/{task}", {})[variant] = {"bodies": bodies, "envs": rows}
    Path(out_json).write_text(json.dumps(out, indent=1))
    for key, variants in out.items():
        for variant, v in variants.items():
            print(key, variant)
            for r in sorted(v["envs"], key=lambda r: (r["class"], r["env"])):
                print(f"   env {r['env']} {r['class']:12s} t0={r['t0']} first_ev={r['first_event_onset']} onset={r['divergence_onset']} "
                      f"d={r['onset_minus_t0']} ({r['onset_minus_t0_s']} s, vs {r['relative_to']}) flip={r['flipped']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
