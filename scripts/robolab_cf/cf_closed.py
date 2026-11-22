"""Closed-loop P0: A (uncapped), A2 (uncapped repeat), C (cap 1.0, all bodies), each a fresh Pi0.5 server.

usage: cf_closed.py OUT_JSON CLOSED_DIR RECORDING_TASK_DIR
Bit-exactness A vs A2 and A vs recording (initial states, per-env object states); per-env divergence onset
A vs A2 and A vs C; success before/after, both flip directions with Wilson intervals, classes from A's events.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cf_summary import tally  # noqa: E402

DT = 1 / 120


def batch(run_dir):
    task_dir = next(p for p in Path(run_dir).iterdir() if (p / "batch" / "run_0.json").exists())
    return task_dir, json.load(open(task_dir / "batch" / "run_0.json"))


def seg_states(task_dir, seg):
    return np.load(Path(task_dir).joinpath(*Path(seg["dir"]).parts[-3:]) / "states.npz")


def klass(task_dir, seg):
    ev = [json.loads(l) for l in (Path(task_dir).joinpath(*Path(seg["dir"]).parts[-3:]) / "events.jsonl").read_text().splitlines()]
    if any(e["status"] == "confirmed" for e in ev):
        return "confirmed", min(e["onset_substep"] for e in ev if e["status"] == "confirmed")
    if any(e["status"] == "flag" for e in ev):
        return "flagged-only", min(e["onset_substep"] for e in ev if e["status"] == "flag")
    return "no-event", None


def onset(a, b):
    """First substep where two segments' object states differ, and whether they are identical."""
    sa, sb = a["states"], b["states"]
    n = min(len(sa), len(sb))
    diff = np.any(sa[:n] != sb[:n], axis=(1, 2))
    same = bool(len(sa) == len(sb) and not diff.any() and str(a["body_ids"]) == str(b["body_ids"]))
    return same, (int(a["substeps"][int(np.argmax(diff))]) if diff.any() else None), (float(np.abs(sa[:n] - sb[:n]).max()) if n else None)


def initial_equal(d1, d2):
    z1, z2 = np.load(Path(d1) / "batch" / "initial_states.npz"), np.load(Path(d2) / "batch" / "initial_states.npz")
    return {k: bool(z1[k].shape == z2[k].shape and np.array_equal(z1[k], z2[k])) for k in z1.files}


def main(out_json, closed_dir, rec_task_dir):
    closed = Path(closed_dir)
    a_dir, a = batch(closed / "A")
    runs = {"A": (a_dir, a)}
    for name in ("A2", "C"):
        runs[name] = batch(closed / name)
    rec = Path(rec_task_dir)
    rec_run = json.load(open(rec / "batch" / "run_0.json"))
    out = {"initial_states": {"A_vs_A2": initial_equal(a_dir, runs["A2"][0]), "A_vs_C": initial_equal(a_dir, runs["C"][0]),
                              "A_vs_recording": initial_equal(a_dir, rec)}}
    segs = {name: {(s["env"], s["segment"]): s for s in r[1]["segments"]} for name, r in runs.items()}
    rec_segs = {(s["env"], s["segment"]): s for s in rec_run["segments"]}
    envs = []
    for key, sa in sorted(segs["A"].items()):
        c, t0 = klass(a_dir, sa)
        row = {"env": key[0], "segment": key[1], "class": c, "t0": t0, "discarded": sa["discarded"], "success": {"A": bool(sa["success"])}}
        st_a = seg_states(a_dir, sa)
        for name in ("A2", "C"):
            sb = segs[name].get(key)
            row["success"][name] = None if sb is None else bool(sb["success"])
            if sb is not None:
                same, first, mx = onset(st_a, seg_states(runs[name][0], sb))
                row[f"A_vs_{name}"] = {"bit_exact": same, "divergence_onset": first, "max_abs_diff": mx,
                                       "onset_minus_t0": None if first is None or t0 is None else first - t0,
                                       "onset_minus_t0_s": None if first is None or t0 is None else round((first - t0) * DT, 3)}
        sr = rec_segs.get(key)
        if sr is not None:
            same, first, mx = onset(st_a, seg_states(rec, sr))
            row["A_vs_recording"] = {"bit_exact": same, "divergence_onset": first, "max_abs_diff": mx, "recording_success": bool(sr["success"])}
        envs.append(row)
    out["envs"] = envs
    kept = [r for r in envs if not r["discarded"]]
    for name in ("A2", "C"):
        rows = [{"class": r["class"], "before": r["success"]["A"], "after": r["success"][name]} for r in kept if r["success"][name] is not None]
        out[f"A_vs_{name}"] = {"bit_exact_envs": sum(r[f"A_vs_{name}"]["bit_exact"] for r in kept), "envs": len(kept), "tally": tally(rows)}
    Path(out_json).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "envs"}, indent=1))
    for r in envs:
        print(r["env"], r["class"], "t0", r["t0"], "succ", r["success"],
              "A2", {k: r.get("A_vs_A2", {}).get(k) for k in ("bit_exact", "divergence_onset")},
              "C", {k: r.get("A_vs_C", {}).get(k) for k in ("divergence_onset", "onset_minus_t0_s")},
              "rec", {k: r.get("A_vs_recording", {}).get(k) for k in ("bit_exact", "divergence_onset", "recording_success")})


if __name__ == "__main__":
    main(*sys.argv[1:])
