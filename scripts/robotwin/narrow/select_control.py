#!/usr/bin/env python3
"""Draw the matched control episodes by the rule in prereg.md (fixed RNG seed 20261107).

Inputs: the inventory JSON lines (inventory.py), the flagged mapping (flagged_mapping.csv) and the
gripper contact onsets (contact_onsets.py) of every pool and flagged episode.  Output:
control_selection.json with one entry per drawn episode (segment, host, anchor body and substep,
the flagged episode it is matched to).

    python select_control.py --inventory inv1.jsonl inv2.jsonl --mapping flagged_mapping.csv \
        --onsets on1.jsonl on2.jsonl --out control_selection.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random

SEED = 20261107
LEAD, AFTER = 25, 50


def group_of(root: str) -> str:
    if "/hyvla_table1/" in root:
        return "hyvla"
    if "/campaign_20260918T220055Z/default/" in root:
        return root.rsplit("/", 1)[1]
    return "p2" if "/align40/" in root else "p1"


def anchor_for(onsets: dict, body: str, flagged_frac: float) -> tuple[str, int, str] | None:
    """(body, substep, how) of the eligible onset closest in normalised time, or None."""

    length = onsets["length"]
    candidates = [(body, "flagged_body")]
    targets = onsets.get("targets") or []
    if targets and targets[0] != body:
        candidates.append((targets[0], "task_target_fallback"))
    for name, how in candidates:
        eligible = [s for s in onsets["onsets"].get(name, []) if s >= LEAD + 1 and s + AFTER <= length]
        if eligible:
            best = min(eligible, key=lambda s: (abs(s / length - flagged_frac), s))
            return name, best, how
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", nargs="+", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--onsets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    onsets = {}
    for path in args.onsets:
        for line in open(path):
            d = json.loads(line)
            if "error" not in d:
                onsets[d["segment"]] = d
    rows = []
    for path in args.inventory:
        rows += [json.loads(line) for line in open(path)]
    rows = [r for r in rows if "error" not in r and "crosstask4_50seeds_copy" not in r["root"]]
    flagged = [r for r in csv.DictReader(open(args.mapping))
               if r["status"] == "found" and r["task"] != "put_object_cabinet"]
    flagged.sort(key=lambda r: (r["task"], r["group"], int(r["seed"])))

    pool: dict[str, dict[bool, list[dict]]] = {}
    for r in rows:
        if r["n_invalid"] or r["task"] == "put_object_cabinet":
            continue
        if not all(r["files"][k] for k in ("controls.npz", "states.npz", "trace.jsonl.gz")):
            continue
        r["episode_id"] = f"{r['task']}/{group_of(r['root'])}/{r['seed']}"
        pool.setdefault(r["task"], {True: [], False: []})[bool(r["success"])].append(r)

    rng = random.Random(SEED)
    chosen, log = [], []
    for f in flagged:
        task_pool = pool.get(f["task"], {True: [], False: []})
        flagged_onsets = onsets.get(f["segment"])
        if flagged_onsets is None:
            raise SystemExit(f"no onsets for flagged {f['segment']}")
        frac = int(f["onset_substep"]) / flagged_onsets["length"]
        picks = 0
        order = [True, False]
        attempts = 0
        while picks < 2 and attempts < 1000:
            attempts += 1
            want = order[picks] if task_pool[order[picks]] else (not order[picks])
            if not task_pool[want]:
                break
            candidates = sorted(task_pool[want], key=lambda r: r["episode_id"])
            r = rng.choice(candidates)
            task_pool[want].remove(r)
            anchor = anchor_for(onsets[r["segment"]], f["body"], frac) if r["segment"] in onsets else None
            if anchor is None:
                log.append({"discarded": r["episode_id"], "matched_to": f["episode_id"], "reason": "no eligible onset"})
                continue
            body, substep, how = anchor
            chosen.append({"episode_id": r["episode_id"], "task": r["task"], "seed": r["seed"], "host": r["host"],
                           "segment": r["segment"], "recorded_success": bool(r["success"]),
                           "anchor_body": body, "anchor_substep": substep, "anchor_how": how,
                           "anchor_frac": round(substep / onsets[r["segment"]]["length"], 4),
                           "matched_to": f["episode_id"], "matched_frac": round(frac, 4),
                           "matched_body": f["body"]})
            picks += 1
    out = {"rule": "prereg.md (seed %d)" % SEED, "n": len(chosen),
           "n_success": sum(c["recorded_success"] for c in chosen), "control": chosen, "log": log}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"{len(chosen)} control episodes ({out['n_success']} S), {len(log)} discarded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
