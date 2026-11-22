#!/usr/bin/env python3
"""Flip counts and replay scores of the narrow-intervention study (prereg.md).

Reads every report written by narrow_replay.py (copied off the hosts), the flagged mapping, the
inventory (for each task's N and S) and the control selection.  Writes:

* <out>.csv            one row per (condition, cap, scope)
* <out>_episodes.csv   one row per (episode, condition)
* <out>.md             the tables with the open-loop caveat

    python narrow_table.py --reports dir1 dir2 --mapping flagged_mapping.csv --inventory inv*.jsonl \
        --control control_selection.json --out results/narrow
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

FIVE = ["place_can_basket", "stack_bowls_three", "place_object_basket", "place_bread_basket", "stack_bowls_two"]
CONDITIONS = ["broad", "body", "partner", "resolved"]
CAPS = [1.0, 0.5, 2.0]
CAVEAT = ("Open-loop replay replays the recorded actions; the policy cannot react to a changed state. "
          "It therefore overestimates S->F (a success whose state is perturbed cannot be corrected by the policy) "
          "and underestimates F->S (a rescued state is still followed by the actions recorded after the failure).")


def wilson(k, n, z=1.959964):
    """95% Wilson score interval of k successes in n trials, as (lo, hi); None if n == 0."""

    if not n:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def fmt_rate(k, n):
    if not n:
        return f"{k}/{n}"
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}%, {100 * lo:.0f}-{100 * hi:.0f})"


def load_reports(dirs):
    reports = {}
    for d in dirs:
        # A report written by the replay child whose batch parent was killed before it could attach the
        # job metadata has no "job" key; take the job from the run's jobs.jsonl, matched by file name.
        jobs = {}
        for jp in Path(d).rglob("jobs.jsonl"):
            for line in jp.read_text().splitlines():
                if line.strip():
                    j = json.loads(line)
                    jobs[(jp.parent, f"{j['key']}__{j['condition'].replace('@', '_')}")] = j
        for p in Path(d).rglob("*.json"):
            if p.name in ("summary.json",) or "__" not in p.name:
                continue
            r = json.loads(p.read_text())
            job = r.get("job") or jobs.get((p.parent, p.stem)) or {}
            if "episode_id" not in job:
                continue
            reports[(job["kind"], job["episode_id"], r["condition"])] = r
    return reports


def valid(r):
    if r["kind"] == "baseline":
        return bool(r.get("bit_identical"))
    return bool(r.get("prefix_bit_identical"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", nargs="+", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--inventory", nargs="+", required=True)
    ap.add_argument("--control")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="LingBot-VA")
    ap.add_argument("--note", default="`put_object_cabinet` (3 flagged, not bit-exact) and the 10 pass-2 episodes "
                                      "without control logs are not included.")
    ap.add_argument("--section", action="store_true",
                    help="write the md as one '## <policy>' section (to append to another policy's narrow.md)")
    args = ap.parse_args()

    inv = []
    for path in args.inventory:
        inv += [json.loads(line) for line in open(path)]
    inv = [r for r in inv if "error" not in r and "crosstask4_50seeds_copy" not in r["root"]]
    N, S = defaultdict(int), defaultdict(int)
    for r in inv:
        N[r["task"]] += 1
        S[r["task"]] += bool(r["success"])
    flagged = [r for r in csv.DictReader(open(args.mapping)) if r["status"] == "found" and r["task"] != "put_object_cabinet"]
    control = json.load(open(args.control))["control"] if args.control else []
    reports = load_reports(args.reports)
    tasks_all = sorted({r["task"] for r in flagged})
    scopes = [(t, [t]) for t in FIVE] + [("five_tasks", FIVE), ("all_analysed", tasks_all)]

    sets = {"flagged": [{"episode_id": r["episode_id"], "task": r["task"], "recorded": r["recorded"] == "S"} for r in flagged],
            "control": [{"episode_id": p["episode_id"], "task": p["task"], "recorded": p["recorded_success"]} for p in control]}
    rows, ep_rows = [], []
    for kind in ("flagged", "control"):
        for cap in CAPS:
            for cond in CONDITIONS:
                if kind == "control" and cond == "broad":
                    continue
                name = f"{cond}@{cap}"
                per_ep = {}
                for e in sets[kind]:
                    r = reports.get((kind, e["episode_id"], name))
                    status = "missing" if r is None else ("invalid_prefix" if not valid(r) else "ok")
                    out = None if status != "ok" else bool(r["success"])
                    per_ep[e["episode_id"]] = (e, status, out, r)
                    if r is not None:
                        ep_rows.append({
                            "set": kind, "episode_id": e["episode_id"], "task": e["task"], "condition": name,
                            "recorded": "S" if e["recorded"] else "F", "replayed": "" if out is None else ("S" if out else "F"),
                            "status": status, "flip": "" if out is None else (
                                "F->S" if out and not e["recorded"] else "S->F" if (not out and e["recorded"]) else ""),
                            "anchor_substep": r.get("anchor_substep"), "anchor_body": r.get("anchor_body"),
                            "window_start": r.get("window_start_substep", r.get("trigger_substep")),
                            "window_last": r.get("window_last_substep"), "window_end_reason": r.get("window_end_reason"),
                            "capped_bodies": ";".join(r.get("capped_bodies") or []), "partner_status": r.get("partner_status"),
                            "restored_exactly": r.get("restored_exactly"),
                            "replay_invalid_after_window": (r.get("replay_gravity_stage") or {}).get("n_after_window"),
                            "wall_s": r.get("wall_s")})
                if not any(s != "missing" for _, s, _, _ in per_ep.values()):
                    continue
                for scope, tasks in scopes:
                    # a headline task without episodes in this set still gets a row (replay score = official)
                    eps = [v for v in per_ep.values() if v[0]["task"] in tasks]
                    ok = [v for v in eps if v[1] == "ok"]
                    nF = sum(not v[0]["recorded"] for v in ok)
                    nS = sum(v[0]["recorded"] for v in ok)
                    fs = sum(1 for v in ok if v[2] and not v[0]["recorded"])
                    sf = sum(1 for v in ok if (not v[2]) and v[0]["recorded"])
                    n_tot, s_tot = sum(N[t] for t in tasks), sum(S[t] for t in tasks)
                    w_fs, w_sf = wilson(fs, nF), wilson(sf, nS)
                    rows.append({
                        "set": kind, "condition": cond, "cap_mps": cap, "scope": scope,
                        "episodes": len(eps), "valid": len(ok), "missing": sum(v[1] == "missing" for v in eps),
                        "invalid_prefix": sum(v[1] == "invalid_prefix" for v in eps),
                        "recorded_F": nF, "F->S": fs, "F->S_rate": round(fs / nF, 3) if nF else "",
                        "F->S_wilson95_lo": round(w_fs[0], 3) if w_fs else "", "F->S_wilson95_hi": round(w_fs[1], 3) if w_fs else "",
                        "recorded_S": nS, "S->F": sf, "S->F_rate": round(sf / nS, 3) if nS else "",
                        "S->F_wilson95_lo": round(w_sf[0], 3) if w_sf else "", "S->F_wilson95_hi": round(w_sf[1], 3) if w_sf else "",
                        "N": n_tot, "S": s_tot, "official": round(100 * s_tot / n_tot, 1),
                        "replay_score": round(100 * (s_tot + fs - sf) / n_tot, 1)})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(f"{out}_episodes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ep_rows[0]))
        w.writeheader()
        w.writerows(ep_rows)

    h = "#" if args.section else ""
    title = (f"## {args.policy}" if args.section
             else f"# RoboTwin narrow intervention and control ({args.policy}, open-loop replay)")
    lines = [title, "",
             f"> **Caveat.** {CAVEAT}", "",
             "Flip = replayed outcome differs from the recorded outcome. Rates: F->S over recorded failures, "
             "S->F over recorded successes, among replays that pass the prefix bit-identity gate; "
             "in brackets the rate and its 95% Wilson interval. "
             "Replay score = (S + #F->S - #S->F) / N per scope (paper convention). " + args.note, ""]
    for scope_set in (("five_tasks", "all_analysed"), tuple(FIVE)):
        title = "Totals" if scope_set[0] == "five_tasks" else "Five headline tasks"
        lines += [f"{h}## {title}", "",
                  "| set | condition | cap | scope | valid/eps | F->S | S->F | official | replay score |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["scope"] not in scope_set:
                continue
            fs = fmt_rate(r["F->S"], r["recorded_F"])
            sf = fmt_rate(r["S->F"], r["recorded_S"])
            lines.append(f"| {r['set']} | {r['condition']} | {r['cap_mps']} | {r['scope']} | {r['valid']}/{r['episodes']} | "
                         f"{fs} | {sf} | {r['official']} | {r['replay_score']} |")
        lines.append("")
    lines += [f"> **Caveat.** {CAVEAT}", "", f"Per-episode rows: `{out.name}_episodes.csv`."]
    Path(f"{out}.md").write_text("\n".join(lines) + "\n")
    print(f"{out}.csv  {len(rows)} rows; {len(ep_rows)} episode rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
