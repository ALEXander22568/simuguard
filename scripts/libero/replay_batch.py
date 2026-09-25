#!/usr/bin/env python3
"""Exact-replay check over a campaign: check_replay.py on a selection of episodes, each in its own
fresh process, and one table.

Selection per suite: every failed episode, every episode with a policy-phase event, and the first
``--per-suite`` successes; ``--all`` takes every episode.  ``--drop-warmstart`` repeats each
replay without the recorded warm starts (robosuite 1.4 needs them).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve()


def select(root: Path, per_suite: int, take_all: bool) -> list[dict]:
    chosen = []
    for suite_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("libero")):
        eps = []
        for log in sorted(suite_dir.glob("t*/episodes.jsonl")):
            eps += [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        eps = [e for e in eps if not e.get("error")]
        if take_all:
            chosen += eps
            continue
        settle = lambda e: int(e.get("settle_substeps") or 0)  # noqa: E731
        special = [e for e in eps if not e["success"] or any(ev["onset"] > settle(e) for ev in e.get("events", []))]
        plain = [e for e in eps if e not in special][:per_suite]
        chosen += special + plain
    return chosen


def run(segment: str, drop: bool) -> dict:
    cmd = [sys.executable, str(HERE.parent / "check_replay.py"), segment] + (["--drop-warmstart"] if drop else [])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = Path(segment) / ("replay_check_nowarm.json" if drop else "replay_check.json")
    if proc.returncode != 0 or not out.is_file():
        return {"segment": segment, "error": (proc.stderr or proc.stdout)[-500:]}
    return json.loads(out.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("--per-suite", type=int, default=3)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--drop-warmstart", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    episodes = select(Path(args.campaign), args.per_suite, args.all)
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda e: run(e["segment"], args.drop_warmstart), episodes))
    rows = []
    for ep, res in zip(episodes, results):
        row = {"suite": ep["suite"], "task_id": ep["task_id"], "episode": ep["episode"], "success": ep["success"],
               "steps": ep["steps"], **{k: res.get(k) for k in (
                   "model_identical", "substeps_replayed", "max_abs_pos_err_m", "full_state_snapshots_compared",
                   "full_state_max_abs_diff", "first_divergence", "error")}}
        rows.append(row)
        print(f"{row['suite']:15s} t{row['task_id']} ep{row['episode']} success={row['success']!s:5s} "
              f"substeps {row['substeps_replayed']}  model {row['model_identical']}  max pos err {row['max_abs_pos_err_m']}"
              f"  full state {row['full_state_max_abs_diff']} over {row['full_state_snapshots_compared']} snapshots"
              + (f"  ERROR {row['error']}" if row.get("error") else ""))
    ok = [r for r in rows if not r.get("error")]
    exact = [r for r in ok if r["max_abs_pos_err_m"] == 0.0 and r["full_state_max_abs_diff"] in (0.0, None)]
    print(f"\n{len(exact)}/{len(rows)} episodes replay bit-exactly "
          f"({sum(r['substeps_replayed'] or 0 for r in ok)} substeps, "
          f"{sum(r['full_state_snapshots_compared'] or 0 for r in ok)} full-state snapshots); "
          f"worst position error {max((r['max_abs_pos_err_m'] for r in ok), default=None)} m")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
