#!/usr/bin/env python3
"""Make a stopped Pi0.5 x LIBERO campaign re-run what did not finish cleanly.

Drops from every ``episodes.jsonl`` the episodes that ended in an infrastructure error (policy
server timeouts and the like: ``error`` is set) and deletes their segments, plus every segment no
remaining episode refers to (episodes cut off by a stop).  The next ``pi05_campaign.sh`` run on the
same OUT then redoes exactly those episodes.  Run it only while no worker writes to OUT.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dropped_eps, dropped_dirs = [], []
    for task_dir in sorted(Path(args.campaign).glob("*/t*")):
        log = task_dir / "episodes.jsonl"
        records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()] if log.exists() else []
        keep = [r for r in records if not r.get("error")]
        dropped_eps += [f"{r['suite']}:t{r['task_id']}:ep{r['episode']} ({r['error'].strip().splitlines()[-1][:60]})"
                        for r in records if r.get("error")]
        referenced = {Path(r["segment"]).name for r in keep}
        stale = [d for d in sorted((task_dir / "segments").glob("*")) if d.is_dir() and d.name not in referenced]
        dropped_dirs += [str(d) for d in stale]
        if args.dry_run:
            continue
        if len(keep) != len(records):
            tmp = log.with_name(log.name + ".tmp")
            tmp.write_text("".join(json.dumps(r) + "\n" for r in keep))
            tmp.replace(log)
        for d in stale:
            shutil.rmtree(d)
    print(f"{'would drop' if args.dry_run else 'dropped'} {len(dropped_eps)} errored episodes and "
          f"{len(dropped_dirs)} unreferenced segments")
    for line in dropped_eps:
        print("  ", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
