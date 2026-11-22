#!/usr/bin/env python3
"""Start a second queue on the same run dir once the first queue has nothing left for some GPUs.

usage: python3 start_after.py --run-dir RUN --free-gpus 4 5 --drain 4 5 6 -- CMD...

Polls RUN/STATUS.json (written by the first rd_queue) every --every s.  When no job is pending except those of
held policies (CONTROL.json hold_policies) and the lanes of --free-gpus are idle, it writes ``drain`` for the
--drain GPUs into CONTROL.json (lanes stop after their current job; held policies stay held), waits until
the --free-gpus lanes report ``off``, then runs CMD (a setsid'd child whose output goes to RUN/start_after.out).
Used for InternVLA_A1 (sim on GPU 4, server on GPU 5) after the other models on the 24 GB box.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path

KST = dt.timezone(dt.timedelta(hours=9))


def log(run: Path, msg: str) -> None:
    line = f"{dt.datetime.now(KST).strftime('%m-%d %H:%M:%S')} start_after: {msg}"
    print(line, flush=True)
    with (run / "queue.log").open("a") as fh:
        fh.write(line + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--free-gpus", nargs="+", type=int, required=True)
    ap.add_argument("--drain", nargs="+", type=int, required=True)
    ap.add_argument("--every", type=float, default=300)
    ap.add_argument("cmd", nargs="+")
    args = ap.parse_args()
    run = Path(args.run_dir).resolve()
    log(run, f"waiting: free GPUs {args.free_gpus}, then {' '.join(args.cmd)}")
    drained = False
    while True:
        time.sleep(args.every)
        try:
            st = json.loads((run / "STATUS.json").read_text())
            ctl = json.loads((run / "CONTROL.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        hold = set(ctl.get("hold_policies", []))
        lanes = {ln["gpu"]: ln for ln in st["lanes"]}
        if not drained:
            pending = [j["job"] for j in st["jobs"] if j["state"] == "pending" and j["remaining"] > 0
                       and j["job"].split("/")[0] not in hold]
            if pending or any(lanes.get(g, {}).get("job") for g in args.free_gpus):
                continue
            ctl["disabled_gpus"] = {**ctl.get("disabled_gpus", {}), **{str(g): "drain" for g in args.drain}}
            tmp = run / "CONTROL.json.tmp"
            tmp.write_text(json.dumps(ctl))
            tmp.replace(run / "CONTROL.json")
            drained = True
            log(run, f"nothing pending outside {sorted(hold)}; CONTROL drain {args.drain}")
            continue
        if all("off" in lanes.get(g, {}).get("phase", "") for g in args.free_gpus):
            out = (run / "start_after.out").open("a")
            subprocess.Popen(args.cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True, cwd=str(run.parent.parent))
            log(run, "lanes off; started the second queue")
            return


if __name__ == "__main__":
    main()
