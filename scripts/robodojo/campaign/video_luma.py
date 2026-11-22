#!/usr/bin/env python3
"""Mean luma of every camera video of each recorded episode (black-camera check for batched runs).

usage: python video_luma.py RUN_DIR > luma.json

Reads every ``*/*/episodes.jsonl`` under RUN_DIR; for each counted episode it globs the official videos
``episode_<detail_index>_cam_<camera>_<outcome>.mp4`` in the record's ``video_dir`` and samples one frame
per second with ffmpeg ``signalstats``: mean / min YAVG and the share of frames with YAVG < 16 ("dark").
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


def luma(path: Path) -> dict:
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf",
                          "fps=1,signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-", "-f", "null", "-"],
                         capture_output=True, text=True)
    vals = [float(v) for v in re.findall(r"lavfi\.signalstats\.YAVG=([0-9.]+)", out.stdout + out.stderr)]
    if not vals:
        return {"frames": 0, "error": (out.stderr or "")[-300:]}
    return {"frames": len(vals), "mean_yavg": round(sum(vals) / len(vals), 1), "min_yavg": round(min(vals), 1),
            "dark_share": round(sum(v < 16 for v in vals) / len(vals), 3)}


def main() -> None:
    root = Path(sys.argv[1])
    rows = []
    for log in sorted(root.glob("*/*/episodes.jsonl")):
        for line in log.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("status") != "counted":
                continue
            vids = sorted(Path(rec["video_dir"]).glob(f"episode_{int(rec['detail_index']):07d}_cam_*.mp4"))
            cams = {re.sub(r"_(success|fail)$", "", v.stem.split("_cam_", 1)[1]): luma(v) for v in vids}
            rows.append({"task": rec["task"], "layout": rec["layout"], "env_idx": rec.get("env_idx"),
                         "num_envs": rec.get("num_envs"), "cameras": cams})
    dark = [(r["layout"], r["env_idx"], c) for r in rows for c, s in r["cameras"].items()
            if not s.get("frames") or s["dark_share"] > 0.5]
    print(json.dumps({"episodes": len(rows), "dark_or_missing": dark, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
