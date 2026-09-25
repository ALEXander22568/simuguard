#!/usr/bin/env python3
"""Tile head-camera frames around every recorded event of a RoboDojo episode (stdlib + ffmpeg).

RoboDojo streams one video frame per observation; XPolicyLab's loop observes once before the first
action and after every action, so video frame k is (to within one frame) the state after control step k.
For each event (default: confirmed contact ejections) this writes ``<segment>/event_frames/<event>.png``,
a 4x2 grid of the head camera at control steps onset-8, -4, -2, -1, 0, +1, +3, +8.

Usage::

    python3 event_frames.py SEGMENT_DIR VIDEO_DIR [--ffmpeg PATH] [--all-statuses] [--camera cam_head]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
from pathlib import Path

OFFSETS = (-8, -4, -2, -1, 0, 1, 3, 8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("segment")
    ap.add_argument("video_dir")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--camera", default="cam_head")
    ap.add_argument("--episode-index", type=int, default=None, help="RoboDojo video index (default: from the layout order)")
    ap.add_argument("--all-statuses", action="store_true", help="also candidates/rejected events and flags")
    args = ap.parse_args()
    seg = Path(args.segment)
    summary = json.loads((seg / "summary.json").read_text())
    events = summary.get("events", [])
    if not args.all_statuses:
        events = [e for e in events if e.get("status") == "confirmed"]
    videos = sorted(glob.glob(str(Path(args.video_dir) / f"episode_*_{args.camera}_*.mp4")))
    if args.episode_index is not None:
        videos = [v for v in videos if f"episode_{args.episode_index:07d}_" in v]
    if not videos:
        print("no video found")
        return 1
    video = videos[0]
    out = seg / "event_frames"
    out.mkdir(exist_ok=True)
    for event in events:
        step = int(event.get("control_step", 0))
        frames = [max(0, step + o) for o in OFFSETS]
        select = "+".join(f"eq(n\\,{f})" for f in frames)
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", event["event_id"])[-120:] + ".png"
        cmd = [args.ffmpeg, "-v", "error", "-y", "-i", video, "-vf",
               f"select={select},scale=320:-1,tile=4x2", "-frames:v", "1", str(out / name)]
        subprocess.run(cmd, check=False)
        print(f"{event['detector']} {event['status']} onset substep {event['onset_substep']} control step {step} "
              f"bodies {event.get('bodies')} -> {out / name} (frames {frames})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
