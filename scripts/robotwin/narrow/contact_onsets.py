#!/usr/bin/env python3
"""Gripper-finger contact onsets per body, from recorded traces (control anchors, see prereg.md).

An onset is a substep s at which the body touches a gripper finger link (``f[lr]_link7`` /
``f[lr]_link8``) and touched none in the preceding ``--gap`` substeps (default 25 = 0.1 s).
Plain Python 3.  One JSON line per segment::

    python contact_onsets.py seg_dir [seg_dir ...] > onsets.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

FINGER = re.compile(r"^link:.*f[lr]_link[78]$")


def onsets(segment: Path, gap: int) -> dict:
    last_touch: dict[str, int] = {}
    found: dict[str, list[int]] = {}
    length = 0
    with gzip.open(segment / "trace.jsonl.gz", "rt") as fh:
        for line in fh:
            frame = json.loads(line)
            s = int(frame["substep"])
            length = max(length, s)
            touching = set()
            for c in frame.get("contacts", []):
                a, b = c["a"], c["b"]
                if FINGER.match(a) and not b.startswith("link:"):
                    touching.add(b)
                elif FINGER.match(b) and not a.startswith("link:"):
                    touching.add(a)
            for body in touching:
                prev = last_touch.get(body)
                if prev is None or s - prev > gap:
                    found.setdefault(body, []).append(s)
                last_touch[body] = s
    targets = []
    try:
        targets = json.loads((segment / "manifest.json").read_text())["task_ground_truth"].get("targets") or []
    except Exception:  # noqa: BLE001
        pass
    return {"segment": str(segment), "length": length, "targets": targets, "onsets": found}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("segments", nargs="+")
    ap.add_argument("--gap", type=int, default=25)
    args = ap.parse_args()
    for seg in args.segments:
        try:
            print(json.dumps(onsets(Path(seg), args.gap)), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"segment": seg, "error": repr(exc)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
