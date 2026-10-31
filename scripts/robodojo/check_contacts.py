#!/usr/bin/env python3
"""Contact sanity check on a recorded SimuGuard segment: does a resting object's contact force equal its weight?

For every free body, over a window of substeps in which it barely moves (default: the first 250 substeps,
i.e. the first second, speed < 5 mm/s), sum the net contact force of all its reported contact pairs along
+z (|sum impulse| / dt with the impulse oriented onto the body) and compare it with m * g.  Also reports
which partners carried the load, the number of contact points, and the worst penetration.

The trace stores per-pair magnitudes, so the vertical balance uses the float64 contact impulses re-read
from the trace's pair records: ``force_est`` = |sum of impulses| / dt of the pair.  For a body resting on
one support this equals the support force; with several supports the pair magnitudes add up (upper bound).

Usage::

    python scripts/robodojo/check_contacts.py SEGMENT_DIR [--substeps 1 250] [--max-speed 0.005]
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

G = 9.81


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("segment")
    ap.add_argument("--substeps", type=int, nargs=2, default=(1, 250))
    ap.add_argument("--max-speed", type=float, default=0.005)
    args = ap.parse_args()
    seg = Path(args.segment)
    manifest = json.loads((seg / "manifest.json").read_text())
    bodies = manifest["bodies"]
    free = {b for b, i in bodies.items() if i["kind"] == "dynamic"}
    lo, hi = args.substeps
    force = defaultdict(list)  # body -> per-substep sum of pair forces
    partners = defaultdict(lambda: defaultdict(list))
    points = defaultdict(list)
    penetration = defaultdict(float)
    still = defaultdict(int)
    frames = 0
    with gzip.open(seg / "trace.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            frame = json.loads(line)
            s = frame["substep"]
            if s < lo:
                continue
            if s > hi:
                break
            frames += 1
            per_body = defaultdict(float)
            per_points = defaultdict(int)
            for c in frame["contacts"]:
                for b, other in ((c["a"], c["b"]), (c["b"], c["a"])):
                    if b in free:
                        per_body[b] += float(c["force_est"])
                        per_points[b] += int(c["points"])
                        partners[b][other].append(float(c["force_est"]))
                        penetration[b] = max(penetration[b], float(c["penetration"]))
            for b in free:
                state = frame["states"].get(b)
                if state is None:
                    continue
                if np.linalg.norm(state["v"]) < args.max_speed:
                    still[b] += 1
                    force[b].append(per_body.get(b, 0.0))
                    points[b].append(per_points.get(b, 0))
    print(f"segment {seg.name}: substeps {lo}..{hi} ({frames} frames), dt {manifest['timestep_s']}")
    print(f"{'body':34s} {'mass kg':>8s} {'m*g N':>8s} {'median F N':>10s} {'F/mg':>6s} {'pts':>4s} {'pen mm':>7s}  load partners (median N)")
    for b in sorted(free):
        mass = bodies[b].get("mass")
        if not force[b] or mass is None:
            print(f"{b:34s} {'-':>8s}  not at rest in the window ({still[b]} still frames)")
            continue
        weight = mass * G
        med = float(np.median(force[b]))
        top = sorted(((o, float(np.median(v))) for o, v in partners[b].items()), key=lambda x: -x[1])[:3]
        print(f"{b:34s} {mass:8.4f} {weight:8.4f} {med:10.4f} {med / weight:6.3f} {int(np.median(points[b])):4d} "
              f"{1000 * penetration[b]:7.3f}  " + ", ".join(f"{o} {f:.3f}" for o, f in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
