#!/usr/bin/env python3
"""Per-task sanity check of a run directory against the official RoboDojo sim numbers (stdlib only).

usage: python3 sanity_vs_official.py RUN_DIR [--json OUT]

For every RUN/<policy>/<task>/episodes.jsonl: counted episodes, mean official score (x100) and
success rate, next to the leaderboard's (score, success %) for that policy and task (official:
150 episodes per task, 3 eval seeds; ours: eval seed 0).  ``z`` is (ours - official) success rate
over its binomial standard error at our n, a rough flag for a model that is clearly off.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# RoboDojo leaderboard, arx_x5 sim: task -> (mean score, success %)
OFFICIAL = {
    "GR00T_N17": {"hang_mugs": (1.8, 0), "sweep_blocks": (0, 0), "classify_objects": (2.77, 0), "fill_pen_holder": (7.4, 1.33),
                  "stack_bowls": (22.53, 17.33), "insert_tubes": (3.47, 0), "cover_blocks": (0.97, 0)},
    "Pi_05": {"hang_mugs": (9.53, 0), "sweep_blocks": (0, 0), "classify_objects": (24.67, 12.67), "fill_pen_holder": (23.23, 7.33),
              "stack_bowls": (76, 72), "insert_tubes": (17.87, 3.33), "cover_blocks": (19.07, 13.33)},
    "Pi_0": {"hang_mugs": (2.13, 0), "sweep_blocks": (0, 0), "classify_objects": (0.9, 0), "fill_pen_holder": (2.97, 0),
             "stack_bowls": (32.67, 26.67), "insert_tubes": (6.53, 0), "cover_blocks": (13.87, 6)},
    "InternVLA_A1": {"hang_mugs": (1.4, 0), "sweep_blocks": (0, 0), "classify_objects": (0.8, 0), "fill_pen_holder": (0.4, 0),
                     "stack_bowls": (31.67, 26.67), "insert_tubes": (5.2, 0), "cover_blocks": (1.93, 0.67)},
    "Xiaomi_Robotics_0": {"hang_mugs": (5.4, 0), "sweep_blocks": (0, 0), "classify_objects": (3.53, 0.67),
                          "fill_pen_holder": (14.87, 2), "stack_bowls": (53.2, 48), "insert_tubes": (16.67, 4.67),
                          "cover_blocks": (12.8, 5.33)},
    "X_VLA": {"hang_mugs": (12.33, 0), "sweep_blocks": (0, 0), "classify_objects": (4.57, 0.67), "fill_pen_holder": (12.17, 1.33),
              "stack_bowls": (56.6, 52), "insert_tubes": (45.07, 30.67), "cover_blocks": (22.57, 15.33)},
    "InternW0_delta": {"hang_mugs": (39.6, 22.67), "sweep_blocks": (9.33, 9.33), "classify_objects": (17.4, 8),
                       "fill_pen_holder": (61.83, 42), "stack_bowls": (88.67, 86.67), "insert_tubes": (63.6, 46.67),
                       "cover_blocks": (94.77, 92.67)},
    "OpenWAM": {"hang_mugs": (29.4, 10.67), "sweep_blocks": (2.67, 2.67), "classify_objects": (5.53, 1.33),
                "fill_pen_holder": (15.37, 6), "stack_bowls": (89.6, 88), "insert_tubes": (26.27, 6.67),
                "cover_blocks": (22.23, 16.67)},
}


def main() -> None:
    run = Path(sys.argv[1])
    rows = []
    for log in sorted(run.glob("*/*/episodes.jsonl")):
        policy, task = log.parent.parent.name, log.parent.name
        recs = {}
        for line in log.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") == "counted":
                recs[int(r["layout"])] = r  # last counted record per layout
        n = len(recs)
        if not n:
            continue
        score = 100 * sum(float(r.get("score") or 0) for r in recs.values()) / n
        succ = 100 * sum(1 for r in recs.values() if r.get("success")) / n
        off = OFFICIAL.get(policy, {}).get(task)
        z = None
        if off is not None:
            p = min(max(off[1] / 100, 0.5 / 150), 1 - 0.5 / 150)
            z = (succ / 100 - p) / math.sqrt(p * (1 - p) / n)
        rows.append({"policy": policy, "task": task, "n": n, "score": round(score, 2), "success_pct": round(succ, 2),
                     "official_score": off and off[0], "official_success_pct": off and off[1],
                     "z_success": z and round(z, 2), "layouts": sorted(recs)})
    print(f"{'policy':<18}{'task':<18}{'n':>4}{'score':>8}{'off':>8}{'succ%':>8}{'off%':>8}{'z':>7}")
    for r in rows:
        print(f"{r['policy']:<18}{r['task']:<18}{r['n']:>4}{r['score']:>8.2f}{r['official_score'] or 0:>8.2f}"
              f"{r['success_pct']:>8.2f}{r['official_success_pct'] or 0:>8.2f}{r['z_success'] if r['z_success'] is not None else '-':>7}")
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
