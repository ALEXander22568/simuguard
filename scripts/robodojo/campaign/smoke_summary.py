#!/usr/bin/env python3
"""Summarise the 48 GB box smoke run: per policy, batch wall time, policy get_action time, sim time, GPU 0 peak.
usage: python3 smoke_summary.py RUN_DIR"""
import csv, json, sys
from pathlib import Path
run = Path(sys.argv[1])
mem = []
p = run / "gpu0_mem.csv"
if p.exists():
    for row in csv.reader(p.open()):
        try:
            mem.append((int(row[0]), int(row[1]), row[2]))
        except (ValueError, IndexError):
            pass
for pol in ("OpenWAM", "InternW0_delta"):
    f = run / pol / "hang_mugs" / "episodes.jsonl"
    if not f.exists():
        print(pol, "no episodes"); continue
    eps = [json.loads(l) for l in f.open() if l.strip()]
    counted = [e for e in eps if e.get("status") == "counted"]
    print(f"== {pol}: {len(eps)} records, {len(counted)} counted, success {sum(bool(e.get('success')) for e in counted)}, "
          f"mean score {sum(e.get('score') or 0 for e in counted)/max(1,len(counted)):.2f}")
    seen = set()
    for e in eps:
        prof = e.get("profile_episode") or {}
        key = json.dumps(prof, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        ga = prof.get("policy_call:get_action", {}); st = prof.get("env.step", {}); ta = prof.get("env.take_action", {})
        go = prof.get("env.get_obs", {}); rc = prof.get("obs.render_for_capture", {})
        acts = e.get("actions") or 0
        print(f"  layout {e.get('layout')} batch? wall {e.get('episode_wall_s', 0):.0f}s reset {e.get('reset_wall_s', 0):.0f}s actions {acts}")
        print(f"    get_action n={ga.get('n')} total {ga.get('s')}s  per call {ga.get('s', 0)/max(1, ga.get('n', 1)):.2f}s  per control step {ga.get('s', 0)/max(1, acts):.3f}s")
        print(f"    env.take_action {ta.get('s')}s  env.step {st.get('s')}s  get_obs {go.get('s')}s  render {rc.get('s')}s  "
              f"sim per control step {(ta.get('s', 0)+go.get('s', 0))/max(1, acts):.3f}s")
        print("    all:", {k: v for k, v in prof.items()})
    t0 = min((e.get("started_utc") for e in eps), default=None)
print("GPU0 peak total MiB:", max((m[1] for m in mem), default=None), "samples", len(mem))
