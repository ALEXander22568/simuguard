#!/usr/bin/env python3
"""Audit LIBERO's official initial states: what does the physics do before any policy acts?

For every init state of a task, the episode is built exactly as the evaluation builds it (fresh
env, ``env.seed(init)``, reset, ``set_init_state``) but without cameras, SimuGuard is attached, and
the 15 settle steps every LIBERO evaluation starts with (zero motion, gripper open) are run.  Per
init state: the deepest initial penetration of every free object into a non-robot body, confirmed
ejection events and what the gravity and carrier stages make of them, the peak speed and how far
each object ended up from where the init state put it.  Writes a SimuGuard segment per init state
and one line per init state to ``<out>/<suite>/t<task>/audit.jsonl``.

Usage::

    $PY scripts/libero/init_state_audit.py --suite libero_object --task 0 --inits 0-49 --out $RUNS/init_audit
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[1] / "robotwin"))

from carrier_filter import classify as carrier_classify  # noqa: E402
from gravity_filter import classify_segment  # noqa: E402

from simuguard.adapters.mujoco.libero import attach_libero, make_env  # noqa: E402
from simuguard.core.monitor import MonitorConfig, SubstepMonitor  # noqa: E402
from simuguard.core.recorder import EpisodeRecorder  # noqa: E402
from simuguard.core.types import BodyKind, BodyRole  # noqa: E402
from simuguard.presets import default_detectors  # noqa: E402

SETTLE = [0.0] * 6 + [-1.0]


def audit_one(suite: str, task: int, init: int, out: Path, settle_steps: int, keep: bool) -> dict:
    t0 = time.time()
    env, _, meta = make_env(suite, task, init, init, use_camera_obs=False, ignore_done=True)
    adapter, info = attach_libero(env, task_name=meta["task"])
    bodies = adapter.bodies()
    free = sorted(b for b, i in bodies.items() if i.kind == BodyKind.DYNAMIC)
    robot = {b for b, i in bodies.items() if i.role == BodyRole.ROBOT}
    depth = {b: 0.0 for b in free}
    partner = {b: None for b in free}
    for pair in adapter.read_contacts():
        if pair.body_a in robot or pair.body_b in robot:
            continue
        pen = max(0.0, -min(p.separation for p in pair.points))
        for b, other in ((pair.body_a, pair.body_b), (pair.body_b, pair.body_a)):
            if b in depth and pen > depth[b]:
                depth[b], partner[b] = pen, other
    start = adapter.read_states(free)
    seg = out / f"i{init:02d}"
    monitor = SubstepMonitor(adapter, default_detectors({"ejection": {"delta_v_reference_timestep_s": 0.004}}),
                             episode_id=f"{suite}:{meta['task']}:init{init}",
                             config=MonitorConfig.from_dict({"snapshot_interval_substeps": 0}),
                             recorder=EpisodeRecorder(seg), metadata={**meta, "phase": "settle"})
    monitor.attach()
    for _ in range(settle_steps):
        env.step(SETTLE)
    summary = monitor.finalize()
    end = adapter.read_states(free)
    log = monitor.state_log  # every substep, every free object
    speeds = np.linalg.norm(log.array()[:, :, 7:10], axis=2) if log is not None and len(log) else None
    peak = {b: float(speeds[:, log.body_ids.index(b)].max()) if speeds is not None and b in log.body_ids else 0.0
            for b in free}
    env.close()
    gravity = classify_segment(seg, 300, 50, 1.5, 0.05) or {"events": []}
    carrier = carrier_classify(seg, 0.04, 0.1, 1.5, 0.05) if any(
        e.get("verdict") == "physics_invalid" for e in gravity["events"]) else {"events": []}
    invalid = [e for e in (carrier or {"events": []})["events"] if e.get("verdict") == "physics_invalid"]
    record = {
        "suite": suite, "task_id": task, "task": meta["task"], "init": init,
        "confirmed": summary["confirmed_count"], "flags": summary["flag_count"], "monitor_errors": summary["error_count"],
        "after_gravity": sum(e.get("verdict") == "physics_invalid" for e in gravity["events"]),
        "physics_invalid": len(invalid), "invalid_bodies": sorted({e["body"] for e in invalid}),
        "objects": {b[5:]: {"init_penetration_mm": round(1000 * depth[b], 2), "into": (partner[b] or "")[5:],
                            "peak_speed_mps": round(peak[b], 3),
                            "moved_cm": round(100 * float(np.linalg.norm(end[b].position - start[b].position)), 2),
                            "mass_kg": round(bodies[b].mass, 4)} for b in free},
        "wall_s": round(time.time() - t0, 1),
    }
    if not keep and not summary["confirmed_count"]:
        shutil.rmtree(seg, ignore_errors=True)  # keep the segments worth looking at
    return record


def parse_range(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--inits", default="0-49")
    ap.add_argument("--settle-steps", type=int, default=15)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-all", action="store_true", help="keep segments without events too")
    args = ap.parse_args()
    out = Path(args.out) / args.suite / f"t{args.task:02d}"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "audit.jsonl"
    done = {json.loads(line)["init"] for line in log.read_text().splitlines() if line.strip()} if log.exists() else set()
    for init in parse_range(args.inits):
        if init in done:
            continue
        record = audit_one(args.suite, args.task, init, out, args.settle_steps, args.keep_all)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        worst = max(record["objects"].items(), key=lambda kv: kv[1]["init_penetration_mm"])
        print(f"{args.suite} t{args.task} init{init}: confirmed {record['confirmed']} invalid {record['physics_invalid']} "
              f"deepest {worst[0]} {worst[1]['init_penetration_mm']} mm into {worst[1]['into']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
