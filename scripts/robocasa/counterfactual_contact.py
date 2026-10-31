#!/usr/bin/env python3
"""Same RoboCasa state, different penetration-recovery time: does the object get thrown?

For one recorded episode (a segment of xr1_rollout.py), rebuild the scene from its seed, check
that restoring the initial MuJoCo state and replaying the recorded actuation reproduces the
recorded object trajectories, then take the moments at which a free object was pressed deepest
into a non-robot body.  From the exact state at each moment the recorded actuation is replayed
for ``--window`` seconds twice: with the contact time constants as recorded, and with every time
constant capped at ``--tau`` seconds.  Only the recovery time changes: same penetration, same
robot, same actuation.

Output: JSON with, per moment and setting, each free object's peak speed and displacement and
the ejection detector's confirmed events.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from xr1_rollout import BASE_SEED, install_persistent_renderer, roles_for, unwrap  # noqa: E402

from simuguard.adapters.mujoco import attach_robosuite  # noqa: E402
from simuguard.core.controls import ControlLog  # noqa: E402
from simuguard.core.detectors.base import DetectorContext  # noqa: E402
from simuguard.core.detectors.ejection import ContactEjectionDetector, EjectionConfig  # noqa: E402
from simuguard.core.events import EventStatus  # noqa: E402
from simuguard.core.snapshot import Snapshot  # noqa: E402
from simuguard.core.statelog import StateLog  # noqa: E402
from simuguard.core.types import BodyKind, BodyRole  # noqa: E402


def deepest_moments(segment: Path, count: int, min_gap: int, min_depth: float) -> list[dict]:
    manifest = json.loads((segment / "manifest.json").read_text())
    roles = {b: i["role"] for b, i in manifest["bodies"].items()}
    free = {b for b, i in manifest["bodies"].items() if i["kind"] == "dynamic"}
    hits = []
    with gzip.open(segment / "trace.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            frame = json.loads(line)
            for c in frame["contacts"]:
                a, b, depth = c["a"], c["b"], float(c["penetration"])
                if depth >= min_depth and (a in free or b in free) and "robot" not in (roles.get(a), roles.get(b)):
                    hits.append((depth, int(frame["substep"]), a, b))
    chosen: list[dict] = []
    for depth, substep, a, b in sorted(hits, reverse=True):
        if all(abs(substep - m["substep"]) >= min_gap for m in chosen):
            chosen.append({"substep": substep, "depth_m": depth, "pair": [a, b]})
        if len(chosen) >= count:
            break
    return sorted(chosen, key=lambda m: m["substep"])


def cap_time_constants(model, tau: float | None, original: np.ndarray) -> int:
    model.geom_solref[:] = original
    if tau is None:
        return 0
    change = (original[:, 0] > 0) & (original[:, 0] > tau)
    model.geom_solref[change, 0] = tau
    return int(change.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("segment")
    ap.add_argument("--moments", type=int, default=3)
    ap.add_argument("--min-depth", type=float, default=0.002, help="m; shallower contacts are not candidates")
    ap.add_argument("--tau", type=float, default=0.004, help="s; cap on every contact time constant")
    ap.add_argument("--window", type=float, default=1.0, help="s replayed from each moment")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    segment = Path(args.segment)
    manifest = json.loads((segment / "manifest.json").read_text())
    meta = manifest["metadata"]
    task, seed = meta["task"], int(meta["seed"])
    if (meta.get("contact") or {}).get("name", "default") != "default":
        raise SystemExit("counterfactuals start from default-contact episodes")
    controls = ControlLog.load(segment / "controls.npz")
    initial = Snapshot.load(segment / "initial_snapshot.json.gz")
    reference = StateLog.load(segment / "states.npz")
    moments = deepest_moments(segment, args.moments, min_gap=250, min_depth=args.min_depth)
    report: dict = {"segment": str(segment), "task": task, "seed": seed, "tau_cap_s": args.tau,
                    "window_s": args.window, "moments": []}
    if not moments:
        report["note"] = f"no free-object contact deeper than {args.min_depth} m"
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(report["note"])
        return 0

    import gymnasium as gym
    import robocasa  # noqa: F401

    install_persistent_renderer()
    env = gym.make(f"robocasa/{task}", split="pretrain", seed=BASE_SEED)
    env.reset(seed=seed)
    rs = unwrap(env)
    roles, _ = roles_for(rs)
    adapter = attach_robosuite(rs, roles=roles, task_name=task)
    model = adapter.model
    original = model.geom_solref.copy()
    bodies = adapter.bodies()
    free = [b for b, i in bodies.items() if i.kind == BodyKind.DYNAMIC]
    dt = adapter.timestep()

    # 1) reach every moment exactly: restore the nearest earlier snapshot (or the initial state) and
    #    replay the recorded actuation; the replayed objects must sit exactly on the recorded ones
    archive = np.load(segment / "snapshots.npz") if (segment / "snapshots.npz").is_file() else None
    ref_ids, ref = reference.body_ids, reference.array()
    ref_row = {int(s): i for i, s in enumerate(reference.substeps)}
    worst, natives = 0.0, {}
    for m in moments:
        start = 0
        if archive is not None:
            earlier = [i for i, k in enumerate(archive["substeps"]) if 0 < k <= m["substep"]]
            if earlier:
                i = earlier[-1]
                start = int(archive["substeps"][i])
                adapter.set_native_state(np.frombuffer(archive["states"][i, : archive["lengths"][i]].tobytes(), dtype=np.float64))
                adapter.mark_post_step()
        if start == 0:
            adapter.restore_snapshot(initial, method="native")
        for record in controls.between(start, m["substep"]):  # record 0 is the actuation at attach, not a step
            adapter.apply_control(record)
            adapter.step_physics()
        now = adapter.read_states(ref_ids)
        row = ref[ref_row[m["substep"]]]
        worst = max(worst, max(float(np.abs(now[b].position - row[j, :3]).max()) for j, b in enumerate(ref_ids)))
        natives[m["substep"]] = adapter.native_state()
        m["replayed_from"] = start
    report["replay_max_abs_pos_err_m"] = worst

    # 2) from each moment: recorded time constants vs capped
    cfg = EjectionConfig(delta_v_reference_timestep_s=0.004)
    steps = int(round(args.window / dt))
    for m in moments:
        entry = dict(m)
        for setting, tau in (("recorded", None), (f"tau_{1000 * args.tau:g}ms", args.tau)):
            changed = cap_time_constants(model, tau, original)
            adapter.set_native_state(natives[m["substep"]])
            adapter.mark_post_step()
            detector = ContactEjectionDetector(cfg)
            context = DetectorContext(episode_id=f"{task}:{seed}:{m['substep']}:{setting}", timestep=dt, bodies=bodies)
            detector.reset(context)
            start = {b: s.position.copy() for b, s in adapter.read_states(free).items()}
            peak = {b: 0.0 for b in free}
            confirmed, robot_contact = [], False
            for k in range(1, steps + 1):
                record = controls.get(m["substep"] + k)
                if record is None:
                    break
                adapter.apply_control(record)
                adapter.step_physics()
                frame = adapter.read_frame(m["substep"] + k, free)
                for b, s in frame.states.items():
                    peak[b] = max(peak[b], s.speed)
                robot_contact |= any(bodies.get(p.body_a) and bodies[p.body_a].role == BodyRole.ROBOT
                                     or bodies.get(p.body_b) and bodies[p.body_b].role == BodyRole.ROBOT
                                     for p in frame.contacts if p.body_a in m["pair"] or p.body_b in m["pair"])
                for event in detector.observe(frame, context):
                    if event.status == EventStatus.CONFIRMED:
                        confirmed.append({"body": event.bodies[0], "onset": event.onset_substep,
                                          "max_speed": event.metrics["max_speed_mps"], "reasons": event.reasons})
                context.previous = frame
            end = adapter.read_states(free)
            entry[setting] = {
                "geoms_changed": changed,
                "peak_speed": {b.split(":", 1)[1]: round(v, 4) for b, v in peak.items()},
                "displacement": {b.split(":", 1)[1]: round(float(np.linalg.norm(end[b].position - start[b])), 4) for b in free},
                "confirmed": confirmed, "robot_touches_pair": robot_contact,
            }
        cap_time_constants(model, None, original)
        report["moments"].append(entry)
        rec, cap = entry["recorded"], entry[f"tau_{1000 * args.tau:g}ms"]
        a, b = m["pair"]
        obj = a if a in free else b
        name = obj.split(":", 1)[1]
        print(f"substep {m['substep']:6d} depth {1000 * m['depth_m']:5.1f} mm {m['pair']}: {name} peak "
              f"{rec['peak_speed'][name]:.2f} -> {cap['peak_speed'][name]:.2f} m/s, events {len(rec['confirmed'])} -> {len(cap['confirmed'])}",
              flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
