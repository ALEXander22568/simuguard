#!/usr/bin/env python3
"""Export one episode as a frame sequence plus a per-substep ground-truth table.

Replays a recorded episode (optionally under a counterfactual), writes a rendered
frame every ``--frame-every`` substeps, and a CSV row for *every* substep in the
range with the physical quantities the figures are built from: target speed and
position, robot link speed, and the contact forces, impulses and penetration on
the target, split by what it touches.

Usage::

    python scripts/robotwin/export_episode_trace.py --robotwin-root <RoboTwin> \\
        --segment-dir <seg> --out-dir runs/trace_x --from-substep 10200 \\
        --to-substep 12400 --frame-every 25
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SIMUGUARD_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIMUGUARD_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

COLUMNS = [
    "substep", "sim_time_s", "control_step", "frame",
    "target_speed_mps", "target_dv_mps", "target_kinetic_energy_j",
    "target_x", "target_y", "target_z",
    "target_vx", "target_vy", "target_vz", "target_angular_speed_radps",
    "container_speed_mps", "target_container_distance_m",
    "fastest_robot_link_speed_mps",
    "contact_force_container_n", "contact_impulse_container_ns", "penetration_container_m",
    "contact_force_robot_n", "contact_impulse_robot_ns",
    "contact_force_other_n", "contact_impulse_other_ns",
    "contact_points_total",
]


def main() -> int:
    from attribute_failures import _install_velocity_clamp
    from render_counterfactual_frames import _figure_camera, _save
    from replay_intervention import build_intervention
    from simuguard.adapters.robotwin import RoboTwinAdapter, make_task_env
    from simuguard.core import ControlLog
    from simuguard.core.types import BodyRole
    from simuguard.integrations.robotwin_eval import reapply_recorded_intervention

    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--segment-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--condition", default="baseline")
    parser.add_argument("--from-substep", type=int, default=0)
    parser.add_argument("--to-substep", type=int, required=True)
    parser.add_argument("--frame-every", type=int, default=25, help="substeps between rendered frames")
    parser.add_argument("--camera", default="figure", choices=("figure", "observer", "head"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--fovy-deg", type=float, default=62.0)
    parser.add_argument("--camera-pos", type=float, nargs=3, default=[0.0, 0.75, 1.45])
    parser.add_argument("--camera-forward", type=float, nargs=3, default=[0.0, -1.0, -0.55])
    args = parser.parse_args()

    segment = Path(args.segment_dir).resolve()
    out = Path(args.out_dir).resolve()
    (out / "frames").mkdir(parents=True, exist_ok=True)
    manifest = json.loads((segment / "manifest.json").read_text())
    summary = json.loads((segment / "summary.json").read_text())
    meta = {**manifest["metadata"], **(summary.get("metadata") or {})}
    controls = ControlLog.load(segment / "controls.npz")

    env, _ = make_task_env(args.robotwin_root, meta["task"], int(meta["seed"]))
    rows: list[dict] = []
    frames: list[dict] = []
    try:
        adapter = RoboTwinAdapter(env)
        recorded_intervention = reapply_recorded_intervention(adapter, meta)
        target = adapter.body_ids_with_role(BodyRole.TARGET)[0]
        containers = adapter.body_ids_with_role(BodyRole.CONTAINER)
        container = containers[0] if containers else None
        robot_ids = adapter.body_ids_with_role(BodyRole.ROBOT)
        mass = adapter.mass(target)
        if args.condition.startswith("vclamp_"):
            _install_velocity_clamp(adapter, target, float(args.condition.split("_", 1)[1]))
        elif args.condition != "baseline":
            _, apply = build_intervention(args.condition)
            if apply is not None:
                apply(adapter)
        camera = _figure_camera(env, args) if args.camera == "figure" else None
        timestep = adapter.timestep()
        previous_velocity = None

        for record in controls.between(0, controls.newest_substep):
            if record.substep == 0:
                continue
            if record.substep > args.to_substep:
                break
            adapter.apply_control(record)
            adapter.scene.step()
            if record.substep < args.from_substep:
                continue

            ids = [target] + ([container] if container else []) + robot_ids
            states = adapter.read_states(ids)
            target_state = states[target]
            velocity = np.asarray(target_state.linear_velocity, dtype=float)
            speed = float(np.linalg.norm(velocity))
            delta_v = float(np.linalg.norm(velocity - previous_velocity)) if previous_velocity is not None else 0.0
            previous_velocity = velocity

            buckets = {"container": [0.0, 0.0, 0.0], "robot": [0.0, 0.0, 0.0], "other": [0.0, 0.0, 0.0]}
            points = 0
            for pair in adapter.read_contacts():
                if not pair.involves(target):
                    continue
                other = pair.other(target)
                key = "container" if other == container else ("robot" if other in robot_ids else "other")
                buckets[key][0] += pair.force_abs_estimate(timestep)
                buckets[key][1] += pair.impulse_abs_sum
                buckets[key][2] = max(buckets[key][2], pair.max_penetration)
                points += pair.point_count

            is_frame = (record.substep - args.from_substep) % args.frame_every == 0
            name = ""
            if is_frame:
                name = f"s{record.substep:07d}.png"
                _save(env, out / "frames" / name, args.camera, camera)
                frames.append({"substep": int(record.substep), "file": f"frames/{name}",
                               "sim_time_s": round(record.substep * timestep, 4),
                               "target_speed_mps": round(speed, 4)})

            rows.append({
                "substep": int(record.substep),
                "sim_time_s": round(record.substep * timestep, 5),
                "control_step": int(record.control_step),
                "frame": name,
                "target_speed_mps": round(speed, 5),
                "target_dv_mps": round(delta_v, 5),
                "target_kinetic_energy_j": round(0.5 * mass * speed * speed, 6),
                "target_x": round(float(target_state.position[0]), 5),
                "target_y": round(float(target_state.position[1]), 5),
                "target_z": round(float(target_state.position[2]), 5),
                "target_vx": round(float(velocity[0]), 5),
                "target_vy": round(float(velocity[1]), 5),
                "target_vz": round(float(velocity[2]), 5),
                "target_angular_speed_radps": round(float(np.linalg.norm(target_state.angular_velocity)), 5),
                "container_speed_mps": round(states[container].speed, 5) if container else "",
                "target_container_distance_m": (
                    round(float(np.abs(np.asarray(target_state.position) -
                                       np.asarray(states[container].position)).sum()), 5) if container else ""),
                "fastest_robot_link_speed_mps": round(max((states[b].speed for b in robot_ids), default=0.0), 5),
                "contact_force_container_n": round(buckets["container"][0], 4),
                "contact_impulse_container_ns": round(buckets["container"][1], 6),
                "penetration_container_m": round(buckets["container"][2], 6),
                "contact_force_robot_n": round(buckets["robot"][0], 4),
                "contact_impulse_robot_ns": round(buckets["robot"][1], 6),
                "contact_force_other_n": round(buckets["other"][0], 4),
                "contact_impulse_other_ns": round(buckets["other"][1], 6),
                "contact_points_total": points,
            })
    finally:
        try:
            env.close_env()
        except Exception:  # noqa: BLE001
            pass

    with (out / "ground_truth.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    events = [e for e in summary.get("events", []) if e["status"] == "confirmed"]
    (out / "episode.json").write_text(json.dumps({
        "segment": str(segment),
        "task": meta["task"],
        "seed": meta["seed"],
        "phase": meta.get("phase"),
        "recorded_outcome": meta.get("outcome"),
        "condition": args.condition,
        "recorded_sim_intervention": recorded_intervention,
        "timestep_s": timestep,
        "target_body": target,
        "target_mass_kg": mass,
        "container_body": container,
        "substep_range": [args.from_substep, args.to_substep],
        "frame_every": args.frame_every,
        "frames": frames,
        "confirmed_events": [
            {k: e[k] for k in ("event_id", "detector", "onset_substep", "control_step", "bodies", "metrics", "reasons")
             if k in e}
            for e in events if args.from_substep <= e["onset_substep"] <= args.to_substep
        ],
    }, indent=2, default=str))
    print(f"wrote {len(rows)} substeps and {len(frames)} frames to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
