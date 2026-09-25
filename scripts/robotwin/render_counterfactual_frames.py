#!/usr/bin/env python3
"""Render matched frames of one episode under two physics, for a figure.

Replays a recorded episode with the recorded actuation, optionally with a
counterfactual applied, and saves a rendered frame at each requested substep
plus the target's speed trace.  Because the actions are identical, frames at
the same substep in two conditions differ only by the change in physics.

Usage::

    python scripts/robotwin/render_counterfactual_frames.py --robotwin-root <RoboTwin> \\
        --segment-dir <seg> --condition baseline --substeps 10950 11030 11150 11600 \\
        --until-success --out-dir figures/raw/baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SIMUGUARD_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIMUGUARD_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    from attribute_failures import EVENT_LEAD_S, _install_velocity_clamp, first_invalid_onset
    from replay_intervention import build_intervention
    from simuguard.adapters.robotwin import RoboTwinAdapter, make_task_env
    from simuguard.core import ControlLog
    from simuguard.core.types import BodyRole
    from simuguard.integrations.robotwin_eval import reapply_recorded_intervention

    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--segment-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--condition", default="baseline",
                        help="baseline | vclamp_<v> | mass_<g>g | solver_high | depen_<v>; append @event to switch it on "
                             "EVENT_LEAD_S before the first physics-invalid onset, as attribute_failures does")
    parser.add_argument("--substeps", type=int, nargs="+", required=True)
    parser.add_argument("--stop-at-success", action="store_true",
                        help="also render the first substep where the benchmark reports success, then stop")
    parser.add_argument("--camera", default="figure", choices=("figure", "observer", "head"),
                        help="figure: a dedicated high-resolution camera framing the table")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--fovy-deg", type=float, default=52.0, help="smaller = tighter crop on the table")
    parser.add_argument("--camera-pos", type=float, nargs=3, default=[0.0, 0.42, 1.18])
    parser.add_argument("--camera-forward", type=float, nargs=3, default=[0.0, -1.0, -0.72])
    args = parser.parse_args()

    segment = Path(args.segment_dir).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((segment / "manifest.json").read_text())
    summary = json.loads((segment / "summary.json").read_text())
    meta = {**manifest["metadata"], **(summary.get("metadata") or {})}
    controls = ControlLog.load(segment / "controls.npz")
    wanted = sorted(set(args.substeps))

    env, _ = make_task_env(args.robotwin_root, meta["task"], int(meta["seed"]))
    speeds, robot_speeds, recorded_substeps = [], [], []
    info = {"segment": str(segment), "condition": args.condition, "frames": {}}
    try:
        adapter = RoboTwinAdapter(env)
        info["recorded_sim_intervention"] = reapply_recorded_intervention(adapter, meta)
        target = adapter.body_ids_with_role(BodyRole.TARGET)[0]
        robot_ids = adapter.body_ids_with_role(BodyRole.ROBOT)
        base, _, when = args.condition.partition("@")
        trigger, apply = None, None
        if when:
            if when != "event":
                raise SystemExit(f"unknown trigger in {args.condition}: only <condition>@event")
            onset = first_invalid_onset(segment)
            if onset is None:
                raise SystemExit(f"{segment} holds no physics-invalid event to trigger on")
            trigger = max(1, onset - max(1, round(EVENT_LEAD_S / adapter.timestep())))
            info["first_invalid_onset"] = onset
        if base.startswith("vclamp_"):
            _install_velocity_clamp(adapter, target, float(base.split("_", 1)[1]))
        elif base != "baseline":
            _, apply = build_intervention(base)
            if apply is not None and trigger is None:
                apply(adapter)

        camera = _figure_camera(env, args) if args.camera == "figure" else None
        success_substep = None
        for record in controls.between(0, controls.newest_substep):
            if record.substep == 0:
                continue
            if trigger is not None and record.substep == trigger and apply is not None:
                apply(adapter)
                info["applied_at_substep"] = record.substep
            adapter.apply_control(record)
            adapter.scene.step()
            states = adapter.read_states([target, *robot_ids])
            speeds.append(states[target].speed)
            robot_speeds.append(max((states[b].speed for b in robot_ids), default=0.0))
            recorded_substeps.append(record.substep)
            if record.substep in wanted:
                _save(env, out / f"s{record.substep:06d}.png", args.camera, camera)
                info["frames"][str(record.substep)] = f"s{record.substep:06d}.png"
            if success_substep is None and env.check_success():
                success_substep = record.substep
                _save(env, out / f"success_s{record.substep:06d}.png", args.camera, camera)
                info["frames"]["success"] = f"success_s{record.substep:06d}.png"
                if args.stop_at_success:
                    break
            if record.substep >= max(wanted) and not args.stop_at_success:
                break
        info["success_substep"] = success_substep
        np.savez_compressed(
            out / "speed.npz", substeps=np.asarray(recorded_substeps), speed=np.asarray(speeds),
            robot_speed=np.asarray(robot_speeds),
        )
    finally:
        try:
            env.close_env()
        except Exception:  # noqa: BLE001
            pass
    (out / "frames.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=1))
    return 0


def _figure_camera(env, args):
    """A camera of our own, so figures are not limited to the benchmark's 320x240 views."""

    import sapien

    camera = env.scene.add_camera(
        name="simuguard_figure", width=args.width, height=args.height,
        fovy=float(np.deg2rad(args.fovy_deg)), near=0.05, far=100.0,
    )
    position = np.asarray(args.camera_pos, dtype=float)
    forward = np.asarray(args.camera_forward, dtype=float)
    left = np.array([1.0, 0.0, 0.0])
    matrix = np.eye(4)
    matrix[:3, :3] = np.stack([forward, left, np.cross(forward, left)], axis=1)
    matrix[:3, 3] = position
    camera.entity.set_pose(sapien.Pose(matrix))
    return camera


def _save(env, path: Path, which: str, camera=None) -> None:
    from PIL import Image

    env._update_render()
    if camera is not None:
        camera.take_picture()
        image = (camera.get_picture("Color") * 255).clip(0, 255).astype("uint8")[:, :, :3]
    else:
        env.cameras.update_picture()
        image = env.cameras.get_observer_rgb() if which == "observer" else env.cameras.get_rgb()["head_camera"]["rgb"]
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


if __name__ == "__main__":
    raise SystemExit(main())
