#!/usr/bin/env python3
"""Render how MuJoCo resolves one of LIBERO's stored initial states during the settle steps (for figures).

The episode is built exactly as the evaluation builds it (fresh env, ``env.seed(init)``, reset,
``set_init_state``), with cameras on, SimuGuard attached, and the 15 settle steps every LIBERO evaluation
starts with (zero motion, gripper open).  Writes, into ``--out``:

* ``step{k:02d}.png``: the camera after k settle steps (0 = the init state as set, before any physics step);
* ``speeds.npz``: ``substeps``, ``bodies``, ``speed_mps`` (every free object at every substep);
* ``meta.json``: the episode description, the substeps per settle step and the initial penetration of every
  free object into a non-robot body (mm, and the body it is in).

Usage::

    $PY scripts/libero/render_init_strip.py --suite libero_object --task 0 --init 0 --out $RUNS/figA_libero/t00_i00
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))

from simuguard.adapters.mujoco.libero import attach_libero, make_env  # noqa: E402
from simuguard.core.monitor import MonitorConfig, SubstepMonitor  # noqa: E402
from simuguard.core.recorder import EpisodeRecorder  # noqa: E402
from simuguard.core.types import BodyKind, BodyRole  # noqa: E402
from simuguard.presets import default_detectors  # noqa: E402

SETTLE = [0.0] * 6 + [-1.0]


def save_image(array: np.ndarray, path: Path) -> None:
    img = np.ascontiguousarray(array[::-1])        # robosuite renders upside down
    try:
        from PIL import Image

        Image.fromarray(img).save(path)
    except ImportError:
        import imageio

        imageio.imwrite(path, img)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--init", type=int, default=0)
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 1, 2, 3, 4, 6, 10, 15])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--camera", default="agentview")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env, obs, meta = make_env(args.suite, args.task, args.init, args.init, camera_size=args.size, ignore_done=True)
    adapter, info = attach_libero(env, task_name=meta["task"])
    bodies = adapter.bodies()
    free = sorted(b for b, i in bodies.items() if i.kind == BodyKind.DYNAMIC)
    robot = {b for b, i in bodies.items() if i.role == BodyRole.ROBOT}
    depth = {b: (0.0, None) for b in free}
    for pair in adapter.read_contacts():
        if pair.body_a in robot or pair.body_b in robot:
            continue
        pen = max(0.0, -min(p.separation for p in pair.points))
        for b, other in ((pair.body_a, pair.body_b), (pair.body_b, pair.body_a)):
            if b in depth and pen > depth[b][0]:
                depth[b] = (pen, other)
    if 0 in args.steps:
        save_image(obs[f"{args.camera}_image"], out / "step00.png")
    monitor = SubstepMonitor(adapter, default_detectors({"ejection": {"delta_v_reference_timestep_s": 0.004}}),
                             episode_id=f"{args.suite}:{meta['task']}:init{args.init}",
                             config=MonitorConfig.from_dict({"snapshot_interval_substeps": 0}),
                             recorder=EpisodeRecorder(out / "segment"), metadata={**meta, "phase": "settle"})
    monitor.attach()
    for k in range(1, max(args.steps) + 1):
        obs, _, _, _ = env.step(SETTLE)
        if k in args.steps:
            save_image(obs[f"{args.camera}_image"], out / f"step{k:02d}.png")
    monitor.finalize()
    log = monitor.state_log
    arr = log.array()
    speeds = np.linalg.norm(arr[:, :, 7:10], axis=2)
    keep = [log.body_ids.index(b) for b in free if b in log.body_ids]
    np.savez_compressed(out / "speeds.npz", substeps=np.asarray(log.substeps), bodies=np.asarray([log.body_ids[j] for j in keep]),
                        speed_mps=speeds[:, keep])
    meta.update({"camera": args.camera, "size": args.size, "steps": args.steps,
                 "substeps_per_step": info.get("substeps_per_step"), "timestep": info.get("timestep"),
                 "init_penetration_mm": {b: [round(d * 1000, 2), o] for b, (d, o) in depth.items()}})
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    env.close()
    peak = {log.body_ids[j]: round(float(speeds[:, j].max()), 3) for j in keep}
    print(json.dumps({"peak_speed_mps": peak, "init_penetration_mm": meta["init_penetration_mm"]}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
