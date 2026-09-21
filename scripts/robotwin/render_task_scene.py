#!/usr/bin/env python3
"""Render one frame of a RoboTwin task's initial scene, for figures."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

ROOT = Path("/mnt/nvme0/shared/zhoujingjing/simuguard/SimuGuard")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "robotwin"))

from simuguard.adapters.robotwin import RoboTwinAdapter, make_task_env  # noqa: E402
from render_counterfactual_frames import _figure_camera, _save  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--robotwin-root", required=True)
parser.add_argument("--task", required=True)
parser.add_argument("--seed", type=int, default=100000)
parser.add_argument("--settle", type=int, default=400, help="substeps to let the scene settle")
parser.add_argument("--out", required=True)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=960)
parser.add_argument("--fovy-deg", type=float, default=62.0)
parser.add_argument("--camera-pos", type=float, nargs=3, default=[0.0, 0.75, 1.45])
parser.add_argument("--camera-forward", type=float, nargs=3, default=[0.0, -1.0, -0.55])
parser.add_argument("--camera", default="figure")
args = parser.parse_args()

env, _ = make_task_env(args.robotwin_root, args.task, args.seed)
try:
    adapter = RoboTwinAdapter(env)
    camera = _figure_camera(env, args)
    for _ in range(args.settle):
        adapter.scene.step()
    _save(env, Path(args.out), args.camera, camera)
    print("wrote", args.out)
finally:
    try:
        env.close_env()
    except Exception:
        pass
