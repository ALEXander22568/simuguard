#!/usr/bin/env python3
"""Latency of each Pi0.5 server on a dummy LIBERO observation.  Usage: pi05_ping.py URL[,URL...] [N]"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pi05_rollout import Pi05Client  # noqa: E402

urls = [u for u in sys.argv[1].split(",") if u]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
rng = np.random.default_rng(0)
obs = {"main_images": rng.integers(0, 255, (1, 256, 256, 3), dtype=np.uint8),
       "wrist_images": rng.integers(0, 255, (1, 256, 256, 3), dtype=np.uint8), "extra_view_images": None,
       "states": np.zeros((1, 8), dtype=np.float32), "task_descriptions": ["put the bowl on the plate"]}
for url in urls:
    client = Pi05Client([url], timeout_s=60.0, attempts=1)
    times = []
    try:
        for _ in range(n):
            t0 = time.time()
            chunk = client.predict(obs)
            times.append(time.time() - t0)
        print(f"{url}: chunk {chunk.shape}, latency median {np.median(times):.3f} s (min {min(times):.3f}, max {max(times):.3f})")
    except Exception as exc:  # noqa: BLE001
        print(f"{url}: FAILED {type(exc).__name__}: {exc}")
