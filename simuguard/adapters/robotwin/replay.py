"""Exact replay on RoboTwin: rebuild the task env with the same seed, then replay controls.

Verified on place_can_basket seed 100000 (RoboTwin 6dde571): the rebuilt env matches
the recorded initial public state exactly and 1750 replayed substeps reproduce every
body position with zero error.  In-place snapshot restore (PhysX pack/unpack or
public state) is not exact in contact because solver warm-start state is not restored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from ...core.adapter import SimAdapter
from ...core.detectors.base import Detector
from ...core.replay import ReplayResult, replay_bundle
from ...core.snapshot import ReplayBundle
from .adapter import RoboTwinAdapter
from .env import make_task_env
from .tasks import TaskSpec


def replay_on_rebuilt_env(
    robotwin_root: str | Path,
    task_name: str,
    seed: int,
    bundle: ReplayBundle,
    *,
    detectors: Iterable[Detector] = (),
    task_spec: TaskSpec | None = None,
    before_replay: Callable[[SimAdapter], None] | None = None,
    position_tolerance_m: float = 1.0e-4,
    env_overrides: dict[str, Any] | None = None,
) -> ReplayResult:
    if bundle.snapshot.substep != 0:
        raise ValueError("rebuild replay requires an episode_start bundle (snapshot at substep 0)")
    env, _ = make_task_env(robotwin_root, task_name, seed, overrides=env_overrides)
    try:
        adapter = RoboTwinAdapter(env, task_spec=task_spec)
        return replay_bundle(
            adapter,
            bundle,
            method="none",
            detectors=detectors,
            before_replay=before_replay,
            position_tolerance_m=position_tolerance_m,
        )
    finally:
        try:
            env.close_env()
        except Exception:  # noqa: BLE001
            pass
