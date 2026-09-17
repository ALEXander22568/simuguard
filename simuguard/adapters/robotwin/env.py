"""Build RoboTwin task environments exactly as the official evaluator does.

SimuGuard never re-implements RoboTwin's argument loading.  It imports the
official ``scripts/eval_policy_xpolicylab.py`` from an *unmodified* RoboTwin
checkout and reuses ``load_task_args`` and ``class_decorator``.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _prepare_robotwin_imports(robotwin_root: str | Path) -> Path:
    root = Path(robotwin_root).resolve()
    if not (root / "envs" / "_base_task.py").is_file():
        raise FileNotFoundError(f"not a RoboTwin checkout: {root}")
    for path in (root, root / "scripts", root / "description" / "utils"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    # RoboTwin resolves assets and configs relative to the repository root.
    os.chdir(root)
    return root


def official_eval_module(robotwin_root: str | Path) -> Any:
    _prepare_robotwin_imports(robotwin_root)
    return importlib.import_module("eval_policy_xpolicylab")


def load_official_task_args(
    robotwin_root: str | Path,
    task_name: str,
    *,
    task_config: str = "demo_clean",
    policy_name: str = "SimuGuard",
) -> dict[str, Any]:
    module = official_eval_module(robotwin_root)
    args, _embodiment = module.load_task_args(
        {"task_name": task_name, "task_config": task_config, "policy_name": policy_name, "ckpt_setting": "simuguard"}
    )
    args["eval_mode"] = True
    args["render_freq"] = 0
    args.pop("eval_video_save_dir", None)
    return args


def make_task_env(
    robotwin_root: str | Path,
    task_name: str,
    seed: int,
    *,
    task_config: str = "demo_clean",
    episode_index: int = 0,
    overrides: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Instantiate ``envs.<task_name>`` and run ``setup_demo`` with official args."""

    module = official_eval_module(robotwin_root)
    args = load_official_task_args(robotwin_root, task_name, task_config=task_config)
    args.update(overrides or {})
    env = module.class_decorator(task_name)
    env.setup_demo(now_ep_num=episode_index, seed=int(seed), is_test=True, **args)
    return env, args


def robotwin_provenance(robotwin_root: str | Path) -> dict[str, Any]:
    root = Path(robotwin_root).resolve()

    def git(*cmd: str, cwd: Path = root) -> str | None:
        try:
            return subprocess.run(["git", *cmd], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return None

    status = git("status", "--porcelain")
    xpl = root / "XPolicyLab"
    return {
        "robotwin_root": str(root),
        "robotwin_commit": git("rev-parse", "HEAD"),
        "robotwin_dirty_entries": len(status.splitlines()) if status else 0,
        "xpolicylab_commit": git("rev-parse", "HEAD", cwd=xpl) if xpl.exists() else None,
    }
