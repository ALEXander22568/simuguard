"""SimuGuard on NVIDIA RoboLab: record, replay and intervene without modifying RoboLab.

Run with RoboLab's own interpreter from the RoboLab checkout (``policies`` must be importable)::

    cd RoboLab
    OMNI_KIT_ACCEPT_EULA=Y PYTHONPATH=/path/to/simuguard .venv/bin/python -m simuguard.integrations.robolab_eval \\
        --policy pi05 --task BananaInBowlTask --num_envs 8 --num_runs 1 --out /data/sg_robolab/pi05 --headless

Modes:

* ``eval`` (default): RoboLab's ``run_evaluation`` with an ``episode_runner`` that records every env
  of every run (:class:`~simuguard.adapters.isaac.robolab.BatchRecorder`).  ``--no-simuguard``
  runs the same loop without recording (timing baseline).
* ``replay``: rebuild the env (same task, seed, num_envs) and re-apply the recorded tensor-API
  writes step by step; reports bit-exactness per run.  With ``--depen`` this is the
  counterfactual: the recorded actuation is replayed with a depenetration cap.

``--depen VALUE[,bodies=a+b][,envs=0+3][,from=S][,to=S]`` (repeatable) caps
``physxRigidBody:maxDepenetrationVelocity``.  Only ``--cap-via prebuild`` (USD written before
the first ``SimulationContext.reset``) reaches GPU PhysX; runtime USD/usdrt writes have no effect.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import cv2  # noqa: F401  (RoboLab: import before isaaclab)

from ..adapters.isaac.robolab_io import (ejection_config, effective_ejection_thresholds, freeze_text, parse_cap, sha256_file,
                                         server_info, sha256_text, tree_sha256)

POLICIES = ("pi05", "gr00t")


def build_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher
    from robolab.eval.runner import add_common_eval_args

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("eval", "replay"), default="eval")
    p.add_argument("--policy", choices=POLICIES, default="pi05")
    p.add_argument("--remote-host", "--remote_host", default="localhost")
    p.add_argument("--remote-port", "--remote_port", type=int, default=None)
    p.add_argument("--open-loop-horizon", "--open_loop_horizon", type=int, default=None)
    p.add_argument("--policy-checkpoint", default=None, help="checkpoint id stamped into the manifest")
    p.add_argument("--server-info", default=None, help="JSON file describing the policy server, copied into the manifest")
    p.add_argument("--out", required=True, help="output root (RoboLab results go to <out>/robolab)")
    p.add_argument("--no-simuguard", action="store_true", help="plain RoboLab loop (timing baseline)")
    p.add_argument("--depen", action="append", default=[], help="depenetration cap spec (repeatable)")
    p.add_argument("--cap-via", choices=("prebuild", "usd", "usdrt"), default="prebuild")
    p.add_argument("--detector-config", default=None, help="JSON detector config (defaults otherwise)")
    p.add_argument("--record-dir", default=None, help="replay: <out>/<task> of the recording")
    p.add_argument("--replay-runs", default="0", help="replay: comma-separated run indices to compare")
    p.add_argument("--replay-render", action="store_true", help="replay: also render every render_interval")
    p.add_argument("--replay-save-states", action="store_true", help="replay: save the replayed object states per physics step")
    p.add_argument("--replay-contacts", action="store_true", help="replay: record and save the replayed contacts per env")
    add_common_eval_args(p)
    AppLauncher.add_app_launcher_args(p)
    return p


# ---------------------------------------------------------------------------- provenance
def _run(cmd: list[str], cwd: str | None = None) -> str | None:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def environment_fingerprint(args: argparse.Namespace, robolab_root: Path) -> dict[str, Any]:
    import importlib.metadata as md

    import torch

    sg_root = Path(__file__).resolve().parents[2]
    commit_file = sg_root / "SIMUGUARD_COMMIT"
    sg_commit = commit_file.read_text().strip() if commit_file.exists() else _run(["git", "rev-parse", "HEAD"], str(sg_root))
    dists = [(d.metadata["Name"], d.version) for d in md.distributions() if d.metadata["Name"]]
    freeze = freeze_text(dists)

    def version(name: str) -> str | None:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None

    return {
        "simuguard_commit": sg_commit,
        "simuguard_source_sha256": tree_sha256(sg_root / "simuguard"),
        "robolab_commit": _run(["git", "rev-parse", "HEAD"], str(robolab_root)),
        "robolab_dirty": bool(_run(["git", "status", "--porcelain", "--untracked-files=no"], str(robolab_root))),
        "uv_lock_sha256": sha256_file(robolab_root / "uv.lock"),
        "pip_freeze_sha256": sha256_text(freeze),
        "pip_freeze_packages": len(dists),
        "driver_version": _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "isaacsim_version": version("isaacsim") or version("isaacsim-kernel"),
        "isaaclab_version": version("isaaclab"),
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "hostname": platform.node(),
        "policy": args.policy,
        "policy_checkpoint": args.policy_checkpoint,
    }, freeze


# ---------------------------------------------------------------------------- clients
def make_client_factory(args: argparse.Namespace):
    if args.policy == "pi05":
        from policies.pi0_family.client import Pi0DroidJointposClient

        def factory(a: argparse.Namespace):
            kwargs = dict(remote_host=a.remote_host, remote_port=a.remote_port or 8000,
                          open_loop_horizon=a.open_loop_horizon, policy_variant="pi05")
            return Pi0DroidJointposClient(**{k: v for k, v in kwargs.items() if v is not None})

        return factory
    from policies.gr00t.client import GR00TDroidJointposClient

    def factory(a: argparse.Namespace):
        kwargs = dict(remote_host=a.remote_host, remote_port=a.remote_port or 5555, open_loop_horizon=a.open_loop_horizon)
        return GR00TDroidJointposClient(**{k: v for k, v in kwargs.items() if v is not None})

    return factory


# ---------------------------------------------------------------------------- build-time options
def install_build_options(args: argparse.Namespace, caps: list[Any]) -> dict[str, Any]:
    """Build-time caps (the only kind GPU PhysX honours).

    RoboLab replicates env_0's physics to all envs (``replicate_physics=True``), so a cap holds
    for every env: per-env selections are rejected, per-body ones apply in all envs.
    """

    state: dict[str, Any] = {}
    if caps and args.cap_via == "prebuild":
        from ..adapters.isaac.robolab import prebuild_caps

        bad = [c for c in caps if c.start > 1 or c.end is not None or c.envs is not None]
        if bad:
            raise SystemExit(f"build-time caps hold from step 1, for every env (replicated physics): {bad}")
        state["undo"] = prebuild_caps(caps, args.num_envs)
    return state


# ---------------------------------------------------------------------------- eval
def run_eval(args: argparse.Namespace, out: Path, caps: list[Any], manifest: dict[str, Any]) -> None:
    from robolab.constants import get_output_dir
    from robolab.eval.episode import run_episode
    from robolab.eval.runner import run_evaluation

    from ..adapters.isaac.robolab import BatchRecorder

    detector_cfg = ejection_config(json.loads(Path(args.detector_config).read_text()) if args.detector_config else None)
    recorders: dict[int, BatchRecorder] = {}
    runtime_caps = [] if args.cap_via == "prebuild" else caps
    timings: list[dict[str, Any]] = []

    def episode_runner(*, env: Any, env_cfg: Any, episode: int, client: Any, **kw: Any):
        task = os.path.basename(get_output_dir())
        t0 = time.time()
        if args.no_simuguard:
            results, msgs, timing = run_episode(env=env, env_cfg=env_cfg, episode=episode, client=client, **kw)
            row = {"task": task, "run": episode, "num_envs": env.num_envs, "wall_s": time.time() - t0, "robolab": timing}
            timings.append(row)
            _append_jsonl(out / "plain_timing.jsonl", row)
            return results, msgs, timing
        rec = recorders.get(id(env))
        if rec is None:
            for old in recorders.values():
                old.remove()
            recorders.clear()
            t_install = time.time()
            rec = BatchRecorder(env, out / task, task_name=task, detector_cfg=detector_cfg, caps=runtime_caps,
                                cap_via=args.cap_via, metadata={"policy": args.policy, "seed": args.seed,
                                          "prebuild_caps": [c.to_dict() for c in caps] if args.cap_via == "prebuild" else []}).install()
            recorders[id(env)] = rec
            manifest.setdefault("tasks", {})[task] = {
                "install_s": time.time() - t_install,
                "replicate_physics": bool(env.cfg.scene.replicate_physics),
                "num_envs": int(env.num_envs),
                "effective_ejection_thresholds": effective_ejection_thresholds(detector_cfg["ejection"], float(env.physics_dt)),
            }
            _write_json(out / "manifest.json", manifest)
        rec.begin_run(episode)
        results, msgs, timing = run_episode(env=env, env_cfg=env_cfg, episode=episode, client=client, **kw)
        report = rec.end_run(results, timing)
        row = {"task": task, "run": episode, "num_envs": env.num_envs, "wall_s": time.time() - t0, "robolab": timing,
               "recorder_overhead_s": report["timing"]["recorder_overhead_s"], "storage_bytes": report["storage_bytes"],
               "segments": len(report["segments"]),
               "confirmed": sum(s["confirmed"] or 0 for s in report["segments"]),
               "flags": sum(s["flags"] or 0 for s in report["segments"])}
        _append_jsonl(out / "run_index.jsonl", row)
        return results, msgs, timing

    run_evaluation(args, policy=args.policy, client_factory=make_client_factory(args), episode_runner=episode_runner)
    for rec in recorders.values():
        rec.remove()


# ---------------------------------------------------------------------------- replay
def run_replay(args: argparse.Namespace, out: Path, caps: list[Any]) -> None:
    from robolab.core.environments.runtime import create_env

    from ..adapters.isaac.robolab import replay_batch

    record_dir = Path(args.record_dir)
    install = json.loads((record_dir / "batch" / "install.json").read_text())
    task = install["task"]
    if int(install["num_envs"]) != args.num_envs:
        raise SystemExit(f"recorded with num_envs={install['num_envs']}, got {args.num_envs}")
    env, _ = create_env(task, device=args.device, seed=args.seed, num_envs=args.num_envs,
                        instruction_type=args.instruction_type, policy=args.policy,
                        renderer=args.renderer, rendering_mode=args.rendering_type)
    runs = [int(r) for r in args.replay_runs.split(",") if r != ""]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    report = replay_batch(env, record_dir, runs=runs, caps=[] if args.cap_via == "prebuild" else caps,
                          cap_via=args.cap_via, render=args.replay_render, out_dir=out,
                          save_states=args.replay_save_states, save_contacts=args.replay_contacts)
    report["wall_s"] = time.time() - t0
    report["record_dir"] = str(record_dir)
    report["prebuild_caps"] = [c.to_dict() for c in caps] if args.cap_via == "prebuild" else []
    name = "replay" if not caps else "counterfactual"
    _write_json(out / f"{name}_{task}_runs{'-'.join(map(str, runs))}.json", report)
    print("SIMUGUARD_REPLAY", json.dumps({k: report[k] for k in ("initial", "runs")}, default=str), flush=True)
    env.close()


# ---------------------------------------------------------------------------- main
def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = build_parser()
    args, _ = parser.parse_known_args()
    args.enable_cameras = True
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.output_folder_name = str(out / "robolab")
    app = AppLauncher(args).app
    code = 0
    try:
        import robolab.constants
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args.enable_subtask
        robolab_root = Path(robolab.__file__).resolve().parents[1]
        sys.path.insert(0, str(robolab_root))
        caps = [parse_cap(s) for s in args.depen]
        if args.mode == "replay":
            install = json.loads((Path(args.record_dir) / "batch" / "install.json").read_text())
            args.task = [install["task"]]
        auto_register_droid_envs(task_dirs=args.task_dirs, task=args.task)
        install_build_options(args, caps)
        fingerprint, freeze = environment_fingerprint(args, robolab_root)
        manifest = {
            "schema": "simuguard-robolab-run-v1",
            "mode": args.mode,
            "argv": sys.argv,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "fingerprint": fingerprint,
            "simuguard": not args.no_simuguard,
            "caps": [c.to_dict() for c in caps],
            "cap_via": args.cap_via,
            "num_envs": args.num_envs,
            "num_runs": args.num_runs,
            "seed": args.seed,
            "tasks_requested": args.task,
            "policy_server": server_info(args.server_info),
            "limitations": ["contacts with the ground plane and other static colliders are not recorded; "
                            "only contacts between rigid bodies with a PhysX contact view are"],
            "args": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, list, type(None)))},
        }
        (out / "pip_freeze.txt").write_text(freeze)
        if args.mode == "eval":
            _write_json(out / "manifest.json", manifest)
            run_eval(args, out, caps, manifest)
            manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _write_json(out / "manifest.json", manifest)
        else:
            _write_json(out / "replay_manifest.json", manifest)
            run_replay(args, out, caps)
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
    return code


if __name__ == "__main__":
    main()
