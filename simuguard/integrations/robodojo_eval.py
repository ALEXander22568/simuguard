"""Run RoboDojo evaluation episodes under SimuGuard (native RoboDojo install or image).

The environment is RoboDojo's own ``EvalEnv`` built exactly like ``src/eval_client/main.py``
builds it (same configs, ``process_randomization`` / ``process_config``, ``eval_batch``
read from the policy's ``deploy.yml``), by default with one env per process (``num_envs = 1``;
the official batch policies run 10 envs side by side).  Each layout id is one official
episode: ``env.reset(seed=[id])`` -> ``env.run_eval()`` (the XPolicyLab policy loop,
RoboDojo's reward manager and video writer) -> ``env.close()``.  As in ``main.py`` a
layout that fails ``check_layout_stability`` (``UnStableError``) or is marked unstable
during the episode uses up its id and is not counted.  What differs from ``main.py`` is
only the outer loop (explicit layout list, ``--layouts all`` = every layout file of the
eval seed, an optional cap on counted episodes, resume from ``episodes.jsonl`` instead of
the resume manifest, no PhysX-crash re-exec) and the instrumentation:

* before the env exists, :func:`install_contact_reporting` makes RoboDojo create every
  scene object with ``PhysxContactReportAPI`` (contact reporting only, no dynamics change);
* after each reset a :class:`RoboDojoAdapter` + :class:`SubstepMonitor` attach to the
  ``SimulationContext.step`` of the fresh scene; they finalize after ``run_eval``;
* ``--replay`` rebuilds the same layout (close + reset, no policy) and replays the
  recorded per-substep actuation from the episode start, reporting the position error of
  every logged body against the recorded trajectory; ``--public-restore`` additionally
  restores a mid-episode public snapshot in place and replays from there.

``--num-envs N`` (only for policies with ``eval_batch: true``) runs N layouts side by side as
``main.py`` does: ``env.reset(seed=[ids...])`` pads a short batch with ``None``, envs that fail
the stability check start ended, and ``run_eval`` scores the rest.  Each running env gets its
own adapter (bodies of that env only) and monitor; a monitor detaches when its env ends, so
every layout still has its own segment directory and record.  ``--replay`` then replays all
envs of the batch together (one physics step for the whole scene, as recorded).

``--policy scripted`` runs a built-in IK push probe instead of a policy server.

Run from ``/workspace/RoboDojo`` with ``PYTHONPATH=/workspace/RoboDojo:/workspace/RoboDojo/XPolicyLab:<simuguard>``::

    python -m simuguard.integrations.robodojo_eval --task put_bottles_into_dustbin --layouts 0 1 \\
        --policy scripted --out /runs/probe --replay all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

KIT_ENABLE_EXTS = ("isaacsim.replicator.behavior", "isaacsim.sensors.camera")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--layouts", nargs="+", default=[],
                    help="RoboDojo layout ids, or 'all' = every layout file of the eval seed, in id order "
                         "(also forced by EVAL_NUM_FORCE=all)")
    ap.add_argument("--max-counted", default="0",
                    help="stop after this many counted episodes: an int, 'official' (eval_nums of the task "
                         "config, as main.py) or 0 = no cap (run every listed layout)")
    ap.add_argument("--resume", action="store_true",
                    help="skip layouts already finished in <out>/<task>/episodes.jsonl (errors are retried)")
    ap.add_argument("--max-attempts", type=int, default=2, help="--resume: give up a layout after this many errors")
    ap.add_argument("--ckpt-name", default="", help="checkpoint dir name, for the official additional_info")
    ap.add_argument("--action-type", default="", help="joint | ee, for the official additional_info")
    ap.add_argument("--env-manifest", default=None, help="environment fingerprint JSON to embed in each run record")
    ap.add_argument("--replay-from", nargs="+", default=[],
                    help="replay recorded segment dirs (of --task) in this fresh process instead of running episodes")
    ap.add_argument("--policy", default="scripted", help="'scripted' or an XPolicyLab policy name (Xiaomi_Robotics_1, Pi_05)")
    ap.add_argument("--policy-url", default="", help="ws://host:port of the policy server")
    ap.add_argument("--additional-info", default=None,
                    help="default: ckpt_name=<ckpt>,action_type=<type> as eval.sh, or 'simuguard'")
    ap.add_argument("--env-cfg", default="arx_x5")
    ap.add_argument("--eval-seed", type=int, default=0, help="RoboDojo eval seed (layout set)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-envs", type=int, default=1,
                    help="envs side by side in one scene (eval_batch policies only, as main.py with "
                         "scene.num_envs): layouts run in batches of this size, one adapter + monitor per env")
    ap.add_argument("--monitor", choices=("on", "off"), default="on")
    ap.add_argument("--contact-report", choices=("on", "off"), default="on")
    ap.add_argument("--replay", choices=("none", "first", "all"), default="none")
    ap.add_argument("--public-restore", type=int, default=0,
                    help="also restore the public snapshot at this substep in place and replay 500 substeps (0: off)")
    ap.add_argument("--max-actions", type=int, default=0, help="scripted policy: stop after this many actions")
    ap.add_argument("--kick-speed", type=float, default=0.0,
                    help="scripted policy positive control: upward velocity (m/s) written to one resting object")
    ap.add_argument("--intervention", default="none",
                    help="'none' or 'depen_<m/s>': cap PhysX max depenetration velocity on every object body "
                         "(set at object creation; meant for --replay-from counterfactual replays)")
    ap.add_argument("--replay-detect", action="store_true",
                    help="run the detectors on replayed frames (reports recurring events and peak speeds)")
    ap.add_argument("--detector-config", default=None)
    ap.add_argument("--replay-frames", type=int, nargs="+", default=[],
                    help="--replay-from: save a camera image (PNG) after each of these control steps of the replay")
    ap.add_argument("--replay-camera", default="cam_head", help="camera for --replay-frames")
    ap.add_argument("--replay-speeds", action="store_true",
                    help="--replay-from: save the speed of every free body at every replayed substep (npz)")
    ap.add_argument("--no-cameras", action="store_true",
                    help="physics-only probe: no RTX rendering, RoboDojo camera managers stubbed out "
                         "(runs on GPUs without RT cores; NOT an official evaluation, scripted policy only)")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args(argv)
    args.headless = True
    args.enable_cameras = not args.no_cameras
    if args.no_cameras and args.policy != "scripted":
        ap.error("--no-cameras only makes sense with --policy scripted (policies need images)")
    args.kit_args = " ".join(f"--enable {ext}" for ext in KIT_ENABLE_EXTS) if not args.no_cameras else ""
    if os.environ.get("EVAL_NUM_FORCE", "").lower() == "all" and not args.replay_from:
        args.layouts = ["all"]
    if not args.layouts and not args.replay_from:
        ap.error("give --layouts or --replay-from")
    if args.layouts != ["all"]:
        args.layouts = [int(x) for x in args.layouts]
    if args.num_envs > 1 and args.replay_from:
        if len(args.replay_from) > args.num_envs or args.replay_detect or args.intervention != "none" \
                or args.replay_frames or args.replay_speeds:
            ap.error("--replay-from with --num-envs N: at most N one-env segments, plain replay only")
    elif args.num_envs > 1 and (args.policy == "scripted" or args.public_restore or args.intervention != "none"):
        ap.error("--num-envs > 1 is for policy episodes only (no scripted probe, public restore, intervention)")
    if args.additional_info is None:
        args.additional_info = (f"ckpt_name={args.ckpt_name},action_type={args.action_type}"
                                if args.ckpt_name else "simuguard")
    return args


def layout_files(env_cfg_name: str, eval_seed: int, task: str) -> dict[int, Path]:
    """Layout files of one eval seed: ``Assets/Eval_Layout/<bench>/<cfg>/<seed>/<task>_<i>.json``."""

    from env.global_configs import ASSETS_PATH, BENCHMARK

    folder = Path(ASSETS_PATH, "Eval_Layout", BENCHMARK, env_cfg_name, str(eval_seed))
    out: dict[int, Path] = {}
    for f in folder.glob(f"{task}_*.json"):
        tail = f.stem[len(task) + 1:]
        if tail.isdigit():
            out[int(tail)] = f
    return dict(sorted(out.items()))


def eval_batch_from_deploy(policy: str) -> bool:
    """``eval_batch`` of ``XPolicyLab/policy/<policy>/deploy.yml``, as ``main.py`` reads it."""

    from env.global_configs import ROOT_DIR
    from utils.load_file import load_yaml

    path = os.path.join(ROOT_DIR, "XPolicyLab", "policy", policy, "deploy.yml")
    return bool((load_yaml(path) if os.path.isfile(path) else {}).get("eval_batch", False))


# ---------------------------------------------------------------------------- env construction
class NullModelClient:
    """Stands in for the XPolicyLab websocket client when no policy server is used."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.calls: list[str] = []

    def call(self, func_name: str | None = None, obs: Any = None, **kwargs: Any) -> Any:
        self.calls.append(str(func_name))
        return None

    def close(self) -> None:
        pass


def build_env(args: argparse.Namespace, simulation_app: Any) -> tuple[Any, int]:
    """RoboDojo EvalEnv for one task, assembled like ``src/eval_client/main.py`` (``--num-envs``, default 1)."""

    import importlib

    from omegaconf import OmegaConf

    from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
    from src.eval_client import eval_env as eval_env_mod
    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_config, process_randomization

    task_registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
    benchmark_path = os.path.join(ROOT_DIR, "task", BENCHMARK)
    policy_name = args.policy if args.policy != "scripted" else "demo_policy"
    eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, args.env_cfg + ".yml"))
    eval_cfg["task_name"] = args.task
    num_envs = int(getattr(args, "num_envs", 1))
    eval_cfg["num_envs"] = num_envs
    eval_cfg["device_id"] = 0
    eval_cfg["eval_batch"] = eval_batch_from_deploy(policy_name) if args.policy != "scripted" else False
    if num_envs > 1 and not eval_cfg["eval_batch"]:
        raise ValueError(f"{policy_name}: eval_batch is false in deploy.yml, main.py runs it with one env")
    eval_cfg["policy_name"] = policy_name
    eval_cfg["additional_info"] = args.additional_info
    eval_cfg["seed"] = args.eval_seed
    eval_cfg["physx_monitor_enabled"] = False
    run_id = os.environ.setdefault("ROBODOJO_RUN_ID", time.strftime("%Y-%m-%d_%H-%M-%S"))
    url = args.policy_url or "ws://localhost:6000"
    deploy_cfg = {
        "policy_name": policy_name,
        "port": int(url.rsplit(":", 1)[-1]) if ":" in url else 6000,
        "host": url.split("//", 1)[-1].rsplit(":", 1)[0],
        "protocol": "ws",
        "policy_server_url": url,
        "evaluation_id": run_id,
        "trial_id": f"{args.task}-{run_id}",
        "action_case_id": f"{args.task}_case",
        "repeat_index": None,
    }
    env_cfg = OmegaConf.create(
        {
            "sim": load_yaml(os.path.join(ENV_CONFIG_PATH, "sim", eval_cfg["config"]["sim"] + ".yml")),
            "scene": load_yaml(os.path.join(ENV_CONFIG_PATH, "scene", eval_cfg["config"]["scene"] + ".yml")),
            "camera": load_yaml(os.path.join(ENV_CONFIG_PATH, "camera", eval_cfg["config"]["camera"] + ".yml")),
            "robot": load_yaml(os.path.join(ENV_CONFIG_PATH, "robot", eval_cfg["config"]["robot"] + ".yml")),
            "task_env": load_yaml(task_registry.task_config_path(os.path.join(benchmark_path, "config"), args.task)),
            "eval_cfg": eval_cfg,
            "deploy_cfg": deploy_cfg,
        }
    )
    OmegaConf.update(env_cfg, "sim.scene.num_envs", num_envs, force_add=True)
    OmegaConf.update(env_cfg, "eval_cfg.num_envs", num_envs, force_add=True)
    env_cfg = process_randomization(env_cfg)
    env_cfg, eval_num = process_config(env_cfg, task_name=args.task)
    eval_cfg["eval_num"] = eval_num
    OmegaConf.update(env_cfg, "camera.default_frequency", eval_cfg["observation"].get("collect_freq", 0), force_add=True)
    env_cfg.sim.seed = [0 for _ in range(num_envs)]
    if args.policy == "scripted" or args.replay_from:
        eval_env_mod.WsModelClient = NullModelClient  # no server: the scripted probe / the replay drives the env
    env = eval_env_mod.create_eval_env(env_cfg, simulation_app, resume_state=None)
    return env, int(eval_num)


class _NoCameras:
    """No-op stand-in for RoboDojo's CameraManager / TiledCaptureManager (physics-only probe)."""

    def __init__(self, num_envs: int = 1, *args: Any, **kwargs: Any) -> None:
        self.num_envs = int(num_envs)
        self.num_cams = 0
        self.camera_names = [[] for _ in range(self.num_envs)]

    def step(self, *args: Any, **kwargs: Any) -> list:
        return []

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


class CallProfiler:
    """Wall time per wrapped method (instance attributes; removed again by ``unwrap``).

    Timing only: the wrapped methods run unchanged.  Nested calls overlap (``env.step`` contains
    the physics substeps whose hook times the adapter reports separately).
    """

    def __init__(self) -> None:
        self.stats: dict[str, list] = {}
        self.wrapped: list[tuple[Any, str]] = []

    def wrap(self, obj: Any, attr: str, key: str, by_func: bool = False) -> None:
        if obj is None or attr in getattr(obj, "__dict__", {}):  # already wrapped, or an instance override
            return
        fn = getattr(obj, attr, None)
        if fn is None:
            return
        stats = self.stats

        def timed(*a: Any, **k: Any) -> Any:
            name = f"{key}:{k.get('func_name', a[0] if a else '?')}" if by_func else key
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                s = stats.setdefault(name, [0, 0.0])
                s[0] += 1
                s[1] += time.perf_counter() - t

        setattr(obj, attr, timed)
        self.wrapped.append((obj, attr))

    def wrap_env(self, env: Any) -> None:
        self.wrap(env.model_client, "call", "policy_call", by_func=True)
        self.wrap(env, "setup_scene", "reset.setup_scene")
        self.wrap(env, "step", "env.step")
        self.wrap(env, "take_action", "env.take_action")
        self.wrap(env, "get_obs", "env.get_obs")
        self.wrap(env, "_stream_vision", "obs.stream_video")
        self.wrap(env, "save_video", "save_video")
        obs = getattr(env, "obs_manager", None)
        self.wrap(obs, "render_for_capture", "obs.render_for_capture")
        self.wrap(obs, "get_obs", "obs.get_obs")

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {k: {"n": v[0], "s": round(v[1], 3)} for k, v in sorted(self.stats.items())}

    def unwrap(self) -> None:
        for obj, attr in reversed(self.wrapped):
            try:
                delattr(obj, attr)
            except AttributeError:
                pass
        self.wrapped.clear()


def install_no_cameras() -> None:
    import env.camera_manager.capture.tiled_capture_manager as capture_mod
    import env.environment.task_env as task_env_mod

    task_env_mod.CameraManager = _NoCameras
    capture_mod.TiledCaptureManager = _NoCameras


class use_null_client:
    """Temporarily swap the env's model client (replay resets must not talk to the policy)."""

    def __init__(self, env: Any) -> None:
        self.env = env

    def __enter__(self) -> None:
        self.saved = self.env.model_client
        self.env.model_client = NullModelClient()

    def __exit__(self, *exc: Any) -> None:
        self.env.model_client = self.saved


# ---------------------------------------------------------------------------- scripted probe policy
def scripted_episode(env: Any, adapter: Any, *, max_actions: int = 0, kick_speed: float = 0.0) -> dict[str, Any]:
    """Hold, then sweep one gripper through the first target object with IK joint actions, then hold.

    Deterministic given the scene: every decision uses the observation (joint/EE state) and
    the target's pose.  Joint actions are used so that exactly the IK solution is applied.
    ``kick_speed > 0`` adds a positive control after the hold: the last target gets an upward
    velocity written between two steps (a synthetic ejection; test runs only).
    """

    from simuguard.core.types import BodyRole

    log: dict[str, Any] = {"actions": 0, "ik_fail": 0, "phases": []}
    robots = {r.arm_name: r for r in env.robot_manager.robot_list if r.type == "target"}
    origin = np.asarray(env.sim.scene.env_origins[0].detach().cpu().numpy(), dtype=float)

    def obs_state() -> dict[str, Any]:
        return env.get_obs()["state"]

    def hold_action(state: dict[str, Any]) -> dict[str, Any]:
        return {k: np.asarray(v, dtype=float).tolist() for k, v in state.items() if k.endswith("_joint_state")}

    def act(action: dict[str, Any]) -> bool:
        if env.is_episode_end() or (max_actions and log["actions"] >= max_actions):
            return False
        env.take_action(action)
        log["actions"] += 1
        return True

    state = obs_state()
    log["phases"].append({"hold": 25})
    # contact cross-check while everything rests: our contact-report reading vs PhysX's net contact
    # force (rigid contact view), both for the last substep of each action
    free = [b for b, i in adapter.bodies().items() if i.kind.value == "dynamic"]
    try:
        net_reader = adapter.net_contact_force_reader(free)
    except Exception as exc:  # noqa: BLE001
        net_reader = None
        log["net_force_view_error"] = f"{type(exc).__name__}: {exc}"
    crosscheck: dict[str, list] = {b: [] for b in free}
    dt = adapter.timestep()
    for _ in range(25):
        if not act(hold_action(state)):
            return log
        if net_reader is None:
            continue
        pairs = adapter.read_contacts()
        net = net_reader()
        for b in free:
            ours = np.zeros(3)
            for pair in pairs:
                if pair.body_a == b:
                    ours += pair.total_impulse / dt
                elif pair.body_b == b:
                    ours -= pair.total_impulse / dt
            crosscheck[b].append([adapter.control_step(), *ours.tolist(), *net.get(b, np.full(3, np.nan)).tolist()])
    log["contact_crosscheck"] = {
        "columns": ["control_step", "ours_fx", "ours_fy", "ours_fz", "physx_net_fx", "physx_net_fy", "physx_net_fz"],
        "mass_kg": {b: adapter.bodies()[b].mass for b in free},
        "rows": crosscheck,
    }
    targets = adapter.body_ids_with_role(BodyRole.TARGET) or adapter.body_ids_with_role(BodyRole.OBJECT)
    targets = [t for t in targets if adapter.bodies()[t].kind.value == "dynamic"]
    if not targets:
        log["phases"].append({"skip": "no dynamic target"})
        return log
    if kick_speed > 0.0 and len(targets) > 1:
        # positive control (test only): throw a resting object that nothing touches except the table,
        # by overwriting its velocity between two steps; the monitor must record the write, the
        # detector must confirm the flight, the gravity and carrier stages must keep it
        kicked = targets[-1]
        velocity = [0.3, 0.0, float(kick_speed)]
        adapter.set_body_velocity(kicked, velocity)
        log["phases"].append({"kick": kicked, "velocity_mps": velocity, "control_step": adapter.control_step()})
    target = targets[0]
    p = adapter.read_states([target])[target].position - origin
    side = "left" if p[0] < 0.0 else "right"
    robot = robots[f"{side}_arm"]
    sign = 1.0 if side == "left" else -1.0
    state = obs_state()
    ee = np.asarray(state[f"{side}_ee_pose"], dtype=float)
    quat = ee[3:7].tolist()
    z = p[2] + 0.12
    waypoints = [
        (np.array([p[0] - sign * 0.12, p[1], z + 0.10]), 30),
        (np.array([p[0] - sign * 0.12, p[1], z]), 25),
        (np.array([p[0] + sign * 0.10, p[1], z]), 35),
        (np.array([p[0] + sign * 0.10, p[1], z + 0.12]), 25),
    ]
    log["phases"].append({"push": target, "side": side, "target_env_xyz": p.tolist(), "ee_start": ee.tolist()})
    current = ee[:3].copy()
    grip_key = f"{side}_ee_joint_state"
    for goal, steps in waypoints:
        start = current.copy()
        for i in range(1, steps + 1):
            pos = start + (goal - start) * (i / steps)
            state = obs_state()
            ik = env.robot_manager.solve_ik(target_pose=pos.tolist() + quat, env_idx=0, robot=robot)
            action = hold_action(state)
            if ik.get("status") == "Success":
                joints = ik["joint_value"]
                if hasattr(joints, "detach"):
                    joints = joints.detach().cpu().numpy()
                action[f"{side}_arm_joint_state"] = np.asarray(joints, dtype=float).reshape(-1).tolist()
            else:
                log["ik_fail"] += 1
            if goal is waypoints[2][0]:
                action[grip_key] = [0.0]  # close the gripper while sweeping (exercises gripper drives)
            if not act(action):
                return log
        current = goal
    state = obs_state()
    for _ in range(40):
        if not act(hold_action(state)):
            break
    return log


# ---------------------------------------------------------------------------- replay checks
def _comparable_ids(adapter: Any, logged: list[str]) -> list[str]:
    """Logged bodies that have a state (static colliders are logged as NaN and have none)."""

    bodies = adapter.bodies()
    return [b for b in logged if b in bodies and bodies[b].kind.value != "static"]


def replay_recorded(
    env: Any,
    adapter: Any,
    episode_dir: Path,
    *,
    limit: int | None = None,
    detectors: list[Any] | None = None,
    before_step: Any = None,
    frames: tuple[set[int], Path, str] | None = None,
    speeds_out: Path | None = None,
) -> dict[str, Any]:
    """Replay ``controls.npz`` on the (already rebuilt) scene; compare with ``states.npz``.

    ``frames`` = (control steps, folder, camera): render and save that camera after each listed control
    step (rendering does not step physics). ``speeds_out``: npz with the speed of every free body at
    every replayed substep.

    With ``detectors`` the replayed frames (states + contacts) also go through them, so an
    intervention replay reports whether the recorded events recur and how fast the free bodies get.
    """

    from simuguard.core.controls import ControlLog
    from simuguard.core.snapshot import Snapshot
    from simuguard.core.statelog import StateLog

    controls = ControlLog.load(episode_dir / "controls.npz")
    ref = StateLog.load(episode_dir / "states.npz")
    initial = Snapshot.load(episode_dir / "initial_snapshot.json.gz")
    report = adapter.restore_snapshot(initial, method="none")
    ids = _comparable_ids(adapter, ref.body_ids)
    arr = ref.array()
    row = {int(s): i for i, s in enumerate(ref.substeps)}
    col = {b: ref.body_ids.index(b) for b in ids}
    free = [b for b in ids if adapter.bodies()[b].kind.value in ("dynamic", "kinematic")]
    newest = controls.newest_substep or 0
    last = newest if limit is None else min(newest, limit)
    worst = 0.0
    worst_free = 0.0
    first: dict[str, Any] | None = None
    curve: list[list[float]] = []
    context = None
    events: dict[str, Any] = {}
    peak_speed = {b: [0.0, 0] for b in free}
    if detectors:
        from simuguard.core.detectors.base import DetectorContext

        context = DetectorContext(episode_id=f"replay:{episode_dir.name}", timestep=adapter.timestep(), bodies=adapter.bodies())
        for detector in detectors:
            detector.reset(context)
        tracked = sorted(b for b, i in adapter.bodies().items() if i.role.value in ("target", "container", "object", "robot"))
    started = time.time()
    frame = None
    current = {"control_step": 0}
    adapter._control_step_fn = lambda: current["control_step"]  # replayed frames carry the recorded step
    speed_rows: list[list[float]] = []
    speed_steps: list[int] = []
    prev_cs = None
    for record in controls.between(0, last):
        if frames is not None and prev_cs is not None and record.control_step != prev_cs and prev_cs in frames[0]:
            _save_camera(env, frames[2], frames[1] / f"cs{prev_cs:05d}.png")
        prev_cs = record.control_step
        current["control_step"] = record.control_step
        if before_step is not None:
            before_step(record.substep)
        adapter.apply_control(record)
        adapter.step_physics()
        k = record.substep
        if context is not None:
            frame = adapter.read_frame(k, tracked, with_contacts=True)
            for detector in detectors:
                for event in detector.observe(frame, context):
                    events[event.event_id] = event
            context.previous = frame
            for b in free:
                s = frame.states.get(b)
                if s is not None and s.speed > peak_speed[b][0]:
                    peak_speed[b] = [s.speed, k]
        if speeds_out is not None:
            st = adapter.read_states(free)
            speed_rows.append([float(np.linalg.norm(st[b].linear_velocity)) for b in free])
            speed_steps.append(k)
        if k not in row:
            continue
        now = adapter.read_states(ids)
        errs = {b: float(np.linalg.norm(now[b].position - arr[row[k], col[b], :3])) for b in ids}
        b_max = max(errs, key=errs.get)
        worst = max(worst, errs[b_max])
        worst_free = max([worst_free] + [errs[b] for b in free])
        if errs[b_max] > 0.0 and first is None:
            first = {"substep": k, "control_step": record.control_step, "body": b_max, "error_m": errs[b_max]}
        if k % 100 == 0 or k == last:
            curve.append([k, errs[b_max], max((errs[b] for b in free), default=0.0)])
    if frames is not None and prev_cs is not None and prev_cs in frames[0]:
        _save_camera(env, frames[2], frames[1] / f"cs{prev_cs:05d}.png")
    if speeds_out is not None:
        np.savez_compressed(speeds_out, substeps=np.asarray(speed_steps), bodies=np.asarray(free),
                            speed_mps=np.asarray(speed_rows, dtype=np.float64))
    result = {
        "initial_public_state_error": report.public_state_error,
        "substeps": last,
        "bodies_compared": len(ids),
        "free_bodies_compared": len(free),
        "max_position_error_m": worst,
        "max_free_body_position_error_m": worst_free,
        "bit_exact": worst == 0.0,
        "first_divergence": first,
        "curve_every_100": curve,
        "wall_s": time.time() - started,
    }
    if context is not None:
        for detector in detectors:
            for event in detector.finalize(frame, context):
                events[event.event_id] = event
        result["events"] = [e.to_dict() for e in events.values()]
        result["confirmed"] = [
            {k: e.to_dict()[k] for k in ("detector", "onset_substep", "control_step", "bodies", "reasons")}
            | {"max_speed_mps": e.metrics.get("max_speed_mps")}
            for e in events.values() if e.status.value == "confirmed"
        ]
        result["peak_speed_mps"] = {b: {"speed": v[0], "substep": v[1]} for b, v in peak_speed.items()}
    return result


def _save_camera(env: Any, camera: str, path: Path, renders: int = 3) -> None:
    """Render (no physics step) and save one camera's RGB image of env 0 as PNG."""

    for _ in range(renders):
        env.render()
    vision = env.obs_manager.get_obs(env_idx_list=[0])[0]["vision"]
    name = next((n for n in vision if camera in n), next(iter(vision)))
    rgb = np.asarray(vision[name]["color"])[:, :, :3].astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(rgb).save(path)


def final_state_digest(adapter: Any) -> dict[str, list[float]]:
    """End-of-episode pose (xyz, wxyz) and velocity of every rigid body the adapter views, read fresh."""

    adapter._invalidate()
    return {
        b: [float(v) for v in (*s.position, *s.quaternion, *s.linear_velocity, *s.angular_velocity)]
        for b, s in adapter.read_states().items()
    }


def public_restore_check(adapter: Any, monitor: Any, episode_dir: Path, substep: int, horizon: int = 500) -> dict[str, Any]:
    """Restore the public snapshot at ``substep`` in place (end-of-episode scene) and replay."""

    from simuguard.core.controls import ControlLog
    from simuguard.core.statelog import StateLog

    snapshot = monitor.snapshots.latest_at_or_before(substep)
    if snapshot is None:
        return {"error": f"no snapshot at or before {substep}"}
    controls = ControlLog.load(episode_dir / "controls.npz")
    ref = StateLog.load(episode_dir / "states.npz")
    report = adapter.restore_snapshot(snapshot, method="public")
    ids = _comparable_ids(adapter, ref.body_ids)
    free = [b for b in ids if adapter.bodies()[b].kind.value in ("dynamic", "kinematic")]
    arr = ref.array()
    row = {int(s): i for i, s in enumerate(ref.substeps)}
    col = {b: ref.body_ids.index(b) for b in ids}
    end = min(snapshot.substep + horizon, controls.newest_substep or snapshot.substep)
    worst, worst_free, first = 0.0, 0.0, None
    for record in controls.between(snapshot.substep, end):
        adapter.apply_control(record)
        adapter.step_physics()
        if record.substep not in row:
            continue
        now = adapter.read_states(ids)
        errs = {b: float(np.linalg.norm(now[b].position - arr[row[record.substep], col[b], :3])) for b in ids}
        b_max = max(errs, key=errs.get)
        worst = max(worst, errs[b_max])
        worst_free = max([worst_free] + [errs[b] for b in free])
        if first is None and errs[b_max] > 1e-6:
            first = {"substep": record.substep, "body": b_max, "error_m": errs[b_max]}
    return {
        "snapshot_substep": snapshot.substep,
        "replayed_to": end,
        "restore_public_state_error": report.public_state_error,
        "max_position_error_m": worst,
        "max_free_body_position_error_m": worst_free,
        "first_over_1um": first,
    }


# ---------------------------------------------------------------------------- episode loop
def monitor_config() -> Any:
    from simuguard.core.monitor import MonitorConfig

    return MonitorConfig.from_dict(
        {
            "snapshot_interval_substeps": 100,
            "snapshot_capacity": 5000,
            "frame_buffer_substeps": 2000,
            "control_log_maxlen": None,
            "bundle_mode": "episode_start",
            "state_log_roles": ["target", "container", "object", "robot"],
        }
    )


def detector_config(path: str | None) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if path:
        cfg = json.loads(Path(path).read_text())
    ejection = cfg.setdefault("ejection", {})
    ejection.setdefault("delta_v_reference_timestep_s", 0.004)  # RoboDojo runs at 4 ms: thresholds hold as given
    return cfg


def replay_recorded_batch(env: Any, adapters: dict[int, Any], seg_dirs: dict[int, Path]) -> dict[int, dict[str, Any]]:
    """Batched counterpart of :func:`replay_recorded`: replay every env's ``controls.npz`` side by side.

    The envs of one scene share every physics step, so the recorded actuation of all monitored envs is
    applied before each raw step (an env whose episode ended keeps its last targets, as it did in the
    run) and each env's logged bodies are compared with its own ``states.npz``.
    """

    from simuguard.core.controls import ControlLog
    from simuguard.core.snapshot import Snapshot
    from simuguard.core.statelog import StateLog

    started = time.time()
    plan: dict[int, dict[str, Any]] = {}
    for k, adapter in adapters.items():
        seg = seg_dirs[k]
        controls = ControlLog.load(seg / "controls.npz")
        ref = StateLog.load(seg / "states.npz")
        report = adapter.restore_snapshot(Snapshot.load(seg / "initial_snapshot.json.gz"), method="none")
        ids = _comparable_ids(adapter, ref.body_ids)
        newest = controls.newest_substep or 0
        plan[k] = {
            "controls": {r.substep: r for r in controls.between(0, newest)}, "last": newest, "ids": ids,
            "arr": ref.array(), "row": {int(s): i for i, s in enumerate(ref.substeps)},
            "col": {b: ref.body_ids.index(b) for b in ids},
            "free": [b for b in ids if adapter.bodies()[b].kind.value in ("dynamic", "kinematic")],
            "worst": 0.0, "worst_free": 0.0, "first": None, "curve": [], "initial": report.public_state_error,
        }
    last = max((p["last"] for p in plan.values()), default=0)
    stepper = next(iter(adapters.values()))
    for k_sub in range(1, last + 1):
        for k, adapter in adapters.items():
            record = plan[k]["controls"].get(k_sub)
            if record is not None:
                adapter.apply_control(record)
        stepper.step_physics()
        for k, adapter in adapters.items():
            if adapter is not stepper:
                adapter._invalidate()
                adapter._internal_step += 1
            p = plan[k]
            if k_sub > p["last"] or k_sub not in p["row"]:
                continue
            now = adapter.read_states(p["ids"])
            errs = {b: float(np.linalg.norm(now[b].position - p["arr"][p["row"][k_sub], p["col"][b], :3])) for b in p["ids"]}
            b_max = max(errs, key=errs.get)
            p["worst"] = max(p["worst"], errs[b_max])
            p["worst_free"] = max([p["worst_free"]] + [errs[b] for b in p["free"]])
            if errs[b_max] > 0.0 and p["first"] is None:
                p["first"] = {"substep": k_sub, "body": b_max, "error_m": errs[b_max]}
            if k_sub % 100 == 0 or k_sub == p["last"]:
                p["curve"].append([k_sub, errs[b_max], max((errs[b] for b in p["free"]), default=0.0)])
    wall = time.time() - started
    return {k: {"initial_public_state_error": p["initial"], "substeps": p["last"], "bodies_compared": len(p["ids"]),
                "free_bodies_compared": len(p["free"]), "max_position_error_m": p["worst"],
                "max_free_body_position_error_m": p["worst_free"], "bit_exact": p["worst"] == 0.0,
                "first_divergence": p["first"], "curve_every_100": p["curve"], "wall_s": wall,
                "batched_with_envs": sorted(adapters)}
            for k, p in plan.items()}


def run_batched(args: argparse.Namespace, env: Any, out: Path, files: dict[int, Path], layouts: list[int],
                done: dict[int, dict[str, Any]], attempts: dict[int, int], max_counted: int, counted: int,
                det_cfg: dict[str, Any]) -> None:
    """``--num-envs N``: the layouts in batches of N side by side, as ``main.py`` runs eval_batch policies.

    One ``env.reset(seed=[l_0 .. l_N-1])`` -> ``env.run_eval()`` (the policy's ``eval_one_episode_batch``)
    per batch; a short last batch is padded with ``None`` seeds like ``main.py`` (those envs start ended).
    Every env gets its own :class:`RoboDojoAdapter` (that env's bodies only) and :class:`SubstepMonitor`
    with its own segment dir, so each layout is recorded and detected exactly as a one-env episode.  A
    monitor detaches when its env's ``end_flag`` is set (the scene keeps stepping for the other envs;
    in a one-env run physics stops there).  One ``episodes.jsonl`` record per layout, as before.
    """

    import hashlib

    from simuguard.adapters.isaac.robodojo import RoboDojoAdapter, physics_settings
    from simuguard.core.monitor import SubstepMonitor
    from simuguard.core.recorder import EpisodeRecorder
    from simuguard.presets import default_detectors
    from utils.cluttered_generator import UnStableError

    n_envs = int(env.num_envs)
    episodes_log = out / "episodes.jsonl"
    pending = [lid for lid in layouts if lid not in done]
    batch_no = -1
    while pending:
        if max_counted and counted >= max_counted:
            break
        take = n_envs if not max_counted else min(n_envs, max_counted - counted)
        batch, pending = pending[:take], pending[take:]
        batch_no += 1
        seeds: list[int | None] = list(batch) + [None] * (n_envs - len(batch))
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        batch_id = f"{started}:{'-'.join(map(str, batch))}"
        records: dict[int, dict[str, Any]] = {}
        segs: dict[int, Path] = {}
        for k, layout in enumerate(batch):
            rec: dict[str, Any] = {"task": args.task, "layout": layout, "policy": args.policy, "monitor": args.monitor,
                                   "attempt": attempts.get(layout, 0) + 1, "started_utc": started,
                                   "num_envs": n_envs, "env_idx": k, "batch": list(seeds), "batch_id": batch_id}
            if layout in files:
                rec["layout_file"] = files[layout].name
                rec["layout_sha256"] = hashlib.sha256(files[layout].read_bytes()).hexdigest()
            rec["status"] = "error"
            seg = out / "segments" / (f"layout{layout:03d}" if rec["attempt"] == 1 else
                                      f"layout{layout:03d}_a{rec['attempt']}")
            if seg.exists():
                seg.rename(seg.with_name(f"{seg.name}_stale{int(time.time())}"))
            rec["segment"] = str(seg)
            records[k], segs[k] = rec, seg
        current = out / "current_layout.json"
        current.write_text(json.dumps({"layouts": batch, "attempts": [records[k]["attempt"] for k in records],
                                       "pid": os.getpid(), "started_utc": started}))
        evaluated_before = int(env.success_nums + env.fail_nums)
        unstable_before = int(env.unstable_nums)
        adapters: dict[int, Any] = {}
        monitors: dict[int, Any] = {}
        digests: dict[int, Any] = {}
        ended_at: dict[int, int] = {}
        prof = CallProfiler()
        try:
            t0 = time.time()
            prof.wrap_env(env)
            try:
                env.reset(seed=list(seeds))
            except UnStableError as exc:  # every env of the batch failed check_layout_stability
                for rec in records.values():
                    rec["status"] = "unstable_layout"
                    rec["error"] = f"unstable layout: {exc}"
                continue
            reset_wall = time.time() - t0
            reset_profile = prof.snapshot()
            prof.stats.clear()
            prof.wrap_env(env)
            physics = physics_settings(env) if batch_no == 0 else None
            for k, rec in records.items():
                rec["reset_wall_s"] = reset_wall
                rec["profile_reset"] = reset_profile
                if physics is not None:
                    rec["physics"] = physics
                if env.end_flag[k]:  # check_layout_stability dropped this env (main.py: unstable, not counted)
                    rec["status"] = "unstable_layout"
                    rec["error"] = "unstable layout: check_layout_stability (batched)"
            running = [k for k in records if not env.end_flag[k]]
            if args.monitor == "on":
                for k in running:
                    adapters[k] = RoboDojoAdapter(env, task_name=args.task, env_idx=k)
                    monitors[k] = SubstepMonitor(
                        adapters[k], default_detectors(det_cfg), episode_id=f"{args.task}:layout{batch[k]}",
                        config=monitor_config(), recorder=EpisodeRecorder(segs[k]),
                        metadata={"phase": "policy", "seed": batch[k], "task": args.task, "policy": args.policy,
                                  "env_idx": k, "num_envs": n_envs, "batch": list(seeds)})
                for k in running:
                    monitors[k].attach()
            official_end = env.is_episode_end

            def is_episode_end() -> bool:  # instance attribute: detach a monitor when its env ends
                result = official_end()
                for k, monitor in monitors.items():
                    if env.end_flag[k] and k not in ended_at:
                        monitor.detach()
                        ended_at[k] = int(monitor.substep)
                        digests[k] = final_state_digest(adapters[k])
                return result

            env.is_episode_end = is_episode_end
            t1 = time.time()
            env.run_eval()
            episode_wall = time.time() - t1
            episode_profile = prof.snapshot()
            del env.is_episode_end
            details = env.eval_result.get("details", {})
            evaluated_after = int(env.success_nums + env.fail_nums)
            by_layout = {}
            for idx in range(evaluated_before, evaluated_after):
                d = details.get(idx, details.get(str(idx)))
                if d is not None:
                    by_layout[int(d["layout_id"])] = (idx, d)
            for k in running:
                rec, layout = records[k], batch[k]
                rec["episode_wall_s"] = episode_wall
                rec["profile_episode"] = episode_profile
                rec["actions"] = int(env.take_action_cnt[k])
                rec["video_dir"] = str(Path(env.save_dir).resolve())
                if layout in by_layout:
                    idx, d = by_layout[layout]
                    rec["status"] = "counted"
                    rec["detail_index"] = idx
                    rec["video"] = str(Path(env.save_dir).resolve() / f"episode_{idx:07d}.mp4")
                    rec["success"] = bool(d.get("success", False))
                    rec["score"] = d.get("score")
                    counted += 1
                else:  # mark_env_unstable during the episode: dropped from the eval set, like main.py
                    rec["status"] = "unstable_episode"
                    rec["success"] = None
                    rec["score"] = None
                rec["official_eval_time"] = evaluated_after
                rec["official_success_rate"] = env.eval_result.get("success_rate")
                rec["official_score"] = env.eval_result.get("score")
                if k in monitors:
                    monitor = monitors[k]
                    if k not in ended_at:  # should not happen: run_eval returns once every env ended
                        monitor.detach()
                        ended_at[k] = int(monitor.substep)
                        digests[k] = final_state_digest(adapters[k])
                    rec["final_states"] = digests[k]
                    rec["monitor_detached_at_substep"] = ended_at[k]
                    monitor.metadata["outcome"] = {"eval_success": rec["success"], "score": rec["score"],
                                                   "actions": rec["actions"]}
                    summary = monitor.finalize()
                    rec["substeps"] = summary["substeps"]
                    rec["confirmed"] = summary["confirmed_count"]
                    rec["flags"] = summary["flag_count"]
                    rec["monitor_errors"] = summary["error_count"]
                    rec["events_by_detector"] = summary["event_counts_by_detector"]
                    rec["hook"] = adapters[k].hook_statistics()
            for adapter in adapters.values():
                adapter.close()
            do_replay = args.monitor == "on" and monitors and (
                args.replay == "all" or (args.replay == "first" and batch_no == 0))
            if do_replay:
                env.close()
                t2 = time.time()
                with use_null_client(env):
                    env.reset(seed=list(seeds))
                replay_adapters = {k: RoboDojoAdapter(env, task_name=args.task, env_idx=k) for k in monitors}
                replay_reset = time.time() - t2
                for k, result in replay_recorded_batch(env, replay_adapters, {k: segs[k] for k in monitors}).items():
                    records[k]["replay_reset_wall_s"] = replay_reset
                    records[k]["replay"] = result
                for adapter in replay_adapters.values():
                    adapter.close()
        except Exception as exc:  # noqa: BLE001
            for rec in records.values():
                if rec["status"] == "error":
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                    rec["traceback"] = traceback.format_exc()
            for monitor in monitors.values():
                try:
                    monitor.finalize()
                except Exception:  # noqa: BLE001
                    pass
        finally:
            prof.unwrap()
            vars(env).pop("is_episode_end", None)
            delta = int(env.unstable_nums) - unstable_before
            finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with episodes_log.open("a", encoding="utf-8") as fh:
                for rec in records.values():
                    rec["unstable_nums_delta_batch"] = delta
                    rec["finished_utc"] = finished
                    fh.write(json.dumps(rec, default=str) + "\n")
            current.unlink(missing_ok=True)
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass
        for rec in records.values():
            print(f"[simuguard] {args.task} layout {rec['layout']} (env {rec['env_idx']}/{n_envs}): " + json.dumps(
                {k: rec.get(k) for k in ("status", "success", "score", "confirmed", "flags", "substeps",
                                         "episode_wall_s", "error")}, default=str), flush=True)
            if rec.get("replay"):
                r = rec["replay"]
                print(f"[simuguard]   replay: max err {r['max_position_error_m']:.3e} m, bit_exact={r['bit_exact']}, "
                      f"first={r['first_divergence']}", flush=True)


def run(args: argparse.Namespace) -> int:
    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app
    from simuguard import __version__
    from simuguard.adapters.isaac.robodojo import RoboDojoAdapter, install_contact_reporting, physics_settings
    from simuguard.core.monitor import SubstepMonitor
    from simuguard.core.recorder import EpisodeRecorder
    from simuguard.presets import default_detectors
    from utils.cluttered_generator import UnStableError

    out = Path(args.out) / args.task
    (out / "segments").mkdir(parents=True, exist_ok=True)
    depen, depen_from = parse_intervention(args.intervention)
    patched = []
    if args.contact_report == "on" or depen is not None:
        patched = install_contact_reporting(
            contact_report=args.contact_report == "on",
            max_depenetration_velocity=depen if depen_from is None else None,  # from creation
            physx_rigid_body_api=depen_from is not None,  # switched on at substep depen_from during the replay
        )
    if args.no_cameras:
        install_no_cameras()
    env, eval_num = build_env(args, app)
    det_cfg = detector_config(args.detector_config)
    episodes_log = out / "episodes.jsonl"
    files = layout_files(args.env_cfg, args.eval_seed, args.task)
    layouts = list(files) if args.layouts == ["all"] else list(args.layouts)
    if args.max_counted == "official":
        max_counted = int(eval_num)
    else:
        max_counted = int(args.max_counted)
    done: dict[int, dict[str, Any]] = {}
    attempts: dict[int, int] = {}
    if args.resume and episodes_log.exists():
        for line in episodes_log.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            lid = int(rec["layout"])
            if rec.get("status") in ("counted", "unstable_layout", "unstable_episode"):
                done[lid] = rec
            else:
                attempts[lid] = attempts.get(lid, 0) + 1
                if attempts[lid] >= args.max_attempts:
                    done[lid] = rec
    counted_before = sum(1 for r in done.values() if r.get("status") == "counted")
    run_info = {
        "task": args.task, "policy": args.policy, "layouts": layouts, "monitor": args.monitor,
        "layout_spec": args.layouts, "max_counted": max_counted, "layout_files": len(files),
        "eval_batch": bool(env.eval_batch), "num_envs": int(env.num_envs),
        "additional_info": args.additional_info, "eval_seed": args.eval_seed,
        "eval_num_force": os.environ.get("EVAL_NUM_FORCE"),
        "count_rule": ("every layout file of the eval seed (EVAL_NUM_FORCE=all / --layouts all), overriding "
                       "the task's eval_nums" if args.layouts == ["all"] else "explicit layout list"),
        "resumed_layouts": sorted(done), "save_dir": str(Path(env.save_dir).resolve()),
        "contact_report": args.contact_report, "contact_patch": patched, "replay": args.replay,
        "no_cameras": bool(args.no_cameras), "intervention": args.intervention,
        "simuguard": __version__, "eval_num_official": eval_num, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "robodojo_run_id": os.environ.get("ROBODOJO_RUN_ID"),
    }
    if args.env_manifest:
        run_info["environment"] = json.loads(Path(args.env_manifest).read_text())
    (out / "run.json").write_text(json.dumps(run_info, indent=2))
    with (out / "runs.jsonl").open("a", encoding="utf-8") as fh:  # one line per process (resume history)
        fh.write(json.dumps(run_info, default=str) + "\n")
    if args.replay_from:
        if int(env.num_envs) > 1:
            replay_cross_batch(args, env, out)
        else:
            replay_only(args, env, out)
        app.close()
        return 0
    import hashlib

    if int(env.num_envs) > 1:
        run_batched(args, env, out, files, layouts, done, attempts, max_counted, counted_before, det_cfg)
        try:
            env.model_client.close()
        except Exception:  # noqa: BLE001
            pass
        app.close()
        return 0

    counted = counted_before
    n = -1
    for layout in layouts:
        if layout in done:
            continue
        if max_counted and counted >= max_counted:
            break
        n += 1
        record: dict[str, Any] = {"task": args.task, "layout": layout, "policy": args.policy, "monitor": args.monitor,
                                  "attempt": attempts.get(layout, 0) + 1, "started_utc": time.strftime(
                                      "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if layout in files:
            record["layout_file"] = files[layout].name
            record["layout_sha256"] = hashlib.sha256(files[layout].read_bytes()).hexdigest()
        evaluated_before = int(env.success_nums + env.fail_nums)
        unstable_before = int(env.unstable_nums)
        record["status"] = "error"
        seg = out / "segments" / (f"layout{layout:03d}" if record["attempt"] == 1 else
                                  f"layout{layout:03d}_a{record['attempt']}")  # never reuse a crashed try's dir
        if seg.exists():  # a preempted try of the same attempt number: keep it aside
            seg.rename(seg.with_name(f"{seg.name}_stale{int(time.time())}"))
        record["segment"] = str(seg)
        monitor = adapter = None
        current = out / "current_layout.json"  # lets a supervisor attribute a hard crash to this layout
        current.write_text(json.dumps({"layout": layout, "attempt": record["attempt"], "pid": os.getpid(),
                                       "started_utc": record["started_utc"]}))
        prof = CallProfiler()
        try:
            t0 = time.time()
            env.env_seeds = [layout]
            prof.wrap_env(env)
            env.reset(seed=[layout])
            record["reset_wall_s"] = time.time() - t0
            record["profile_reset"] = prof.snapshot()
            prof.stats.clear()
            prof.wrap_env(env)  # objects rebuilt by reset get wrapped too
            if n == 0:
                record["physics"] = physics_settings(env)
            if args.monitor == "on":
                adapter = RoboDojoAdapter(env, task_name=args.task)
                monitor = SubstepMonitor(
                    adapter,
                    default_detectors(det_cfg),
                    episode_id=f"{args.task}:layout{layout}",
                    config=monitor_config(),
                    recorder=EpisodeRecorder(seg),
                    metadata={"phase": "policy", "seed": layout, "task": args.task, "policy": args.policy},
                )
                monitor.attach()
            t1 = time.time()
            if args.policy == "scripted":
                probe_adapter = adapter or RoboDojoAdapter(env, task_name=args.task, physics_step_counter=False)
                record["scripted"] = None

                def scripted(_env: Any = env, _adapter: Any = probe_adapter) -> None:
                    record["scripted"] = scripted_episode(_env, _adapter, max_actions=args.max_actions,
                                                          kick_speed=args.kick_speed)

                env.eval_one_episode = scripted  # instance attribute: replaces the XPolicyLab module call
            env.run_eval()
            record["episode_wall_s"] = time.time() - t1
            record["profile_episode"] = prof.snapshot()
            reader = adapter
            if reader is None and args.policy == "scripted":
                reader = probe_adapter
            if reader is not None:  # lets runs with and without the monitor be compared bit for bit
                record["final_states"] = final_state_digest(reader)
            details = env.eval_result.get("details", {})
            evaluated_after = int(env.success_nums + env.fail_nums)
            if evaluated_after > evaluated_before:  # run_eval wrote details[evaluated_before] for this episode
                record["status"] = "counted"
                last = details.get(evaluated_before, details.get(str(evaluated_before), {}))
                record["detail_index"] = evaluated_before
                record["video"] = str(Path(env.save_dir).resolve() / f"episode_{evaluated_before:07d}.mp4")
                if int(last.get("layout_id", layout)) != layout:
                    raise RuntimeError(f"details[{evaluated_before}] is layout {last.get('layout_id')}, not {layout}")
                counted += 1
            else:  # mark_env_unstable during the episode: dropped from the eval set, like main.py
                record["status"] = "unstable_episode"
                last = {}
            record["success"] = bool(last.get("success", False)) if last else None
            record["score"] = last.get("score")
            record["official_eval_time"] = evaluated_after
            record["official_success_rate"] = env.eval_result.get("success_rate")
            record["official_score"] = env.eval_result.get("score")
            record["actions"] = int(env.take_action_cnt[0])
            record["video_dir"] = str(Path(env.save_dir).resolve())
            if monitor is not None:
                monitor.metadata["outcome"] = {"eval_success": record["success"], "score": record["score"],
                                               "actions": record["actions"]}
                summary = monitor.finalize()
                record["substeps"] = summary["substeps"]
                record["confirmed"] = summary["confirmed_count"]
                record["flags"] = summary["flag_count"]
                record["monitor_errors"] = summary["error_count"]
                record["events_by_detector"] = summary["event_counts_by_detector"]
                record["hook"] = adapter.hook_statistics()
                if args.public_restore:
                    record["public_restore"] = public_restore_check(adapter, monitor, seg, args.public_restore)
                adapter.close()
            do_replay = args.monitor == "on" and (args.replay == "all" or (args.replay == "first" and n == 0))
            if do_replay:
                env.close()
                t2 = time.time()
                with use_null_client(env):
                    env.env_seeds = [layout]
                    env.reset(seed=[layout])
                replay_adapter = RoboDojoAdapter(env, task_name=args.task)
                record["replay_reset_wall_s"] = time.time() - t2
                record["replay"] = replay_recorded(env, replay_adapter, seg)
                replay_adapter.close()
        except UnStableError as exc:  # check_layout_stability failed: the id is used up, not counted (main.py)
            record["status"] = "unstable_layout"
            record["error"] = f"unstable layout: {exc}"
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
            if monitor is not None:
                try:
                    monitor.finalize()
                except Exception:  # noqa: BLE001
                    pass
        finally:
            prof.unwrap()
            record["unstable_nums_delta"] = int(env.unstable_nums) - unstable_before
            record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with episodes_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
            current.unlink(missing_ok=True)
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass
        print(f"[simuguard] {args.task} layout {layout}: " + json.dumps(
            {k: record.get(k) for k in ("status", "success", "score", "confirmed", "flags", "substeps", "episode_wall_s",
                                        "error")}, default=str),
            flush=True)
        if record.get("replay"):
            r = record["replay"]
            print(f"[simuguard]   replay: max err {r['max_position_error_m']:.3e} m (free bodies "
                  f"{r['max_free_body_position_error_m']:.3e}), bit_exact={r['bit_exact']}, first={r['first_divergence']}",
                  flush=True)
    try:
        env.model_client.close()
    except Exception:  # noqa: BLE001
        pass
    app.close()
    return 0


def parse_intervention(text: str) -> tuple[float | None, int | None]:
    """'none' | 'depen_<m/s>' (from creation) | 'depen_<m/s>@<substep>' (switched on during replay)."""

    if text == "none":
        return None, None
    if not text.startswith("depen_"):
        raise ValueError(f"unknown intervention {text}")
    value, _, start = text[len("depen_"):].partition("@")
    return float(value), (int(start) if start else None)


def replay_only(args: argparse.Namespace, env: Any, out: Path) -> None:
    """Fresh-process replay: rebuild each recorded segment's layout and replay its actuation."""

    from simuguard.adapters.isaac.robodojo import RoboDojoAdapter

    log = out / "replays.jsonl"
    for seg_path in args.replay_from:
        seg = Path(seg_path)
        record: dict[str, Any] = {"segment": str(seg)}
        try:
            meta = json.loads((seg / "manifest.json").read_text())["metadata"]
            layout = int(meta["seed"])
            record["layout"] = layout
            if meta.get("task") != args.task:
                raise ValueError(f"segment task {meta.get('task')} != --task {args.task}")
            t0 = time.time()
            with use_null_client(env):
                env.env_seeds = [layout]
                env.reset(seed=[layout])
            record["reset_wall_s"] = time.time() - t0
            adapter = RoboDojoAdapter(env, task_name=args.task, physics_step_counter=False)
            detectors = None
            if args.replay_detect:
                from simuguard.presets import default_detectors

                detectors = default_detectors(detector_config(args.detector_config))
            record["intervention"] = args.intervention
            depen, depen_from = parse_intervention(args.intervention)
            before_step = None
            if depen is not None and depen_from is not None:
                def before_step(substep: int, _a: Any = adapter, _v: float = depen, _s: int = depen_from) -> None:
                    if substep == _s:
                        record["intervention_bodies"] = _a.set_max_depenetration_velocity(_v)
            frames = (set(args.replay_frames), out / f"frames_{seg.name}", args.replay_camera) if args.replay_frames else None
            speeds_out = out / f"speeds_{seg.name}.npz" if args.replay_speeds else None
            record["replay"] = replay_recorded(env, adapter, seg, detectors=detectors, before_step=before_step,
                                               frames=frames, speeds_out=speeds_out)
            adapter.close()
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
        finally:
            with log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass
        r = record.get("replay") or {}
        print(f"[simuguard] fresh-process replay {seg.name} ({args.intervention}): bit_exact={r.get('bit_exact')} "
              f"max err {r.get('max_position_error_m')} first={r.get('first_divergence')} error={record.get('error')}",
              flush=True)
        if "confirmed" in r:
            print(f"[simuguard]   replayed confirmed events: {r['confirmed']}", flush=True)


def replay_cross_batch(args: argparse.Namespace, env: Any, out: Path) -> None:
    """Replay one-env segments side by side in an N-env scene (batch vs one-env physics check).

    Segment ``i`` (recorded with ``num_envs = 1``) is rebuilt in env ``i`` of one ``env.reset(seed=[...])``
    and gets that segment's recorded actuation at every substep; env ``i``'s bodies (``env{i}/`` ids, world
    positions minus the env origin) are compared with the segment's ``states.npz``.  Same actuation, so any
    difference comes from the scene holding N envs (PhysX solves them in one scene).  Spare envs start
    ended (``None`` seed) like ``main.py``'s padding.
    """

    from simuguard.adapters.isaac.robodojo import RoboDojoAdapter
    from simuguard.core.controls import ControlLog
    from simuguard.core.snapshot import Snapshot
    from simuguard.core.snapshot_types import ControlRecord
    from simuguard.core.statelog import StateLog

    n_envs = int(env.num_envs)
    segs = [Path(s) for s in args.replay_from]
    metas = [json.loads((seg / "manifest.json").read_text())["metadata"] for seg in segs]
    if all(int(m.get("num_envs", 1)) > 1 for m in metas):
        replay_batch_fresh(args, env, out, segs, metas)
        return
    layouts = []
    for seg, meta in zip(segs, metas):
        if meta.get("task") != args.task or int(meta.get("num_envs", 1)) != 1:
            raise ValueError(f"{seg}: task {meta.get('task')}, num_envs {meta.get('num_envs', 1)} (need {args.task}, 1)")
        layouts.append(int(meta["seed"]))
    seeds: list[int | None] = layouts + [None] * (n_envs - len(layouts))
    record: dict[str, Any] = {"mode": "cross_batch", "num_envs": n_envs, "batch": seeds, "segments": [str(s) for s in segs]}
    try:
        t0 = time.time()
        with use_null_client(env):
            env.env_seeds = list(seeds)
            env.reset(seed=list(seeds))
        record["reset_wall_s"] = time.time() - t0
        record["end_flag_after_reset"] = [bool(env.end_flag[k]) for k in range(n_envs)]
        origins = _to_np(env.env_origins)
        record["env_origins"] = origins.tolist()

        def remap(payload: dict[str, Any], pre: str) -> dict[str, Any]:
            if payload.get("__writes__"):
                raise RuntimeError("segment has between-step writes: cross-batch replay does not remap them")
            return {pre + k: v for k, v in payload.items() if k != "__writes__"}

        plan = {}
        for k, seg in enumerate(segs):
            pre = f"env{k}/"
            adapter = RoboDojoAdapter(env, task_name=args.task, env_idx=k, physics_step_counter=False)
            bodies = adapter.bodies()
            ref = StateLog.load(seg / "states.npz")
            ids = [b for b in ref.body_ids if pre + b in bodies and bodies[pre + b].kind.value != "static"]
            missing = [b for b in ref.body_ids if pre + b not in bodies]
            snap = Snapshot.load(seg / "initial_snapshot.json.gz")
            adapter.apply_control(ControlRecord(snap.substep, snap.control.control_step, remap(snap.control.payload, pre)))
            off = origins[k]
            now = adapter.read_states([pre + b for b in ids])
            init_err = {}
            for b, d in snap.public_state.get("bodies", {}).items():
                if pre + b in now:
                    init_err[b] = float(np.linalg.norm(now[pre + b].position - off - np.asarray(d["pose_xyzw"][:3])))
            controls = ControlLog.load(seg / "controls.npz")
            newest = controls.newest_substep or 0
            plan[k] = {"adapter": adapter, "pre": pre, "off": off, "ids": ids, "missing": missing,
                       "free": [b for b in ids if bodies[pre + b].kind.value in ("dynamic", "kinematic")],
                       "controls": {r.substep: r for r in controls.between(0, newest)}, "last": newest,
                       "arr": ref.array(), "row": {int(s): i for i, s in enumerate(ref.substeps)},
                       "col": {b: ref.body_ids.index(b) for b in ids}, "worst": 0.0, "worst_free": 0.0,
                       "first": None, "curve": [], "final": {},
                       "initial_position_error_m": max(init_err.values(), default=None),
                       "initial_bodies_compared": len(init_err)}
        started = time.time()
        last = max(p["last"] for p in plan.values())
        stepper = plan[0]["adapter"]
        for k_sub in range(1, last + 1):
            for p in plan.values():
                rec = p["controls"].get(k_sub)
                if rec is not None:
                    p["adapter"].apply_control(ControlRecord(rec.substep, rec.control_step, remap(rec.payload, p["pre"])))
            stepper.step_physics()
            for p in plan.values():
                a = p["adapter"]
                if a is not stepper:
                    a._invalidate()
                    a._internal_step += 1
                if k_sub > p["last"] or k_sub not in p["row"]:
                    continue
                now = a.read_states([p["pre"] + b for b in p["ids"]])
                r = p["arr"][p["row"][k_sub]]
                errs = {b: float(np.linalg.norm(now[p["pre"] + b].position - p["off"] - r[p["col"][b], :3])) for b in p["ids"]}
                b_max = max(errs, key=errs.get)
                p["worst"] = max(p["worst"], errs[b_max])
                p["worst_free"] = max([p["worst_free"]] + [errs[b] for b in p["free"]])
                if errs[b_max] > 0.0 and p["first"] is None:
                    p["first"] = {"substep": k_sub, "body": b_max, "error_m": errs[b_max]}
                if k_sub % 100 == 0 or k_sub == p["last"]:
                    p["curve"].append([k_sub, errs[b_max], max((errs[b] for b in p["free"]), default=0.0)])
                if k_sub == p["last"]:
                    p["final"] = {b: errs[b] for b in p["free"]}
        record["wall_s"] = time.time() - started
        record["envs"] = {
            str(k): {"segment": str(segs[k]), "layout": layouts[k], "substeps": p["last"], "bodies_compared": len(p["ids"]),
                     "free_bodies_compared": len(p["free"]), "bodies_missing_in_batch": p["missing"],
                     "initial_position_error_m": p["initial_position_error_m"],
                     "initial_bodies_compared": p["initial_bodies_compared"],
                     "max_position_error_m": p["worst"], "max_free_body_position_error_m": p["worst_free"],
                     "bit_exact": p["worst"] == 0.0, "first_divergence": p["first"],
                     "final_free_body_error_m": p["final"], "curve_every_100": p["curve"]}
            for k, p in plan.items()}
        for p in plan.values():
            p["adapter"].close()
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    finally:
        with (out / "replays.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass
    for k, r in (record.get("envs") or {}).items():
        print(f"[simuguard] cross-batch replay env {k} layout {r['layout']}: bit_exact={r['bit_exact']} "
              f"init err {r['initial_position_error_m']} max err {r['max_position_error_m']:.3e} "
              f"free {r['max_free_body_position_error_m']:.3e} first={r['first_divergence']}", flush=True)
    if record.get("error"):
        print(f"[simuguard] cross-batch replay error: {record['error']}", flush=True)


def replay_batch_fresh(args: argparse.Namespace, env: Any, out: Path, segs: list[Path], metas: list[dict[str, Any]]) -> None:
    """Fresh-process replay of one recorded batch: same ``env.reset(seed=batch)``, every env's actuation.

    The segments must be all monitored envs of one batch (same ``batch`` seeds, recorded with this
    ``num_envs``); each env's bodies are compared with its own ``states.npz`` (:func:`replay_recorded_batch`).
    """

    from simuguard.adapters.isaac.robodojo import RoboDojoAdapter

    n_envs = int(env.num_envs)
    record: dict[str, Any] = {"mode": "batch_fresh_process", "num_envs": n_envs, "segments": [str(s) for s in segs]}
    try:
        batches = {json.dumps(m.get("batch")) for m in metas}
        if len(batches) != 1 or any(int(m["num_envs"]) != n_envs or m.get("task") != args.task for m in metas):
            raise ValueError(f"segments are not one {n_envs}-env {args.task} batch: {sorted(batches)}")
        seeds = list(metas[0]["batch"])
        by_env = {int(m["env_idx"]): seg for m, seg in zip(metas, segs)}
        record["batch"] = seeds
        t0 = time.time()
        with use_null_client(env):
            env.env_seeds = list(seeds)
            env.reset(seed=list(seeds))
        record["reset_wall_s"] = time.time() - t0
        running = [k for k in range(n_envs) if not env.end_flag[k]]
        record["running_after_reset"] = running
        if sorted(by_env) != running:
            raise ValueError(f"segments for envs {sorted(by_env)} but envs {running} are running after reset")
        adapters = {k: RoboDojoAdapter(env, task_name=args.task, env_idx=k) for k in running}
        record["envs"] = {str(k): {"segment": str(by_env[k]), "layout": seeds[k], **r}
                          for k, r in replay_recorded_batch(env, adapters, by_env).items()}
        for adapter in adapters.values():
            adapter.close()
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    finally:
        with (out / "replays.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass
    for k, r in (record.get("envs") or {}).items():
        print(f"[simuguard] fresh-process batch replay env {k} layout {r['layout']}: bit_exact={r['bit_exact']} "
              f"max err {r['max_position_error_m']:.3e} first={r['first_divergence']}", flush=True)
    if record.get("error"):
        print(f"[simuguard] fresh-process batch replay error: {record['error']}", flush=True)


def _to_np(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except BaseException:  # noqa: BLE001 - Kit's shutdown would otherwise turn a crash into exit code 0
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
