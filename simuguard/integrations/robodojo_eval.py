"""Run RoboDojo evaluation episodes under SimuGuard (inside the ``hexa/robodojo`` image).

The environment is RoboDojo's own ``EvalEnv`` built exactly like ``src/eval_client/main.py``
builds it (same configs, ``process_randomization`` / ``process_config``, one env per
process as for every ``eval_batch: false`` policy).  Each layout id is one official
episode: ``env.reset(seed=[id])`` -> ``env.run_eval()`` (the XPolicyLab policy loop,
RoboDojo's reward manager and video writer) -> ``env.close()``.  What differs from
``main.py`` is only the outer loop (explicit layout ids, no resume manifest, no
PhysX-crash re-exec) and the instrumentation:

* before the env exists, :func:`install_contact_reporting` makes RoboDojo create every
  scene object with ``PhysxContactReportAPI`` (contact reporting only, no dynamics change);
* after each reset a :class:`RoboDojoAdapter` + :class:`SubstepMonitor` attach to the
  ``SimulationContext.step`` of the fresh scene; they finalize after ``run_eval``;
* ``--replay`` rebuilds the same layout (close + reset, no policy) and replays the
  recorded per-substep actuation from the episode start, reporting the position error of
  every logged body against the recorded trajectory; ``--public-restore`` additionally
  restores a mid-episode public snapshot in place and replays from there.

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
    ap.add_argument("--layouts", type=int, nargs="+", default=[], help="RoboDojo layout ids (eval seeds)")
    ap.add_argument("--replay-from", nargs="+", default=[],
                    help="replay recorded segment dirs (of --task) in this fresh process instead of running episodes")
    ap.add_argument("--policy", default="scripted", help="'scripted' or an XPolicyLab policy name (Xiaomi_Robotics_1, Pi_05)")
    ap.add_argument("--policy-url", default="", help="ws://host:port of the policy server")
    ap.add_argument("--additional-info", default="simuguard")
    ap.add_argument("--env-cfg", default="arx_x5")
    ap.add_argument("--eval-seed", type=int, default=0, help="RoboDojo eval seed (layout set)")
    ap.add_argument("--out", required=True)
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
    if not args.layouts and not args.replay_from:
        ap.error("give --layouts or --replay-from")
    return args


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
    """RoboDojo EvalEnv for one task, assembled like ``src/eval_client/main.py`` (num_envs = 1)."""

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
    eval_cfg["num_envs"] = 1
    eval_cfg["device_id"] = 0
    eval_cfg["eval_batch"] = False
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
    OmegaConf.update(env_cfg, "sim.scene.num_envs", 1, force_add=True)
    OmegaConf.update(env_cfg, "eval_cfg.num_envs", 1, force_add=True)
    env_cfg = process_randomization(env_cfg)
    env_cfg, eval_num = process_config(env_cfg, task_name=args.task)
    eval_cfg["eval_num"] = eval_num
    OmegaConf.update(env_cfg, "camera.default_frequency", eval_cfg["observation"].get("collect_freq", 0), force_add=True)
    env_cfg.sim.seed = [0]
    if args.policy == "scripted":
        eval_env_mod.WsModelClient = NullModelClient  # no server: the scripted probe drives the env
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
) -> dict[str, Any]:
    """Replay ``controls.npz`` on the (already rebuilt) scene; compare with ``states.npz``.

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
    for record in controls.between(0, last):
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
    depen = None
    if args.intervention.startswith("depen_"):
        depen = float(args.intervention.split("_", 1)[1])
    elif args.intervention != "none":
        raise ValueError(f"unknown intervention {args.intervention}")
    patched = []
    if args.contact_report == "on" or depen is not None:
        patched = install_contact_reporting(contact_report=args.contact_report == "on", max_depenetration_velocity=depen)
    if args.no_cameras:
        install_no_cameras()
    env, eval_num = build_env(args, app)
    det_cfg = detector_config(args.detector_config)
    episodes_log = out / "episodes.jsonl"
    run_info = {
        "task": args.task, "policy": args.policy, "layouts": args.layouts, "monitor": args.monitor,
        "contact_report": args.contact_report, "contact_patch": patched, "replay": args.replay,
        "no_cameras": bool(args.no_cameras), "intervention": args.intervention,
        "simuguard": __version__, "eval_num_official": eval_num, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "robodojo_run_id": os.environ.get("ROBODOJO_RUN_ID"),
    }
    (out / "run.json").write_text(json.dumps(run_info, indent=2))
    if args.replay_from:
        replay_only(args, env, out)
        app.close()
        return 0
    for n, layout in enumerate(args.layouts):
        record: dict[str, Any] = {"task": args.task, "layout": layout, "policy": args.policy, "monitor": args.monitor}
        seg = out / "segments" / f"layout{layout:03d}"
        record["segment"] = str(seg)
        monitor = adapter = None
        try:
            t0 = time.time()
            env.env_seeds = [layout]
            env.reset(seed=[layout])
            record["reset_wall_s"] = time.time() - t0
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
            details = env.eval_result.get("details", {})
            last = details[max(details)] if details else {}
            record["success"] = bool(last.get("success", False))
            record["score"] = last.get("score")
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
        except UnStableError as exc:
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
            with episodes_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass
        print(f"[simuguard] {args.task} layout {layout}: " + json.dumps(
            {k: record.get(k) for k in ("success", "confirmed", "flags", "substeps", "episode_wall_s", "error")}, default=str),
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
            record["replay"] = replay_recorded(env, adapter, seg, detectors=detectors)
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
