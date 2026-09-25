#!/usr/bin/env python3
"""XR-1 (Xiaomi-Robotics-1) on RoboCasa365 under SimuGuard monitoring.

Follows the XR-1 reference evaluation (``eval_robocasa365/entry.py``): split ``pretrain``,
episode seed ``7 + task_index * 50 + episode`` with the task index in ``target50``, the
official task horizon, 4 observations spaced 2 steps apart, 16 actions per query, stop at
the first success.  The policy is a running XR-1 replica pool (crop and state conversion
happen server-side); this process only simulates.

``--contact`` changes the contact solver for the scored rollout, applied after every reset
(a hard reset rebuilds the model) and before the first action:

  default    RoboCasa as shipped (most geoms: time constant 20 ms = 10 physics steps)
  stiff2dt   every geom time constant capped at 2 physics steps (4 ms), the shortest MuJoCo
             accepts without disabling its stability guard; damping ratios unchanged

Every episode writes the usual SimuGuard segment (manifest, summary, events, states, controls,
trace) and one line in ``episodes.jsonl``.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

VIDEO_KEYS = ("video.robot0_agentview_left", "video.robot0_agentview_right", "video.robot0_eye_in_hand")
STATE_KEYS = ("state.end_effector_position_relative", "state.end_effector_rotation_relative",
              "state.gripper_qpos", "state.base_position", "state.base_rotation")
LANGUAGE_KEY = "annotation.human.task_description"
ACTION_FIELDS = (("end_effector_position", 3), ("end_effector_rotation", 3), ("gripper_close", 1),
                 ("base_motion", 4), ("control_mode", 1))
BASE_SEED, TRIALS_PER_TASK = 7, 50
CONTACTS = ("default", "stiff2dt")


# ----------------------------------------------------------------------------- XR-1 client
class Xr1Client:
    """REQ/REP msgpack client of the XR-1 pool (the wire format of its serving protocol)."""

    MARK = "__capx_ndarray__"

    def __init__(self, endpoint: str, timeout_s: float = 300.0) -> None:
        import zmq

        self.zmq, self.endpoint, self.timeout_s = zmq, f"tcp://{endpoint}", timeout_s
        self.ctx = zmq.Context.instance()
        self.sock = None

    def _encode(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, value, allow_pickle=False)
            return {self.MARK: True, "npy": buf.getvalue()}
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)

    def _decode(self, value: dict) -> Any:
        if value.get("__hexaanything_ndarray_v2__"):
            return np.frombuffer(value["data"], dtype=np.dtype(value["dtype"])).reshape(tuple(value["shape"]))
        if value.get(self.MARK) or value.get("__hexa_sdk_ndarray__"):
            return np.load(io.BytesIO(value["npy"]), allow_pickle=False)
        return value

    def call(self, operation: str, payload: dict) -> Any:
        import msgpack

        if self.sock is None:
            self.sock = self.ctx.socket(self.zmq.REQ)
            self.sock.setsockopt(self.zmq.LINGER, 0)
            self.sock.connect(self.endpoint)
        self.sock.send(msgpack.Packer(default=self._encode, use_bin_type=True).pack({"operation": operation, "payload": payload}))
        if not self.sock.poll(int(self.timeout_s * 1000), self.zmq.POLLIN):
            self.sock.close(0)
            self.sock = None
            raise TimeoutError(f"XR-1 {operation} timed out at {self.endpoint}")
        reply = msgpack.unpackb(self.sock.recv(), object_hook=self._decode, raw=False)
        if not isinstance(reply, dict) or reply.get("ok") is not True:
            raise RuntimeError(f"XR-1 {operation} failed: {reply.get('error') if isinstance(reply, dict) else reply}")
        return reply["result"]

    def infer(self, observation: dict, session_id: str, reset_memory: bool) -> dict:
        result = self.call("infer", {"observation": observation, "session_id": session_id, "reset_memory": reset_memory})
        return result["action"]

    def close_session(self, session_id: str) -> None:
        try:
            self.call("close_session", {"session_id": session_id})
        except Exception:  # noqa: BLE001
            pass


def action_block(actions: dict) -> np.ndarray:
    """Named XR-1 chunk -> (horizon, 12); binary channels cut at 0.5 as RoboCasa's converter does."""
    parts = []
    for name, size in ACTION_FIELDS:
        a = np.asarray(actions.get(f"action.{name}", actions.get(name)), dtype=np.float32)
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if a.ndim == 1:
            a = a[None, :]
        if a.shape[1] != size or not np.isfinite(a).all():
            raise ValueError(f"XR-1 action {name} has shape {a.shape}")
        if name in ("gripper_close", "control_mode"):
            a = (a >= 0.5).astype(np.float32)
        parts.append(a)
    return np.clip(np.concatenate(parts, axis=1), -1.0, 1.0)


# ----------------------------------------------------------------------------- env helpers
def install_persistent_renderer() -> None:
    """One ``mujoco.Renderer`` per MjSim for RGB observations.

    The robosuite build in this venv creates and destroys a renderer for every camera image;
    on this host's EGL that returns pixel noise.  Same fix as the team's RoboCasa runtime
    (robots/robocasa/stable_renderer.py): keep one renderer per sim and copy its pixels.
    """
    import mujoco
    from robosuite.utils import binding_utils

    cls = binding_utils.MjSim
    if getattr(cls, "_simuguard_rgb_patch", False):
        return
    original, lock = cls.render, binding_utils._MjSim_render_lock

    def render(self, width=None, height=None, *, camera_name=None, depth=False, mode="offscreen",
               device_id=-1, segmentation=False):
        if depth or segmentation or mode != "offscreen" or width is None or height is None:
            return original(self, width, height, camera_name=camera_name, depth=depth, mode=mode,
                            device_id=device_id, segmentation=segmentation)
        with lock:
            vis = self.model.vis.global_
            vis.offwidth, vis.offheight = max(width, vis.offwidth), max(height, vis.offheight)
            renderer = getattr(self, "_simuguard_renderer", None)
            if renderer is None or getattr(self, "_simuguard_renderer_size", None) != (height, width):
                if renderer is not None:
                    renderer.close()
                renderer = mujoco.Renderer(self.model._model, height=height, width=width, max_geom=10000)
                self._simuguard_renderer, self._simuguard_renderer_size = renderer, (height, width)
            renderer.update_scene(self.data._data, camera=camera_name if camera_name is not None else -1,
                                  scene_option=self._render_context_offscreen.vopt)
            return np.array(renderer.render()[::-1], dtype=np.uint8, order="C", copy=True)

    cls.render = render
    cls._simuguard_rgb_patch = True


def unwrap(env: Any) -> Any:
    cur = env
    for _ in range(10):
        if hasattr(cur, "sim") and hasattr(cur, "_check_success"):
            return cur
        nxt = getattr(cur, "env", None) or getattr(cur, "unwrapped", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    raise RuntimeError("no robosuite env inside the gym wrapper")


def apply_contact(rs: Any, name: str) -> dict:
    m = rs.sim.model._model
    info: dict = {"name": name, "timestep": float(m.opt.timestep)}
    if name == "default":
        return info
    if name != "stiff2dt":
        raise ValueError(f"unknown contact setting {name}")
    tau = 2.0 * float(m.opt.timestep)
    standard = m.geom_solref[:, 0] > 0  # (time constant, damping ratio) form; negative = direct stiffness
    changed = standard & (m.geom_solref[:, 0] > tau)
    info.update({"time_constant_s": tau, "geoms_changed": int(changed.sum()), "geoms": int(m.ngeom)})
    m.geom_solref[changed, 0] = tau
    if m.npair:
        pc = (m.pair_solref[:, 0] > tau)
        m.pair_solref[pc, 0] = tau
        info["pairs_changed"] = int(pc.sum())
    return info


def roles_for(rs: Any):
    from simuguard.adapters.mujoco import TaskRoles

    objects = {name: obj.root_body for name, obj in getattr(rs, "objects", {}).items()}
    containers = {body for name, body in objects.items() if "container" in name and name != "obj"}
    targets = set(objects.values()) - containers
    return TaskRoles(targets=targets, containers=containers), objects


def history_frame(obs: dict) -> dict:
    return {k: np.array(obs[k], copy=True) for k in VIDEO_KEYS + STATE_KEYS}


def sample_history(history: collections.deque, frames: int = 4, interval: int = 2) -> list:
    items = list(history)
    last = len(items) - 1
    return [items[max(0, last - interval * (frames - 1 - i))] for i in range(frames)]


# ----------------------------------------------------------------------------- one episode
def run_episode(env: Any, client: Xr1Client, task: str, episode: int, seed: int, horizon: int,
                contact: str, out_dir: Path, args: argparse.Namespace) -> dict:
    from robocasa.utils.env_utils import convert_action

    from simuguard.adapters.mujoco import attach_robosuite
    from simuguard.core.monitor import MonitorConfig, SubstepMonitor
    from simuguard.core.recorder import EpisodeRecorder
    from simuguard.presets import default_detectors

    t0 = time.time()
    try:  # a hard reset builds a new sim; release the old sim's renderer first
        old_sim = unwrap(env).sim
        if getattr(old_sim, "_simuguard_renderer", None) is not None:
            old_sim._simuguard_renderer.close()
            old_sim._simuguard_renderer = None
    except Exception:  # noqa: BLE001
        pass
    obs, _ = env.reset(seed=seed)
    reset_s = time.time() - t0
    instruction = str(obs[LANGUAGE_KEY])
    rs = unwrap(env)
    contact_info = apply_contact(rs, contact)
    roles, objects = roles_for(rs)
    adapter = attach_robosuite(rs, roles=roles, task_name=task)
    name = f"ep{episode:02d}_seed{seed}"
    seg = out_dir / "segments" / name
    monitor = SubstepMonitor(
        adapter,
        default_detectors({"ejection": {"delta_v_reference_timestep_s": 0.004}}),
        episode_id=f"{task}:{contact}:{name}",
        config=MonitorConfig.from_dict({
            "snapshot_interval_substeps": 250, "snapshot_capacity": 2000, "frame_buffer_substeps": 2000,
            "control_log_maxlen": None, "bundle_mode": "snapshot", "bundle_post_substeps": 500,
            "save_snapshots": True,
        }),
        recorder=EpisodeRecorder(seg),
        metadata={"task": task, "episode": episode, "seed": seed, "contact": contact_info, "objects": objects,
                  "instruction": instruction, "policy": "XR-1", "phase": "policy"},
    )
    monitor.attach()
    session = f"simuguard-{task}-{contact}-{seed}-{os.getpid()}"
    history: collections.deque = collections.deque(maxlen=7)
    for _ in range(7):
        history.append(history_frame(obs))
    plan: collections.deque = collections.deque()
    frames = [np.concatenate([obs[k] for k in VIDEO_KEYS], axis=1)] if args.video else []
    success, steps, queries, error = False, 0, 0, None
    try:
        while steps < horizon:
            if not plan:
                hist = sample_history(history)
                observation = {k: np.stack([h[k] for h in hist]).astype(np.uint8) for k in VIDEO_KEYS}
                observation.update({k: np.stack([np.asarray(h[k], dtype=np.float32).reshape(-1) for h in hist]) for k in STATE_KEYS})
                observation[LANGUAGE_KEY] = [instruction]
                block = action_block(client.infer(observation, session, reset_memory=(queries == 0)))
                queries += 1
                plan.extend(block[:16])
            obs, _, done, truncated, info = env.step(convert_action(plan.popleft()))
            steps += 1
            history.append(history_frame(obs))
            if args.video and steps % 2 == 0:
                frames.append(np.concatenate([obs[k] for k in VIDEO_KEYS], axis=1))
            success = bool(info.get("success", False))
            if success or done or truncated:
                break
    except Exception:  # noqa: BLE001
        error = traceback.format_exc(limit=5)
    client.close_session(session)
    monitor.metadata["outcome"] = {"success": success, "steps": steps, "horizon": horizon, "error": error}
    summary = monitor.finalize()
    if frames:
        import imageio.v2 as imageio

        imageio.mimsave(seg / f"video_{'success' if success else 'failure'}.mp4", frames, fps=10)
    confirmed = [e for e in summary["events"] if e.get("status") == "confirmed"]
    return {
        "task": task, "episode": episode, "seed": seed, "contact": contact, "success": success, "steps": steps,
        "horizon": horizon, "queries": queries, "error": error, "segment": str(seg),
        "confirmed": len(confirmed), "flags": summary["flag_count"], "monitor_errors": summary["error_count"],
        "events": [{"bodies": e.get("bodies"), "onset": e.get("onset_substep"), "reasons": e.get("reasons"),
                    "max_speed": (e.get("metrics") or {}).get("max_speed_mps")} for e in confirmed],
        "reset_s": reset_s, "wall_s": time.time() - t0, "substeps": summary["substeps"],
    }


def parse_range(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--episodes", default="0-9")
    ap.add_argument("--contact", choices=CONTACTS, default="default")
    ap.add_argument("--out", required=True, help="campaign root; writes <out>/<contact>/<task>/")
    ap.add_argument("--endpoint", default="127.0.0.1:15555")
    ap.add_argument("--horizon", type=int, default=None, help="default: the official task horizon")
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()

    import gymnasium as gym
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

    install_persistent_renderer()
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    tasks = list(TASK_SET_REGISTRY["target50"])
    task_index = tasks.index(args.task)
    horizon = args.horizon or int(get_task_horizon(args.task))
    out_dir = Path(args.out) / args.contact / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    log = out_dir / "episodes.jsonl"
    if log.exists():
        done = {json.loads(line)["episode"] for line in log.read_text().splitlines() if line.strip()}
    env = gym.make(f"robocasa/{args.task}", split="pretrain", seed=BASE_SEED)
    client = Xr1Client(args.endpoint)
    try:
        for episode in parse_range(args.episodes):
            if episode in done:
                continue
            seed = BASE_SEED + task_index * TRIALS_PER_TASK + episode
            record = run_episode(env, client, args.task, episode, seed, horizon, args.contact, out_dir, args)
            with log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            print(f"[{time.strftime('%H:%M:%S')}] {args.task} {args.contact} ep{episode} seed{seed}: "
                  f"success={record['success']} steps={record['steps']} confirmed={record['confirmed']} "
                  f"wall={record['wall_s']:.0f}s" + (f" ERROR {record['error'].splitlines()[-1]}" if record["error"] else ""),
                  flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
