#!/usr/bin/env python3
"""Pi0.5 (RLinf-Pi05-LIBERO-130-fullshot-SFT) on LIBERO under SimuGuard monitoring.

Episode protocol = RLinf's LIBERO evaluation, the one this checkpoint was trained and scored with:
a fresh ``OffScreenRenderEnv`` per episode (256x256 agentview + wrist cameras), ``env.seed(seed)``,
reset, ``set_init_state(init_states[init])``; 15 steps of zero motion with the gripper open (LIBERO
places objects a few cm above their supports; they fall and settle); then the policy: both images
rotated 180 degrees, state = eef position + axis-angle (robosuite ``quat2axisangle``) + the two
gripper joint positions, the task's language; 5 actions per query, all executed; stop at the first
success or after ``--max-steps`` policy steps (default 600, LIBERO's own evaluation limit).

The policy runs in a separate server (``pi05_server.sh``: RPent's ``pi05_vla_server.py``, JSON over
HTTP); this process only simulates, so it runs in the plain LIBERO venv.  SimuGuard attaches right
after ``set_init_state``: the settle steps are monitored too and tagged (``settle_substeps`` in the
metadata).  Every episode writes a SimuGuard segment and one line of ``episodes.jsonl``; with
``--video`` also an mp4 (agentview | wrist, upright, 20 fps).

Usage::

    source scripts/libero/env.h800-2.sh official
    $PY scripts/libero/pi05_rollout.py --suite libero_spatial --task 0 --episodes 0-4 \
        --out $RUNS/pi05_v1 --endpoint http://127.0.0.1:58261 --video
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SETTLE_STEPS = 15
SETTLE_ACTION = [0.0] * 6 + [-1.0]
POLICY = "Pi0.5 (RLinf-Pi05-LIBERO-130-fullshot-SFT, 5 actions/query)"


# ----------------------------------------------------------------------------- policy client
class _Encoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": base64.b64encode(np.ascontiguousarray(obj).tobytes()).decode("ascii"),
                    "dtype": str(obj.dtype), "shape": list(obj.shape)}
        if isinstance(obj, np.generic):
            return {"__npscalar__": obj.item(), "dtype": str(obj.dtype)}
        return super().default(obj)


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            return np.frombuffer(base64.b64decode(obj["__ndarray__"]), dtype=obj["dtype"]).reshape(obj["shape"]).copy()
        if "__npscalar__" in obj:
            return np.dtype(obj["dtype"]).type(obj["__npscalar__"])
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


class Pi05Client:
    """Client of RPent's Pi0.5 servers (``rpent.utils.rpc.http_rpc`` wire format), stdlib only.

    The policy is stateless per query and the simulation waits for it, so a query that times out
    is simply sent again, to the next server in the list.
    """

    def __init__(self, urls: list[str], timeout_s: float = 180.0, attempts: int = 4) -> None:
        self.urls = [u.rstrip("/") + "/call" for u in urls]
        self.timeout_s, self.attempts = timeout_s, attempts
        self.primary = 0
        self.retries = 0
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # localhost: no proxy

    @property
    def url(self) -> str:
        return self.urls[self.primary]

    def call(self, method: str, *args: Any) -> Any:
        body = json.dumps({"method": method, "args": list(args), "kwargs": {}, "session_id": None},
                          cls=_Encoder).encode("utf-8")
        error: Exception | None = None
        for attempt in range(self.attempts):
            url = self.urls[(self.primary + attempt) % len(self.urls)]
            request = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
            try:
                with self.opener.open(request, timeout=self.timeout_s) as response:
                    reply = _decode(json.loads(response.read()))
            except OSError as exc:  # timeouts, refused connections, URLError
                error = exc
                self.retries += 1
                continue
            if not reply.get("ok"):
                raise RuntimeError(f"{method}: {reply.get('error')}")
            return reply.get("result")
        raise error if error is not None else RuntimeError("no policy endpoint")

    def predict(self, obs: dict) -> np.ndarray:
        return np.asarray(self.call("vla.predict", obs, None), dtype=np.float32)[0]  # (5, 7)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """robosuite's quat2axisangle (xyzw), as RLinf builds the Pi0.5 state (no hemisphere fix)."""
    quat = np.array(quat, dtype=np.float64)
    quat[3] = min(1.0, max(-1.0, quat[3]))
    den = math.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(quat[3]) / den


def policy_obs(obs: dict, language: str) -> dict:
    main = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])  # RLinf get_libero_image
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    state = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]])
    return {"main_images": main[None].astype(np.uint8), "wrist_images": wrist[None].astype(np.uint8),
            "extra_view_images": None, "states": state[None].astype(np.float32), "task_descriptions": [language]}


# ----------------------------------------------------------------------------- video
class VideoWriter:
    """Raw RGB frames piped to ffmpeg (H.264); falls back to OpenCV mp4v."""

    def __init__(self, path: Path, fps: int = 20) -> None:
        self.path, self.fps, self.proc, self.cv = path, fps, None, None

    def add(self, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if self.proc is None and self.cv is None:
            h, w = frame.shape[:2]
            ffmpeg = os.environ.get("FFMPEG", "ffmpeg")
            try:
                self.proc = subprocess.Popen(
                    [ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                     "-r", str(self.fps), "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23",
                     str(self.path)], stdin=subprocess.PIPE)
            except OSError:
                import cv2

                self.cv = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        if self.proc is not None:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.cv.write(frame[..., ::-1])

    def close(self) -> None:
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait(timeout=120)
        if self.cv is not None:
            self.cv.release()


def view(obs: dict) -> np.ndarray:
    return np.concatenate([obs["agentview_image"][::-1], obs["robot0_eye_in_hand_image"][::-1]], axis=1)


# ----------------------------------------------------------------------------- one episode
def run_episode(client: Pi05Client, suite: str, task_id: int, init: int, seed: int, out_dir: Path,
                args: argparse.Namespace) -> dict:
    from simuguard.adapters.mujoco.libero import attach_libero, make_env
    from simuguard.core.monitor import MonitorConfig, SubstepMonitor
    from simuguard.core.recorder import EpisodeRecorder
    from simuguard.presets import default_detectors

    t0 = time.time()
    client.primary = (init + 3 * task_id + 7 * len(suite)) % len(client.urls)  # spread episodes over servers
    retries_before = client.retries
    env, obs, meta = make_env(suite, task_id, init, seed, camera_size=256)
    reset_s = time.time() - t0
    adapter, info = attach_libero(env, task_name=meta["task"])
    name = f"ep{init:02d}_seed{seed}"
    seg = out_dir / "segments" / name
    settle_substeps = SETTLE_STEPS * info["substeps_per_step"]
    monitor = SubstepMonitor(
        adapter,
        default_detectors({"ejection": {"delta_v_reference_timestep_s": 0.004}}),
        episode_id=f"{suite}:{meta['task']}:{name}",
        config=MonitorConfig.from_dict({
            "snapshot_interval_substeps": 250, "snapshot_capacity": 2000, "frame_buffer_substeps": 2000,
            "control_log_maxlen": None, "bundle_mode": "snapshot", "bundle_post_substeps": 500,
            "save_snapshots": True,
        }),
        recorder=EpisodeRecorder(seg),
        metadata={**meta, "policy": POLICY, "endpoint": client.url, "phase": "policy", "settle_steps": SETTLE_STEPS,
                  "settle_substeps": settle_substeps, "max_steps": args.max_steps,
                  "roles": {k: info[k] for k in ("target_objects", "container_objects")}},
    )
    monitor.attach()
    video = VideoWriter(seg / "video.mp4") if args.video else None
    success, steps, queries, error, latency = False, 0, 0, None, []
    try:
        for _ in range(SETTLE_STEPS):
            obs, _, _, _ = env.step(SETTLE_ACTION)
            if video:
                video.add(view(obs))
        plan: list[np.ndarray] = []
        while steps < args.max_steps:
            if not plan:
                q0 = time.time()
                chunk = client.predict(policy_obs(obs, meta["language"]))
                latency.append(time.time() - q0)
                if chunk.ndim != 2 or chunk.shape[1] != 7 or not np.isfinite(chunk).all():
                    raise ValueError(f"policy returned {chunk.shape}")
                plan = list(chunk)
                queries += 1
            obs, _, done, _ = env.step(plan.pop(0).tolist())
            steps += 1
            if video:
                video.add(view(obs))
            if done:
                success = True
                break
    except Exception:  # noqa: BLE001
        error = traceback.format_exc(limit=6)
    if video:
        try:
            video.close()
        except Exception:  # noqa: BLE001
            pass
    try:
        success = success or bool(env.check_success())
    except Exception:  # noqa: BLE001
        pass
    monitor.metadata["outcome"] = {"success": success, "eval_success": success, "steps": steps, "queries": queries,
                                   "max_steps": args.max_steps, "error": error}
    summary = monitor.finalize()
    env.close()
    confirmed = [e for e in summary["events"] if e.get("status") == "confirmed"]
    return {
        "suite": suite, "task_id": task_id, "task": meta["task"], "episode": init, "init_index": init, "seed": seed,
        "success": success, "steps": steps, "max_steps": args.max_steps, "queries": queries, "error": error,
        "segment": str(seg), "confirmed": len(confirmed), "flags": summary["flag_count"],
        "monitor_errors": summary["error_count"], "settle_substeps": settle_substeps,
        "events": [{"bodies": e.get("bodies"), "onset": e.get("onset_substep"), "reasons": e.get("reasons"),
                    "max_speed": (e.get("metrics") or {}).get("max_speed_mps"),
                    "in_settle": int(e.get("onset_substep", 0)) <= settle_substeps} for e in confirmed],
        "reset_s": round(reset_s, 2), "wall_s": round(time.time() - t0, 2), "substeps": summary["substeps"],
        "policy_latency_s": round(float(np.mean(latency)), 4) if latency else None,
        "policy_endpoint": client.url, "policy_retries": client.retries - retries_before,
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
    ap.add_argument("--suite", required=True)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--episodes", default="0-4", help="init-state indices; episode k uses init state k and seed base+k")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--out", required=True, help="campaign root; writes <out>/<suite>/t<task>/")
    ap.add_argument("--endpoint", default="http://127.0.0.1:58261",
                    help="Pi0.5 server URL, or a comma list: episodes are spread over them, failed queries move on")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out) / args.suite / f"t{args.task:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "episodes.jsonl"
    done = set()
    if log.exists():
        done = {json.loads(line)["episode"] for line in log.read_text().splitlines() if line.strip()}
    client = Pi05Client([u for u in args.endpoint.split(",") if u])
    for init in parse_range(args.episodes):
        if init in done:
            continue
        record = run_episode(client, args.suite, args.task, init, args.seed_base + init, out_dir, args)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        print(f"[{time.strftime('%H:%M:%S')}] {args.suite} t{args.task} ep{init}: success={record['success']} "
              f"steps={record['steps']} confirmed={record['confirmed']} flags={record['flags']} wall={record['wall_s']:.0f}s"
              + (f" ERROR {record['error'].strip().splitlines()[-1]}" if record["error"] else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
