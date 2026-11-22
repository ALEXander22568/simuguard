#!/usr/bin/env python3
"""Job queue for the SimuGuard x RoboDojo runs (native install, stdlib only).

usage:
  python3 rd_queue.py --run-dir RUN --gpus 3 4 5 6 --policies GR00T_N17 Pi_05 [--tasks ...]
                      [--max-counted 3] [--env-manifest FILE]

One job = one (policy, task).  A lane = one GPU.  For each job the lane starts the policy's
official ``setup_eval_policy_server.sh`` (same arguments as the policy's ``eval.sh``), waits
with the official ``wait_for_policy_server.sh``, then runs
``simuguard.integrations.robodojo_eval`` in the RoboDojo conda env in place of
``src/eval_client/main.py`` (``--layouts all``, ``--resume``).  Server and sim share the
lane's GPU unless ``--server-gpu`` maps a lane to another card.

Restart-safe: job progress lives in ``RUN/<policy>/<task>/episodes.jsonl`` (written by the
runner); rerunning this script with the same arguments continues where it stopped.  If the
sim process dies without writing a record, the queue appends an ``error`` record for the
layout that was running (the first layout, in file order, without a final record), so a
layout that keeps crashing the process is given up after ``--max-attempts`` tries.

Control (re-read every loop) from ``RUN/CONTROL.json``::

  {"disabled_gpus": {"3": "drain"},      # drain: finish the lane's job, then stop the lane
                                          # on (or key removed): a CONTROL-stopped lane runs again
                                          # now: stop it at once (the episode in flight is
                                          # dropped without an error record and rerun later)
   "hold_policies": ["OpenWAM"]}          # do not start new jobs of these policies

A lane also stops at once ("now") when a compute process that is not ours shows up on its
GPU (e.g. another project taking the card back).

Disk watch (``--min-free-gb N``): checked once a minute on the run dir's filesystem.  Below N GB
free, every running launch is stopped at once (the episodes in flight are dropped without an error
record and rerun later) and no new launch starts; the queue stays up and carries on by itself once
``--resume-free-gb`` (default N + 10) GB are free again.

Writes ``RUN/STATUS.txt`` / ``RUN/STATUS.json`` every ``--status-every`` seconds and logs
to ``RUN/queue.log``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("SGRD_ROOT", "/path/to/simuguard-robodojo"))
RD = ROOT / "RoboDojo"
XPL = RD / "XPolicyLab"
SG_SRC = Path(os.environ.get("SG_SRC", ROOT / "simuguard-src"))  # synced SimuGuard tree (override: tests)
PY_SIM = ROOT / "miniconda3/envs/RoboDojo/bin/python"
# coordinator: RoboDojo uses GPUs 3-6 only on the 24 GB box; another box sets SGRD_ALLOWED_GPUS (e.g. "0" on the 48 GB box)
ALLOWED_GPUS = {int(g) for g in os.environ.get("SGRD_ALLOWED_GPUS", "3,4,5,6").split(",")}
KST = dt.timezone(dt.timedelta(hours=9))
DONE_STATUSES = ("counted", "unstable_layout", "unstable_episode")

# policy -> (ckpt_name, action_type, env arg of setup_eval_policy_server.sh)
POLICIES = {
    "GR00T_N17": ("RoboDojo-sim-arx_x5-joint-0", "joint", "uv"),
    "Pi_05": ("RoboDojo-sim-arx_x5-joint-0", "joint", "uv"),
    "Pi_0": ("RoboDojo-sim-arx_x5-joint-0", "joint", "uv"),
    "InternW0_delta": ("RoboDojo-sim-arx_x5-joint-0", "joint", "internw0-delta"),
    "InternVLA_A1": ("RoboDojo-sim-arx_x5-joint-0", "joint", "internvla_a1"),
    "Xiaomi_Robotics_0": ("RoboDojo-sim-arx_x5-ee-0", "ee", "mibot"),
    "X_VLA": ("RoboDojo-sim-arx_x5-ee-0", "ee", "XVLA"),
    "OpenWAM": ("OpenWAM-Alpha-Sim-RoboDojo", "ee", "openwam"),
}
# extra env for a policy's official server script. Xiaomi_Robotics_0: its setup_eval_policy_server.sh exports
# a fixed LAN proxy (<lan-proxy>) and documents HF_HUB_OFFLINE=1 for offline deploy; every file it
# loads is local (processor under xr0/, Qwen/Qwen3-VL-4B-Instruct in the HF hub cache, see 00b_base_models.sh)
SERVER_ENV = {"Xiaomi_Robotics_0": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}}
TASKS = ("hang_mugs", "sweep_blocks", "classify_objects", "fill_pen_holder", "stack_bowls", "insert_tubes",
         "cover_blocks")


def eval_batch(policy: str) -> bool:
    """``eval_batch`` of the policy's deploy.yml (a top-level ``eval_batch: true|false`` line)."""
    path = XPL / "policy" / policy / "deploy.yml"
    for line in path.read_text().splitlines() if path.exists() else []:
        key, _, value = line.partition(":")
        if key.strip() == "eval_batch" and not line.startswith((" ", "\t")):
            return value.split("#")[0].strip().lower() == "true"
    return False


def official_num_envs() -> int:
    """``scene.num_envs`` of RoboDojo's sim_config.yml: the num_envs ``scripts/eval_policy.sh`` passes to main.py."""
    import yaml

    cfg = yaml.safe_load((RD / "env_cfg" / "sim" / "sim_config.yml").read_text())
    return int((cfg.get("scene") or {}).get("num_envs", 1))


def run_modes(policies: list[str], batch_envs: int) -> dict[str, dict]:
    """Per policy: one env or batched, and why (main.py runs eval_batch=false policies with one env)."""
    out = {}
    for p in policies:
        eb = eval_batch(p)
        n = batch_envs if eb and batch_envs > 1 else 1
        out[p] = {"deploy_yml_eval_batch": eb, "num_envs": n, "mode": "batched" if n > 1 else "single_env",
                  "server_env": SERVER_ENV.get(p, {}),
                  "reason": (f"deploy.yml eval_batch: true -> num_envs {n} (sim_config.yml scene.num_envs, as eval_policy.sh)"
                             if n > 1 else "deploy.yml eval_batch: false -> main.py forces num_envs 1" if not eb
                             else f"deploy.yml eval_batch: true but this run uses --batch-envs {batch_envs}")}
    return out


def now_kst() -> str:
    return dt.datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")


def layout_ids(task: str, seed: int = 0) -> list[int]:
    folder = RD / "Assets/Eval_Layout/RoboDojo/arx_x5" / str(seed)
    ids = []
    for p in folder.glob(f"{task}_*.json"):
        tail = p.stem[len(task) + 1:]
        if tail.isdigit():
            ids.append(int(tail))
    return sorted(ids)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class Job:
    def __init__(self, run_dir: Path, policy: str, task: str, max_counted: int, max_attempts: int):
        self.policy, self.task = policy, task
        self.dir = run_dir / policy / task
        self.dir.mkdir(parents=True, exist_ok=True)
        self.layouts = layout_ids(task)
        self.max_counted = max_counted
        self.max_attempts = max_attempts
        self.launches = 0
        self.state = "pending"
        self.lane: int | None = None
        self.mem_mib = {"server": 0, "sim": 0}

    @property
    def name(self) -> str:
        return f"{self.policy}/{self.task}"

    def records(self) -> list[dict]:
        log = self.dir / "episodes.jsonl"
        out = []
        if log.exists():
            for line in log.read_text().splitlines():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out

    def progress(self) -> dict:
        """Same bookkeeping as robodojo_eval --resume."""
        done: dict[int, dict] = {}
        attempts: dict[int, int] = {}
        for rec in self.records():
            lid = int(rec["layout"])
            if rec.get("status") in DONE_STATUSES:
                done[lid] = rec
            else:
                attempts[lid] = attempts.get(lid, 0) + 1
                if attempts[lid] >= self.max_attempts:
                    done[lid] = rec
        counted = sum(1 for r in done.values() if r.get("status") == "counted")
        statuses: dict[str, int] = {}
        for r in done.values():
            statuses[r.get("status", "?")] = statuses.get(r.get("status", "?"), 0) + 1
        next_layout = next((x for x in self.layouts if x not in done), None)
        finished = next_layout is None or bool(self.max_counted and counted >= self.max_counted)
        if self.max_counted:
            remaining = 0 if finished else self.max_counted - counted
        else:
            remaining = sum(1 for x in self.layouts if x not in done)
        succ = [r for r in done.values() if r.get("status") == "counted"]
        return {"done": len(done), "counted": counted, "statuses": statuses, "finished": finished,
                "remaining": remaining, "next_layout": next_layout, "total": len(self.layouts),
                "success": sum(1 for r in succ if r.get("success")),
                "mean_score": (sum(float(r.get("score") or 0) for r in succ) / len(succ)) if succ else None}


class Lane:
    def __init__(self, gpu: int, server_gpu: int):
        self.gpu, self.server_gpu = gpu, server_gpu
        self.job: Job | None = None
        self.server: subprocess.Popen | None = None
        self.sim: subprocess.Popen | None = None
        self.phase = "idle"
        self.port = 0
        self.t_phase = time.time()
        self.t_launch = 0.0
        self.records_at_launch = 0
        self.disabled = ""  # "", "drain", "off"
        self.by_control = False  # disabled through CONTROL.json (can be switched back on there)
        self.foreign = False  # stopped because another user's process appeared on its GPU


def base_env(gpu: int) -> dict:
    env = dict(os.environ)
    env.update({
        "HOME": str(ROOT), "HF_HOME": str(ROOT / "hf/home"), "OPENPI_DATA_HOME": str(ROOT / ".cache/openpi"),
        "OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y", "PYTHONUNBUFFERED": "1",
        "PATH": f"{ROOT}/miniconda3/bin:{ROOT}/bin:" + env.get("PATH", ""),
        "CUDA_VISIBLE_DEVICES": str(gpu),
    })
    env.pop("CONDA_PREFIX", None)
    return env


class Queue:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_dir = Path(args.run_dir).resolve()
        if args.env_manifest:
            args.env_manifest = str(Path(args.env_manifest).resolve())
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_fh = (self.run_dir / "queue.log").open("a", encoding="utf-8")
        gpus = [int(g) for g in args.gpus]
        bad = set(gpus) - ALLOWED_GPUS
        if bad:
            sys.exit(f"GPUs {sorted(bad)} are not RoboDojo's (allowed: {sorted(ALLOWED_GPUS)})")
        smap = dict(pair.split(":") for pair in args.server_gpu)  # lane gpu -> server gpu
        self.lanes = [Lane(g, int(smap.get(str(g), g))) for g in gpus]
        if {lane.server_gpu for lane in self.lanes} - ALLOWED_GPUS:
            sys.exit(f"server GPU outside {sorted(ALLOWED_GPUS)}")
        self.jobs = [Job(self.run_dir, p, t, args.max_counted, args.max_attempts)
                     for p in args.policies for t in args.tasks]
        self.started = time.time()
        self.stop = False
        self.disk_paused = False
        self.disk_free_gb: float | None = None
        self.last_disk = 0.0
        self.events: list[tuple[float, int]] = []  # (time, records finished) for throughput
        self.hold: set[str] = set()
        sfx = f"-{args.name}" if args.name else ""  # a second queue on the same run dir has its own files
        self.control_path = self.run_dir / f"CONTROL{sfx}.json"
        self.status_stem = f"STATUS{sfx}"
        self.args_path = self.run_dir / f"queue_args{sfx}.json"
        self.hook_proc: subprocess.Popen | None = None
        self.hook_due = False
        self.hook_last = time.time()

    def log(self, msg: str) -> None:
        line = f"{dt.datetime.now(KST).strftime('%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        self.log_fh.write(line + "\n")
        self.log_fh.flush()

    # ------------------------------------------------------------------ processes
    def start_server(self, lane: Lane, job: Job) -> None:
        ckpt, action, penv = POLICIES[job.policy]
        lane.port = free_port()
        script = XPL / "policy" / job.policy / "setup_eval_policy_server.sh"
        cmd = ["bash", str(script), "RoboDojo", job.task, ckpt, "arx_x5", action, "0", str(lane.server_gpu), penv,
               str(lane.port), "localhost"]
        fh = (job.dir / "server.log").open("a")
        extra = SERVER_ENV.get(job.policy, {})
        fh.write(f"\n===== {now_kst()} launch {job.launches} gpu {lane.server_gpu} port {lane.port}: "
                 f"{' '.join(f'{k}={v}' for k, v in extra.items())} {' '.join(cmd)}\n")
        fh.flush()
        lane.server = subprocess.Popen(cmd, cwd=str(XPL / "policy" / job.policy), env={**base_env(lane.server_gpu), **extra},
                                       stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        lane.phase, lane.t_phase = "server_wait", time.time()
        self.log(f"[gpu{lane.gpu}] {job.name}: server pid {lane.server.pid} port {lane.port} (gpu {lane.server_gpu})")

    def server_ready(self, lane: Lane) -> bool:
        # LISTEN-state check like the official wait_for_policy_server.sh (a bare TCP connect
        # makes the websocket server log a failed handshake)
        try:
            out = subprocess.run(["ss", "-ltnH", f"sport = :{lane.port}"], capture_output=True, text=True,
                                 timeout=10).stdout
            return bool(out.strip())
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            with socket.create_connection(("127.0.0.1", lane.port), timeout=1):
                return True
        except OSError:
            return False

    def start_sim(self, lane: Lane, job: Job) -> None:
        ckpt, action, _ = POLICIES[job.policy]
        env = base_env(lane.gpu)
        env["PYTHONPATH"] = f"{RD}:{XPL}:{SG_SRC}"
        env["ROBODOJO_RUN_ID"] = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{job.policy}_{job.task}_{os.getpid()}_{job.launches}"
        if self.args.eval_num_force:
            env["EVAL_NUM_FORCE"] = "all"
        cmd = [str(PY_SIM), "-u", "-m", "simuguard.integrations.robodojo_eval", "--task", job.task,
               "--policy", job.policy, "--policy-url", f"ws://localhost:{lane.port}", "--ckpt-name", ckpt,
               "--action-type", action, "--layouts", "all", "--resume", "--max-attempts", str(job.max_attempts),
               "--max-counted", str(job.max_counted), "--out", str(self.run_dir / job.policy), "--eval-seed", "0"]
        if self.args.env_manifest:
            cmd += ["--env-manifest", self.args.env_manifest]
        if self.args.batch_envs > 1 and eval_batch(job.policy):
            cmd += ["--num-envs", str(self.args.batch_envs)]
        cmd += self.args.extra
        fh = (job.dir / "sim.log").open("a")
        fh.write(f"\n===== {now_kst()} launch {job.launches} gpu {lane.gpu}: {' '.join(cmd)}\n")
        fh.flush()
        lane.records_at_launch = len(job.records())
        lane.sim = subprocess.Popen(cmd, cwd=str(RD), env=env, stdout=fh, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        lane.phase, lane.t_phase = "sim", time.time()
        self.log(f"[gpu{lane.gpu}] {job.name}: sim pid {lane.sim.pid}")

    def kill(self, proc: subprocess.Popen | None) -> None:
        if proc is None or proc.poll() is not None:
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)  # leftovers of an exited leader
                except ProcessLookupError:
                    pass
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        for _ in range(30):
            if proc.poll() is not None:
                break
            time.sleep(1)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()

    def end_launch(self, lane: Lane, reason: str, preempted: bool = False) -> None:
        job = lane.job
        assert job is not None
        rc = lane.sim.poll() if lane.sim else None
        self.kill(lane.sim)
        self.kill(lane.server)
        current = job.dir / "current_layout.json"  # written by robodojo_eval at episode start
        if preempted:  # we stopped it on purpose: the episode in flight is simply rerun later
            current.unlink(missing_ok=True)
            job.launches -= 1
        if lane.sim is not None and current.exists():  # the process died inside an episode
            try:
                cur = json.loads(current.read_text())
            except json.JSONDecodeError:
                cur = None
            if cur is not None:  # one layout, or the layouts of a batch (--num-envs)
                lids = cur["layouts"] if "layouts" in cur else [cur["layout"]]
                tries = cur["attempts"] if "attempts" in cur else [cur.get("attempt")]
                with (job.dir / "episodes.jsonl").open("a") as fh:
                    for lid, attempt in zip(lids, tries):
                        rec = {"task": job.task, "layout": int(lid), "policy": job.policy, "status": "error",
                               "attempt": attempt, "started_utc": cur.get("started_utc"),
                               "error": f"sim process ended ({reason}, rc={rc}) during this layout",
                               "source": "rd_queue", "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                        fh.write(json.dumps(rec) + "\n")
            current.unlink(missing_ok=True)
        prog = job.progress()
        self.log(f"[gpu{lane.gpu}] {job.name}: launch {job.launches} ended ({reason}, rc={rc}); "
                 f"{prog['done']}/{prog['total']} done, counted {prog['counted']}")
        if prog["finished"]:
            job.state = "done"
        elif job.launches >= self.args.job_launches:
            job.state = "failed"
        else:
            job.state = "pending"
        job.lane = None
        lane.job, lane.server, lane.sim, lane.phase = None, None, None, "idle"
        self.hook_due = True

    def run_hook(self, wait: bool = False) -> None:
        """--on-job-end (e.g. rsync the run dir to another box): after every launch end and every --hook-every s.
        One at a time; a request while one runs is served when it ends."""
        if not self.args.on_job_end:
            return
        if self.hook_proc is not None and self.hook_proc.poll() is None:
            if wait:
                self.hook_proc.wait()
            else:
                return
        if self.hook_proc is not None:
            if self.hook_proc.returncode:
                self.log(f"hook rc={self.hook_proc.returncode} (see hook.log)")
            self.hook_proc = None
        every = self.args.hook_every
        if self.hook_due or wait or (every and time.time() - self.hook_last > every):
            self.hook_due, self.hook_last = False, time.time()
            fh = (self.run_dir / "hook.log").open("a")
            fh.write(f"=== {now_kst()} {self.args.on_job_end}\n"); fh.flush()
            self.hook_proc = subprocess.Popen(["bash", "-c", self.args.on_job_end], stdout=fh, stderr=subprocess.STDOUT,
                                              start_new_session=True)
            if wait:
                self.hook_proc.wait()

    # ------------------------------------------------------------------ memory
    def sample_memory(self) -> None:
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,used_memory",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
            gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=30).stdout
        except Exception:  # noqa: BLE001
            return
        bus_to_idx = {}
        for line in gpus.splitlines():
            idx, bus = (x.strip() for x in line.split(","))
            bus_to_idx[bus] = int(idx)
        ours = set()
        for lane in self.lanes:
            for proc in (lane.server, lane.sim):
                if proc is not None:
                    ours.add(proc.pid)
        for line in out.splitlines():
            try:
                bus, pid_s, mib_s = (x.strip() for x in line.split(","))
                pid, mib = int(pid_s), int(mib_s)
            except ValueError:
                continue
            try:
                pgid = os.getpgid(pid)
            except (ProcessLookupError, PermissionError):
                pgid = None
            gpu = bus_to_idx.get(bus)
            if pgid not in ours:  # someone else's process (only its GPU index is used)
                for lane in self.lanes:
                    if gpu in (lane.gpu, lane.server_gpu) and lane.disabled != "off":
                        self.log(f"[gpu{lane.gpu}] foreign compute process on GPU {gpu}: stopping this lane")
                        self.disable_lane(lane, "now")
                        lane.foreign = True
                continue
            for lane in self.lanes:
                if lane.job is None:
                    continue
                for role, proc in (("server", lane.server), ("sim", lane.sim)):
                    if proc is not None and pgid == proc.pid:
                        lane.job.mem_mib[role] = max(lane.job.mem_mib[role], mib)

    def disable_lane(self, lane: Lane, mode: str) -> None:
        if mode == "now":
            if lane.job is not None:
                self.end_launch(lane, "lane_stopped", preempted=True)
            lane.disabled = "off"
        elif mode == "drain" and not lane.disabled:
            lane.disabled = "drain"
            self.log(f"[gpu{lane.gpu}] draining: finishes {lane.job.name if lane.job else 'nothing'}, then stops")

    def read_control(self) -> None:
        path = self.control_path
        try:
            ctl = json.loads(path.read_text()) if path.exists() else {}
        except json.JSONDecodeError:
            return
        self.hold = set(ctl.get("hold_policies", []))
        for lane in self.lanes:
            mode = ctl.get("disabled_gpus", {}).get(str(lane.gpu))
            if mode in ("now", "drain") and lane.disabled != "off":
                if mode == "now" or lane.disabled != "drain":
                    self.log(f"[gpu{lane.gpu}] CONTROL: {mode}")
                self.disable_lane(lane, mode)
                lane.by_control = True
            elif mode in (None, "on") and lane.disabled and lane.by_control and not lane.foreign:
                self.log(f"[gpu{lane.gpu}] CONTROL: lane back on")  # a lane stopped for a foreign process stays off
                lane.disabled, lane.phase, lane.by_control = "", "idle", False

    # ------------------------------------------------------------------ status
    def write_status(self) -> None:
        rows, total_rem, total_done = [], 0, 0
        for job in self.jobs:
            p = job.progress()
            total_rem += p["remaining"]
            total_done += p["done"]
            rows.append({"job": job.name, "state": job.state, "lane": job.lane, "launches": job.launches,
                         "mem_mib": job.mem_mib, **{k: p[k] for k in ("done", "total", "counted", "statuses",
                                                                      "success", "mean_score", "remaining")}})
        t = time.time()
        self.events.append((t, total_done))
        window = [e for e in self.events if t - e[0] <= 7200]
        rate = None
        if len(window) >= 2 and window[-1][0] - window[0][0] > 300:
            rate = (window[-1][1] - window[0][1]) / ((window[-1][0] - window[0][0]) / 3600)
        eta = None
        if rate and rate > 0:
            eta = (dt.datetime.now(KST) + dt.timedelta(hours=total_rem / rate)).strftime("%Y-%m-%d %H:%M KST")
        status = {"updated": now_kst(), "started": dt.datetime.fromtimestamp(self.started, KST).isoformat(),
                  "gpus": [lane.gpu for lane in self.lanes], "episodes_done": total_done,
                  "episodes_remaining": total_rem, "episodes_per_hour_2h": rate, "eta": eta,
                  "disk_free_gb": self.disk_free_gb and round(self.disk_free_gb), "disk_paused": self.disk_paused,
                  "lanes": [{"gpu": lane.gpu, "server_gpu": lane.server_gpu, "phase": lane.phase + (f" [{lane.disabled}]" if lane.disabled else ""),
                             "job": lane.job.name if lane.job else None,
                             "phase_min": round((t - lane.t_phase) / 60, 1)} for lane in self.lanes],
                  "jobs": rows}
        (self.run_dir / f"{self.status_stem}.json").write_text(json.dumps(status, indent=1))
        lines = [f"updated {status['updated']}  done {total_done}  remaining {total_rem}  "
                 f"rate {rate and round(rate, 1)}/h  ETA {eta}"
                 + (f"  disk {round(self.disk_free_gb)} GB free" if self.disk_free_gb is not None else "")
                 + ("  PAUSED (disk)" if self.disk_paused else "")]
        for ln in status["lanes"]:
            lines.append(f"  gpu{ln['gpu']} (server gpu{ln['server_gpu']}): {ln['phase']:<11} {ln['job']} "
                         f"{ln['phase_min']} min")
        lines.append(f"{'job':<36}{'state':<9}{'done':>9}{'cnt':>5}{'succ':>5}{'score':>7}  mem srv/sim MiB  statuses")
        for r in rows:
            ms = f"{r['mean_score']:.1f}" if r["mean_score"] is not None else "-"
            lines.append(f"{r['job']:<36}{r['state']:<9}{r['done']:>4}/{r['total']:<4}{r['counted']:>5}"
                         f"{r['success']:>5}{ms:>7}  {r['mem_mib']['server']:>6}/{r['mem_mib']['sim']:<6}  "
                         f"{json.dumps(r['statuses'])}")
        (self.run_dir / f"{self.status_stem}.txt").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------------ main loop
    def step_lane(self, lane: Lane) -> None:
        if lane.job is None:
            if lane.disabled:
                lane.disabled, lane.phase = "off", "off"
                return
            job = next((j for j in self.jobs if j.state == "pending" and j.policy not in self.hold
                        and not j.progress()["finished"]), None)
            for j in self.jobs:
                if j.state == "pending" and j.progress()["finished"]:
                    j.state = "done"
            if job is None or self.stop or self.disk_paused:
                return
            job.state, job.lane, lane.job = "running", lane.gpu, job
            job.launches += 1
            lane.t_launch = time.time()
            self.start_server(lane, job)
            return
        job = lane.job
        if lane.phase == "server_wait":
            if lane.server.poll() is not None:
                self.log(f"[gpu{lane.gpu}] {job.name}: server exited rc={lane.server.returncode} before listening")
                self.end_launch(lane, "server_exit")
            elif self.server_ready(lane):
                self.log(f"[gpu{lane.gpu}] {job.name}: server up after {time.time() - lane.t_phase:.0f} s")
                self.start_sim(lane, job)
            elif time.time() - lane.t_phase > self.args.server_timeout:
                self.end_launch(lane, "server_timeout")
            return
        if lane.phase == "sim":
            if lane.sim.poll() is not None:
                self.end_launch(lane, "sim_exit")
            elif lane.server.poll() is not None:
                self.log(f"[gpu{lane.gpu}] {job.name}: server died rc={lane.server.returncode}")
                self.end_launch(lane, "server_died")
            else:
                log = job.dir / "episodes.jsonl"
                last = max(log.stat().st_mtime if log.exists() else 0, lane.t_phase)
                if time.time() - last > self.args.stall:
                    self.end_launch(lane, "stall")

    def check_disk(self) -> None:
        """Pause the queue below --min-free-gb free on the run dir's filesystem, resume at --resume-free-gb."""
        if not self.args.min_free_gb or time.time() - self.last_disk < 60:
            return
        self.last_disk = time.time()
        self.disk_free_gb = shutil.disk_usage(self.run_dir).free / 1e9
        resume = self.args.resume_free_gb or self.args.min_free_gb + 10
        if not self.disk_paused and self.disk_free_gb < self.args.min_free_gb:
            self.disk_paused = True
            self.log(f"disk watch: {self.disk_free_gb:.0f} GB free < {self.args.min_free_gb} GB: stopping running "
                     f"launches, no new ones until {resume} GB free")
            for lane in self.lanes:
                if lane.job is not None:
                    self.end_launch(lane, "disk_low", preempted=True)
        elif self.disk_paused and self.disk_free_gb >= resume:
            self.disk_paused = False
            self.log(f"disk watch: {self.disk_free_gb:.0f} GB free >= {resume} GB: resuming")

    def run(self) -> None:
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop", True))
        self.log(f"queue start: lanes {[(lane.gpu, lane.server_gpu) for lane in self.lanes]}, "
                 f"{len(self.jobs)} jobs, max_counted={self.args.max_counted}")
        self.args_path.write_text(json.dumps(vars(self.args), indent=1))
        modes = run_modes(list(self.args.policies), self.args.batch_envs)
        prev = self.run_dir / "modes.json"
        old = {}
        if prev.exists():  # never mix one-env and batched episodes of a policy across restarts
            old = json.loads(prev.read_text())
            for p, m in modes.items():
                if p in old and old[p]["num_envs"] != m["num_envs"] and (self.run_dir / p).exists() and any(
                        (self.run_dir / p).glob("*/episodes.jsonl")):
                    raise SystemExit(f"{p}: num_envs {old[p]['num_envs']} in modes.json, {m['num_envs']} now; "
                                     "episodes exist, refusing to mix modes")
            modes = {**old, **modes}  # a queue for a subset of policies keeps the other rows
        prev.write_text(json.dumps(modes, indent=1))
        self.log("modes: " + json.dumps({p: m["num_envs"] for p, m in modes.items() if p in self.args.policies}))
        last_status = last_mem = 0.0
        while True:
            self.read_control()
            self.check_disk()
            for lane in self.lanes:
                self.step_lane(lane)
            t = time.time()
            if t - last_mem > 60:  # nvidia-smi is slow under driver contention: once a minute
                self.sample_memory()
                last_mem = t
            if t - last_status > self.args.status_every:
                self.write_status()
                last_status = t
            self.run_hook()
            if self.stop:
                for lane in self.lanes:
                    if lane.job is not None:
                        self.end_launch(lane, "queue_stopped")
                break
            if all(lane.job is None for lane in self.lanes) and (
                    all(lane.disabled == "off" for lane in self.lanes)
                    or not any(j.state == "pending" and not j.progress()["finished"] for j in self.jobs)):
                if not any(j.state == "pending" and j.policy in self.hold for j in self.jobs) or all(
                        lane.disabled == "off" for lane in self.lanes):
                    break
            time.sleep(5)
        self.write_status()
        self.run_hook(wait=True)
        self.log("queue end: " + json.dumps({s: sum(1 for j in self.jobs if j.state == s)
                                             for s in ("done", "failed", "pending")}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gpus", nargs="+", required=True)
    ap.add_argument("--server-gpu", nargs="*", default=[], help="LANE:SERVER pairs, e.g. 3:4 (default: same card)")
    ap.add_argument("--policies", nargs="+", required=True, choices=sorted(POLICIES))
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--max-counted", type=int, default=0, help="0 = every layout file (full run)")
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--job-launches", type=int, default=6, help="give a job up after this many launches")
    ap.add_argument("--server-timeout", type=float, default=1200, help="seconds (official wait_for_policy_server)")
    ap.add_argument("--stall", type=float, default=3600, help="kill the sim if no episode record for this long")
    ap.add_argument("--status-every", type=float, default=60)
    ap.add_argument("--env-manifest", default=None)
    ap.add_argument("--batch-envs", default="1",
                    help="eval_batch policies (deploy.yml): run N envs side by side (robodojo_eval --num-envs); "
                         "'official' = scene.num_envs of RoboDojo's env_cfg/sim/sim_config.yml, what eval_policy.sh uses")
    ap.add_argument("--eval-num-force", action="store_true", help="set EVAL_NUM_FORCE=all (recorded in run.json)")
    ap.add_argument("--name", default="", help="second queue on the same run dir: CONTROL-NAME.json, STATUS-NAME.*")
    ap.add_argument("--on-job-end", default="", help="shell command after each launch end (e.g. rsync to another box)")
    ap.add_argument("--hook-every", type=float, default=0, help="also run --on-job-end every this many s (0 = off)")
    ap.add_argument("--min-free-gb", type=float, default=0, help="disk watch: pause below this many GB free (0 = off)")
    ap.add_argument("--resume-free-gb", type=float, default=0, help="disk watch: resume at this many GB free "
                    "(default: --min-free-gb + 10)")
    ap.add_argument("extra", nargs="*", help="extra robodojo_eval args after --")
    args = ap.parse_args()
    args.batch_envs = official_num_envs() if args.batch_envs == "official" else int(args.batch_envs)
    Queue(args).run()


if __name__ == "__main__":
    main()
