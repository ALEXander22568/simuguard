#!/usr/bin/env python3
"""Throughput probe: the official RoboDojo client (``src/eval_client/main.py``, no SimuGuard) at a given num_envs.

usage: python3 bench_batch.py --out DIR --policy Pi_05 --task hang_mugs --num-envs 4 --sim-gpu 5 --server-gpu 6

Starts the policy's official ``setup_eval_policy_server.sh``, waits for it, then runs ``main.py`` with the
arguments of ``scripts/eval_policy.sh`` and ``EVAL_NUM=<num-envs>`` (one batch: layouts 0..N-1 of seed 0).
Writes ``DIR/bench.json``: wall times (app start -> first action -> end), peak GPU memory of server and sim,
the official ``_result.json``.  Timing study only; nothing here feeds the reported numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rd_queue import ALLOWED_GPUS, POLICIES, PY_SIM, RD, XPL, base_env, free_port, now_kst  # noqa: E402

KIT_ARGS = " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"


def gpu_mem_by_pid() -> dict[int, int]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    res = {}
    for line in out.splitlines():
        try:
            pid, mem = (int(x) for x in line.split(","))
            res[pid] = mem
        except ValueError:
            pass
    return res


def group_pids(pgid: int) -> set[int]:
    out = subprocess.run(["ps", "-eo", "pid=,pgid="], capture_output=True, text=True).stdout
    return {int(p) for p, g in (line.split() for line in out.splitlines()) if int(g) == pgid}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--num-envs", type=int, required=True)
    ap.add_argument("--eval-num", type=int, default=0, help="default: num-envs (one batch)")
    ap.add_argument("--sim-gpu", type=int, required=True)
    ap.add_argument("--server-gpu", type=int, required=True)
    args = ap.parse_args()
    if {args.sim_gpu, args.server_gpu} - ALLOWED_GPUS:
        sys.exit("GPUs outside 3-6")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ckpt, action, penv = POLICIES[args.policy]
    info = f"ckpt_name={ckpt},action_type={action}"
    port = free_port()
    res: dict = {"policy": args.policy, "task": args.task, "num_envs": args.num_envs, "started_kst": now_kst(),
                 "sim_gpu": args.sim_gpu, "server_gpu": args.server_gpu}
    t0 = time.time()
    srv = subprocess.Popen(
        ["bash", str(XPL / "policy" / args.policy / "setup_eval_policy_server.sh"), "RoboDojo", args.task, ckpt,
         "arx_x5", action, "0", str(args.server_gpu), penv, str(port), "localhost"],
        cwd=str(XPL / "policy" / args.policy), env=base_env(args.server_gpu),
        stdout=(out / "server.log").open("w"), stderr=subprocess.STDOUT, start_new_session=True)
    sim = None
    peak = {"server": 0, "sim": 0}
    stop = threading.Event()

    def sampler() -> None:
        while not stop.wait(60):  # once a minute: nvidia-smi contends with Isaac in the driver
            mem = gpu_mem_by_pid()
            for role, proc in (("server", srv), ("sim", sim)):
                if proc is not None and proc.poll() is None:
                    peak[role] = max(peak[role], sum(mem.get(p, 0) for p in group_pids(proc.pid)))

    threading.Thread(target=sampler, daemon=True).start()
    try:
        while not subprocess.run(["ss", "-ltnH", f"sport = :{port}"], capture_output=True, text=True).stdout.strip():
            if srv.poll() is not None or time.time() - t0 > 1200:
                raise RuntimeError("server did not come up")
            time.sleep(3)
        res["server_start_s"] = round(time.time() - t0, 1)
        env = base_env(args.sim_gpu)
        run_id = f"bench_{args.policy}_{args.task}_n{args.num_envs}_{int(time.time())}"
        env.update({"ROBODOJO_RUN_ID": run_id, "EVAL_NUM": str(args.eval_num or args.num_envs),
                    "PYTHONPATH": f"{RD}:{XPL}"})
        cmd = [str(PY_SIM), "-u", "src/eval_client/main.py", "--task_name", args.task, "--env_cfg_type", "arx_x5",
               "--num_envs", str(args.num_envs), "--enable_cameras", "--kit_args", KIT_ARGS, "--device_id", "0",
               "--policy_name", args.policy, "--port", str(port), "--protocol", "ws",
               "--policy_server_url", f"ws://localhost:{port}", "--additional_info", info, "--seed", "0",
               "--host", "localhost", "--headless"]
        res["cmd"] = cmd
        res["run_id"] = run_id
        t1 = time.time()
        sim = subprocess.Popen(cmd, cwd=str(RD), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=True)
        marks: dict[str, float] = {}
        tail = b""
        with (out / "sim.log").open("wb") as log:
            while True:
                chunk = os.read(sim.stdout.fileno(), 65536)
                if not chunk:
                    break
                log.write(chunk)
                tail = (tail + chunk)[-4096:]
                now = time.time() - t1
                if b"step: \x1b[92m1 /" in tail and "first_action" not in marks:
                    marks["first_action"] = round(now, 1)
                for env_idx in range(args.num_envs):
                    if f"env{env_idx} step".encode() in tail:
                        marks.setdefault(f"env{env_idx}_first_print", round(now, 1))
        sim.wait()
        res["sim_rc"] = sim.returncode
        res["sim_wall_s"] = round(time.time() - t1, 1)
        res["marks_s"] = marks
        results = sorted((RD / "eval_result").glob(f"RoboDojo/{args.task}/{args.policy}/arx_x5/0_{info}/{run_id}*/_result.json"))
        res["result_files"] = [str(p) for p in results]
        if results:
            res["result"] = json.loads(results[-1].read_text())
    finally:
        stop.set()
        res["peak_mem_mib"] = dict(peak)
        for proc in (sim, srv):
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        time.sleep(5)
        for proc in (sim, srv):
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        res["finished_kst"] = now_kst()
        (out / "bench.json").write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps({k: v for k, v in res.items() if k != "cmd"}, indent=2, default=str))


if __name__ == "__main__":
    main()
