#!/usr/bin/env python3
"""Start each policy's official server alone on one GPU and record its GPU memory once it listens.

usage: python3 probe_servers.py --gpu 5 --out PROBE.json [--policies Pi_0 X_VLA ...] [--task hang_mugs]

Same command as rd_queue (``setup_eval_policy_server.sh`` with the eval.sh arguments).  For each policy:
GPU memory before, wait until the port listens (``--timeout``), wait ``--settle`` s, read memory again
(one nvidia-smi call per policy), stop the server's process group.  This is load-time memory: activations
of the first requests come on top (calibrate with a policy whose in-run memory is known, e.g. Pi_05).
The GPU must have no other compute process (checked before each policy).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rd_queue as q  # noqa: E402


def gpu_mem(gpu: int) -> tuple[int, int]:
    out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    used, total = (int(x) for x in out.strip().split(","))
    return used, total


def gpu_apps(gpu: int) -> list[str]:
    uuid = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True).stdout.strip()
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    return [line.split(",")[1].strip() for line in out.splitlines() if line.startswith(uuid)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policies", nargs="*", default=list(q.POLICIES))
    ap.add_argument("--task", default="hang_mugs")
    ap.add_argument("--timeout", type=float, default=1200)
    ap.add_argument("--settle", type=float, default=60)
    ap.add_argument("--sample", type=float, default=60, help="GPU memory sample interval during load (peak)")
    args = ap.parse_args()
    if args.gpu not in q.ALLOWED_GPUS:
        sys.exit(f"GPU outside {sorted(q.ALLOWED_GPUS)}")
    out_path = Path(args.out)
    logs = out_path.parent / (out_path.stem + "_logs")
    logs.mkdir(parents=True, exist_ok=True)
    rows = json.loads(out_path.read_text()) if out_path.exists() else {}
    for policy in args.policies:
        if policy in rows and rows[policy].get("listening"):
            continue
        apps = gpu_apps(args.gpu)
        if apps:
            sys.exit(f"GPU {args.gpu} has compute processes {apps}; stopping")
        ckpt, action, penv = q.POLICIES[policy]
        port = q.free_port()
        cmd = ["bash", str(q.XPL / "policy" / policy / "setup_eval_policy_server.sh"), "RoboDojo", args.task, ckpt,
               "arx_x5", action, "0", str(args.gpu), penv, str(port), "localhost"]
        before, total = gpu_mem(args.gpu)
        t0 = time.time()
        with (logs / f"{policy}.log").open("w") as fh:
            proc = subprocess.Popen(cmd, cwd=str(q.XPL / "policy" / policy), env={**q.base_env(args.gpu), **q.SERVER_ENV.get(policy, {})}, stdout=fh,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            listening = False
            peak, t_sample = before, t0
            while time.time() - t0 < args.timeout and proc.poll() is None:
                if subprocess.run(["ss", "-ltnH", f"sport = :{port}"], capture_output=True, text=True).stdout.strip():
                    listening = True
                    break
                if time.time() - t_sample >= args.sample:
                    peak, t_sample = max(peak, gpu_mem(args.gpu)[0]), time.time()
                time.sleep(5)
            load_s = round(time.time() - t0, 1)
            if listening:
                time.sleep(args.settle)
            after, _ = gpu_mem(args.gpu)
            peak = max(peak, after)
            rc = proc.poll()
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    break
                time.sleep(15)
        rows[policy] = {"listening": listening, "exit_code_before_stop": rc, "load_s": load_s,
                        "gpu_mem_before_mib": before, "gpu_mem_after_load_mib": after,
                        "server_mib": after - before, "gpu_mem_peak_sampled_mib": peak,
                        "server_peak_mib": peak - before, "sample_s": args.sample, "gpu_total_mib": total, "cmd": cmd}
        out_path.write_text(json.dumps(rows, indent=1))
        print(f"{policy}: listening={listening} load {load_s}s server {after - before} MiB (peak sampled {peak - before})",
              flush=True)
        for _ in range(5):  # wait until the card is empty again (nvidia-smi at most once a minute)
            time.sleep(60)
            if not gpu_apps(args.gpu):
                break


if __name__ == "__main__":
    main()
