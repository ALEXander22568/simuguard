"""Resumable driver for the full RoboLab recording pass (one sim process per task).

Runs ``simuguard.integrations.robolab_eval`` for every task of a shard, one process each,
into ``<out>/<task>/``.  Any number of drivers can share one ``<out>``: a driver claims a
task with an exclusive flock on ``<out>/_claims/<task>.lock``, held by the driver and its sim
process until both exit, so lanes can be added or moved to another policy at any time.  A task
is done when its process exited 0 and its ``run_index.jsonl`` holds ``--num-runs`` runs; then
``<out>/<task>/DONE.json`` is written and the task is skipped from then on.  DONE.json also
lists data issues found in the batch files (contact views that failed, monitor errors,
writes that replay cannot re-apply, missing files); such tasks are not retried, the issues
are reported.  An unfinished task directory (crash, or no exit within ``--task-timeout``
seconds) is moved to ``<out>/_failed/`` and the task is retried until it has failed
``--attempts`` times.  ``touch <out>/STOP`` stops every driver after its current task.

A driver on another machine (``--remote-claim``, an ssh command that runs scripts/remote_claim.py on
the machine holding the shared ``--remote-out``) also claims each task there before running it into its
own ``<out>``; tasks that machine has finished or is running are skipped, and drivers there skip tasks
with a ``<out>/_claims/<task>.remote`` directory.  ``--reverse`` walks the task list from the end, so
two machines meet in the middle.

Plain Python (no Isaac imports); start it once per sim process, e.g.::

    python scripts/robolab_formal.py --policy pi05 --gpu 2 --port 8000 \\
        --out /path/to/simuguard-robolab/runs/formal_pi05
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path


def benchmark_tasks(robolab: Path) -> list[str]:
    names = set()
    for path in (robolab / "robolab" / "tasks" / "benchmark").glob("*.py"):
        names.update(re.findall(r"^class (\w+)\(Task\)", path.read_text(), flags=re.M))
    return sorted(names)


def runs_done(task_dir: Path) -> int:
    index = task_dir / "run_index.jsonl"
    if not index.exists():
        return 0
    return len({json.loads(line)["run"] for line in index.read_text().splitlines() if line.strip()})


def task_issues(task_dir: Path, num_runs: int) -> list[str]:
    batch = next(task_dir.glob("*/batch"), None)
    if batch is None:
        return ["no batch directory"]
    issues = []
    install = json.loads((batch / "install.json").read_text()) if (batch / "install.json").exists() else None
    if install is None:
        issues.append("no install.json")
    elif install.get("contact_view_errors"):
        issues.append(f"contact view errors: {sorted(install['contact_view_errors'])}")
    for k in range(num_runs):
        for name in (f"run_{k}.json", f"run_{k}_writes.npz", f"run_{k}_robot.npz"):
            if not (batch / name).exists():
                issues.append(f"missing {name}")
        if not (batch / f"run_{k}.json").exists():
            continue
        run = json.loads((batch / f"run_{k}.json").read_text())
        if run.get("unknown_views"):
            issues.append(f"run {k}: writes on unknown views {run['unknown_views']}")
        if run.get("apply_force_calls"):
            issues.append(f"run {k}: {run['apply_force_calls']} apply_* calls")
        errors = sum(seg.get("monitor_errors") or 0 for seg in run.get("segments", []))
        if errors:
            issues.append(f"run {k}: {errors} monitor errors")
    return issues


def claim(out: Path, task: str) -> int | None:
    """An open fd holding an exclusive flock on ``<out>/_claims/<task>.lock``, or None if another driver holds it.

    The fd is passed on to the sim process, so the lock lasts until both the driver and the sim are gone."""
    claims = out / "_claims"
    claims.mkdir(exist_ok=True)
    fd = os.open(claims / f"{task}.lock", os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    os.ftruncate(fd, 0)
    os.write(fd, json.dumps({"pid": os.getpid(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}).encode())
    return fd


def remote_taken(out: Path, task: str) -> bool:
    return (out / "_claims" / f"{task}.remote").is_dir()


def remote_call(prefix: list[str], args: list[str], tries: int = 1, wait: float = 30.0) -> str | None:
    """First word printed by remote_claim.py on the other machine, or None if ssh failed every try."""
    for i in range(tries):
        try:
            r = subprocess.run(prefix + [shlex.join(args)], capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
            if r.returncode != 255 and r.stdout.split():
                return r.stdout.split()[0]
        except subprocess.TimeoutExpired:
            pass
        if i + 1 < tries:
            time.sleep(wait)
    return None


def failures(out: Path, task: str) -> int:
    return len(list((out / "_failed").glob(f"{task}.*")))


def move_failed(out: Path, task: str) -> None:
    failed = out / "_failed"
    failed.mkdir(exist_ok=True)
    shutil.move(str(out / task), str(failed / f"{task}.{time.time_ns()}"))


def run_task(cmd: list[str], cwd: str, env: dict[str, str], log: Path, timeout: float,
             pass_fds: tuple[int, ...] = ()) -> int | None:
    """Exit code, or None after killing the process group on timeout."""
    with log.open("w") as fh:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True, pass_fds=pass_fds)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return None
        except BaseException:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["pi05", "gr00t"], required=True)
    p.add_argument("--policy-checkpoint", default=None)
    p.add_argument("--server-info", default=None, help="policy server JSON (pid, start time, memory fraction) for the manifests")
    p.add_argument("--open-loop-horizon", type=int, default=None,
                   help="default: 8 for gr00t (RoboLab's validated N1.7 baseline), the client default for pi05")
    p.add_argument("--gpu", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--num-envs", type=int, default=10)
    p.add_argument("--num-runs", type=int, default=2)
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--task-timeout", type=float, default=6 * 3600, help="seconds per task process")
    p.add_argument("--tasks", nargs="*", default=None, help="default: every class in robolab/tasks/benchmark")
    p.add_argument("--robolab", default="/path/to/simuguard-robolab/RoboLab")
    p.add_argument("--simuguard", default="/path/to/simuguard-robolab/simuguard")
    p.add_argument("--reverse", action="store_true", help="take tasks from the end of the list")
    p.add_argument("--remote-claim", default=None, help="ssh command to the machine that holds --remote-out")
    p.add_argument("--remote-out", default=None, help="the shared output dir on that machine")
    p.add_argument("--owner", default=None, help="claim owner name sent with --remote-claim")
    a = p.parse_args()
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    if a.open_loop_horizon is None and a.policy == "gr00t":
        a.open_loop_horizon = 8

    robolab = Path(a.robolab)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = a.tasks or benchmark_tasks(robolab)
    mine = [t for i, t in enumerate(tasks) if i % a.shards == a.shard]
    if a.reverse:
        mine.reverse()
    remote = shlex.split(a.remote_claim) if a.remote_claim else None
    remote_done: set[str] = set()
    log = out / f"driver_gpu{a.gpu}_shard{a.shard}.jsonl"
    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "Y", "CUDA_VISIBLE_DEVICES": str(a.gpu), "PYTHONPATH": a.simuguard}
    python = str(robolab / ".venv" / "bin" / "python")

    def note(**row) -> None:
        with log.open("a") as fh:
            fh.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "shard": a.shard, **row}) + "\n")

    note(event="start", tasks=len(mine), args=vars(a))
    while True:
        if (out / "STOP").exists():
            note(event="stopped")
            return 0
        task, lock = None, None
        busy = set()
        for t in mine:
            if t in remote_done or t in busy or remote_taken(out, t):
                continue
            if (out / t / "DONE.json").exists() or failures(out, t) >= a.attempts:
                continue
            lock = claim(out, t)
            if lock is not None and remote:
                word = remote_call(remote, ["acquire", a.remote_out, t, a.owner, str(a.attempts)])
                note(event="remote_acquire", task=t, word=word)
                if word != "claimed":
                    os.close(lock)
                    lock = None
                    (remote_done if word == "done" else busy).add(t)
            if lock is not None:
                task = t
                break
        if task is None:
            break
        try:
            if (out / task / "DONE.json").exists() or failures(out, task) >= a.attempts:
                continue
            task_dir = out / task
            if task_dir.exists():
                move_failed(out, task)
            task_dir.mkdir(parents=True)
            cmd = [python, "-m", "simuguard.integrations.robolab_eval", "--policy", a.policy, "--task", task,
                   "--num_envs", str(a.num_envs), "--num_runs", str(a.num_runs), "--remote-port", str(a.port),
                   "--out", str(task_dir), "--headless", "--video-mode", "none"]
            if a.policy_checkpoint:
                cmd += ["--policy-checkpoint", a.policy_checkpoint]
            if a.server_info:
                cmd += ["--server-info", a.server_info]
            if a.open_loop_horizon is not None:
                cmd += ["--open-loop-horizon", str(a.open_loop_horizon)]
            t0 = time.time()
            code = run_task(cmd, str(robolab), env, task_dir / "process.log", a.task_timeout, pass_fds=(lock,))
            runs = runs_done(task_dir)
            ok = code == 0 and runs == a.num_runs
            row = {"task": task, "gpu": a.gpu, "exit": code, "timed_out": code is None, "runs": runs,
                   "wall_s": round(time.time() - t0, 1), "ok": ok}
            if ok:
                row["issues"] = task_issues(task_dir, a.num_runs)
            note(event="task", **row)
            if ok:
                (task_dir / "DONE.json").write_text(json.dumps(row, indent=2))
            else:
                move_failed(out, task)
            if remote:
                word = remote_call(remote, ["release", a.remote_out, task, a.owner, "done" if ok else "failed"], tries=5)
                note(event="remote_release", task=task, ok=ok, word=word)
        finally:
            os.close(lock)
    left = [t for t in mine if not (out / t / "DONE.json").exists() and t not in remote_done and not remote_taken(out, t)]
    note(event="end", left=left)
    return 0 if not left else 1


if __name__ == "__main__":
    raise SystemExit(main())
