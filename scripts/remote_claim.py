"""Claims on a formal output dir for drivers on another machine (flock does not cross machines).

usage: remote_claim.py acquire OUT TASK OWNER [ATTEMPTS]
       remote_claim.py release OUT TASK OWNER done|failed
       remote_claim.py status OUT TASK

Runs on the machine that holds OUT; the other machine calls it over ssh (as a forced command it
reads SSH_ORIGINAL_COMMAND).  acquire takes the same flock that robolab_formal.py drivers take on
``OUT/_claims/TASK.lock`` and creates ``OUT/_claims/TASK.remote/`` with mkdir; a detached holder
process keeps the flock while that directory exists, so drivers here skip the task (old drivers by
the flock, new ones by the directory too), and a held flock or an existing directory makes acquire
fail.  release done writes ``TASK.remote/DONE.json`` and keeps the holder, so neither side runs the
task again; release failed removes the directory, the holder exits and the task is free again.
Prints one word (claimed, busy, done, released) and exits 0 for claimed/released, 1 for busy,
2 for done (finished here, finished remotely, or out of attempts)."""
from __future__ import annotations

import fcntl
import json
import os
import re
import shlex
import shutil
import sys
import time
from pathlib import Path

ALLOWED_OUT = re.compile(r"^/path/to/simuguard-robolab/runs/formal_(pi05|gr00t)$")
NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_request(argv: list[str]) -> list[str]:
    """Validated argument list; raises ValueError on anything else."""
    if not argv:
        raise ValueError("empty request")
    op = argv[0]
    shape = {"acquire": (4, 5), "release": (5, 5), "status": (3, 3)}
    if op not in shape or not shape[op][0] <= len(argv) <= shape[op][1]:
        raise ValueError(f"bad request {argv}")
    if not ALLOWED_OUT.match(argv[1]):
        raise ValueError(f"out not allowed: {argv[1]}")
    for name in argv[2:4]:
        if not NAME.match(name):
            raise ValueError(f"bad name {name}")
    if op == "acquire" and len(argv) == 5 and not argv[4].isdigit():
        raise ValueError(f"bad attempts {argv[4]}")
    if op == "release" and argv[4] not in ("done", "failed"):
        raise ValueError(f"bad state {argv[4]}")
    return argv


def marker(out: Path, task: str) -> Path:
    return out / "_claims" / f"{task}.remote"


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def hold(fd: int, mark: Path, poll: float) -> None:
    """Detached child that keeps the flock fd open while the marker directory exists."""
    child = os.fork()
    if child:
        os.waitpid(child, 0)
        return
    os.setsid()
    if os.fork():
        os._exit(0)
    devnull = os.open(os.devnull, os.O_RDWR)
    for k in (0, 1, 2):
        os.dup2(devnull, k)
    (mark / "holder.pid").write_text(str(os.getpid()))
    while mark.is_dir():
        time.sleep(poll)
    os._exit(0)


def take_lock(out: Path, task: str) -> int | None:
    claims = out / "_claims"
    claims.mkdir(exist_ok=True)
    fd = os.open(claims / f"{task}.lock", os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def start_holder(fd: int, mark: Path, owner: str, poll: float) -> None:
    (mark / "holder.pid").unlink(missing_ok=True)
    hold(fd, mark, poll)
    os.close(fd)
    for _ in range(200):
        if (mark / "holder.pid").exists():
            break
        time.sleep(0.01)
    (mark / "owner.json").write_text(json.dumps({"owner": owner, "utc": utc(), "holder_pid": int((mark / "holder.pid").read_text())}))


def holder_pid(mark: Path) -> int | None:
    try:
        return int((mark / "holder.pid").read_text())
    except (FileNotFoundError, ValueError):
        return None


def acquire(out: Path, task: str, owner: str, attempts: int = 3, poll: float = 5.0) -> str:
    if (out / task / "DONE.json").exists() or len(list((out / "_failed").glob(f"{task}.*"))) >= attempts:
        return "done"
    mark = marker(out, task)
    if mark.is_dir():
        if (mark / "DONE.json").exists():
            return "done"
        info = json.loads((mark / "owner.json").read_text()) if (mark / "owner.json").exists() else {}
        if info.get("owner") != owner:
            return "busy"
        pid = holder_pid(mark)
        if pid is not None and alive(pid):
            return "claimed"
        fd = take_lock(out, task)
        if fd is None:
            return "busy"
        start_holder(fd, mark, owner, poll)
        return "claimed"
    fd = take_lock(out, task)
    if fd is None:
        return "busy"
    try:
        mark.mkdir()
    except FileExistsError:
        os.close(fd)
        return "busy"
    start_holder(fd, mark, owner, poll)
    return "claimed"


def release(out: Path, task: str, owner: str, state: str) -> str:
    mark = marker(out, task)
    info = json.loads((mark / "owner.json").read_text()) if (mark / "owner.json").exists() else {}
    if info.get("owner") != owner:
        return "busy"
    if state == "done":
        (mark / "DONE.json").write_text(json.dumps({"owner": owner, "utc": utc()}))
    else:
        shutil.rmtree(mark)
    return "released"


def status(out: Path, task: str) -> str:
    mark = marker(out, task)
    if (out / task / "DONE.json").exists() or (mark / "DONE.json").exists():
        return "done"
    return "busy" if mark.is_dir() else "free"


def main(argv: list[str]) -> int:
    if not argv and "SSH_ORIGINAL_COMMAND" in os.environ:
        argv = shlex.split(os.environ["SSH_ORIGINAL_COMMAND"])
    try:
        argv = parse_request(argv)
    except ValueError as e:
        print(f"error {e}")
        return 3
    op, out, task = argv[0], Path(argv[1]), argv[2]
    if op == "acquire":
        word = acquire(out, task, argv[3], int(argv[4]) if len(argv) == 5 else 3)
    elif op == "release":
        word = release(out, task, argv[3], argv[4])
    else:
        word = status(out, task)
    print(word)
    return {"claimed": 0, "released": 0, "free": 0, "busy": 1, "done": 2}[word]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
