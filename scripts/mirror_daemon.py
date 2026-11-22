"""Copy finished, re-packed formal tasks to a second machine and check every file by sha256.  Copy only: nothing
is deleted on either side.

usage: mirror_daemon.py --dest USER@HOST:DIR --port P --key KEY [--bwlimit KBPS] [--period S]

Every --period seconds: a task of runs/formal_pi05 or runs/formal_gr00t is ready when it has DONE.json and the
repack daemon is done with it (every .npz except run_<k>_robot.npz holds only LZMA entries).  Each ready task with
no matching row in runs/<policy>/MIRROR.jsonl is taken under its claim lock (skipped while a lane or the repack
daemon holds it), rsynced to DIR/<policy>/<task>/ under nice/ionice with --bwlimit, then hashed on both sides.
One row per attempt in runs/<policy>/MIRROR.jsonl.  runs/MIRROR_BWLIMIT overrides --bwlimit at the next task;
runs/MIRROR_STOP stops after the current task."""
import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import time
import zipfile
from pathlib import Path

RUNS = Path(os.environ.get("MIRROR_RUNS", "/path/to/simuguard-robolab/runs"))
POLICY_DIRS = ("formal_pi05", "formal_gr00t")
SKIP = ".repack-tmp"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def repacked(task_dir: Path) -> bool:
    for path in task_dir.rglob("*.npz"):
        if path.name.endswith("_robot.npz"):
            continue
        with zipfile.ZipFile(path) as z:
            if any(info.compress_type != zipfile.ZIP_LZMA for info in z.infolist()):
                return False
    return True


def local_hashes(task_dir: Path) -> dict[str, str]:
    out = {}
    for path in sorted(p for p in task_dir.rglob("*") if p.is_file() and not p.name.endswith(SKIP)):
        h = hashlib.sha256()
        with path.open("rb") as fh:
            while chunk := fh.read(1 << 24):
                h.update(chunk)
        out[str(path.relative_to(task_dir))] = h.hexdigest()
    return out


def parse_sha256sum(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        digest, name = line.split(maxsplit=1)
        out[name.lstrip("*").removeprefix("./")] = digest
    return out


def compare(local: dict[str, str], remote: dict[str, str]) -> dict:
    return {"missing_remote": sorted(set(local) - set(remote)), "extra_remote": sorted(set(remote) - set(local)),
            "mismatched": sorted(k for k in set(local) & set(remote) if local[k] != remote[k])}


def mirrored(log: Path) -> set[str]:
    if not log.exists():
        return set()
    return {row["task"] for row in map(json.loads, log.read_text().splitlines()) if row.get("match")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--bwlimit", type=int, default=8000)
    ap.add_argument("--period", type=float, default=120)
    a = ap.parse_args()
    host, base = a.dest.split(":", 1)
    ssh = ["ssh", "-i", a.key, "-p", str(a.port), "-o", "BatchMode=yes",
           "-o", f"UserKnownHostsFile={Path(a.key).parent / 'known_hosts'}"]
    nice = ["nice", "-n", "19", "ionice", "-c3"]
    os.nice(19)
    while not (RUNS / "MIRROR_STOP").exists():
        did = False
        for pdir in POLICY_DIRS:
            out = RUNS / pdir
            done_tasks = mirrored(out / "MIRROR.jsonl")
            for done in sorted(out.glob("*/DONE.json")):
                task_dir = done.parent
                if (RUNS / "MIRROR_STOP").exists() or task_dir.name in done_tasks or not repacked(task_dir):
                    continue
                fd = os.open(out / "_claims" / f"{task_dir.name}.lock", os.O_RDWR)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(fd)
                    continue
                try:
                    bw = int((RUNS / "MIRROR_BWLIMIT").read_text()) if (RUNS / "MIRROR_BWLIMIT").exists() else a.bwlimit
                    dest = f"{base}/{pdir}/{task_dir.name}"
                    t0 = time.time()
                    subprocess.run(ssh + [host, f"mkdir -p {shlex.quote(dest)}"])
                    rsync = subprocess.run(nice + ["rsync", "-a", "--partial", f"--bwlimit={bw}", f"--exclude=*{SKIP}",
                                                   f"--rsync-path={' '.join(nice)} rsync", "-e", " ".join(ssh),
                                                   f"{task_dir}/", f"{host}:{dest}/"])
                    copy_s = time.time() - t0
                    local = local_hashes(task_dir)
                    remote_cmd = (f"cd {shlex.quote(dest)} && find . -type f ! -name '*{SKIP}' -print0 | sort -z | "
                                  f"{' '.join(nice)} xargs -0 sha256sum")
                    remote = subprocess.run(ssh + [host, remote_cmd], capture_output=True, text=True)
                    diff = compare(local, parse_sha256sum(remote.stdout))
                    row = {"utc": utc(), "task": task_dir.name, "dest": f"{host}:{dest}", "files": len(local),
                           "bytes": sum(p.stat().st_size for p in task_dir.rglob("*") if p.is_file() and not p.name.endswith(SKIP)),
                           "bwlimit_kbps": bw, "rsync_exit": rsync.returncode, "copy_s": round(copy_s, 1),
                           "seconds": round(time.time() - t0, 1), **diff,
                           "match": rsync.returncode == 0 and remote.returncode == 0 and not any(diff.values())}
                finally:
                    os.close(fd)
                with (out / "MIRROR.jsonl").open("a") as log:
                    log.write(json.dumps(row) + "\n")
                print(f"{utc()} {pdir}/{task_dir.name} files={row['files']} bytes={row['bytes']} match={row['match']} "
                      f"{row['seconds']}s", flush=True)
                did = True
                if not row["match"]:
                    time.sleep(a.period)
        if not did:
            time.sleep(a.period)
    print(f"{utc()} stopped", flush=True)


if __name__ == "__main__":
    main()
