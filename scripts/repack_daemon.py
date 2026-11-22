"""Re-pack the .npz files of finished formal tasks with LZMA entries (lossless; np.load reads them unchanged).

usage: repack_daemon.py [--workers N] [--hold-after-first]
       repack_daemon.py --to-deflate FILE...   (put files back to deflate entries, same verify and log)

Every --period seconds: for each task of runs/formal_pi05 and runs/formal_gr00t that has DONE.json and still
holds a non-LZMA .npz, take the task's claim lock (skip the task if a lane holds it) and re-pack its files in a
pool of single-core workers.  Per file: sha256 of the original, write ``.<name>.repack-tmp`` in the same
directory, compare every array (key set, dtype, shape, bytes), then os.replace.  One row per file in
runs/<policy dir>/REPACK.jsonl.  A failed verify keeps the original and stops the daemon (REPACK_STOP).
runs/REPACK_WORKERS overrides --workers at the next task; runs/REPACK_STOP stops after the current task.
With --hold-after-first the daemon waits for runs/REPACK_GO after the first task.
run_<k>_robot.npz stays deflate: replay indexes it once per substep (whole-array read each time), LZMA makes
that ~6x slower and saves only ~11% on those files."""
import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import time
import zipfile
from pathlib import Path

import numpy as np

RUNS = Path(os.environ.get("REPACK_RUNS", "/path/to/simuguard-robolab/runs"))
POLICY_DIRS = ("formal_pi05", "formal_gr00t")
KEEP_DEFLATE = "_robot.npz"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def needs_repack(path: Path) -> bool:
    with zipfile.ZipFile(path) as z:
        return any(info.compress_type != zipfile.ZIP_LZMA for info in z.infolist())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def same_arrays(a_path: Path, b_path: Path) -> str:
    with np.load(a_path, allow_pickle=False) as a, np.load(b_path, allow_pickle=False) as b:
        if sorted(a.files) != sorted(b.files):
            return f"keys differ: {sorted(a.files)} vs {sorted(b.files)}"
        for k in a.files:
            x, y = a[k], b[k]
            if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                return f"array {k} differs"
    return "identical"


def repack(path_str: str, compression: int = zipfile.ZIP_LZMA) -> dict:
    os.nice(19)
    path = Path(path_str)
    tmp = path.with_name(f".{path.name}.repack-tmp")
    t0 = time.time()
    row = {"utc": utc(), "file": str(path.relative_to(RUNS)), "to": "lzma" if compression == zipfile.ZIP_LZMA else "deflate", "old_bytes": path.stat().st_size, "sha256_orig": sha256(path)}
    try:
        with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", compression=compression) as zout:
            for info in zin.infolist():
                with zin.open(info) as fin, zout.open(info.filename, "w", force_zip64=True) as fout:
                    while chunk := fin.read(1 << 24):
                        fout.write(chunk)
        row["verify"] = same_arrays(path, tmp)
    except Exception as exc:
        row["verify"] = f"error: {exc!r}"
    if row["verify"] == "identical":
        row["new_bytes"] = tmp.stat().st_size
        os.replace(tmp, path)
    else:
        row["new_bytes"] = None
        tmp.unlink(missing_ok=True)
    row["seconds"] = round(time.time() - t0, 1)
    return row


def candidates():
    for pdir in POLICY_DIRS:
        out = RUNS / pdir
        for done in sorted(out.glob("*/DONE.json")):
            task_dir = done.parent
            for tmp in task_dir.rglob(".*.repack-tmp"):
                tmp.unlink()
            files = [p for p in sorted(task_dir.rglob("*.npz")) if not p.name.endswith(KEEP_DEFLATE) and needs_repack(p)]
            if files:
                yield out, task_dir, files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--period", type=float, default=60)
    ap.add_argument("--hold-after-first", action="store_true")
    ap.add_argument("--to-deflate", nargs="+", type=Path)
    a = ap.parse_args()
    if a.to_deflate:
        for path in a.to_deflate:
            row = repack(str(path.resolve()), zipfile.ZIP_DEFLATED)
            with (RUNS / row["file"].split("/")[0] / "REPACK.jsonl").open("a") as log:
                log.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        return
    first_done = False
    while not (RUNS / "REPACK_STOP").exists():
        did = False
        for out, task_dir, files in candidates():
            if (RUNS / "REPACK_STOP").exists():
                break
            fd = os.open(out / "_claims" / f"{task_dir.name}.lock", os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue
            workers = int((RUNS / "REPACK_WORKERS").read_text()) if (RUNS / "REPACK_WORKERS").exists() else a.workers
            print(f"{utc()} {out.name}/{task_dir.name} {len(files)} files, {workers} workers", flush=True)
            try:
                with mp.get_context("spawn").Pool(workers) as pool, (out / "REPACK.jsonl").open("a") as log:
                    for row in pool.imap_unordered(repack, [str(f) for f in files]):
                        log.write(json.dumps(row) + "\n")
                        log.flush()
                        if row["verify"] != "identical":
                            (RUNS / "REPACK_STOP").write_text(json.dumps(row))
            finally:
                os.close(fd)
            print(f"{utc()} {out.name}/{task_dir.name} done", flush=True)
            did = True
            if a.hold_after_first and not first_done:
                first_done = True
                (RUNS / "REPACK_FIRST").write_text(json.dumps({"utc": utc(), "task": f"{out.name}/{task_dir.name}"}))
                while not (RUNS / "REPACK_GO").exists() and not (RUNS / "REPACK_STOP").exists():
                    time.sleep(20)
        if not did:
            time.sleep(a.period)
    print(f"{utc()} stopped: {(RUNS / 'REPACK_STOP').read_text()}", flush=True)


if __name__ == "__main__":
    main()
