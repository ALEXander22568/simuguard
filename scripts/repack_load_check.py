"""Load every .npz of a task through the readers SimuGuard uses (no code change): StateLog, ControlLog,
WriteLog, and np.load for contacts / initial states / robot logs.  Prints one JSON line.

usage: repack_load_check.py <task_dir>"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

from simuguard.adapters.isaac.robolab_io import WriteLog
from simuguard.core.controls import ControlLog
from simuguard.core.statelog import StateLog

task_dir = Path(sys.argv[1])
counts, errors, not_lzma = {}, [], []
for path in sorted(task_dir.rglob("*.npz")):
    kind = path.name if path.name in ("states.npz", "controls.npz", "contacts.npz") else path.name.split("_")[-1]
    try:
        if not path.name.endswith("_robot.npz") and any(i.compress_type != zipfile.ZIP_LZMA for i in zipfile.ZipFile(path).infolist()):
            not_lzma.append(str(path))
        if path.name == "states.npz":
            n = len(StateLog.load(path).substeps)
        elif path.name == "controls.npz":
            n = len(ControlLog.load(path)._substeps)
        elif path.name.endswith("_writes.npz"):
            n = len(WriteLog.load(path)[0])
        else:
            with np.load(path, allow_pickle=False) as d:
                n = sum(d[k].size for k in d.files)
        counts.setdefault(kind, []).append(n)
    except Exception as exc:
        errors.append(f"{path}: {exc!r}")
print(json.dumps({"task_dir": str(task_dir), "files": {k: len(v) for k, v in counts.items()},
                  "items": {k: int(sum(v)) for k, v in counts.items()}, "not_lzma": not_lzma, "errors": errors,
                  "ok": bool(counts) and not errors and not not_lzma}))
