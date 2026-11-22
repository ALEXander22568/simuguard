"""Offline "object stuck to an open gripper" detector for recorded RoboLab segments.

Rule (frozen, see RULE.md): a scene body is STUCK at a physics substep s when
  (1) the gripper command (position target of finger_joint, controls.npz values[:, 7]) is "open" (< 0.1 rad)
      and has been open continuously for at least GRACE_S seconds,
  (2) the body has at least one contact point with a robot finger link with separation <= TOUCH_SEP_M,
  (3) the body has no reported contact point (any separation) with any non-robot body.
Stuck substeps are merged into windows; a gap of <= GAP_S between two stuck runs is bridged if every gap
substep still satisfies (1) and (3).  The episode is flagged if some window lasts >= MIN_DUR_S and in it the
body moves >= MIN_DISP_M from where it was at the window start.

usage (RoboLab venv, PYTHONPATH not needed):
  python stick_detect.py <segs.json> <out.jsonl>
segs.json: [{"key": [...], "seg_dir": str}, ...] (same lists as the baseline/sensitivity reruns)
Every window that lasts >= REPORT_MIN_S is written out (with its features) so a human can look; only the frozen
thresholds decide "flag".
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

RULE = dict(OPEN_CMD_RAD=0.1, GRACE_S=0.5, TOUCH_SEP_M=0.001, GAP_S=0.1, MIN_DUR_S=2.0, MIN_DISP_M=0.10)
REPORT_MIN_S = 0.25
SKIP_ROLES = {"scene"}


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def finger_joint(seg: Path, S: np.ndarray):
    try:
        task_dir = seg.parents[2]
        k = int(seg.parents[1].name.split("_")[1]); e = int(seg.parent.name.split("_")[1])
        run = json.loads((task_dir / "batch" / f"run_{k}.json").read_text())
        sg = [s for s in run["segments"] if s["env"] == e and str(Path(s["dir"]).name) == seg.name]
        sg = sg[-1] if sg else [s for s in run["segments"] if s["env"] == e and not s.get("discarded")][-1]
        rob = np.load(task_dir / "batch" / f"run_{k}_robot.npz", allow_pickle=False)
        idx = S + int(sg["start_substep"]) - 1 - int(rob["first_substep"])
        dp = rob["dof_pos"]
        if idx.min() < 0 or idx.max() >= dp.shape[0]:
            return None
        return dp[idx, e, 7].astype(np.float64)
    except Exception:  # noqa: BLE001
        return None


def run_segment(seg: Path, rule: dict = RULE) -> dict:
    man = json.loads((seg / "manifest.json").read_text())
    dt = float(man["timestep_s"])
    st = np.load(seg / "states.npz", allow_pickle=False)
    ids = json.loads(str(st["body_ids"])); S = st["substeps"].astype(np.int64); X = st["states"]
    T = len(S); s0 = int(S[0])
    contiguous = bool(T == 0 or S[-1] - S[0] + 1 == T)
    co = np.load(seg / "controls.npz", allow_pickle=False)
    w = co["widths"]
    if not np.all(w == 39):
        raise ValueError(f"unexpected control widths {np.unique(w)}")
    V = co["values"].reshape(-1, 39); cs = co["substeps"]
    gi = np.clip(np.searchsorted(cs, S, side="right") - 1, 0, len(cs) - 1)
    gcmd = V[gi, 7].astype(np.float64)
    opn = gcmd < rule["OPEN_CMD_RAD"]
    held = np.zeros(T, np.int64)  # substeps the open command has been held, including this one
    c = 0
    for i in range(T):
        c = c + 1 if opn[i] else 0
        held[i] = c
    grace = held * dt >= rule["GRACE_S"] - 1e-9
    fj = finger_joint(seg, S)

    ct = np.load(seg / "contacts.npz", allow_pickle=False)
    names = json.loads(str(ct["names"]))
    rows = ct["rows"]
    sub = rows[:, 0].astype(np.int64); a = rows[:, 1].astype(np.int32); b = rows[:, 2].astype(np.int32); sep = rows[:, 12].copy()
    del rows
    robot = np.array([n.startswith("robot/") for n in names])
    finger = np.array([n.startswith("robot/") and "finger" in n for n in names])
    gap = int(round(rule["GAP_S"] / dt))
    windows = []
    for body in ids:
        bd = man["bodies"].get(body, {})
        if bd.get("role") in SKIP_ROLES or bd.get("kind") != "dynamic" or body not in names:
            continue
        bi = names.index(body)
        m = (a == bi) | (b == bi)
        if not m.any():
            continue
        oth = np.where(a[m] == bi, b[m], a[m]); ss = sub[m] - s0; sp = sep[m]
        ok = (ss >= 0) & (ss < T)
        oth, ss, sp = oth[ok], ss[ok], sp[ok]
        fsel = finger[oth] & (sp <= rule["TOUCH_SEP_M"])
        if not fsel.any():
            continue
        fing = np.zeros(T, bool); fing[ss[fsel]] = True
        supp = np.zeros(T, bool); supp[ss[~robot[oth]]] = True
        C = grace & fing & ~supp
        U = grace & ~supp
        rr = runs(C)
        merged = []
        for u, v in rr:
            if merged and u - merged[-1][1] <= gap and U[merged[-1][1]:u].all():
                merged[-1][1] = v
            else:
                merged.append([u, v])
        col = ids.index(body)
        for u, v in merged:
            dur = (v - u) * dt
            if dur < REPORT_MIN_S:
                continue
            p = X[u:v, col, :3].astype(np.float64)
            disp = float(np.linalg.norm(p - p[0], axis=1).max())
            fs = fsel & (ss >= u) & (ss < v)
            links = sorted({names[int(o)] for o in np.unique(oth[fs])})
            j = u - int(held[u]) + 1
            win = {"body": body, "role": bd.get("role"), "start": int(S[u]), "end": int(S[v - 1]), "dur_s": round(dur, 3),
                   "disp_m": round(disp, 4), "path_m": round(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()), 4),
                   "z0": round(float(p[0, 2]), 4), "z_max": round(float(p[:, 2].max()), 4),
                   "open_since": int(S[j]), "open_held_at_start_s": round(float(held[u] * dt), 3),
                   "links": links, "min_sep_m": round(float(sp[fs].min()), 5),
                   "fj_min": None if fj is None else round(float(fj[u:v].min()), 4),
                   "fj_max": None if fj is None else round(float(fj[u:v].max()), 4)}
            win["flag"] = bool(dur >= rule["MIN_DUR_S"] - 1e-9 and disp >= rule["MIN_DISP_M"])
            windows.append(win)
    windows.sort(key=lambda x: -x["dur_s"])
    flagged = [x for x in windows if x["flag"]]
    return {"flag": bool(flagged), "flag_bodies": sorted({x["body"] for x in flagged}),
            "onset": min((x["start"] for x in flagged), default=None), "windows": windows[:40], "n_windows": len(windows),
            "n_substeps": T, "contiguous": contiguous, "has_fj": fj is not None, "open_frac": round(float(opn.mean()), 4)}


def main() -> int:
    segs = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                done.add(json.loads(line)["seg_dir"])
            except Exception:  # noqa: BLE001
                pass
    with open(out, "a") as f:
        for it in segs:
            if it["seg_dir"] in done:
                continue
            t0 = time.time()
            try:
                r = run_segment(Path(it["seg_dir"]))
                r["error"] = None
            except Exception as e:  # noqa: BLE001
                r = {"flag": None, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-800:]}
            r.update(key=it.get("key"), seg_dir=it["seg_dir"], wall_s=round(time.time() - t0, 2))
            f.write(json.dumps(r) + "\n"); f.flush()
            print(it.get("key"), r.get("flag"), r.get("error"), r["wall_s"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
