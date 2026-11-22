"""stick_series.py <spec.json> <out.json>: per-substep series around a flagged sticking window, from the records only.
spec: [{"name", "seg_dir", "body", "onset", "lo_s", "hi_s"}]. Output per case: t (s from onset), gripper command (rad),
finger joint (rad), object position (m), min separation to any finger link (m, NaN = no finger contact),
number of contact points with non-robot bodies."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stick_detect import finger_joint
res = {}
for c in json.loads(Path(sys.argv[1]).read_text()):
    seg = Path(c["seg_dir"])
    st = np.load(seg / "states.npz", allow_pickle=False)
    ids = json.loads(str(st["body_ids"])); S = st["substeps"].astype(np.int64); X = st["states"]; s0 = int(S[0])
    on = int(c["onset"]) - s0; lo = max(0, on + int(c["lo_s"] * 120)); hi = min(len(S), on + int(c["hi_s"] * 120))
    co = np.load(seg / "controls.npz", allow_pickle=False); V = co["values"].reshape(-1, 39); cs = co["substeps"]
    g = V[np.clip(np.searchsorted(cs, S, side="right") - 1, 0, len(cs) - 1), 7]
    fj = finger_joint(seg, S)
    ct = np.load(seg / "contacts.npz", allow_pickle=False); names = json.loads(str(ct["names"])); rows = ct["rows"]
    bi = names.index(c["body"]); a = rows[:, 1].astype(int); b = rows[:, 2].astype(int)
    m = (a == bi) | (b == bi); oth = np.where(a[m] == bi, b[m], a[m]); ss = rows[m, 0].astype(np.int64) - s0; sp = rows[m, 12]
    fing = np.array([names[o].startswith("robot/") and "finger" in names[o] for o in oth])
    rob = np.array([names[o].startswith("robot/") for o in oth])
    msep = np.full(len(S), np.nan); nsup = np.zeros(len(S), int)
    for s_, p_ in zip(ss[fing], sp[fing]):
        if 0 <= s_ < len(S): msep[s_] = p_ if np.isnan(msep[s_]) else min(msep[s_], p_)
    np.add.at(nsup, ss[~rob][(ss[~rob] >= 0) & (ss[~rob] < len(S))], 1)
    col = ids.index(c["body"]); r = slice(lo, hi)
    res[c["name"]] = {"t": ((np.arange(lo, hi) - on) / 120).round(4).tolist(), "cmd": g[r].astype(float).round(4).tolist(),
                      "fj": None if fj is None else fj[r].round(4).tolist(), "pos": X[r, col, :3].astype(float).round(5).tolist(),
                      "sep": [None if np.isnan(x) else round(float(x), 5) for x in msep[r]], "nsup": nsup[r].tolist()}
Path(sys.argv[2]).write_text(json.dumps(res))
print({k: len(v["t"]) for k, v in res.items()})
