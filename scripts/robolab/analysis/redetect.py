"""Offline re-run of the contact_ejection detector on recorded RoboLab segments (states.npz + contacts.npz).

Same detector class as the live run, fed with the recorded per-substep object states (float64) and
per-point contacts (float32).  Used for the threshold sensitivity study: one pass over the frames,
several detector instances with different configs.

usage (any python with numpy, PYTHONPATH=<simuguard>):
  python redetect.py <segs.json> <out.jsonl> [--configs configs.json] [--check]
segs.json: [{"key": [...], "seg_dir": str}, ...]
configs.json: {"name": {field: value, ...}, ...}  (fields override the recorded config); "base" is always added.
--check: also compare the base config's events with the recorded events.jsonl and report mismatches.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from simuguard.adapters.isaac.robolab_io import ArrayContactPair
from simuguard.core.detectors.base import DetectorContext
from simuguard.core.detectors.ejection import ContactEjectionDetector, EjectionConfig
from simuguard.core.types import BodyInfo, BodyKind, BodyRole, BodyState, SubstepFrame


class SharedEvidenceDetector(ContactEjectionDetector):
    """Same detector; the contact evidence of a body in a frame does not depend on the thresholds, so it is
    computed once per (frame, body) and shared by all configs."""

    cache: dict = {}

    def _evidence(self, frame, body_id, contacts, context):  # noqa: ANN001
        key = (frame.substep, body_id)
        c = SharedEvidenceDetector.cache
        if key not in c:
            c[key] = ContactEjectionDetector._evidence(self, frame, body_id, contacts, context)
        return c[key]


class FastState(BodyState):
    """BodyState whose finiteness is computed once per frame instead of once per config."""

    def is_finite(self) -> bool:  # noqa: D401
        return self._fin


def make_state(b, v):  # noqa: ANN001
    x = FastState(b, v[0:3], v[3:7], v[7:10], v[10:13])
    object.__setattr__(x, "_fin", bool(np.isfinite(v).all()))
    return x


def load_bodies(manifest: dict) -> dict[str, BodyInfo]:
    out = {}
    for b, d in manifest["bodies"].items():
        out[b] = BodyInfo(body_id=d["body_id"], name=d["name"], kind=BodyKind(d["kind"]), role=BodyRole(d["role"]),
                          mass=d.get("mass"), articulation=d.get("articulation"),
                          collision_shape_count=d.get("collision_shape_count", 0), metadata=d.get("metadata") or {})
    return out


def run_segment(seg_dir: Path, configs: dict[str, dict]) -> dict:
    manifest = json.loads((seg_dir / "manifest.json").read_text())
    dt = float(manifest["timestep_s"])
    bodies = load_bodies(manifest)
    rec_cfg = manifest["detectors"]["contact_ejection"]
    rec_cfg = {k: v for k, v in rec_cfg.items() if k != "gate"}
    dets = {}
    for name, over in configs.items():
        cfg = EjectionConfig.from_dict({**rec_cfg, **over})
        dets[name] = SharedEvidenceDetector(cfg)
    ctx = DetectorContext(episode_id=manifest["episode_id"], timestep=dt, bodies=bodies)
    for d in dets.values():
        d.reset(ctx)
    free = ctx.free_bodies(tuple(BodyRole(r) for r in rec_cfg["monitored_roles"]))

    st = np.load(seg_dir / "states.npz", allow_pickle=False)
    ids = json.loads(str(st["body_ids"]))
    col = {b: i for i, b in enumerate(ids)}
    use = [b for b in free if b in col]
    missing = [b for b in free if b not in col]
    substeps, states = st["substeps"], st["states"]
    ct = np.load(seg_dir / "contacts.npz", allow_pickle=False)
    names = json.loads(str(ct["names"]))
    rows = ct["rows"]
    cs = rows[:, 0].astype(np.int64)
    if len(cs) > 1 and np.any(np.diff(cs) < 0):
        order = np.argsort(cs, kind="stable")
        rows, cs = rows[order], cs[order]
    a_ids = rows[:, 1].astype(np.int64)
    b_ids = rows[:, 2].astype(np.int64)
    lo = np.searchsorted(cs, substeps, side="left")
    hi = np.searchsorted(cs, substeps, side="right")
    keep_names = set(use)

    events = {name: {} for name in dets}
    zero = np.zeros(3)
    for i, s in enumerate(substeps):
        s = int(s)
        x = states[i]
        fs = {}
        for b in use:
            v = x[col[b]]
            fs[b] = make_state(b, v)
        pairs = []
        if hi[i] > lo[i]:
            j0, j1 = int(lo[i]), int(hi[i])
            aa, bb = a_ids[j0:j1], b_ids[j0:j1]
            brk = np.flatnonzero((aa[1:] != aa[:-1]) | (bb[1:] != bb[:-1])) + 1
            starts = np.concatenate([[0], brk]); ends = np.concatenate([brk, [j1 - j0]])
            for p, q in zip(starts, ends):
                na, nb = names[int(aa[p])], names[int(bb[p])]
                if na not in keep_names and nb not in keep_names:
                    continue
                r = rows[j0 + p:j0 + q]
                pairs.append(ArrayContactPair(na, nb, r[:, 3:6].astype(np.float64), r[:, 6:9].astype(np.float64),
                                              r[:, 9:12].astype(np.float64), r[:, 12].astype(np.float64)))
        SharedEvidenceDetector.cache.clear()
        frame = SubstepFrame(substep=s, sim_time=s * dt, control_step=(s - 1) // 8 + 1, timestep=dt, states=fs, contacts=pairs)
        for name, d in dets.items():
            for e in d.observe(frame, ctx):
                events[name][e.event_id] = e
    for name, d in dets.items():
        for e in d.finalize(None, ctx):
            events[name][e.event_id] = e
    out = {}
    for name, evs in events.items():
        out[name] = [{"body": e.event_id.rsplit(":", 2)[-2], "onset": e.onset_substep, "status": e.status.value,
                      "reasons": list(e.reasons), "bodies": list(e.bodies),
                      "m": {k: e.metrics.get(k) for k in ("onset_speed_mps", "onset_delta_v_mps", "max_speed_mps", "max_delta_v_mps",
                                                          "max_distance_m", "max_free_flight_substeps", "risky_contact")}}
                     for e in evs.values() if e.status.value in ("confirmed",)]
    return {"events": out, "missing_free_bodies": missing, "n_substeps": int(len(substeps)),
            "contiguous": bool(len(substeps) == 0 or (substeps[-1] - substeps[0] + 1 == len(substeps)))}


def recorded(seg_dir: Path) -> list[tuple[str, int]]:
    out = []
    for line in open(seg_dir / "events.jsonl"):
        e = json.loads(line)
        if e["detector"] == "contact_ejection" and e["status"] == "confirmed":
            out.append((e["event_id"].rsplit(":", 2)[-2], int(e["onset_substep"])))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("segs")
    ap.add_argument("out")
    ap.add_argument("--configs")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    segs = json.loads(Path(a.segs).read_text())
    configs = {"base": {}}
    if a.configs:
        configs.update(json.loads(Path(a.configs).read_text()))
    done = set()
    outp = Path(a.out)
    if outp.exists():
        for line in open(outp):
            try:
                done.add(json.loads(line)["seg_dir"])
            except Exception:  # noqa: BLE001
                pass
    with open(outp, "a") as fh:
        for item in segs:
            if item["seg_dir"] in done:
                continue
            t0 = time.time()
            seg = Path(item["seg_dir"])
            try:
                res = run_segment(seg, configs)
                if a.check:
                    got = sorted((e["body"], e["onset"]) for e in res["events"]["base"])
                    want = recorded(seg)
                    res["check"] = {"equal": got == want, "got": got, "want": want}
                res["error"] = None
            except Exception as exc:  # noqa: BLE001
                res = {"error": repr(exc)}
            res.update(key=item.get("key"), seg_dir=item["seg_dir"], wall_s=round(time.time() - t0, 2))
            fh.write(json.dumps(res) + "\n")
            fh.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
