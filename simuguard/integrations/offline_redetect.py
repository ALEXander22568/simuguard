"""Re-run the SimuGuard detectors offline on a recorded episode segment (CPU only, no simulator).

usage::

    python -m simuguard.integrations.offline_redetect SEG_DIR [SEG_DIR ...] [--config CFG.json] [--check] [--json OUT]

A segment directory holds what :class:`SubstepMonitor` wrote online: ``manifest.json`` (body inventory,
timestep, the detector configs in force), ``states.npz`` (float64 state of every tracked body at every
substep) and ``trace.jsonl.gz`` (every substep: the contact pairs the monitor saw, as per-pair aggregates).
This rebuilds each substep's :class:`SubstepFrame` from them and feeds the detectors in the online order
(``context.previous`` = last frame, ``observe`` per substep, ``finalize`` on the last frame).

* default: the detectors are built from the configs stored in the manifest (the thresholds used online);
  ``--config`` instead takes a ``default_detectors`` config (keys ejection / penetration / impulse_spike /
  actuation_bound / non_finite / disabled), e.g. for threshold sensitivity.
* ``--check`` compares the offline event updates with the online ``events.jsonl``, in order: event id,
  detector, kind, status, onset/substep/control step, bodies and reasons must be equal; metric values must
  be equal up to the rounding of the trace (contact impulse 1e-9 N s, force 1e-6 N, penetration 1e-9 m).

Contact pairs are restored from their aggregates: the detectors only read per-pair point count, net impulse,
force estimates and maximum penetration, which the trace stores (rounded as above).  States come from the
full-precision ``states.npz``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

from simuguard.core.detectors.base import DetectorContext
from simuguard.core.types import BodyInfo, BodyKind, BodyRole, BodyState, SubstepFrame
from simuguard.core.statelog import StateLog
from simuguard.presets import default_detectors

# manifest detector name -> default_detectors config key
CONFIG_KEYS = {"contact_ejection": "ejection", "non_finite_state": "non_finite", "deep_penetration": "penetration",
               "impulse_spike": "impulse_spike", "actuation_bound": "actuation_bound"}
ALL_DETECTORS = tuple(CONFIG_KEYS)
# absolute tolerance per metric for contact-derived values (trace rounding); everything else must be equal
CONTACT_TOL = 2.0e-6


class StoredPair:
    """A contact pair restored from its trace aggregates (same interface the detectors use)."""

    __slots__ = ("body_a", "body_b", "point_count", "shape_pairs", "impulse_magnitude", "_force", "_force_abs",
                 "max_penetration", "impulse_abs_sum")

    def __init__(self, row: dict[str, Any], timestep: float) -> None:
        self.body_a = row["a"]
        self.body_b = row["b"]
        self.point_count = int(row["points"])
        self.shape_pairs = int(row.get("shape_pairs", 1))
        self.impulse_magnitude = float(row["impulse"])
        self._force = float(row["force_est"])
        self._force_abs = float(row["force_abs_est"])
        self.max_penetration = float(row["penetration"])
        self.impulse_abs_sum = self._force_abs * max(float(timestep), 1.0e-12)

    def force_estimate(self, timestep: float) -> float:
        return self._force

    def force_abs_estimate(self, timestep: float) -> float:
        return self._force_abs

    def involves(self, body_id: str) -> bool:
        return body_id in (self.body_a, self.body_b)

    def other(self, body_id: str) -> str:
        return self.body_b if self.body_a == body_id else self.body_a


def manifest_config(manifest: dict[str, Any]) -> dict[str, Any]:
    stored = manifest.get("detectors", {})
    cfg: dict[str, Any] = {CONFIG_KEYS[name]: dict(value) for name, value in stored.items() if name in CONFIG_KEYS}
    cfg["disabled"] = [name for name in ALL_DETECTORS if name not in stored]
    gate = cfg.get("ejection", {}).pop("gate", None)  # Detector.config() adds the gate's name
    if gate is not None:
        raise ValueError(f"online ejection gate {gate!r}: not reproducible offline")
    return cfg


def bodies_from_manifest(manifest: dict[str, Any]) -> dict[str, BodyInfo]:
    out = {}
    for body_id, d in manifest["bodies"].items():
        out[body_id] = BodyInfo(body_id=d["body_id"], name=d["name"], kind=BodyKind(d["kind"]), role=BodyRole(d["role"]),
                                mass=d.get("mass"), articulation=d.get("articulation"),
                                collision_shape_count=int(d.get("collision_shape_count", 0)),
                                metadata=d.get("metadata") or {})
    return out


def redetect(seg: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = json.loads((seg / "manifest.json").read_text())
    dt = float(manifest["timestep_s"])
    bodies = bodies_from_manifest(manifest)
    tracked = list(manifest["tracked_body_ids"])
    states = StateLog.load(seg / "states.npz")
    arr = states.array()
    row_of = {int(s): i for i, s in enumerate(states.substeps)}
    col_of = {b: states.body_ids.index(b) for b in tracked if b in states.body_ids}
    missing = sorted(set(tracked) - set(col_of))
    cfg = manifest_config(manifest) if config is None else config
    detectors = default_detectors(cfg)
    context = DetectorContext(episode_id=manifest["episode_id"], timestep=dt, bodies=bodies)
    for detector in detectors:
        detector.reset(context)
    updates: list[dict[str, Any]] = []
    last: SubstepFrame | None = None
    frames = 0
    with gzip.open(seg / "trace.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            t = json.loads(line)
            substep = int(t["substep"])
            r = arr[row_of[substep]]
            # which bodies the online frame had: the trace lists every non-robot body of the frame (states.npz
            # writes NaN for a body the adapter did not return, e.g. a static body without a rigid-body view)
            present = t.get("states", {})
            frame_states = {b: BodyState(b, r[c, 0:3].copy(), r[c, 3:7].copy(), r[c, 7:10].copy(), r[c, 10:13].copy())
                            for b, c in col_of.items()
                            if (b in present if bodies[b].role != BodyRole.ROBOT else not np.isnan(r[c]).all())}
            frame = SubstepFrame(substep=substep, sim_time=float(t["t"]), control_step=int(t["control_step"]),
                                 timestep=dt, states=frame_states,
                                 contacts=[StoredPair(p, dt) for p in t.get("contacts", [])],  # type: ignore[misc]
                                 extras=t.get("extras", {}))
            context.previous = last
            for detector in detectors:
                updates.extend(e.to_dict() for e in detector.observe(frame, context))
            last = frame
            frames += 1
    for detector in detectors:
        updates.extend(e.to_dict() for e in detector.finalize(last, context))
    return {"segment": str(seg), "frames": frames, "state_rows": len(states), "missing_state_bodies": missing,
            "config": cfg, "events": updates}


def _diff_values(a: Any, b: Any, path: str, out: list[dict[str, Any]], worst: dict[str, float]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append({"path": f"{path}.{k}", "online": a.get(k, "<missing>"), "offline": b.get(k, "<missing>")})
            else:
                _diff_values(a[k], b[k], f"{path}.{k}", out, worst)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            _diff_values(x, y, f"{path}[{i}]", out, worst)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        if a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)):
            return
        err = abs(float(a) - float(b))
        key = path.split(".")[-1]
        worst[key] = max(worst.get(key, 0.0), err)
        if not err <= CONTACT_TOL:
            out.append({"path": path, "online": a, "offline": b, "abs_diff": err})
    elif a != b:
        out.append({"path": path, "online": a, "offline": b})


def check(seg: Path, result: dict[str, Any]) -> dict[str, Any]:
    # updates in substep order; within one substep the order of bodies follows the inventory order, which
    # the manifest stores sorted, so both lists are sorted by (substep, detector, event id) keeping update order
    def ordered(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [e for _, e in sorted(enumerate(events), key=lambda ie: (ie[1].get("substep", -1), ie[1].get("detector", ""),
                                                                          ie[1].get("event_id", ""), ie[0]))]

    online = ordered([json.loads(l) for l in (seg / "events.jsonl").read_text().splitlines() if l.strip()])
    offline = ordered(result["events"])
    mismatches: list[dict[str, Any]] = []
    worst: dict[str, float] = {}
    exact_fields = ("event_id", "detector", "kind", "status", "onset_substep", "substep", "control_step", "bodies",
                    "reasons", "review")
    for i in range(max(len(online), len(offline))):
        if i >= len(online) or i >= len(offline):
            mismatches.append({"index": i, "only": "offline" if i >= len(online) else "online",
                               "event": (offline if i >= len(online) else online)[i]})
            continue
        a, b = online[i], offline[i]
        for f in exact_fields:
            if a.get(f) != b.get(f):
                mismatches.append({"index": i, "field": f, "online": a.get(f), "offline": b.get(f)})
        diffs: list[dict[str, Any]] = []
        _diff_values(a.get("metrics", {}), b.get("metrics", {}), "metrics", diffs, worst)
        for d in diffs:
            mismatches.append({"index": i, "event_id": a.get("event_id"), **d})
    return {"online_updates": len(online), "offline_updates": len(offline), "match": not mismatches,
            "mismatches": mismatches[:50], "mismatch_count": len(mismatches),
            "max_metric_abs_diff": worst}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("segments", nargs="+", type=Path)
    ap.add_argument("--config", default=None, help="default_detectors config JSON (default: the manifest's)")
    ap.add_argument("--check", action="store_true", help="compare with the online events.jsonl")
    ap.add_argument("--json", default=None, help="write per-segment results here")
    args = ap.parse_args(argv)
    config = json.loads(Path(args.config).read_text()) if args.config else None
    results = []
    ok = True
    for seg in args.segments:
        if not (seg / "trace.jsonl.gz").exists() or not (seg / "states.npz").exists():
            results.append({"segment": str(seg), "skipped": "no trace.jsonl.gz / states.npz"})
            print(f"{seg}: skipped (no trace/states)")
            continue
        res = redetect(seg, config)
        line = f"{seg}: {res['frames']} frames, {len(res['events'])} event updates"
        if args.check:
            res["check"] = check(seg, res)
            c = res["check"]
            ok &= c["match"]
            line += (f"; online {c['online_updates']} -> {'MATCH' if c['match'] else 'MISMATCH x' + str(c['mismatch_count'])}"
                     f", max metric diff {max(c['max_metric_abs_diff'].values(), default=0.0):.2e}")
        print(line, flush=True)
        results.append(res)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1, default=str))
    if args.check:
        checked = [r for r in results if "check" in r]
        print(f"checked {len(checked)} segments: {sum(r['check']['match'] for r in checked)} match, "
              f"{sum(not r['check']['match'] for r in checked)} mismatch, {len(results) - len(checked)} skipped")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
