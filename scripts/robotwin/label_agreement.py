#!/usr/bin/env python3
"""Compare human labels of a blind clip set with what the detector said.

Input: the label set's ``key.json`` (which clip came from which moment) and a
JSON/CSV of human answers keyed by clip id, with ``q1_nonphysical_motion`` and
``q2_policy_on_track`` in {yes, no, unsure}.

Reported:

* precision of confirmed events - how many detector events a human also calls
  physically implausible, with a Wilson 95% interval;
* the hard negatives (unconfirmed flags, and the fastest target motion in
  episodes with no event) a human nevertheless calls implausible: evidence of
  missed anomalies, bounded by how the negatives were sampled;
* whether the policy was on track before the moment (the human's answer to the
  question the detectors cannot answer);
* for confirmed events, whether the target actually left the container, using
  the benchmark's own distance criterion - a violent bounce that stays inside
  is still an anomaly but need not change the outcome.

Usage::

    python scripts/robotwin/label_agreement.py --label-set runs/label_set_v1 \\
        --labels runs/label_set_v1/human_labels_v1.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from simuguard.core import StateLog  # noqa: E402

POSITIVE = "confirmed_event"


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if not total:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_labels(path: Path) -> dict[str, dict]:
    if path.suffix == ".csv":
        with path.open() as handle:
            return {row["clip_id"]: row for row in csv.DictReader(handle)}
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return {row["clip_id"]: row for row in data}
    return data


def containment(item: dict, threshold: float) -> dict:
    """Did the target leave the container, by the benchmark's own distance test?"""

    log = StateLog.load(Path(item["segment"]) / "states.npz")
    ids = log.body_ids
    target = next(i for i, b in enumerate(ids) if "can" in b.lower())
    container = next(i for i, b in enumerate(ids) if "basket" in b.lower())
    array, substeps, onset = log.array(), log.substeps, item["onset_substep"]
    speed = np.linalg.norm(array[:, target, 7:10], axis=1)
    window = (substeps >= onset - 50) & (substeps <= onset + 750)
    distance = np.abs(array[:, target, 0:3] - array[:, container, 0:3]).sum(axis=1)
    after = (substeps >= onset) & (substeps <= onset + 2000)
    return {
        "peak_speed_mps": float(np.nanmax(speed[window])) if window.any() else float("nan"),
        "left_container": bool(np.nanmax(distance[after]) > threshold) if after.any() else None,
        "outside_at_episode_end": bool(distance[-1] > threshold),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-set", required=True, help="directory holding key.json")
    parser.add_argument("--labels", required=True, help="human answers (json or csv)")
    parser.add_argument("--containment-threshold-m", type=float, default=0.15,
                        help="RoboTwin place_can_basket counts success below this |dx|+|dy|+|dz|")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.label_set).resolve()
    key = {item["clip_id"]: item for item in json.loads((root / "key.json").read_text())}
    labels = load_labels(Path(args.labels).resolve())
    missing = [c for c in key if c not in labels]

    def answer(clip: str, field: str) -> str:
        return (labels.get(clip, {}) or {}).get(field, "") or ""

    report: dict = {
        "label_set": str(root),
        "clips": len(key),
        "labelled": len(key) - len(missing),
        "unlabelled": missing,
        "annotators": 1,
        "by_kind": {},
    }

    for kind in sorted({item["kind"] for item in key.values()}):
        clips = [c for c, item in key.items() if item["kind"] == kind]
        counts = Counter(answer(c, "q1_nonphysical_motion") for c in clips)
        report["by_kind"][kind] = {
            "clips": len(clips),
            "q1": dict(counts),
            "q2_policy_on_track": dict(Counter(answer(c, "q2_policy_on_track") for c in clips)),
        }

    positives = [c for c, item in key.items() if item["kind"] == POSITIVE]
    negatives = [c for c in key if c not in positives]
    agreed = sum(1 for c in positives if answer(c, "q1_nonphysical_motion") == "yes")
    disputed = [c for c in positives if answer(c, "q1_nonphysical_motion") == "no"]
    missed = [c for c in negatives if answer(c, "q1_nonphysical_motion") == "yes"]
    low, high = wilson(agreed, len(positives))
    report["detector_precision"] = {
        "confirmed_events": len(positives),
        "human_agrees": agreed,
        "precision": round(agreed / len(positives), 4) if positives else None,
        "wilson_ci95": [round(low, 4), round(high, 4)],
        "disputed_clips": [
            {"clip": c, **{k: key[c][k] for k in ("config", "repeat", "phase", "seed", "onset_substep", "detector")}}
            for c in disputed
        ],
        "negatives_called_anomalous": [
            {"clip": c, **{k: key[c][k] for k in ("kind", "config", "phase", "seed")}} for c in missed
        ],
        "negatives": len(negatives),
    }

    on_track = Counter(answer(c, "q2_policy_on_track") for c in positives)
    report["policy_on_track_at_confirmed_events"] = dict(on_track)

    rows = []
    for clip in sorted(positives):
        measures = containment(key[clip], args.containment_threshold_m)
        rows.append({"clip": clip, "config": key[clip]["config"], "phase": key[clip]["phase"],
                     "human_q1": answer(clip, "q1_nonphysical_motion"), **measures})
    report["confirmed_event_outcomes"] = {
        "left_container_within_2000_substeps": sum(1 for r in rows if r["left_container"]),
        "outside_container_at_episode_end": sum(1 for r in rows if r["outside_at_episode_end"]),
        "by_peak_speed": {
            f"{lo}-{hi}": {
                "clips": len(band),
                "human_yes": sum(1 for r in band if r["human_q1"] == "yes"),
                "left_container": sum(1 for r in band if r["left_container"]),
            }
            for lo, hi in ((0, 3), (3, 6), (6, 12), (12, 1000))
            if (band := [r for r in rows if lo <= r["peak_speed_mps"] < hi])
        },
        "rows": rows,
    }

    text = json.dumps(report, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(text)
    print(json.dumps({k: v for k, v in report.items() if k != "confirmed_event_outcomes"}, indent=1, default=str))
    print(json.dumps(
        {k: v for k, v in report["confirmed_event_outcomes"].items() if k != "rows"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
