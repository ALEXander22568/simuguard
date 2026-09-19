#!/usr/bin/env python3
"""Build a blind human-labelling set of video clips from recorded episodes.

Clips are rendered by replaying the recorded actuation (no video is needed at
evaluation time) around two kinds of moments:

* positives - confirmed anomaly events (detector onset);
* hard negatives - the fastest target motion in episodes with *no* confirmed
  event (e.g. the can dropping into the basket), and unconfirmed detector flags.

Clip ids are shuffled and carry no hint of which kind they are.  ``labels.csv``
is what the annotator fills in; ``key.json`` (kept away from the annotator)
maps each clip back to its source so precision/recall can be computed.

Clips are rendered around the onset at ``--every`` substeps per frame (default
5 substeps = 0.02 s -> half speed at 25 fps), head camera + observer camera
side by side.

Usage::

    python scripts/robotwin/build_label_set.py --robotwin-root <RoboTwin> \\
        --runs runs/campaign_x/default runs/campaign_x/mass_100g \\
        --output-dir runs/label_set --positives 40 --negatives 20 --workers 8 --gpus 5 6
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SIMUGUARD_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIMUGUARD_ROOT))
HERE = Path(__file__).resolve()

QUESTIONS = [
    ("q1_nonphysical_motion", "Does any object move in a physically implausible way (e.g. launched without a matching push)? yes/no/unsure"),
    ("q2_policy_on_track", "Before that moment, was the robot behaving sensibly for the task? yes/no/unsure"),
    ("notes", "free text"),
]


# ---------------------------------------------------------------------------- sampling
def collect(roots: list[str]) -> tuple[list[dict], list[dict]]:
    from simuguard.core import StateLog

    positives, negatives = [], []
    for root in roots:
        for summary_path in sorted(Path(root).resolve().glob("**/simuguard/segments/*/summary.json")):
            segment = summary_path.parent
            if not (segment / "controls.npz").is_file():
                continue
            summary = json.loads(summary_path.read_text())
            meta = summary.get("metadata") or {}
            run = segment.parents[1]
            base = {"segment": str(segment), "config": run.parent.name, "repeat": run.name,
                    "phase": meta.get("phase"), "seed": meta.get("seed")}
            events = summary.get("events", [])
            confirmed = [e for e in events if e["status"] == "confirmed"]
            for event in confirmed:
                positives.append({**base, "kind": "confirmed_event", "onset_substep": event["onset_substep"],
                                  "detector": event["detector"], "event_id": event["event_id"]})
            for event in events:
                if event["status"] == "flag" and event["detector"] in ("impulse_spike", "actuation_bound"):
                    if all(abs(event["onset_substep"] - c["onset_substep"]) > 1000 for c in confirmed):
                        negatives.append({**base, "kind": "unconfirmed_flag", "onset_substep": event["onset_substep"],
                                          "detector": event["detector"], "event_id": event["event_id"]})
            if not confirmed:
                states = StateLog.load(segment / "states.npz")
                target = next((b for b in states.body_ids if "can" in b.lower()), states.body_ids[0])
                speeds = np.linalg.norm(states.array()[:, states.body_ids.index(target), 7:10], axis=1)
                if speeds.size and np.isfinite(speeds).any():
                    i = int(np.nanargmax(speeds))
                    negatives.append({**base, "kind": "fastest_normal_motion", "onset_substep": int(states.substeps[i]),
                                      "detector": None, "peak_speed_mps": round(float(speeds[i]), 3)})
    return positives, negatives


def stratified(items: list[dict], n: int, rng: random.Random, keys=("config", "phase", "kind")) -> list[dict]:
    groups: dict = {}
    for item in items:
        groups.setdefault(tuple(item.get(k) for k in keys), []).append(item)
    for group in groups.values():
        rng.shuffle(group)
    picked = []
    while len(picked) < n and any(groups.values()):
        for key in sorted(groups, key=str):
            if groups[key] and len(picked) < n:
                picked.append(groups[key].pop())
    return picked


# ---------------------------------------------------------------------------- rendering
def render_clip(args) -> int:
    from simuguard.adapters.robotwin import RoboTwinAdapter, make_task_env
    from simuguard.core import ControlLog, StateLog

    segment = Path(args.segment_dir).resolve()
    manifest = json.loads((segment / "manifest.json").read_text())
    summary = json.loads((segment / "summary.json").read_text())
    meta = {**manifest["metadata"], **(summary.get("metadata") or {})}
    controls = ControlLog.load(segment / "controls.npz")
    states = StateLog.load(segment / "states.npz")
    start, end = max(1, args.onset - args.pre), args.onset + args.post
    row_of = {int(s): i for i, s in enumerate(states.substeps)}
    recorded = states.array()

    env, _ = make_task_env(args.robotwin_root, meta["task"], int(meta["seed"]))
    frames, mismatches, checked = [], 0, 0
    try:
        adapter = RoboTwinAdapter(env)
        ids = states.body_ids
        for record in controls.between(0, controls.newest_substep):
            if record.substep == 0:
                continue
            if record.substep > end:
                break
            adapter.apply_control(record)
            adapter.scene.step()
            if record.substep >= start and (record.substep - start) % args.every == 0:
                live = adapter.read_states(ids)
                row = row_of.get(int(record.substep))
                if row is not None:
                    checked += 1
                    got = np.concatenate([live[b].position for b in ids])
                    want = recorded[row][:, 0:3].reshape(-1)
                    mismatches += int(not np.array_equal(got, want))
                env._update_render()
                env.cameras.update_picture()
                head = env.cameras.get_rgb()["head_camera"]["rgb"]
                observer = env.cameras.get_observer_rgb()
                frames.append(_side_by_side(head, observer))
    finally:
        try:
            env.close_env()
        except Exception:  # noqa: BLE001
            pass

    out = Path(args.clip)
    out.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    proc = subprocess.Popen(
        [args.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(args.fps), "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "20", str(out)],
        stdin=subprocess.PIPE,
    )
    for frame in frames:
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    proc.wait()
    info = {"frames": len(frames), "position_checks": checked, "position_mismatches": mismatches,
            "window": [start, end], "every": args.every}
    Path(str(out) + ".json").write_text(json.dumps(info))
    print(json.dumps(info))
    return 0 if mismatches == 0 else 1


def _side_by_side(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    height = max(a.shape[0], b.shape[0])

    def pad(img):
        extra = height - img.shape[0]
        return np.pad(img, ((0, extra), (0, 0), (0, 0))) if extra else img

    frame = np.concatenate([pad(a), pad(b)], axis=1)
    h, w = frame.shape[:2]
    return frame[: h - h % 2, : w - w % 2]  # yuv420p needs even dimensions


# ---------------------------------------------------------------------------- batch
def build(args) -> int:
    out = Path(args.output_dir).resolve()
    (out / "clips").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    positives, negatives = collect(args.runs)
    picked = stratified(positives, args.positives, rng) + stratified(negatives, args.negatives, rng, keys=("kind", "config"))
    rng.shuffle(picked)
    for index, item in enumerate(picked, start=1):
        item["clip_id"] = f"clip{index:03d}"
    print(f"[labels] {len(positives)} positives, {len(negatives)} negative candidates -> {len(picked)} clips", flush=True)
    gpus = args.gpus or [None]

    def work(indexed):
        index, item = indexed
        clip = out / "clips" / f"{item['clip_id']}.mp4"
        if clip.exists() and Path(str(clip) + ".json").exists():
            return
        env = dict(os.environ)
        if gpus[index % len(gpus)] is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpus[index % len(gpus)])
        proc = subprocess.run(
            [args.python, "-B", str(HERE), "--robotwin-root", args.robotwin_root, "--segment-dir", item["segment"],
             "--onset", str(item["onset_substep"]), "--clip", str(clip), "--pre", str(args.pre), "--post", str(args.post),
             "--every", str(args.every), "--fps", str(args.fps), "--ffmpeg", args.ffmpeg],
            capture_output=True, text=True, env=env,
        )
        (out / "clips" / f"{item['clip_id']}.log").write_text(proc.stdout + proc.stderr)
        print(f"[labels] {item['clip_id']} exit={proc.returncode}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, enumerate(picked)))

    (out / "key.json").write_text(json.dumps(picked, indent=2))
    with open(out / "labels.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clip_id"] + [q for q, _ in QUESTIONS])
        for item in sorted(picked, key=lambda x: x["clip_id"]):
            writer.writerow([item["clip_id"]] + [""] * len(QUESTIONS))
    (out / "INSTRUCTIONS.txt").write_text(
        "Each clip: left = robot head camera, right = fixed observer camera; half speed.\n"
        "The clip is centred on a moment of interest (2 s before, 3 s after at real time).\n"
        "Fill labels.csv:\n" + "\n".join(f"  {q}: {text}" for q, text in QUESTIONS) + "\n"
        "Do not open key.json before labelling is finished.\n"
    )
    print(f"[labels] wrote {out}/labels.csv and key.json")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", required=True)
    # single clip
    parser.add_argument("--segment-dir")
    parser.add_argument("--onset", type=int)
    parser.add_argument("--clip")
    # batch
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--output-dir")
    parser.add_argument("--positives", type=int, default=40)
    parser.add_argument("--negatives", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", nargs="*", default=None)
    # rendering
    parser.add_argument("--pre", type=int, default=500, help="substeps before the onset")
    parser.add_argument("--post", type=int, default=750, help="substeps after the onset")
    parser.add_argument("--every", type=int, default=5, help="substeps per frame")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if args.segment_dir:
        if args.onset is None or not args.clip:
            parser.error("--onset and --clip are required with --segment-dir")
        return render_clip(args)
    if not (args.runs and args.output_dir):
        parser.error("batch mode needs --runs and --output-dir")
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
