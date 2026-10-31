#!/usr/bin/env python3
"""Write down what one evaluation run actually covered, so any number can be traced back.

For every seed the evaluator touched it records: whether the scripted expert check
passed (RoboTwin skips the seed otherwise), the policy's outcome, the detected events,
the official video, and the SimuGuard segment that replays the episode exactly.

Outputs, inside the run directory:

* ``manifest.json`` - machine-readable, one entry per seed;
* ``SEEDS.md``      - the same as a table a person can read;
* ``videos/``       - the official evaluation videos, copied next to the recordings.

Usage::

    python scripts/robotwin/build_run_manifest.py --run-dir runs/x/stack_bowls_two \\
        --task stack_bowls_two --robotwin-root <RoboTwin> --started-epoch 1790000000
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def newest_video_dir(robotwin_root: Path, task: str, started_epoch: float) -> Path | None:
    base = robotwin_root / "eval_result" / task
    if not base.is_dir():
        return None
    candidates = [p for p in base.glob("*/*/*/*") if p.is_dir() and p.stat().st_mtime >= started_epoch - 60]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--started-epoch", type=float, default=0.0)
    args = parser.parse_args()

    run = Path(args.run_dir).resolve()
    segments_dir = run / "simuguard" / "segments"
    seeds: dict[int, dict] = {}
    for summary_path in sorted(segments_dir.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        meta = summary.get("metadata") or {}
        seed, phase = meta.get("seed"), meta.get("phase")
        if seed is None or phase is None:
            continue
        outcome = meta.get("outcome") or {}
        events = [e for e in summary.get("events", []) if e.get("status") == "confirmed"]
        entry = seeds.setdefault(int(seed), {"seed": int(seed), "episode_index": meta.get("episode_index")})
        entry[phase] = {
            "segment": str(summary_path.parent.relative_to(run)),
            "substeps": summary.get("substeps"),
            "confirmed_events": len(events),
            "event_onsets": [e["onset_substep"] for e in events][:12],
            "peak_event_speed_mps": round(max((e.get("metrics", {}).get("max_speed_mps", 0.0) for e in events),
                                              default=0.0), 3),
            "plan_success": outcome.get("plan_success"),
            "check_success": outcome.get("check_success"),
            "eval_success": outcome.get("eval_success"),
            "policy_actions": outcome.get("take_action_cnt"),
            "instruction": outcome.get("instruction"),
            "replayable": (summary_path.parent / "controls.npz").is_file(),
        }

    # the evaluator numbers its videos by scored episode, in seed order
    video_dir = newest_video_dir(Path(args.robotwin_root).resolve(), args.task, args.started_epoch)
    scored = [s for s in sorted(seeds) if "policy" in seeds[s]]
    copied = 0
    if video_dir is not None:
        (run / "videos").mkdir(exist_ok=True)
        for index, seed in enumerate(scored):
            source = video_dir / f"episode{index}.mp4"
            if source.is_file():
                target = run / "videos" / f"seed{seed}_episode{index}.mp4"
                shutil.copy2(source, target)
                seeds[seed]["video"] = str(target.relative_to(run))
                copied += 1
        result_file = video_dir / "_result.txt"
        if result_file.is_file():
            shutil.copy2(result_file, run / "official_result.txt")

    successes = sum(1 for s in scored if seeds[s]["policy"].get("eval_success"))
    manifest = {
        "task": args.task,
        "run_dir": str(run),
        "official_video_dir": str(video_dir) if video_dir else None,
        "seeds_touched": sorted(seeds),
        "seeds_scored": scored,
        "seeds_skipped_by_expert_check": [s for s in sorted(seeds) if "policy" not in seeds[s]],
        "policy_success": successes,
        "policy_episodes": len(scored),
        "policy_episodes_with_events": sum(1 for s in scored if seeds[s]["policy"]["confirmed_events"]),
        "videos_copied": copied,
        "episodes": [seeds[s] for s in sorted(seeds)],
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    lines = [
        f"# {args.task}",
        "",
        f"Scored {len(scored)} seeds, {successes} successes "
        f"({100.0 * successes / len(scored):.1f}%)." if scored else "No seed was scored.",
        f"Seeds skipped because the scripted expert check failed: "
        f"{manifest['seeds_skipped_by_expert_check'] or 'none'}.",
        "",
        "| seed | expert check | expert events | policy result | policy events | peak event speed | video | replay |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for seed in sorted(seeds):
        entry = seeds[seed]
        expert, policy = entry.get("expert", {}), entry.get("policy")
        expert_ok = bool(expert.get("plan_success")) and bool(expert.get("check_success"))
        lines.append(
            f"| {seed} | {'pass' if expert_ok else 'FAIL'} | {expert.get('confirmed_events', '-')} | "
            + (f"{'success' if policy['eval_success'] else 'FAIL'} | {policy['confirmed_events']} | "
               f"{policy['peak_event_speed_mps']} m/s | {entry.get('video', '-')} | {policy['segment']} |"
               if policy else "not scored | - | - | - | - |"))
    (run / "SEEDS.md").write_text("\n".join(lines) + "\n")
    print(f"[manifest] {args.task}: scored {len(scored)}, success {successes}, "
          f"with events {manifest['policy_episodes_with_events']}, videos {copied}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
