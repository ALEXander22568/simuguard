#!/usr/bin/env python3
"""Roll back before an anomaly and let the live policy try again.

For a failed episode that contains a confirmed anomaly event:

1. rebuild the environment and replay the recording bit-exactly up to
   ``distance`` substeps before the event;
2. hand control to the live LingBot-VA bridge with the episode's remaining
   action budget (``step_lim - take_action_cnt``), running the official
   observe/act loop;
3. if a new confirmed anomaly occurs after the resume point, abort and roll
   back again, up to ``--max-attempts`` times.

``--mode progressive`` (default) rolls back to just before the *new* event,
replaying the aborted attempt's own actuation - so progress the policy made
before the new event is kept.  ``--mode fixed`` always restarts from the
first resume point.

Per episode outcome:

* ``rescued``            - an attempt reached task success;
* ``anomaly_persists``   - every attempt ended in a new anomaly;
* ``failed_without_anomaly`` - an attempt used up the budget with no anomaly
  (from that state the policy itself did not finish the task).

Report the rescued rate *together with* the number of attempts: retrying until
success is an upper bound, not a replacement for the official success rate.

Caveat: the policy is reset at each resume point and restarts its visual
history from that frame.

Usage (one episode)::

    python scripts/robotwin/rollback_retry.py --robotwin-root <RoboTwin> \\
        --segment-dir <seg> --port <bridge> --output-dir runs/retry/<seg>

Usage (batch; one bridge per port, one worker per bridge)::

    python scripts/robotwin/rollback_retry.py --robotwin-root <RoboTwin> \\
        --runs runs/campaign_x/default --ports 60001 60002 --output-dir runs/retry
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SIMUGUARD_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIMUGUARD_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

HERE = Path(__file__).resolve()


def confirmed(events) -> list[dict]:
    return sorted((e for e in events if e["status"] == "confirmed"), key=lambda e: e["onset_substep"])


def run_attempt(args, module, meta: dict, base_controls, base_states, resume: int, attempt: int, out: Path) -> dict:
    from resume_from_replay import policy_actions, replay_to
    from replay_intervention import build_intervention
    from simuguard.adapters.robotwin import RoboTwinAdapter, make_task_env
    from simuguard.core import EpisodeRecorder, SubstepMonitor
    from simuguard.core.events import EventStatus
    from simuguard.integrations.robotwin_eval import eval_monitor_config
    from simuguard.presets import default_detectors

    task, seed = meta["task"], int(meta["seed"])
    env, _ = make_task_env(args.robotwin_root, task, seed)
    record: dict = {"attempt": attempt, "resume_substep": resume}
    try:
        adapter = RoboTwinAdapter(env)
        gate = None
        try:
            gate = adapter.containment_gate()
        except Exception:  # noqa: BLE001
            pass
        monitor = SubstepMonitor(
            adapter,
            default_detectors(ejection_gate=gate),
            episode_id=f"retry:{attempt}",
            # initial snapshot only (substep 0) so the attempt log stays replayable by the other tools
            config=eval_monitor_config({"snapshot_interval_substeps": 10**9, "bundle_statuses": []}),
            recorder=EpisodeRecorder(out / f"attempt{attempt}"),
            metadata={"task": task, "seed": seed, "phase": "policy", "resume_substep": resume, "attempt": attempt,
                      "source_segment": args.segment_dir, "intervention": args.intervention},
        )
        monitor.attach()
        replayed = replay_to(adapter, base_controls.between(0, base_controls.newest_substep), resume)

        mask = base_states.substeps <= resume
        replay_log = monitor.state_log.array()
        cols = [monitor.state_log.body_ids.index(b) for b in base_states.body_ids]
        n = min(int(mask.sum()), replay_log.shape[0])
        record["prefix_bit_identical"] = bool(
            np.array_equal(base_states.array()[mask][:n], replay_log[:n][:, cols, :], equal_nan=True)
        )

        if args.intervention != "baseline":
            _, apply = build_intervention(args.intervention)
            if apply is not None:
                apply(adapter)

        env.take_action_cnt = int(base_controls.get(resume).control_step)
        env.eval_success = False
        budget = int(env.step_lim) - env.take_action_cnt
        record.update({"take_action_cnt_at_resume": env.take_action_cnt, "action_budget": budget})

        def new_anomaly() -> bool:
            return any(
                e.status == EventStatus.CONFIRMED and e.onset_substep > resume for e in monitor.events.values()
            )

        policy_args = SimpleNamespace(
            host=args.host, port=args.port, task=task, seed_for_policy=seed, resume_substep=resume,
            request_timeout_s=args.request_timeout_s, robotwin_root=args.robotwin_root, frequency=args.frequency,
        )
        instruction = meta.get("outcome", {}).get("instruction") or env.get_instruction()
        started = time.time()
        stats = policy_actions(module, env, policy_args, instruction, budget, should_stop=new_anomaly)
        record["wall_s"] = round(time.time() - started, 1)
        record["policy"] = stats
        record["success"] = bool(env.eval_success)

        final = monitor.finalize()
        anomalies = [e for e in confirmed(final["events"]) if e["onset_substep"] > resume]
        record["substeps_replayed"] = replayed
        record["substeps_total"] = final["substeps"]
        record["anomalies_after_resume"] = [
            {"onset_substep": e["onset_substep"], "control_step": e["control_step"], "bodies": e["bodies"],
             "peak_speed_mps": (e.get("metrics") or {}).get("max_speed_mps")}
            for e in anomalies
        ]
        if record["success"]:
            record["outcome"] = "success"
        elif anomalies:
            record["outcome"] = "anomaly"
        else:
            record["outcome"] = "budget_exhausted"
        return {"record": record, "controls": monitor.controls, "states": monitor.state_log}
    finally:
        try:
            env.close_env()
        except Exception:  # noqa: BLE001
            pass


def retry_episode(args) -> int:
    from simuguard.adapters.robotwin.env import official_eval_module
    from simuguard.core import ControlLog, StateLog

    segment = Path(args.segment_dir).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((segment / "manifest.json").read_text())
    summary = json.loads((segment / "summary.json").read_text())
    meta = {**manifest["metadata"], **(summary.get("metadata") or {})}
    events = confirmed(summary["events"])
    if not events:
        raise SystemExit(f"{segment} has no confirmed events")

    module = official_eval_module(args.robotwin_root)
    base_controls = ControlLog.load(segment / "controls.npz")
    base_states = StateLog.load(segment / "states.npz")
    first_resume = max(1, events[0]["onset_substep"] - args.distance)
    report = {
        "segment": str(segment), "task": meta["task"], "seed": meta["seed"],
        "recorded_success": (meta.get("outcome") or {}).get("eval_success"),
        "recorded_events": [e["onset_substep"] for e in events],
        "mode": args.mode, "distance": args.distance, "max_attempts": args.max_attempts,
        "intervention": args.intervention, "attempts": [],
    }

    resume = first_resume
    for attempt in range(1, args.max_attempts + 1):
        print(f"[retry] attempt {attempt}: resume at substep {resume}", flush=True)
        result = run_attempt(args, module, meta, base_controls, base_states, resume, attempt, out)
        record = result["record"]
        report["attempts"].append(record)
        print("   " + json.dumps({k: record.get(k) for k in ("outcome", "prefix_bit_identical", "take_action_cnt_at_resume", "wall_s")}), flush=True)
        (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
        if record["outcome"] != "anomaly":
            break
        if args.mode == "progressive":
            base_controls, base_states = result["controls"], result["states"]
            resume = max(1, record["anomalies_after_resume"][0]["onset_substep"] - args.distance)
        else:
            resume = first_resume

    outcomes = [a["outcome"] for a in report["attempts"]]
    if "success" in outcomes:
        report["verdict"] = "rescued"
    elif outcomes and outcomes[-1] == "budget_exhausted":
        report["verdict"] = "failed_without_anomaly"
    else:
        report["verdict"] = "anomaly_persists"
    report["attempts_used"] = len(outcomes)
    report["all_prefixes_bit_identical"] = all(a.get("prefix_bit_identical") for a in report["attempts"])
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print("verdict:", report["verdict"], "attempts:", report["attempts_used"])
    return 0


# ---------------------------------------------------------------------------- batch
def find_candidates(roots: list[str]) -> list[dict]:
    items = []
    for root in roots:
        for summary_path in sorted(Path(root).resolve().glob("**/simuguard/segments/*/summary.json")):
            summary = json.loads(summary_path.read_text())
            meta = summary.get("metadata") or {}
            if meta.get("phase") != "policy" or (meta.get("outcome") or {}).get("eval_success"):
                continue
            if not confirmed(summary.get("events", [])) or not (summary_path.parent / "controls.npz").is_file():
                continue
            run = summary_path.parents[2]
            items.append({"segment": str(summary_path.parent), "key": f"{run.parent.name}_{run.name}_{summary_path.parent.name}"})
    return items


def batch(args) -> int:
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    items = find_candidates(args.runs)
    if args.limit:
        items = items[: args.limit]
    print(f"[retry] {len(items)} failed episodes with anomalies; {len(args.ports)} bridges", flush=True)
    queue = list(items)
    lock = threading.Lock()

    def worker(port: str) -> None:
        while True:
            with lock:
                if not queue:
                    return
                item = queue.pop(0)
            target = out / item["key"]
            if (target / "report.json").exists() and "verdict" in json.loads((target / "report.json").read_text()):
                continue
            target.mkdir(parents=True, exist_ok=True)
            cmd = [args.python, "-B", str(HERE), "--robotwin-root", args.robotwin_root, "--segment-dir", item["segment"],
                   "--output-dir", str(target), "--port", str(port), "--host", args.host, "--mode", args.mode,
                   "--distance", str(args.distance), "--max-attempts", str(args.max_attempts),
                   "--intervention", args.intervention]
            env = dict(os.environ)
            if args.sim_gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(args.sim_gpu)
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            (out / f"{item['key']}.log").write_text(proc.stdout + proc.stderr)
            verdict = "error"
            if (target / "report.json").exists():
                verdict = json.loads((target / "report.json").read_text()).get("verdict", "error")
            print(f"[retry] {item['key']} (port {port}) -> {verdict}", flush=True)

    threads = [threading.Thread(target=worker, args=(p,)) for p in args.ports]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    write_summary(out, items)
    return 0


def write_summary(out: Path, items: list[dict]) -> None:
    rows = []
    for item in items:
        path = out / item["key"] / "report.json"
        report = json.loads(path.read_text()) if path.exists() else {}
        rows.append({
            "key": item["key"],
            "verdict": report.get("verdict", "missing"),
            "attempts_used": report.get("attempts_used"),
            "outcomes": [a["outcome"] for a in report.get("attempts", [])],
            "all_prefixes_bit_identical": report.get("all_prefixes_bit_identical"),
        })
    done = [r for r in rows if r["verdict"] not in ("missing", "error")]
    rescued = [r for r in done if r["verdict"] == "rescued"]
    summary = {
        "episodes": len(rows),
        "completed": len(done),
        "verdicts": {v: sum(1 for r in done if r["verdict"] == v) for v in ("rescued", "anomaly_persists", "failed_without_anomaly")},
        "rescued_rate": round(len(rescued) / len(done), 3) if done else None,
        "rescued_at_attempt": {str(k): sum(1 for r in rescued if r["attempts_used"] == k) for k in sorted({r["attempts_used"] for r in rescued})},
        "rows": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--segment-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--port", help="bridge port (single episode)")
    parser.add_argument("--runs", nargs="+", help="batch: roots searched for **/simuguard/segments")
    parser.add_argument("--ports", nargs="+", help="batch: one worker per bridge port")
    parser.add_argument("--sim-gpu", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--mode", choices=("progressive", "fixed"), default="progressive")
    parser.add_argument("--distance", type=int, default=250, help="substeps before the event to resume from")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--intervention", default="baseline")
    parser.add_argument("--frequency", type=int, default=30)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    if args.segment_dir:
        if not args.port:
            parser.error("--port is required")
        return retry_episode(args)
    if not args.runs:
        parser.error("give --segment-dir or --runs")
    if args.summary_only:
        write_summary(Path(args.output_dir).resolve(), find_candidates(args.runs))
        return 0
    if not args.ports:
        parser.error("batch mode needs --ports")
    return batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
