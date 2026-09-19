#!/usr/bin/env python3
"""Attribute recorded failures to the policy or to the simulator by counterfactual replay.

Question per failed episode: *would the policy's own actions have succeeded if
the simulator had not misbehaved?*  Each failed episode is replayed from its
start with the recorded actuation (bit-exact), under several conditions:

* ``baseline``   - unchanged.  Gate: the replay must be bit-identical to the
  recording and must also fail; otherwise the episode is not interpretable.
* ``vclamp_<v>`` - after every substep, if the target moves faster than
  ``v`` m/s, its velocity and that substep's displacement are scaled down to
  ``v``.  Normal target motion in this task peaks below ~1.7 m/s, so the clamp
  only acts on the anomalous energy and leaves everything else unchanged.
* ``mass_<g>g``  - target mass changed (removes the anomaly, but also changes
  grasp dynamics; used as a second, independent counterfactual).

Success is evaluated exactly as the benchmark does: RoboTwin's
``check_success()`` after every substep for the policy phase (the evaluator
stops at the first success), and at the end for the expert check.

Verdicts (policy and expert phases alike):

* ``environment_caused``     - the recording has confirmed anomaly events and
  the same actions succeed once the anomaly is removed;
* ``anomaly_unresolved``     - anomaly events, but no counterfactual succeeds
  (policy may also be wrong; needs the closed-loop retry test);
* ``outcome_changed_without_anomaly`` - no events, yet a counterfactual
  succeeds: the intervention changes more than the anomaly (confound check);
* ``policy_failure``         - no events and no counterfactual succeeds;
* ``unverifiable``           - baseline gate failed.

Usage (batch)::

    python scripts/robotwin/attribute_failures.py --robotwin-root <RoboTwin> \\
        --runs runs/campaign_x/default runs/campaign_x/mass_100g \\
        --output-dir runs/attribution --workers 4 --gpus 0 2 5 6

Usage (one segment)::

    python scripts/robotwin/attribute_failures.py --robotwin-root <RoboTwin> \\
        --segment-dir <seg> --report out.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SIMUGUARD_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIMUGUARD_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

HERE = Path(__file__).resolve()
DEFAULT_CONDITIONS = ["baseline", "vclamp_2.0", "mass_100g"]


# ---------------------------------------------------------------------------- selection
def recorded_outcome(summary: dict) -> tuple[str | None, bool | None]:
    meta = summary.get("metadata") or {}
    outcome = meta.get("outcome") or {}
    phase = meta.get("phase")
    if phase == "policy":
        return phase, bool(outcome.get("eval_success"))
    if phase == "expert":
        return phase, bool(outcome.get("plan_success")) and bool(outcome.get("check_success"))
    return phase, None


def find_failures(roots: list[str], phases: list[str]) -> list[dict]:
    found = []
    for root in roots:
        root_path = Path(root).resolve()
        for summary_path in sorted(root_path.glob("**/simuguard/segments/*/summary.json")):
            segment = summary_path.parent
            if not (segment / "controls.npz").is_file():
                continue
            summary = json.loads(summary_path.read_text())
            phase, success = recorded_outcome(summary)
            if phase not in phases or success is not False:
                continue
            outcome = (summary.get("metadata") or {}).get("outcome") or {}
            if phase == "expert" and not outcome.get("plan_success"):
                continue  # planner failure: nothing was executed that could be attributed
            run = segment.parents[2]  # <config>/<rep>/simuguard/segments/<seg> -> <rep>
            found.append(
                {
                    "segment": str(segment),
                    "key": f"{run.parent.name}_{run.name}_{segment.name}",
                    "config": run.parent.name,
                    "repeat": run.name,
                    "phase": phase,
                    "seed": summary["metadata"].get("seed"),
                    "recorded_confirmed_events": int(summary.get("confirmed_count", 0)),
                }
            )
    return found


# ---------------------------------------------------------------------------- one segment
def run_condition(robotwin_root: str, segment: Path, name: str, phase: str, stop_on_success: bool) -> dict:
    from replay_intervention import build_intervention
    from simuguard.adapters.robotwin import RoboTwinAdapter, make_task_env
    from simuguard.core import ControlLog, StateLog, SubstepMonitor
    from simuguard.core.types import BodyRole
    from simuguard.integrations.robotwin_eval import eval_monitor_config, reapply_recorded_intervention
    from simuguard.presets import default_detectors

    summary = json.loads((segment / "summary.json").read_text())
    manifest = json.loads((segment / "manifest.json").read_text())
    meta = {**manifest["metadata"], **(summary.get("metadata") or {})}
    controls = ControlLog.load(segment / "controls.npz")
    recorded_states = StateLog.load(segment / "states.npz")

    env, _ = make_task_env(robotwin_root, meta["task"], int(meta["seed"]))
    result: dict = {"name": name}
    try:
        adapter = RoboTwinAdapter(env)
        target = adapter.body_ids_with_role(BodyRole.TARGET)[0]
        # the physics the episode was recorded under (closed-loop campaigns), then the counterfactual on top
        result["recorded_sim_intervention"] = reapply_recorded_intervention(adapter, meta)
        clamp = None
        if name.startswith("vclamp_"):
            clamp = _install_velocity_clamp(adapter, target, float(name.split("_", 1)[1]))
            result["description"] = f"target speed clamped to {clamp.cap} m/s after each substep"
        else:
            description, apply = build_intervention(name)
            result["description"] = description
            if apply is not None:
                apply(adapter)
        gate = None
        try:
            gate = adapter.containment_gate()
        except Exception:  # noqa: BLE001
            pass
        monitor = SubstepMonitor(
            adapter,
            default_detectors(ejection_gate=gate),
            episode_id=f"attribution:{name}",
            config=eval_monitor_config(
                {"snapshot_interval_substeps": 0, "bundle_statuses": [], "save_episode_logs": False}
            ),
        )
        monitor.attach()

        first_success = None
        applied = 0
        started = time.time()
        for record in controls.between(0, controls.newest_substep):
            if record.substep == 0:
                continue
            adapter.apply_control(record)
            adapter.scene.step()
            applied += 1
            if first_success is None and env.check_success():
                first_success = record.substep
                if stop_on_success:
                    break
        final_success = bool(env.check_success())
        summary_after = monitor.finalize()

        result.update(
            {
                "substeps_replayed": applied,
                "wall_s": round(time.time() - started, 1),
                "first_success_substep": first_success,
                "success_at_end": final_success,
                # policy phase: the evaluator stops at the first successful substep
                "success": (first_success is not None) if phase == "policy" else final_success,
                "confirmed_events": summary_after["confirmed_count"],
                "target_mass_kg": adapter.mass(target),
            }
        )
        speeds = np.linalg.norm(monitor.state_log.array()[:, monitor.state_log.body_ids.index(target), 7:10], axis=1)
        result["peak_target_speed_mps"] = float(np.nanmax(speeds)) if speeds.size else None
        if clamp is not None:
            result["clamped_substeps"] = clamp.count
            result["clamp_max_raw_speed_mps"] = round(clamp.max_raw, 3)
        if name == "baseline":
            live = recorded_states.array()
            replayed = monitor.state_log.array()[:, [monitor.state_log.body_ids.index(b) for b in recorded_states.body_ids], :]
            result["bit_identical"] = bool(live.shape == replayed.shape and np.array_equal(live, replayed, equal_nan=True))
            result["reproduces_recorded_events"] = summary_after["confirmed_count"] == summary["confirmed_count"]
    finally:
        try:
            env.close_env()
        except Exception:  # noqa: BLE001
            pass
    return result


class _Clamp:
    def __init__(self, cap: float) -> None:
        self.cap = cap
        self.count = 0
        self.max_raw = 0.0


def _install_velocity_clamp(adapter, target: str, cap: float) -> _Clamp:
    """Wrap the raw physics step (installed *before* the monitor, so it observes the clamped state)."""

    handle = adapter._bodies[target]
    component, entity = handle.component, handle.entity
    scene = adapter.scene
    raw_step = scene.step
    clamp = _Clamp(cap)

    def clamped_step(*args, **kwargs):
        before = entity.get_pose()
        out = raw_step(*args, **kwargs)
        velocity = np.asarray(component.get_linear_velocity(), dtype=float)
        speed = float(np.linalg.norm(velocity))
        if speed > cap:
            scale = cap / speed
            after = entity.get_pose()
            p0, p1 = np.asarray(before.p, dtype=float), np.asarray(after.p, dtype=float)
            corrected = type(after)(p0 + (p1 - p0) * scale, after.q)
            entity.set_pose(corrected)
            component.set_linear_velocity(velocity * scale)
            clamp.count += 1
            clamp.max_raw = max(clamp.max_raw, speed)
        return out

    scene.step = clamped_step
    return clamp


def attribute_segment(args) -> int:
    segment = Path(args.segment_dir).resolve()
    summary = json.loads((segment / "summary.json").read_text())
    phase, recorded_success = recorded_outcome(summary)
    report = {
        "segment": str(segment),
        "phase": phase,
        "seed": (summary.get("metadata") or {}).get("seed"),
        "recorded_success": recorded_success,
        "recorded_confirmed_events": int(summary.get("confirmed_count", 0)),
        "recorded_event_onsets": [e["onset_substep"] for e in summary.get("events", []) if e["status"] == "confirmed"],
        "conditions": [],
    }
    for name in args.conditions:
        # the policy evaluator stops at the first successful substep; the expert check is judged at the end
        stop = name != "baseline" and phase == "policy"
        condition = run_condition(args.robotwin_root, segment, name, phase, stop_on_success=stop)
        report["conditions"].append(condition)
        print(json.dumps(condition, default=str)[:400], flush=True)
    report["verdict"] = verdict(report)
    Path(args.report).write_text(json.dumps(report, indent=2, default=str))
    print("verdict:", report["verdict"])
    return 0


def verdict(report: dict) -> str:
    conditions = {c["name"]: c for c in report["conditions"]}
    baseline = conditions.get("baseline")
    if baseline is not None and not (baseline.get("bit_identical") and not baseline.get("success")):
        return "unverifiable"
    counterfactual_success = any(c.get("success") for n, c in conditions.items() if n != "baseline")
    if report["recorded_confirmed_events"] > 0:
        return "environment_caused" if counterfactual_success else "anomaly_unresolved"
    return "outcome_changed_without_anomaly" if counterfactual_success else "policy_failure"


# ---------------------------------------------------------------------------- batch
def batch(args) -> int:
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    failures = find_failures(args.runs, args.phases)
    if args.limit:
        failures = failures[: args.limit]
    print(f"[attribution] {len(failures)} failed segments", flush=True)
    gpus = args.gpus or [None]

    def work(indexed):
        index, item = indexed
        report = out / f"{item['key']}.json"
        if report.exists():
            return
        env = dict(os.environ)
        gpu = gpus[index % len(gpus)]
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        proc = subprocess.run(
            [args.python, "-B", str(HERE), "--robotwin-root", args.robotwin_root, "--segment-dir", item["segment"],
             "--report", str(report), "--conditions", *args.conditions],
            capture_output=True, text=True, env=env,
        )
        (out / f"{item['key']}.log").write_text(proc.stdout + proc.stderr)
        status = json.loads(report.read_text())["verdict"] if report.exists() else f"error {proc.returncode}"
        print(f"[attribution] {item['key']} phase={item['phase']} events={item['recorded_confirmed_events']} -> {status}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, enumerate(failures)))
    write_summary(out, failures, args.runs)
    return 0


def write_summary(out: Path, failures: list[dict], roots: list[str]) -> None:
    rows = []
    for item in failures:
        path = out / f"{item['key']}.json"
        if not path.exists():
            rows.append({**item, "verdict": "missing"})
            continue
        report = json.loads(path.read_text())
        conditions = {c["name"]: c for c in report["conditions"]}
        rows.append(
            {
                **item,
                "verdict": report["verdict"],
                "counterfactual_success": {n: c.get("success") for n, c in conditions.items() if n != "baseline"},
                "baseline_bit_identical": conditions.get("baseline", {}).get("bit_identical"),
            }
        )

    summary: dict = {"failures": len(rows), "by_phase": {}, "corrected_policy_success": {}, "rows": rows}
    for phase in sorted({r["phase"] for r in rows}):
        phase_rows = [r for r in rows if r["phase"] == phase]
        by_config: dict = {}
        for row in phase_rows:
            by_config.setdefault(row["config"], Counter())[row["verdict"]] += 1
        conditions = sorted({n for r in phase_rows for n in (r.get("counterfactual_success") or {})})
        agreement = {}
        for with_events in (True, False):
            group = [r for r in phase_rows if (r["recorded_confirmed_events"] > 0) == with_events and r.get("counterfactual_success")]
            agreement["with_events" if with_events else "without_events"] = {
                "episodes": len(group),
                **{n: sum(1 for r in group if r["counterfactual_success"].get(n)) for n in conditions},
            }
        summary["by_phase"][phase] = {
            "verdicts": dict(Counter(r["verdict"] for r in phase_rows)),
            "verdicts_by_config": {c: dict(v) for c, v in by_config.items()},
            "counterfactual_success_by_group": agreement,
        }

    # policy phase: recorded successes + failures attributed to the environment, per run
    for root in roots:
        for rep_dir in sorted(Path(root).resolve().glob("**/simuguard")):
            run = rep_dir.parent
            total = succeeded = 0
            for summary_path in rep_dir.glob("segments/*/summary.json"):
                phase, success = recorded_outcome(json.loads(summary_path.read_text()))
                if phase == "policy":
                    total += 1
                    succeeded += bool(success)
            if not total:
                continue
            key_prefix = f"{run.parent.name}_{run.name}_"
            attributed = sum(
                1 for r in rows if r["phase"] == "policy" and r["key"].startswith(key_prefix) and r["verdict"] == "environment_caused"
            )
            summary["corrected_policy_success"][f"{run.parent.name}/{run.name}"] = {
                "episodes": total,
                "official_success": succeeded,
                "environment_caused_failures": attributed,
                "official_rate": round(succeeded / total, 4),
                "corrected_rate_upper": round((succeeded + attributed) / total, 4),
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1, default=str))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--segment-dir", help="attribute one segment")
    parser.add_argument("--report", help="output for --segment-dir")
    parser.add_argument("--runs", nargs="+", help="batch: roots searched for **/simuguard/segments")
    parser.add_argument("--phases", nargs="+", default=["policy", "expert"])
    parser.add_argument("--output-dir")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpus", nargs="*", default=None, help="render devices, assigned round-robin")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--summary-only", action="store_true", help="rebuild summary.json from existing reports")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if args.segment_dir:
        if not args.report:
            parser.error("--report is required with --segment-dir")
        return attribute_segment(args)
    if not (args.runs and args.output_dir):
        parser.error("batch mode needs --runs and --output-dir")
    if args.summary_only:
        write_summary(Path(args.output_dir).resolve(), find_failures(args.runs, args.phases), args.runs)
        return 0
    return batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
