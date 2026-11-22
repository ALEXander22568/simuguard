#!/usr/bin/env python3
"""Open-loop replay of a recorded RoboTwin episode under a broad or narrow depenetration cap.

Same replay as ``scripts/robotwin/attribute_failures.py`` (scene rebuilt from the seed, recorded
per-substep actuation from substep 0, bit-exact; the recorded closed-loop physics is re-applied
first; the policy-phase evaluator stops at the first success).  Conditions:

* ``baseline``          - unchanged; must reproduce the recording bit for bit.
* ``broad@<cap>``       - the paper's ``depen_<cap>@event``: every dynamic body and link capped at
  ``cap`` m/s from 0.1 s before the anchor until the episode ends.
* ``body@<cap>``        - only the anchor body (the ejected body), window [t0 - 0.1 s, t0 + 0.2 s],
  then each body's stored original value is written back.
* ``partner@<cap>``     - as ``body``, plus its contact partner at the anchor (the other body
  with the deepest penetration in the recorded trace over substeps t0-2..t0, ties and all-zero
  broken by the largest impulse).  A static partner has no such setting: only the body is
  capped and the report says ``partner_static``.
* ``resolved@<cap>``    - as ``body``, but the window ends once the body's penetration is resolved
  (see ``window.py``: <= rest offset + 1 mm for 5 consecutive substeps from t0 on; at most 1 s).

Anchor ``t0`` and anchor body: the first physics-invalid event of the recording (carrier stage if
it ran, else gravity stage; frozen detector output), or ``--anchor-substep`` / ``--anchor-body``
(control episodes).

One process replays one (episode, condition)::

    python scripts/robotwin/narrow/narrow_replay.py --robotwin-root <RoboTwin> --segment-dir <seg> \
        --condition body@1.0 --report out.json

Batch (one subprocess per job line, resumable)::

    python scripts/robotwin/narrow/narrow_replay.py --robotwin-root <RoboTwin> --jobs jobs.jsonl \
        --output-dir <dir> --gpus 2 3 --workers 4
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
SIMUGUARD_ROOT = HERE.parents[3]
sys.path.insert(0, str(SIMUGUARD_ROOT))
sys.path.insert(0, str(HERE.parents[1]))  # scripts/robotwin
sys.path.insert(0, str(HERE.parent))

from window import DepenWindow, WindowRule  # noqa: E402

LEAD_S = 0.1          # window starts this long before the anchor (same as the paper's @event)
AFTER_S = 0.2         # fixed window ends this long after the anchor
RESOLVED_N = 5        # consecutive substeps with penetration <= rest offset + tolerance
RESOLVED_TOL_M = 0.001
RESOLVED_MAX_S = 1.0
PARTNER_FRAMES = 2    # partner searched over recorded trace substeps t0-2..t0


def first_invalid_event(segment: Path) -> dict | None:
    """Same rule as ``attribute_failures.first_invalid_onset``, but returns the event (with its body)."""

    for name in ("events_carrier.json", "events_gravity.json"):
        path = segment / name
        if path.is_file():
            events = [e for e in json.loads(path.read_text()).get("events", []) if e.get("verdict") == "physics_invalid"]
            if not events:
                return None
            event = min(events, key=lambda e: int(e["onset_substep"]))
            body = event.get("body") or (event.get("bodies") or [None])[0]
            return {"onset_substep": int(event["onset_substep"]), "body": body, "event_id": event.get("event_id"),
                    "stage": name}
    return None


def trace_partner(segment: Path, body: str, t0: int, frames: int = PARTNER_FRAMES) -> dict | None:
    """Contact partner of ``body`` in the recorded trace around the anchor."""

    best = None
    path = segment / "trace.jsonl.gz"
    if not path.is_file():
        return None
    with gzip.open(path, "rt") as fh:
        for line in fh:
            frame = json.loads(line)
            s = int(frame["substep"])
            if s < t0 - frames:
                continue
            if s > t0:
                break
            for c in frame.get("contacts", []):
                if body not in (c["a"], c["b"]):
                    continue
                other = c["b"] if c["a"] == body else c["a"]
                key = (float(c.get("penetration") or 0.0), float(c.get("impulse") or 0.0))
                if best is None or key > best[0]:
                    best = (key, other, s)
    if best is None:
        return None
    return {"partner": best[1], "penetration_m": best[0][0], "impulse": best[0][1], "substep": best[2]}


def body_penetration(contacts, body: str) -> float:
    deepest = 0.0
    for pair in contacts:
        if body in (pair.body_a, pair.body_b):
            deepest = max(deepest, float(pair.max_penetration))
    return deepest


def run(robotwin_root: str, segment: Path, condition: str, anchor_substep: int | None,
        anchor_body: str | None, report_path: Path | None = None) -> dict:
    from attribute_failures import recorded_outcome
    from replay_intervention import build_intervention
    from simuguard.adapters.robotwin import RoboTwinAdapter, make_task_env
    from simuguard.core import ControlLog, StateLog, SubstepMonitor
    from simuguard.integrations.robotwin_eval import eval_monitor_config, reapply_recorded_intervention
    from simuguard.presets import default_detectors

    summary = json.loads((segment / "summary.json").read_text())
    manifest = json.loads((segment / "manifest.json").read_text())
    meta = {**manifest["metadata"], **(summary.get("metadata") or {})}
    phase, recorded_success = recorded_outcome(summary)
    controls = ControlLog.load(segment / "controls.npz")
    recorded_states = StateLog.load(segment / "states.npz")

    kind, _, cap_text = condition.partition("@")
    if kind not in ("baseline", "broad", "body", "partner", "resolved"):
        raise SystemExit(f"unknown condition {condition}")
    cap = float(cap_text) if kind != "baseline" else None

    event = first_invalid_event(segment)
    t0 = anchor_substep if anchor_substep is not None else (event or {}).get("onset_substep")
    body = anchor_body or (event or {}).get("body")
    result: dict = {"segment": str(segment), "task": meta.get("task"), "seed": meta.get("seed"), "phase": phase,
                    "condition": condition, "kind": kind, "cap_mps": cap, "recorded_success": bool(recorded_success),
                    "anchor_substep": t0, "anchor_body": body,
                    "anchor_source": "argument" if anchor_substep is not None else "first_physics_invalid_event",
                    "first_invalid_event": event, "recorded_substeps": int(controls.newest_substep)}
    if kind != "baseline" and (t0 is None or body is None):
        raise SystemExit("no anchor: the episode has no physics-invalid event and none was given")

    env, _ = make_task_env(robotwin_root, meta["task"], int(meta["seed"]))
    try:
        adapter = RoboTwinAdapter(env)
        result["recorded_sim_intervention"] = reapply_recorded_intervention(adapter, meta)
        dt = adapter.timestep()
        lead = max(1, round(LEAD_S / dt))
        window = None
        broad_apply = None
        trigger = None
        if kind != "baseline":
            if body not in adapter._bodies:
                raise SystemExit(f"anchor body {body} is not in the scene")
            trigger = max(1, t0 - lead)
            if kind == "broad":
                description, broad_apply = build_intervention(f"depen_{cap}")
                result["description"] = description + ", from 0.1 s before the anchor (= paper depen_<cap>@event)"
            else:
                bodies = [body]
                rest = adapter.rest_offset(body)
                if kind == "partner":
                    partner = trace_partner(segment, body, t0)
                    result["partner"] = partner
                    if partner is None:
                        result["partner_status"] = "no_contact_in_trace"
                    elif adapter._rigid_body_component(partner["partner"]) is None:
                        result["partner_status"] = "partner_static"
                    else:
                        result["partner_status"] = "capped"
                        bodies.append(partner["partner"])
                rule = WindowRule(t0=t0, lead=lead, mode="resolved" if kind == "resolved" else "fixed",
                                  after=round(AFTER_S / dt), n_resolved=RESOLVED_N,
                                  threshold_m=rest + RESOLVED_TOL_M, max_after=round(RESOLVED_MAX_S / dt))
                result["window_rule"] = {"start": rule.start, "mode": rule.mode, "after": rule.after,
                                         "n_resolved": rule.n_resolved, "threshold_m": rule.threshold_m,
                                         "rest_offset_m": rest, "max_after": rule.max_after}
                window = DepenWindow(rule=rule, bodies=bodies, cap=cap)
                result["description"] = (f"max depenetration velocity {cap} m/s on {bodies} only, "
                                         f"{rule.mode} window from 0.1 s before the anchor; originals restored")
            result["trigger_substep"] = trigger

        gate = None
        try:
            gate = adapter.containment_gate()
        except Exception:  # noqa: BLE001
            pass
        monitor = SubstepMonitor(
            adapter, default_detectors(ejection_gate=gate), episode_id=f"narrow:{condition}",
            config=eval_monitor_config({"snapshot_interval_substeps": 0, "bundle_statuses": [], "save_episode_logs": False}),
        )
        monitor.attach()

        stop_on_success = kind != "baseline" and phase == "policy"
        first_success = None
        applied = 0
        started = time.time()
        for record in controls.between(0, controls.newest_substep):
            if record.substep == 0:
                continue
            if broad_apply is not None and record.substep == trigger:
                broad_apply(adapter)
                result["applied_at_substep"] = record.substep
            if window is not None:
                window.before_step(adapter, record.substep)
            adapter.apply_control(record)
            adapter.scene.step()
            applied += 1
            if window is not None and window.active:
                pen = body_penetration(adapter.read_contacts(), body) if window.rule.mode == "resolved" else None
                window.after_step(adapter, record.substep, pen)
            if first_success is None and env.check_success():
                first_success = record.substep
                if stop_on_success:
                    break
        if window is not None and window.active:  # the episode ended (or succeeded) inside the window
            window.restored_values = adapter.restore_max_depenetration_velocity(window.originals)
            window.ended_after = record.substep
            window.end_reason = "episode_ended_in_window"
        final_success = bool(env.check_success())
        summary_after = monitor.finalize()
        result.update({
            "substeps_replayed": applied,
            "wall_s": round(time.time() - started, 1),
            "first_success_substep": first_success,
            "success_at_end": final_success,
            "success": (first_success is not None) if phase == "policy" else final_success,
            "confirmed_events": summary_after["confirmed_count"],
        })
        if window is not None:
            result.update(window.report())
            result["values_now"] = adapter.max_depenetration_velocities(window.originals)
        last = result.get("window_last_substep") if window is not None else None
        events = [e for e in summary_after.get("events", []) if e.get("status") == "confirmed"]
        result["confirmed_events_detail"] = [
            {"onset_substep": e.get("onset_substep"), "detector": e.get("detector"), "bodies": e.get("bodies")}
            for e in events]

        live = recorded_states.array()
        replayed = monitor.state_log.array()[:, [monitor.state_log.body_ids.index(b) for b in recorded_states.body_ids], :]
        if trigger is not None:
            n = min(trigger - 1, len(live), len(replayed))
            result["prefix_substeps"] = n
            result["prefix_bit_identical"] = bool(np.array_equal(live[:n], replayed[:n], equal_nan=True))
            # first substep at which the replay leaves the recording (sanity: should be >= trigger)
            m = min(len(live), len(replayed))
            diff = np.nonzero(~np.all(np.isclose(live[:m], replayed[:m], rtol=0, atol=0, equal_nan=True), axis=(1, 2)))[0]
            result["first_divergent_row"] = int(diff[0]) if diff.size else None
        else:
            result["bit_identical"] = bool(live.shape == replayed.shape and np.array_equal(live, replayed, equal_nan=True))
            result["reproduces_recorded_outcome"] = bool(result["success"]) is bool(recorded_success)
            result["reproduces_recorded_events"] = summary_after["confirmed_count"] == summary["confirmed_count"]

        # frozen gravity stage on the replay's own events (diagnostic: is there still an artifact, and when?)
        if report_path is not None:
            result["replay_gravity_stage"] = _gravity_stage(report_path, monitor, summary_after, meta, last)
    finally:
        try:
            env.close_env()
        except Exception:  # noqa: BLE001
            pass
    return result


def _gravity_stage(report_path: Path, monitor, summary_after: dict, meta: dict, window_last: int | None) -> dict:
    try:
        from gravity_filter import classify_segment
        work = report_path.with_suffix(".replayseg")
        work.mkdir(parents=True, exist_ok=True)
        monitor.state_log.save(work / "states.npz")
        (work / "summary.json").write_text(json.dumps({"events": summary_after.get("events", []),
                                                         "metadata": {"phase": meta.get("phase"), "seed": meta.get("seed")}},
                                                        default=str))
        out = classify_segment(work, before=150, after=25, min_ratio=1.5, v_floor=0.05) or {"events": []}
        shutil.rmtree(work, ignore_errors=True)
        invalid = [e for e in out.get("events", []) if e.get("verdict") == "physics_invalid"]
        after = [e for e in invalid if window_last is not None and int(e["onset_substep"]) > window_last]
        return {"physics_invalid": [{"onset_substep": e["onset_substep"], "bodies": e.get("bodies"),
                                     "ratio": e.get("speed_over_free_fall")} for e in invalid],
                "n_physics_invalid": len(invalid), "n_after_window": len(after)}
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


# ---------------------------------------------------------------------------- batch
def batch(args) -> int:
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    jobs = [json.loads(line) for line in Path(args.jobs).read_text().splitlines() if line.strip()]
    gpus = args.gpus or [None]
    print(f"[narrow] {len(jobs)} jobs, workers={args.workers}, gpus={gpus}", flush=True)

    def work(indexed):
        index, job = indexed
        name = f"{job['key']}__{job['condition'].replace('@', '_')}"
        report = out / f"{name}.json"
        if report.exists():
            return
        env = dict(os.environ)
        gpu = gpus[index % len(gpus)]
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [args.python, "-B", str(HERE), "--robotwin-root", args.robotwin_root, "--segment-dir", job["segment"],
               "--condition", job["condition"], "--report", str(report)]
        if job.get("anchor_substep") is not None:
            cmd += ["--anchor-substep", str(job["anchor_substep"]), "--anchor-body", job["anchor_body"]]
        started = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        (out / f"{name}.log").write_text(proc.stdout + proc.stderr)
        status = "ok" if report.exists() else f"error {proc.returncode}"
        if report.exists():
            r = json.loads(report.read_text())
            r["job"] = job
            report.write_text(json.dumps(r, indent=1, default=str))
            status = f"success={r.get('success')} prefix={r.get('prefix_bit_identical', r.get('bit_identical'))}"
        print(f"[narrow] {time.strftime('%H:%M:%S')} {name} -> {status} ({time.time() - started:.0f}s)", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, enumerate(jobs)))
    print("[narrow] ALLDONE", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robotwin-root", required=True)
    ap.add_argument("--segment-dir")
    ap.add_argument("--condition")
    ap.add_argument("--report")
    ap.add_argument("--anchor-substep", type=int)
    ap.add_argument("--anchor-body")
    ap.add_argument("--jobs", help="JSON lines: key, segment, condition[, anchor_substep, anchor_body]")
    ap.add_argument("--output-dir")
    ap.add_argument("--gpus", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    if args.jobs:
        return batch(args)
    report = Path(args.report).resolve()
    result = run(args.robotwin_root, Path(args.segment_dir).resolve(), args.condition, args.anchor_substep,
                 args.anchor_body, report)
    report.write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({k: result.get(k) for k in ("condition", "success", "prefix_bit_identical", "bit_identical",
                                                  "restored_exactly", "window_last_substep", "wall_s")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
