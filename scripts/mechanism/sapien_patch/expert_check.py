"""RoboTwin's own scripted demonstration (the expert check of the official evaluator) on a range of seeds,
monitored by SimuGuard.  Run once with the original engine and once with the patched copy of SAPIEN first on
PYTHONPATH; the task, seeds and planner are the same, only the PhysX GJK threshold differs.
usage: expert_check.py TASK SEED_START N OUT.jsonl
"""
import sys, json, time, traceback
import numpy as np
W = "/mnt/nvme0/shared/USER/simuguard"
sys.path.insert(0, W + "/SimuGuard"); sys.path.insert(0, W + "/SimuGuard/scripts/robotwin")
task, start, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
import sapien
from simuguard.adapters.robotwin import RoboTwinAdapter, make_task_env
from simuguard.core import SubstepMonitor
from simuguard.core.types import BodyRole
from simuguard.integrations.robotwin_eval import eval_monitor_config
from simuguard.presets import default_detectors
lib = [l.split()[-1] for l in open("/proc/self/maps") if "libsapien.so" in l][:1]
print("sapien", sapien.__file__, lib, flush=True)
for seed in range(start, start + n):
    row = {"task": task, "seed": seed, "lib": lib[0] if lib else None}
    t0 = time.time(); env = None
    try:
        env, _ = make_task_env(W + "/RoboTwin", task, seed)
        adapter = RoboTwinAdapter(env)
        targets = adapter.body_ids_with_role(BodyRole.TARGET)
        gate = None
        try: gate = adapter.containment_gate()
        except Exception: pass
        monitor = SubstepMonitor(adapter, default_detectors(ejection_gate=gate), episode_id=f"expert:{seed}",
                                 config=eval_monitor_config({"snapshot_interval_substeps": 0, "bundle_statuses": [], "save_episode_logs": False}))
        monitor.attach()
        env.play_once()
        row["plan_success"] = bool(env.plan_success); row["check_success"] = bool(env.check_success())
        summary = monitor.finalize()
        row["confirmed_events"] = int(summary["confirmed_count"]); row["substeps"] = len(monitor.state_log)
        cols = [monitor.state_log.body_ids.index(t) for t in targets]
        sp = np.linalg.norm(monitor.state_log.array()[:, cols, 7:10], axis=2)
        row["peak_target_speed_mps"] = float(np.nanmax(sp)) if sp.size else None
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"; traceback.print_exc(limit=3)
    finally:
        try:
            if env is not None: env.close_env()
        except Exception: pass
    row["wall_s"] = round(time.time() - t0, 1)
    with open(out, "a") as f: f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
