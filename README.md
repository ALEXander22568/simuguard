# SimuGuard

Detect, verify and analyze **contact artifacts** in simulation-based evaluation
of vision-language-action (VLA) policies.

Standard benchmark success rates cannot tell a policy error from a failure
induced by the simulator (e.g. an object ejected from a container by a
non-physical contact impulse). SimuGuard instruments the benchmark at physics
**substep** granularity from simulator ground truth, flags suspicious contact
events, routes them to human verification, and records in-rollout snapshots so
events can be replayed deterministically under different simulation settings
(collision decomposition, solver iterations, object mass).

Status: `0.1.0.dev0` skeleton. RoboTwin 2.0 (SAPIEN 3) adapter implemented and
exercised on an unmodified upstream checkout; see *Validation* below.

## Layout

```
simuguard/
  core/                   simulator-agnostic, numpy only
    types.py              BodyInfo, BodyState, ContactPair, SubstepFrame
    adapter.py            SimAdapter contract (ground truth, hooks, snapshots)
    monitor.py            SubstepMonitor
    detectors/            ejection, penetration, impulse spike, non-finite, actuation bound
    snapshot.py           Snapshot, SnapshotRing, ControlLog, ReplayBundle
    replay.py             replay_bundle() + fidelity metrics
    recorder.py           on-disk episode artefacts
  adapters/robotwin/      RoboTwin 2.0 / SAPIEN 3 adapter (lazy SAPIEN import)
    env.py                official evaluator arg loading (no re-implementation)
    adapter.py            RoboTwinAdapter
    sapien_io.py          PhysX components, contacts, pack/unpack, control
    tasks.py              TaskSpec roles, ContainmentGate
  presets.py              default detector suite
configs/detectors_default.json
scripts/check_robotwin_integration.py
tests/                    unittest suite with a deterministic toy adapter
docs/ARCHITECTURE.md
```

## Quick start (RoboTwin)

```python
from simuguard import MonitorConfig, SubstepMonitor, EpisodeRecorder, default_detectors
from simuguard.adapters.robotwin import RoboTwinAdapter, make_task_env

env, _ = make_task_env("/path/to/RoboTwin", "place_can_basket", seed=100000)
adapter = RoboTwinAdapter(env)                       # roles: can=target, basket=container
monitor = SubstepMonitor(
    adapter,
    default_detectors(ejection_gate=adapter.containment_gate()),
    episode_id="place_can_basket-100000",
    config=MonitorConfig(snapshot_interval_substeps=100),
    recorder=EpisodeRecorder("runs/place_can_basket-100000"),
)
monitor.attach()
# ... run the policy / evaluator loop unchanged (env.take_action(...)) ...
summary = monitor.finalize()          # events, bundles, needs_human_review
```

Replay a confirmed event exactly (rebuilt env), then under an intervention:

```python
from simuguard import ReplayBundle
from simuguard.adapters.robotwin import replay_on_rebuilt_env
bundle = ReplayBundle.load(summary["bundles"][event_id])       # episode_start bundle
baseline = replay_on_rebuilt_env(root, "place_can_basket", 100000, bundle, detectors=default_detectors())
assert baseline.within_tolerance and baseline.confirmed_events  # fidelity gate
treated = replay_on_rebuilt_env(root, "place_can_basket", 100000, bundle, detectors=default_detectors(),
                                before_replay=lambda a: a.set_solver_iterations(position=32, velocity=8))
```

## Official evaluator integration (single process)

```bash
python -m simuguard.integrations.robotwin_eval \
    --robotwin-root /path/to/RoboTwin --simuguard-out runs/eval_place_can_basket \
    -- --bench_name RoboTwin --task_name place_can_basket --env_cfg_type aloha_agilex \
       --policy_name LingBot_VA --host 127.0.0.1 --port <bridge port> --protocol ws \
       --eval_batch false --seed 0 --test_num 10 --instruction_type seen --expert_check true
```

The official `eval_policy_xpolicylab.py` runs unmodified; the returned task env
gets instance-level `setup_demo / play_once / close_env` hooks.  Each
`setup_demo .. close_env` span is a *segment*; the official expert check
(`play_once`) yields an `expert` segment and the scored rollout a `policy`
segment.  Output: `segments.jsonl`, `run_summary.json`, and per segment
`manifest.json, summary.json, trace.jsonl.gz, events.jsonl, controls.npz,
states.npz, initial_snapshot.json.gz, bundles/`.  `--simuguard-disable` runs the
identical path without monitors; `eval_batch=true` is not supported yet.

## Validation (2026-09-17, 4090-hexa-node2)

Unmodified RoboTwin `6dde571` + XPolicyLab `fa431ec`, task `place_can_basket`,
seed 100000, scripted expert `play_once()` (no policy server), SAPIEN on one GPU:

| check | result |
|---|---|
| substep hook | instance attribute on `scene.step`; 3280 substeps observed, 0 monitor errors |
| expert run under monitoring | `plan_success` and `check_success` both true (not a transparency proof; paired on/off comparison still TODO) |
| inventory / roles | 59 bodies: target `actor:071_can`, container `actor:110_basket`, 54 robot links, 3 scene |
| contacts | merged per body pair (SAPIEN reports per shape pair) |
| exact replay (`episode_start`) | rebuilt env initial state error 0.0; 1750 replayed substeps, max position error **0.0 m** (reproduced in 2 independent runs, 2nd on commit a87baea) |
| in-place snapshot replay | pre-contact <=1.1e-6 m; in contact 0.4-20 mm over 250 substeps; two replays from the same snapshot differ by 1.7e-4-2.1e-3 m (approximate only) |

### Evaluator wrapper (official call sequence, no policy server)

2 seeds x (expert `play_once` segment + scripted 40-action `take_action` policy segment),
each segment rebuilt and replayed in a fresh process:

| check | result |
|---|---|
| lifecycle | 4/4 segments, phases expert/policy/expert/policy, all monitored, hooks removed, 0 errors |
| monitoring on vs off (same controls) | **bit-identical** for all 59 bodies in 4/4 segments |
| rebuilt (new process) vs live recording | **bit-identical** (max abs diff 0.0) in 4/4 segments; float32 control log |
| overhead, physics loop | bare 0.45–0.82 ms/substep; monitor adds 7.5–11.6 ms/substep |
| overhead, live wall time | expert segment 41–49 s vs 12 s; scripted policy segment 22 s vs 3.3 s |
| artefacts | 108 B/substep (quiet policy segment) to 322–851 B/substep (contact-rich expert, incl. bundles); `controls.npz` 85–105 B/substep |

Overhead is not yet optimized (per-substep Python reads of 54 robot links, contact
merging, template encoding).  Reports: `runs/wrapper_validation_20260917T153902Z/`.

Reports: `runs/simuguard_integration_20260917T131127Z/`, `runs/simuguard_integration_20260917T131551Z/` (`integration_report.json`)
(workspace `/mnt/nvme0/twinguar/simuguard`).

## Tests

```bash
PYTHONPATH=.:tests python -m unittest discover -s tests
CUDA_VISIBLE_DEVICES=<sim gpu> PYTHONPATH=. python scripts/check_robotwin_integration.py \
    --robotwin-root /path/to/RoboTwin --task place_can_basket --seed 100000 --output-dir runs/integration
```

See `docs/ARCHITECTURE.md` for data flow, timing semantics and limitations.
