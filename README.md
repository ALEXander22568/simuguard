# SimuGuard

A **side-channel audit tool for simulated policy evaluation**: it records
physics-substep ground truth while a benchmark runs unmodified, flags suspicious
contact events, and can **replay any episode bit-exactly** so a failure can be
attributed to the policy or to the simulator.

Standard success rates cannot separate the two.  A 10 g can that reaches
14.5 m/s inside a basket (1.05 J, the speed of a 10.7 m free fall) while the
robot links never exceed 0.3 m/s is not something the policy did - but it
changes the score all the same.

Status `0.1.0.dev0`: core plus one validated adapter (RoboTwin 2.0 / SAPIEN 3).

## What it supports

### 1. Per-substep monitoring, without changing the benchmark
Hooks the simulator's `step` and records, every substep (4 ms on RoboTwin):
poses, velocities, mass and role of every body; and every contact *pair* with
impulse, force estimate, penetration depth and point count (simulators usually
report per collision *shape* pair, so pairs are merged per body).
Verified: with monitoring on vs off, all 59 bodies follow bit-identical
trajectories.  Cost: ~7-12 ms per substep on top of ~0.5 ms of physics.

### 2. Anomaly detection
Five detectors, all thresholds configurable, every event carrying its metrics
and a `review: pending` marker:
* **contact ejection** - candidate -> confirmed/rejected lifecycle, two
  confirmation paths (ballistic free flight, or violent displacement while the
  contact manifold persists), with an optional containment gate (is the object
  fully inside the container?);
* deep penetration, impulse spike, non-finite/absurd state, and speed above
  what the robot links could have imparted.

### 3. Exact replay
Rebuild the environment from the episode's seed and replay the recorded
per-substep actuation.  Verified at **0.0 m error** over complete episodes
(139,613 and 53,745 substeps), across processes.
This also rules out an approach: restoring a mid-episode snapshot (PhysX
pack/unpack or public state) is *not* exact once bodies touch - mm to cm drift.

### 4. Intervention experiments
Replay an event with exactly one simulation setting changed (solver iterations,
object mass; collision geometry not yet).  Example from
`place_can_basket` seed 100000: baseline reproduces the recorded 14.51 m/s
bit-exactly, raising solver iterations to 32/8 drops it to 4.73 m/s, and a
10 g -> 100 g target mass drops it to 2.82 m/s.

### 5. Reproducible artefacts
Per episode: `manifest.json` (bodies, roles, detector config, task ground truth,
commits), `states.npz` (float64 state per substep), `controls.npz` (actuation
per substep, ~85-105 B/substep, the exact-replay input), `trace.jsonl.gz`
(contact time series), `events.jsonl`, `bundles/*.json.gz` (self-contained
replay bundle per confirmed event) and `summary.json`.

### RoboTwin specifics
The official `eval_policy_xpolicylab.py` runs unmodified; both the official
expert check and the scored policy rollout are monitored.  LingBot-VA runs
through its end-effector action path, with the server config, action conversion
and 24 GB memory strategy applied as runtime shims (no upstream file is edited)
and recorded in every run's provenance.  `place_can_basket` has roles and a
containment gate configured; other tasks run with a generic fallback until a
few lines of `TaskSpec` are added.

## Not supported yet
* collision-geometry (convex decomposition) interventions - they need the scene
  rebuilt with different meshes;
* batch evaluation (`eval_batch=true`);
* a human review UI (SimuGuard produces the review queue, not the interface);
* other benchmarks - ManiSkill / LIBERO / RoboCasa need one `SimAdapter` each;
* calibrated detector thresholds: **current detector output is a candidate list
  for human verification, not ground truth**.

## Requirements
Simulation and monitoring must share one process.  Exact replay assumes the same
machine, the same simulator build and single-process CPU physics.  A new
benchmark needs an adapter implementing `simuguard.core.adapter.SimAdapter`.

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

## Validation

### RoboTwin adapter (2026-09-17, 4090-hexa-node2)

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

### LingBot-VA through the official evaluator (2026-09-18, 4090-hexa-node1)

Clean RoboTwin `6dde571`, `place_can_basket` seed 100000, end-effector action
path, upstream attention window, policy on one GPU and simulation on another:

| check | result |
|---|---|
| official evaluation | ran to completion (`exit 0`), expert check passed, policy scored 0/1 |
| monitoring | 3,279 expert + 53,745 policy substeps, 0 monitor errors |
| detector output | 7 confirmed ejections in the *successful* expert segment, 22 in the policy segment |
| ground truth | target can peaks at **14.51 m/s** (expert) and 13.40 m/s (policy); basket never exceeds 0.31 m/s |
| exact replay of one event | baseline reproduces the recorded trajectory at **0.0 m** error and the detector fires again |
| intervention, solver iterations 32/8 | peak target speed 14.51 -> **4.73 m/s** |
| intervention, target mass 10 g -> 100 g | peak target speed 14.51 -> **2.82 m/s** |

The action path matters: driving the 30-dim joint channels of
`robotwin30_train` (whose de-normalisation statistics do not match this
checkpoint) commands joint targets that jump up to 1.337 rad between policy
steps with a 0.234 direction-reversal rate - visible as arm/gripper jitter.
The end-effector path lowers this to 0.660 rad and 0.144.  Both measured from
`controls.npz`; the expert planner commands 0.00022 rad per substep with no
reversals.

Reports: `runs/ee_upstream_20260918T105408Z/`, `runs/intervention_*/`.

## Tests

```bash
PYTHONPATH=.:tests python -m unittest discover -s tests
# reproduce one event, then repeat it with a single setting changed
python scripts/robotwin/replay_intervention.py --robotwin-root /path/to/RoboTwin \
    --bundle runs/<run>/simuguard/segments/<seg>/bundles/<event>.json.gz \
    --report out.json --interventions baseline solver_high mass_100g
CUDA_VISIBLE_DEVICES=<sim gpu> PYTHONPATH=. python scripts/check_robotwin_integration.py \
    --robotwin-root /path/to/RoboTwin --task place_can_basket --seed 100000 --output-dir runs/integration
```

See `docs/ARCHITECTURE.md` for data flow, timing semantics and limitations.
