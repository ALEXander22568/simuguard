# SimuGuard architecture

## Goal

Separate policy failures from simulator-induced failures in simulation-based
VLA evaluation:

1. **Detect** suspicious contact events at physics-substep granularity from
   simulator ground truth.
2. **Verify** them: route flagged episodes to human review and, independently,
   reproduce them by deterministic replay from state saved inside the same rollout.
3. **Analyze** them: replay under interventions (collision decomposition,
   solver iterations, mass, ...) and measure whether the event disappears.

## Layers

```
 benchmark env (RoboTwin Base_Task, unmodified)
        │  scene.step()   ← instance-attribute hook (fallback: scene proxy)
        ▼
 ┌──────────────────────────── adapters/robotwin ────────────────────────────┐
 │ env.py        official arg loader + class_decorator (no re-implementation) │
 │ adapter.py    RoboTwinAdapter(SimAdapter): inventory, states, contacts,    │
 │               hook, control, snapshot/restore, interventions               │
 │ sapien_io.py  SAPIEN 3 I/O (PhysX components, contacts, pack/unpack)       │
 │ tasks.py      TaskSpec roles (target/container), ContainmentGate           │
 └──────────────────────────────────────┬─────────────────────────────────────┘
                                        │ SimAdapter contract (core/adapter.py)
 ┌──────────────────────────────── core (numpy only) ─────────────────────────┐
 │ types.py      BodyInfo/BodyState/ContactPair/SubstepFrame                  │
 │ monitor.py    SubstepMonitor: per substep read frame → control log →       │
 │               periodic snapshot → detectors → events → recorder → bundles  │
 │ detectors/    ContactEjection, Penetration, ImpulseSpike, NonFinite,       │
 │               ActuationBound (experimental plausibility bound)             │
 │ snapshot.py   Snapshot, SnapshotRing, ControlLog, ReplayBundle             │
 │ replay.py     replay_bundle(): restore → apply controls → step → fidelity  │
 │ recorder.py   manifest / trace / events / bundles / summary on disk        │
 └────────────────────────────────────────────────────────────────────────────┘
```

The core never imports a simulator.  Supporting another benchmark means writing one `SimAdapter`
(ManiSkill); MuJoCo benchmarks (RoboCasa, LIBERO: `docs/libero.md`) share `adapters/mujoco`.

## Per-substep data flow

```
scene.step() ─► hook ─► monitor._on_substep()
                         ├─ adapter.read_frame(tracked bodies, contacts)      ground truth
                         ├─ (pre-step hook) adapter.capture_control() → ControlLog  drive targets, qf
                         ├─ every N substeps: adapter.capture_snapshot()      public + PhysX pack
                         ├─ detectors.observe(frame) → Event(status)          candidate/confirmed/rejected/flag
                         ├─ recorder.frame / recorder.event
                         └─ confirmed event + post window elapsed → ReplayBundle
```

All monitor work runs inside `_guard`: exceptions are counted and recorded,
never propagated, so instrumentation cannot change evaluation outcomes.

## Timing semantics

* A frame at substep `k` is the state **after** physics step `k`.
* A `ControlRecord` at substep `k` holds the drive targets / `qf` in effect
  **during** step `k`; it is captured by the *pre-step* hook because RoboTwin
  recomputes gravity-compensation `qf` from the current state right before
  every `scene.step()` (capturing after the step broke replay fidelity).
* A snapshot at substep `s` is the state after step `s`; replaying a bundle
  restores it and then, for `k = s+1..e`, applies control `k` and steps once.

## Roles

Detectors reason about roles rather than names:
`TARGET`, `CONTAINER`, `OBJECT`, `ROBOT`, `SCENE`.  RoboTwin roles come from a
`TaskSpec` (`place_can_basket`: target=`env.can`, container=`env.basket`);
robot links are articulations referenced by `env.robot`; static actors and
`table`/`wall`/`ground` are scene.  Unknown tasks fall back to `OBJECT` for all
free bodies and can be configured with `get_task_spec(name, overrides)`.

## Detectors

| detector | kind | lifecycle | purpose |
|---|---|---|---|
| `contact_ejection` | contact_triggered_ejection | candidate → confirmed/rejected | port of TwinGuard phase-1 logic (ballistic free flight or violent risky contact displacement), optional containment gate |
| `deep_penetration` | deep_penetration | flag | contact separation below −5 mm |
| `impulse_spike` | impulse_spike | flag | \|Σ impulse\|/dt above 200 N on a pair with a free body |
| `non_finite_state` | non_finite_or_absurd_state | flag | NaN/Inf or >100 m/s |
| `actuation_bound` | speed_exceeds_actuation_bound | flag | experimental: object speed ≫ fastest robot link after non-robot contact |

Thresholds are dataclass configs (`configs/detectors_default.json`) and are
written into every episode manifest.

## Replay and fidelity

Re-running a *policy* from the same env seed does not reproduce contact events
(TwinGuard fidelity audits v1–v4), because policy inference is not replayed.
SimuGuard replays the recorded *actuation* instead.  Two bundle modes:

| mode | start state | restore | fidelity on RoboTwin (place_can_basket, seed 100000) |
|---|---|---|---|
| `episode_start` (default) | initial snapshot at attach | rebuild env with same seed, `method="none"` | **exact**: rebuilt initial state error 0.0; 1750 replayed substeps, max position error 0.0 m |
| `snapshot` | latest periodic snapshot before onset | in place, `native` (PhysX pack/unpack) or `public` | pre-contact ~1e-6 m; in contact 1 mm – 2 cm within 250 substeps; two replays from the same snapshot already differ by 1.7e-4 m |

PhysX `pack()/unpack()` does not restore solver warm-start/friction-anchor
state, so in-place restore is only approximate once bodies are in contact.
`episode_start` costs re-simulating from the beginning (tens of seconds per
episode) but is the only mode suitable as causal evidence.

`replay_bundle()` reports per-body max position error, the first substep
exceeding tolerance, and whether detectors fire again.  Only a bundle whose
unchanged-settings replay reproduces the event may be used for intervention
analysis.

## Compact episode logs

* `ControlLog` keeps the `ControlRecord` API but stores one template per payload
  structure plus a flat numeric row per substep; float32 is used only while every
  value round-trips exactly, otherwise the log is promoted to float64.  Measured on
  aloha-agilex: 4,032 JSON bytes -> 114 numbers per substep; `controls.npz` ≈ 85–105
  bytes per substep after compression.  Rebuilt replays from float32 logs were
  bit-identical to the live runs.
* `StateLog` stores float64 pose/velocity of non-robot tracked bodies every substep
  (`states.npz`) as the exact-replay reference.
* Replay bundles embed their controls in the same compact encoding
  (`controls_encoding = control_log_npz_b64`); legacy JSON lists still load.

## Official evaluator integration

`simuguard.integrations.robotwin_eval` imports the official evaluator, wraps
`class_decorator`, and installs instance-level hooks on the task env:
`setup_demo` (after) starts a segment, `play_once` marks it `expert`,
`close_env` (before) finalizes it.  Monitoring failures are recorded in
`run_summary.json` and never raised into the evaluator.

## Known limitations

* RoboTwin's Python planner state (TOPP trajectories, gripper schedules) is not
  part of a snapshot; replay drives recorded joint targets, not the planner.
* Decomposition interventions need the env rebuilt with alternative collision
  meshes; replaying recorded actuation there is well defined, but exactness can
  only be verified for the unchanged configuration.
* Exact replay relies on deterministic single-process CPU PhysX and identical
  RoboTwin/SAPIEN versions; bundles record the RoboTwin commit for this reason.
* SAPIEN reports contacts per collision-shape pair; the adapter merges them per
  body pair.  Net impulses of different hulls can cancel (e.g. a squeezing
  grasp), so both net and absolute impulse sums are reported.
* Contact force is estimated as impulse/dt (PhysX reports impulses).
* Monitoring overhead is currently ~7–12 ms per substep (physics alone 0.5–0.8 ms).
* Episode-start bundles embed the whole episode's controls, duplicating
  `controls.npz` for every confirmed event.
