# SimuGuard on LIBERO

LIBERO is BDDL task suites on robosuite + MuJoCo, so it runs on the MuJoCo adapter
(`simuguard/adapters/mujoco`) plus a thin LIBERO layer (`simuguard/adapters/mujoco/libero.py`).
Everything below was measured on h800-2 on 2026-09-25 (times in KST).

## Stacks

| | official (Cap-bench `libero_runtime.cpu.env`) | rpent-venv (`libero_runtime.rpentvenv.cpu.env`) |
|---|---|---|
| python / robosuite / MuJoCo | 3.10 / **1.4.0** / 3.2.3 | 3.11 / **1.5.2** / 3.3.0 |
| LIBERO | upstream 8f1084e | RPent fork `rpent_libero` 0.2.0 |
| physics loop per substep | `sim.forward(); _pre_action(); sim.step()` (OSC calls `sim.forward()` again) | `lite_physics=True`: `sim.step1(); _pre_action(); sim.step2()` |
| SimuGuard hook (`attach_robosuite`, unchanged) | `sim.step` | `sim.step2` |
| replay needs | recorded warm start (`qacc_warmstart`) every substep | mocap targets (robosuite 1.5 eef targets), recorded fixture placement |
| fixture placement from `env.seed` | reproduced | **not** reproduced (per-env `default_rng(seed=None)`) |

Both: dt 2 ms, control 20 Hz (25 substeps per action), Euler, elliptic cones, Newton solver, 100
iterations, impratio 20.  Object geoms: solref (0.001, 1) (clamped by MuJoCo to 2 dt = 4 ms),
solimp (0.998, 0.998, 0.001); objects weigh 5-30 g.  Rendering: Mesa software EGL (no GPU).
The campaign below uses the official stack (upstream LIBERO); the rpent stack is verified for
attach and replay only.

## Attaching

```python
from simuguard.adapters.mujoco.libero import attach_libero, make_env
from simuguard.core.monitor import MonitorConfig, SubstepMonitor
from simuguard.core.recorder import EpisodeRecorder
from simuguard.presets import default_detectors

env, obs, meta = make_env("libero_spatial", task_id=0, init_index=3, seed=3)   # seed, reset, set_init_state
adapter, info = attach_libero(env, task_name=meta["task"])                     # after every reset
monitor = SubstepMonitor(adapter, default_detectors({"ejection": {"delta_v_reference_timestep_s": 0.004}}),
                         episode_id="...", recorder=EpisodeRecorder(seg),
                         config=MonitorConfig(snapshot_interval_substeps=250, snapshot_capacity=2000,
                                              save_snapshots=True), metadata=meta)
monitor.attach()
# ... the evaluation loop, unchanged: env.step(action) ...
summary = monitor.finalize()
```

Roles come from the BDDL goal (`parsed_problem["goal_state"]`): the first argument of a binary
predicate is a **target**, the object or fixture owning the second argument (regions resolve to
their parent) is a **container** together with every body below it (drawers, microwave door);
table regions leave the table as scenery.  Other free objects are **objects** (still monitored),
robot/gripper/mount bodies **robot**.  E.g. `[["on","akita_black_bowl_1","plate_1"]]` → bowl
target, plate container; `[["in","white_yellow_mug_1","microwave_1_heating_region"],["close","microwave_1"]]`
→ mug target, microwave body + door containers.

The manifest's task ground truth carries the stack (`robosuite`, `mujoco`, `lite_physics`,
hooked method), the roles and a model fingerprint (SHA-256 of the compiled model + exact body
poses).

## What was verified

**Inventory / roles** (libero_spatial task 0, both stacks): 38 bodies (40 with robosuite 1.5's two
eef mocap targets), 5 free bodies: bowl_1 target, plate container, bowl_2 / cookies / ramekin
objects; cabinet drawers and stove button are articulated links.

**Contacts.** After LIBERO's 15 settle steps, the net contact force on every free object divided
by its weight is **1.0000** for all five objects on both stacks (horizontal part 0.0000), i.e. the
per-contact impulses (`mj_contactForce`, oriented body_a → body_b) add up to the weights of
5.6-11.5 g objects.

**Detectors.** 0 monitor errors in every run (probe, campaign, audit).  At episode start most
scenes drop their objects (LIBERO-Spatial task 0: from 7.2 cm, 1.1 m/s at impact = g·0.114 s),
which `actuation_bound` flags and the ejection detector rejects.

**Overhead** (official stack, 256×256 agentview + wrist rendered on CPU every step; probe):
0.302 s per env step unmonitored, 0.362 s with the monitor (+2.4 ms per substep), 0.372 s with
the campaign configuration (recorder writing trace/states/controls/events + snapshots every
250 substeps: +2.8 ms per substep, +23 %); finalize 0.05 s.

**Exact replay.**  `scripts/libero/check_replay.py SEGMENT` rebuilds the episode in a fresh
process from (suite, task, init, seed), compares the model fingerprint and the initial state
(every `mjSTATE_INTEGRATION` component), restores the recorded initial snapshot, replays the
recorded actuation of the whole episode and compares every logged object at every substep and the
complete integration state (robot included) at every archived snapshot (one per 250 substeps).

* official stack, Pi0.5 episode (libero_spatial t0, 2225 substeps): model identical, initial
  state identical, **max position error 0.0 m**, full state identical at 8/8 snapshots.
  Without the recorded warm starts: first divergence at substep 59 (the first table impact,
  1.1e-16 m), 7.2e-8 m by the end, and the full integration state already differs at the first
  archived snapshot (substep 250; up to 1.5e-4).
* robosuite 1.5.2 stack, Pi0.5 episode (libero_goal t3, 4500 substeps): rebuilt model differs
  (a fixture 8 mm away: placement comes from an unseeded generator); after writing the recorded
  body poses back the model hash matches and the replay is **0.0 m**, full state identical at
  18/18 snapshots.
* **every campaign episode** (`replay_batch.py --all`, one fresh process each): **400/400
  bit-exact**, 1,880,350 substeps, max position error 0.0 m, full integration state identical at
  all 7,334 archived snapshots, model hash identical 400/400.  The same replays with the recorded
  warm starts dropped (33 episodes: failures, policy-phase events, 3 successes per suite): 0/33
  exact, median first divergence at substep 48, final position error median 0.11 m, worst 1.46 m.
* in-process (probe, scripted reach, 1500 substeps): 0.0 m on both stacks.

Two fixes to the shared MuJoCo adapter came out of this (they apply to RoboCasa as well):
1. **Warm start.** `mj_forward` rewrites `qacc_warmstart`; robosuite 1.4 calls it twice before
   every `mj_step`, so the adapter now records the warm start whenever it changed since the last
   step and writes it back on replay (43 floats per substep on LIBERO; controls.npz ~0.36 KB per
   substep).  robosuite ≥ 1.5 with `lite_physics` never changes it between steps (no records).
2. **Centre-of-mass velocity.** `mj_objectVelocity(mjOBJ_BODY)` already returns the COM velocity;
   the adapter added `ω × (xipos - xpos)` on top, over-reporting spinning bodies with an offset
   COM (test: 1.5 m/s reported for 0.9 m/s).  RoboCasa numbers recorded before this fix carry the
   same bias for spinning objects.

## Evaluation: Pi0.5 on LIBERO under SimuGuard

Campaign `pi05_v1` (h800-2, 2026-09-25 16:36-17:39 KST, official stack): all 40 tasks of
LIBERO-Spatial / Object / Goal / 10, init states 0-9 of each (seed = init index), 400 episodes.

* Policy: Pi0.5, checkpoint `RLinf-Pi05-LIBERO-130-fullshot-SFT` (the team's LIBERO VLA), served by
  RPent's `pi05_vla_server.py` (RLinf openpi model, 5 actions of 7 per query, ~0.14 s per query on
  an idle H800), three servers on GPUs 0-2 at the end.  `scripts/libero/pi05_rollout.py` follows
  RLinf's LIBERO evaluation, which this checkpoint was trained and scored with: 256×256 agentview
  and wrist images rotated 180°, state = eef position + robosuite axis-angle + gripper qpos, the
  task's language, all 5 actions executed, 15 settle steps (zero motion, gripper open) before the
  policy, stop at the first success, cap 600 policy steps (LIBERO's own evaluation limit).
* SimuGuard: default detectors (ejection |dv| thresholds as accelerations,
  `delta_v_reference_timestep_s = 0.004`), snapshots every 250 substeps archived, full trace;
  attached right after `set_init_state`, so the settle steps are monitored too.
* Three stages: detector (confirmed contact ejection) → gravity stage (`gravity_filter.py`,
  windows 0.6 s / 0.1 s = 300 / 50 substeps, ratio > 1.5) → carrier stage (`carrier_filter.py`,
  0.04 s / 0.1 s, ratio ≥ 1.5).
* 0 infrastructure errors and 0 monitor errors in the 400 recorded episodes.  (A first launch with
  two policy servers on a GPU saturated by other users' Habitat renderers timed out in 26
  episodes; they were discarded (`requeue.py`) and re-run after the client got retries and more
  servers.  Slow servers only cost wall time: the simulation waits for the policy.)

Episodes with events after each stage, split into settle phase (before the policy acts) / policy
phase; an episode can have both:

| suite | episodes | successes | detected | after gravity | physics-invalid (after carrier) |
|---|---|---|---|---|---|
| LIBERO-Spatial | 100 | 99 | 0 | 0 | **0** |
| LIBERO-Object | 100 | 97 | 100 (100 / 1) | 100 (100 / 1) | **100 (100 / 1)** |
| LIBERO-Goal | 100 | 98 | 0 | 0 | **0** |
| LIBERO-10 | 100 | 87 | 31 (30 / 2) | 31 (30 / 1) | **31 (30 / 1)** |
| total | 400 | 381 (95.2 %) | 131 (130 / 3) | 131 (130 / 2) | **131 (130 / 2)** |

* **Settle phase (130 episodes):** the init-state launches described below, in every episode of
  LIBERO-Object and of LIBERO-10 tasks 0, 1, 7 (exactly the audit's prediction for init states
  0-9).  They happen before the policy acts.  5 of these 130 episodes failed (96.2 % success,
  against 94.8 % in the other 270), so with this policy there is no sign they cost successes;
  they do change the scene every such episode starts from.
* **Policy phase:** 3 episodes with a confirmed event, 2 left after the three stages:
  * LIBERO-10 task 9 (mug into the microwave) ep 5, **failed**: while the gripper holds the mug at
    0.15 m/s, a contact between the mug's collision box `white_yellow_mug_1_g9` and the
    microwave's 2 cm plate `microwave_1_g6` appears in one substep with a depth of −242.8 mm and
    412 N normal force; MuJoCo's own `mj_geomDistance` gives +1.4 mm for the same two geoms at the
    same state, i.e. the box-box contact is spurious.  The 31 g mug leaves the grip at 1.57 m/s
    (arm links ≤ 0.30 m/s), flies 9 substeps, hits the microwave.  Found by replaying substeps
    3251-3469 from the archived snapshot (0.0 m error) with `scripts/libero/replay_inspect.py`.
    Whether this event caused the failure is not established (it needs a counterfactual replay).
  * LIBERO-Object task 4 (ketchup) ep 9, succeeded: during release one finger re-contacts
    (22 N), the bottle's speed jumps 0.25 → 0.62 m/s for one substep and is back at 0.25 m/s two
    substeps later; the confirmation comes from the ordinary 4 cm drop into the basket that
    follows.  The stages keep it because the carrier stage does not examine direct robot contact
    (the RoboCasa campaign had the same gap); on inspection it is a benign one-substep contact
    transient, not an ejection.
  * LIBERO-10 task 1 ep 1: cream cheese released 18 cm above the basket, gravity-explained
    (ratio 0.83).
* Flags in the policy phase (no confirmation logic, context only): 371 `deep_penetration`
  (> 5 mm; fingers pressing into the thin plate and grasped boxes, the mug inside the microwave),
  55 `actuation_bound` (objects released and falling faster than the arm moves), 42
  `impulse_spike` (> 200 N; mostly the jammed mug in LIBERO-10 task 9, the wine bottle against the
  cabinet in LIBERO-Goal task 2), in 108 episodes.
* Cost: 1.88 M substeps, 69 k policy steps; artefacts 1.8 GB (controls 355 B/substep, states
  304 B, trace 122 B, snapshots 7 MB in total, videos 126 MB).  Wall time per episode 50-300 s
  with 16 workers sharing the host (CPU rendering dominates).

Videos: `<run>/<suite>/tNN/segments/epKK_seedK/video.mp4` (agentview | wrist, upright, 20 fps,
settle steps included).  Frames checked: the smoke episode (grasp and place), LIBERO-Goal t3 and
Spatial t5 (drawer, bowl), LIBERO-Object t0 and LIBERO-10 t0 starts (the launches).

Per task (campaign and init-state audit side by side): `docs/libero_pi05_v1_tasks.csv`.

## Finding: LIBERO's init states start objects inside their supports

`scripts/libero/init_state_audit.py` builds every one of the 50 official init states of all 40
tasks exactly as the evaluation does (fresh env, `env.seed(i)`, reset, `set_init_state`; cameras
off), attaches SimuGuard and runs only the 15 settle steps (no policy).  2000 init states, 0
monitor errors:

| suite | init states | confirmed ejection | after gravity stage | after carrier stage = physics-invalid | tasks affected |
|---|---|---|---|---|---|
| LIBERO-Spatial | 500 | 3 | 0 | **0** | 0/10 |
| LIBERO-Object | 500 | 500 | 500 | **500** | 10/10 |
| LIBERO-Goal | 500 | 0 | 0 | **0** | 0/10 |
| LIBERO-10 | 500 | 152 | 150 | **150** | 3/10 (tasks 0, 1, 7) |

Cause: in the floor scene of LIBERO-Object and the living-room table of LIVING_ROOM_SCENE1/2
(LIBERO-10 tasks 0, 1, 7) the stored init states put objects **inside** their support: salad
dressing 37.7 mm, ketchup 37.4 mm, milk and orange juice 34.8 mm, alphabet soup 23.6 mm, bbq sauce
19.6 mm, butter 13.7 mm, cream cheese 8.9 mm into the floor; ketchup 14.6 mm, milk / orange juice
13.3 mm, alphabet soup 7.7 mm into the living-room table.  The depth is the same in every init
state of a scene (the init states differ in x/y only).  MuJoCo resolves it within the first
few substeps (peak speed at substep 2 on the table, 5 on the floor, i.e. ≤ 10 ms) by launching
the objects: up to 1.38 m/s off the floor and 1.83 m/s off the table, e.g. the salad dressing hops
10.6 cm and the ketchup 17 cm, with no robot contact (gravity ratio 17-37, no carrier).  Objects
starting shallower (bbq sauce, butter, cream cheese) pop too but stay under the detector's
thresholds (≤ 0.72 m/s).  They land in a rearranged scene during the settle steps, so the policy
starts from a layout the physics produced, not the one the init state specifies.  The same depths
and launches occur on the robosuite 1.5.2 / MuJoCo 3.3.0 stack (checked on 6 init states).  The
3 + 2 confirmed-but-gravity-explained candidates (LIBERO-Spatial; LIBERO-10 tasks 3 and 6) are
falls: LIBERO's other scenes hold their objects a few cm above their supports.

The videos show it: `pi05_v1/libero_10/t00/segments/ep00_seed0/video.mp4` frames 0-14 (ketchup,
milk and orange juice leave the table between env steps 0 and 4 and land by step 14).
(`deepest_start_mm` in the CSV also counts object-object overlap: in some init states of
LIBERO-Spatial tasks 0 and 2 a bowl starts up to 7.5 mm inside the plate while both are still in
the air; nothing survives the gravity stage there.)

## Open issues

* The carrier stage does not examine direct robot contact, so a contact transient under the
  fingers (LIBERO-Object t4 ep9) stays in the physics-invalid count until a human looks; a
  robot-contact stage (e.g. the object's speed against the gripper's) is still to be written.
* Whether the spurious box-box contact caused the LIBERO-10 t9 ep5 failure needs a counterfactual
  replay (e.g. the same episode with that contact pair excluded); not done.
* Only the official stack was evaluated with the policy; on robosuite 1.5.2 the fixture placement
  is not reproducible from the seed, which also affects anyone evaluating there (RPent's native
  LIBERO arm runs on that stack).
* RoboCasa segments recorded before the centre-of-mass fix over-report the speed of spinning
  objects with an offset centre of mass; RoboCasa campaigns would need re-running (the traces
  store the biased velocities).
* Policy servers: RPent's server serialises requests; on a GPU another tenant saturated, queries
  averaged 27 s over the episodes routed there.  Check `pi05_ping.py` and per-endpoint latency (`policy_endpoint`,
  `policy_latency_s` in `episodes.jsonl`) before long campaigns.

## Where things are (h800-2)

* code: `/data/shared/zhoujingjing/simuguard-libero` (rsync of this repo, no git)
* runs: `/data/shared/zhoujingjing/simuguard-libero-runs/`: `pi05_v1/` (campaign, `summary.json`,
  `replay_all.json`, `replay_nowarm.json`, per-task CSV), `init_audit/` (audit, `audit_summary.json`),
  `probe/`, `smoke/`, `smoke_rpent/` (robosuite 1.5.2 recordings and replays), `servers/` (logs)

## How to run (h800-2)

```bash
source scripts/libero/env.h800-2.sh official          # or: rpent (robosuite 1.5.2)
# probe: inventory, contacts, overhead, in-process replay, a frame
$PY scripts/libero/probe_adapter.py --suite libero_spatial --task 0 --out $RUNS/probe/x
# policy servers (RPent's pi05_vla_server.py, RLinf-Pi05-LIBERO-130-fullshot-SFT, ~8-12 GB each)
bash scripts/libero/pi05_server.sh GPU PORT $RUNS/servers/pi05_PORT.log
$PY scripts/libero/pi05_ping.py http://127.0.0.1:PORT
# one task
$PY scripts/libero/pi05_rollout.py --suite libero_goal --task 3 --episodes 0-4 --out $RUNS/x \
    --endpoint http://127.0.0.1:P1,http://127.0.0.1:P2 --video
# campaign (JOB = SUITE:TASK:EPISODES), stop / requeue infrastructure failures / resume
bash scripts/libero/pi05_campaign.sh $RUNS/pi05_v1 16 URL1,URL2,... libero_10:0:0-4 ...
bash scripts/libero/stop_campaign.sh $RUNS/pi05_v1 && $PY scripts/libero/requeue.py $RUNS/pi05_v1
# three stages, replay, init-state audit
$PY scripts/libero/campaign_summary.py $RUNS/pi05_v1 --per-task --out summary.json
$PY scripts/libero/check_replay.py SEGMENT [--drop-warmstart]
$PY scripts/libero/replay_batch.py $RUNS/pi05_v1 --all --jobs 16 --out replay_all.json
$PY scripts/libero/replay_inspect.py SEGMENT SUBSTEP --body white_yellow_mug_1_main   # contacts geom by geom
$PY scripts/libero/init_state_audit.py --suite libero_10 --task 0 --inits 0-49 --out $RUNS/init_audit
$PY scripts/libero/audit_summary.py $RUNS/init_audit --out audit_summary.json
$PY scripts/libero/task_table.py summary.json audit_summary.json tasks.csv
```

Tests: `python -m pytest tests` (the MuJoCo ones need `mujoco`; on h800-2 use either LIBERO venv).
