# SimuGuard on RoboDojo (Isaac Sim 5.1 / Isaac Lab / PhysX 5)

RoboDojo (arXiv 2607.04434) evaluates XPolicyLab policies on 42 simulated tasks built on
Isaac Sim 5.1.0 + Isaac Lab 0.54.3 (image `hexa/robodojo:0.2.4-f465f8ae`, sha256 `9e8f8334…`),
dual ARX X5 arms.  This page covers how RoboDojo steps physics, what the Isaac adapter
reads, how exact replay is, how to run it, and what was measured.

Status and numbers: see [Measured results](#measured-results).

## How RoboDojo simulates (measured in the image, not assumed)

| item | value | source |
|---|---|---|
| physics step | `dt = 0.004 s` (250 Hz), `decimation = 1`, `render_interval = 10` | `env_cfg/sim/sim_config.yml`, read back from `SimulationContext` |
| substep call | `CustomDirectRLEnv.sim_step`: `scene.write_data_to_sim()` → `SimulationContext.step(render=False)` → `scene.update(dt)` | `env/environment/isaac/direct_rl_env.py` |
| one action | `take_action` interpolates the joint target over `collect_interval = 1/(dt·25 Hz) = 10` substeps (80 % ramp, 20 % hold) | `src/eval_client/eval_env.py` |
| PhysX scene (USD) | TGS, MBP broadphase, `enableGPUDynamics = false`, enhanced determinism off, CCD off, min velocity iterations 0 | `physics_settings()` in the manifest |
| PhysX override | RoboDojo calls `omni.physx.get_physx_interface().overwrite_gpu_setting(1)` ("enable cpu garment and deformable"); read back as 1 in every run, i.e. **GPU dynamics is forced on** despite the USD attribute | `env/environment/base_env.py`, `physx_overwrite_gpu_setting` in the manifest |
| Isaac Lab device | `cpu` (tensors on host; `use_fabric = False`, PhysX writes transforms back to USD) | `SimulationContext.device` |
| actuation | Isaac Lab implicit actuators: PhysX joint drives (arm stiffness 4400 / damping 40, gripper 2300 / 100); inputs are drive position/velocity targets and joint efforts | `env/robot_manager/robot_config/x5.py` |
| objects | Isaac Sim `SingleRigidPrim`, mass capped at 0.5 kg by RoboDojo; static "Geometry" objects (dustbin, tube rack, toolbox) are colliders without a rigid body | `env/scene_manager/objects/*.py` |
| episode flow | one layout per `reset(seed=[layout])`, `run_eval()`, `close()` (the stage is closed and rebuilt for every batch) | `src/eval_client/main.py` |

## The adapter (`simuguard/adapters/isaac/`)

* `adapter.py` - `IsaacAdapter`, generic for any Isaac Sim stage; `physx_io.py` - the
  simulator-free parts (tensor layouts, contact merging; unit-tested in `tests/test_isaac_pure.py`);
  `robodojo.py` - `RoboDojoAdapter`, the body inventory and roles from RoboDojo's scene and layout
  managers, and `install_contact_reporting()`.

| contract item | implementation | exact? |
|---|---|---|
| bodies / kinds / roles | robot links from Isaac Lab articulation views; layout objects by label (Rigid/Dynamic → rigid bodies below the prim, Articulation → articulation + links, Geometry and Table/Ground/Rooms → STATIC groups matched by path prefix); roles from `TASK_SPECS` (label regexes per task), unknown tasks: every free object is OBJECT | - |
| `read_states` | one PhysX rigid-body tensor view over all links and free bodies: pose (body frame, wxyz) and centre-of-mass velocity read right after each step | exact (float32 as PhysX stores it) |
| `read_contacts` | omni.physx contact report: per shape pair, per point position / normal / separation / normal impulse of the step; merged per body pair, impulse oriented onto `body_a`.  Needs `PhysxContactReportAPI` on the watched bodies: `install_contact_reporting()` patches RoboDojo's object loaders so it is applied before PhysX parses the prim (applying it to a live body re-creates the actor) | exact to PhysX's own net contact force (checked, below); friction impulses not included |
| `install_substep_hook` | instance attribute over `SimulationContext.step`; an independent `subscribe_physics_step_events` counter proves every PhysX step went through the hook | exact |
| `capture_control` / `apply_control` | per articulation (both arms, articulated objects): drive position targets, velocity targets and joint actuation forces read from PhysX right before the step; plus any pose/velocity/joint-state write made between two steps (compared with the state the previous step left) | exact |
| `capture_snapshot` / `restore_snapshot("public")` | all free-body poses and velocities, articulation root states, joint positions/velocities/targets | approximate in contact: PhysX contact caches and warm start are not restorable, no native snapshot exists in Isaac Sim |
| exact replay | restore method `none`: rebuild the layout the same way (close + reset), then apply the recorded control before every step | bit-exact where measured (below) |

## Running it

Everything runs in throwaway containers of the pinned image; nothing in RoboDojo, XPolicyLab or the
shared assets is modified.

1. Assets.  The shared asset trees on the 4090 nodes are partial (only what the URAI lines needed)
   and lack `Robots/*/curobo.yml` (RoboDojo's installer writes it from `curobo_tmp.yml`).
   `scripts/robodojo/build_asset_overlay.py --tasks ...` builds `$SG_ROOT/assets_overlay`: symlinks
   into the shared tree, a copy of `Robots/` with generated `curobo.yml`, and the missing object
   folders downloaded from the official ModelScope mirror (`RoboDojo-Benchmark/RoboDojo`, `Assets/`).
2. One job: `scripts/robodojo/sg_container.sh NAME GPU|auto OUT robodojo_eval-args...`
   (`simuguard.integrations.robodojo_eval`).  It refuses a GPU with less than `MIN_FREE_MIB` free.
3. A queue: `scripts/robodojo/sg_queue.sh JOBS RUN_ROOT` waits for a GPU with >= 15 GB free on two
   polls a minute apart, stops our container at the first GPU out-of-memory message (the cards are
   shared) and retries later, and only starts the next job if the previous one ended cleanly.
4. Policy: XR-1 (`Xiaomi_Robotics_1`, official RoboDojo checkpoint `RoboDojo-sim-arx_x5-ee-0`)
   is served from h800-2 by `xr1_robodojo_server.sh` (XPolicyLab code from the image, unmodified),
   reached through an SSH tunnel whose key may only forward that one port.
5. Summary: `scripts/robodojo/campaign_summary.py RUN_ROOT` (detector → gravity → carrier stages),
   `scripts/robodojo/check_contacts.py SEGMENT` (resting contact force vs weight).

Deployment used here (2026-09-25):

| host | path | what |
|---|---|---|
| 4090-node1 | `/mnt/nvme0/shared/zhoujingjing/simuguard-robodojo/` | `simuguard/` (rsync of this repo), `assets_overlay/` (base `/mnt/nvme0/shared/geshijia/urai-robodojo/assets`), `run_queue.sh`, `runs/v1`, `runs/v2`, `sg_policy_tunnel.sh` (127.0.0.1:16602) |
| 4090-node3 | `/data/shared/zhoujingjing/simuguard-robodojo/` | same layout (base `/data/shared/geshijia/urai-robodojo/assets`), tunnel 127.0.0.1:16601 |
| h800-2 | `/data/shared/zhoujingjing/simuguard-robodojo-policy/` | XPolicyLab + `env_cfg` copied from the image, `pylib/` (scipy, opencv-headless, h5py, websockets, msgpack-numpy, liger-kernel for `~/xr1-deploy/mibot-env`), `xr1_robodojo_server.sh`, `logs/` |
| h800-2 | `/data/shared/zhoujingjing/checkpoints/robodojo/` | XR-1 checkpoint (`model_states.pt` converted from the DeepSpeed file, `config.py` for inference, `config.orig.py` as released), `Qwen3-VL-4B-Instruct-processor/` (backbone weights + processor) |

State after the runs (2026-09-25 22:30 KST): both XR-1 servers on h800-2 (ports 16601/16602) and both
tunnels are stopped, and no SimuGuard container or queue is running.  h800-2 `~/.ssh/authorized_keys`
still holds the two forwarding-only keys (`permitopen="127.0.0.1:16601"` / `"127.0.0.1:16602"`, backup
of the file before them: `authorized_keys.bak-20260925-sg-robodojo`); delete those two lines when the
tunnels are no longer wanted.

Policy setup (`scripts/robodojo/policy/`): `download_xr1_robodojo_ckpt.sh`, `download_qwen3vl4b.sh`,
`convert_ds_checkpoint.py` (the release is a DeepSpeed `mp_rank_00_model_states.pt` with the weights under
`module`; XPolicyLab's `helper()` expects a flat `model_states.pt`), `make_inference_config.py` (drops
the training-only `stop_gradient_to_vlm` / `training_repeat` that the vendored `XR1.__init__` rejects),
`xr1_robodojo_server.sh` (`GPU=7 PORT=16601`), `sg_policy_tunnel.sh` (node side; the key is restricted on
h800-2 to forwarding that one port).

Example (node1):

```bash
cd /mnt/nvme0/shared/zhoujingjing/simuguard-robodojo
export SG_ROOT=$PWD SG_ASSETS_BASE=/mnt/nvme0/shared/geshijia/urai-robodojo/assets
python3 simuguard/scripts/robodojo/build_asset_overlay.py --tasks insert_tubes
printf 'tubes --task insert_tubes --layouts 0 1 --policy Xiaomi_Robotics_1 --policy-url ws://127.0.0.1:16602 --replay all\n' > jobs.txt
nohup setsid bash simuguard/scripts/robodojo/sg_queue.sh jobs.txt runs/mine > runs/mine.queue.log 2>&1 &
python3 simuguard/scripts/robodojo/campaign_summary.py runs/mine          # needs numpy
```

Runner options: `--policy scripted` (built-in IK push probe, no server; its RoboDojo "success" is not
meaningful), `--kick-speed V` (probe only: positive control, writes a velocity onto an object between two
steps), `--replay first|all` (close, rebuild the layout, replay the recorded actuation, compare every
logged body), `--public-restore K` (restore the public snapshot at substep K in place and replay 500
substeps), `--replay-from SEG...` (fresh-process replay of recorded segments; `--replay-detect` runs
the detectors on the replayed frames), `--intervention depen_V@S` (counterfactual replay: cap
`maxDepenetrationVelocity` of every free object at V m/s from substep S on), `--contact-report off`,
`--monitor off`.  Every episode record carries `final_states` (exact end pose and velocity of every
rigid body), so runs with and without the monitor can be compared bit for bit.

## Measured results

RTX 4090s of 4090-node1 and 4090-node3, image `hexa/robodojo:0.2.4-f465f8ae`, 2026-09-25.  Runs:
`node1:/mnt/nvme0/shared/zhoujingjing/simuguard-robodojo/runs/{v1,v2}` and
`node3:/data/shared/zhoujingjing/simuguard-robodojo/runs/{v1,v2}` (each job: `container.log`,
`<task>/episodes.jsonl`, `<task>/segments/layoutNNN/` with SimuGuard artefacts, `eval_result/` with
RoboDojo's official `_result.json` and the three camera videos per episode).

### Physics configuration actually in effect

* `physx_overwrite_gpu_setting = 1`: RoboDojo forces **GPU dynamics**, although the USD scene says
  `enableGPUDynamics = false` (MBP, TGS).  So everything below is GPU PhysX with host readback.
* Every PhysX step went through the hook: an independent `subscribe_physics_step_events` counter equals
  the hooked count in every monitored episode (1800/1800 in the probes, 3020/3020 to 11000/11000 in the
  ten XR-1 episodes).
  No between-step state writes occurred in any policy episode; the positive control's injected velocity
  was the only one recorded.

### Contacts

* Our contact-report reading equals PhysX's own net contact force (rigid contact view,
  `get_net_contact_forces`, a different code path) to 1.4e-6 N for all eight free bodies of the two
  probes (four tools, four bottles; 25 samples each while resting).
* Resting objects on `put_bottles_into_dustbin`: median support force 1.000 × weight for all four bottles
  over the first second (`scripts/robodojo/check_contacts.py`; 3-4 points each, penetration ≤ 1 µm).
* Dynamic contact is physically consistent: a bottle falling into the dustbin (`put_bottles_into_dustbin`
  layout 0, substep 486) has free-fall acceleration -9.81 m/s² before impact, and the reported impact
  impulse 1.522 N·s equals its momentum change |mΔv - mg·dt| = 1.531 N·s (0.6 %); next substeps
  0.1170 vs 0.1191 and 0.0102 vs 0.0103 N·s.
* Resting contact on `store_tools_in_toolbox` is not: PhysX reports a support force of 1.0-2.0 × the
  weight (hammer 0.99, pliers 2.00, tape measure 2.04, wrench 2.01 over the first second; the tools'
  meshes give 22-88 contact points against the table, the bottles 3-4) and resting "phantom" velocities
  of 1-11 mm/s whose positions drift < 0.1 mm/s.  This is PhysX's own output (see the net-force match),
  a TGS artefact of steady contact (this task runs without `enable_stabilization`, the bottles task with
  it); force-based flags (`impulse_spike`) should be read as relative on RoboDojo.  The kinematic
  thresholds of the ejection detector (0.35-0.5 m/s) are two orders above these velocities.

### Replay fidelity

| check | episodes | substeps | max position error |
|---|---|---|---|
| rebuild (close + reset, same process) + replay recorded actuation | probe store_tools L0; XR-1 bottles L0, L1 | 1800; 4160; 3020 | **0.0 m** (bit-exact, 26 bodies incl. 22 robot links; initial state identical) |
| same, in a fresh process on another GPU (card 2 vs 3) | XR-1 bottles L0, L1 | 4160; 3020 | **0.0 m** |
| same, on another machine (recorded node1 GPU 0, replayed node3 GPU 4), detectors on the replayed frames | XR-1 tubes L2 | 5000 | **0.0 m**; the same three events fire again (tube2 at substep 3901, 4.31 m/s) |
| every episode run with `--replay` in this study (3 scripted probes incl. the positive control, 10 XR-1) | 13 episodes | 1800-11000 each | **0.0 m** in all |
| in-place public-snapshot restore at substep 400, replay 500 substeps | probes store_tools L0, bottles L0 | 500 | tools: 10 µm after 1 substep, 1.0 mm after 500; bottles: 75 µm after 1, 0.2 mm after 500 (restored pose error 1.2e-7 m = float32) |

Positive control (`--kick-speed 3`, `store_tools_in_toolbox` L0, node3): the wrench, resting on the
table, gets v = (0.3, 0, 3.0) m/s written between two steps at control step 25.  The adapter records it
as the only between-step write of the episode; the detector confirms it at the next substep (ballistic
free flight, 2.98 m/s); gravity (59.5× the free-fall speed) and carrier stages keep it as physics-invalid;
the rebuild replay re-applies the write and is bit-exact over 1800 substeps.  (The same run shows 57
`impulse_spike` flags from the wrench landing and bouncing: force flags are noisy on RoboDojo.)

GPU PhysX is deterministic here as long as the scene is rebuilt the same way; what cannot be restored
in place is PhysX's contact/solver state.  Replays therefore start from the episode start
(`bundle_mode = episode_start`, restore method `none`).

### Monitoring does not change the simulation

The same scripted probe (`store_tools_in_toolbox` L0, 180 actions = 1800 substeps, deterministic IK
pushes) was run three times on node1 GPU 0, one after another, plus once on node3 earlier:

| run | `PhysxContactReportAPI` on objects | SimuGuard hook | episode wall | dynamics vs. `probe_on` |
|---|---|---|---|---|
| `probe_off` (`--monitor off`) | yes | none | 29.9 s | end state of all 26 rigid bodies bit-identical |
| `probe_nocontact` (`--contact-report off`) | no | states, actuation, detectors | 30.4 s | all 28 bodies × 13 values bit-identical at every one of the 1800 substeps |
| `probe_on` (default) | yes | everything incl. contacts | 42.0 s | - |
| `probe_on` again, on node3 GPU 4 | yes | everything | 43.2 s | bit-identical at every substep (the whole closed-loop episode, on another machine) |

So neither the contact-report schema, the extra contact views nor the hook alter a single bit of the
trajectory, and RoboDojo's own outcome is untouched.  Every episode record now carries `final_states`,
so the same check can be made on any task.

### Overhead

Per substep, measured inside the hook (XR-1 episodes; physics step = PhysX `simulate` + fetch):

| episode | contact points per substep | monitor | of which contact report | physics step | share of episode wall time |
|---|---|---|---|---|---|
| bottles L0 / L1 | 42 / 39 | 7.3 / 6.9 ms | 3.1 / 2.9 ms | 8.5 / 8.2 ms | 4.0 / 3.0 % |
| tubes L0 / L1 / L2 | 380 / 596 / 163 | 13.4 / 16.8 / 11.6 ms | 7.6 / 9.8 / 4.4 ms | 9.8 / 9.4 / 10.5 ms | 5.4 / 6.3 / 4.5 % |
| tools L0 / L1 | 613 / 400 | 18.4 / 15.6 ms | 10.1 / 7.6 ms | 8.9 / 9.2 ms | 6.7 / 6.2 % |
| pens L0 / L1 / L2 | 219 / 272 / 265 | 13.8 / 13.2 / 13.5 ms | 6.8 / 7.0 / 6.8 ms | 10.6 / 11.1 / 10.8 ms | 6.8 / 6.2 / 6.4 % |

The wall time of an XR-1 episode is dominated by rendering three cameras per action and shipping each
observation to the policy (1.8-2.8 s per action through the gateway), so SimuGuard adds 3-7 %.  Without
a policy in the loop the cost is visible end to end: the scripted probe above takes 29.9 s with the
monitor off, 30.4 s (+2 %) with the monitor but no contact reading, and 42.0 s (+40 %) with everything
(226 contact points per substep).  Per substep the monitor costs 0.8-2.1 physics steps, and the contact
report dominates: PhysX's call itself takes
0.02-0.04 ms, the rest is reading ~15-20 µs per contact point through the Python bindings (tubes and tools
rest on the rack/table with hundreds of mesh contact points).  PhysX's rigid contact view returns the same
data as tensors (`get_contact_data`, it needs every partner listed as a filter); switching to it is the
obvious speed-up, not done here to keep the reader that was validated against `get_net_contact_forces`.

### XR-1 evaluation (three-stage event classification)

XR-1 = XPolicyLab `Xiaomi_Robotics_1`, official RoboDojo checkpoint `RoboDojo-sim-arx_x5-ee-0`, `ee`
actions, one env per process, eval seed 0, layouts in order.  Stage 1: confirmed contact-ejection events;
stage 2: gravity filter; stage 3: carrier filter (`scripts/robodojo/campaign_summary.py`).

10 episodes on 4 tasks, 4 successes.  Stage 1 confirmed 10 contact-ejection events in 6 episodes; the
gravity filter explained 5 of them as falls; the carrier filter found no carrier for the other 5, so
**5 physics-invalid events remain, in 2 episodes (tubes L2, tools L1), both failed**.  Every episode was
replayed bit-exactly after its rebuild.

| task | layout | host | success | actions | substeps | stage 1 | stage 2 | stage 3 | flags | replay | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fill_pen_holder | 0 | n1 | yes | 641 | 6410 | 1 | 0 | 0 | 0 | bit-exact | 1307 s |
| fill_pen_holder | 1 | n1 | no | 1100 | 11000 | 0 | 0 | 0 | 2 | bit-exact | 2355 s |
| fill_pen_holder | 2 | n1 | no | 1100 | 11000 | 1 | 0 | 0 | 1 | bit-exact | 2296 s |
| insert_tubes | 0 | n1 | yes | 305 | 3050 | 0 | 0 | 0 | 0 | bit-exact | 751 s |
| insert_tubes | 1 | n1 | no | 500 | 5000 | 0 | 0 | 0 | 0 | bit-exact | 1345 s |
| insert_tubes | 2 | n1 | no | 500 | 5000 | 3 | 2 | 2 | 5 | bit-exact | 1294 s |
| put_bottles_into_dustbin | 0 | n1 | yes | 416 | 4160 | 1 | 0 | 0 | 3 | bit-exact | 762 s |
| put_bottles_into_dustbin | 1 | n1 | yes | 302 | 3020 | 0 | 0 | 0 | 0 | bit-exact | 684 s |
| store_tools_in_toolbox | 0 | n3 | no | 900 | 9000 | 1 | 0 | 0 | 14 | bit-exact | 2483 s |
| store_tools_in_toolbox | 1 | n3 | no | 900 | 9000 | 3 | 3 | 3 | 23 | bit-exact | 2266 s |

(stage k = events still classified physics-invalid after stage k; flags = unconfirmed detector flags:
31 `impulse_spike` (30 on the tools task), 10 `actuation_bound`, 7 `deep_penetration`; wall = episode
wall time incl. policy calls.)

Confirmed events and their fate through the stages (frames checked for every one; video frame k is the
observation after control step k):

| episode | control step | body (partners) | onset → peak speed | stage 2 (speed / free-fall) | stage 3 | what the frames show | result |
|---|---|---|---|---|---|---|---|
| bottles L0 (success) | 49 | bottle3 (dustbin) | 0.67 → 0.70 m/s, Δv 3.0 m/s in one step | 0.21 | - | bottle released over the floor dustbin, lands in it | gravity-explained |
| tools L0 (fail) | 701 | pliers (table) | 0.68 → 0.74 m/s, 16 ms free flight | 0.87 | - | pliers released just above the toolbox rim | gravity-explained |
| pens L0 (success) | 471 | pen `target1` (gripper fingers) | 0.68 → 0.98 m/s, 44 ms free flight | 1.00 | - | pen slips out of the left gripper and falls to the table | gravity-explained |
| pens L2 (fail) | 588 | pen `target0` (pen `target2`) | 0.93 → 1.01 m/s after a 15 cm drop | 0.59 | - | pen let go 15 cm up beside the holder, falls onto a pen leaning against the holder (states: free fall from substep 5840 at -9.81 m/s²) | gravity-explained |
| tubes L2 (fail) | 391 | **tube2** (tube rack, 11 mm penetration) | 0 → **4.31 m/s in one 4 ms step**, 194 N; robot links ≤ 1.24 m/s | 86 | no carrier | tube shoots out of the rack over the far table edge | **physics-invalid**; tube ends on the floor 1.7 m away |
| tubes L2 | 391 | tube0 (hit by tube2) | 0.99 → 1.00 m/s | 20 | no carrier | knocked out of the rack | **physics-invalid** (secondary) |
| tubes L2 | 404 | tube2 (ground) | 2.0 m/s | 0.49 | - | the ejected tube falling to the floor | gravity-explained |
| tools L1 (fail) | 139 | **hammer** (toolbox only) | 0.73 → 2.95 m/s; links ≤ 0.39 m/s | 2.04 | no carrier | hammer, held upright over the box, tilts against the rim (141-143), leaves the gripper and lies at the back of the table by frame 180 | **physics-invalid** (closest to the 1.5 cut); ends 0.6 m away |
| tools L1 | 790 | **wrench** (gripper fingers, pliers, toolbox; 4 mm penetration) | **6.15 → 6.38 m/s in one step**, 813 N; links ≤ 0.40 m/s | 4.25 | no carrier | wrench shoots out of the toolbox to the upper left | **physics-invalid** |
| tools L1 | 793 | pliers (wrench, tape, toolbox) | 0.50 → 1.57 m/s | 31 | no carrier | knocked by the wrench | **physics-invalid** (secondary) |

Unconfirmed flags were checked the same way where they are not force flags: pens L1 has an
`actuation_bound` flag (pen at 1.14 m/s while the arms move ≤ 0.21 m/s) and a `deep_penetration` flag
(6 mm into the table) - the states show a pen released 17 cm above the table in free fall, landing at
1.57 m/s, i.e. one 4 ms step of travel.  Pens L2's `deep_penetration` flag (6.5 mm) is a pen that fell
tumbling and then rocks on the table at ≤ 0.1 m/s; PhysX lifts it back by 5 mm within 16 ms, without
any speed-up.  Neither is an ejection, which is why only confirmed events are counted.

The objects carry no authored `maxDepenetrationVelocity` or solver-iteration override (PhysX defaults;
recorded per episode in `task_ground_truth.physx_body_properties`).  All three physics-invalid incidents
happen while the policy places an object into a static receptacle (tube rack, toolbox), the RoboDojo
analogue of RoboTwin's concave-container ejections.

### Counterfactual: the tube ejection is set by a solver limit

The tubes L2 episode was replayed in fresh processes on node1 with the detectors on the replayed
frames, capping `physxRigidBody:maxDepenetrationVelocity` of the three tubes from substep 3800 on (100
substeps before the event).  The tubes get `PhysxRigidBodyAPI` at creation without any value authored;
the value is written on the live prim at substep 3800 (`--intervention depen_V@3800`).

| replay | first difference from the recording | tube2 at the ejection (substep 3901) | other confirmed events |
|---|---|---|---|
| control: schema at creation, no value ever written | none, **bit-exact** over 5000 substeps | 4.31 m/s (same event) | tube0 knocked out (1.00 m/s), tube2 lands on the floor (2.04 m/s) |
| cap 1.0 m/s from 3800 | substep 3901, tube2 10.7 mm | **1.88 m/s** | none |
| cap 0.3 m/s from 3800 | substep 3877, tube2 8 µm | **0.62 m/s** | none |

The actuation is the same recorded one in all three replays and they are identical until the tube is
pushed into the rack; the speed at which it then leaves the rack follows the cap (4.31 → 1.88 →
0.62 m/s).  So it is set by how fast PhysX may resolve the 11 mm penetration: the confirmed event is a
solver artefact, as stages 2 and 3 classified it.  (Capping from creation instead diverges at substep
789, long before the event, so it cannot isolate the cause.)  An open-loop replay cannot say whether the
episode would have succeeded with the cap, since the policy does not react to the changed scene.

## What does not work / open issues

* **GPU memory on the shared 4090 nodes** is the bottleneck.  A RoboDojo process (1 env, 3 cameras, cuRobo)
  holds ~8 GB and peaks higher while loading textures; on a card that another job also grows on, an 11 GiB
  margin was not enough (node3, GPU 6: our container was stopped at the first allocation failure).  The
  queue therefore waits for 15 GB free on two polls and stops its container on any out-of-memory message.
* **H800 / A100 hosts cannot run RoboDojo**: Isaac Sim 5.1 crashes right after "app ready" even headless
  without cameras (no RT cores; the headless kit still enables the RTX renderer).
* **Shared asset trees are incomplete** (node1/node3/h800-1 copies hold what the URAI lines used):
  `make_toast` alone misses 165 object folders; the overlay builder downloads what a task needs.
* **Resting contact forces** on some tasks are 2× the weight in PhysX's own report (TGS; see above), so
  force-threshold flags are not calibrated for RoboDojo.
* **Public-snapshot restore** is approximate (contact caches); exact replay needs the rebuild path.
* Overhead is dominated by per-point contact parsing on contact-heavy tasks (see Overhead): 3-7 % of an
  XR-1 episode, but +40 % on a policy-free probe.
* **Small sample**: 10 XR-1 episodes on 4 of the 42 tasks (layouts 0-2), one seed; pi0.5 was not set
  up.  The rates above are not benchmark-level numbers.
* **Counterfactual replays are open loop**: they show what sets the ejection speed, not whether the
  episode would have succeeded (the recorded policy actions do not react to the changed scene).
* Friction impulses are not in the contact report reading (normal impulses only); the carrier stage and
  the ejection detector do not need them, but a friction-based check would.
* The collaborator's RoboDojo integration (Feishu wiki "Simuguard", 霍逸逍: pi0.5 8/425 and XR-1 30/425
  confirmed events on 10 tasks) was not found on node1/2/3 or h800-1/2; this adapter was written from
  scratch, so those numbers could not be compared episode by episode.
