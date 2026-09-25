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
| PhysX override | RoboDojo calls `omni.physx.get_physx_interface().overwrite_gpu_setting(1)` ("enable cpu garment and deformable"); the effective value is recorded as `physx_overwrite_gpu_setting` | `env/environment/base_env.py` |
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

Runner options: `--policy scripted` (built-in IK push probe, no server), `--replay first|all`
(close, rebuild the layout, replay the recorded actuation, compare every logged body),
`--public-restore K` (restore the public snapshot at substep K in place and replay 500 substeps),
`--replay-from SEG...` (fresh-process replay of recorded segments), `--contact-report off`,
`--monitor off`.

## Measured results

(filled in below)
