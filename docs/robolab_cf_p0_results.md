# RoboLab counterfactual P0: results (2026-11-07)

P0 follows the pre-registration in `docs/robolab_cf_prereg.md`. After P0 we decided not to run a
counterfactual for RoboLab (no B1, B2 or B3, no replay-corrected score). RoboLab reports the raw score,
the exclusion score, its interval and the artifact rate. The findings below go into the limitations.

Scripts: `scripts/robolab_cf/`. Outputs stay on the 48 GB box under `/path/to/simuguard-robolab/runs/cf/p0/`
(`onset_final.json`, `summary_ref_cap1000.json`, `summary_ref_recording.json`, `closed_summary.json`,
`probe7/`, `probe8/`). On that box this note is also `runs/cf/P0_RESULTS.md`. probe5 and probe6 ran earlier on
the 24 GB box; their logs are there in `/path/to/simuguard-robolab/runs/` (`probe5.log`, `probe6a.log`, `probe6b.log`).

## Setup

- Recordings: gr00t BananaInBowlTask run 0, pi05 BananasInCrateTask run 0, pi05 BananasInBinOneMoreTask
  run 0; 10 envs each, all segments compared.
- Physics step dt = 1/120 s. The recorded scene sets `maxDepenetrationVelocity` = 3.0 on every body.
- Caps are set in USD before the sim is built (`--cap-via prebuild`). This is the only way that works (see probe7).
- Replay feeds the recorded joint targets open loop. A replay saves only its first divergence from the recording.
  For two variants X and Y, both match the recording before min(onset_X, onset_Y). So when the two onsets
  differ, X and Y first differ at that substep. When they are equal, X and Y match up to it and any
  difference comes at or after it.

## Reference variants

- No cap change and an explicit 3.0 are bit-exact with the recording on all 30 episodes.
- Cap 1000 is not bit-exact (first divergence at substeps 595, 446, 618). So the 3.0 cap already acts
  during normal play, and the primary contrast is cap 1.0 against cap 1000.
- Whole scene, cap 1.0 against cap 1000: divergence in 30 of 30 envs, mostly at substeps 400-900, which is
  the first grasp and before the events.

## Per-body caps: onset relative to the event

t0 is the onset substep of the confirmed event on the capped body B (gr00t env 9: t0 = 695; it was
confirmed at 698). P is the partner body. The table gives the first substep where cap 1.0 and cap 1000
on the same bodies differ, as onset - t0.

| episode | B | t0 | B only | B + P |
|---|---|---|---|---|
| gr00t Bowl env 9 | banana | 695 | -136 (-1.13 s) | -136 (-1.13 s), P = left_inner_finger |
| pi05 Crate env 4 | banana | 2430 | -1830 (-15.25 s) | -1830 (-15.25 s), P = banana_03 |
| pi05 Crate env 9 | banana_03 | 1360 | -336 (-2.80 s) | -449 (-3.74 s), P = purple_crate |
| pi05 BinOneMore env 9 | banana_01 | 627 | -51 (-0.43 s) | -51 (-0.43 s), P = banana_03 |

In none of the 4 episodes does the per-body cap first act at or after t0. The onsets fall where B is
first handled.

Cap 1000 on B alone is not bit-exact with the recording for banana (both tasks) and banana_01, so the
3.0 cap limits ordinary grasps of these bodies. For banana_03 it stays exact until substep 3447.

## Control: envs with no event

The same contrast on envs with no flagged or confirmed event, where we expected no divergence:

| set | diverged | onsets (substeps) |
|---|---|---|
| gr00t Bowl, B = banana | 6 / 6 | 593, 591, >=607, 576, 567, 539 |
| pi05 Crate, B = banana | 3 / 3 | 611, 649, 618 |
| pi05 Crate, B = banana_03 | 0 / 3 | - |
| pi05 BinOneMore, B = banana_01 | 4 / 5 | 599, -, 607, 689, 597 |

13 of 17 control envs diverge, for B only and for B + P alike.

## Per-body flips against cap 1000 on the same bodies

Pooled over the four B choices. Wilson 95% intervals in brackets. The confirmed class holds only
episodes whose confirmed event is on B.

| set | class | n | success before -> after | success -> fail | fail -> success |
|---|---|---|---|---|---|
| B | confirmed | 4 | 3 -> 1 | 2/3 [.21, .94] | 0/1 |
| B | flagged only | 17 | 12 -> 7 | 7/12 [.32, .81] | 2/5 [.12, .77] |
| B | no event | 17 | 16 -> 14 | 2/16 [.04, .36] | 0/1 |
| B + P | confirmed | 4 | 1 -> 0 | 1/1 | 0/3 |
| B + P | flagged only | 17 | 7 -> 7 | 4/7 | 4/10 |
| B + P | no event | 17 | 16 -> 14 | 2/16 | 0/1 |

The cap acts before the event and also changes episodes with no event. These flips cannot be
attributed to the artifact.

## probe7: the cap cannot be changed during an episode

Test (BananaInBowlTask, 8 envs): push the banana 12 mm into its support and measure the rise in one
step. With no cap it rises about 12 mm. With a 0.1 m/s cap set before the build it rises 0.82 mm.

Writes of 0.1 during the episode, through USD on envs 0, 2, 3 (2, 3 in probe5 and probe6) and through usdrt (Fabric) on envs 4, 5:

| probe | write path | rise in step 1 (mm) |
|---|---|---|
| probe5, probe6 | plain write | 11.87 - 11.99 |
| probe7a | + `IPhysxSimulation.flush_changes()`, change tracking on | 11.98 - 12.0 |
| probe7b | `use_fabric=False` + flush | 11.98 - 12.0 |
| probe7c | `replicate_physics=False` + flush, change tracking on | 11.87 - 12.0 |

The `omni.physics.tensors` views have no setter for this value. In this Isaac Lab and GPU PhysX stack
the value is read only when PhysX creates the body at sim start. A cap only inside a window around t0
(the RoboTwin design) is therefore not possible here.

## probe8 and the closed-loop floor

Closed loop on pi05 BananasInCrateTask run 0, 10 envs, a fresh policy server for each run: A and A'
with no change, C with a whole-scene cap of 1.0. The initial states are the same in all three and in
the recording.

| comparison | class | n | success before -> after | success -> fail | fail -> success |
|---|---|---|---|---|---|
| A vs A' | confirmed | 2 | 2 -> 2 | 0/2 | - |
| A vs A' | flagged only | 5 | 4 -> 5 | 0/4 | 1/1 |
| A vs A' | no event | 3 | 3 -> 3 | 0/3 | - |
| A vs C | confirmed | 2 | 2 -> 2 | 0/2 | - |
| A vs C | flagged only | 5 | 4 -> 3 | 2/4 [.15, .85] | 1/1 |
| A vs C | no event | 3 | 3 -> 3 | 0/3 | - |

A and A' are bit-exact in 0 of 10 envs and diverge at substeps 420-723. C diverges at almost the same
substeps as A'. With 10 episodes the cap effect cannot be told apart from this floor.

The cause is rendering. probe8 built the same task twice in two processes, with the same seed,
initial state and fixed action. The camera images differ in every env and every frame: 29-58% of
pixels differ, by 0.5-1.25 grey levels on average and up to 43-142. The policy server gives the same
output for the same input. So the first action chunk already differs (A vs A': 3e-4 to 1e-2 rad), and
the episodes drift apart. A closed-loop rerun also cannot reproduce the recording.
