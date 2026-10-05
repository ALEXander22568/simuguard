# Where the RoboTwin contact artifacts come from

Measured 2026-10-03/04. RoboTwin 2.0 runs on SAPIEN 3.0.0b1, which embeds PhysX 5.3.1
(PCM on, TGS, CCD off, 4 ms step, contact offset 10 mm, rest offset 0, no depenetration limit).
Scripts are in `scripts/mechanism/` unless a path under `scripts/robotwin/` is given.
The scripts are the ones that were run; their absolute paths are kept, with the account and workspace directory
names replaced by `USER` and `WORKSPACE`.

## Cause

`physx/source/geomutils/src/gjk/GuGJKSimplex.h`, `closestPtPointTriangle` (two overloads):

```cpp
const FloatV eps  = FEps();                    // 1.1920929e-7
const FloatV area = V3Dot(signArea, signArea); // |ab x ac|^2
if (FAllGrtrOrEq(eps, area)) { /* degenerate: fall back to a segment */ }
```

The threshold is absolute. A triangle of the GJK simplex with `|ab x ac| <= 3.45e-4 m^2` is treated as
degenerate, which a well shaped triangle reaches once the geometry is at the centimeter scale. PhysX 5.4.0
compares with `PX_EPS_REAL * PX_EPS_REAL` (1.42e-14) instead; that is the only change in the file.

What follows in 5.3.1, traced with prints in a standalone build (`physx_repro/sg_patch*.py`):

1. `gjkPenetration` adds a support point, the triangle is rejected, the distance does not decrease, and the loop
   leaves through `bNotDegenerated == false`: `GJK_DEGENERATE` with the closest points of an unconverged simplex.
   This exit is not compared with the contact distance.
2. `addGJKEPAContacts` accepts the result when the search direction and the normal agree; no SAT test.
3. `generateFullContactManifold` picks the witness faces from those closest points. The point on one piece lies
   inside it or on its far side, so `getWitnessPolygonIndex` returns a face that points away from the other piece
   (or, in a third of the bowl cases, the right reference face with a sideways witness face on the other piece).
4. The contact is generated against that face: a gap is reported as a depth of about gap + piece thickness.
5. `addManifoldContactsToContactBuffer` outputs every point with separation <= contact offset; no lower bound.
6. The solver removes the reported depth within the step.

Steps 2 to 5 are unchanged up to PhysX 5.9.0.

## Evidence

| level | what was run | result |
|---|---|---|
| value at the faulty step | `physx_repro/versions/trace_area.sh` | `|ab x ac|^2` = 8.81e-8, 1.189e-7, 1.180e-7, 5.99e-8 in four Place Can Basket episodes; threshold 1.192e-7; triangle edges 11 to 96 mm |
| geometry of the reported contact | `robotwin_probe/onset_table.py`, `onset_sign.py` | 35 onsets: pieces a median 15 mm apart, reported -24 mm; normal opposite to the healthy convention in 34 of 35 (0 of 1958 healthy contacts); depth = far vertex of the can piece to the contact point along the normal, median error 0.00 mm |
| standalone PhysX, recorded poses | `physx_repro/repro.cpp` | 5.3.1 reproduces SAPIEN's onset contact to the printed digits in four episodes (fig1: n = 0.6579 -0.0292 0.7525, -27.54 mm) |
| one file swapped | `physx_repro/versions/variant2.sh`; `variant.sh` (its `extent1` variant only) | 5.3.1 with only `GuGJKSimplex.h` from 5.4.0: 0 of 4; with only the `hullData1` extent fix: 4 of 4 |
| newer versions | `physx_repro/versions/build_tag.sh TAG` | 5.4.0, 5.4.2, 5.5.1, 5.6.1, 5.9.0: 0 of 4 |
| Stack Bowls Three | `physx_repro/bowls_analyze.py`, `versions/bowls_fix_test.py` | 21 onsets with a deep contact: 16 false depths on the same path (true gap 0 to 21.7 mm), 4 true penetrations (EPA, depth equals the SAT value), 1 not reproduced. With the 5.4.0 header: 0 of 16 false depths, the 4 true penetrations unchanged |
| SAPIEN itself, open loop | `sapien_patch/patch_lib.py`, `replay_check.py` | patched copy of `libsapien.so` (4 loads of the threshold, 19 bytes): four episodes without any can-basket contact deeper than 5 mm, peak can speed below 1 m/s (fig1 unpatched: -27.54 mm, 21.8 m/s). The patched replay leaves the recording at step 3500 to 5300, so it is not a paired counterfactual |
| SAPIEN itself, scripted expert | `sapien_patch/expert_check.py` | 60 seeds each. Place Can Basket: episodes with events 18 -> 1, planned but failed 6 -> 1, success 51 -> 59. Stack Bowls Three: 6 -> 0, 5 -> 0, 30 -> 36 |
| SAPIEN itself, closed loop | `closed_loop/launch*.sh` | see below |

`physx_repro/data/fig1` holds the two hulls and the last 27 pose steps of the Figure 1 episode; it is enough to
reproduce the contact on 5.3.1 and its absence with the 5.4.0 header.

## Conditions

The artifact needs both PhysX <= 5.3.1 and collision pieces at the centimeter scale (RoboTwin's basket pieces are
about 12 x 15 x 19 mm, can pieces 4 to 25 mm). PyPI `sapien` 3.0.0 and 3.0.3 still ship PhysX 5.3.1.
Three settings amplify it without causing it:

* contact offset 10 mm per shape: 25 to 90 can-basket piece pairs are within range each step. Replays with the
  offset of can and basket changed 0.1 s before the first artifact (`scripts/robotwin/attribute_failures.py`,
  condition `offset_<mm>mm@event`, 37 Place Can Basket
  episodes, 152 recorded events, all prefixes bit identical): episodes with events 3 at 0.5 mm, 4 at 1 mm, 9 at 2 mm,
  11 at 3 mm, 13 at 5 mm, 37 at 10 mm. At 1 mm: 19 of 21 failures pass, 12 of 16 successes kept
  (depenetration limit 1 m/s: 26 episodes with events, 18 of 21, 9 of 16). Stack Bowls Three at 1 mm, 12 episodes:
  6 keep events, 4 of 12 pass; pieces that touch are not excluded by a smaller offset.
* no limit on the depenetration velocity;
* 10 g objects.

A value preserving `set_pose` on the can just before the onset removes the artifact in 6 of 9 episodes
(`robotwin_probe/noop_reset.py`): in those the GJK warm start stored in the contact cache is needed; the other
three reproduce from an empty cache.

## Closed loop

LingBot-VA, 20 scored seeds per task, sampling noise reseeded at every reset (`SIMUGUARD_POLICY_SEED`),
run directory `runs/cl_seeded_v1` (manifests from `scripts/robotwin/build_run_manifest.py`, artifact counts
from `scripts/robotwin/gravity_filter.py`). Success / scored, in parentheses the episodes with an artifact after the gravity stage.

| task | default | offset 1 mm | 5.4.0 threshold (`gjk_fix`) |
|---|---|---|---|
| Place Can Basket | 15/20 (5) | 19/20 (3) | 20/20 (0) |
| Stack Bowls Three | 15/20 (6) | 17/20 (3) | 19/20 (0) |
| Place Object Basket | 20/20 (0) | 20/20 (0) | 20/20 (0) |
| Place Bread Basket | 19/20 (1) | 19/20 (0) | 20/20 (0) |
| Stack Bowls Two | 20/20 (0) | 20/20 (0) | 19/20 (0) |
| five tasks | 89/100 (12) | 95/100 (6) | 98/100 (0) |

Default: 10 of the 11 failures contain an artifact. Paired on the seeds scored in both arms, default against
`gjk_fix`: 83 seeds, 73 both, 2 default only, 8 `gjk_fix` only. The expert check is not deterministic across
runs, so the scored seeds differ between arms. The two control tasks, default and offset 1 mm: Blocks Ranking
Size 18/20 and 17/20, Open Microwave 15/20 and 18/20, none with an artifact; that is the run to run spread at
20 episodes. The `gjk_fix` arm of the control tasks and a fourth arm with PCM off were still running when this
was written (PCM off: Place Can Basket 20/20 with 0 artifacts, Place Object Basket 20/20 with 1).

## RoboDojo

Isaac Sim 5.1 (omni.physx 107.3.26, GPU dynamics). Insert Tubes layout 2, substep 3901
(`robodojo/mech_dump.py`, `analyze_*.py`): three new contact points with separation -10.9 to -10.1 mm that lie
7.64 mm outside the convex hull of the tube; the tube is paired with the far side of a 2 to 5 mm thin wall of
the rack. The same kind of false depth, on another path: convex pieces against a static SDF triangle mesh in the
GPU contact generation. The 5.4.0 change does not apply to it and the faulty step there is not located. The tube's
depenetration velocity limit reads 3 m/s at run time.

## Not done

* Between the rejected triangle and the distance that no longer decreases, the simplex solver was not followed
  step by step; the link rests on the printed value and on the one file swap.
* RoboTwin was not run on a complete newer PhysX, only on 5.3.1 with the one threshold changed.
* One bowl onset is not reproduced by the standalone program; 19 bowl onsets have no deep object to object contact
  and were not analysed.
* HyVLA was not run in the patched engine.
