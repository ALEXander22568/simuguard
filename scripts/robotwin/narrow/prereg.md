# Narrow intervention and matched control on RoboTwin

Pre-registration.  Written and committed before any replay of this study runs.  Nothing below is
changed after the first replay; any deviation is reported as a deviation.

## Question

The paper's counterfactual (`depen_1.0@event`) caps PhysX max depenetration velocity at 1 m/s on
*every* dynamic body and link, from 0.1 s before the first physics-invalid event until the episode
ends.  The question is whether the outcome flips come from removing the artifact or from changing the
physics of the whole scene for the rest of the episode.  We test (a) narrow versions of the same
intervention that touch one body (or one body pair) for a short window, and (b) the identical narrow
intervention on matched episodes that have *no* artifact (control).

## Episodes

* Flagged: the 70 LingBot-VA episodes with a physics-invalid event (gravity stage, carrier stage if it
  ran; detector outputs frozen) whose recordings exist (controls, states, trace).  List:
  `flagged_mapping.csv`.  The 3 `put_object_cabinet` episodes are reported but excluded from all
  counts, as in the paper: their baseline replay is not bit-exact before the switch.  The 10 pass-2
  episodes of the second team run have no control log and cannot be replayed (listed as missing).
  So 67 flagged episodes are analysed (45 F, 22 S by recorded outcome).
* Control: unflagged episodes (no physics-invalid event) of the same task, drawn by the rule below.

## Conditions (cap c in {0.5, 1.0, 2.0} m/s; c = 1.0 is primary)

Anchor `t0` = onset substep of the first physics-invalid event; anchor body = its ejected body.
Substep dt = 4 ms.  Window start = t0 - 25 substeps (0.1 s), same as the paper.

| name | bodies capped | window |
|---|---|---|
| `broad@c` | every dynamic body and link (paper) | t0 - 0.1 s to episode end |
| `body@c` | anchor body | t0 - 0.1 s to t0 + 0.2 s (substeps t0-25 .. t0+50) |
| `partner@c` | anchor body + its contact partner at t0 | same as `body` |
| `resolved@c` | anchor body | t0 - 0.1 s until resolved (below), at most t0 + 1 s |

* Restore: at window start each capped body's current value is read and stored; after the last
  window substep the stored value is written back (no assumed default).  Every report records the
  stored, capped and read-back values.
* Contact partner: in the recorded trace, over substeps t0-2..t0, the other body of the anchor body's
  contacts with the largest penetration (ties and all-zero: largest impulse).  If the partner is static
  (no depenetration setting), only the body is capped and the episode is flagged `partner_static`.
* Resolved: after each step from t0 on, the anchor body's deepest penetration over all its contacts
  (live replay, `-min separation`) is compared with its largest shape rest offset + 1 mm.  The window
  ends after the first substep that completes 5 consecutive substeps (counted from t0) at or below
  that threshold, or after substep t0 + 250 (1 s), whichever comes first.
* `baseline` (no change) is replayed on a subset as a gate (below).

## Control selection rule (fixed RNG seed 20261107)

1. Pool per task: policy episodes of the same task with full recordings and no physics-invalid event,
   from the same run directories as the flagged episodes (the copy `crosstask4_50seeds_copy` and
   `put_object_cabinet` excluded).
2. Gripper contact onsets per pool episode and body: substeps s where the body touches a gripper
   finger link (`f[lr]_link7`, `f[lr]_link8` of the aloha-agilex arms) in the recorded trace and did
   not touch any finger in the preceding 25 substeps.
3. Flagged episodes are processed in the order (task, run group, seed).  For each, with
   `rng = random.Random(20261107)` created once: draw one success uniformly from the task's remaining
   pool successes and one failure from the remaining pool failures.  If one class is empty, draw the
   second episode from the other class (at most 2 per flagged episode).  Drawn episodes leave the pool
   (no reuse).  Candidates are sorted by episode id before each draw.
4. Anchor body = the flagged episode's anchor body if it has a gripper contact onset in the control
   episode, else the task's first target object (`task_ground_truth.targets[0]`).  Anchor `t0` = the
   gripper contact onset of that body whose normalised time `s / L` (L = recorded substeps) is closest
   to the flagged episode's `t0 / L`; ties go to the earlier onset.  Onsets with s < 26 or
   s + 50 > L are not eligible.  A drawn episode with no eligible onset is discarded and the draw is
   repeated from the same class.
5. Control conditions: `body@c`, `partner@c`, `resolved@c` (same code as flagged).  The output of the
   draw (`control_selection.json`) is written before any control replay.

## Outcomes and analysis

* Flip = replayed outcome differs from the recorded outcome.  F->S and S->F are counted separately,
  per condition and cap, per task and over all analysed episodes.  Rates: F->S over recorded failures,
  S->F over recorded successes of that set.
* Replay score in the paper's convention, per task: (S + sum over replayed e of (r_e - y_e)) / N,
  with N, S the task's scored episodes and successes, y_e the recorded and r_e the replayed outcome.
  Rows: the five headline tasks, their total, the analysed total.  For the control the same formula
  over the control episodes shows how far the score would move from the intervention alone.
* Validity gate per replay: the state log must be bit-identical to the recording before the window
  start (`prefix_bit_identical`).  Replays that fail it are reported and excluded from the counts.
  `baseline` is replayed on at least 3 flagged episodes per host before the batch and must be
  bit-identical end to end; `broad@1.0` must reproduce the outcome of the existing paper reports on at
  least 3 episodes.  If a gate fails, the batch does not start.
* Reported regardless of direction.  Expected: control flips near zero; if narrow conditions flip far
  fewer flagged failures than `broad`, the paper's attribution claim is weakened and we say so.
* Caveat kept with every table: open-loop replay overestimates S->F (the policy cannot react to a
  changed state) and underestimates F->S (a rescued state cannot be followed by corrected actions).

## Run order

`broad@1.0`, `body@1.0`, `partner@1.0`, `resolved@1.0` on flagged; the three narrow conditions at 1.0
on control; then caps 0.5 and 2.0 (flagged first, then control).
