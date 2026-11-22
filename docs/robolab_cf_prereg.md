# RoboLab counterfactual and control: pre-registration

Fixed on 2026-11-07 (13:15 KST), before any flip count of this pass was computed. Copy of
`/path/to/simuguard-robolab/runs/cf/PREREG.md` on the 48 GB box.

## Data

The formal RoboLab recordings: Pi0.5-DROID and GR00T N1.7-DROID, 120 tasks, 10 envs x 2 runs per task.
Sources: `runs-mirror/formal_<policy>/<task>/` on the 48 GB box (copied with sha256 checks from the 24 GB box),
and `runs/formal_gr00t/<task>/` for GR00T tasks recorded on the 48 GB box. The detector and its thresholds are
the frozen ones used for the recordings. Nothing is re-tuned here.

## Selection

- Unit: one recorded task batch (one policy, one task).
- A task is selected if any non-discarded episode in it has a confirmed detector event.
- Replay covers runs 0..k, where k is the last run of the task with a confirmed episode. A replay of run 1 has
  to replay run 0 anyway, so every non-discarded episode of the replayed runs is counted.
- Cutoff 1: every task with DONE.json in the sources at 2026-11-07T04:15Z (19 Pi0.5 tasks, 10 GR00T tasks).
- Tasks that reach the sources later are added in waves with the same rule. Every interim summary states its
  cutoff. The final summary uses every task of the finished formal pass.
- P0 (pilot) uses the 3 shortest selected runs from 3 different tasks. Replay is deterministic, so P0 results
  are the same as the B1/B2 results for those runs and are counted once.

## Episode classes

Each non-discarded episode falls in exactly one class:

- confirmed: at least one confirmed event;
- flagged-only: at least one flag and no confirmed event;
- no-event: no flag and no confirmed event (the control). Rejected candidates may be present.

## Caps (all through `--cap-via prebuild`)

Only build-time writes reach GPU PhysX. Every cap therefore holds from substep 1 to the end and applies to the
named bodies in all 10 envs of the batch (replicate_physics=True). No windowed cap is used.

1. Reference variants, per selected task: no cap (control), cap 1000 m/s, cap 3.0 m/s written explicitly (the
   PhysxSchema default used when the attribute is unset), cap 1.0 m/s on every body.
2. Per-body variants. For each confirmed event: B = the event's first body (the ejected body). P = the first of
   the event's other bodies that is not the table, preferring a scene object over a robot link. If there is
   none, the event has no B+P variant. Variants: cap 1.0 on B, and cap 1.0 on B+P. The table is never capped.
   Each distinct (B, P) set in a task is replayed once. In such a replay, the confirmed class holds only the
   episodes whose confirmed event has that B. Episodes with confirmed events on other bodies are left out of
   that replay. No-event and flagged-only episodes are counted per replay.

## Primary contrast

Chosen from the P0 bit-exactness outcome only, never from flip counts:

- If the cap 1000 replay is bit-exact to the recording on all three P0 tasks (all object and dof values), the
  primary contrast is cap 1.0 vs the recording. The per-body variants are compared with the recording.
- Otherwise the primary contrast is cap 1.0 vs cap 1000. Each per-body variant is compared with a cap 1000
  replay on the same bodies.

The no-cap control must be bit-exact. If it is not, the pass stops.

## Metrics

For each variant, contrast, class and policy:

- n episodes, the success rate before and after the cap, and the change;
- k/n flips success -> failure and k/n flips failure -> success, reported separately;
- Wilson 95% intervals for the success rates and for each flip fraction (the flip denominators are the
  episodes that succeeded or failed before the cap).

Outcome = RoboLab's own `success` termination term, evaluated during replay at every policy step.

## Closed loop

Open-loop replay applies the recorded actions. Once a trajectory diverges, later actions no longer react to
the world they act on. In particular, they do not react to the world after the artifact. The closed-loop runs
bound this.

- Our own Pi0.5 server on 48 GB box GPU 3, restarted before every run, with a single client.
- Step 1: the same 3 seeded observations on a fresh server, again on the same server, and again after a
  restart. This shows whether actions depend only on the call order since start.
- Per task: A = uncapped, A' = uncapped repeat, C = cap 1.0 on every body. All use the formal seed and
  10 envs x 1 run.
- Classes come from the detector events of A.
- Floor = the A vs A' flips. If A and A' are bit-exact, the floor is 0 and A vs C is an exact paired
  counterfactual.
- The metrics are the same as above, for A vs C and for A vs A'.
- P0 uses the shortest selected Pi0.5 task. B3 uses the 3 Pi0.5 tasks with the most confirmed episodes at
  cutoff 1 (ties: shorter episodes first).
