# Stack Bowls Three: which PhysX branch produces the onset contacts (2026-10-04)

Cases: `bowls_cases.json` (21 onsets whose onset step holds an object-object contact deeper than 5 mm; from
`probe/mech/tasks/stack_bowls_three__*.json`, script `bowls_list_cases.py`).
Export: `export_pair_generic.py` (any two bodies, any piece pair, several cases per replay), driver `bowls_export.sh`,
data in `data/bowls/<segment>_o<onset>/` (same file layout as data/fig1; hull_can = piece of the first shape of the
reported contact, hull_basket = piece of the second).
Analysis: `bowls_analyze.py` (runs build/repro with debug prints over the whole pose history, keeps the onset step's
block, compares with SAPIEN, SAT ground truth from the hull vertices) -> `bowls_results.json`, log `bowls_analyze.out`.
`bowls_trace.py TAG` compares the program with SAPIEN step by step over onset +- 400.

Result (see bowls_results.json):
* 16 of 21: GJK_DEGENERATE exit -> accepted by addGJKEPAContacts (doOverlapTest = 0) -> fullContactGen -> witness faces
  from the unconverged closest points -> negative pen although the pieces do not intersect (SAT: separated by 0.2 to
  21.7 mm, or touching within 0.01 mm).  Same branch as Place Can Basket.  Three variants of the wrong face:
  reference face on shape 0 facing away from shape 1; reference face on shape 1 facing away from shape 0;
  reference face correct but the other shape's witness face nearly perpendicular to the GJK normal (|dot| < 0.1),
  with a GJK distance of 14 to 25 mm for pieces that are 0 to 0.8 mm apart.
* 4 of 21 (all late in seg0034, steps 97487 to 97713): GJK finds an overlap, EPA_CONTACT, and the reported depth equals
  the SAT minimum translation depth (11.7 to 18.0 mm): real interpenetration, correct contact.
* 1 of 21 (seg0043 o10976): not reproduced by the program (it outputs a healthy +3 mm contact at that step).
* For 2 cases the deepest point and normal are reproduced but the number of points differs; for 6 cases the full
  history drifts and a shorter history starting from an empty cache reproduces SAPIEN's contact.  The program's cache
  can differ from the engine's after hundreds of steps (bowls_trace.py: e.g. one cached point differs from step 10932
  in seg0043); the cause was not found.
