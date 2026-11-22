# Objects stuck to an open gripper (RoboLab)

`stick_detect.py` reads recorded RoboLab segments and flags episodes in which an object
keeps moving with a finger after the gripper has opened. The rule and its thresholds were
fixed on a development set of 18 episodes before the full run.

## Rule

For each dynamic scene body, at each physics substep (1/120 s), three conditions are checked.

| Condition | Definition | Threshold |
|---|---|---|
| Gripper commanded open | finger joint target below `OPEN_CMD_RAD`, held for at least `GRACE_S` | 0.1 rad, 0.5 s |
| Touching a finger | at least one contact point with a finger link, separation at most `TOUCH_SEP_M` | 1 mm |
| No other support | no contact point with any body that is not part of the robot | none |

Substeps that meet all three form a window. Two windows merge when the gap between them is
at most 0.1 s and the first and third conditions hold through the gap. An episode is flagged
when one window lasts at least 2.0 s and the object moves at least 0.10 m from where the
window started.

## Result

The rule flags 179 of the 4800 RoboLab episodes. In 10 of 20 flagged clips checked by eye,
the object sticks to the gripper or passes through it. An open finger inside a mug or
hooked under a bowl rim can lift the object for real, and the rule cannot tell these cases
apart, so flagged episodes are candidates for review.

## Files

`stick_detect.py` applies the rule to a list of segments. `analyze.py` builds the counts per
policy and object. `stick_series.py` writes the object and finger positions of one window.
