#!/bin/bash
# Replay estimate on the seeded default arm: every episode with an artifact replayed with the depenetration
# limit of 1 m/s (the paper's rerun) and with contact offset 1 mm, each switched on 0.1 s before the first artifact.
W=/path/to/workspace
PY=/path/to/RoboTwin-2.0/runtime/lingbot-va-repro/.venv-client/bin/python
export PYTHONPATH=$W/SimuGuard:$W/RoboTwin:$W/RoboTwin/XPolicyLab
cd $W/SimuGuard
D=$W/runs/cl_seeded_v1/default; O=$W/runs/cl_seeded_v1/replay_estimate; mkdir -p $O
$PY scripts/robotwin/attribute_failures.py --robotwin-root $W/RoboTwin \
    --runs $D/place_can_basket $D/stack_bowls_three $D/place_object_basket $D/place_bread_basket $D/stack_bowls_two \
    --select invalid --phases policy --conditions depen_1.0@event offset_1mm@event --primary depen_1.0@event \
    --output-dir $O --gpus 1 3 --workers 2 > $O.log 2>&1
echo ALLDONE > $O/STATUS
