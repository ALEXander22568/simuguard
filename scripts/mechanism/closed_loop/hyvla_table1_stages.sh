#!/usr/bin/env bash
# Gravity and carrier stages on the finished HyVLA tasks (CPU only, low priority).
source "$(dirname "$0")/hyvla_env.sh"
cd /path/to/workspace/SimuGuard
D=/path/to/workspace/runs/hyvla_table1
RUNS=$(for t in blocks_ranking_size open_microwave place_bread_basket place_can_basket put_bottles_dustbin put_object_cabinet stack_bowls_three stack_bowls_two stamp_seal turn_switch; do echo $D/$t; done)
nice -n 10 "$MANIFEST_PYTHON" scripts/robotwin/gravity_filter.py --runs $RUNS > $D/stages_gravity.log 2>&1
nice -n 10 "$MANIFEST_PYTHON" scripts/robotwin/carrier_filter.py --runs $RUNS > $D/stages_carrier.log 2>&1
echo DONE > $D/STAGES_STATUS
