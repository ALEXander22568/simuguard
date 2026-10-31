#!/usr/bin/env bash
# gjk_fix arm of runs/cl_seeded_v1, reordered 2026-10-04: start now on lane 7 (free since the offset_1mm arm finished
# its tasks there), container tasks first; lane 8 joins with the tasks not yet started once the offset_1mm arm is done.
cd /path/to/workspace/SimuGuard
export SIMUGUARD_WORKSPACE=/path/to/workspace ROBOTWIN_ROOT=/path/to/workspace/RoboTwin
export LINGBOT_MODEL=/path/to/RoboTwin-2.0/runtime/lingbot-va-repro/checkpoints/lingbot-va-posttrain-robotwin-modelscope
export SIMUGUARD_CONFIG=/path/to/workspace/SimuGuard/configs/scale_no_bundles.json
export MANIFEST_PYTHON=/path/to/RoboTwin-2.0/runtime/lingbot-va-repro/.venv-client/bin/python
export SIM_GPUS="0 2 1 3" SIM_FREE_MIB=7000
export SIMUGUARD_PATCHED_SAPIEN=/path/to/workspace/sapien_patched/site
OUT=/path/to/workspace/runs/cl_seeded_v1
TASKS="place_can_basket stack_bowls_three place_object_basket place_bread_basket stack_bowls_two blocks_ranking_size open_microwave"
SIM_INTERVENTION=gjk_fix bash scripts/robotwin/parallel_campaign.sh $OUT/gjk_fix 20 "7:29607" $TASKS &
until [ -f $OUT/offset_1mm/STATUS ] && grep -q DONE $OUT/offset_1mm/STATUS; do sleep 120; done
sleep 90
REM=""; for t in $TASKS; do [ -d $OUT/gjk_fix/$t ] || REM="$REM $t"; done
echo "[$(date -u +%FT%TZ)] lane 8 joins with:$REM"
[ -n "$REM" ] && SIM_INTERVENTION=gjk_fix bash scripts/robotwin/parallel_campaign.sh $OUT/gjk_fix 20 "8:29608" $REM &
wait
echo DONE > $OUT/STATUS.gjk_fix
