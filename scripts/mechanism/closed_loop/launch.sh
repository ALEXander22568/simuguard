#!/usr/bin/env bash
# Seeded, paired closed loop (2026-10-03): LingBot-VA with the policy's sampling noise reseeded at every reset
# (SIMUGUARD_POLICY_SEED in va_host.env on the inference host), 20 seeds per task, default physics against contact offset 1 mm
# on the task's objects and containers (applied to the policy rollout only, so the expert check runs under default physics
# in both arms; it is not deterministic across runs, so compare on the seeds scored in both), then PCM off (scene level, expert check included).
cd /mnt/nvme0/WORKSPACE/simuguard/SimuGuard
export SIMUGUARD_WORKSPACE=/mnt/nvme0/WORKSPACE/simuguard ROBOTWIN_ROOT=/mnt/nvme0/WORKSPACE/simuguard/RoboTwin
export LINGBOT_MODEL=/mnt/nvme0/WORKSPACE/RoboTwin-2.0/runtime/lingbot-va-repro/checkpoints/lingbot-va-posttrain-robotwin-modelscope
export SIMUGUARD_CONFIG=/mnt/nvme0/WORKSPACE/simuguard/SimuGuard/configs/scale_no_bundles.json
export MANIFEST_PYTHON=/mnt/nvme0/WORKSPACE/RoboTwin-2.0/runtime/lingbot-va-repro/.venv-client/bin/python
export SIM_GPUS="0 2 1 3" SIM_FREE_MIB=7000
OUT=/mnt/nvme0/WORKSPACE/simuguard/runs/cl_seeded_v1
TASKS="place_can_basket stack_bowls_three place_object_basket place_bread_basket stack_bowls_two blocks_ranking_size open_microwave"
( SIM_INTERVENTION=none bash scripts/robotwin/parallel_campaign.sh $OUT/default 20 "5:29605 6:29606" $TASKS
  SIM_INTERVENTION=pcm_off bash scripts/robotwin/parallel_campaign.sh $OUT/pcm_off 20 "5:29605 6:29606" $TASKS ) &
sleep 60
SIM_INTERVENTION=offset_1mm bash scripts/robotwin/parallel_campaign.sh $OUT/offset_1mm 20 "7:29607 8:29608" $TASKS &
wait
echo DONE > $OUT/STATUS
