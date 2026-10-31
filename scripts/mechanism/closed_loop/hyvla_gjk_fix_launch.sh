#!/usr/bin/env bash
# HyVLA closed loop in the repaired engine (gjk_fix), five container tasks, 20 seeds, same settings as runs/hyvla_table1.
source "$(dirname "$0")/hyvla_env.sh"
export MIN_FREE_POLICY_MIB=11000 SIM_INTERVENTION=gjk_fix
export SIMUGUARD_PATCHED_SAPIEN=/path/to/workspace/sapien_patched/site
cd /path/to/workspace/SimuGuard
bash scripts/robotwin/xpolicylab_campaign.sh /path/to/workspace/runs/hyvla_gjk_fix 20 "6:7 7:6" place_can_basket stack_bowls_three stack_bowls_two place_bread_basket place_object_basket
echo ALLDONE > /path/to/workspace/runs/hyvla_gjk_fix/STATUS
