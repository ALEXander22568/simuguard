#!/usr/bin/env bash
# Hy-Embodied-0.5-VLA on the 10 Table 1 tasks, 20 episodes each (same seeds/settings as the LingBot-VA runs)
source "$(dirname "$0")/hyvla_env.sh"
export MIN_FREE_POLICY_MIB=11000
cd /path/to/workspace/SimuGuard
exec bash scripts/robotwin/xpolicylab_campaign.sh /path/to/workspace/runs/hyvla_table1 20 "1:1 5:0 6:2 7:3" \
  put_bottles_dustbin open_microwave stack_bowls_three blocks_ranking_size stack_bowls_two \
  place_can_basket place_bread_basket put_object_cabinet turn_switch stamp_seal
