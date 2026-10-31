#!/usr/bin/env bash
# Two more distinct seeds for HyVLA place_object_basket: shards s1 and s2 both scored 100015/100016
# (the scripted demonstration passes or fails differently across runs), so 18 distinct seeds so far.
source "$(dirname "$0")/hyvla_env.sh"
export MIN_FREE_POLICY_MIB=11000
cd /path/to/workspace/SimuGuard
O=/path/to/workspace/runs/hyvla_table1/place_object_basket_shards
t0=$(date +%s)
echo "[$(date -u +%FT%TZ)] s4: start seed 100026, 2 episodes, lane p1s1" >> $O/campaign.log
SEED_START=100026 bash scripts/robotwin/run_xpolicylab_eval.sh $O/s4 2 place_object_basket 1 1 > $O/s4.supervisor.log 2>&1
rc=$?
echo "[$(date -u +%FT%TZ)] s4: exit=$rc after $(( ($(date +%s) - t0) / 60 )) min" >> $O/campaign.log
[ "$rc" -eq 0 ] && "$MANIFEST_PYTHON" scripts/robotwin/build_run_manifest.py --run-dir $O/s4 --task place_object_basket \
    --robotwin-root "$ROBOTWIN_ROOT" --started-epoch "$t0" >> $O/campaign.log 2>&1
echo "DONE rc=$rc" > $O/STATUS.s4
