#!/usr/bin/env bash
# HyVLA on place_object_basket: the 20 seeds LingBot-VA was scored on (runs/crosstask5_20ep) in 4 shards
# of 5, pulled from a queue by parallel lanes (policy GPU:sim GPU). Each shard starts at its first seed
# (SEED_START -> robotwin_eval --seed-start) and scores 5 episodes; the expert check skips seeds as usual.
source "$(dirname "$0")/hyvla_env.sh"
export MIN_FREE_POLICY_MIB=11000   # HyVLA holds ~10 GB; other tenants leave 13 GB on GPUs 6/7
S=/path/to/workspace/SimuGuard
OUT=/path/to/workspace/runs/hyvla_table1/place_object_basket_shards
TASK=place_object_basket
mkdir -p "$OUT"; cd "$S"
Q="$OUT/.queue"
printf '%s\n' "s0 100000 5" "s1 100009 5" "s2 100015 5" "s3 100021 5" > "$Q"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$OUT/campaign.log"; }
take() { ( flock 8; l=$(head -n 1 "$Q"); [ -n "$l" ] && sed -i '1d' "$Q"; echo "$l" ) 8> "$Q.lock"; }
back() { ( flock 8; echo "$1" >> "$Q" ) 8> "$Q.lock"; }
lane() {
  local pg=$1 sg=$2 spec name start n rc t0 run
  while spec=$(take) && [ -n "$spec" ]; do
    read -r name start n <<< "$spec"
    run="$OUT/$name"; mkdir -p "$run"
    echo "RUNNING $name start=$start n=$n policy_gpu=$pg sim_gpu=$sg" > "$OUT/STATUS.lane_p${pg}s${sg}"
    log "$name: start seed $start, $n episodes, lane p${pg}s${sg}"
    t0=$(date +%s)
    SEED_START=$start bash scripts/robotwin/run_xpolicylab_eval.sh "$run" "$n" "$TASK" "$pg" "$sg" > "$run/supervisor.log" 2>&1
    rc=$?
    log "$name: exit=$rc after $(( ($(date +%s) - t0) / 60 )) min (lane p${pg}s${sg})"
    if [ "$rc" -eq 75 ]; then log "$name: not enough GPU memory on p${pg}s${sg}; requeued"; back "$spec"; sleep 120; continue; fi
    if [ "$rc" -eq 0 ]; then
      "$MANIFEST_PYTHON" scripts/robotwin/build_run_manifest.py --run-dir "$run" --task "$TASK" \
        --robotwin-root "$ROBOTWIN_ROOT" --started-epoch "$t0" >> "$OUT/campaign.log" 2>&1
    else
      log "$name: FAILED, see $run/supervisor.log"
    fi
  done
  echo IDLE > "$OUT/STATUS.lane_p${pg}s${sg}"
}
log "shards: $(wc -l < "$Q"), lanes: $*"
for spec in "$@"; do lane "${spec%%:*}" "${spec##*:}" & sleep 20; done
wait
echo DONE > "$OUT/STATUS"
