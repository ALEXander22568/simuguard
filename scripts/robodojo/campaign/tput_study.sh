#!/usr/bin/env bash
# Throughput study on GPUs 5-6 (coordinator, 11-07): batch size, time split, recording overhead.
#   P1  official client, Pi_05 hang_mugs, num_envs 4 (sim 5 / server 6) || num_envs 2 (sim 6 / server 5)
#   P2  SimuGuard runner, GR00T_N17 hang_mugs layout 0, monitor on (GPU 5) || monitor + contact report off (GPU 6)
#   P3  official client, Pi_05 hang_mugs, num_envs 1 (GPU 5) || num_envs 8 (GPU 6)
# usage: bash tput_study.sh OUT_DIR
set -uo pipefail
OUT=${1:?out dir}; mkdir -p "$OUT"
S=/path/to/simuguard-robodojo/sgscripts
log() { echo "$(TZ=Asia/Seoul date '+%m-%d %H:%M:%S') $*" >> "$OUT/study.log"; }
bench() { python3 "$S/bench_batch.py" --out "$OUT/$1" --policy Pi_05 --task hang_mugs --num-envs "$2" --sim-gpu "$3" --server-gpu "$4" > "$OUT/$1.out" 2>&1; }
sgq() { python3 "$S/rd_queue.py" --run-dir "$OUT/$1" --gpus "$2" --policies GR00T_N17 --tasks hang_mugs --max-counted 1 "${@:3}" > "$OUT/$1.out" 2>&1; }

log "P1 start"; bench n4 4 5 6 & bench n2 2 6 5 & wait; log "P1 done"
log "P2 start"; sgq sg-on 5 & sgq sg-off 6 -- --monitor off --contact-report off & wait; log "P2 done"
log "P3 start"; bench n1 1 5 5 & bench n8 8 6 6 & wait; log "P3 done"
