#!/usr/bin/env bash
# XR-1 RoboDojo checkpoint (ModelScope dataset RoboDojo-Benchmark/RoboDojo), resumable.
set -u
cd "$(dirname "$0")"
URL="https://modelscope.cn/api/v1/datasets/RoboDojo-Benchmark/RoboDojo/repo?Revision=master&FilePath=ckpt/RoboDojo/Xiaomi_Robotics_1/RoboDojo-sim-arx_x5-ee-0/last.ckpt/checkpoint/mp_rank_00_model_states.pt"
for i in $(seq 1 20); do
  curl -s -L -C - --retry 5 --retry-delay 10 -o last.ckpt/checkpoint/mp_rank_00_model_states.pt "$URL" && break
  echo "attempt $i failed rc=$?"; sleep 15
done
ls -la last.ckpt/checkpoint/
echo DONE $(date -u +%FT%TZ)
