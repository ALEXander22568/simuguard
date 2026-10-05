#!/bin/bash
# export every Stack Bowls Three case of bowls_cases.json, one replay per segment, 3 at a time on GPU 7
D=/mnt/nvme0/shared/USER/simuguard/probe/physx_repro; P=/home/USER/lingbot-va-repro/.venv-client/bin/python
cd $D; mkdir -p data/bowls; n=0
for seg in $(grep '"segment"' bowls_cases.json | sed 's/.*: "\(.*\)",/\1/' | sort -u); do
  CUDA_VISIBLE_DEVICES=7 timeout 3000 $P $D/export_pair_generic.py --cases $D/bowls_cases.json --segment $seg --out-root $D/data/bowls > $D/data/bowls/export_$(basename $seg).log 2>&1 &
  n=$((n+1)); [ $((n % 3)) = 0 ] && wait
done
wait; echo DONE > $D/data/bowls/EXPORT_STATUS
