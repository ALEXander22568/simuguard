#!/bin/bash
# probe every episode with a physics-invalid event of four further container tasks, 4 replays at a time on GPU 5
N=/mnt/nvme0/shared/USER/simuguard; P=/home/USER/lingbot-va-repro/.venv-client/bin/python; D=$N/probe/mech/tasks
mkdir -p $D; cd $N/probe/mech; n=0
for root in $N/runs/crosstask4_50seeds_copy/stack_bowls_three $N/runs/crosstask_20260921T160736Z/stack_bowls_two $N/runs/crosstask_20260921T160736Z/place_bread_basket $N/runs/crosstask2_20260922T063959Z/place_object_basket; do
  task=$(basename $root)
  for seg in $root/simuguard/segments/*/; do
    seg=${seg%/}
    grep -q '"phase": "policy"' $seg/summary.json 2>/dev/null || continue
    f=$seg/events_carrier.json; [ -f $f ] || f=$seg/events_gravity.json
    [ -f $f ] && grep -q '"physics_invalid"' $f || continue
    out=$D/${task}__$(basename $seg).json; [ -f $out ] && continue
    CUDA_VISIBLE_DEVICES=5 timeout 1500 $P probe_all_onsets.py --segment $seg --out $out > ${out%.json}.log 2>&1 &
    n=$((n+1)); [ $((n % 4)) = 0 ] && wait
  done
done
wait; echo DONE $n > $D/STATUS
