#!/bin/bash
# value-preserving pose write k steps before the onset, k = 1..8 and none, on 8 further episodes
N=/mnt/nvme0/shared/USER/simuguard; P=/home/USER/lingbot-va-repro/.venv-client/bin/python; D=$N/probe/mech/noop
cd $N/probe/mech
sed -n '2,9p' $N/probe/episodes.txt | while read -r seg onset; do
  tag=$(basename $(dirname $(dirname $(dirname $seg))))_$(basename $seg)
  for k in -1 1 2 3 4 5 6 7 8; do
    CUDA_VISIBLE_DEVICES=7 timeout 600 $P noop_reset.py --segment $seg --onset $onset --k $k --out $D/${tag}_k$k.json > $D/${tag}_k$k.log 2>&1 &
  done
  wait
done
echo DONE > $D/BATCH_STATUS
