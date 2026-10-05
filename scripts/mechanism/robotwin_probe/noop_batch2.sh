#!/bin/bash
# rerun of the missing runs of noop_batch.sh, 4 at a time (9 at once ran GPU 7 out of memory)
N=/mnt/nvme0/shared/USER/simuguard; P=/home/USER/lingbot-va-repro/.venv-client/bin/python; D=$N/probe/mech/noop
cd $N/probe/mech; n=0
while read -r seg onset; do
  tag=$(basename $(dirname $(dirname $(dirname $seg))))_$(basename $seg)
  for k in -1 1 2 3 4 5 6 7 8; do
    [ -f $D/${tag}_k$k.json ] && continue
    CUDA_VISIBLE_DEVICES=7 timeout 900 $P noop_reset.py --segment $seg --onset $onset --k $k --out $D/${tag}_k$k.json > $D/${tag}_k$k.log 2>&1 &
    n=$((n+1)); [ $((n % 4)) = 0 ] && wait
  done
done < <(sed -n '2,9p' $N/probe/episodes.txt)
wait; echo DONE > $D/BATCH2_STATUS
