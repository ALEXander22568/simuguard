#!/bin/bash
# expert check on 60 seeds of Place Can Basket and Stack Bowls Three, original and patched engine, 3 shards each on GPUs 3/4/6
N=/mnt/nvme0/shared/USER/simuguard; P=/home/USER/lingbot-va-repro/.venv-client/bin/python; D=$N/probe/sapien_patched/expert; mkdir -p $D; cd $D
run() { # engine task shard gpu
  local site=$N/probe/sapien_patched/$1; local s=$((100000 + $3 * 20))
  CUDA_VISIBLE_DEVICES=$4 PYTHONPATH=$site $P $N/probe/sapien_patched/expert_check.py $2 $s 20 $D/$1__$2__$s.jsonl > $D/$1__$2__$s.log 2>&1
}
for task in place_can_basket stack_bowls_three; do
  for engine in site_orig site; do
    run $engine $task 0 3 & run $engine $task 1 4 & run $engine $task 2 6 &
    wait
  done
done
echo DONE > $D/STATUS
