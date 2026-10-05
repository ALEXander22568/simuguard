#!/bin/bash
# export three further episodes (pair = deepest onset contact from probe/mech/onset_table.py)
D=/mnt/nvme0/shared/USER/simuguard/probe/physx_repro; R=/mnt/nvme0/shared/USER/simuguard/runs/campaign_20260918T220055Z/default/rep1/simuguard/segments
P=/home/USER/lingbot-va-repro/.venv-client/bin/python
cd $D
run() { CUDA_VISIBLE_DEVICES=7 timeout 1200 $P $D/export_pair.py --segment $R/$1 --onset $2 --can-piece $3 --basket-piece $4 --out $D/data/$5 > $D/data/$5.log 2>&1; }
run seg0012_seed100006_ep5 11044 17 46 ep5 &
run seg0032_seed100017_ep14 11301 19 18 ep14 &
run seg0008_seed100004_ep3 12618 16 41 ep3 &
wait
echo DONE > $D/data/EXPORT_STATUS
