#!/bin/bash
# a: unpatched copy on the Figure 1 episode; b: patched copy on it; c: patched copy on three more episodes. <= 3 at a time, GPU 7.
D=/mnt/nvme0/shared/USER/simuguard/probe/sapien_patched; P=/home/USER/lingbot-va-repro/.venv-client/bin/python
R=/mnt/nvme0/shared/USER/simuguard/runs/campaign_20260918T220055Z/default
mkdir -p $D/out; cd $D
run() { site=$1; tag=$2; seg=$3; onset=$4
  PYTHONPATH=$D/$site CUDA_VISIBLE_DEVICES=7 $P $D/replay_check.py --segment $seg --onset $onset --out $D/out/${tag}.json > $D/out/${tag}.log 2>&1; }
run site_orig a_orig_fig1 $R/rep2/simuguard/segments/seg0036_seed100019_ep16 11023 &
run site b_patched_fig1 $R/rep2/simuguard/segments/seg0036_seed100019_ep16 11023 &
run site c_patched_ep5 $R/rep1/simuguard/segments/seg0012_seed100006_ep5 11044 &
wait
run site c_patched_ep14 $R/rep1/simuguard/segments/seg0032_seed100017_ep14 11301 &
run site c_patched_ep3 $R/rep1/simuguard/segments/seg0008_seed100004_ep3 12618 &
run site_orig a_orig_ep5 $R/rep1/simuguard/segments/seg0012_seed100006_ep5 11044 &
wait
echo DONE > $D/out/STATUS
