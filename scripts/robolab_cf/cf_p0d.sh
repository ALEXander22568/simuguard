#!/bin/bash
# After cf_p0c: camera-observation reproducibility probe (two processes, GPU 3, one at a time).
R=/path/to/simuguard-robolab/runs; O=$R/cf/p0; S=/path/to/simuguard-robolab/simuguard
PY=/path/to/simuguard-robolab/RoboLab/.venv/bin/python
export OMNI_KIT_ACCEPT_EULA=Y CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$S
log() { echo "$(date -u +%FT%TZ) $*" >> $O/p0.log; }
until grep -q "p0c done" $O/p0.log; do sleep 30; done
mkdir -p $O/probe8
cd /path/to/simuguard-robolab/RoboLab
for k in 1 2; do
  log "probe8 run$k start"
  $PY $S/probes/probe8.py --out $O/probe8/run$k.npz > $O/probe8/run$k.log 2>&1
  log "probe8 run$k exit $?"
done
$PY $S/probe8_cmp.py $O/probe8/run1.npz $O/probe8/run2.npz > $O/probe8/cmp.txt 2>&1
log "p0d done"
