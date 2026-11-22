#!/bin/bash
# P0 part b, after cf_p0.sh: cap 1000 on the per-body sets (cap 1000 was not bit-exact, so per PREREG the
# per-body variants are compared with cap 1000 on the same bodies)
R=/path/to/simuguard-robolab/runs
M=/path/to/simuguard-robolab/runs-mirror
O=$R/cf/p0
PY=/path/to/simuguard-robolab/RoboLab/.venv/bin/python
export OMNI_KIT_ACCEPT_EULA=Y CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/path/to/simuguard-robolab/simuguard
cd /path/to/simuguard-robolab/RoboLab
log() { echo "$(date -u +%FT%TZ) $*" >> $O/p0.log; }
while ! grep -q "p0 done" $O/p0.log; do sleep 30; done
replay() {
  local pol=$1 task=$2 tag=$3 out=$O/$1/$2/$3
  [ -f $out/done ] && return
  mkdir -p $out
  log "replay $pol $task $tag start"
  $PY -m simuguard.integrations.robolab_eval --mode replay --policy $pol --record-dir $M/formal_$pol/$task/$task \
    --replay-runs 0 --num_envs 10 --depen "$4" --cap-via prebuild --out $out --headless > $out/process.log 2>&1
  local code=$?
  log "replay $pol $task $tag exit $code"
  [ $code = 0 ] && touch $out/done
}
replay gr00t BananaInBowlTask cap1000_banana "1000,bodies=banana"
replay gr00t BananaInBowlTask cap1000_banana+left_inner_finger "1000,bodies=banana+robot/left_inner_finger"
replay pi05 BananasInCrateTask cap1000_banana "1000,bodies=banana"
replay pi05 BananasInCrateTask cap1000_banana+banana_03 "1000,bodies=banana+banana_03"
replay pi05 BananasInCrateTask cap1000_banana_03 "1000,bodies=banana_03"
replay pi05 BananasInCrateTask cap1000_banana_03+purple_crate "1000,bodies=banana_03+purple_crate"
replay pi05 BananasInBinOneMoreTask cap1000_banana_01 "1000,bodies=banana_01"
replay pi05 BananasInBinOneMoreTask cap1000_banana_01+banana_03 "1000,bodies=banana_01+banana_03"
log "p0b done"
