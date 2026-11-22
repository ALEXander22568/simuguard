#!/bin/bash
# After cf_p0b: rerun the robot-link per-body replay (body_prim fix 55e2935), then the runtime-cap probes (GPU 3, one at a time).
R=/path/to/simuguard-robolab/runs; O=$R/cf/p0; S=/path/to/simuguard-robolab/simuguard; M=/path/to/simuguard-robolab/runs-mirror
PY=/path/to/simuguard-robolab/RoboLab/.venv/bin/python
export OMNI_KIT_ACCEPT_EULA=Y CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$S
log() { echo "$(date -u +%FT%TZ) $*" >> $O/p0.log; }
until grep -q "p0b done" $O/p0.log; do sleep 30; done
cd /path/to/simuguard-robolab/RoboLab
t="gr00t/BananaInBowlTask/cap1_banana+left_inner_finger"
[ -d "$O/$t" ] && mv "$O/$t" "$O/${t}_failed1"
mkdir -p "$O/$t"
log "replay gr00t BananaInBowlTask cap1_banana+left_inner_finger start (rerun)"
$PY -m simuguard.integrations.robolab_eval --mode replay --policy gr00t --record-dir $M/formal_gr00t/BananaInBowlTask/BananaInBowlTask \
  --replay-runs 0 --num_envs 10 --depen "1.0,bodies=banana+robot/left_inner_finger" --cap-via prebuild --out "$O/$t" --headless > "$O/$t/process.log" 2>&1
log "replay gr00t BananaInBowlTask cap1_banana+left_inner_finger exit $?"
mkdir -p $O/probe7
RT="--rt usd:0,2,3=0.1 --rt usdrt:4,5=0.1"
for spec in "a:--flush --unpause" "b:--no-fabric --flush" "c:--no-replicate --flush --unpause"; do
  k=${spec%%:*}; args=${spec#*:}
  log "probe7$k start ($args)"
  $PY $S/probes/probe7.py --num-envs 8 $RT $args > $O/probe7/probe7$k.log 2>&1
  log "probe7$k exit $?"
done
log "p0c done"
