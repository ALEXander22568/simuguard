#!/bin/bash
# P0 counterfactual pilot on 48 GB box GPU 3, one sim process at a time: open-loop replays of 3 mirrored tasks under the
# reference and per-body caps, then the closed-loop Pi0.5 determinism step and A / A' / C on BananasInCrateTask
R=/path/to/simuguard-robolab/runs
M=/path/to/simuguard-robolab/runs-mirror
O=$R/cf/p0
PY=/path/to/simuguard-robolab/RoboLab/.venv/bin/python
OP=/path/to/simuguard-robolab/openpi
PORT=8011
export OMNI_KIT_ACCEPT_EULA=Y CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/path/to/simuguard-robolab/simuguard
cd /path/to/simuguard-robolab/RoboLab
mkdir -p $O
log() { echo "$(date -u +%FT%TZ) $*" >> $O/p0.log; }

replay() {  # policy task tag [depen spec]
  local pol=$1 task=$2 tag=$3 out=$O/$1/$2/$3
  [ -f $out/done ] && return
  mkdir -p $out
  log "replay $pol $task $tag start"
  $PY -m simuguard.integrations.robolab_eval --mode replay --policy $pol --record-dir $M/formal_$pol/$task/$task \
    --replay-runs 0 --num_envs 10 ${4:+--depen "$4"} --cap-via prebuild --out $out --headless > $out/process.log 2>&1
  local code=$?
  log "replay $pol $task $tag exit $code"
  [ $code = 0 ] && touch $out/done
}

for spec in "gr00t BananaInBowlTask" "pi05 BananasInCrateTask" "pi05 BananasInBinOneMoreTask"; do
  set -- $spec
  replay $1 $2 nocap
  replay $1 $2 cap1000 1000
  replay $1 $2 cap3 3.0
  replay $1 $2 cap1 1.0
done
replay gr00t BananaInBowlTask cap1_banana "1.0,bodies=banana"
replay gr00t BananaInBowlTask cap1_banana+left_inner_finger "1.0,bodies=banana+robot/left_inner_finger"
replay pi05 BananasInCrateTask cap1_banana "1.0,bodies=banana"
replay pi05 BananasInCrateTask cap1_banana+banana_03 "1.0,bodies=banana+banana_03"
replay pi05 BananasInCrateTask cap1_banana_03 "1.0,bodies=banana_03"
replay pi05 BananasInCrateTask cap1_banana_03+purple_crate "1.0,bodies=banana_03+purple_crate"
replay pi05 BananasInBinOneMoreTask cap1_banana_01 "1.0,bodies=banana_01"
replay pi05 BananasInBinOneMoreTask cap1_banana_01+banana_03 "1.0,bodies=banana_01+banana_03"

# closed loop: own Pi0.5 server on GPU 3, restarted before every run, one client
SPID=
server_start() {
  cd $OP
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 setsid nohup .venv/bin/python scripts/serve_policy.py --port $PORT policy:checkpoint \
    --policy.config=pi05_droid_jointpos --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos > $O/closed/server_$1.log 2>&1 < /dev/null &
  SPID=$!
  cd /path/to/simuguard-robolab/RoboLab
  for i in $(seq 180); do (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break; sleep 5; done
  sleep 5
  log "server $1 pid $SPID up"
}
server_stop() {
  [ -n "$SPID" ] && { kill -- -$SPID 2>/dev/null || kill $SPID; }
  for i in $(seq 60); do kill -0 $SPID 2>/dev/null || break; sleep 2; done
  log "server pid $SPID stopped"
  SPID=
  sleep 5
}
closed() {  # tag [depen spec]
  local out=$O/closed/$1
  [ -f $out/done ] && return
  rm -rf $out; mkdir -p $out
  server_start $1
  log "closed $1 start"
  $PY -m simuguard.integrations.robolab_eval --policy pi05 --task BananasInCrateTask --num_envs 10 --num_runs 1 \
    --remote-port $PORT --policy-checkpoint "pi05_droid_jointpos (gs://openpi-assets-simeval)" ${2:+--depen "$2"} --cap-via prebuild \
    --out $out --headless --video-mode none > $out/process.log 2>&1
  local code=$?
  log "closed $1 exit $code"
  server_stop
  [ $code = 0 ] && touch $out/done
}
mkdir -p $O/closed
if [ ! -f $O/closed/step1.done ]; then
  server_start step1a
  $PY /path/to/simuguard-robolab/simuguard/pi05_action_check.py $O/closed/step1_fresh1.npy $PORT >> $O/p0.log 2>&1
  $PY /path/to/simuguard-robolab/simuguard/pi05_action_check.py $O/closed/step1_same_server2.npy $PORT >> $O/p0.log 2>&1
  server_stop
  server_start step1b
  $PY /path/to/simuguard-robolab/simuguard/pi05_action_check.py $O/closed/step1_fresh2.npy $PORT >> $O/p0.log 2>&1
  server_stop
  touch $O/closed/step1.done
fi
closed A
closed A2
closed C 1.0
log "p0 done"
