#!/usr/bin/env bash
# Checks before eval_batch policies run formally with the official num_envs (10), one GPU, Pi_05 / hang_mugs:
#   b10  num_envs 10 (scene.num_envs of sim_config.yml = what eval_policy.sh runs), first 10 layouts, with
#        --replay first (in-process self-replay of the batch): wall time per env-episode, peak memory
#   fb10 fresh-process replay of b10's first batch (all its segments): bit-exact check
#   wrist  mean luma of every view of each b10 episode video (black wrist cameras show as ~0)
# optional (EXTRA=1): s1 one-env run of layouts 0-3, x10 cross-batch replay of s1 in a 10-env scene,
#   r1 fresh-process one-env replay, s2 one-env repeat
# usage: bash batch_equiv.sh OUT GPU [SG_SRC_EPISODES] [SG_SRC_REPLAY]
set -uo pipefail
OUT=${1:?out dir}; GPU=${2:?gpu}
R=/path/to/simuguard-robodojo
SRC_EP=${3:-$R/simuguard-src-formal}; SRC_RP=${4:-$R/simuguard-src-next}
Q=$R/sgscripts-formal/rd_queue.py
PY=$R/miniconda3/envs/RoboDojo/bin/python
mkdir -p "$OUT"
log() { echo "$(TZ=Asia/Seoul date '+%m-%d %H:%M:%S') $*" | tee -a "$OUT/study.log"; }
queue() {  # name, extra queue args...
  local name=$1; shift
  log "$name: start"
  SG_SRC=$SRC_EP python3 -u "$Q" --run-dir "$OUT/$name" --gpus "$GPU" --policies Pi_05 --tasks hang_mugs \
    --max-counted 4 --eval-num-force --env-manifest "$OUT/env_manifest.json" --stall 7200 "$@" > "$OUT/$name.out" 2>&1
  log "$name: end rc=$?"
}
replay() {  # name, robodojo_eval args...
  local name=$1; shift
  log "$name: start"
  mkdir -p "$OUT/$name"
  (cd $R/RoboDojo && env HOME=$R HF_HOME=$R/hf/home OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PYTHONUNBUFFERED=1 \
      CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$R/RoboDojo:$R/RoboDojo/XPolicyLab:$SRC_RP" ROBODOJO_RUN_ID="beq_${name}_$$" \
      "$PY" -u -m simuguard.integrations.robodojo_eval --task hang_mugs --policy Pi_05 \
      --ckpt-name RoboDojo-sim-arx_x5-joint-0 --action-type joint --eval-seed 0 --out "$OUT/$name" \
      --env-manifest "$OUT/env_manifest.json" "$@") > "$OUT/$name.out" 2>&1
  log "$name: end rc=$?"
}
segs() { ls -d "$OUT"/s1/Pi_05/hang_mugs/segments/layout??? 2>/dev/null | head -n "${1:-4}"; }

segs() { ls -d "$OUT"/s1/Pi_05/hang_mugs/segments/layout??? 2>/dev/null | head -n "${1:-4}"; }
batch0() {  # segments of b10's first batch
  "$PY" - "$OUT/b10/Pi_05/hang_mugs/segments" <<'PY'
import json, sys
from pathlib import Path
segs = sorted(p for p in Path(sys.argv[1]).glob("layout???") if (p / "manifest.json").exists())
meta = {p: json.loads((p / "manifest.json").read_text())["metadata"] for p in segs}
first = meta[segs[0]]["batch"] if segs else None
print(" ".join(str(p) for p in segs if meta[p]["batch"] == first))
PY
}

python3 $R/sgscripts-formal/fingerprint_robodojo.py --policies Pi_05 --out "$OUT/env_manifest.json" > /dev/null 2>&1
queue b10 --batch-envs 10 --max-counted 10 -- --replay first
# shellcheck disable=SC2046
replay fb10 --num-envs 10 --replay-from $(batch0)
log "wrist: start"
"$PY" "$R/sgscripts-next/video_luma.py" "$OUT/b10" > "$OUT/wrist_luma.json" 2> "$OUT/wrist_luma.err"
log "wrist: end rc=$?"
if [ "${EXTRA:-0}" = 1 ]; then
  queue s1
  # shellcheck disable=SC2046
  replay x10 --num-envs 10 --replay-from $(segs 4)
  # shellcheck disable=SC2046
  replay r1 --replay-from $(segs 2)
  queue s2
fi
log "all done"
