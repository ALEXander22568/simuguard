#!/usr/bin/env bash
# Wait until two GPUs have enough free memory, then run one LingBot evaluation.
#
# Usage: wait_for_gpu_then_run.sh OUT_DIR TEST_NUM [TASK]
#   MODEL_FREE_MIB (default 20000), SIM_FREE_MIB (default 6000), POLL_SECONDS (default 300)
#   MAX_WAIT_HOURS (default 48); all LINGBOT_* variables are passed through.
set -uo pipefail

OUT=${1:?usage: wait_for_gpu_then_run.sh OUT_DIR TEST_NUM [TASK]}
TEST_NUM=${2:?TEST_NUM required}
TASK=${3:-place_can_basket}
MODEL_FREE_MIB=${MODEL_FREE_MIB:-20000}
SIM_FREE_MIB=${SIM_FREE_MIB:-6000}
POLL_SECONDS=${POLL_SECONDS:-300}
MAX_WAIT_HOURS=${MAX_WAIT_HOURS:-48}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "${OUT}"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "${OUT}/wait.log"; }

deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
while :; do
    mapfile -t free < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -t, -rn)
    best=${free[0]}; second=${free[1]}
    model_gpu=${best%%,*}; model_free=$(echo "${best#*,}" | tr -d ' ')
    sim_gpu=${second%%,*}; sim_free=$(echo "${second#*,}" | tr -d ' ')
    if (( model_free >= MODEL_FREE_MIB )) && (( sim_free >= SIM_FREE_MIB )); then
        log "running on model GPU${model_gpu} (${model_free} MiB) sim GPU${sim_gpu} (${sim_free} MiB)"
        echo "RUNNING model_gpu=${model_gpu} sim_gpu=${sim_gpu}" > "${OUT}/STATUS"
        bash "${HERE}/run_lingbot_eval.sh" "${OUT}/eval" "${TEST_NUM}" "${TASK}" "${model_gpu}" "${sim_gpu}" \
            > "${OUT}/supervisor.log" 2>&1
        rc=$?
        log "eval exit=${rc}"
        echo "DONE exit=${rc}" > "${OUT}/STATUS"
        exit "${rc}"
    fi
    if (( $(date +%s) > deadline )); then
        log "timeout: best free ${model_free} MiB on GPU${model_gpu}"
        echo "TIMEOUT" > "${OUT}/STATUS"; exit 3
    fi
    log "waiting: best GPU${model_gpu}=${model_free} MiB, next GPU${sim_gpu}=${sim_free} MiB (need ${MODEL_FREE_MIB}/${SIM_FREE_MIB})"
    sleep "${POLL_SECONDS}"
done
