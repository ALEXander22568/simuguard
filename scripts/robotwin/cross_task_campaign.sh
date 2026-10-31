#!/usr/bin/env bash
# Evaluate one policy on several RoboTwin tasks under SimuGuard, one task after another,
# and leave behind everything needed to trace a result back: the seeds that were scored,
# the official videos, and the recorded actuation that replays each episode exactly.
#
# Usage: cross_task_campaign.sh OUT TEST_NUM MODEL_GPU SIM_GPU TASK [TASK ...]
#   MODEL_GPU / SIM_GPU may be "auto": before every task the script then waits until one GPU
#   has MODEL_FREE_MIB free (default 19500; the LingBot server holds about 18 GB) and another
#   has SIM_FREE_MIB (default 8000), and uses those.  On a shared machine the free GPU changes
#   from one hour to the next, so the choice is made per task rather than once.
set -uo pipefail

OUT="$1"; TEST_NUM="$2"; MODEL_GPU="$3"; SIM_GPU="$4"; shift 4
TASKS=("$@")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:?set ROBOTWIN_ROOT}"
PYTHON="${MANIFEST_PYTHON:-python3}"
MODEL_FREE_MIB="${MODEL_FREE_MIB:-19500}"
SIM_FREE_MIB="${SIM_FREE_MIB:-8000}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-72}"
mkdir -p "${OUT}"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "${OUT}/campaign.log"; }

# sets PICK_MODEL / PICK_SIM; returns 1 on timeout
pick_gpus() {
    local deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 )) announced=0
    while :; do
        mapfile -t free < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -rn)
        local best="${free[0]}" second="${free[1]}"
        local model_free sim_free
        PICK_MODEL="${best%%,*}";  model_free="$(echo "${best#*,}" | tr -d ' ')"
        PICK_SIM="${second%%,*}";  sim_free="$(echo "${second#*,}" | tr -d ' ')"
        if (( model_free >= MODEL_FREE_MIB )) && (( sim_free >= SIM_FREE_MIB )); then
            log "GPUs: model GPU${PICK_MODEL} (${model_free} MiB free), sim GPU${PICK_SIM} (${sim_free} MiB free)"
            return 0
        fi
        if (( announced % 30 == 0 )); then
            log "waiting for a GPU: best GPU${PICK_MODEL}=${model_free} MiB, need ${MODEL_FREE_MIB}"
        fi
        announced=$(( announced + 1 ))
        echo "WAITING_FOR_GPU best=${model_free}MiB need=${MODEL_FREE_MIB}MiB" > "${OUT}/STATUS"
        (( $(date +%s) > deadline )) && return 1
        sleep "${POLL_SECONDS}"
    done
}

for TASK in "${TASKS[@]}"; do
    RUN="${OUT}/${TASK}"
    if [[ -f "${RUN}/manifest.json" ]]; then
        log "${TASK}: already done, skipping"
        continue
    fi
    mkdir -p "${RUN}"
    USE_MODEL="${MODEL_GPU}"; USE_SIM="${SIM_GPU}"
    if [[ "${MODEL_GPU}" == "auto" || "${SIM_GPU}" == "auto" ]]; then
        if ! pick_gpus; then
            log "${TASK}: gave up waiting for a GPU after ${MAX_WAIT_HOURS} h"
            echo "TIMEOUT waiting for GPU" > "${OUT}/STATUS"
            exit 3
        fi
        USE_MODEL="${PICK_MODEL}"; USE_SIM="${PICK_SIM}"
    fi
    echo "RUNNING ${TASK} model_gpu=${USE_MODEL} sim_gpu=${USE_SIM}" > "${OUT}/STATUS"
    log "${TASK}: start (${TEST_NUM} seeds) on model GPU${USE_MODEL}, sim GPU${USE_SIM}"
    STARTED="$(date +%s)"
    MIN_FREE_MODEL_MIB="${MIN_FREE_MODEL_MIB:-19000}" \
    bash "${HERE}/run_lingbot_eval.sh" "${RUN}" "${TEST_NUM}" "${TASK}" "${USE_MODEL}" "${USE_SIM}" \
        > "${RUN}/supervisor.log" 2>&1
    RC=$?
    log "${TASK}: exit=${RC} after $(( ($(date +%s) - STARTED) / 60 )) min"
    if [[ "${RC}" -ne 0 ]]; then
        log "${TASK}: FAILED, see ${RUN}/supervisor.log (no manifest written, so a rerun retries it)"
        continue
    fi
    "${PYTHON}" "${HERE}/build_run_manifest.py" --run-dir "${RUN}" --task "${TASK}" \
        --robotwin-root "${ROBOTWIN_ROOT}" --started-epoch "${STARTED}" >> "${OUT}/campaign.log" 2>&1
done
echo "DONE" > "${OUT}/STATUS"
log "campaign finished"
