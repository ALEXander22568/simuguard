#!/usr/bin/env bash
# Evaluate several tasks with an XPolicyLab policy, one lane per (policy GPU, simulator GPU) pair on
# this machine.  Same queue, manifest and output layout as parallel_campaign.sh, so gravity_filter.py
# and results_table.py work unchanged.
#
# Usage: xpolicylab_campaign.sh OUT TEST_NUM "POLICY_GPU:SIM_GPU [POLICY_GPU:SIM_GPU ...]" TASK [TASK ...]
#   A lane starts a fresh policy server for every task (run_xpolicylab_eval.sh), so every task run
#   begins from a freshly loaded policy.  A task already holding manifest.json is skipped.
# Env: ROBOTWIN_ROOT (required), MANIFEST_PYTHON, plus everything run_xpolicylab_eval.sh reads
#      (POLICY_NAME, DEPLOY_YML, POLICY_PY, POLICY_PYTHONPATH, CKPT_NAME, ACTION_TYPE, ...).
set -uo pipefail

OUT="$1"; TEST_NUM="$2"; LANES="$3"; shift 3
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:?set ROBOTWIN_ROOT}"
PYTHON="${MANIFEST_PYTHON:-python3}"
mkdir -p "${OUT}"
QUEUE="${OUT}/.queue"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "${OUT}/campaign.log"; }

( flock 9; for t in "$@"; do [[ -f "${OUT}/${t}/manifest.json" ]] || echo "${t}"; done > "${QUEUE}" ) 9> "${QUEUE}.lock"

next_task() {
    local t
    exec 8> "${QUEUE}.lock"; flock 8
    t=$(head -n 1 "${QUEUE}")
    [[ -n "${t}" ]] && sed -i '1d' "${QUEUE}"
    flock -u 8
    [[ -n "${t}" ]] && echo "${t}"
}
requeue() { ( flock 8; echo "$1" >> "${QUEUE}" ) 8> "${QUEUE}.lock"; }

lane() {
    local pgpu=$1 sgpu=$2 name="p${1}s${2}" task run started rc
    while task=$(next_task); do
        run="${OUT}/${task}"
        [[ -f "${run}/manifest.json" ]] && continue
        mkdir -p "${run}"
        echo "RUNNING ${task} policy_gpu=${pgpu} sim_gpu=${sgpu}" > "${OUT}/STATUS.lane${name}"
        log "${task}: start (${TEST_NUM} seeds) lane ${name}"
        started=$(date +%s)
        bash "${HERE}/run_xpolicylab_eval.sh" "${run}" "${TEST_NUM}" "${task}" "${pgpu}" "${sgpu}" > "${run}/supervisor.log" 2>&1
        rc=$?
        log "${task}: exit=${rc} after $(( ($(date +%s) - started) / 60 )) min (lane ${name})"
        if [[ "${rc}" -eq 75 ]]; then
            log "${task}: not enough GPU memory at start; requeued"; requeue "${task}"; sleep 120; continue
        elif [[ "${rc}" -ne 0 ]]; then
            log "${task}: FAILED, see ${run}/supervisor.log (no manifest written, a rerun retries it)"; continue
        fi
        "${PYTHON}" "${HERE}/build_run_manifest.py" --run-dir "${run}" --task "${task}" \
            --robotwin-root "${ROBOTWIN_ROOT}" --started-epoch "${started}" >> "${OUT}/campaign.log" 2>&1
    done
    echo "IDLE" > "${OUT}/STATUS.lane${name}"
}

log "campaign: $(wc -l < "${QUEUE}") tasks, lanes: ${LANES}, policy: ${POLICY_NAME:-?}"
pids=()
for spec in ${LANES}; do
    lane "${spec%%:*}" "${spec##*:}" &
    pids+=($!)
    sleep 60   # stagger model loading and simulator start-up
done
wait "${pids[@]}"
echo "DONE" > "${OUT}/STATUS"
log "campaign finished"
