#!/usr/bin/env bash
# Evaluate several tasks at once, one lane per remote LingBot-VA backend (split deployment:
# the model runs on the inference host, the simulator here).
#
# Usage: parallel_campaign.sh OUT TEST_NUM "SLOT:PORT [SLOT:PORT ...]" TASK [TASK ...]
#
# Each lane takes the next task from a shared queue and, for every task:
#   1. restarts its backend on the inference host (va_ctl.sh restart SLOT TASK), so every task
#      run starts from a fresh policy server exactly as in the single-machine campaign;
#   2. picks the simulator GPU with the most free memory (reservations keep lanes that start
#      together from piling onto one card);
#   3. runs run_lingbot_eval.sh with VA_REMOTE_PORT=PORT (the SSH tunnel from va_tunnels.sh);
#   4. writes manifest.json / SEEDS.md / videos exactly like cross_task_campaign.sh.
# A task already holding manifest.json is skipped.  Output layout is the same as
# cross_task_campaign.sh, so gravity_filter.py / results_table.py / run_tiers.sh work unchanged.
#
# Env: ROBOTWIN_ROOT (required), MANIFEST_PYTHON, SIM_GPUS (allowed simulator GPUs, default all),
#      SIM_FREE_MIB (8000), VA_SSH_CONFIG, VA_SSH_HOST, VA_CTL_PATH (va_ctl.sh on the inference
#      host), plus whatever run_lingbot_eval.sh reads.
set -uo pipefail

OUT="$1"; TEST_NUM="$2"; LANES="$3"; shift 3
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:?set ROBOTWIN_ROOT}"
PYTHON="${MANIFEST_PYTHON:-python3}"
SIM_FREE_MIB="${SIM_FREE_MIB:-8000}"
RESERVE_MIB="${RESERVE_MIB:-6500}"      # what one simulator takes once it is up
RESERVE_SECONDS="${RESERVE_SECONDS:-300}"
CONFIG=${VA_SSH_CONFIG:-$HOME/.ssh/config.simuguard}
HOST=${VA_SSH_HOST:-h800-2-sg}
VA_CTL_PATH=${VA_CTL_PATH:-/data/shared/zhoujingjing/simuguard-va/SimuGuard/scripts/robotwin/va_ctl.sh}
mkdir -p "${OUT}"
QUEUE="${OUT}/.queue"; RES="${OUT}/.gpu_reservations"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "${OUT}/campaign.log"; }

( flock 9; for t in "$@"; do [[ -f "${OUT}/${t}/manifest.json" ]] || echo "${t}"; done > "${QUEUE}" ) 9> "${QUEUE}.lock"
touch "${RES}"

next_task() {
    local t
    exec 8> "${QUEUE}.lock"; flock 8
    t=$(head -n 1 "${QUEUE}")
    [[ -n "${t}" ]] && sed -i '1d' "${QUEUE}"
    flock -u 8
    [[ -n "${t}" ]] && echo "${t}"
}
requeue() { ( flock 8; echo "$1" >> "${QUEUE}" ) 8> "${QUEUE}.lock"; }

pick_sim_gpu() {
    local now best bestfree idx free reserved eff
    while :; do
        exec 7> "${RES}.lock"; flock 7
        now=$(date +%s); best=; bestfree=-1
        while read -r idx free; do
            if [[ -n "${SIM_GPUS:-}" ]] && [[ " ${SIM_GPUS} " != *" ${idx} "* ]]; then continue; fi
            reserved=$(awk -v g="${idx}" -v t=$(( now - RESERVE_SECONDS )) '$2 == g && $1 > t' "${RES}" | wc -l)
            eff=$(( free - reserved * RESERVE_MIB ))
            (( eff > bestfree )) && { best=${idx}; bestfree=${eff}; }
        done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
        if (( bestfree >= SIM_FREE_MIB )); then
            echo "${now} ${best}" >> "${RES}"; flock -u 7; echo "${best}"; return 0
        fi
        flock -u 7; sleep 120
    done
}

lane() {
    local slot=$1 port=$2 task run gpu started rc
    while task=$(next_task); do
        run="${OUT}/${task}"
        [[ -f "${run}/manifest.json" ]] && continue
        mkdir -p "${run}"
        log "${task}: lane ${slot} restarting backend (port ${port})"
        if ! ssh -F "${CONFIG}" -o BatchMode=yes "${HOST}" "${VA_CTL_PATH}" restart "${slot}" "${task}" >> "${run}/va_ctl.log" 2>&1; then
            log "${task}: lane ${slot} backend did not come up (see ${run}/va_ctl.log); requeued"
            requeue "${task}"; sleep 300; continue
        fi
        gpu=$(pick_sim_gpu)
        echo "RUNNING ${task} slot=${slot} port=${port} sim_gpu=${gpu}" > "${OUT}/STATUS.lane${slot}"
        log "${task}: start (${TEST_NUM} seeds) lane ${slot}, sim GPU${gpu}"
        started=$(date +%s)
        VA_REMOTE_PORT="${port}" VA_REMOTE_DESC="host=${HOST} slot=${slot}" \
            bash "${HERE}/run_lingbot_eval.sh" "${run}" "${TEST_NUM}" "${task}" remote "${gpu}" \
            > "${run}/supervisor.log" 2>&1
        rc=$?
        log "${task}: exit=${rc} after $(( ($(date +%s) - started) / 60 )) min (lane ${slot})"
        if [[ "${rc}" -eq 75 ]]; then
            log "${task}: not enough GPU memory at start; requeued"; requeue "${task}"; sleep 120; continue
        elif [[ "${rc}" -ne 0 ]]; then
            log "${task}: FAILED, see ${run}/supervisor.log (no manifest written, a rerun retries it)"; continue
        fi
        "${PYTHON}" "${HERE}/build_run_manifest.py" --run-dir "${run}" --task "${task}" \
            --robotwin-root "${ROBOTWIN_ROOT}" --started-epoch "${started}" >> "${OUT}/campaign.log" 2>&1
    done
    ssh -F "${CONFIG}" -o BatchMode=yes "${HOST}" "${VA_CTL_PATH}" stop "${slot}" >> "${OUT}/campaign.log" 2>&1
    echo "IDLE" > "${OUT}/STATUS.lane${slot}"
}

log "campaign: $(wc -l < "${QUEUE}") tasks, lanes: ${LANES}"
pids=()
for spec in ${LANES}; do
    lane "${spec%%:*}" "${spec##*:}" &
    pids+=($!)
    sleep 60   # stagger backend restarts and simulator start-up
done
wait "${pids[@]}"
echo "DONE" > "${OUT}/STATUS"
log "campaign finished"
