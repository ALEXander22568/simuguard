#!/usr/bin/env bash
# Repeated official evaluations for one simulation configuration.
#
# Usage: scale_campaign.sh OUT_ROOT CONFIG TEST_NUM REPEATS MODEL_GPU SIM_GPU [TASK]
#   CONFIG: none | solver_high | solver_default | mass_100g | mass_50g
# Each repeat is a full official evaluation of TEST_NUM episodes; seeds are
# paired across configurations afterwards (the expert check is nondeterministic,
# so the evaluated seed sets can differ slightly).
set -uo pipefail

OUT=${1:?usage: scale_campaign.sh OUT_ROOT CONFIG TEST_NUM REPEATS MODEL_GPU SIM_GPU [TASK]}
CONFIG=${2:?CONFIG required}
TEST_NUM=${3:?TEST_NUM required}
REPEATS=${4:?REPEATS required}
MODEL_GPU=${5:?MODEL_GPU required}
SIM_GPU=${6:?SIM_GPU required}
TASK=${7:-place_can_basket}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "${OUT}"
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "${OUT}/campaign.log"; }

CHILD=
cleanup() {
    [[ -n "${CHILD}" ]] && kill -TERM -- "-$(ps -o pgid= -p "${CHILD}" | tr -d ' ')" 2>/dev/null
    echo "STOPPED" >> "${OUT}/STATUS"
}
on_signal() { cleanup; exit 143; }
trap cleanup EXIT
trap on_signal INT TERM

log "config=${CONFIG} test_num=${TEST_NUM} repeats=${REPEATS} model_gpu=${MODEL_GPU} sim_gpu=${SIM_GPU}"
for repeat in $(seq 1 "${REPEATS}"); do
    dir="${OUT}/rep${repeat}"
    if [[ -f "${dir}/simuguard/run_summary.json" ]]; then
        log "rep${repeat}: already complete, skipping"
        continue
    fi
    echo "RUNNING rep${repeat}" > "${OUT}/STATUS"
    log "rep${repeat}: start"
    SIM_INTERVENTION="${CONFIG}" bash "${HERE}/run_lingbot_eval.sh" "${dir}" "${TEST_NUM}" "${TASK}" \
        "${MODEL_GPU}" "${SIM_GPU}" > "${OUT}/rep${repeat}.supervisor.log" 2>&1 &
    CHILD=$!
    rc=0
    wait "${CHILD}" || rc=$?
    CHILD=
    log "rep${repeat}: exit=${rc}"
done
echo "DONE" > "${OUT}/STATUS"
log "campaign finished"
