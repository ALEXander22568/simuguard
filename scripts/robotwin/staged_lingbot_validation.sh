#!/usr/bin/env bash
# Wait for the policy/simulation GPUs, then run LingBot-VA through SimuGuard in gated stages:
#   stage1: official eval, test_num=1  -> gate: eval exit 0, 0 instrumentation/monitor errors,
#           >=1 policy segment, every segment replays bit-identically in a fresh process
#   stage2: official eval, test_num=10 -> same checks, reported (no further stage)
#
# Usage: staged_lingbot_validation.sh OUT_ROOT [TASK]
#   WAIT_FOR_STATE_FILE   file whose content must stop starting with RUNNING (optional)
#   MODEL_GPU/SIM_GPU     default 0/1; MIN_FREE_MODEL_MIB/MIN_FREE_SIM_MIB default 20000/8000
#   POLL_SECONDS          default 600; MAX_WAIT_HOURS default 96
set -uo pipefail

OUT=${1:?usage: staged_lingbot_validation.sh OUT_ROOT [TASK]}
TASK=${2:-place_can_basket}
MODEL_GPU=${MODEL_GPU:-0}
SIM_GPU=${SIM_GPU:-1}
MIN_FREE_MODEL_MIB=${MIN_FREE_MODEL_MIB:-20000}
MIN_FREE_SIM_MIB=${MIN_FREE_SIM_MIB:-8000}
POLL_SECONDS=${POLL_SECONDS:-600}
MAX_WAIT_HOURS=${MAX_WAIT_HOURS:-96}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SIMUGUARD_REPO=$(cd "${HERE}/../.." && pwd)
BASE=${SIMUGUARD_WORKSPACE:-/mnt/nvme0/twinguar/simuguard}
REPO=${ROBOTWIN_ROOT:-${BASE}/RoboTwin}
CLIENT_PY=${LINGBOT_RUNTIME:-/mnt/nvme0/twinguar/RoboTwin-2.0/runtime/lingbot-va-repro}/.venv-client/bin/python
CACHE=${BASE}/.cache
mkdir -p "${OUT}"
STATUS="${OUT}/STATUS"

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "${OUT}/staged.log"; }
CHILD_PID=
cleanup() {  # propagate termination to the launcher's process group
    local pid=${CHILD_PID:-}
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        local pgid; pgid=$(ps -o pgid= -p "${pid}" | tr -d " ")
        kill -TERM "${pid}" 2>/dev/null || true
        [[ -n "${pgid}" ]] && kill -TERM -- "-${pgid}" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "${pid}" 2>/dev/null || break; sleep 1; done
        [[ -n "${pgid}" ]] && kill -KILL -- "-${pgid}" 2>/dev/null || true
    fi
    echo "STOPPED" >> "${STATUS}"
}
on_signal() { cleanup; exit 143; }   # a bare trap would clean up and then keep running
trap cleanup EXIT
trap on_signal INT TERM
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }

# ---------------------------------------------------------------- wait for resources
echo "WAITING" > "${STATUS}"
deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
while :; do
    busy=""
    if [[ -n "${WAIT_FOR_STATE_FILE:-}" && -f "${WAIT_FOR_STATE_FILE}" ]] && grep -q '^RUNNING\|^STARTING' "${WAIT_FOR_STATE_FILE}"; then
        busy="state_file=$(head -c 40 "${WAIT_FOR_STATE_FILE}")"
    fi
    fm=$(free_mib "${MODEL_GPU}"); fs=$(free_mib "${SIM_GPU}")
    if [[ -z "${busy}" ]] && (( fm >= MIN_FREE_MODEL_MIB )) && (( fs >= MIN_FREE_SIM_MIB )); then
        log "resources ready: GPU${MODEL_GPU} free=${fm}MiB GPU${SIM_GPU} free=${fs}MiB"
        break
    fi
    if (( $(date +%s) > deadline )); then
        log "timeout waiting for resources (${busy} GPU${MODEL_GPU}=${fm} GPU${SIM_GPU}=${fs})"
        echo "TIMEOUT_WAITING" > "${STATUS}"; exit 3
    fi
    log "waiting: ${busy:-state_ok} GPU${MODEL_GPU} free=${fm}MiB GPU${SIM_GPU} free=${fs}MiB"
    sleep "${POLL_SECONDS}"
done

# ---------------------------------------------------------------- stage runner + gate
run_stage() {
    local name=$1 test_num=$2 dir="${OUT}/$1"
    echo "RUNNING_${name}" > "${STATUS}"
    log "${name}: official eval test_num=${test_num}"
    bash "${HERE}/run_lingbot_eval.sh" "${dir}" "${test_num}" "${TASK}" "${MODEL_GPU}" "${SIM_GPU}" > "${dir}.supervisor.log" 2>&1 &
    CHILD_PID=$!
    local rc=0
    wait "${CHILD_PID}" || rc=$?
    CHILD_PID=
    log "${name}: eval exit=${rc}"
    local replay_fail=0
    for seg in "${dir}"/simuguard/segments/*; do
        [[ -d "${seg}" && -f "${seg}/controls.npz" ]] || continue
        local n; n=$(basename "${seg}")
        env CUDA_VISIBLE_DEVICES="${SIM_GPU}" HOME="${CACHE}/home" XDG_CACHE_HOME="${CACHE}" PYTHONPATH="${SIMUGUARD_REPO}" \
            "${CLIENT_PY}" -B "${SIMUGUARD_REPO}/scripts/validate_robotwin_eval_wrapper.py" replay \
            --robotwin-root "${REPO}" --segment-dir "${seg}" --report "${dir}/replay_${n}.json" > "${dir}/replay_${n}.log" 2>&1
        local r=$?
        log "${name}: replay ${n} exit=${r}"
        (( r == 0 )) || replay_fail=$((replay_fail + 1))
    done
    "${CLIENT_PY}" - "${dir}" "${rc}" "${replay_fail}" > "${dir}/gate.json" <<'PY'
import glob, json, sys
d, rc, replay_fail = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
summary = json.load(open(f"{d}/simuguard/run_summary.json")) if glob.glob(f"{d}/simuguard/run_summary.json") else {}
segments = [json.loads(l) for l in open(f"{d}/simuguard/segments.jsonl")] if glob.glob(f"{d}/simuguard/segments.jsonl") else []
policy = [s for s in segments if s["phase"] == "policy"]
gate = {
    "eval_exit": rc,
    "instrumentation_errors": len(summary.get("instrumentation_errors", [None])),
    "segments": len(segments),
    "policy_segments": len(policy),
    "monitor_errors": sum(s.get("monitor", {}).get("error_count", 1) for s in segments),
    "replay_failures": replay_fail,
    "policy_success": [s["outcome"].get("eval_success") for s in policy],
    "policy_substeps": [s.get("monitor", {}).get("substeps") for s in policy],
    "confirmed_events": [s.get("monitor", {}).get("confirmed_count") for s in segments],
    "artifact_bytes": [s.get("artifact_bytes", {}).get("_total") for s in segments],
}
gate["pass"] = rc == 0 and gate["instrumentation_errors"] == 0 and gate["monitor_errors"] == 0 and len(policy) >= 1 and replay_fail == 0
print(json.dumps(gate, indent=1))
PY
    log "${name}: gate $(tr -d '\n ' < "${dir}/gate.json")"
    grep -q '"pass": true' "${dir}/gate.json"
}

if run_stage stage1_1episode 1; then
    if run_stage stage2_10seeds 10; then
        echo "DONE_STAGE2_PASS" > "${STATUS}"
    else
        echo "DONE_STAGE2_FAIL" > "${STATUS}"
    fi
else
    echo "STOPPED_STAGE1_FAIL" > "${STATUS}"
fi
log "finished: $(cat "${STATUS}")"
