#!/usr/bin/env bash
# Official RoboTwin evaluation of an XPolicyLab policy with SimuGuard monitoring.
#
# The policy server is XPolicyLab/setup_policy_server.py in the policy's own environment; the
# evaluator is the official eval_policy_xpolicylab.py run through simuguard.integrations.robotwin_eval
# with exactly the arguments run_lingbot_eval.sh uses (seed 0, seen instructions, expert check,
# demo_clean), so runs of different policies are comparable seed for seed.
#
# Usage: run_xpolicylab_eval.sh RUN_DIR TEST_NUM TASK POLICY_GPU SIM_GPU
#
# Env (policy):
#   POLICY_NAME         XPolicyLab policy directory name, e.g. Hy_Embodied_05_VLA (required)
#   DEPLOY_YML          deploy config, absolute or relative to the RoboTwin root (required)
#   POLICY_PY           python of the policy environment (required)
#   POLICY_PYTHONPATH   extra import roots for the server, ':'-separated (e.g. the policy's source tree)
#   POLICY_OVERRIDES    extra key=value overrides for setup_policy_server.py (space-separated)
#   CKPT_NAME           recorded in --additional_info (default: policy name)
#   ACTION_TYPE         ee | joint (default ee)
#   POLICY_REMOTE_PORT  do not start a server; talk to 127.0.0.1:PORT (an SSH tunnel)
#   SERVERS_ONLY=1      start the server, write RUN_DIR/ports.env and wait
# Env (shared with run_lingbot_eval.sh): SIMUGUARD_WORKSPACE, ROBOTWIN_ROOT, LINGBOT_RUNTIME (client
#   env), CLIENT_PY, TOOLS_BIN, SIMUGUARD_CONFIG, SIMUGUARD=1, SIM_INTERVENTION, POLICY_REQUEST_TIMEOUT_S,
#   MIN_FREE_POLICY_MIB (default 14000), MIN_FREE_SIM_MIB (default 7000)
set -Eeuo pipefail

RUN_DIR=${1:?usage: run_xpolicylab_eval.sh RUN_DIR TEST_NUM TASK POLICY_GPU SIM_GPU}
TEST_NUM=${2:?TEST_NUM required}
TASK=${3:?TASK required}
POLICY_GPU=${4:-0}
SIM_GPU=${5:-1}
POLICY_NAME=${POLICY_NAME:?set POLICY_NAME}
DEPLOY_YML=${DEPLOY_YML:?set DEPLOY_YML}
POLICY_PY=${POLICY_PY:?set POLICY_PY}
POLICY_PYTHONPATH=${POLICY_PYTHONPATH:-}
POLICY_OVERRIDES=${POLICY_OVERRIDES:-}
CKPT_NAME=${CKPT_NAME:-${POLICY_NAME}}
ACTION_TYPE=${ACTION_TYPE:-ee}
POLICY_REMOTE_PORT=${POLICY_REMOTE_PORT:-}
SERVERS_ONLY=${SERVERS_ONLY:-0}
SIMUGUARD=${SIMUGUARD:-1}
SIM_INTERVENTION=${SIM_INTERVENTION:-none}
POLICY_REQUEST_TIMEOUT_S=${POLICY_REQUEST_TIMEOUT_S:-900}
MIN_FREE_POLICY_MIB=${MIN_FREE_POLICY_MIB:-14000}
MIN_FREE_SIM_MIB=${MIN_FREE_SIM_MIB:-7000}
SIMUGUARD_CONFIG=${SIMUGUARD_CONFIG:-}

BASE=${SIMUGUARD_WORKSPACE:-/mnt/nvme0/twinguar/simuguard}
REPO=${ROBOTWIN_ROOT:-${BASE}/RoboTwin}
SIMUGUARD_REPO=${SIMUGUARD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
RUNTIME=${LINGBOT_RUNTIME:-/mnt/nvme0/twinguar/RoboTwin-2.0/runtime/lingbot-va-repro}
CLIENT_PY=${CLIENT_PY:-${RUNTIME}/.venv-client/bin/python}
TOOLS_BIN=${TOOLS_BIN:-/mnt/nvme0/twinguar/RoboTwin-2.0/.tools/bin}
CACHE=${BASE}/.cache
mkdir -p "${RUN_DIR}" "${CACHE}/home"
SERVER_PID= ; EVAL_PID=

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" | tr -d ' '; }
stop_group() {
    local pid=${1:-}
    [[ -z "${pid}" ]] && return 0
    kill -0 "${pid}" 2>/dev/null || return 0
    local pgid; pgid=$(ps -o pgid= -p "${pid}" | tr -d ' ')
    [[ -z "${pgid}" ]] && return 0
    kill -TERM -- "-${pgid}" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "${pid}" 2>/dev/null || return 0; sleep 1; done
    kill -KILL -- "-${pgid}" 2>/dev/null || true
}
cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    stop_group "${EVAL_PID}"; stop_group "${SERVER_PID}"
    echo "${rc}" > "${RUN_DIR}/exit_code"
    echo "[run] finished rc=${rc} at $(date --iso-8601=seconds)"
    exit "${rc}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
pick_port() { "${CLIENT_PY}" -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])'; }
wait_port() {
    local port=$1 pid=$2 name=$3 timeout=$4 t=0
    until "${CLIENT_PY}" -c "import socket,sys;s=socket.socket();s.settimeout(1);sys.exit(s.connect_ex(('127.0.0.1',${port})))"; do
        if [[ -n "${pid}" ]]; then
            kill -0 "${pid}" 2>/dev/null || { echo "[run] ${name} exited before listening"; return 1; }
        fi
        sleep 3; t=$((t+3))
        (( t >= timeout )) && { echo "[run] ${name} not up after ${timeout}s"; return 1; }
    done
    echo "[run] ${name} up on ${port} after ${t}s"
}

# ---- provenance -----------------------------------------------------------
{
    echo "started_at=$(date --iso-8601=seconds)"
    echo "task=${TASK} test_num=${TEST_NUM} simuguard=${SIMUGUARD} policy_gpu=${POLICY_GPU} sim_gpu=${SIM_GPU}"
    echo "policy=${POLICY_NAME} deploy_yml=${DEPLOY_YML} action_type=${ACTION_TYPE} ckpt_name=${CKPT_NAME}"
    echo "policy_overrides=${POLICY_OVERRIDES}"
    echo "sim_intervention=${SIM_INTERVENTION} policy_request_timeout_s=${POLICY_REQUEST_TIMEOUT_S}"
    echo "simuguard_commit=$(git -C "${SIMUGUARD_REPO}" rev-parse HEAD 2>/dev/null) simuguard_dirty=$(git -C "${SIMUGUARD_REPO}" status --porcelain 2>/dev/null | wc -l)"
    echo "robotwin_commit=$(git -C "${REPO}" rev-parse HEAD)"
    echo "robotwin_dirty=$(git -C "${REPO}" status --porcelain | wc -l)"
    echo "xpolicylab_commit=$(git -C "${REPO}/XPolicyLab" rev-parse HEAD)"
    echo "xpolicylab_dirty=$(git -C "${REPO}/XPolicyLab" status --porcelain | wc -l)"
    if [[ -n "${POLICY_REMOTE_PORT}" ]]; then
        echo "policy_server=remote local_port=${POLICY_REMOTE_PORT}"
    else
        echo "policy_server=local host=$(hostname) python=${POLICY_PY}"
    fi
    echo "client_python=${CLIENT_PY}"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
} > "${RUN_DIR}/provenance.txt"
cat "${RUN_DIR}/provenance.txt"

# ---- GPU guard -------------------------------------------------------------
fp=$( [[ -n "${POLICY_REMOTE_PORT}" ]] && echo 999999 || free_mib "${POLICY_GPU}" )
fs=$( [[ "${SERVERS_ONLY}" == "1" ]] && echo 999999 || free_mib "${SIM_GPU}" )
if (( fp < MIN_FREE_POLICY_MIB )) || (( fs < MIN_FREE_SIM_MIB )); then
    echo "[run] insufficient free memory: policy GPU${POLICY_GPU}=${fp}MiB sim GPU${SIM_GPU}=${fs}MiB"
    exit 75
fi

# ---- 1) policy server --------------------------------------------------------
cd "${REPO}"
if [[ -n "${POLICY_REMOTE_PORT}" ]]; then
    PORT=${POLICY_REMOTE_PORT}
else
    PORT=$(pick_port)
    action_dim=$(bash XPolicyLab/utils/get_action_dim.sh "${REPO}" aloha_agilex 2>/dev/null || echo 14)
    # shellcheck disable=SC2206
    extra=( ${POLICY_OVERRIDES} )
    setsid env CUDA_VISIBLE_DEVICES="${POLICY_GPU}" PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore::UserWarning \
        PYTHONPATH="${REPO}${POLICY_PYTHONPATH:+:${POLICY_PYTHONPATH}}" \
        "${POLICY_PY}" XPolicyLab/setup_policy_server.py --config_path "${DEPLOY_YML}" \
            --overrides port="${PORT}" host=127.0.0.1 protocol=ws bench_name=RoboTwin task_name="${TASK}" \
                ckpt_name="${CKPT_NAME}" env_cfg_type=aloha_agilex seed=0 policy_name="${POLICY_NAME}" \
                action_type="${ACTION_TYPE}" action_dim="${action_dim}" "${extra[@]}" \
        > "${RUN_DIR}/policy_server.log" 2>&1 &
    SERVER_PID=$!
    wait_port "${PORT}" "${SERVER_PID}" "policy server" 1800
fi
if [[ "${SERVERS_ONLY}" == "1" ]]; then
    printf 'POLICY_PORT=%s\nSERVER_PID=%s\n' "${PORT}" "${SERVER_PID}" > "${RUN_DIR}/ports.env"
    echo "[run] server ready on ${PORT} (ports.env written); waiting"
    wait "${SERVER_PID}"
    exit $?
fi

# ---- 2) official RoboTwin evaluator (optionally through SimuGuard) --------
OFFICIAL_ARGS=(
    --bench_name RoboTwin --task_name "${TASK}" --env_cfg_type aloha_agilex
    --policy_name "${POLICY_NAME}" --host 127.0.0.1 --port "${PORT}" --protocol ws
    --eval_batch false --root_dir "${REPO}" --device_id 0
    --additional_info "ckpt_name=${CKPT_NAME},action_type=${ACTION_TYPE}"
    --seed 0 --test_num "${TEST_NUM}" --instruction_type seen --expert_check true
)
if [[ "${SIMUGUARD}" == "1" ]]; then
    EVAL_CMD=("${CLIENT_PY}" -m simuguard.integrations.robotwin_eval
        --robotwin-root "${REPO}" --simuguard-out "${RUN_DIR}/simuguard"
        --policy-request-timeout-s "${POLICY_REQUEST_TIMEOUT_S}"
        --sim-intervention "${SIM_INTERVENTION}"
        ${SIMUGUARD_CONFIG:+--simuguard-config "${SIMUGUARD_CONFIG}"} -- "${OFFICIAL_ARGS[@]}")
    EVAL_PYTHONPATH="${SIMUGUARD_REPO}:${REPO}:${REPO}/XPolicyLab"
else
    EVAL_CMD=("${CLIENT_PY}" scripts/eval_policy_xpolicylab.py "${OFFICIAL_ARGS[@]}")
    EVAL_PYTHONPATH="${REPO}:${REPO}/XPolicyLab"
fi
printf '%q ' "${EVAL_CMD[@]}" > "${RUN_DIR}/eval_command.txt"
setsid env CUDA_VISIBLE_DEVICES="${SIM_GPU}" PATH="${TOOLS_BIN}:${PATH}" HOME="${CACHE}/home" XDG_CACHE_HOME="${CACHE}" \
    MPLCONFIGDIR="${CACHE}/matplotlib" NUMBA_CACHE_DIR="${CACHE}/numba" \
    PYTHONPATH="${EVAL_PYTHONPATH}" "${EVAL_CMD[@]}" \
    > "${RUN_DIR}/eval.log" 2>&1 &
EVAL_PID=$!
echo "[run] eval pid=${EVAL_PID}"
eval_rc=0
wait "${EVAL_PID}" || eval_rc=$?
EVAL_PID=
echo "[run] eval exit=${eval_rc}"
grep -h "_result.txt\|Success rate\|success rate" "${RUN_DIR}/eval.log" | tail -3 || true
exit "${eval_rc}"
