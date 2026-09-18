#!/usr/bin/env bash
# Official RoboTwin evaluation of LingBot-VA with SimuGuard monitoring.
# Upstream RoboTwin/XPolicyLab are not modified: the VA backend and XPolicyLab
# bridge use official launch semantics, and the evaluator is the official
# eval_policy_xpolicylab.py run through simuguard.integrations.robotwin_eval.
#
# Usage: run_lingbot_eval.sh RUN_DIR TEST_NUM [TASK] [MODEL_GPU] [SIM_GPU]
#   SIMUGUARD=0        run the official evaluator without monitoring
#   paths default to the 4090-hexa-node2 workspace; override via env vars below
set -Eeuo pipefail

RUN_DIR=${1:?usage: run_lingbot_eval.sh RUN_DIR TEST_NUM [TASK] [MODEL_GPU] [SIM_GPU]}
TEST_NUM=${2:?TEST_NUM required}
TASK=${3:-place_can_basket}
MODEL_GPU=${4:-0}
SIM_GPU=${5:-1}
SIMUGUARD=${SIMUGUARD:-1}
POLICY_REQUEST_TIMEOUT_S=${POLICY_REQUEST_TIMEOUT_S:-900}  # upstream client default 120 s is too short for first reset
# upstream = unmodified wan_va_server.py (prompt padded to 512 tokens, ~6.4 min CPU reset on node2)
# longest  = same server via simuguard/integrations/lingbot_va_server_shim.py (tokenizer padding override only)
LINGBOT_PROMPT_PADDING=${LINGBOT_PROMPT_PADDING:-longest}
# upstream keeps the VAE on CPU under offload (>14 min for the first action chunk on node2)
LINGBOT_VAE_DEVICE=${LINGBOT_VAE_DEVICE:-gpu_staged}
# robotwin30_train at the upstream window 72 needs a 10.1 GiB KV cache and does not fit a
# 24 GB card next to the 11 GB transformer; 48 -> 6.7 GiB but CHANGES POLICY BEHAVIOUR.
# The "robotwin" config used by the EE path already fits, so it keeps the upstream window.
if [[ "${LINGBOT_ACTION_PATH}" == "ee" ]]; then
    LINGBOT_ATTN_WINDOW=${LINGBOT_ATTN_WINDOW:-}
else
    LINGBOT_ATTN_WINDOW=${LINGBOT_ATTN_WINDOW:-48}
fi
MIN_FREE_MODEL_MIB=${MIN_FREE_MODEL_MIB:-20000}
MIN_FREE_SIM_MIB=${MIN_FREE_SIM_MIB:-7000}

BASE=${SIMUGUARD_WORKSPACE:-/mnt/nvme0/twinguar/simuguard}
REPO=${ROBOTWIN_ROOT:-${BASE}/RoboTwin}
SIMUGUARD_REPO=${SIMUGUARD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
POLICY_DIR=${REPO}/XPolicyLab/policy/LingBot_VA
LINGBOT_DIR=${POLICY_DIR}/lingbot_va
RUNTIME=${LINGBOT_RUNTIME:-/mnt/nvme0/twinguar/RoboTwin-2.0/runtime/lingbot-va-repro}
SERVER_PY=${BASE}/.venv-lingbot/bin/python   # overlay: runtime .venv-server site-packages + h5py
CLIENT_PY=${RUNTIME}/.venv-client/bin/python
MODEL=${LINGBOT_MODEL:-/mnt/nvme0/twinguar/models/lingbot-va-posttrain-robotwin}
TOOLS_BIN=${TOOLS_BIN:-/mnt/nvme0/twinguar/RoboTwin-2.0/.tools/bin}  # ffmpeg
# ee   : server config "robotwin" (16-dim relative EE actions, matches the posttrain
#        checkpoint) + bridge shim that converts them to RoboTwin ee actions
# joint: upstream default config robotwin30_train (30-dim, joint channels)
LINGBOT_ACTION_PATH=${LINGBOT_ACTION_PATH:-ee}
if [[ "${LINGBOT_ACTION_PATH}" == "ee" ]]; then
    CONFIG_NAME=${CONFIG_NAME:-robotwin}
else
    CONFIG_NAME=${CONFIG_NAME:-robotwin30_train}
fi
CACHE=${BASE}/.cache

mkdir -p "${RUN_DIR}" "${CACHE}/home"
VA_PID= ; BRIDGE_PID= ; EVAL_PID=

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
    stop_group "${EVAL_PID}"; stop_group "${BRIDGE_PID}"; stop_group "${VA_PID}"
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
        kill -0 "${pid}" 2>/dev/null || { echo "[run] ${name} exited before listening"; return 1; }
        sleep 3; t=$((t+3))
        (( t >= timeout )) && { echo "[run] ${name} not up after ${timeout}s"; return 1; }
    done
    echo "[run] ${name} up on ${port} after ${t}s"
}

# ---- provenance -----------------------------------------------------------
{
    echo "started_at=$(date --iso-8601=seconds)"
    echo "task=${TASK} test_num=${TEST_NUM} simuguard=${SIMUGUARD} model_gpu=${MODEL_GPU} sim_gpu=${SIM_GPU} config=${CONFIG_NAME}"
    echo "lingbot_action_path=${LINGBOT_ACTION_PATH} server_config=${CONFIG_NAME}"
    echo "lingbot_attn_window=${LINGBOT_ATTN_WINDOW:-upstream}"
    echo "lingbot_vae_device=${LINGBOT_VAE_DEVICE} lingbot_prompt_padding=${LINGBOT_PROMPT_PADDING} policy_request_timeout_s=${POLICY_REQUEST_TIMEOUT_S}"
    echo "simuguard_commit=$(git -C "${SIMUGUARD_REPO}" rev-parse HEAD 2>/dev/null) simuguard_dirty=$(git -C "${SIMUGUARD_REPO}" status --porcelain 2>/dev/null | wc -l)"
    echo "robotwin_commit=$(git -C "${REPO}" rev-parse HEAD)"
    echo "robotwin_dirty=$(git -C "${REPO}" status --porcelain | wc -l)"
    echo "xpolicylab_commit=$(git -C "${REPO}/XPolicyLab" rev-parse HEAD)"
    echo "xpolicylab_dirty=$(git -C "${REPO}/XPolicyLab" status --porcelain | wc -l)"
    echo "model=${MODEL}"
    echo "python_envs=server+bridge:${SERVER_PY} client:${CLIENT_PY}"
    echo "compat_shims=${BASE}/compat/flash_attn (import-only; server uses attn_mode=torch)"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
} > "${RUN_DIR}/provenance.txt"
cat "${RUN_DIR}/provenance.txt"

# ---- GPU guard -------------------------------------------------------------
fm=$(free_mib "${MODEL_GPU}"); fs=$(free_mib "${SIM_GPU}")
if (( fm < MIN_FREE_MODEL_MIB )) || (( fs < MIN_FREE_SIM_MIB )); then
    echo "[run] insufficient free memory: model GPU${MODEL_GPU}=${fm}MiB sim GPU${SIM_GPU}=${fs}MiB"
    exit 75
fi

VA_PORT=$(pick_port); VA_MASTER=$(pick_port); BR_PORT=$(pick_port); BR_MASTER=$(pick_port)

# ---- official merged checkpoint (symlinks only) ---------------------------
"${SERVER_PY}" "${POLICY_DIR}/prepare_merged_ckpt.py" \
    --checkpoint-path "${MODEL}" --base-model-path "${MODEL}" \
    --merged-dir "${POLICY_DIR}/.merged_ckpt" 2>&1 | tee "${RUN_DIR}/merged_ckpt.log"

# ---- 1) backend VA server (official launch semantics) ---------------------
if [[ "${LINGBOT_PROMPT_PADDING}" == "upstream" && "${LINGBOT_VAE_DEVICE}" == "upstream" ]]; then
    SERVER_ENTRY="wan_va/wan_va_server.py"   # unmodified upstream launch
else
    SERVER_ENTRY="${SIMUGUARD_REPO}/simuguard/integrations/lingbot_va_server_shim.py"
fi
[[ "${LINGBOT_PROMPT_PADDING}" == "upstream" ]] && LINGBOT_PROMPT_PADDING=max_length
cd "${LINGBOT_DIR}"
setsid env CUDA_VISIBLE_DEVICES="${MODEL_GPU}" MASTER_ADDR=127.0.0.1 MASTER_PORT="${VA_MASTER}" \
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${REPO}:${REPO}/XPolicyLab:${LINGBOT_DIR}:${BASE}/compat" \
    SIMUGUARD_WAN_VA_SERVER="${LINGBOT_DIR}/wan_va/wan_va_server.py" SIMUGUARD_PROMPT_PADDING="${LINGBOT_PROMPT_PADDING}" \
    SIMUGUARD_VAE_DEVICE="${LINGBOT_VAE_DEVICE}" SIMUGUARD_ATTN_WINDOW="${LINGBOT_ATTN_WINDOW}" \
    SIMUGUARD_MODEL_PATH="${POLICY_DIR}/.merged_ckpt" SIMUGUARD_ENABLE_OFFLOAD=1 \
    "${SERVER_PY}" -m torch.distributed.run --nproc_per_node=1 --master_port="${VA_MASTER}" \
        "${SERVER_ENTRY}" --config-name "${CONFIG_NAME}" --port "${VA_PORT}" \
        --save_root "${RUN_DIR}/va_visualization" \
    > "${RUN_DIR}/va_server.log" 2>&1 &
VA_PID=$!
wait_port "${VA_PORT}" "${VA_PID}" "VA backend" 1800

# ---- 2) XPolicyLab forward bridge (CPU, same env as VA backend per upstream) -----------------------------------
cd "${REPO}"
if [[ "${LINGBOT_ACTION_PATH}" == "ee" ]]; then
    BRIDGE_ENTRY=("${SIMUGUARD_REPO}/simuguard/integrations/lingbot_bridge_shim.py")
else
    BRIDGE_ENTRY=("XPolicyLab/setup_policy_server.py")
fi
setsid env CUDA_VISIBLE_DEVICES= MASTER_ADDR=127.0.0.1 MASTER_PORT="${BR_MASTER}" \
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 PYTHONPATH="${SIMUGUARD_REPO}:${REPO}:${REPO}/XPolicyLab:${BASE}/compat" \
    SIMUGUARD_XPOLICYLAB_SERVER="${REPO}/XPolicyLab/setup_policy_server.py" \
    "${SERVER_PY}" "${BRIDGE_ENTRY[@]}" \
        --config_path XPolicyLab/policy/LingBot_VA/deploy.yml \
        --overrides port="${BR_PORT}" host=127.0.0.1 protocol=ws bench_name=RoboTwin \
            task_name="${TASK}" ckpt_name="${MODEL}" checkpoint_path="${MODEL}" \
            base_model_path="${MODEL}" env_cfg_type=aloha_agilex env_cfg=aloha_agilex \
            seed=0 policy_name=LingBot_VA action_type=joint action_dim=14 \
            config_name="${CONFIG_NAME}" va_server_host=127.0.0.1 va_server_port="${VA_PORT}" \
    > "${RUN_DIR}/bridge.log" 2>&1 &
BRIDGE_PID=$!
wait_port "${BR_PORT}" "${BRIDGE_PID}" "LingBot bridge" 600

# ---- 3) official RoboTwin evaluator (optionally through SimuGuard) --------
OFFICIAL_ARGS=(
    --bench_name RoboTwin --task_name "${TASK}" --env_cfg_type aloha_agilex
    --policy_name LingBot_VA --host 127.0.0.1 --port "${BR_PORT}" --protocol ws
    --eval_batch false --root_dir "${REPO}" --device_id 0
    --additional_info "ckpt_name=${MODEL},action_type=joint"
    --seed 0 --test_num "${TEST_NUM}" --instruction_type seen --expert_check true
)
if [[ "${SIMUGUARD}" == "1" ]]; then
    EVAL_CMD=("${CLIENT_PY}" -m simuguard.integrations.robotwin_eval
        --robotwin-root "${REPO}" --simuguard-out "${RUN_DIR}/simuguard"
        --policy-request-timeout-s "${POLICY_REQUEST_TIMEOUT_S}" -- "${OFFICIAL_ARGS[@]}")
    EVAL_PYTHONPATH="${SIMUGUARD_REPO}:${REPO}:${REPO}/XPolicyLab"
else
    EVAL_CMD=("${CLIENT_PY}" scripts/eval_policy_xpolicylab.py "${OFFICIAL_ARGS[@]}")
    EVAL_PYTHONPATH="${REPO}:${REPO}/XPolicyLab"
fi
printf '%q ' "${EVAL_CMD[@]}" > "${RUN_DIR}/eval_command.txt"
cd "${REPO}"
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
