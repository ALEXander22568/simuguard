#!/usr/bin/env bash
# Start one Pi0.5 LIBERO policy server (RLinf-Pi05-LIBERO-130-fullshot-SFT behind RPent's
# pi05_vla_server.py: HTTP POST /call, method "vla.predict", 5 actions of 7 per query).
#
# Usage: pi05_server.sh GPU PORT LOG
# Env (defaults are h800-2's): RPENT (checkout providing rpent/robots/components/pi05_vla_server.py),
#   RPENT_PY (python with RLinf + openpi), PI05_CHECKPOINT_PATH.
# Needs ~8.2 GB on the GPU.  Stop it with: kill $(cat LOG.pid)
set -euo pipefail
GPU="$1"; PORT="$2"; LOG="$3"
RPENT=${RPENT:-/home/zhoujingjing/rpent-captrain}
RPENT_PY=${RPENT_PY:-/data/shared/zhoujingjing/rpent-venv/bin/python}
export PI05_CHECKPOINT_PATH=${PI05_CHECKPOINT_PATH:-/data/shared/zhoujingjing/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT}
export PYTHONNOUSERSITE=1 PYTHONPATH="${RPENT}" HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
cd "${RPENT}"
nohup setsid "${RPENT_PY}" rpent/robots/components/pi05_vla_server.py --embodiment libero --transport http \
    --host 127.0.0.1 --port "${PORT}" --cuda-device "${GPU}" > "${LOG}" 2>&1 < /dev/null &
echo $! > "${LOG}.pid"
echo "pi05 server pid $(cat "${LOG}.pid") gpu ${GPU} port ${PORT} log ${LOG}"
