#!/usr/bin/env bash
# Start one Pi0.5 LIBERO policy server (RLinf-Pi05-LIBERO-130-fullshot-SFT behind RPent's
# pi05_vla_server.py: HTTP POST /call, method "vla.predict", 5 actions of 7 per query).
#
# Usage: pi05_server.sh GPU PORT LOG
# Env (defaults are H800-B's): RPENT (checkout providing rpent/robots/components/pi05_vla_server.py),
#   RPENT_PY (python with RLinf + openpi), PI05_CHECKPOINT_PATH.
# Needs ~8.2 GB on the GPU.  Stop it with: kill $(cat LOG.pid)
set -euo pipefail
GPU="$1"; PORT="$2"; LOG="$3"
RPENT=${RPENT:-/path/to/rpent}
RPENT_PY=${RPENT_PY:-/path/to/rpent-venv/bin/python}
export PI05_CHECKPOINT_PATH=${PI05_CHECKPOINT_PATH:-/path/to/checkpoints/RLinf-Pi05-LIBERO-130-fullshot-SFT}
export PYTHONNOUSERSITE=1 PYTHONPATH="${RPENT}" HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
cd "${RPENT}"
nohup setsid "${RPENT_PY}" rpent/robots/components/pi05_vla_server.py --embodiment libero --transport http \
    --host 127.0.0.1 --port "${PORT}" --cuda-device "${GPU}" > "${LOG}" 2>&1 < /dev/null &
echo $! > "${LOG}.pid"
echo "pi05 server pid $(cat "${LOG}.pid") gpu ${GPU} port ${PORT} log ${LOG}"
