#!/usr/bin/env bash
# Pi0.5 on LIBERO under SimuGuard, many worker processes.
#
# Usage: pi05_campaign.sh OUT CONCURRENCY ENDPOINTS JOB [JOB ...]
#   ENDPOINTS  comma list of Pi0.5 servers (pi05_server.sh), e.g. http://127.0.0.1:58261,http://127.0.0.1:58262;
#              every job gets the list: episodes are spread over the servers, failed queries move on
#   JOB        SUITE:TASK:EPISODES, e.g. libero_spatial:0:0-4 (one worker process per job; list long jobs first)
# Env: PY, SG (source scripts/libero/env.h800-2.sh first).  Finished episodes are skipped, so a rerun
# resumes (requeue.py first drops episodes that ended in an infrastructure error); `touch OUT/STOP`
# lets running jobs finish and starts no new one.
set -uo pipefail
OUT="$1"; CONC="$2"; ENDPOINTS="$3"; shift 3
: "${PY:?}" "${SG:?}"
mkdir -p "${OUT}/logs"
export OUT PY SG
echo "[$(date -u +%FT%TZ)] start: $# jobs, concurrency ${CONC}, endpoints ${ENDPOINTS}" >> "${OUT}/campaign.log"
for job in "$@"; do
    echo "${job} ${ENDPOINTS}"
done | xargs -P "${CONC}" -L 1 bash -c '
    IFS=: read -r suite task episodes <<< "$0"; endpoint=$1
    [ -e "${OUT}/STOP" ] && exit 0
    log="${OUT}/logs/${suite}_t${task}_${episodes}.log"
    "${PY}" "${SG}/scripts/libero/pi05_rollout.py" --suite "${suite}" --task "${task}" --episodes "${episodes}" \
        --out "${OUT}" --endpoint "${endpoint}" --video >> "${log}" 2>&1
    echo "[$(date -u +%FT%TZ)] ${suite} t${task} ${episodes}: exit $?" >> "${OUT}/campaign.log"'
echo "DONE" > "${OUT}/STATUS"
echo "[$(date -u +%FT%TZ)] finished" >> "${OUT}/campaign.log"
