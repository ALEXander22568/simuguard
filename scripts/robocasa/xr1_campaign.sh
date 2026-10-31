#!/usr/bin/env bash
# XR-1 on RoboCasa365 under SimuGuard, several contact settings, many worker processes.
#
# Usage: xr1_campaign.sh OUT CONCURRENCY CONTACTS JOB [JOB ...]
#   CONTACTS  comma list of xr1_rollout.py --contact values, e.g. default,stiff2dt
#   JOB       TASK:EPISODES, e.g. ArrangeBreadBasket:0-1 (one worker process per job and contact;
#             list long jobs first).  Finished episodes are skipped, so a rerun resumes.
# Env: PY (python with robocasa), SG (SimuGuard checkout); source the RoboCasa runtime env first.
set -uo pipefail
OUT="$1"; CONC="$2"; CONTACTS="$3"; shift 3
: "${PY:?}" "${SG:?}"
mkdir -p "${OUT}/logs"
export OUT PY SG
echo "[$(date -u +%FT%TZ)] start: contacts ${CONTACTS}, $# jobs each, concurrency ${CONC}" >> "${OUT}/campaign.log"
for job in "$@"; do
    for contact in ${CONTACTS//,/ }; do
        echo "${contact} ${job}"
    done
done | xargs -P "${CONC}" -L 1 bash -c '
    contact=$0; task=${1%%:*}; episodes=${1##*:}
    log="${OUT}/logs/${contact}_${task}_${episodes}.log"
    "${PY}" "${SG}/scripts/robocasa/xr1_rollout.py" --task "${task}" --episodes "${episodes}" \
        --contact "${contact}" --out "${OUT}" --video > "${log}" 2>&1
    echo "[$(date -u +%FT%TZ)] ${contact} ${task} ${episodes}: exit $?" >> "${OUT}/campaign.log"'
echo "DONE" > "${OUT}/STATUS"
echo "[$(date -u +%FT%TZ)] finished" >> "${OUT}/campaign.log"
