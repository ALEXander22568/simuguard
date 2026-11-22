#!/usr/bin/env bash
# Copy a run dir from this box to the 24 GB box (rd_queue --on-job-end), so results survive this box going down.
# The key (rd_sync_ed25519) is accepted on the 24 GB box only by rrsync, write-only, under
# /path/to/simuguard-robodojo/runs-48g; the run lands at runs-48g/<run name>. Never deletes on the receiving side.
# usage: bash push_run.sh RUN_DIR
set -euo pipefail
R=${SGRD_ROOT:-/path/to/simuguard-robodojo}
RUN=$(cd "${1:?run dir}" && pwd)
DEST=${PUSH_DEST:-user@host-b}; PORT=${PUSH_PORT:-22}
exec 9>"$RUN/.push.lock"; flock 9  # two queues on one run dir (one per GPU) push one at a time
t0=$(date +%s)
rsync -a --partial --exclude current_layout.json --exclude .push.lock \
  -e "ssh -i $R/.ssh/rd_sync_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=$R/.ssh/known_hosts -o ClearAllForwardings=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 -p $PORT" \
  "$RUN" "$DEST:/"
echo "$(TZ=Asia/Seoul date '+%m-%d %H:%M:%S KST') pushed $RUN in $(( $(date +%s) - t0 )) s"
