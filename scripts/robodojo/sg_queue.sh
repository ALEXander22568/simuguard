#!/usr/bin/env bash
# Run SimuGuard x RoboDojo jobs one after another, each in its own container (sg_container.sh).
#
# usage: sg_queue.sh JOBS_FILE RUN_ROOT
#   JOBS_FILE: one job per line, "<name> <robodojo_eval args...>" (blank lines and # comments ignored)
#   each job writes RUN_ROOT/<name>/ (container.log, <task>/episodes.jsonl, segments/, eval_result/)
# A job waits for a GPU (WAIT_GPU=1, MIN_FREE_MIB, default 15000 here, stable over two polls).
# OOM guard: GPUs are shared, so if our container logs a GPU out-of-memory error it is stopped at
# once (it cannot finish, and it must not keep squeezing the other job on that card) and the job is
# retried later, up to SG_MAX_ATTEMPTS (default 3).  The next job starts only if the previous one
# exited 0 with no episode error (SG_QUEUE_STRICT=0 keeps going).  Finished jobs get
# RUN_ROOT/<name>/DONE and are skipped when the queue is restarted.
set -u
jobs=${1:?jobs file}
root=${2:?run root}
mkdir -p "$root" && root=$(cd "$root" && pwd)  # docker bind mounts need absolute paths
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export WAIT_GPU=1 MIN_FREE_MIB=${MIN_FREE_MIB:-15000}
oom_pattern="Out of GPU memory allocating|cudaErrorMemoryAllocation|CUDA out of memory"

run_job() {  # name out args... ; returns container rc, 75 for an OOM stop
  local name=$1 out=$2
  shift 2
  bash "$here/sg_container.sh" "$name" auto "$out" "$@" > "$out/container.log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if grep -q -E "$oom_pattern" "$out/container.log" 2>/dev/null; then
      echo "[$(date -u +%FT%TZ)] $name: GPU out-of-memory in our container, stopping it" | tee -a "$out/container.log"
      docker --context default stop -t 20 "sg-robodojo-$name" >/dev/null 2>&1
      wait "$pid"
      return 75
    fi
    sleep 10
  done
  wait "$pid"
}

while read -r name rest; do
  [[ -z "${name}" || "${name}" == \#* ]] && continue
  out="$root/$name"
  if [ -f "$out/DONE" ]; then echo "[$(date -u +%FT%TZ)] skip $name (DONE)"; continue; fi
  for attempt in $(seq 1 "${SG_MAX_ATTEMPTS:-3}"); do
    if [ "$attempt" -gt 1 ]; then  # keep the failed attempt's output
      mv "$out" "$out.attempt$((attempt - 1)).$(date -u +%H%M%S)"
    fi
    mkdir -p "$out"
    echo "[$(date -u +%FT%TZ)] job $name (attempt $attempt): $rest"
    # shellcheck disable=SC2086  # job arguments are whitespace-separated on purpose
    run_job "$name" "$out" $rest
    rc=$?
    [ "$rc" -ne 75 ] && break
    sleep 300  # let the card settle before waiting for a GPU again
  done
  errors=$(cat "$out"/*/episodes.jsonl 2>/dev/null | grep -c '"error"' || true)
  echo "[$(date -u +%FT%TZ)] job $name rc=$rc episode_errors=${errors:-0}"
  if [ "$rc" -eq 0 ] && [ "${errors:-0}" -eq 0 ]; then
    touch "$out/DONE"
  elif [ "${SG_QUEUE_STRICT:-1}" = 1 ]; then
    echo "[$(date -u +%FT%TZ)] stopping the queue after $name"
    exit 1
  fi
done < "$jobs"
echo "[$(date -u +%FT%TZ)] queue finished"
