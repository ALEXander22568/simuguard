#!/usr/bin/env bash
# Control the LingBot-VA backends on the inference host (split deployment).
#
#   va_ctl.sh start|restart SLOT [LABEL]   blocks until the backend listens (<= 30 min)
#   va_ctl.sh stop SLOT
#   va_ctl.sh status
# LABEL (e.g. the task name) is appended to the run directory name for traceability.
#
# Slots are fixed (GPU, port, master port) rows in ${VA_HOME}/slots.conf:
#     # slot gpu port master
#     1    7   29601 29701
# Each backend is run_lingbot_eval.sh in VA_ONLY mode with the same launch settings as a
# single-machine run, bound to 127.0.0.1 and reached from the simulator host through an SSH
# tunnel.  A fresh process per task keeps the one-server-per-run semantics of the single-machine
# setup.
#
# Also usable as a forced command in authorized_keys: the command then comes from
# SSH_ORIGINAL_COMMAND and only the verbs above are accepted.
set -uo pipefail
VA_HOME=${VA_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
[[ -f "${VA_HOME}/va_host.env" ]] && source "${VA_HOME}/va_host.env"
SG=${VA_HOME}/SimuGuard
SLOTS=${VA_HOME}/slots.conf

if [[ $# -eq 0 && -n "${SSH_ORIGINAL_COMMAND:-}" ]]; then
    read -r -a argv <<< "${SSH_ORIGINAL_COMMAND}"
    [[ "${argv[0]:-}" == */va_ctl.sh ]] && argv=("${argv[@]:1}")   # callers may name the script
    set -- "${argv[@]}"
fi
verb=${1:-status}; slot=${2:-}; label=${3:-}
# strict arguments: this script can run as an SSH forced command
[[ "${verb}" =~ ^(start|stop|restart|status)$ ]] || { echo "usage: va_ctl.sh start|stop|restart SLOT [LABEL] | status"; exit 2; }
[[ -z "${slot}" || "${slot}" =~ ^[0-9]{1,2}$ ]] || { echo "bad slot"; exit 2; }
[[ -z "${label}" || "${label}" =~ ^[A-Za-z0-9_.-]{1,64}$ ]] || { echo "bad label"; exit 2; }

slot_row() { awk -v s="$1" '$1 == s {print $2, $3, $4}' "${SLOTS}"; }
listening() { "${SERVER_PY}" -c "import socket,sys;s=socket.socket();s.settimeout(1);sys.exit(s.connect_ex(('127.0.0.1',$1)))" 2>/dev/null; }

stop_slot() {
    local pidfile="${VA_HOME}/runs/slot$1/launcher.pid"
    [[ -f "${pidfile}" ]] || return 0
    local pid; pid=$(cat "${pidfile}")
    if kill -0 "${pid}" 2>/dev/null; then
        kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null
        for _ in $(seq 1 40); do kill -0 "${pid}" 2>/dev/null || break; sleep 1; done
        kill -KILL -- "-${pid}" 2>/dev/null || true
    fi
    rm -f "${pidfile}"
}

start_slot() {
    local row gpu port master
    row=$(slot_row "$1"); [[ -n "${row}" ]] || { echo "unknown slot $1"; return 2; }
    read -r gpu port master <<< "${row}"
    if listening "${port}"; then echo "slot $1 port ${port} already in use"; return 3; fi
    local run="${VA_HOME}/runs/slot$1/$(date -u +%Y%m%dT%H%M%SZ)${label:+_${label}}"
    mkdir -p "${run}"
    ln -sfn "${run}" "${VA_HOME}/runs/slot$1/current"
    VA_ONLY=1 VA_PORT_FIXED="${port}" VA_MASTER_FIXED="${master}" SIMUGUARD_VA_HOST=127.0.0.1 \
    MERGED_DIR="${run}/.merged_ckpt" SIMUGUARD_WORKSPACE="${VA_HOME}" ROBOTWIN_ROOT="${VA_HOME}/RoboTwin" \
    SERVER_PY="${SERVER_PY}" LINGBOT_MODEL="${LINGBOT_MODEL}" MIN_FREE_MODEL_MIB="${MIN_FREE_MODEL_MIB:-20000}" \
        setsid nohup bash "${SG}/scripts/robotwin/run_lingbot_eval.sh" "${run}" 0 va_only "${gpu}" 0 \
        > "${run}/launcher.log" 2>&1 < /dev/null &
    echo $! > "${VA_HOME}/runs/slot$1/launcher.pid"
    echo "slot $1 starting on GPU${gpu} port ${port} (${run})"
}

wait_ready() {
    local row gpu port master pid t=0
    row=$(slot_row "$1"); read -r gpu port master <<< "${row}"
    pid=$(cat "${VA_HOME}/runs/slot$1/launcher.pid" 2>/dev/null)
    until listening "${port}"; do
        if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
            echo "slot $1 launcher exited:"; tail -5 "${VA_HOME}/runs/slot$1/current/launcher.log"; return 4
        fi
        sleep 5; t=$((t + 5))
        (( t >= 1800 )) && { echo "slot $1 not listening after ${t}s"; return 5; }
    done
    echo "slot $1 ready on port ${port} after ${t}s"
}

case "${verb}" in
    start)   start_slot "${slot}" && wait_ready "${slot}" ;;
    stop)    stop_slot "${slot}"; echo "slot ${slot} stopped" ;;
    restart) stop_slot "${slot}"; start_slot "${slot}" && wait_ready "${slot}" ;;
    status)
        while read -r s gpu port master; do
            [[ -z "${s}" || "${s}" == \#* ]] && continue
            state=down; listening "${port}" && state=up
            echo "slot ${s} gpu${gpu} port ${port} ${state}"
        done < "${SLOTS}" ;;
esac
