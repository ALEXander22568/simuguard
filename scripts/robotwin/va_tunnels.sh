#!/usr/bin/env bash
# Keep one SSH tunnel per remote LingBot-VA backend (split deployment, simulator side).
#
#   va_tunnels.sh start PORT [PORT ...]    local port = remote port, bound to 127.0.0.1
#   va_tunnels.sh stop | status
#
# One ssh process per port on purpose: separate TCP streams add up (about 2x the throughput of a
# single multiplexed connection through the gateway we use), and one dropped tunnel does not take
# the other lanes down.  Each tunnel is restarted by its own loop if ssh exits.
# Env: VA_SSH_CONFIG (default ~/.ssh/config.simuguard), VA_SSH_HOST (default h800-2-sg).
set -uo pipefail
CONFIG=${VA_SSH_CONFIG:-$HOME/.ssh/config.simuguard}
HOST=${VA_SSH_HOST:-h800-2-sg}
STATE=${VA_TUNNEL_STATE:-$HOME/.simuguard_tunnels}
mkdir -p "${STATE}"
verb=${1:-status}; shift || true

probe() { python3 -c "import socket,sys;s=socket.socket();s.settimeout(1);sys.exit(s.connect_ex(('127.0.0.1',$1)))" 2>/dev/null; }

case "${verb}" in
    start)
        for port in "$@"; do
            if [[ -f "${STATE}/${port}.pid" ]] && kill -0 "$(cat "${STATE}/${port}.pid")" 2>/dev/null; then
                echo "tunnel ${port} already running"; continue
            fi
            setsid nohup bash -c "while true; do
                ssh -F '${CONFIG}' -N -o BatchMode=yes -o ExitOnForwardFailure=yes \
                    -L 127.0.0.1:${port}:127.0.0.1:${port} '${HOST}'
                echo \"[\$(date -u +%FT%TZ)] tunnel ${port} exited rc=\$?, restarting\"
                sleep 5
            done" >> "${STATE}/${port}.log" 2>&1 < /dev/null &
            echo $! > "${STATE}/${port}.pid"
            echo "tunnel ${port} started"
        done ;;
    stop)
        for pidfile in "${STATE}"/*.pid; do
            [[ -f "${pidfile}" ]] || continue
            kill -TERM -- "-$(cat "${pidfile}")" 2>/dev/null; rm -f "${pidfile}"
        done
        echo "tunnels stopped" ;;
    status)
        for pidfile in "${STATE}"/*.pid; do
            [[ -f "${pidfile}" ]] || continue
            port=$(basename "${pidfile}" .pid)
            alive=dead; kill -0 "$(cat "${pidfile}")" 2>/dev/null && alive=running
            open=closed; probe "${port}" && open=open
            echo "tunnel ${port} ${alive} local-port ${open}"
        done ;;
    *) echo "usage: va_tunnels.sh start PORT... | stop | status"; exit 2 ;;
esac
