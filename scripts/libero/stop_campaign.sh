#!/usr/bin/env bash
# Stop a pi05_campaign.sh run launched with `setsid ... & echo $! > OUT/driver.pid`: its whole process
# group (driver, xargs, workers, their ffmpeg pipes) and nothing else.  Episodes in flight are lost;
# run requeue.py before relaunching.   Usage: stop_campaign.sh OUT
set -u
OUT="$1"
touch "${OUT}/STOP"
PGID=$(cat "${OUT}/driver.pid")
[ "$(ps -o pgid= -p "${PGID}" | tr -d ' ')" = "${PGID}" ] || { echo "driver ${PGID} not running"; exit 0; }
kill -TERM -- "-${PGID}"
for _ in 1 2 3 4 5 6 7 8 9 10; do
    sleep 1
    pgrep -g "${PGID}" > /dev/null || break
done
pgrep -g "${PGID}" > /dev/null && kill -KILL -- "-${PGID}"
echo "stopped process group ${PGID}"
