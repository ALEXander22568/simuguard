#!/bin/bash
# every 5 min log free space on /; below 400 GB write runs/DISK_WARN (once) and a WARN line; below 150 GB touch
# STOP in the formal output dirs (lanes stop after their current task) and exit
R=${DISK_WATCH_RUNS:-/path/to/simuguard-robolab/runs}
while true; do
  free=$(df --output=avail -B1G / | tail -1 | tr -d " ")
  echo "$(date -u +%FT%TZ) ${free}G" >> $R/disk_watch.log
  if [ $free -lt 400 ] && [ ! -f $R/DISK_WARN ]; then
    echo "$(date -u +%FT%TZ) ${free}G free, below 400G" > $R/DISK_WARN; echo "WARN below 400G" >> $R/disk_watch.log
  fi
  if [ $free -lt 150 ]; then mkdir -p $R/formal_pi05 $R/formal_gr00t; touch $R/formal_pi05/STOP $R/formal_gr00t/STOP; echo STOP >> $R/disk_watch.log; exit; fi
  sleep 300
done
