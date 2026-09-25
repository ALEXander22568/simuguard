#!/bin/bash
# 4090-node3 127.0.0.1:16601 -> h800-2 127.0.0.1:16601 (XR-1 RoboDojo policy server for SimuGuard).
# Key ~/.ssh/sg_robodojo_h8002 is restricted on h800-2 to port-forwarding 127.0.0.1:16601
# (authorized_keys backup: ~/.ssh/authorized_keys.bak-20260925-sg-robodojo on h800-2).
# Reconnects 5 s after a drop; flock prevents duplicates.  Stop: kill the loop, then the ssh child.
exec 9>/tmp/sg-robodojo-tunnel-$(id -u).lock
flock -n 9 || exit 0
while true; do
  ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o IdentitiesOnly=yes -i ~/.ssh/sg_robodojo_h8002 \
      -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o Compression=no \
      -o ClearAllForwardings=no -F /dev/null \
      -L 127.0.0.1:16601:127.0.0.1:16601 -p 2204 zhoujingjing@8.152.195.165
  echo "$(date -u +%FT%TZ) tunnel exited rc=$?, reconnecting in 5s" >> /data/shared/zhoujingjing/simuguard-robodojo/sg-policy-tunnel.log
  sleep 5
done
