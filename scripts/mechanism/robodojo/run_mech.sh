#!/usr/bin/env bash
# usage: run_mech.sh NAME GPU SEGMENT_DIR TASK LO:HI NAME1,NAME2
# One throwaway container of the pinned RoboDojo image (same mounts as simuguard/scripts/robodojo/sg_container.sh,
# plus /runs on PYTHONPATH so mech_dump.py is importable). Output: runs/mech_probe/NAME/.
set -euo pipefail
name=$1; gpu=$2; seg=$3; task=$4; window=$5; names=$6
root=/mnt/nvme0/shared/USER/simuguard-robodojo
base=$root/runs/mech_probe; out=$base/$name
assets=$(readlink -f "${ROBODOJO_ASSETS:?set ROBODOJO_ASSETS to the RoboDojo asset tree}")
image=${ROBODOJO_IMAGE:?set ROBODOJO_IMAGE to the RoboDojo container image}
mkdir -p "$out/eval_result" "$out/src"
rm -rf "$out/src/$(basename $seg)"; cp -r "$seg" "$out/src/"; cp "$base/mech_dump.py" "$out/"
free=$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
[ "$free" -ge 11000 ] || { echo "GPU $gpu has only ${free} MiB free; refusing to start" >&2; exit 3; }
cname="sg-robodojo-mech-$name"
echo "[$(date -u +%FT%TZ)] starting $cname on GPU $gpu (${free} MiB free)"
docker --context default rm -f "$cname" >/dev/null 2>&1 || true
set +e
docker --context default run --rm --name "$cname" \
  --gpus "\"device=$gpu\"" --shm-size=8g --network host \
  --label owner="$(id -un)" --label workload=simuguard-robodojo \
  --log-opt max-size=50m --log-opt max-file=2 \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e OMNI_KIT_ACCEPT_EULA=YES \
  -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONNOUSERSITE=1 \
  -e ROBODOJO_RUN_ID="$cname-$(date -u +%Y%m%dT%H%M%SZ)" \
  -e PYTHONPATH=/workspace/RoboDojo:/workspace/RoboDojo/XPolicyLab:/simuguard:/runs \
  -e MECH_WINDOW="$window" -e MECH_NAMES="$names" -e MECH_OUT=/runs/dump.json \
  -v "$root/assets_overlay:/workspace/RoboDojo/Assets:ro" \
  -v "$assets:$assets:ro" \
  -v "$root/simuguard:/simuguard:ro" \
  -v "$out:/runs" \
  -v "$out/eval_result:/workspace/RoboDojo/eval_result" \
  -v "$root/cache/ov:/root/.cache/ov" -v "$root/cache/nv:/root/.nv" -v "$root/cache/warp:/root/.cache/warp" \
  -w /workspace/RoboDojo \
  --entrypoint /root/miniconda3/envs/RoboDojo/bin/python \
  "$image" /runs/mech_dump.py --out /runs --task "$task" --replay-from "/runs/src/$(basename $seg)" --intervention none --no-cameras
rc=$?
set -e
docker --context default run --rm --network none -v "$out:/runs" --entrypoint chown "$image" -R "$(id -u):$(id -g)" /runs || true
echo "[$(date -u +%FT%TZ)] $cname exited rc=$rc"
