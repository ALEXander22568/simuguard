#!/usr/bin/env bash
# Run one SimuGuard x RoboDojo job in a throwaway container of the pinned RoboDojo image.
#
# usage: sg_container.sh NAME GPU|auto HOST_OUT_DIR [robodojo_eval args...]
#   e.g. sg_container.sh probe auto /data/shared/zhoujingjing/simuguard-robodojo/runs/probe1 \
#          --task put_bottles_into_dustbin --layouts 0 --policy scripted --replay all
#
# env: SG_ROOT (default /data/shared/zhoujingjing/simuguard-robodojo; holds the simuguard copy in
#      $SG_ROOT/simuguard, the asset overlay in $SG_ROOT/assets_overlay and Kit caches in
#      $SG_ROOT/cache), SG_ASSETS_BASE (the shared RoboDojo asset tree the overlay links to),
#      MIN_FREE_MIB (default 11000: same rule as start-sim-zjj.sh), WAIT_GPU=1 to poll every 60 s
#      until a GPU has that much free memory, SG_GPUS (candidate GPUs for auto, default all),
#      SG_MODULE (default simuguard.integrations.robodojo_eval).
# Only this container is started or removed; the GPU is chosen by free memory, never by killing
# anything.  Output files are chowned back to the calling user.
set -euo pipefail
name="sg-robodojo-${1:?name}"
gpu=${2:?gpu or auto}
out=${3:?host output dir}
mkdir -p "$out" && out=$(cd "$out" && pwd)  # docker bind mounts need absolute paths
shift 3
image=hexa/robodojo:0.2.4-f465f8ae
expected=sha256:9e8f833427b2e23876715e49a24f60c87622f6d221f895f28706a5e3203ddfe8
root=${SG_ROOT:-/data/shared/zhoujingjing/simuguard-robodojo}
assets=$(readlink -f "${SG_ASSETS_BASE:-/data/shared/zhoujingjing/urai-robodojo/assets}")   # shared tree, read-only
# RoboDojo generates Robots/*/curobo.yml from curobo_tmp.yml at install time; the shared tree has
# none, so an overlay holds a copy of Robots/ with generated files and symlinks for the rest.
overlay=${SG_ASSETS_OVERLAY:-$root/assets_overlay}
min_free=${MIN_FREE_MIB:-11000}
module=${SG_MODULE:-simuguard.integrations.robodojo_eval}
test "$(docker --context default image inspect "$image" --format '{{.Id}}')" = "$expected"
[ -d "$assets/Eval_Layout" ] || { echo "missing assets $assets" >&2; exit 2; }
[ -f "$overlay/Robots/x5/curobo.yml" ] || { echo "missing overlay $overlay (see docs/robodojo.md)" >&2; exit 2; }
[ -f "$root/simuguard/simuguard/__init__.py" ] || { echo "missing simuguard copy in $root/simuguard" >&2; exit 2; }

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F', ' -v min="$min_free" -v allowed=",${SG_GPUS:-0,1,2,3,4,5,6,7}," \
      '$2 >= min && index(allowed, "," $1 ",") {print $2, $1}' | sort -rn | head -1 | awk '{print $2}'
}
if [ "$gpu" = auto ]; then
  prev=""
  while :; do
    gpu=$(pick_gpu || true)
    # require the same GPU to qualify on two polls a minute apart (other jobs' memory fluctuates)
    if [ -n "$gpu" ] && { [ "${SG_STABLE_POLLS:-2}" -le 1 ] || [ "$gpu" = "$prev" ]; }; then break; fi
    prev=$gpu
    if [ -n "$gpu" ]; then sleep 60; continue; fi
    if [ "${WAIT_GPU:-0}" = 1 ]; then
      echo "[$(date -u +%FT%TZ)] no GPU with >= ${min_free} MiB free; waiting" >&2
      sleep 60
    else
      echo "no GPU with >= ${min_free} MiB free:" >&2
      nvidia-smi --query-gpu=index,memory.free --format=csv,noheader >&2
      exit 3
    fi
  done
else
  free=$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$free" -ge "$min_free" ] || { echo "GPU $gpu has only ${free} MiB free; refusing to start" >&2; exit 3; }
fi
mkdir -p "$out/eval_result" "$root/cache/ov" "$root/cache/nv" "$root/cache/warp" "$root/cache/kitdata"
echo "[$(date -u +%FT%TZ)] starting $name on GPU $gpu ($(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader) free)"
docker --context default rm -f "$name" >/dev/null 2>&1 || true
set +e
docker --context default run --rm --name "$name" \
  --gpus "\"device=$gpu\"" --shm-size=8g --network host \
  --label owner=zhoujingjing --label workload=simuguard-robodojo \
  --log-opt max-size=50m --log-opt max-file=2 \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e OMNI_KIT_ACCEPT_EULA=YES \
  -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONNOUSERSITE=1 \
  -e ROBODOJO_RUN_ID="$name-$(date -u +%Y%m%dT%H%M%SZ)" \
  -e PYTHONPATH=/workspace/RoboDojo:/workspace/RoboDojo/XPolicyLab:/simuguard \
  -v "$overlay:/workspace/RoboDojo/Assets:ro" \
  -v "$assets:$assets:ro" \
  -v "$root/simuguard:/simuguard:ro" \
  -v "$out:/runs" \
  -v "$out/eval_result:/workspace/RoboDojo/eval_result" \
  -v "$root/cache/ov:/root/.cache/ov" \
  -v "$root/cache/nv:/root/.nv" \
  -v "$root/cache/warp:/root/.cache/warp" \
  -w /workspace/RoboDojo \
  --entrypoint /root/miniconda3/envs/RoboDojo/bin/python \
  "$image" -m "$module" --out /runs "$@"
rc=$?
set -e
docker --context default run --rm --network none -v "$out:/runs" --entrypoint chown "$image" -R "$(id -u):$(id -g)" /runs || true
echo "[$(date -u +%FT%TZ)] $name exited rc=$rc"
exit $rc
