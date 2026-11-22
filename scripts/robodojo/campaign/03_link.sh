#!/usr/bin/env bash
# Wire the downloaded files into the RoboDojo tree with symlinks (no copies of
# checkpoints): Assets, each policy's checkpoints/ (pointing into $CKPT_ROOT, the
# coordinator's download), and the base models of 00b_base_models.sh at the
# paths the unmodified deploy.yml files expect. Idempotent.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/pins.env"
RD="$SGRD_ROOT/RoboDojo"; POL="$RD/XPolicyLab/policy"; BASE="$SGRD_ROOT/hf/base"
CK="$CKPT_ROOT/hf/ckpt/RoboDojo"
lnk() { mkdir -p "$(dirname "$2")"; [[ -e "$1" ]] || { echo "missing $1" >&2; exit 1; }; ln -sfn "$1" "$2"; }

lnk "$SGRD_ROOT/hf/robodojo/Assets" "$RD/Assets"
# RoboDojo's own step that turns Assets/Robots/*/curobo_tmp.yml into curobo.yml
# (${ASSETS_PATH} -> the repo root); run from the repo root as its README does.
(cd "$RD" && python3 utils/update_embodiment_config_path.py < /dev/null)

for P in Pi_05 Pi_0 InternVLA_A1 X_VLA Xiaomi_Robotics_0; do
  [[ -L "$POL/$P/checkpoints" ]] && rm "$POL/$P/checkpoints"
  mkdir -p "$POL/$P/checkpoints"
  for e in "$CK/$P"/*; do lnk "$e" "$POL/$P/checkpoints/$(basename "$e")"; done
done
# InternVLA_A1: the release nests the step dirs one level deeper (<run>/checkpoints/060000/pretrained_model)
# than model.py looks (<run>/<step>/pretrained_model, then checkpoint_num 60000 picks 060000), so the run name
# points at the release's inner checkpoints/ dir. Same files, no code change.
for e in "$CK/InternVLA_A1"/*; do
  [[ -d "$e/checkpoints" ]] && lnk "$e/checkpoints" "$POL/InternVLA_A1/checkpoints/$(basename "$e")"
done
# GR00T_N17's model.py rewrites processor_config.json inside the checkpoint dir
# while it loads (cosmos path override, restored afterwards). Mirror the tree:
# directories real, files symlinked, that one small json copied, so the shared
# download is never written to and parallel servers do not race on one file.
src="$CK/GR00T_N17"; dst="$POL/GR00T_N17/checkpoints"
mkdir -p "$dst"
(cd "$src" && find . -type d) | while read -r d; do mkdir -p "$dst/$d"; done
(cd "$src" && find . -type f) | while read -r f; do
  if [[ "$(basename "$f")" == processor_config.json ]]; then
    [[ -f "$dst/$f" && ! -L "$dst/$f" ]] || { rm -f "$dst/$f"; cp "$src/$f" "$dst/$f"; }
  else
    lnk "$src/$f" "$dst/$f"
  fi
done
lnk "$BASE/Cosmos-Reason2-2B" "$dst/shared/Cosmos-Reason2-2B"
# The released checkpoint's config.json names its backbone by the trainer's absolute
# path ("model_name": "/mnt/xspark-data/xspark_shared/model_weights/Cosmos-Reason2-2B");
# model.py only overrides that path in processor_config.json, so the model build fails
# with HFValidationError on any other machine.  Copy config.json and point model_name at
# the same local Cosmos-Reason2-2B that model.py resolves for the processor (only this
# one string changes; all backbone weights are then loaded from the checkpoint itself).
cosmos_real="$(readlink -f "$dst/shared/Cosmos-Reason2-2B")"
(cd "$src" && find . -type f -name config.json -path '*checkpoint-*') | while read -r f; do
  rm -f "$dst/$f"
  python3 - "$src/$f" "$dst/$f" "$cosmos_real" <<'PY'
import json, sys
src, dst, cosmos = sys.argv[1:]
cfg = json.load(open(src))
old = cfg.get("model_name")
assert old and "Cosmos-Reason2-2B" in old, old
cfg["model_name"] = cosmos
json.dump(cfg, open(dst, "w"), indent=2)
print(f"GR00T config.json model_name: {old} -> {cosmos}")
PY
done
lnk "$BASE/X-VLA-Pt" "$POL/X_VLA/checkpoints/shared/X-VLA-Pt"
lnk "$BASE/Cosmos-Tokenizer-CI8x8" "$POL/InternVLA_A1/checkpoints/shared/Cosmos-Tokenizer-CI8x8"
lnk "$BASE/Qwen3-VL-2B-Instruct" "$POL/InternVLA_A1/checkpoints/shared/Qwen3-VL-2B-Instruct"
lnk "$BASE/Xiaomi-Robotics-0-Pretrain" "$POL/Xiaomi_Robotics_0/xiaomi_robotics_0/xr0/XiaomiRobotics/Xiaomi-Robotics-0-Pretrain"
# InternW0_delta: deploy.yml checkpoint_path checkpoints/robodojo.pt, assets/ for Wan2.2 + RynnBrain
# Only the weight file: a checkpoints/<run> dir would make setup_eval_policy_server.sh look for training
# slots (checkpoints/weights/slot_*.pt) and stop; with no run dir it uses checkpoints/robodojo.pt (its README).
# The release's config/ (eval_model.yaml, dataset_stats.json, artifacts.lock.json) equals the policy's config/.
mkdir -p "$POL/InternW0_delta/checkpoints"
[[ -L "$POL/InternW0_delta/checkpoints/RoboDojo-sim-arx_x5-joint-0" ]] && rm "$POL/InternW0_delta/checkpoints/RoboDojo-sim-arx_x5-joint-0"
lnk "$CK/InternW0_delta/RoboDojo-sim-arx_x5-joint-0/checkpoints/robodojo.pt" "$POL/InternW0_delta/checkpoints/robodojo.pt"
[[ -d "$BASE/internw0_assets" ]] && lnk "$BASE/internw0_assets" "$POL/InternW0_delta/assets"
mkdir -p "$POL/OpenWAM/checkpoints"
lnk "$CKPT_ROOT/OpenWAM-Alpha-Sim-RoboDojo" "$POL/OpenWAM/checkpoints/OpenWAM-Alpha-Sim-RoboDojo"
echo "links done"
