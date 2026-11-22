#!/usr/bin/env bash
# Base models / processors the 8 policy servers load besides their RoboDojo
# checkpoints, at pinned revisions, into $SGRD_ROOT/hf/base/. 03_link.sh links
# them to the paths the (unmodified) deploy.yml files point at.
#   GR00T_N17          checkpoints/shared/Cosmos-Reason2-2B   <- nvidia/Cosmos-Reason2-2B. Gated on HF (no token
#                      here), so the files come from the ModelScope mirror nv-community/Cosmos-Reason2-2B and are
#                      checked against cosmos_reason2_2b.sha256 (= HF rev 9ce19a19: model.safetensors LFS sha256
#                      and the git blob ids of all small files matched HF on 2026-11-06).
#   X_VLA              checkpoints/shared/X-VLA-Pt            <- 2toINF/X-VLA-Pt
#   InternVLA_A1       checkpoints/shared/Cosmos-Tokenizer-CI8x8, checkpoints/shared/Qwen3-VL-2B-Instruct
#   Xiaomi_Robotics_0  xiaomi_robotics_0/xr0/XiaomiRobotics/Xiaomi-Robotics-0-Pretrain (processor files only)
#                      + Qwen/Qwen3-VL-4B-Instruct in the HF hub cache ($HF_HOME): XR0._build_model builds the VLM
#                      with from_pretrained("Qwen/Qwen3-VL-4B-Instruct") before loading the checkpoint, and the
#                      official server script exports a fixed LAN proxy, so the server runs with HF_HUB_OFFLINE=1
#                      (the script's documented offline switch; set by rd_queue.py)
#   InternW0_delta     assets/: Wan2.2-TI2V-5B subset from ModelScope (the files and source download_assets.sh uses)
#                      + Alibaba-DAMO-Academy/RynnBrain1.1-2B from HF (same org; ModelScope has no such repo)
#   Pi_05 / Pi_0       $SGRD_ROOT/.cache/openpi/big_vision/paligemma_tokenizer.model (openpi's GCS tokenizer)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/pins.env"
export HOME="$SGRD_ROOT" HF_HOME="$SGRD_ROOT/hf/home" PYTHONNOUSERSITE=1
BASE="$SGRD_ROOT/hf/base"; mkdir -p "$BASE"
HF="$SGRD_ROOT/dlvenv/bin/hf"
sections=("$@"); [[ ${#sections[@]} -gt 0 ]] || sections=(cosmos hf openpi internw0)
want() { local x; for x in "${sections[@]}"; do [[ $x == "$1" ]] && return 0; done; return 1; }
get() {  # repo revision [include...]
  local repo=$1 rev=$2; shift 2
  local inc=(); for p in "$@"; do inc+=(--include "$p"); done
  "$HF" download "$repo" --revision "$rev" --local-dir "$BASE/${repo##*/}" "${inc[@]}" --max-workers 8
}
if want cosmos; then
  d="$BASE/Cosmos-Reason2-2B"; mkdir -p "$d"
  while read -r sum f; do
    [[ -s "$d/$f" ]] && echo "$sum  $d/$f" | sha256sum -c --quiet - 2>/dev/null && continue
    curl -fsSL --retry 3 -o "$d/$f" "https://modelscope.cn/api/v1/models/nv-community/Cosmos-Reason2-2B/repo?Revision=master&FilePath=$f"
  done < "$HERE/cosmos_reason2_2b.sha256"
  (cd "$d" && sha256sum -c "$HERE/cosmos_reason2_2b.sha256")
fi
if want hf; then
  get 2toINF/X-VLA-Pt "$XVLA_PT_REV"
  get nvidia/Cosmos-Tokenizer-CI8x8 "$COSMOS_TOK_REV"
  get Qwen/Qwen3-VL-2B-Instruct "$QWEN3VL_2B_REV"
  get XiaomiRobotics/Xiaomi-Robotics-0-Pretrain "$XR0_PRETRAIN_REV" '*.json' '*.txt' '*.jinja' '*.py'
  "$HF" download Qwen/Qwen3-VL-4B-Instruct --revision "$QWEN3VL_4B_REV" --max-workers 8  # hub cache, not hf/base
  # the code asks for revision main; offline, the cache resolves main through refs/main (= the pinned commit,
  # main on HF when pinned 2026-11-07)
  q="$HF_HOME/hub/models--Qwen--Qwen3-VL-4B-Instruct"; mkdir -p "$q/refs"; printf %s "$QWEN3VL_4B_REV" > "$q/refs/main"
fi
if want openpi; then  # openpi tokenizer (public bucket), the file openpi's maybe_download fetches
  mkdir -p "$SGRD_ROOT/.cache/openpi/big_vision"
  tok="$SGRD_ROOT/.cache/openpi/big_vision/paligemma_tokenizer.model"
  [[ -s "$tok" ]] || curl -fsSL -o "$tok" https://storage.googleapis.com/big_vision/paligemma_tokenizer.model
  echo "$PALIGEMMA_TOK_SHA256  $tok" | sha256sum -c -
fi
if want internw0; then  # InternW0_delta assets at the paths its download_assets.sh uses
  A="$BASE/internw0_assets"
  eval "$("$SGRD_ROOT/miniconda3/bin/conda" shell.bash hook)"
  conda activate internw0-delta
  if [[ "${WAN_SOURCE:-modelscope}" == hf ]]; then
    # same files from the same org's HF repo (ModelScope stalls from some hosts, e.g. the 48 GB box on 11-07);
    # checked below against the policy's artifacts.lock.json
    "$HF" download Wan-AI/Wan2.2-TI2V-5B --revision "$WAN22_HF_REV" --local-dir "$A/Wan-AI/Wan2.2-TI2V-5B" \
      --include Wan2.2_VAE.pth --include models_t5_umt5-xxl-enc-bf16.pth --include 'google/umt5-xxl/*' --max-workers 8
  else
  python - "$A/Wan-AI/Wan2.2-TI2V-5B" <<'PY'
import sys
from modelscope import snapshot_download
# the "wan-eval" selection of download_assets.sh
snapshot_download("Wan-AI/Wan2.2-TI2V-5B", local_dir=sys.argv[1],
                  allow_file_pattern=["Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl/*"])
PY
  fi
  python - "$A/Wan-AI/Wan2.2-TI2V-5B" "$SGRD_ROOT/RoboDojo/XPolicyLab/policy/InternW0_delta/config/artifacts.lock.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
d, lock = Path(sys.argv[1]), json.load(open(sys.argv[2]))
for name, want in lock["wan_base"]["files"].items():
    h = hashlib.sha256()
    with open(d / name, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    assert h.hexdigest() == want["sha256"], (name, h.hexdigest())
    print(f"{name}: sha256 matches artifacts.lock.json")
PY
  conda deactivate
  d="$A/Alibaba-DAMO-Academy/RynnBrain1.1-2B"
  "$HF" download Alibaba-DAMO-Academy/RynnBrain1.1-2B --revision "$RYNNBRAIN_REV" --local-dir "$d" --max-workers 8
  echo "$RYNNBRAIN_SAFETENSORS_SHA256  $d/model.safetensors" | sha256sum -c -
fi
(cd "$BASE" && find . -type f ! -path '*/.cache/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > "$SGRD_ROOT/locks/base_models.sha256"
echo "base models done"
