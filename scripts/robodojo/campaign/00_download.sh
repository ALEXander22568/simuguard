#!/usr/bin/env bash
# Download RoboDojo assets (Assets/** of the HF dataset at a pinned revision)
# and the base models / processors the policies need. The fine-tuned seed-0
# checkpoints are fetched separately into $CKPT_ROOT (see pins.env) and only
# symlinked by 03_link_ckpts.sh.
# Usage: bash 00_download.sh [assets|base|all]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/pins.env"
export HF_HOME="$SGRD_ROOT/hf/home" HF_HUB_ENABLE_HF_TRANSFER=0 HF_XET_HIGH_PERFORMANCE=1
DL="$SGRD_ROOT/dlvenv"
if [[ ! -x "$DL/bin/hf" ]]; then
  "${UV:-$HOME/.local/bin/uv}" venv -q --python 3.11 "$DL"
  "${UV:-$HOME/.local/bin/uv}" pip install -q --python "$DL/bin/python" "huggingface_hub[hf_xet]==2.4.0"
fi
HFD="$SGRD_ROOT/hf/robodojo"   # mirror of the dataset repo layout
mkdir -p "$HFD"
what="${1:-all}"

if [[ "$what" == assets || "$what" == all ]]; then
  "$DL/bin/hf" download "$HF_DATASET" --repo-type dataset --revision "$HF_REV" \
    --include 'Assets/**' --local-dir "$HFD" --max-workers 16
fi
if [[ "$what" == base || "$what" == all ]]; then
  bash "$HERE/00b_base_models.sh"
fi
echo "download($what) done"
