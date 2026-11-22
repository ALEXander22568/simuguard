#!/usr/bin/env bash
# Install the policy-server environments of the 8 RoboDojo policies, each with
# its own XPolicyLab install.sh (unmodified), inside the isolated tree. Every
# environment is frozen into $SGRD_ROOT/locks/ afterwards.
# Usage: bash 02_install_policies.sh <group|policy>...
#   groups: uv (GR00T_N17 Pi_05 Pi_0), conda (the other five), all
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/pins.env"
export HOME="$SGRD_ROOT" PIP_USER=0 PYTHONNOUSERSITE=1 HF_HOME="$SGRD_ROOT/hf/home"
export PATH="$SGRD_ROOT/bin:$SGRD_ROOT/miniconda3/bin:$PATH"
if [[ ! -x "$SGRD_ROOT/bin/uv" ]] || ! "$SGRD_ROOT/bin/uv" --version | grep -q "$UV_VERSION"; then
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_INSTALL_DIR="$SGRD_ROOT/bin" UV_NO_MODIFY_PATH=1 sh
fi
eval "$(conda shell.bash hook)"
POL="$SGRD_ROOT/RoboDojo/XPolicyLab/policy"
LOCKS="$SGRD_ROOT/locks"; mkdir -p "$LOCKS"

freeze_uv()    { "$SGRD_ROOT/bin/uv" pip freeze --python "$1/.venv/bin/python" | sort > "$LOCKS/$2.pip-freeze.txt"; }
freeze_conda() { conda run -n "$1" python -m pip freeze --all | sort > "$LOCKS/$2.pip-freeze.txt"; conda list -n "$1" --explicit > "$LOCKS/$2.conda-explicit.txt"; }

# GR00T_N17: install.sh's `uv run` would try to lock pyproject for aarch64 too
# (required-environments) and fail on the missing aarch64 wheels; UV_NO_SYNC=1
# makes `uv run` use the venv that `uv pip install` just built (no other change).
install_GR00T_N17() { (cd "$POL/GR00T_N17" && UV_NO_SYNC=1 bash install.sh); freeze_uv "$POL/GR00T_N17/gr00t_n17" GR00T_N17; }
install_Pi_05()     { (cd "$POL/Pi_05" && bash install.sh); freeze_uv "$POL/Pi_05/openpi" Pi_05; }
install_Pi_0()      { (cd "$POL/Pi_0" && bash install.sh); freeze_uv "$POL/Pi_0/openpi" Pi_0; }
# InternW0_delta: causal-conv1d==1.6.2.post1 is an sdist whose setup.py needs nvcc
# (the host has no CUDA toolkit) before it would fetch its own prebuilt wheel.  We
# fetch that same upstream release wheel (torch2.10/cu12/cxx11abi/cp311, sha pinned)
# and expose it with PIP_FIND_LINKS; install.sh itself is unchanged.
# The release asset is named ...+cu12torch2.10cxx11abiTRUE-...whl but its metadata says
# plain 1.6.2.post1, which pip rejects as inconsistent; it is saved under the plain name.
CCONV_URL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1+cu12torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
CCONV_WHL="causal_conv1d-1.6.2.post1-cp311-cp311-linux_x86_64.whl"
CCONV_SHA=6ef3667660ef4404f9f833bb6988ab46e5024aec29af0f0575c195e307acf4ed
install_InternW0_delta() {
  [ -f "$LOCKS/InternW0_delta.ok" ] && { echo "already installed"; return; }
  mkdir -p "$SGRD_ROOT/wheels"
  [ -f "$SGRD_ROOT/wheels/$CCONV_WHL" ] || curl -sSL -o "$SGRD_ROOT/wheels/$CCONV_WHL" "$CCONV_URL"
  echo "$CCONV_SHA  $SGRD_ROOT/wheels/$CCONV_WHL" | sha256sum -c -
  conda env list | awk '{print $1}' | grep -qx internw0-delta && conda env remove -n internw0-delta -y
  (cd "$POL/InternW0_delta" && PIP_FIND_LINKS="$SGRD_ROOT/wheels" bash install.sh internw0-delta)
  # unpinned transitive dep that moved after the 24 GB box was installed (11-06): the 24 GB box's version
  conda run -n internw0-delta python -m pip install -q --no-deps "regex==2026.9.29"
  freeze_conda internw0-delta InternW0_delta
  touch "$LOCKS/InternW0_delta.ok"
}
install_InternVLA_A1()      { (cd "$POL/InternVLA_A1" && bash install.sh); freeze_conda internvla_a1 InternVLA_A1; }
install_Xiaomi_Robotics_0() { (cd "$POL/Xiaomi_Robotics_0" && bash install.sh); freeze_conda mibot Xiaomi_Robotics_0; }
# X_VLA pins av==14.4.0, which has no cp310 wheel on PyPI; the sdist needs the FFmpeg dev
# libraries (no apt on this host).  Same version from conda-forge (bundles FFmpeg) first, so
# requirements.txt finds it satisfied.
install_X_VLA() {
  conda env list | awk '{print $1}' | grep -qx XVLA || conda create -q -n XVLA python=3.10 -y
  conda install -q -n XVLA -c conda-forge "av=14.4.0" -y
  (cd "$POL/X_VLA" && bash install.sh)
  freeze_conda XVLA X_VLA
}
install_OpenWAM() {
  # policy README: conda env + torch 2.7.1 cu128 first, then install.sh
  conda env list | awk '{print $1}' | grep -qx openwam || conda create -q -n openwam python=3.10 -y
  conda activate openwam
  python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  (cd "$POL/OpenWAM" && bash install.sh)
  conda deactivate
  freeze_conda openwam OpenWAM
}

targets=()
for a in "$@"; do
  case "$a" in
    uv) targets+=(GR00T_N17 Pi_05 Pi_0) ;;
    conda) targets+=(InternW0_delta InternVLA_A1 Xiaomi_Robotics_0 X_VLA OpenWAM) ;;
    all) targets+=(GR00T_N17 Pi_05 Pi_0 InternW0_delta InternVLA_A1 Xiaomi_Robotics_0 X_VLA OpenWAM) ;;
    *) targets+=("$a") ;;
  esac
done
for t in "${targets[@]}"; do
  echo "=== install $t $(date -u +%FT%TZ)"
  "install_$t"
  echo "=== done $t $(date -u +%FT%TZ)"
done
