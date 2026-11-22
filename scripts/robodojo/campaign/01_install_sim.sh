#!/usr/bin/env bash
# Install the RoboDojo simulator stack (Isaac Sim 5.1 + Isaac Lab + cuRobo) in an
# isolated tree under $SGRD_ROOT, at the SHAs in pins.env. It follows RoboDojo's
# own scripts/install.sh; the only changes are: miniconda lives in
# $SGRD_ROOT/miniconda3 (install.sh hardcodes $HOME/miniconda3, so HOME is set
# to $SGRD_ROOT), and submodules are checked out at pinned SHAs instead of
# `git submodule update --remote`.
#
# System packages (cmake build-essential ffmpeg git-lfs) must already be
# installed: this script never runs apt (on the 24 GB box any apt install
# reconfigures a half-installed nvidia-dkms and drops the kernel module).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/pins.env"
for pkg in build-essential ffmpeg git-lfs; do
  dpkg -s "$pkg" >/dev/null 2>&1 || { echo "missing system package $pkg; install it first (then check dkms status)"; exit 1; }
done
mkdir -p "$SGRD_ROOT"
# cmake: the system package if present; otherwise (no apt) the same version from PyPI in its own venv,
# linked as $SGRD_ROOT/bin/cmake, so the RoboDojo env itself is unchanged
export PATH="$SGRD_ROOT/bin:$PATH"
if ! command -v cmake >/dev/null 2>&1; then
  "${UV:-$HOME/.local/bin/uv}" venv -q --python 3.11 "$SGRD_ROOT/cmakevenv"
  "${UV:-$HOME/.local/bin/uv}" pip install -q --python "$SGRD_ROOT/cmakevenv/bin/python" "cmake==$CMAKE_VERSION"
  mkdir -p "$SGRD_ROOT/bin"; ln -sfn "$SGRD_ROOT/cmakevenv/bin/cmake" "$SGRD_ROOT/bin/cmake"
fi
cmake --version | head -1
cd "$SGRD_ROOT"

# 1. source at pinned SHAs
if [[ ! -d RoboDojo/.git ]]; then git clone -q "$RD_REPO_URL" RoboDojo; fi
cd RoboDojo
git fetch -q origin
git checkout -q "$RD_SHA"
git submodule sync -q
git submodule update --init -q XPolicyLab third_party/IsaacLab third_party/curobo
for pair in "XPolicyLab:$XPL_SHA" "third_party/IsaacLab:$ISAACLAB_SHA" "third_party/curobo:$CUROBO_SHA"; do
  d="${pair%%:*}"; sha="${pair#*:}"
  git -C "$d" fetch -q origin main
  git -C "$d" checkout -q "$sha"
done
git submodule status

# 2. isolated miniconda + RoboDojo env (install.sh steps "conda" and "base_deps")
export HOME="$SGRD_ROOT" PIP_USER=0 PYTHONNOUSERSITE=1
export PATH="$SGRD_ROOT/miniconda3/bin:$PATH"
if [[ ! -x "$SGRD_ROOT/miniconda3/bin/conda" ]]; then
  wget -q "$MINICONDA_URL" -O "$SGRD_ROOT/miniconda.sh"
  bash "$SGRD_ROOT/miniconda.sh" -b -p "$SGRD_ROOT/miniconda3"
  rm -f "$SGRD_ROOT/miniconda.sh"
fi
eval "$(conda shell.bash hook)"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
# the policy server scripts read deploy.yml with the conda *base* python (Pi_05/Pi_0:
# YAML_PYTHON="$(conda info --base)/bin/python"), which Miniconda ships without PyYAML
"$SGRD_ROOT/miniconda3/bin/python" -m pip install -q pyyaml==6.0.3
conda env list | grep -q "^RoboDojo " || conda create -q -n RoboDojo python=3.11 -y
conda activate RoboDojo
python -m pip install -q -r scripts/requirements.txt
python -m pip install -q opencv-python-headless==4.11.0.86 pillow matplotlib "scipy==1.15.3" scikit-learn
python -m pip install -q numpy==1.26.0

# 3. RoboDojo's own installer for isaacsim / isaaclab / curobo
export OMNI_KIT_ACCEPT_EULA=YES TERM=xterm-256color
bash scripts/install.sh --from isaacsim

# unpinned transitive deps that moved after the 24 GB box was installed (11-06): set to the 24 GB box's lock
python -m pip install -q --no-deps "regex==2026.9.29"

# 4. lock file for replay on another machine
mkdir -p "$SGRD_ROOT/locks"
python -m pip freeze --all | sort > "$SGRD_ROOT/locks/RoboDojo.pip-freeze.txt"
python -c "import isaacsim, torch; print('torch', torch.__version__, torch.version.cuda)"
dkms status 2>/dev/null | grep -i nvidia || true
echo "install_sim done"
