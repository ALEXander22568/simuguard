#!/usr/bin/env bash
# Install the whole native stack on a new box, in the order the numbered scripts need, with timings.
#   chain A: 01_install_sim.sh -> 02_install_policies.sh POLICIES
#   chain B (in parallel): 00_download.sh assets -> 00b_base_models.sh cosmos hf openpi
#   then: 00b_base_models.sh internw0 (needs the internw0-delta env, if installed) -> 03_link.sh
#         -> fingerprint_robodojo.py --policies POLICIES
# All base models are downloaded even for a subset of policies, so 03_link.sh and base_models.sha256
# are the same as on a full install.  Logs: $SGRD_ROOT/logs/install_box/.
# usage: bash install_box.sh POLICY...      (e.g. OpenWAM InternW0_delta)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/pins.env"
L="$SGRD_ROOT/logs/install_box"; mkdir -p "$L"
log() { echo "$(TZ=Asia/Seoul date '+%m-%d %H:%M:%S KST') $*" | tee -a "$L/install.log"; }
step() {  # name, command...
  local name=$1; shift; local t0; t0=$(date +%s)
  log "$name: start"
  "$@" > "$L/$name.log" 2>&1; local rc=$?
  log "$name: end rc=$rc after $(( ($(date +%s) - t0) / 60 )) min"
  return $rc
}
[[ $# -gt 0 ]] || { echo "usage: $0 POLICY..."; exit 2; }
log "install_box start: $*"
( step 00_assets bash "$HERE/00_download.sh" assets && step 00b_base bash "$HERE/00b_base_models.sh" cosmos hf openpi ) &
b=$!
step 01_sim bash "$HERE/01_install_sim.sh" || { log "01 failed"; wait $b; exit 1; }
step 02_policies bash "$HERE/02_install_policies.sh" "$@" || { log "02 failed"; wait $b; exit 1; }
wait $b || { log "downloads failed"; exit 1; }
if [[ " $* " == *" InternW0_delta "* ]]; then
  step 00b_internw0 bash "$HERE/00b_base_models.sh" internw0 || exit 1
fi
step 03_link bash "$HERE/03_link.sh" || exit 1
step fingerprint python3 "$HERE/fingerprint_robodojo.py" --policies "$@" --out "$SGRD_ROOT/locks/env_fingerprint.json"
log "install_box done"
