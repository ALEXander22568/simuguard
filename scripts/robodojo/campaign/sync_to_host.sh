#!/usr/bin/env bash
# Copy this SimuGuard checkout (package + campaign scripts) to a host, with a
# SOURCE.json that names the commit and hashes the uncommitted diff.
# usage: bash sync_to_host.sh "<ssh args>" [remote SGRD_ROOT]
#   e.g. bash sync_to_host.sh "-p 22 user@host-b"
set -euo pipefail
ssh_args=${1:?ssh args}; R=${2:-/path/to/simuguard-robodojo}
D=${SG_DEST:-simuguard-src}; S=${SG_SCRIPTS_DEST:-sgscripts}  # other names: a test copy next to the live one
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo"
commit=$(git rev-parse HEAD); branch=$(git branch --show-current)
diff_sha=$( { git diff HEAD; git ls-files --others --exclude-standard simuguard | sort | while read -r f; do echo "== $f"; cat "$f"; done; } | shasum -a 256 | cut -d' ' -f1)
dirty=$(git status --porcelain simuguard | wc -l | tr -d ' ')
src_json="$(mktemp -d "${TMPDIR:-/tmp}/sgsync.XXXX")/SOURCE.json"
printf '{"commit": "%s", "branch": "%s", "uncommitted_diff_sha256": "%s", "dirty_paths_in_simuguard": %s, "synced_utc": "%s"}\n' \
  "$commit" "$branch" "$diff_sha" "$dirty" "$(date -u +%FT%TZ)" > "$src_json"
# shellcheck disable=SC2086
rsync -a --delete -e "ssh -o ClearAllForwardings=yes ${ssh_args% *}" --exclude __pycache__ simuguard pyproject.toml "${ssh_args##* }:$R/$D/"
rsync -a -e "ssh -o ClearAllForwardings=yes ${ssh_args% *}" "$src_json" "${ssh_args##* }:$R/$D/SOURCE.json"
rsync -a --delete -e "ssh -o ClearAllForwardings=yes ${ssh_args% *}" --exclude __pycache__ scripts/robodojo/campaign/ "${ssh_args##* }:$R/$S/"
rm -rf "$(dirname "$src_json")"
echo "synced $commit ($branch, diff $diff_sha) to $R"
