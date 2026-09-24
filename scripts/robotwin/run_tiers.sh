#!/usr/bin/env bash
# Run the remaining RoboTwin tasks tier by tier (20 scored episodes each by default, EPISODES=… to change), filter the events,
# fill the unified results table and pack what has to be sent back.
#
# Usage: run_tiers.sh WS RUNTIME TIER [TIER ...]      TIER in 1 2 3 4 (or "all")
#   WS      = SimuGuard workspace (holds SimuGuard/, RoboTwin/, .venv-lingbot/, runs/)
#   RUNTIME = LingBot-VA repro directory (.venv-client, .venv-server, checkpoints/)
# Optional env: EPISODES (default 50), TOOLS_BIN (ffmpeg dir), MODEL_FREE_MIB (default 18800)
set -uo pipefail
WS="${1:?workspace}"; RUNTIME="${2:?lingbot runtime}"; shift 2
EPISODES="${EPISODES:-20}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${RUNTIME}/.venv-client/bin/python"

declare -A TIER
# lift_pot, move_can_pot, place_bread_skillet and hanging_mug were run by us (2026-09-24, 20 episodes each)
TIER[1]="stack_blocks_three place_burger_fries place_container_plate place_empty_cup place_object_scale place_object_stand"
TIER[2]="place_phone_stand place_shoe place_dual_shoes place_mouse_pad place_fan place_a2b_left place_a2b_right handover_block handover_mic pick_dual_bottles"
TIER[3]="pick_diverse_bottles move_pillbottle_pad move_stapler_pad move_playingcard_away blocks_ranking_rgb blocks_ranking_size adjust_bottle grab_roller open_laptop open_microwave"
TIER[4]="turn_switch click_alarmclock click_bell press_stapler stamp_seal beat_block_hammer rotate_qrcode scan_object shake_bottle shake_bottle_horizontally"

tiers=("$@"); [ "${tiers[*]}" = "all" ] && tiers=(1 2 3 4)
for t in "${tiers[@]}"; do
    OUT="${WS}/runs/tier${t}_$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "${OUT}"
    echo "[tiers] tier ${t} -> ${OUT}"
    MODEL_FREE_MIB="${MODEL_FREE_MIB:-18800}" SIMUGUARD_WORKSPACE="${WS}" ROBOTWIN_ROOT="${WS}/RoboTwin" \
    LINGBOT_RUNTIME="${RUNTIME}" LINGBOT_MODEL="${LINGBOT_MODEL:-${RUNTIME}/checkpoints/lingbot-va-posttrain-robotwin-modelscope}" \
    TOOLS_BIN="${TOOLS_BIN:-${WS}/.tools/bin}" SIMUGUARD_CONFIG="${WS}/SimuGuard/configs/scale_no_bundles.json" \
    MANIFEST_PYTHON="${PY}" \
    bash "${HERE}/cross_task_campaign.sh" "${OUT}" "${EPISODES}" auto auto ${TIER[$t]}
    # post-processing needs numpy only; the client venv has it
    "${PY}" "${HERE}/gravity_filter.py" --runs "${OUT}"/*/ --output "${OUT}/gravity_filter.json"
    "${PY}" "${HERE}/results_table.py" --runs "${OUT}"/*/ --out "${OUT}/results_table.csv" | tee "${OUT}/results_table.md"
    # what to send back: everything but the per-substep contact traces (the replay needs controls + states only)
    tar --exclude='trace.jsonl.gz' -czf "${OUT}.tar.gz" -C "${WS}/runs" "$(basename "${OUT}")"
    echo "[tiers] tier ${t} done: ${OUT}.tar.gz ($(du -h "${OUT}.tar.gz" | cut -f1))"
done
