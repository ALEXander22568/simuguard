#!/usr/bin/env python3
"""Paired analysis of a seeds x configs x repeats campaign (scale_campaign.sh output).

The seed is the pairing unit: every configuration runs the same seeds, so a
seed's outcome under one configuration is compared with the same seed under
another.  Repeats of one configuration are *not* paired with each other; they
measure the policy's run-to-run variation and are averaged per seed.

Tests (reference config vs each other config):
* per repeat: exact McNemar on the seed-paired success table;
* pooled: Wilcoxon signed-rank on per-seed success rate (mean over repeats);
* Wilcoxon signed-rank on per-seed mean confirmed events (policy phase);
* seed-cluster bootstrap 95% CI of the success-rate difference.

Usage::

    python scripts/robotwin/campaign_analysis.py --campaign runs/campaign_x \\
        --reference default --output runs/campaign_x/analysis.json
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", "Mean of empty slice")


def load(campaign: Path) -> dict:
    """{config: {repeat: {seed: {phase: {...}}}}}"""
    data: dict = {}
    for config_dir in sorted(p for p in campaign.iterdir() if p.is_dir()):
        for rep_dir in sorted(config_dir.glob("rep*")):
            segments = rep_dir / "simuguard" / "segments"
            if not segments.is_dir():
                continue
            rep = int(rep_dir.name[3:])
            for summary_path in sorted(segments.glob("*/summary.json")):
                summary = json.loads(summary_path.read_text())
                meta = summary.get("metadata") or {}
                seed, phase = meta.get("seed"), meta.get("phase")
                if seed is None or phase is None:
                    continue
                entry = data.setdefault(config_dir.name, {}).setdefault(rep, {}).setdefault(int(seed), {})
                record = entry.setdefault(phase, {"events": 0, "segments": 0, "success": None})
                record["events"] += int(summary.get("confirmed_count", 0))
                record["segments"] += 1
                if phase == "policy":
                    record["success"] = bool((meta.get("outcome") or {}).get("eval_success"))
    return data


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    n = only_a + only_b
    p = 1.0 if n == 0 else float(stats.binomtest(only_a, n, 0.5).pvalue)
    return {"ref_only_success": only_a, "other_only_success": only_b, "p_value": p}


def wilcoxon(x: np.ndarray, y: np.ndarray) -> dict:
    diff = y - x
    if not np.any(diff):
        return {"n_nonzero": 0, "p_value": 1.0}
    result = stats.wilcoxon(x, y, zero_method="wilcox")
    return {"n_nonzero": int(np.count_nonzero(diff)), "statistic": float(result.statistic), "p_value": float(result.pvalue)}


def bootstrap_ci(diff: np.ndarray, iters: int = 20000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(iters, len(diff)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def per_config(data: dict, config: str, seeds: list[int]) -> dict:
    """(seed x repeat) arrays; NaN where the seed never reached the policy in that repeat."""

    reps = sorted(data[config])

    def cell(r: int, s: int, phase: str, key: str) -> float:
        record = data[config][r].get(s, {}).get(phase)
        if record is None:
            return 0.0 if phase == "expert" else float("nan")
        return float(record[key])

    success = np.array([[cell(r, s, "policy", "success") for r in reps] for s in seeds])
    events = np.array([[cell(r, s, "policy", "events") for r in reps] for s in seeds])
    expert = np.array([[cell(r, s, "expert", "events") for r in reps] for s in seeds])
    return {"reps": reps, "success": success, "events": events, "expert_events": expert}


def describe(block: dict) -> dict:
    success, events = block["success"], block["events"]
    ran = ~np.isnan(success)
    per_seed = np.nanmean(np.where(ran, success, np.nan), axis=1)
    rates = np.nansum(success, axis=0) / ran.sum(axis=0)
    ok, fail = ran & (success == 1), ran & (success == 0)
    return {
        "repeats": block["reps"],
        "policy_episodes_per_repeat": [int(v) for v in ran.sum(axis=0)],
        "success_per_repeat": [int(v) for v in np.nansum(success, axis=0)],
        "success_rate_per_repeat": [round(float(v), 4) for v in rates],
        "success_rate_pooled": round(float(np.nansum(success) / ran.sum()), 4),
        "success_rate_repeat_sd": round(float(rates.std(ddof=1)), 4) if len(rates) > 1 else None,
        "seeds_run_at_least_once": int(np.sum(ran.any(axis=1))),
        "seeds_always_success": int(np.sum(per_seed == 1)),
        "seeds_never_success": int(np.sum(per_seed == 0)),
        "seeds_mixed": int(np.sum((per_seed > 0) & (per_seed < 1))),
        "policy_episodes_with_events": [int(v) for v in ((events > 0) & ran).sum(axis=0)],
        "policy_events_per_episode_mean": round(float(np.nanmean(events)), 3),
        "policy_events_in_success_mean": round(float(events[ok].mean()), 3) if ok.any() else None,
        "policy_events_in_failure_mean": round(float(events[fail].mean()), 3) if fail.any() else None,
        "expert_events_per_seed_mean": round(float(block["expert_events"].mean()), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--reference", default="default")
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--min-policy-episodes", type=int, default=25, help="drop repeats that stopped early")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    campaign = Path(args.campaign).resolve()
    data = load(campaign)
    configs = args.configs or sorted(data)
    # RoboTwin skips a seed whose expert check fails, and that check is not
    # deterministic across runs, so repeats cover different seed sets.  Pairing
    # is therefore done on the seeds both sides actually ran.
    complete, dropped = {}, {}
    for config in configs:
        for rep, seeds in data.get(config, {}).items():
            n_policy = sum(1 for phases in seeds.values() if "policy" in phases)
            target = complete if n_policy >= args.min_policy_episodes else dropped
            target.setdefault(config, {})[rep] = seeds
    all_seeds = sorted({s for reps in complete.values() for seeds in reps.values() for s in seeds})

    blocks = {c: per_config(complete, c, all_seeds) for c in complete}
    report = {
        "campaign": str(campaign),
        "seeds_seen": len(all_seeds),
        "reference": args.reference,
        "incomplete_repeats_dropped": {c: sorted(r) for c, r in dropped.items()},
        "configs": {c: describe(b) for c, b in blocks.items()},
        "comparisons": {},
    }

    ref = blocks[args.reference]
    for config, block in blocks.items():
        if config == args.reference:
            continue
        n_rep = min(ref["success"].shape[1], block["success"].shape[1])
        per_repeat = []
        for i in range(n_rep):
            both = ~np.isnan(ref["success"][:, i]) & ~np.isnan(block["success"][:, i])
            test = mcnemar_exact(ref["success"][both, i] == 1, block["success"][both, i] == 1)
            test["paired_seeds"] = int(both.sum())
            per_repeat.append(test)
        ref_rate, other_rate = np.nanmean(ref["success"], axis=1), np.nanmean(block["success"], axis=1)
        common = ~np.isnan(ref_rate) & ~np.isnan(other_rate)
        ref_rate, other_rate = ref_rate[common], other_rate[common]
        ref_ev, other_ev = np.nanmean(ref["events"], axis=1)[common], np.nanmean(block["events"], axis=1)[common]
        report["comparisons"][f"{args.reference}_vs_{config}"] = {
            "paired_seeds": int(common.sum()),
            "per_seed_success_rate_mean": [round(float(ref_rate.mean()), 4), round(float(other_rate.mean()), 4)],
            "mcnemar_per_repeat": per_repeat,
            "wilcoxon_per_seed_success_rate": wilcoxon(ref_rate, other_rate),
            "success_diff_bootstrap_ci95": [round(v, 4) for v in bootstrap_ci(other_rate - ref_rate)],
            "seeds_better_worse_tied": [
                int(np.sum(other_rate > ref_rate)), int(np.sum(other_rate < ref_rate)), int(np.sum(other_rate == ref_rate))
            ],
            "policy_events_per_seed_mean": [round(float(ref_ev.mean()), 3), round(float(other_ev.mean()), 3)],
            "wilcoxon_per_seed_policy_events": wilcoxon(ref_ev, other_ev),
        }

    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
