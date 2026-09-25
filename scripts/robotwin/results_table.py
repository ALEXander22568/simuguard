#!/usr/bin/env python3
"""One row per task in the convention of the paper (Eq. 1): the unified results table.

    task | N | S | official | skipped | failures | E_fail | E_succ | R1 | R0 | audited | bound | open

* N         scored episodes, the benchmark's own count
* S         official successes; official = S / N
* skipped   seeds the evaluator discarded before reaching N, taken from the scored seed range, so seeds
            rejected as unstable before planning (which leave no SimuGuard segment) are counted too
* E_fail    failures that contain a physics-invalid event (after gravity_filter.py)
* E_succ    successes that contain one
* R1        of the E_fail failures, those that succeed when the policy resumes 1 s before the first event
            with the injected energy removed (rollback_retry.py --takeover --intervention vclamp_2.0)
* R0        the same failures resumed with the physics unchanged: the matched control arm
* audited   (S + R1 - R0) / N; equal to official when E_fail = 0; "-" until both arms cover every E_fail
* bound     (S + E_fail) / N: every failure that contains an event counted as a success
* open      E_fail failures that identical-action replay with the speed clamp turns into successes
            (attribute_failures.py, success of the vclamp_2.0 condition); a diagnostic, not a score

Run directories of one task (repetitions) are pooled into one row.  Reports are matched to episodes
by (parent directory, run directory, segment) names, so tables can be built from metadata copied off
other hosts.

Usage::

    python scripts/robotwin/results_table.py --runs runs/crosstask_x/* runs/campaign_y/default/rep* \\
        --takeover runs/takeover_can/vclamp_2.0 --control runs/takeover_can/baseline \\
        --attr runs/attr_x --out docs/robotwin_results_table.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

COLUMNS = ["task", "N", "S", "official", "skipped", "failures", "E_fail", "E_succ", "R1", "R0",
           "audited", "bound", "open", "runs"]


def segment_key(path: str | Path) -> tuple[str, str, str]:
    """(parent, run directory, segment) for .../<parent>/<run>/simuguard/segments/<segment>.

    The parent keeps repetitions of different configurations apart (default/rep1 vs mass_100g/rep1).
    """
    parts = Path(path).parts
    return parts[-5], parts[-4], parts[-1]


def load_reports(roots: list[str], name_glob: str, phase: str | None = None,
                 alias: dict[str, str] | None = None) -> dict[tuple[str, str, str], str]:
    verdicts: dict[tuple[str, str, str], str] = {}
    for root in roots:
        for report in Path(root).rglob(name_glob):
            if report.name == "summary.json":
                continue
            try:
                data = json.loads(report.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(data, dict) or "verdict" not in data or "segment" not in data:
                continue
            if phase and data.get("phase", phase) != phase:
                continue
            parent, run, segment = segment_key(data["segment"])
            verdicts[((alias or {}).get(parent, parent), run, segment)] = data["verdict"]
    return verdicts


def load_open_loop(roots: list[str], primary: str, alias: dict[str, str]) -> dict[tuple[str, str, str], bool]:
    """Whether identical-action replay under the primary counterfactual succeeded, per policy episode.

    Read from the per-condition results, not from the verdict string, which older reports computed with
    a different rule (any counterfactual succeeding).
    """
    outcome: dict[tuple[str, str, str], bool] = {}
    for root in roots:
        for report in Path(root).rglob("*.json"):
            try:
                data = json.loads(report.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(data, dict) or data.get("phase") != "policy" or "segment" not in data:
                continue
            conditions = {c.get("name"): c for c in data.get("conditions", []) if isinstance(c, dict)}
            if primary in conditions:
                ok = bool(conditions[primary].get("success"))
            elif "verdict" in data:
                ok = data["verdict"] == "environment_caused"
            else:
                continue
            parent, run, segment = segment_key(data["segment"])
            outcome[(alias.get(parent, parent), run, segment)] = ok
    return outcome


def start_seed(run: Path) -> int:
    command = run / "eval_command.txt"
    if command.is_file():
        match = re.search(r"--seed\s+(\d+)", command.read_text())
        if match:
            return 100000 * (1 + int(match.group(1)))
    return 100000


def run_counts(run: Path, treated: dict, control: dict, opened: dict) -> dict | None:
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    n, s = int(manifest["policy_episodes"]), int(manifest["policy_success"])
    scored = manifest.get("seeds_scored") or []
    counts = {"task": manifest["task"], "N": n, "S": s, "skipped": (max(scored) - start_seed(run) + 1 - n) if scored else 0,
              "E_fail": 0, "E_succ": 0, "R1": 0, "R0": 0, "treated": 0, "controlled": 0, "open": 0,
              "attributed": 0, "run": str(run)}
    for episode in manifest["episodes"]:
        policy = episode.get("policy")
        if not policy:
            continue
        segment = run / policy["segment"]
        gravity = segment / "events_gravity.json"
        if gravity.is_file():
            invalid = any(e.get("verdict") == "physics_invalid" for e in json.loads(gravity.read_text())["events"])
        else:
            invalid = policy["confirmed_events"] > 0  # gravity filter not run yet: raw detector output
        ok = bool(policy.get("eval_success"))
        counts["E_succ"] += ok and invalid
        if ok or not invalid:
            continue
        counts["E_fail"] += 1
        key = segment_key(segment)
        if key in treated:
            counts["treated"] += 1
            counts["R1"] += treated[key] == "rescued"
        if key in control:
            counts["controlled"] += 1
            counts["R0"] += control[key] == "rescued"
        if key in opened:
            counts["attributed"] += 1
            counts["open"] += opened[key]
    return counts


def pct(numerator: int, n: int) -> str:
    return f"{100.0 * numerator / n:.1f}" if n else "-"


def task_rows(runs: list[str], treated: dict, control: dict, opened: dict) -> list[dict]:
    pooled: dict[str, dict] = {}
    for run in runs:
        c = run_counts(Path(run), treated, control, opened)
        if c is None:
            continue
        row = pooled.setdefault(c["task"], {k: 0 for k in ("N", "S", "skipped", "E_fail", "E_succ", "R1", "R0",
                                                              "treated", "controlled", "open", "attributed")} | {"runs": []})
        for k in ("N", "S", "skipped", "E_fail", "E_succ", "R1", "R0", "treated", "controlled", "open", "attributed"):
            row[k] += c[k]
        row["runs"].append(c["run"])
    rows = []
    for task, r in sorted(pooled.items()):
        n, s, e_fail = r["N"], r["S"], r["E_fail"]
        if e_fail == 0:
            audited = pct(s, n)
        elif r["treated"] == e_fail and r["controlled"] == e_fail:
            audited = pct(s + r["R1"] - r["R0"], n)
        else:
            audited = "-"
        covered = r["treated"] == e_fail and r["controlled"] == e_fail and e_fail > 0
        rows.append({
            "task": task, "N": n, "S": s, "official": pct(s, n), "skipped": r["skipped"], "failures": n - s,
            "E_fail": e_fail, "E_succ": r["E_succ"],
            "R1": r["R1"] if covered else "-", "R0": r["R0"] if covered else "-",
            "audited": audited, "bound": pct(s + e_fail, n),
            "open": r["open"] if e_fail and r["attributed"] == e_fail else "-",
            "runs": ";".join(r["runs"]),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="task run directories (repetitions are pooled)")
    parser.add_argument("--takeover", nargs="*", default=[], help="rollback_retry --takeover output, energy removed")
    parser.add_argument("--control", nargs="*", default=[], help="rollback_retry --takeover output, physics unchanged")
    parser.add_argument("--attr", nargs="*", default=[], help="attribute_failures.py output (open-loop diagnostic)")
    parser.add_argument("--alias", nargs="*", default=[], metavar="REPORTED=ACTUAL",
                        help="parent directory named in reports -> name in --runs (reports made on a copy)")
    parser.add_argument("--primary", default="vclamp_2.0", help="open-loop counterfactual counted in the open column")
    parser.add_argument("--out", help="CSV to write; rows of other tasks already in it are kept")
    args = parser.parse_args()

    alias = dict(a.split("=", 1) for a in args.alias)
    treated = load_reports(args.takeover, "report.json", alias=alias)
    control = load_reports(args.control, "report.json", alias=alias)
    opened = load_open_loop(args.attr, args.primary, alias)
    rows = task_rows(args.runs, treated, control, opened)
    existing: dict[str, dict] = {}
    if args.out and Path(args.out).is_file():
        with open(args.out, newline="") as handle:
            existing = {r["task"]: r for r in csv.DictReader(handle) if set(r) >= set(COLUMNS[:-1])}
    for row in rows:
        existing[row["task"]] = row
    shown = COLUMNS[:-1]
    print("| " + " | ".join(shown) + " |")
    print("|" + "---|" * len(shown))
    for row in sorted(existing.values(), key=lambda r: r["task"]):
        print("| " + " | ".join(str(row[c]) for c in shown) + " |")
    if args.out:
        with open(args.out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(sorted(existing.values(), key=lambda r: r["task"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
