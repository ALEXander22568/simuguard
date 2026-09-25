"""Run the official RoboTwin evaluator with SimuGuard monitoring (single process).

Nothing in RoboTwin or XPolicyLab is modified.  The official
``scripts/eval_policy_xpolicylab.py`` is imported and its ``class_decorator`` is
wrapped so that the returned task env gets *instance-level* lifecycle hooks:

    setup_demo()  returns -> start a segment: RoboTwinAdapter + SubstepMonitor.attach()
    play_once()           -> mark the segment as phase "expert" (official expert check)
    close_env()   before  -> finalize the monitor while the scene still exists

Every ``setup_demo .. close_env`` span is one *segment*.  In the official flow
each evaluated seed produces an ``expert`` segment (scripted planner, seed
filter) followed by a ``policy`` segment (the model rollout that is scored).

Usage::

    python -m simuguard.integrations.robotwin_eval \\
        --robotwin-root /path/to/RoboTwin --simuguard-out runs/eval_x \\
        -- <official eval_policy_xpolicylab.py arguments>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .. import __version__
from ..adapters.robotwin.adapter import RoboTwinAdapter
from ..adapters.robotwin.env import official_eval_module, robotwin_provenance
from ..core.monitor import MonitorConfig, SubstepMonitor
from ..core.recorder import EpisodeRecorder
from ..core.types import BodyRole
from ..presets import default_detectors

INDEX_SCHEMA = "simuguard-robotwin-eval-segments-v1"


def eval_monitor_config(overrides: dict[str, Any] | None = None) -> MonitorConfig:
    base = {
        "snapshot_interval_substeps": 100,
        "snapshot_capacity": 5000,
        "frame_buffer_substeps": 2000,
        "control_log_maxlen": None,
        "bundle_mode": "episode_start",
    }
    base.update(overrides or {})
    return MonitorConfig.from_dict(base)


SIM_INTERVENTIONS = ("none", "solver_high", "solver_default", "mass_100g", "mass_50g", "depen_1.0", "depen_0.1")
# depen_<v>: PhysX max depenetration velocity <v> m/s on every dynamic body and link (RoboTwin: unbounded)


def recorded_sim_intervention(metadata: dict[str, Any]) -> str | None:
    """Name of the closed-loop intervention a recorded segment ran under (None if default physics)."""

    name = ((metadata or {}).get("sim_intervention") or {}).get("name")
    return None if name in (None, "none") else str(name)


def reapply_recorded_intervention(adapter: RoboTwinAdapter, metadata: dict[str, Any]) -> str | None:
    """Re-apply a segment's recorded intervention before replaying it.

    The evaluator applies it at the first ``take_action``, whose first substep
    is substep 1 of the segment, so applying it before the replay reproduces
    the recorded physics exactly.
    """

    name = recorded_sim_intervention(metadata)
    if name is not None:
        apply_sim_intervention(adapter, name)
    return name


def apply_sim_intervention(adapter: RoboTwinAdapter, name: str) -> dict[str, Any]:
    """Change one simulation setting for the scored rollout (closed loop).

    Applied at the first ``take_action`` of a segment, i.e. only to the policy
    rollout: the official expert check keeps default settings so the same seeds
    pass the gate and the two runs stay paired.
    """

    if name.startswith("depen_"):
        cap = float(name.split("_", 1)[1])
        count = adapter.set_max_depenetration_velocity(cap)
        return {"name": name, "max_depenetration_velocity_mps": cap, "bodies": count}
    before = {"solver_iterations": adapter.solver_iterations()}
    targets = adapter.body_ids_with_role(BodyRole.TARGET)
    before["target_mass_kg"] = {b: adapter.mass(b) for b in targets}
    if name in ("none", None):
        return {"name": "none", "before": before, "after": before}
    if name == "solver_high":
        adapter.set_solver_iterations(position=32, velocity=8)
    elif name == "solver_default":
        adapter.set_solver_iterations(position=10, velocity=1)
    elif name.startswith("mass_"):
        grams = float(name.split("_")[1].rstrip("g"))
        for body_id in targets:
            adapter.set_mass(body_id, grams / 1000.0)
    else:
        raise ValueError(f"unknown sim intervention: {name}")
    after = {
        "solver_iterations": adapter.solver_iterations(),
        "target_mass_kg": {b: adapter.mass(b) for b in targets},
    }
    return {"name": name, "before": before, "after": after}


@dataclass
class Segment:
    index: int
    seed: int | None
    episode_index: int | None
    output_dir: str | None
    started_at: float
    phase: str = "policy"
    monitored: bool = False
    play_once_wall_s: float | None = None
    setup_wall_s: float | None = None
    info: dict[str, Any] = field(default_factory=dict)


class RoboTwinEvalInstrumentation:
    """Instance-level lifecycle hooks for a RoboTwin task env."""

    def __init__(
        self,
        output_root: str | Path,
        *,
        enabled: bool = True,
        monitor_expert: bool = True,
        monitor_config: dict[str, Any] | None = None,
        detector_config: dict[str, Any] | None = None,
        task_name: str | None = None,
        sim_intervention: str = "none",
    ) -> None:
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.monitor_expert = monitor_expert
        self.monitor_config = monitor_config or {}
        self.detector_config = detector_config or {}
        self.task_name = task_name
        if sim_intervention not in SIM_INTERVENTIONS:
            raise ValueError(f"sim_intervention must be one of {SIM_INTERVENTIONS}")
        self.sim_intervention = sim_intervention
        self.segments: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self._count = 0
        self._current: Segment | None = None
        self._monitor: SubstepMonitor | None = None
        self._adapter: RoboTwinAdapter | None = None

    # ------------------------------------------------------------------ patching
    def wrap_class_decorator(self, original: Callable[[str], Any]) -> Callable[[str], Any]:
        def class_decorator(task_name: str) -> Any:
            env = original(task_name)
            self.task_name = self.task_name or task_name
            return self.instrument(env)

        return class_decorator

    def instrument(self, env: Any) -> Any:
        if getattr(env, "_simuguard_instrumented", False):
            return env
        setup_demo, play_once, close_env, take_action = (
            env.setup_demo, env.play_once, env.close_env, env.take_action,
        )

        def patched_setup_demo(*args: Any, **kwargs: Any) -> Any:
            if self._current is not None:  # previous segment never closed (e.g. exception path)
                self._end_segment(env, reason="superseded_by_setup_demo")
            started = time.time()
            result = setup_demo(*args, **kwargs)
            self._begin_segment(env, kwargs, setup_wall_s=time.time() - started)
            return result

        def patched_play_once(*args: Any, **kwargs: Any) -> Any:
            if self._current is not None:
                self._current.phase = "expert"
                if self._monitor is not None and not self.monitor_expert:
                    self._discard_monitor()
            started = time.time()
            try:
                return play_once(*args, **kwargs)
            finally:
                if self._current is not None:
                    self._current.play_once_wall_s = time.time() - started

        def patched_take_action(*args: Any, **kwargs: Any) -> Any:
            segment = self._current
            if (
                segment is not None
                and self.sim_intervention != "none"
                and "sim_intervention" not in segment.info
                and self._adapter is not None
            ):
                try:
                    segment.info["sim_intervention"] = apply_sim_intervention(self._adapter, self.sim_intervention)
                    if self._monitor is not None:
                        self._monitor.metadata["sim_intervention"] = segment.info["sim_intervention"]
                except Exception:  # noqa: BLE001
                    segment.info["sim_intervention_error"] = traceback.format_exc(limit=3)
                    self.errors.append({"segment": segment.index, "where": "intervention", "traceback": traceback.format_exc()})
            return take_action(*args, **kwargs)

        def patched_close_env(*args: Any, **kwargs: Any) -> Any:
            if self._current is not None:
                self._end_segment(env, reason="close_env")
            return close_env(*args, **kwargs)

        env.setup_demo = patched_setup_demo
        env.play_once = patched_play_once
        env.close_env = patched_close_env
        env.take_action = patched_take_action
        env._simuguard_instrumented = True
        return env

    # ------------------------------------------------------------------ segments
    def _begin_segment(self, env: Any, kwargs: dict[str, Any], *, setup_wall_s: float) -> None:
        seed = kwargs.get("seed")
        episode_index = kwargs.get("now_ep_num")
        index = self._count
        self._count += 1
        name = f"seg{index:04d}_seed{seed}_ep{episode_index}"
        segment = Segment(index, seed, episode_index, None, time.time(), setup_wall_s=setup_wall_s)
        self._current = segment
        if not self.enabled:
            return
        try:
            adapter = RoboTwinAdapter(env, task_name=self.task_name)
            gate = None
            try:
                gate = adapter.containment_gate()
            except Exception as exc:  # noqa: BLE001
                segment.info["gate_error"] = f"{type(exc).__name__}: {exc}"
            out = self.output_root / "segments" / name
            monitor = SubstepMonitor(
                adapter,
                default_detectors(self.detector_config, ejection_gate=gate),
                episode_id=f"{self.task_name}:{name}",
                config=eval_monitor_config(self.monitor_config),
                recorder=EpisodeRecorder(out),
                metadata={"task": self.task_name, "seed": seed, "episode_index": episode_index, "segment": index},
            )
            monitor.attach()
            self._adapter, self._monitor = adapter, monitor
            segment.output_dir = str(out)
            segment.monitored = True
        except Exception:  # noqa: BLE001 - never break the evaluation
            self.errors.append({"segment": index, "where": "begin", "traceback": traceback.format_exc()})
            self._adapter = self._monitor = None

    def _discard_monitor(self) -> None:
        assert self._current is not None
        try:
            self._monitor.detach()
            if self._monitor.recorder is not None:
                self._monitor.recorder.end({"discarded": "expert monitoring disabled"})
        except Exception:  # noqa: BLE001
            pass
        self._monitor = self._adapter = None
        self._current.monitored = False
        self._current.info["monitoring"] = "disabled_for_expert"

    def _end_segment(self, env: Any, *, reason: str) -> None:
        segment = self._current
        assert segment is not None
        self._current = None
        ground_truth = _episode_outcome(env)
        record = asdict(segment)
        record.update(
            {
                "task": self.task_name,
                "end_reason": reason,
                "wall_s": time.time() - segment.started_at,
                "outcome": ground_truth,
            }
        )
        if self._monitor is not None:
            try:
                self._monitor.metadata.update({"phase": segment.phase, "outcome": ground_truth})
                summary = self._monitor.finalize()
                record["monitor"] = {
                    "substeps": summary["substeps"],
                    "error_count": summary["error_count"],
                    "confirmed_count": summary["confirmed_count"],
                    "flag_count": summary["flag_count"],
                    "event_counts_by_detector": summary["event_counts_by_detector"],
                    "needs_human_review": summary["needs_human_review"],
                    "hook_removed": bool(self._monitor._hook and self._monitor._hook.removed),
                }
                record["artifact_bytes"] = _dir_sizes(Path(segment.output_dir))
            except Exception:  # noqa: BLE001
                self.errors.append({"segment": segment.index, "where": "end", "traceback": traceback.format_exc()})
        self._monitor = self._adapter = None
        self.segments.append(record)
        with open(self.output_root / "segments.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def write_run_summary(self, extra: dict[str, Any] | None = None) -> Path:
        payload = {
            "schema_version": INDEX_SCHEMA,
            "simuguard_version": __version__,
            "simuguard_commit": _git_commit(Path(__file__).resolve().parents[2]),
            "enabled": self.enabled,
            "monitor_expert": self.monitor_expert,
            "sim_intervention": self.sim_intervention,
            "task": self.task_name,
            "segments": len(self.segments),
            "by_phase": _count_by(self.segments, "phase"),
            "instrumentation_errors": self.errors,
            **(extra or {}),
        }
        path = self.output_root / "run_summary.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path


def _episode_outcome(env: Any) -> dict[str, Any]:
    outcome: dict[str, Any] = {}
    for attr in ("eval_success", "plan_success", "take_action_cnt", "step_lim"):
        value = getattr(env, attr, None)
        outcome[attr] = value.item() if hasattr(value, "item") else value
    try:
        outcome["check_success"] = bool(env.check_success())
    except Exception as exc:  # noqa: BLE001
        outcome["check_success_error"] = f"{type(exc).__name__}: {exc}"
    try:
        outcome["instruction"] = env.get_instruction()
    except Exception:  # noqa: BLE001
        pass
    return outcome


def _dir_sizes(path: Path) -> dict[str, int]:
    sizes = {p.name if p.parent == path else str(p.relative_to(path)): p.stat().st_size for p in path.rglob("*") if p.is_file()}
    sizes["_total"] = sum(sizes.values())
    return sizes


def _count_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[str(item.get(key))] = counts.get(str(item.get(key)), 0) + 1
    return counts


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def set_default_policy_request_timeout(robotwin_root: str | Path, timeout_s: float) -> dict[str, Any]:
    """Default XPolicyLab ``WsModelClient(request_timeout_s=None)`` to ``timeout_s``.

    Upstream's client default is 120 s and the official evaluator exposes no flag
    for it.  LingBot-VA's first ``reset`` encodes the instruction with an 11 GB
    text encoder offloaded to CPU, which exceeded 120 s on 4090-hexa-node2.
    Only an unset value is replaced; an explicit caller value is kept.
    """

    xpl = str(Path(robotwin_root).resolve() / "XPolicyLab")
    if xpl not in sys.path:
        sys.path.insert(0, xpl)
    from client_server.ws import model_client  # noqa: WPS433 - XPolicyLab runtime import

    cls = model_client.WsModelClient
    if getattr(cls, "_simuguard_timeout_patch", None) is not None:
        return {"request_timeout_s_default": cls._simuguard_timeout_patch, "already_patched": True}
    original_init = cls.__init__

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("request_timeout_s") is None:
            kwargs["request_timeout_s"] = float(timeout_s)
        original_init(self, *args, **kwargs)

    cls.__init__ = __init__
    cls._simuguard_timeout_patch = float(timeout_s)
    return {"request_timeout_s_default": float(timeout_s), "upstream_default_s": 120.0}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        split = argv.index("--")
        own, official = argv[:split], argv[split + 1 :]
    else:
        own, official = argv, []
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--simuguard-out", required=True)
    parser.add_argument("--simuguard-config", default=None, help='JSON: {"detectors": {...}, "monitor": {...}}')
    parser.add_argument("--simuguard-disable", action="store_true", help="run the identical path without monitors")
    parser.add_argument("--no-expert-monitoring", action="store_true")
    parser.add_argument(
        "--sim-intervention", default="none", choices=SIM_INTERVENTIONS,
        help="change one simulation setting for the scored rollout only (closed-loop intervention)",
    )
    parser.add_argument(
        "--policy-request-timeout-s",
        type=float,
        default=None,
        help="default XPolicyLab ws request timeout when the evaluator leaves it unset (upstream 120 s)",
    )
    args = parser.parse_args(own)

    config = json.loads(Path(args.simuguard_config).read_text()) if args.simuguard_config else {}
    out = Path(args.simuguard_out).resolve()
    module = official_eval_module(args.robotwin_root)  # chdir + sys.path like the official script
    instrumentation = RoboTwinEvalInstrumentation(
        out,
        enabled=not args.simuguard_disable,
        monitor_expert=not args.no_expert_monitoring,
        monitor_config=config.get("monitor"),
        detector_config=config.get("detectors"),
        sim_intervention=args.sim_intervention,
    )
    module.class_decorator = instrumentation.wrap_class_decorator(module.class_decorator)
    runtime_overrides: dict[str, Any] = {}
    if args.policy_request_timeout_s is not None:
        runtime_overrides["policy_client"] = set_default_policy_request_timeout(
            args.robotwin_root, args.policy_request_timeout_s
        )

    sys.argv = [str(Path(module.__file__).resolve())] + official
    started = time.time()
    status = 0
    try:
        from test_render import Sapien_TEST  # same preflight as the official __main__

        Sapien_TEST()
        parsed = module.parse_args()
        if module.parse_bool(parsed.get("eval_batch", False)):
            raise SystemExit("simuguard robotwin_eval currently supports eval_batch=false only")
        module.main(parsed)
    except SystemExit as exc:
        status = int(exc.code) if isinstance(exc.code, int) else 1
        raise
    except Exception:  # noqa: BLE001
        status = 1
        traceback.print_exc()
    finally:
        instrumentation.write_run_summary(
            {
                "wall_s": time.time() - started,
                "exit_status": status,
                "official_argv": official,
                "runtime_overrides": runtime_overrides,
                "robotwin": robotwin_provenance(args.robotwin_root),
            }
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
