"""Per-substep monitor: ground truth -> detectors -> events -> replay bundles.

The monitor is strictly observational.  Every exception raised while reading
state, running detectors or writing artefacts is caught and recorded, so an
instrumented evaluation behaves exactly like an uninstrumented one.
"""

from __future__ import annotations

import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .adapter import HookHandle, SimAdapter
from .detectors.base import Detector, DetectorContext
from .events import Event, EventStatus
from .recorder import EpisodeRecorder
from .controls import ControlLog
from .snapshot import ReplayBundle, SnapshotRing, build_replay_bundle
from .statelog import StateLog
from .types import BodyRole, SubstepFrame

MONITOR_SCHEMA = "simuguard-monitor-v1"


@dataclass
class MonitorConfig:
    track_roles: tuple[str, ...] = ("target", "container", "object", "robot")
    read_contacts: bool = True
    frame_buffer_substeps: int = 2000  # reference frames kept for bundles (only substeps still in buffer are compared)
    # snapshots / replay
    snapshot_interval_substeps: int = 100  # 0 disables snapshots and bundles
    snapshot_capacity: int = 16
    snapshot_native: bool = True
    control_log_maxlen: int | None = None  # None keeps the whole episode (needed for episode_start bundles)
    bundle_statuses: tuple[str, ...] = ("confirmed",)
    # "episode_start": initial snapshot + every control since attach. Exact on RoboTwin when the
    #                  env is rebuilt with the same seed (verified bit-exact); restore method "none".
    # "snapshot":      latest periodic snapshot before onset; fast but approximate in contact
    #                  (PhysX pack/unpack does not restore solver warm-start state).
    bundle_mode: str = "episode_start"
    bundle_post_substeps: int = 250
    # artefacts
    save_episode_logs: bool = True  # controls.npz + states.npz + initial_snapshot.json.gz (exact replay inputs)
    save_snapshots: bool = False  # also every native snapshot still held (snapshots.npz): replay from any of them
    state_log_roles: tuple[str, ...] = ("target", "container", "object")
    trace_decimation: int = 1
    max_recorded_errors: int = 50

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MonitorConfig":
        data = dict(data or {})
        for key in ("track_roles", "bundle_statuses", "state_log_roles"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)


class SubstepMonitor:
    def __init__(
        self,
        adapter: SimAdapter,
        detectors: Iterable[Detector],
        *,
        episode_id: str = "episode",
        config: MonitorConfig | None = None,
        recorder: EpisodeRecorder | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.adapter = adapter
        self.detectors = list(detectors)
        self.episode_id = str(episode_id)
        self.cfg = config or MonitorConfig()
        self.recorder = recorder
        self.metadata = dict(metadata or {})

        self.substep = 0
        self.context: DetectorContext | None = None
        self.tracked_ids: list[str] = []
        self.events: dict[str, Event] = {}
        self.errors: list[dict[str, Any]] = []
        self.error_count = 0
        self.frames: deque[SubstepFrame] = deque(maxlen=max(1, self.cfg.frame_buffer_substeps))
        self.snapshots = SnapshotRing(
            capacity=self.cfg.snapshot_capacity,
            interval_substeps=max(1, self.cfg.snapshot_interval_substeps),
        )
        if self.cfg.bundle_mode not in ("episode_start", "snapshot"):
            raise ValueError(f"unknown bundle_mode: {self.cfg.bundle_mode}")
        self.controls = ControlLog(maxlen=self.cfg.control_log_maxlen)
        self.bundles: dict[str, str] = {}
        self._pending_bundles: dict[str, tuple[int, int, dict[str, Any]]] = {}
        self.state_log: StateLog | None = None
        self._hook: HookHandle | None = None
        self._last_frame: SubstepFrame | None = None
        self._started = 0.0
        self._summary: dict[str, Any] | None = None

    # ------------------------------------------------------------------ lifecycle
    @property
    def attached(self) -> bool:
        return self._hook is not None and not self._hook.removed

    @property
    def snapshots_enabled(self) -> bool:
        return self.cfg.snapshot_interval_substeps > 0

    def attach(self) -> "SubstepMonitor":
        if self.attached:
            raise RuntimeError("monitor already attached")
        bodies = self.adapter.bodies(refresh=True)
        roles = {BodyRole(role) for role in self.cfg.track_roles}
        self.tracked_ids = sorted(b for b, info in bodies.items() if info.role in roles)
        self.context = DetectorContext(episode_id=self.episode_id, timestep=self.adapter.timestep(), bodies=bodies)
        for detector in self.detectors:
            detector.reset(self.context)
        state_roles = {BodyRole(role) for role in self.cfg.state_log_roles}
        self.state_log = StateLog(sorted(b for b in self.tracked_ids if bodies[b].role in state_roles))
        if self.recorder is not None:
            trace_ids = {b for b in self.tracked_ids if bodies[b].role != BodyRole.ROBOT}
            if self.recorder.trace_body_ids is None:
                self.recorder.trace_body_ids = trace_ids
            self.recorder.begin(self._manifest())
        if self.snapshots_enabled:
            self._guard("initial_snapshot", self._capture_snapshot)
            self.snapshots.pin_before(0)
            self._guard("initial_control", self._capture_control, 0)
        self._started = time.time()
        self._hook = self.adapter.install_substep_hook(
            self._on_substep, before=self._before_substep if self.snapshots_enabled else None
        )
        return self

    def detach(self) -> None:
        if self._hook is not None:
            self._hook.remove()

    def __enter__(self) -> "SubstepMonitor":
        return self.attach()

    def __exit__(self, *exc: Any) -> None:
        self.finalize()

    # ------------------------------------------------------------------ per substep
    def _before_substep(self) -> None:
        # Actuation for step k is only observable right before it runs.
        self._guard("capture_control", self._capture_control, self.substep + 1)

    def _on_substep(self) -> None:
        self.substep += 1
        self._guard("process_substep", self._process)

    def _process(self) -> None:
        assert self.context is not None
        frame = self.adapter.read_frame(self.substep, self.tracked_ids, with_contacts=self.cfg.read_contacts)
        self.context.previous = self._last_frame
        self.frames.append(frame)
        self._last_frame = frame
        if self.state_log is not None:
            self.state_log.append(frame.substep, frame.states)

        if self.snapshots_enabled and self.snapshots.due(self.substep):
            self._guard("capture_snapshot", self._capture_snapshot)

        for detector in self.detectors:
            new_events = self._guard(f"detector:{detector.name}", detector.observe, frame, self.context) or []
            for event in new_events:
                self._record_event(event)

        if self.recorder is not None:
            self._guard("record_frame", self.recorder.frame, frame)
        self._flush_bundles(final=False)

    def _capture_control(self, substep: int) -> None:
        record = self.adapter.capture_control()
        record.substep = substep
        self.controls.append(record)

    def _capture_snapshot(self) -> None:
        self.snapshots.add(self.adapter.capture_snapshot(self.substep, include_native=self.cfg.snapshot_native))

    def _record_event(self, event: Event) -> None:
        self.events[event.event_id] = event
        if self.recorder is not None:
            self._guard("record_event", self.recorder.event, event)
        if (
            self.snapshots_enabled
            and event.status.value in self.cfg.bundle_statuses
            and event.event_id not in self._pending_bundles
            and event.event_id not in self.bundles
        ):
            if self.cfg.bundle_mode == "snapshot":
                self.snapshots.pin_before(event.onset_substep)
            self._pending_bundles[event.event_id] = (
                event.onset_substep,
                event.onset_substep + self.cfg.bundle_post_substeps,
                event.to_dict(),
            )

    def _flush_bundles(self, *, final: bool) -> None:
        for event_id, (onset, end, trigger) in list(self._pending_bundles.items()):
            if not final and self.substep < end:
                continue
            end = min(end, self.substep)
            bundle = self._guard("build_bundle", self._build_bundle, onset, end, trigger)
            self._pending_bundles.pop(event_id, None)
            if bundle is None:
                continue
            if self.recorder is not None:
                path = self._guard("save_bundle", self.recorder.bundle, event_id, bundle)
                self.bundles[event_id] = str(path) if path is not None else "save_failed"
            else:
                self.bundles[event_id] = "in_memory"

    def _build_bundle(self, onset: int, end: int, trigger: dict[str, Any]) -> ReplayBundle:
        start_from = 0 if self.cfg.bundle_mode == "episode_start" else onset
        snapshot = self.snapshots.latest_at_or_before(start_from)
        start = snapshot.substep if snapshot is not None else onset
        reference_ids = {b for b in self.tracked_ids if self.context and self.context.bodies[b].role != BodyRole.ROBOT}
        frames = [f.to_dict(reference_ids, decimals=None) for f in self.frames if start < f.substep <= end]
        return build_replay_bundle(
            self.snapshots,
            self.controls,
            trigger_substep=start_from,
            end_substep=end,
            reference_frames=frames,
            trigger={**trigger, "bundle_mode": self.cfg.bundle_mode},
            metadata={"episode_id": self.episode_id, "bundle_mode": self.cfg.bundle_mode, **self.metadata},
        )

    # ------------------------------------------------------------------ end of episode
    def finalize(self) -> dict[str, Any]:
        if self._summary is not None:
            return self._summary
        self.detach()
        if self.context is not None:
            for detector in self.detectors:
                for event in self._guard(f"finalize:{detector.name}", detector.finalize, self._last_frame, self.context) or []:
                    self._record_event(event)
        self._flush_bundles(final=True)
        if self.recorder is not None and self.cfg.save_episode_logs:
            self._guard("save_episode_logs", self._save_episode_logs)
        summary = self.summary()
        if self.recorder is not None:
            self._guard("recorder_end", self.recorder.end, summary)
        self._summary = summary
        return summary

    def _save_episode_logs(self) -> None:
        assert self.recorder is not None
        initial = self.snapshots.latest_at_or_before(0) if self.snapshots_enabled else None
        self.recorder.episode_logs(controls=self.controls, states=self.state_log, initial_snapshot=initial)
        if self.cfg.save_snapshots and self.snapshots_enabled:
            held = [self.snapshots.latest_at_or_before(k) for k in self.snapshots.substeps()]
            self.recorder.snapshot_archive([s for s in held if s is not None and s.native_state is not None])

    def summary(self) -> dict[str, Any]:
        latest = [event.to_dict() for event in self.events.values()]
        confirmed = [e for e in self.events.values() if e.status == EventStatus.CONFIRMED]
        flags = [e for e in self.events.values() if e.status == EventStatus.FLAG]
        by_detector: dict[str, dict[str, int]] = {}
        for event in self.events.values():
            counts = by_detector.setdefault(event.detector, {})
            counts[event.status.value] = counts.get(event.status.value, 0) + 1
        return {
            "schema_version": MONITOR_SCHEMA,
            "episode_id": self.episode_id,
            "adapter": self.adapter.name,
            "substeps": self.substep,
            "timestep_s": self.context.timestep if self.context else None,
            "simulated_time_s": self.substep * (self.context.timestep if self.context else 0.0),
            "wall_time_s": time.time() - self._started if self._started else 0.0,
            "tracked_bodies": len(self.tracked_ids),
            "event_counts_by_detector": by_detector,
            "confirmed_count": len(confirmed),
            "flag_count": len(flags),
            "needs_human_review": bool(confirmed or flags),
            "events": latest,
            "bundles": self.bundles,
            "snapshots_held": self.snapshots.substeps(),
            "control_log": {
                "records": len(self.controls),
                "oldest": self.controls.oldest_substep,
                "newest": self.controls.newest_substep,
                "dtype": str(self.controls.dtype.__name__ if hasattr(self.controls.dtype, "__name__") else self.controls.dtype),
                "in_memory_bytes": self.controls.nbytes,
            },
            "state_log": {"bodies": self.state_log.body_ids if self.state_log else [], "substeps": len(self.state_log) if self.state_log else 0},
            "error_count": self.error_count,
            "errors": self.errors,
            "metadata": self.metadata,
        }

    # ------------------------------------------------------------------ helpers
    def _manifest(self) -> dict[str, Any]:
        assert self.context is not None
        caps = self.adapter.capabilities()
        return {
            "schema_version": MONITOR_SCHEMA,
            "episode_id": self.episode_id,
            "adapter": self.adapter.name,
            "capabilities": asdict(caps),
            "timestep_s": self.context.timestep,
            "monitor_config": asdict(self.cfg),
            "detectors": {d.name: d.config() for d in self.detectors},
            "bodies": {k: v.to_dict() for k, v in sorted(self.context.bodies.items())},
            "tracked_body_ids": self.tracked_ids,
            "task_ground_truth": self._guard("task_ground_truth", self.adapter.task_ground_truth) or {},
            "metadata": self.metadata,
        }

    def _guard(self, where: str, fn: Any, *args: Any) -> Any:
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001 - instrumentation must never break evaluation
            self.error_count += 1
            if len(self.errors) < self.cfg.max_recorded_errors:
                self.errors.append(
                    {
                        "substep": self.substep,
                        "where": where,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=4),
                    }
                )
            return None
