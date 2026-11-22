"""Time window and per-body bookkeeping of the narrow depenetration intervention.

Pure Python, no SAPIEN: the replay runner drives it, and the unit tests drive it with a fake
adapter.  Substep convention (same as ``attribute_failures.py``): the replay loop handles control
record ``s`` by applying its actuation and then stepping the scene once; "the cap is active at
substep s" means it is set while that step runs.

Window of every narrow condition starts at ``start = max(1, t0 - lead)`` (lead = 0.1 s = 25
substeps at 4 ms, as in the paper's broad condition).  It ends:

* ``fixed``:    after substep ``t0 + after`` (after = 0.2 s = 50 substeps);
* ``resolved``: after the first substep ``s >= t0`` that completes ``n_resolved`` consecutive
  substeps (counted from t0 on) in which the body's deepest penetration is at most
  ``rest_offset + tolerance``; at the latest after substep ``t0 + max_after`` (1 s).

The original value of every capped body is read and stored when the window starts and written
back when it ends (it is not assumed to be any particular number).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WindowRule:
    t0: int
    lead: int = 25
    mode: str = "fixed"  # fixed | resolved | persistent
    after: int = 50
    n_resolved: int = 5
    threshold_m: float = 0.001  # rest offset + tolerance
    max_after: int = 250

    @property
    def start(self) -> int:
        return max(1, self.t0 - self.lead)


@dataclass
class DepenWindow:
    """Applies the cap at ``rule.start`` and restores at the end of the window."""

    rule: WindowRule
    bodies: list[str]
    cap: float
    originals: dict[str, float] = field(default_factory=dict)
    capped_values: dict[str, float] = field(default_factory=dict)
    restored_values: dict[str, float] = field(default_factory=dict)
    started_at: int | None = None
    ended_after: int | None = None  # last substep the cap was active
    end_reason: str | None = None
    _quiet: int = 0
    max_penetration_m: float = 0.0
    max_penetration_substep: int | None = None

    @property
    def active(self) -> bool:
        return self.started_at is not None and self.ended_after is None

    def before_step(self, adapter, substep: int) -> None:
        if self.started_at is None and substep == self.rule.start:
            self.originals = adapter.set_body_max_depenetration_velocity(self.bodies, self.cap)
            self.capped_values = {b: v for b, v in adapter.max_depenetration_velocities(self.originals).items()}
            self.started_at = substep

    def after_step(self, adapter, substep: int, penetration_m: float | None = None) -> None:
        """Call after the step of ``substep``; ``penetration_m`` is needed in ``resolved`` mode."""

        if not self.active or self.rule.mode == "persistent":
            return
        rule = self.rule
        end = False
        if rule.mode == "fixed":
            end = substep >= rule.t0 + rule.after
            reason = "fixed"
        elif rule.mode == "resolved":
            if penetration_m is not None and penetration_m > self.max_penetration_m:
                self.max_penetration_m, self.max_penetration_substep = penetration_m, substep
            if substep >= rule.t0:
                self._quiet = self._quiet + 1 if (penetration_m or 0.0) <= rule.threshold_m else 0
            reason = "resolved"
            if substep >= rule.t0 and self._quiet >= rule.n_resolved:
                end = True
            elif substep >= rule.t0 + rule.max_after:
                end, reason = True, "max_after"
        else:
            raise ValueError(f"unknown window mode {rule.mode}")
        if end:
            self.restored_values = adapter.restore_max_depenetration_velocity(self.originals)
            self.ended_after = substep
            self.end_reason = reason

    def report(self) -> dict:
        return {
            "window_start_substep": self.started_at,
            "window_last_substep": self.ended_after,
            "window_end_reason": self.end_reason,
            "capped_bodies": list(self.originals),
            "original_values": self.originals,
            "values_during_window": self.capped_values,
            "values_after_restore": self.restored_values,
            "restored_exactly": (self.ended_after is not None and self.restored_values == self.originals)
            if self.rule.mode != "persistent" else None,
            "window_max_penetration_m": self.max_penetration_m if self.rule.mode == "resolved" else None,
            "window_max_penetration_substep": self.max_penetration_substep,
        }
