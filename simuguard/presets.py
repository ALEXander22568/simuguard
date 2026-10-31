"""Default detector suite with configuration loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .core.detectors import (
    ActuationBoundConfig,
    ActuationBoundDetector,
    ContactEjectionDetector,
    Detector,
    EjectionConfig,
    ImpulseSpikeConfig,
    ImpulseSpikeDetector,
    NonFiniteConfig,
    NonFiniteStateDetector,
    PenetrationConfig,
    PenetrationDetector,
)
from .core.detectors.ejection import GateFn


def default_detectors(config: dict[str, Any] | None = None, *, ejection_gate: GateFn | None = None) -> list[Detector]:
    """Build the default suite; ``config`` keys: ejection, penetration, impulse_spike, actuation_bound, non_finite, disabled."""

    cfg = dict(config or {})
    disabled = set(cfg.get("disabled", []))
    suite: list[Detector] = []
    if "contact_ejection" not in disabled:
        suite.append(ContactEjectionDetector(EjectionConfig.from_dict(cfg.get("ejection")), gate=ejection_gate))
    if "non_finite_state" not in disabled:
        suite.append(NonFiniteStateDetector(NonFiniteConfig(**cfg.get("non_finite", {}))))
    if "deep_penetration" not in disabled:
        suite.append(PenetrationDetector(PenetrationConfig(**cfg.get("penetration", {}))))
    if "impulse_spike" not in disabled:
        suite.append(ImpulseSpikeDetector(ImpulseSpikeConfig(**cfg.get("impulse_spike", {}))))
    if "actuation_bound" not in disabled:
        suite.append(ActuationBoundDetector(ActuationBoundConfig(**cfg.get("actuation_bound", {}))))
    return suite


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))
