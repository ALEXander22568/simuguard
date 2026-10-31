"""SimuGuard: detect, verify and analyze contact artifacts in simulation-based policy evaluation."""

__version__ = "0.1.0.dev0"

from .core import *  # noqa: F401,F403
from .core import __all__ as _core_all
from .presets import default_detectors

__all__ = ["__version__", "default_detectors", *_core_all]
