"""Isaac Sim / Isaac Lab adapter (PhysX 5 tensor views + contact report) and RoboDojo glue.

Importing this package does not import ``omni`` / ``pxr``; those load when an adapter is built.
"""

from .adapter import ArticulationSpec, BodySpec, IsaacAdapter, enable_contact_reporting
from .robodojo import TASK_SPECS, RoboDojoAdapter, RoboDojoTaskSpec, install_contact_reporting, robodojo_specs

__all__ = [
    "ArticulationSpec",
    "BodySpec",
    "IsaacAdapter",
    "RoboDojoAdapter",
    "RoboDojoTaskSpec",
    "TASK_SPECS",
    "enable_contact_reporting",
    "install_contact_reporting",
    "robodojo_specs",
]
