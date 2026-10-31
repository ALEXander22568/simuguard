#!/usr/bin/env python3
"""Run upstream ``XPolicyLab/setup_policy_server.py`` with LingBot EE-action support.

The bridge is started exactly as upstream documents it; this shim only imports
``XPolicyLab.policy.LingBot_VA.model`` first and installs the 16-dim
end-effector action path (see :mod:`simuguard.integrations.lingbot_ee_actions`)
before handing control to the upstream ``__main__``.

Environment:
    SIMUGUARD_XPOLICYLAB_SERVER  path to upstream XPolicyLab/setup_policy_server.py (required)

All other arguments are forwarded to upstream unchanged.
"""

from __future__ import annotations

import importlib
import os
import runpy
import sys
from pathlib import Path

from simuguard.integrations import lingbot_ee_actions


def main() -> None:
    entry = os.environ.get("SIMUGUARD_XPOLICYLAB_SERVER")
    if not entry:
        raise SystemExit("SIMUGUARD_XPOLICYLAB_SERVER must point to XPolicyLab/setup_policy_server.py")
    entry_path = Path(entry).resolve()
    xpl_root = entry_path.parent
    for path in (str(xpl_root.parent), str(xpl_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    model_module = importlib.import_module("XPolicyLab.policy.LingBot_VA.model")
    result = lingbot_ee_actions.install(model_module)
    print(f"[simuguard] LingBot EE action path installed: {result}", flush=True)

    runpy.run_path(str(entry_path), run_name="__main__")


if __name__ == "__main__":
    main()
