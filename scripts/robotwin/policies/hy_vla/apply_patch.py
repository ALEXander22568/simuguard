"""Add `execute_first_action` to XPolicyLab/policy/Hy_Embodied_05_VLA/model.py (idempotent).

The upstream RoboTwin wrapper (robotwin_eval/policy_wrapper.py) executes chunk slots 0..exc-1; the
XPolicyLab adapter, written for the RoboDojo checkpoint, skips slot 0.  With the flag set, the adapter
executes the same slots as the upstream RoboTwin evaluation.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1])
src = path.read_text()
if "execute_first_action" in src:
    print("already patched"); raise SystemExit(0)
old_cfg = '        self.exc_action_interval = int(model_cfg.get("exc_action_interval", 1))\n'
new_cfg = old_cfg + '        self.execute_first_action = bool(model_cfg.get("execute_first_action", False))\n'
old_slice = """        if self.exc_action_interval > 1:
            needed = self.exc_action_size * self.exc_action_interval
            actions_wxyz = actions_wxyz[1 : needed + 1 : self.exc_action_interval]
        else:
            actions_wxyz = actions_wxyz[1 : self.exc_action_size + 1]
"""
new_slice = """        start = 0 if self.execute_first_action else 1
        if self.exc_action_interval > 1:
            needed = self.exc_action_size * self.exc_action_interval
            actions_wxyz = actions_wxyz[start : needed + start : self.exc_action_interval]
        else:
            actions_wxyz = actions_wxyz[start : self.exc_action_size + start]
"""
assert src.count(old_cfg) == 1 and src.count(old_slice) == 1, "model.py changed upstream; review the patch"
path.write_text(src.replace(old_cfg, new_cfg).replace(old_slice, new_slice))
print("patched", path)
