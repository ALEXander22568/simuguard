#!/usr/bin/env python3
"""Write the config.py that XPolicyLab's Xiaomi_Robotics_1 helper() can load, from the released one.

The RoboDojo XR-1 checkpoint (ModelScope RoboDojo-Benchmark/RoboDojo, ckpt/RoboDojo/Xiaomi_Robotics_1/
RoboDojo-sim-arx_x5-ee-0) ships its training config.  Two of its model keys, ``stop_gradient_to_vlm``
and ``training_repeat``, are training-only and the vendored inference class ``XR1.__init__`` rejects
them; the backbone is loaded with ``from_pretrained("Qwen/Qwen3-VL-4B-Instruct")``, which needs a local
path on hosts without Hugging Face access.  Nothing else changes: the XR-1 weights themselves come from
model_states.pt (convert_ds_checkpoint.py) and override the backbone.

Usage (in the checkpoint directory)::

    python3 make_inference_config.py --backbone /path/to/Qwen3-VL-4B-Instruct
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--backbone", required=True, help="local Qwen3-VL-4B-Instruct folder (weights + processor files)")
    args = ap.parse_args()
    d = Path(args.dir)
    original = d / "config.orig.py"
    if not original.exists():
        shutil.copy2(d / "config.py", original)
    text = original.read_text()
    for key in ("stop_gradient_to_vlm", "training_repeat"):
        text, n = re.subn(rf"^(\s*){key}\s*=.*\n", "", text, flags=re.M)
        print(key, "removed", n)
    text = text.replace("'Qwen/Qwen3-VL-4B-Instruct'", repr(args.backbone)).replace('"Qwen/Qwen3-VL-4B-Instruct"', repr(args.backbone))
    header = ("# Inference copy of config.orig.py (released with the checkpoint): training-only keys\n"
              "# stop_gradient_to_vlm / training_repeat removed, backbone path made local.\n")
    (d / "config.py").write_text(header + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
