"""DeepSpeed mp_rank_00_model_states.pt -> flat model_states.pt that XPolicyLab Xiaomi_Robotics_1 helper() loads.

helper() reads <model_dir>/config.py and <model_dir>/model_states.pt and keeps the keys that start
with "model.".  The released RoboDojo checkpoint keeps them under sd["module"], with the frozen
token embedding also stored in sd["frozen_param_fragments"] and lm_head tied to it.
"""
import time
import torch

t = time.time()
sd = torch.load("last.ckpt/checkpoint/mp_rank_00_model_states.pt", map_location="cpu", mmap=True, weights_only=False)
flat = dict(sd["module"])
added = []
for key, value in sd.get("frozen_param_fragments", {}).items():
    if key not in flat:
        flat[key] = value
        added.append(key)
for key, source in sd.get("shared_params", {}).items():
    if key not in flat and source in flat:
        flat[key] = flat[source]
        added.append(f"{key} (tied to {source})")
print("keys", len(flat), "added", added)
torch.save(flat, "model_states.pt.tmp")
import os
os.replace("model_states.pt.tmp", "model_states.pt")
print("saved in", round(time.time() - t, 1), "s")
