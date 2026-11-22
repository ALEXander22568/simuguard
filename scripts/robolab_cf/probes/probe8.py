"""probe8: are camera observations reproducible?  Build the task env, reset twice, step with a fixed
action, and save every image tensor in the observation at reset and after each step.  Run twice in
separate processes and compare the saved arrays (and two resets inside one process)."""
import argparse, sys, os, traceback
import cv2  # noqa
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--task", default="BananasInCrateTask")
p.add_argument("--num-envs", type=int, default=10)
p.add_argument("--steps", type=int, default=4)
p.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(p)
a, _ = p.parse_known_args()
a.enable_cameras = True
a.headless = True
app = AppLauncher(a).app

def images(obs, tag, out):
    import torch
    def walk(d, pre):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, pre + k + "/")
            elif isinstance(v, torch.Tensor) and v.dim() >= 4:
                out[f"{tag}|{pre}{k}"] = v.detach().cpu().numpy()
    walk(obs, "")

def main():
    import torch, numpy as np
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
    from robolab.constants import DEFAULT_TASK_SUBFOLDERS
    auto_register_droid_envs(task_dirs=DEFAULT_TASK_SUBFOLDERS, task=[a.task])
    from robolab.core.environments.runtime import create_env
    env, _ = create_env(a.task, device="cuda:0", seed=0, num_envs=a.num_envs, policy="pi05")
    out = {}
    for r in range(2):
        obs, _ = env.reset(seed=0)
        images(obs, f"reset{r}", out)
    act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    act[:] = env.scene.articulations["robot"].data.joint_pos[:, :act.shape[1]]
    for k in range(a.steps):
        obs, *_ = env.step(act)
        images(obs, f"step{k}", out)
    np.savez_compressed(a.out, **{k.replace("/", "~"): v for k, v in out.items()})
    for k, v in out.items():
        print("PROBE", k, v.shape, v.dtype, flush=True)
    keys = [k for k in out if k.startswith("reset0|")]
    for k in keys:
        k1 = k.replace("reset0|", "reset1|")
        d = np.abs(out[k].astype(np.float64) - out[k1].astype(np.float64))
        print("PROBE in-process reset0 vs reset1", k, "equal", bool((d == 0).all()), "max", float(d.max()), "frac_px_diff", float((d > 0).mean()), flush=True)

try:
    main()
except Exception:
    traceback.print_exc()
sys.stdout.flush(); os._exit(0)
