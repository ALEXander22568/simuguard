"""Which write path of physxRigidBody:maxDepenetrationVelocity reaches GPU PhysX (fabric on)?
8 envs: 0,1 control; 2,3 runtime USD; 4,5 runtime usdrt (fabric); 6,7 USD before the first sim reset."""
import argparse, sys, os, traceback
import cv2  # noqa
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--task", default="BananaInBowlTask")
p.add_argument("--dz", type=float, default=0.012)
p.add_argument("--cap", type=float, default=0.1)
AppLauncher.add_app_launcher_args(p)
a, _ = p.parse_known_args()
a.enable_cameras = True
a.headless = True
app = AppLauncher(a).app
ATTR = "physxRigidBody:maxDepenetrationVelocity"
GROUPS = {"usd": (2, 3), "usdrt": (4, 5), "prebuild": (6, 7)}

def main():
    import torch, numpy as np
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
    from robolab.constants import DEFAULT_TASK_SUBFOLDERS
    auto_register_droid_envs(task_dirs=DEFAULT_TASK_SUBFOLDERS, task=[a.task])
    from robolab.core.environments.runtime import create_env
    import omni.usd, usdrt
    from pxr import PhysxSchema
    import isaaclab.sim as isim

    def path(e):
        return f"/World/envs/env_{e}/scene/banana"

    def set_usd(envs, v):
        st = omni.usd.get_context().get_stage()
        for e in envs:
            PhysxSchema.PhysxRigidBodyAPI.Apply(st.GetPrimAtPath(path(e))).CreateMaxDepenetrationVelocityAttr().Set(float(v))

    def set_usdrt(envs, v):
        rt = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
        for e in envs:
            attr = rt.GetPrimAtPath(path(e)).GetAttribute(ATTR)
            ok = attr.Set(float(v)) if attr and attr.IsValid() else "no-attr"
            print("PROBE usdrt set", e, v, ok, flush=True)

    def show(tag):
        st = omni.usd.get_context().get_stage()
        rt = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
        usd_v = [st.GetPrimAtPath(path(e)).GetAttribute(ATTR).Get() for e in range(8)]
        rt_v = []
        for e in range(8):
            attr = rt.GetPrimAtPath(path(e)).GetAttribute(ATTR)
            rt_v.append(attr.Get() if attr and attr.IsValid() else None)
        print(f"PROBE attrs[{tag}] usd={usd_v} usdrt={rt_v}", flush=True)

    orig_reset = isim.SimulationContext.reset
    done = {"v": False}
    def reset(self, *args, **kw):
        if not done["v"]:
            set_usd(GROUPS["prebuild"], a.cap)
            done["v"] = True
            print("PROBE prebuild applied", flush=True)
        return orig_reset(self, *args, **kw)
    isim.SimulationContext.reset = reset

    env, _ = create_env(a.task, device="cuda:0", seed=0, num_envs=8, policy="pi05")
    env.reset(); env.reset()
    import carb
    print("PROBE fabric", env.cfg.sim.use_fabric, carb.settings.get_settings().get("/physics/fabricEnabled"), flush=True)
    rv = env.scene.rigid_objects["banana"].root_physx_view
    idx = torch.arange(8, dtype=torch.int32, device=env.device)
    def settle(n=120):
        for _ in range(n):
            env.sim.step(render=False)
    settle(240)
    rest = rv.get_transforms().clone()
    def trial(label, steps=24):
        q = rest.clone(); q[:, 2] -= a.dz
        rv.set_transforms(q, idx)
        rv.set_velocities(torch.zeros(8, 6, device=env.device), idx)
        z0 = q[:, 2].cpu().numpy(); rec = {}; vmax = np.zeros(8)
        for k in range(steps):
            env.sim.step(render=False)
            if k + 1 in (1, 2, 4, 8):
                rec[k + 1] = np.round((rv.get_transforms().cpu().numpy()[:, 2] - z0) * 1000, 2).tolist()
            vmax = np.maximum(vmax, np.linalg.norm(rv.get_velocities().cpu().numpy()[:, :3], axis=1))
        print(f"PROBE {label}: rise_mm(step:[env0..7]) {rec} peak|v| {np.round(vmax, 3).tolist()}", flush=True)
    show("after build")
    settle(); trial("A_prebuild_only")
    set_usd(GROUPS["usd"], a.cap); set_usdrt(GROUPS["usdrt"], a.cap); show("runtime cap")
    settle(); trial("B_runtime_cap(usd 2,3; usdrt 4,5)")
    set_usd(GROUPS["prebuild"][:1], 3.0); set_usdrt(GROUPS["prebuild"][1:], 3.0); show("runtime restore prebuilt")
    settle(); trial("C_restore_prebuilt(usd 6; usdrt 7)")
    set_usd(GROUPS["prebuild"][:1], a.cap); set_usdrt(GROUPS["prebuild"][1:], a.cap); show("runtime re-cap")
    settle(); trial("D_recap(usd 6; usdrt 7)")

try:
    main()
except Exception:
    traceback.print_exc()
sys.stdout.flush(); os._exit(0)
