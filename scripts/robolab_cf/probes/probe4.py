import argparse, sys, os, traceback, time
import cv2  # noqa
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--task", default="BananaInBowlTask")
p.add_argument("--num-envs", type=int, default=4)
p.add_argument("--prebuild-cap", type=float, default=None)
p.add_argument("--dz", type=float, default=0.012)
p.add_argument("--via", default="usd")
AppLauncher.add_app_launcher_args(p)
a, _ = p.parse_known_args()
a.enable_cameras = True
a.headless = True
app = AppLauncher(a).app
CAPPED = ("env_2", "env_3")

def main():
    import torch, numpy as np
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
    from robolab.constants import DEFAULT_TASK_SUBFOLDERS
    auto_register_droid_envs(task_dirs=DEFAULT_TASK_SUBFOLDERS, task=[a.task])
    from robolab.core.environments.runtime import create_env
    import omni.usd
    from pxr import PhysxSchema
    def set_cap(value):
        stage = omni.usd.get_context().get_stage()
        if a.via == "usdrt":
            import usdrt
            from omni.usd import get_context
            rt = usdrt.Usd.Stage.Attach(get_context().get_stage_id())
            for e in CAPPED:
                prim = rt.GetPrimAtPath(f"/World/envs/{e}/scene/banana")
                attr = prim.GetAttribute("physxRigidBody:maxDepenetrationVelocity")
                ok = attr.Set(float(value)) if attr and attr.IsValid() else None
                print("PROBE usdrt set", e, value, bool(attr and attr.IsValid()), ok, attr.Get() if attr else None, flush=True)
            return
        for e in CAPPED:
            prim = stage.GetPrimAtPath(f"/World/envs/{e}/scene/banana")
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateMaxDepenetrationVelocityAttr(value)
    if a.prebuild_cap is not None:
        import isaaclab.scene as scn
        orig_init = scn.InteractiveScene.__init__
        def init(self, *args, **kw):
            orig_init(self, *args, **kw)
            set_cap(a.prebuild_cap)
        scn.InteractiveScene.__init__ = init
    env, env_cfg = create_env(a.task, device="cuda:0", seed=0, num_envs=a.num_envs, policy="pi05")
    env.reset(); env.reset()
    ban = env.scene.rigid_objects["banana"]
    rv = ban.root_physx_view
    idx = torch.arange(env.num_envs, dtype=torch.int32, device=env.device)
    base = ban.data.default_root_state.clone()
    base[:, :3] += env.scene.env_origins
    def settle(n=120):
        for _ in range(n):
            env.sim.step(render=False)
    st = omni.usd.get_context().get_stage()
    pr = st.GetPrimAtPath("/World/envs/env_0/scene/banana")
    print("PROBE attr", pr.GetAttribute("physxRigidBody:maxDepenetrationVelocity").Get(), [str(x) for x in pr.GetAppliedSchemas()], flush=True)
    import omni.physx
    try:
        print("PROBE change_tracking_paused", omni.physx.get_physx_simulation_interface().is_change_tracking_paused(), flush=True)
    except Exception as exc:
        print("PROBE change_tracking_paused err", exc, flush=True)
    settle(240)
    rest = rv.get_transforms().clone()
    print("PROBE rest z", rest[:, 2].cpu().numpy().round(4).tolist(), "default z", base[:, 2].cpu().numpy().round(4).tolist(), flush=True)
    def trial(label, steps=40):
        q = rest.clone(); q[:, 2] -= a.dz
        pose = q
        rv.set_transforms(q, idx)
        rv.set_velocities(torch.zeros(env.num_envs, 6, device=env.device), idx)
        z0 = pose[:, 2].cpu().numpy()
        vmax = np.zeros(env.num_envs); vz = []; rec = {}
        for k in range(steps):
            env.sim.step(render=False)
            if k + 1 in (1, 2, 4, 8, 16):
                rec[k + 1] = np.round((rv.get_transforms().cpu().numpy()[:, 2] - z0) * 1000, 2).tolist()
            vel = rv.get_velocities().cpu().numpy()
            vmax = np.maximum(vmax, np.linalg.norm(vel[:, :3], axis=1))
            vz.append(vel[:, 2])
        z = rv.get_transforms().cpu().numpy()[:, 2]
        print(f"PROBE {label}: peak|v| {np.round(vmax,3).tolist()} peak vz {np.round(np.max(vz,0),3).tolist()} dz_after {np.round(z - z0,4).tolist()} rise_mm {rec}", flush=True)
    settle(); trial("A_default")
    settle(); set_cap(0.1); trial("B_runtime_cap0.1_env2,3")
    settle(); trial("B2_runtime_cap_again")
    settle(); set_cap(3.0); trial("C_restored3.0")
    settle(); set_cap(0.1); env.sim.step(render=False); trial("D_cap_then_one_step")

try:
    main()
except Exception:
    traceback.print_exc()
sys.stdout.flush(); os._exit(0)
