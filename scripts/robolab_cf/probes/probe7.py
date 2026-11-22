"""probe7: runtime depenetration-cap write paths on GPU PhysX that probe6 did not try:
USD/usdrt writes followed by IPhysxSimulation.flush_changes(), usdrt with fabric change tracking unpaused,
and Isaac Lab with use_fabric=False.  Same trial as probe6:
Trial: push the banana 12 mm into its support, zero velocity, step; rise in step 1 is ~12 mm
uncapped (3.0 m/s allows 25 mm/step) and <= 0.83 mm with a 0.1 m/s cap."""
import argparse, sys, os, traceback
import cv2  # noqa
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--task", default="BananaInBowlTask")
p.add_argument("--body", default="banana")
p.add_argument("--num-envs", type=int, default=8)
p.add_argument("--dz", type=float, default=0.012)
p.add_argument("--cap", type=float, default=0.1)
p.add_argument("--no-replicate", action="store_true")
p.add_argument("--pre", default="")  # envs capped in USD before the first sim reset ("all" or "0,6,7")
p.add_argument("--rt", action="append", default=[])  # "usd:2,3=0.1" / "usdrt:4,5=0.1"  (runtime writes, trial B)
p.add_argument("--rt2", action="append", default=[])  # runtime writes before trial C
p.add_argument("--flush", action="store_true")  # flush_changes() after runtime writes
p.add_argument("--unpause", action="store_true")  # pause_change_tracking(False) before runtime writes
p.add_argument("--no-fabric", action="store_true")  # env cfg sim.use_fabric = False
AppLauncher.add_app_launcher_args(p)
a, _ = p.parse_known_args()
a.enable_cameras = True
a.headless = True
app = AppLauncher(a).app
ATTR = "physxRigidBody:maxDepenetrationVelocity"

def envs_of(s, n):
    return list(range(n)) if s == "all" else [int(x) for x in s.split(",") if x != ""]

def main():
    import torch, numpy as np
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
    from robolab.constants import DEFAULT_TASK_SUBFOLDERS
    auto_register_droid_envs(task_dirs=DEFAULT_TASK_SUBFOLDERS, task=[a.task])
    from robolab.core.environments.runtime import create_env
    import omni.usd, usdrt
    from pxr import PhysxSchema
    import isaaclab.sim as isim
    import isaaclab.scene as iscene
    n = a.num_envs

    def path(e):
        return f"/World/envs/env_{e}/scene/{a.body}"

    def set_usd(envs, v):
        st = omni.usd.get_context().get_stage()
        for e in envs:
            PhysxSchema.PhysxRigidBodyAPI.Apply(st.GetPrimAtPath(path(e))).CreateMaxDepenetrationVelocityAttr().Set(float(v))

    def set_usdrt(envs, v):
        rt = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
        for e in envs:
            prim = rt.GetPrimAtPath(path(e))
            attr = prim.GetAttribute(ATTR)
            if not (attr and attr.IsValid()):
                attr = prim.CreateAttribute(ATTR, usdrt.Sdf.ValueTypeNames.Float, True)
            attr.Set(float(v))

    import omni.physx
    simi = omni.physx.get_physx_simulation_interface()

    def apply(specs):
        print("PROBE change_tracking_paused", simi.is_change_tracking_paused(), flush=True)
        if a.unpause:
            simi.pause_change_tracking(False)
            print("PROBE unpaused ->", simi.is_change_tracking_paused(), flush=True)
        for spec in specs:
            via, rest = spec.split(":")
            envs, v = rest.split("=")
            (set_usd if via == "usd" else set_usdrt)(envs_of(envs, n), float(v))
            print("PROBE write", via, envs, v, flush=True)
        if a.flush:
            simi.flush_changes()
            print("PROBE flushed", flush=True)

    if a.no_replicate:
        orig_init = iscene.InteractiveScene.__init__
        def init(self, cfg, *args, **kw):
            cfg.replicate_physics = False
            return orig_init(self, cfg, *args, **kw)
        iscene.InteractiveScene.__init__ = init
    if a.no_fabric:
        import isaaclab.envs as ienvs
        orig_env_init = ienvs.ManagerBasedRLEnv.__init__
        def env_init(self, cfg, *args, **kw):
            cfg.sim.use_fabric = False
            return orig_env_init(self, cfg, *args, **kw)
        ienvs.ManagerBasedRLEnv.__init__ = env_init
    orig_reset = isim.SimulationContext.reset
    done = {"v": False}
    def reset(self, *args, **kw):
        if not done["v"] and a.pre:
            set_usd(envs_of(a.pre, n), a.cap)
            print("PROBE prebuild", a.pre, a.cap, flush=True)
        done["v"] = True
        return orig_reset(self, *args, **kw)
    isim.SimulationContext.reset = reset

    env, _ = create_env(a.task, device="cuda:0", seed=0, num_envs=n, policy="pi05")
    env.reset(); env.reset()
    print("PROBE replicate_physics", env.scene.cfg.replicate_physics, "fabric", env.cfg.sim.use_fabric, flush=True)
    rv = env.scene.rigid_objects[a.body].root_physx_view
    idx = torch.arange(n, dtype=torch.int32, device=env.device)
    def settle(k=120):
        for _ in range(k):
            env.sim.step(render=False)
    settle(240)
    rest = rv.get_transforms().clone()
    def trial(label, steps=16):
        q = rest.clone(); q[:, 2] -= a.dz
        rv.set_transforms(q, idx)
        rv.set_velocities(torch.zeros(n, 6, device=env.device), idx)
        z0 = q[:, 2].cpu().numpy(); rec = {}
        for k in range(steps):
            env.sim.step(render=False)
            if k + 1 in (1, 2, 4, 16):
                rec[k + 1] = np.round((rv.get_transforms().cpu().numpy()[:, 2] - z0) * 1000, 2).tolist()
        print(f"PROBE {label} rise_mm {rec}", flush=True)
    settle(); trial("A_after_build")
    if a.rt:
        apply(a.rt); settle(); trial("B_after_rt")
    if a.rt2:
        apply(a.rt2); settle(); trial("C_after_rt2")

try:
    main()
except Exception:
    traceback.print_exc()
sys.stdout.flush(); os._exit(0)
