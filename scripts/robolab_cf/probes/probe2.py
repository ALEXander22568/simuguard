import argparse, sys, os, traceback, time
import cv2  # noqa
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--task", default="BananaInBowlTask")
p.add_argument("--num-envs", type=int, default=2)
p.add_argument("--prebuild-cap", type=float, default=None)
AppLauncher.add_app_launcher_args(p)
a, _ = p.parse_known_args()
a.enable_cameras = True
a.headless = True
app = AppLauncher(a).app

def main():
    import torch, numpy as np
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
    from robolab.constants import DEFAULT_TASK_SUBFOLDERS
    auto_register_droid_envs(task_dirs=DEFAULT_TASK_SUBFOLDERS, task=[a.task])
    from robolab.core.environments.runtime import create_env
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics
    if a.prebuild_cap is not None:
        import isaaclab.scene as scn
        orig_init = scn.InteractiveScene.__init__
        def init(self, *args, **kw):
            orig_init(self, *args, **kw)
            stage = omni.usd.get_context().get_stage()
            n = 0
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if path.startswith("/World/envs/env_0/scene/banana") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateMaxDepenetrationVelocityAttr(a.prebuild_cap); n += 1
            print("PROBE prebuild cap applied to", n)
        scn.InteractiveScene.__init__ = init
    env, env_cfg = create_env(a.task, device="cuda:0", seed=0, num_envs=a.num_envs, policy="pi05")
    from isaacsim.core.simulation_manager import SimulationManager
    sv = SimulationManager.get_physics_sim_view()
    print("PROBE same simview", sv is env.sim.physics_sim_view)
    stage = omni.usd.get_context().get_stage()
    objs = env.scene.rigid_objects
    for name, o in objs.items():
        prim = stage.GetPrimAtPath(o.root_physx_view.prim_paths[0])
        print("PROBE obj", name, "kin", prim.GetAttribute("physics:kinematicEnabled").Get(), "rb", prim.GetAttribute("physics:rigidBodyEnabled").Get(),
              "depen", prim.GetAttribute("physxRigidBody:maxDepenetrationVelocity").Get(), "mass", o.root_physx_view.get_masses()[0].tolist())
    # patching instance method
    art = env.scene.articulations["robot"]
    v = art.root_physx_view
    log = []
    orig = v.set_dof_position_targets
    def w(data, indices):
        log.append((tuple(data.shape), tuple(indices.shape)))
        return orig(data, indices)
    v.set_dof_position_targets = w
    env.reset(); env.reset()
    act = torch.zeros(env.num_envs, 8, device=env.device)
    env.step(act)
    print("PROBE patched calls", len(log), log[:2])
    # contact view with list sensor patterns
    names = [n for n in objs]
    pats = [objs[n].root_physx_view.prim_paths[0].replace("env_0", "env_*") for n in names]
    links = art.body_names
    filt = pats + [f"/World/envs/env_*/robot/{l}" for l in links]
    try:
        cv = sv.create_rigid_contact_view(pats, filter_patterns=filt, max_contact_data_count=4096)
        print("PROBE cv", cv.sensor_count, cv.filter_count, cv.sensor_paths[:4] if hasattr(cv, "sensor_paths") else None)
        print("PROBE cv attrs", [m for m in dir(cv) if not m.startswith("_")])
    except Exception as e:
        traceback.print_exc(); cv = None
    env.step(act)
    if cv is not None:
        f, pt, nm, sep, cnt, st = cv.get_contact_data(env.physics_dt)
        cnt = cnt.cpu().numpy() if hasattr(cnt, "cpu") else cnt.numpy()
        print("PROBE counts\n", cnt)
        st = st.cpu().numpy() if hasattr(st, "cpu") else st.numpy()
        pt = pt.cpu().numpy() if hasattr(pt, "cpu") else pt.numpy()
        f = f.cpu().numpy() if hasattr(f, "cpu") else f.numpy()
        sep = sep.cpu().numpy() if hasattr(sep, "cpu") else sep.numpy()
        nm = nm.cpu().numpy() if hasattr(nm, "cpu") else nm.numpy()
        for s in range(cnt.shape[0]):
            for j in range(cnt.shape[1]):
                if cnt[s, j]:
                    i0 = st[s, j]
                    print("PROBE pair", s, j, cnt[s, j], "pt", pt[i0].round(3), "f", f[i0], "n", nm[i0].round(3), "sep", sep[i0])
        fm = cv.get_contact_force_matrix(env.physics_dt)
        print("PROBE fm shape", tuple(fm.shape))
        print("PROBE env origins", env.scene.env_origins.cpu().numpy().round(2).tolist())
    # depenetration test: teleport banana into table in both envs
    ban = objs["banana"]
    def peak_after_teleport(dz, steps=30):
        pose = ban.data.root_state_w[:, :7].clone()
        pose[:, 2] -= dz
        ban.write_root_pose_to_sim(pose)
        ban.write_root_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))
        vmax = np.zeros(env.num_envs)
        for _ in range(steps):
            env.sim.step(render=False)
            vel = ban.root_physx_view.get_velocities().cpu().numpy()
            vmax = np.maximum(vmax, np.linalg.norm(vel[:, :3], axis=1))
        ban.update(env.physics_dt)
        return vmax
    for _ in range(60):
        env.sim.step(render=False)
    ban.update(env.physics_dt)
    print("PROBE base", peak_after_teleport(0.03))
    for _ in range(60):
        env.sim.step(render=False)
    prim = stage.GetPrimAtPath("/World/envs/env_0/scene/banana")
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateMaxDepenetrationVelocityAttr(0.2)
    print("PROBE runtime usd set env0 0.2 ->", prim.GetAttribute("physxRigidBody:maxDepenetrationVelocity").Get())
    print("PROBE after runtime", peak_after_teleport(0.03))
    for _ in range(60):
        env.sim.step(render=False)
    print("PROBE after runtime again", peak_after_teleport(0.03))

try:
    main()
except Exception:
    traceback.print_exc()
sys.stdout.flush(); os._exit(0)
