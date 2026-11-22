import argparse, sys, json, time
import cv2  # noqa
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--task", default="BananaInBowlTask")
p.add_argument("--num-envs", type=int, default=2)
AppLauncher.add_app_launcher_args(p)
a, _ = p.parse_known_args()
a.enable_cameras = True
a.headless = True
app = AppLauncher(a).app
import os, traceback
def main():

    import torch, numpy as np
    import robolab.constants
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
    from robolab.constants import DEFAULT_TASK_SUBFOLDERS
    auto_register_droid_envs(task_dirs=DEFAULT_TASK_SUBFOLDERS, task=[a.task])
    from robolab.core.environments.runtime import create_env
    env, env_cfg = create_env(a.task, device="cuda:0", seed=0, num_envs=a.num_envs, policy="pi05")
    print("PROBE env", type(env), "dt", env.physics_dt, "decimation", env.cfg.decimation)
    print("PROBE physx cfg", env.cfg.sim.physx)
    import isaaclab.sim as sim_utils
    from isaacsim.core.simulation_manager import SimulationManager
    sv = SimulationManager.get_physics_sim_view()
    print("PROBE simview", type(sv), [m for m in dir(sv) if not m.startswith("_")])
    for name, obj in env.scene.rigid_objects.items():
        v = obj.root_physx_view
        print("PROBE rigid", name, obj.cfg.prim_path, v.count, v.prim_paths[:3])
    for name, art in env.scene.articulations.items():
        v = art.root_physx_view
        print("PROBE art", name, v.count, v.max_dofs, v.max_links, art.joint_names, art.body_names)
        print("PROBE artview methods", [m for m in dir(v) if not m.startswith("_")])
    rv = list(env.scene.rigid_objects.values())[0].root_physx_view
    print("PROBE rbview type", type(rv), type(rv).__mro__)
    print("PROBE rbview methods", [m for m in dir(rv) if not m.startswith("_")])
    print("PROBE sensors", list(env.scene.sensors.keys())[:40])
    # multi-pattern view
    try:
        pats = [o.root_physx_view.prim_paths[0].replace("env_0", "env_*") for o in env.scene.rigid_objects.values()]
        mv = sv.create_rigid_body_view(pats)
        print("PROBE multiview", mv.count, mv.prim_paths)
    except Exception as e:
        print("PROBE multiview failed", repr(e))
    # count physics steps
    import omni.physx
    n = {"k": 0}
    sub = omni.physx.get_physx_interface().subscribe_physics_step_events(lambda dt: n.__setitem__("k", n["k"] + 1))
    calls = {"k": 0}
    orig = env.sim.step
    def wrapped(*args, **kw):
        calls["k"] += 1
        return orig(*args, **kw)
    env.sim.step = wrapped
    env.reset(); env.reset()
    print("PROBE after reset physx steps", n["k"], "sim.step calls", calls["k"])
    act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    print("PROBE action dim", env.action_manager.total_action_dim, env.action_manager.active_terms)
    t = time.time()
    for i in range(3):
        env.step(act)
    print("PROBE after 3 env.step physx steps", n["k"], "sim.step calls", calls["k"], "wall", time.time() - t)
    from omni.physx import get_physx_simulation_interface
    from pxr import PhysicsSchemaTools
    h, d = get_physx_simulation_interface().get_contact_report()
    print("PROBE contact headers", len(h), "points", len(d))
    seen = set()
    for x in h:
        pa = str(PhysicsSchemaTools.intToSdfPath(x.actor0)); pb = str(PhysicsSchemaTools.intToSdfPath(x.actor1))
        seen.add((pa, pb))
    for s in sorted(seen)[:30]:
        print("PROBE pair", s)
    if len(d):
        print("PROBE point", d[0])
    # depenetration attribute on objects
    import omni.usd
    from pxr import UsdPhysics, PhysxSchema
    stage = omni.usd.get_context().get_stage()
    for path in rv.prim_paths[:2] + list(env.scene.articulations.values())[0].root_physx_view.link_paths[0][:2]:
        prim = stage.GetPrimAtPath(path)
        api = PhysxSchema.PhysxRigidBodyAPI(prim)
        print("PROBE prim", path, prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI), api.GetMaxDepenetrationVelocityAttr().Get() if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI) else None, prim.HasAPI(PhysxSchema.PhysxContactReportAPI))
    scene_prim = stage.GetPrimAtPath("/physicsScene")
    print("PROBE scenes", [str(x.GetPath()) for x in stage.Traverse() if x.HasAPI(PhysxSchema.PhysxSceneAPI)][:3])
    for x in stage.Traverse():
        if x.HasAPI(PhysxSchema.PhysxSceneAPI):
            api = PhysxSchema.PhysxSceneAPI(x)
            print("PROBE sceneattrs", {at.GetName(): at.Get() for at in x.GetAttributes() if "physx" in at.GetName().lower()})
            break
try:
    main()
except Exception:
    traceback.print_exc()
sys.stdout.flush(); os._exit(0)
