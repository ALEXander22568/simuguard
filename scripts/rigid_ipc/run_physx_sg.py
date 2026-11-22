# -*- coding: utf-8 -*-
"""Rigid IPC unit tests in PhysX (SAPIEN) with SimuGuard attached; same scenes, hulls, poses and zero friction as run_mujoco_sg.py.
usage: python run_physx_sg.py DT TAG > results/physx_TAG_DT.json     (set SIMUGUARD_PATCHED_SAPIEN to use the repaired library)
"""
import json, os, sys, collections, types
import numpy as np
SG = os.path.expanduser(os.environ.get('SIMUGUARD_REPO', '~/simuguard')); sys.path.insert(0, SG)
if os.environ.get('SIMUGUARD_PATCHED_SAPIEN'): sys.path.insert(0, os.environ['SIMUGUARD_PATCHED_SAPIEN'])
import sapien
from common import scenes, parse, Energy, gravity_ratio, freefall_excess
from simuguard.adapters.robotwin.adapter import RoboTwinAdapter
from simuguard.adapters.robotwin.tasks import TaskSpec
from simuguard.core.monitor import SubstepMonitor, MonitorConfig
from simuguard.presets import default_detectors
CFG = json.load(open(SG + '/configs/detectors_default.json'))
DT = float(sys.argv[1]); TAG = sys.argv[2]; T = 5.0
out = []
for name, f in scenes():
    if os.environ.get('ONLY') and name not in os.environ['ONLY'].split(','): continue
    bodies = parse(f)
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()]); scene.set_timestep(DT)
    env = types.SimpleNamespace(scene=scene, targets=[], containers=[], task_name=name, take_action_cnt=0)
    for i, b in enumerate(bodies):
        mat = sapien.physx.PhysxMaterial(b['friction'], b['friction'], 0.0)
        bl = scene.create_actor_builder()
        bl.add_convex_collision_from_file(b['mesh'], scale=b['scale'], material=mat, density=1000)
        bl.set_initial_pose(sapien.Pose(b['pos'], b['quat']))
        a = bl.build(name='free%d' % i) if b['free'] else bl.build_static(name='fixed%d' % i)
        (env.targets if b['free'] else env.containers).append(a)
        if b['free'] and os.environ.get('DEPEN'): a.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).set_max_depenetration_velocity(float(os.environ['DEPEN']))
    ad = RoboTwinAdapter(env, task_spec=TaskSpec(task_name=name, target_attrs=('targets',), container_attrs=('containers',)), task_name=name)
    mon = SubstepMonitor(ad, default_detectors(CFG), episode_id=name, config=MonitorConfig(snapshot_interval_substeps=0, save_episode_logs=False)).attach()
    comps = [a.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent) for a in env.targets]
    def state():
        m, z, ke, sp = [], [], [], []
        for c in comps:
            cm = c.entity.get_pose() * c.cmass_local_pose; v = np.asarray(c.get_linear_velocity()); w = np.asarray(c.get_angular_velocity())
            Rw = cm.to_transformation_matrix()[:3, :3]
            m.append(c.mass); z.append(cm.p[2]); sp.append(float(np.linalg.norm(v)))
            ke.append(0.5 * c.mass * v @ v + 0.5 * np.sum(np.asarray(c.inertia) * (Rw.T @ w) ** 2))
        return m, z, ke, sp
    en = Energy(); vmax = 0.0; pen = 0.0; note = ''; Z, S = [], []
    m, z, ke, sp = state(); en.add(m, z, ke); Z.append(z); S.append(sp)
    for _ in range(int(round(T / DT))):
        env.scene.step()
        m, z, ke, sp = state()
        if not np.all(np.isfinite(z)): note = 'diverged'; break
        en.add(m, z, ke); Z.append(z); S.append(sp); vmax = max(vmax, max(sp))
        for c in scene.get_contacts():
            for p in c.points: pen = min(pen, float(p.separation))
    mon.finalize(); Z, S = np.array(Z), np.array(S)
    ev = collections.Counter((e.detector, e.status.value) for e in mon.events.values())
    conf = []; names = [x.get_name() for x in env.targets]
    for e in mon.events.values():
        if e.status.value != 'confirmed': continue
        d = e.to_dict(); d.pop('metrics', None)
        j = next((names.index(n) for n in names if any(str(bb).endswith(n) for bb in e.bodies)), None)
        if j is not None: d['gravity_ratio'], d['v_peak'], d['drop_m'] = gravity_ratio(Z[:, j], S[:, j], min(e.onset_substep, len(Z) - 1), DT)
        conf.append(d)
    ex = freefall_excess(Z, S)
    out.append(dict(scene=name, engine='physx %s (sapien %s)' % (TAG, sapien.__version__), dt=DT, free=len(comps), peak_speed=vmax, penetration_mm=-1000 * pen,
                    freefall_excess=ex[0], excess_time_s=ex[1] * DT, excess_speed=ex[3], energy_rise_J=en.rise, energy_rise_ratio=en.ratio(), note=note, monitor_errors=mon.error_count,
                    events={'%s:%s' % k: v for k, v in sorted(ev.items())}, confirmed=conf))
    print('%-26s excess %.2f E rise %10.3f J (%.1f%%) peak %.2f pen %.1f mm  %s  kept %s' % (name, ex[0], en.rise, 100 * en.ratio(), vmax, -1000 * pen, {k: v for k, v in ev.items() if k[0] == 'contact_ejection'}, [round(c.get('gravity_ratio', -1), 2) for c in conf]), file=sys.stderr)
json.dump(out, sys.stdout, indent=1)
