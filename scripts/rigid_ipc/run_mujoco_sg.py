# -*- coding: utf-8 -*-
"""Rigid IPC unit tests in MuJoCo with SimuGuard attached (detector thresholds: configs/detectors_default.json, unchanged).
Scenes are rebuilt from the repository's MuJoCo MJCF files: same meshes (convex hulls, as MuJoCo uses), poses, zero friction.
usage: ../engine_stress/venv/bin/python run_mujoco_sg.py [dt] > results/mujoco_<dt>.json
"""
import json, os, sys, collections
import numpy as np, mujoco
from common import scenes, parse, Energy, gravity_ratio, freefall_excess
SG = os.path.expanduser(os.environ.get('SIMUGUARD_REPO', '~/simuguard')); sys.path.insert(0, SG)
from simuguard.adapters.mujoco import MujocoAdapter, TaskRoles
from simuguard.core.monitor import SubstepMonitor, MonitorConfig
from simuguard.presets import default_detectors
CFG = json.load(open(SG + '/configs/detectors_default.json'))
DT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.002; T = 5.0
out = []
for name, f in scenes():
    bodies = parse(f)
    xml = '<mujoco><option timestep="%g"><flag energy="enable"/></option><asset>' % DT
    for i, b in enumerate(bodies): xml += '<mesh name="m%d" file="%s" scale="%g %g %g"/>' % ((i, b['mesh']) + tuple(b['scale']))
    xml += '</asset><worldbody>'
    for i, b in enumerate(bodies):
        xml += '<body name="%s%d" pos="%g %g %g" quat="%g %g %g %g"><geom type="mesh" mesh="m%d" friction="%g"/>%s</body>' % (
            ('free' if b['free'] else 'fixed', i) + tuple(b['pos']) + tuple(b['quat']) + (i, b['friction'], '<freejoint/>' if b['free'] else ''))
    xml += '</worldbody></mujoco>'
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    free = [i + 1 for i, b in enumerate(bodies) if b['free']]
    nm = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
    ad = MujocoAdapter(m, d, roles=TaskRoles(targets={nm(b) for b in free}, containers={nm(i + 1) for i, b in enumerate(bodies) if not b['free']}, robot_prefixes=()), task_name=name)
    mon = SubstepMonitor(ad, default_detectors(CFG), episode_id=name, config=MonitorConfig(snapshot_interval_substeps=0, save_episode_logs=False)).attach()
    en = Energy(); vmax = pen = 0.0; note = ''
    def ke():
        r = []
        for b in free:
            v = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)  # [ang, lin] at body origin, world frame
            w = v[:3]; vc = v[3:] + np.cross(w, d.xipos[b] - d.xpos[b]); Rw = d.ximat[b].reshape(3, 3)
            r.append(0.5 * m.body_mass[b] * vc @ vc + 0.5 * np.sum(m.body_inertia[b] * (Rw.T @ w) ** 2))
        return r
    en.add(m.body_mass[free], d.xipos[free, 2], ke()); Z = [d.xipos[free, 2].copy()]; S = [np.linalg.norm(d.cvel[free, 3:], axis=1)]
    for _ in range(int(round(T / DT))):
        ad.step()
        if not np.all(np.isfinite(d.qpos)): note = 'diverged'; break
        k = ke(); en.add(m.body_mass[free], d.xipos[free, 2], k)
        S.append(np.linalg.norm(d.cvel[free, 3:], axis=1)); Z.append(d.xipos[free, 2].copy()); vmax = max(vmax, float(S[-1].max()))
        if d.ncon: pen = min(pen, float(d.contact.dist[:d.ncon].min()))
    mon.finalize()
    ev = collections.Counter((e.detector, e.status.value) for e in mon.events.values())
    Z, S = np.array(Z), np.array(S); conf = []; names = [nm(b) for b in free]
    for e in mon.events.values():
        if e.status.value != 'confirmed': continue
        dd = e.to_dict(); dd.pop('metrics', None)
        j = next((names.index(n) for n in names if any(str(bb).endswith(n) for bb in e.bodies)), None)
        if j is not None: dd['gravity_ratio'], dd['v_peak'], dd['drop_m'] = gravity_ratio(Z[:, j], S[:, j], min(e.onset_substep, len(Z) - 1), DT)
        conf.append(dd)
    ex = freefall_excess(Z, S)
    out.append(dict(scene=name, engine='mujoco %s' % mujoco.__version__, dt=DT, free=len(free), peak_speed=vmax, penetration_mm=-1000 * pen,
                    freefall_excess=ex[0], excess_time_s=ex[1] * DT, excess_speed=ex[3], energy_rise_J=en.rise, energy_rise_ratio=en.ratio(), note=note, monitor_errors=mon.error_count,
                    events={'%s:%s' % k: v for k, v in sorted(ev.items())},
                    confirmed=conf))
    print('%-26s excess %.2f peak %.2f pen %.1f mm  %s kept %s' % (name, ex[0], vmax, -1000 * pen, {k: v for k, v in ev.items() if k[0] == 'contact_ejection'}, [round(c.get('gravity_ratio', -1), 2) for c in conf]), file=sys.stderr)
json.dump(out, sys.stdout, indent=1)
