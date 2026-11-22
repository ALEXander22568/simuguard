# -*- coding: utf-8 -*-
"""The 13 unit tests as simulated by the Rigid IPC reference solver itself (run_ripc.sh, its own fixtures and 10 ms step),
read back from sim.json: free-fall check and the SimuGuard detectors (unchanged thresholds) on the recorded trajectory.
Rigid IPC stores body states and the smallest distance in the scene, not per-pair contacts. We give the detector a
contact between every pair of bodies whenever that distance is below 5 mm; this can only add flags.
usage: ../engine_stress/venv/bin/python run_ripc_sg.py > results/rigid_ipc_0.01.json
"""
import json, os, sys, collections, itertools
import numpy as np
from common import gravity_ratio, freefall_excess
SG = os.path.expanduser(os.environ.get('SIMUGUARD_REPO', '~/simuguard')); sys.path.insert(0, SG)
from simuguard.core.adapter import SimAdapter, AdapterCapabilities, HookHandle
from simuguard.core.types import BodyInfo, BodyKind, BodyRole, BodyState, ContactPair, ContactPoint
from simuguard.core.monitor import SubstepMonitor, MonitorConfig
from simuguard.presets import default_detectors
CFG = json.load(open(SG + '/configs/detectors_default.json'))
NAMES = ['5-cubes', 'edge-edge-parallel', 'edge-edge', 'erleben/cliff-edges', 'erleben/internal-edges', 'erleben/spike-and-wedge', 'erleben/spikes',
         'erleben/wedges', 'face-vertex', 'large-mass-ratio', 'tet-corner', 'vertex-face', 'vertex-vertex']
up = lambda v: np.array([v[0], -v[2], v[1]])  # Rigid IPC is y-up; SimuGuard and the gravity filter are z-up

class Traj(SimAdapter):
    name = 'trajectory'
    def __init__(self, seq, fixed, dt, tol=0.005):
        self.seq, self.fixed, self.dt, self.tol, self.k, self.cb = seq, fixed, dt, tol, 0, None
        self.ids = ['body:%s%d' % ('fixed' if f else 'free', i) for i, f in enumerate(fixed)]
    def capabilities(self): return AdapterCapabilities(substep_hook=True, contacts=True, public_state_snapshot=False, native_state_snapshot=False, control_capture=False, notes=['recorded trajectory'])
    def timestep(self): return self.dt
    def bodies(self, refresh=False):
        return {b: BodyInfo(body_id=b, name=b.split(':')[1], kind=BodyKind.STATIC if f else BodyKind.DYNAMIC, role=BodyRole.CONTAINER if f else BodyRole.TARGET,
                            mass=None if f else 1.0, articulation=None, collision_shape_count=1) for b, f in zip(self.ids, self.fixed)}
    def read_states(self, body_ids=None):
        out = {}
        for b, s in zip(self.ids, self.seq[self.k]['rigid_bodies']):
            if body_ids is None or b in body_ids:
                out[b] = BodyState(body_id=b, position=up(s['position']), quaternion=np.array([1.0, 0, 0, 0]), linear_velocity=up(s['linear_velocity']), angular_velocity=up(s['angular_velocity']))
        return out
    def read_contacts(self):
        d = self.seq[self.k].get('min_distance')
        if d is None or d > self.tol: return []
        pt = lambda: [ContactPoint(position=np.zeros(3), normal=np.array([0.0, 0, 1]), impulse=np.zeros(3), separation=float(d))]
        return [ContactPair(a, b, pt()) for (a, fa), (b, fb) in itertools.combinations(zip(self.ids, self.fixed), 2) if not (fa and fb)]
    def control_step(self): return self.k
    def install_substep_hook(self, after, before=None):
        self.cb = after; return HookHandle(lambda: None, "trajectory")
    def capture_control(self): raise NotImplementedError
    def apply_control(self, control): raise NotImplementedError
    def capture_snapshot(self, substep, *, include_native=True): raise NotImplementedError
    def restore_snapshot(self, snapshot, *, method='auto'): raise NotImplementedError
    def step_physics(self): self.k += 1

out = []
for name in NAMES:
    d = json.load(open('ripc_out/%s/sim.json' % name.replace('/', '_')))
    seq = d['animation']['state_sequence']; dt = d['args']['timestep']
    fixed = [bool(b.get('is_dof_fixed', False)) for b in d['args']['rigid_body_problem']['rigid_bodies']]
    assert len(fixed) == len(seq[0]['rigid_bodies'])
    free = [i for i, f in enumerate(fixed) if not f]
    Z = np.array([[s['rigid_bodies'][i]['position'][1] for i in free] for s in seq]); S = np.array([[np.linalg.norm(s['rigid_bodies'][i]['linear_velocity']) for i in free] for s in seq])
    ad = Traj(seq, fixed, dt)
    mon = SubstepMonitor(ad, default_detectors(CFG), episode_id=name, config=MonitorConfig(snapshot_interval_substeps=0, save_episode_logs=False)).attach()
    for k in range(1, len(seq)):
        ad.step_physics(); ad.cb()
    mon.finalize()
    ev = collections.Counter((e.detector, e.status.value) for e in mon.events.values()); conf = []
    names = [ad.ids[i] for i in free]
    for e in mon.events.values():
        if e.status.value != 'confirmed': continue
        dd = e.to_dict(); dd.pop('metrics', None)
        j = next((names.index(n) for n in names if n in e.bodies), None)
        if j is not None: dd['gravity_ratio'], dd['v_peak'], dd['drop_m'] = gravity_ratio(Z[:, j], S[:, j], min(e.onset_substep, len(Z) - 1), dt)
        conf.append(dd)
    ex = freefall_excess(Z, S); md = [s['min_distance'] for s in seq if s.get('min_distance') is not None]
    out.append(dict(scene=name, engine='rigid-ipc dfff467', dt=dt, free=len(free), steps=len(seq) - 1, peak_speed=float(S.max()), freefall_excess=ex[0], excess_time_s=ex[1] * dt, excess_speed=ex[3],
                    min_distance_mm=1000 * min(md) if md else None, contact_steps=sum(1 for s in seq if s.get('min_distance') is not None and s['min_distance'] <= 0.005),
                    monitor_errors=mon.error_count, events={'%s:%s' % k: v for k, v in sorted(ev.items())}, confirmed=conf))
    print('%-26s excess %.2f peak %.2f min dist %s mm contact steps %d  %s kept %s err %d' % (name, ex[0], S.max(), None if not md else round(1000 * min(md), 3), out[-1]['contact_steps'],
          {k: v for k, v in ev.items() if k[0] == 'contact_ejection'}, [round(c.get('gravity_ratio', -1), 2) for c in conf], mon.error_count), file=sys.stderr)
json.dump(out, sys.stdout, indent=1)
