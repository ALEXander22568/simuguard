# -*- coding: utf-8 -*-
"""Shared by the MuJoCo and PhysX runs of the Rigid IPC unit tests: scene parsing and the energy check.
A scene has no robot and no motor, so its total mechanical energy can never rise. We report the largest
rise over the initial value, relative to the energy the bodies could release by falling to the lowest point they reach."""
import glob, os, re
import numpy as np
import xml.etree.ElementTree as ET
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, 'repo'); G = 9.81

def slab():
    """The zero-thickness plane mesh as a 0.1 m slab under the same surface (a convex hull needs volume)."""
    out = os.path.join(H, 'work', 'plane_slab.obj'); os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out): return out
    V = np.array([[float(x) for x in l.split()[1:4]] for l in open(R + '/meshes/plane.obj') if l.startswith('v ')])
    ax = int(np.argmin(np.ptp(V, axis=0))); W = V.copy(); W[:, ax] -= 0.1
    with open(out, 'w') as f:
        for v in np.vstack([V, W]): f.write('v %g %g %g\n' % tuple(v))
        # V order: (-,+) (+,+) (-,-) (+,-) in the two flat axes; 12 triangles of the box
        for a, b, c in [(1,2,4),(1,4,3),(5,8,6),(5,7,8),(1,5,6),(1,6,2),(2,6,8),(2,8,4),(4,8,7),(4,7,3),(3,7,5),(3,5,1)]:
            f.write('f %d %d %d\n' % (a, b, c))
    return out

def mesh_path(name):
    if name.startswith('plane'): return slab()
    base = os.path.splitext(name)[0]
    for ext in ('.obj', '.stl'):
        if os.path.exists(R + '/meshes/' + base + ext): return R + '/meshes/' + base + ext
    raise FileNotFoundError(name)

def scenes():
    for f in sorted(glob.glob(R + '/comparisons/MuJoCo/unit-tests/**/*.xml', recursive=True)):
        yield os.path.relpath(f, R + '/comparisons/MuJoCo/unit-tests').replace('_mjc.xml', ''), f

def parse(f):
    """-> list of bodies: dict(mesh=path, scale, pos, quat(wxyz), free, friction)."""
    root = ET.parse(f).getroot()
    meshes = {m.get('name'): (mesh_path(m.get('file')), [float(x) for x in (m.get('scale') or '1 1 1').split()]) for m in root.iter('mesh')}
    out = []
    for b in root.find('worldbody').findall('body'):
        g = b.find('geom'); path, scale = meshes[g.get('mesh')]
        out.append(dict(mesh=path, scale=scale, pos=[float(x) for x in b.get('pos').split()],
                        quat=[float(x) for x in (b.get('quat') or '1 0 0 0').split()],
                        free=b.find('freejoint') is not None, friction=float((g.get('friction') or '1').split()[0])))
    return out

class Energy:
    """Feed (mass, com z, kinetic energy) per free body each step."""
    def __init__(self): self.e0 = None; self.rise = 0.0; self.z0 = None; self.zmin = None; self.m = None; self.ke0 = 0.0
    def add(self, m, z, ke):
        m, z = np.asarray(m, float), np.asarray(z, float); e = float(np.sum(m * G * z) + np.sum(ke))
        if self.e0 is None: self.e0, self.z0, self.zmin, self.m, self.ke0 = e, z.copy(), z.copy(), m, float(np.sum(ke))
        self.zmin = np.minimum(self.zmin, z); self.rise = max(self.rise, e - self.e0)
    def ratio(self):
        ref = float(np.sum(self.m * G * (self.z0 - self.zmin))) + self.ke0
        return self.rise / ref if ref > 1e-9 else float('nan')

def gravity_ratio(z, speed, onset, dt, before_s=0.6, after_s=0.1, v_floor=0.05):
    """Same rule as scripts/robotwin/gravity_filter.py: peak speed within 0.1 s after onset, against the free-fall
    speed for the drop from the highest point in the 0.6 s before onset to that peak. Above 1.5 = gravity cannot explain it."""
    z, speed = np.asarray(z), np.asarray(speed)
    a, b = max(0, onset - int(round(before_s / dt))), min(len(z) - 1, onset + int(round(after_s / dt)))
    k = onset + int(np.argmax(speed[onset:b + 1])) if b >= onset else onset
    dz = max(0.0, float(z[a:k + 1].max() - z[k]))
    return float(speed[k] / max(np.sqrt(2 * G * dz), v_floor)), float(speed[k]), dz

def freefall_excess(Z, S, v_floor=0.5):
    """Z, S: (steps, bodies) com height and speed. Largest ratio of a body's speed to the free-fall speed from the
    highest starting point in the scene down to where the body is. No passive, inelastic scene can exceed 1 by much."""
    Z, S = np.asarray(Z), np.asarray(S); top = Z[0].max()
    bound = np.maximum(np.sqrt(2 * G * np.maximum(top - Z, 0.0)), v_floor)
    r = S / bound; k = np.unravel_index(np.argmax(r), r.shape)
    return float(r[k]), int(k[0]), int(k[1]), float(S[k])
