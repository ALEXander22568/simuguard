# -*- coding: utf-8 -*-
"""Feasibility probe: run the Rigid IPC unit-test scenes (their MuJoCo MJCF files) in MuJoCo.
Reports, per scene: peak speed of any free body, deepest reported penetration, and the largest
one-step speed jump of a body. Not the SimuGuard detector; only checks that the scenes load and run.
usage: ../engine_stress/venv/bin/python -I run_mujoco.py
"""
import glob, os, re, sys
import numpy as np, mujoco
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'repo')
rows = []
for f in sorted(glob.glob(R + '/comparisons/MuJoCo/unit-tests/**/*.xml', recursive=True)):
    name = os.path.relpath(f, R + '/comparisons/MuJoCo/unit-tests').replace('_mjc.xml', '')
    xml = open(f).read().replace('<mujoco>', '<mujoco><compiler meshdir="%s/meshes"/>' % R, 1)
    # the repository ships OBJ meshes; the MJCF files name STL copies
    xml = re.sub(r'file="([^"]+)\.stl"', lambda k: 'file="%s.%s"' % (k.group(1), 'obj' if os.path.exists('%s/meshes/%s.obj' % (R, k.group(1))) else 'stl'), xml)
    # a flat plane mesh has no convex hull: MuJoCo needs inertia from the hull only for moving bodies
    xml = re.sub(r'(<mesh [^>]*file="plane[^"]*"[^>]*)/>', r'\1 inertia="shell"/>', xml)
    try:
        m = mujoco.MjModel.from_xml_string(xml)
    except Exception as e:
        rows.append((name, 'LOAD FAIL: ' + str(e).splitlines()[0][:90])); continue
    d = mujoco.MjData(m)
    vmax = jump = 0.0; pen = 0.0; nan = False
    prev = np.zeros((m.nbody, 3))
    for _ in range(int(5.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)): nan = True; break
        v = d.cvel[:, 3:]
        sp = np.linalg.norm(v, axis=1)
        vmax = max(vmax, sp.max()); jump = max(jump, np.linalg.norm(v - prev, axis=1).max()); prev = v.copy()
        if d.ncon: pen = min(pen, d.contact.dist[:d.ncon].min())
    rows.append((name, 'dt=%g nbody=%d peak speed %.2f m/s, max one-step jump %.2f m/s, deepest penetration %.1f mm%s' % (m.opt.timestep, m.nbody - 1, vmax, jump, -1000 * pen, ' DIVERGED' if nan else '')))
for r in rows: print('%-28s %s' % r)
