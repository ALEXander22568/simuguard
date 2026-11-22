"""Per-episode statistics for simple baseline detectors, from the recorded RoboLab segments (no physics).

usage: PYTHONPATH=simuguard python baseline_stats.py <segs.json> <out.jsonl>
Same monitored bodies as contact_ejection (free bodies with role target/container/object). Per episode:
  vmax   max linear speed of a monitored body                      (speed threshold)
  dvmax  max |v(t) - v(t-1)| of a monitored body between substeps   (velocity-jump / impulse threshold)
  kemax  max increase of 0.5 |v|^2 between substeps                 (energy-jump threshold, per unit mass)
  pen    max penetration depth (-separation) of a contact on a monitored body
each also as *_late, ignoring the first second (substep <= 120, where objects settle after spawn).
"""
import json, sys
from pathlib import Path
import numpy as np
from redetect import load_bodies
from simuguard.core.detectors.base import DetectorContext
from simuguard.core.types import BodyRole


def stats(seg: Path) -> dict:
    man = json.loads((seg / 'manifest.json').read_text())
    roles = tuple(BodyRole(r) for r in man['detectors']['contact_ejection']['monitored_roles'])
    ctx = DetectorContext(episode_id=man['episode_id'], timestep=float(man['timestep_s']), bodies=load_bodies(man))
    free = ctx.free_bodies(roles)
    st = np.load(seg / 'states.npz', allow_pickle=False)
    ids = json.loads(str(st['body_ids'])); col = [i for i, b in enumerate(ids) if b in free]
    sub = st['substeps']; v = st['states'][:, col, 7:10]
    sp = np.linalg.norm(v, axis=2)
    late = sub > 120
    out = {}
    spm = sp.max(axis=1) if sp.size else np.zeros(len(sub))
    out['vmax'] = float(spm.max()) if len(spm) else 0.0
    out['vmax_late'] = float(spm[late].max()) if late.any() else 0.0
    if len(sub) > 1 and sp.size:
        cont = np.diff(sub) == 1
        dv = np.linalg.norm(np.diff(v, axis=0), axis=2).max(axis=1); dv[~cont] = 0
        ke = np.diff(0.5 * sp ** 2, axis=0).max(axis=1); ke[~cont] = 0
        l2 = late[1:]
        out.update(dvmax=float(np.nanmax(dv)), dvmax_late=float(np.nanmax(dv[l2])) if l2.any() else 0.0,
                   kemax=float(np.nanmax(ke)), kemax_late=float(np.nanmax(ke[l2])) if l2.any() else 0.0)
    else:
        out.update(dvmax=0.0, dvmax_late=0.0, kemax=0.0, kemax_late=0.0)
    ct = np.load(seg / 'contacts.npz', allow_pickle=False)
    names = json.loads(str(ct['names'])); rows = ct['rows']
    mon = np.array([n in free for n in names], bool)
    if len(rows):
        a = rows[:, 1].astype(int); b = rows[:, 2].astype(int)
        m = mon[a] | mon[b]
        depth = -rows[:, 12].astype(np.float64)
        out['pen'] = float(depth[m].max()) if m.any() else 0.0
        ml = m & (rows[:, 0] > 120)
        out['pen_late'] = float(depth[ml].max()) if ml.any() else 0.0
    else:
        out['pen'] = out['pen_late'] = 0.0
    out['n_free'] = len(col)
    return out


def main():
    segs = json.loads(Path(sys.argv[1]).read_text()); outp = Path(sys.argv[2])
    done = set()
    if outp.exists():
        done = {json.loads(l)['seg_dir'] for l in open(outp)}
    with open(outp, 'a') as fh:
        for it in segs:
            if it['seg_dir'] in done: continue
            try:
                r = stats(Path(it['seg_dir'])); r['error'] = None
            except Exception as e:  # noqa: BLE001
                r = {'error': repr(e)}
            r.update(key=it['key'], seg_dir=it['seg_dir']); fh.write(json.dumps(r) + '\n'); fh.flush()


if __name__ == '__main__':
    main()
