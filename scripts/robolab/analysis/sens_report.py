"""Threshold sensitivity of the RoboLab results (offline re-run of contact_ejection, see redetect.py).

usage: python3 sens_report.py sens/out_24g.jsonl sens/out_48g.jsonl  -> sens/sensitivity.md, sens/sensitivity.json
For every detector config: flagged episodes |A|, artifact rate, reported / exclusion / interval, per policy,
with and without the fruit-plate spawn overlap; plus overlap of the flagged set with the frozen config.
"""
import json
import sys
import collections as C

P = json.load(open('episodes.json'))
E = json.load(open('events.json'))
key = lambda r: (r['policy'], r['task'], r['run'], r['env'])
rows = {key(r): r for r in P}
res = {}
for f in sys.argv[1:]:
    for line in open(f):
        x = json.loads(line)
        res[tuple(x['key'])] = x
missing = [k for k in rows if k not in res]
errors = [k for k, x in res.items() if x.get('error')]
mism = [k for k, x in res.items() if not x.get('error') and not x['check']['equal']]


def fruit(e):
    return {'lemon_02', 'pomegranate01'} <= set(e['bodies']) and (e['onset'] - 1) // 8 <= 3


configs = list(next(iter(res.values()))['events'].keys())
base_cfg = dict(min_speed_mps=0.5, min_delta_v_mps=0.35, min_flight_distance_m=0.04, min_free_flight_ms=16.0,
                violent_delta_v_mps=0.75, violent_speed_mps=1.0)
out = {}
for cfg in configs:
    out[cfg] = {}
    for pol in ('pi05', 'gr00t'):
        ks = [k for k in rows if k[0] == pol and k in res and not res[k].get('error')]
        N = len(ks); S = sum(rows[k]['success_rl'] for k in ks)
        for variant in ('all', 'nofruit'):
            def flagged(k):
                ev = res[k]['events'][cfg]
                return any(not fruit(e) for e in ev) if variant == 'nofruit' else bool(ev)
            A = [k for k in ks if flagged(k)]
            SA = sum(rows[k]['success_rl'] for k in A); a = len(A)
            base_A = {k for k in ks if (any(not fruit(e) for e in res[k]['events']['base']) if variant == 'nofruit' else bool(res[k]['events']['base']))}
            out[cfg][f'{pol}/{variant}'] = dict(N=N, S=S, A=a, SA=SA, reported=S / N, exclusion=(S - SA) / (N - a) if N > a else None,
                                               lo=(S - SA) / N, hi=(S - SA + a) / N, rate=a / N,
                                               n_events=sum(len(res[k]['events'][cfg]) for k in ks),
                                               jaccard_vs_base=len(set(A) & base_A) / max(1, len(set(A) | base_A)),
                                               added=len(set(A) - base_A), removed=len(base_A - set(A)))
json.dump(out, open('sens/sensitivity.json', 'w'), indent=1)

# base must equal the frozen live results
for pol, want in (('pi05', 496), ('gr00t', 263)):
    got = out['base'][f'{pol}/all']['A']
    print(f'check base |A| {pol}: {got} (live {want})')

L = ['# RoboLab: threshold sensitivity of contact_ejection (offline re-run on the recorded episodes)', '',
     f'Episodes re-run: {len(res)}/{len(rows)}; errors {len(errors)}; base config differs from the live events in {len(mism)} episodes.',
     'One threshold changed at a time (x0.5 … x2), plus all six together x0.75 and x1.5. Scores in %.', '',
     'Frozen thresholds: ' + ', '.join(f'{k}={v}' for k, v in base_cfg.items()), '']
for variant, title in (('all', 'all flags'), ('nofruit', 'without the fruit-plate spawn overlap')):
    L += [f'## {title}', '', '| config | pi05 rate | pi05 excl. | pi05 interval | gr00t rate | gr00t excl. | gr00t interval | Jaccard vs frozen (pi05/gr00t) |',
          '|---|---|---|---|---|---|---|---|']
    for cfg in configs:
        cells = []
        for pol in ('pi05', 'gr00t'):
            s = out[cfg][f'{pol}/{variant}']
            cells += [f"{100*s['rate']:.1f} ({s['A']})", f"{100*s['exclusion']:.1f}", f"[{100*s['lo']:.1f}, {100*s['hi']:.1f}]"]
        j = f"{out[cfg]['pi05/'+variant]['jaccard_vs_base']:.2f} / {out[cfg]['gr00t/'+variant]['jaccard_vs_base']:.2f}"
        L.append(f'| {cfg} | ' + ' | '.join(cells) + f' | {j} |')
    L += ['', f"reported: pi05 {100*out['base']['pi05/all']['reported']:.1f}, gr00t {100*out['base']['gr00t/all']['reported']:.1f}", '']
open('sens/sensitivity.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
if missing:
    print('missing', len(missing), missing[:5])
if errors:
    print('errors', errors[:5])
if mism:
    print('mismatch', mism[:5])
