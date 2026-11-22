"""Compare contact_ejection with simple baselines on the human-labeled RoboLab episodes.

usage: python3 baseline_report.py <labels.json> bl/out_24g.jsonl bl/out_48g.jsonl -> baseline/baseline_report.md/.json
Truth: P item real=yes, or R item miss=yes ('harmful' variant: P also needs effect=yes).
Each labeled episode stands for part of the population: a P item in stratum s for A_s/n_s flagged episodes,
an R item for U/n_R unflagged episodes of its policy. Weighted counts give population precision / recall.
Baselines flag an episode when its statistic exceeds a threshold; thresholds shown: (a) same flag rate as
contact_ejection on all episodes of that policy pooled, (b) fixed physical values, (c) best weighted F1 on the labels
(tuned on the test labels, so optimistic).
"""
import json, sys, collections as C
import numpy as np
from select_review import stratum, ev, key, P as EPIS
items = json.load(open('review_items.json')); lab = json.load(open(sys.argv[1]))['labels']
st = {}
for f in sys.argv[2:]:
    for l in open(f):
        x = json.loads(l)
        if not x.get('error'): st[tuple(x['key'])] = x
k_ = lambda it: (it['policy'], it['task'], it['run'], it['env'])
pop = C.Counter(); popU = C.Counter(); nS = C.Counter(); nR = C.Counter()
for r in EPIS:
    if key(r) in ev: pop[(r['policy'], stratum(ev[key(r)]))] += 1
    else: popU[r['policy']] += 1
lab_items = [it for it in items if it['id'] in lab]
for it in lab_items:
    (nS if it['set'] == 'P' else nR)[(it['policy'], it['stratum']) if it['set'] == 'P' else it['policy']] += 1
def w(it): return pop[(it['policy'], it['stratum'])] / nS[(it['policy'], it['stratum'])] if it['set'] == 'P' else popU[it['policy']] / nR[it['policy']]
def truth(it, notion):
    l = lab[it['id']]
    if it['set'] == 'R': return l.get('miss') == 'yes'
    return l.get('real') == 'yes' and (notion == 'real' or l.get('effect') == 'yes')
miss_stats = [it['id'] for it in lab_items if k_(it) not in st]
lab_items = [it for it in lab_items if k_(it) in st]
STATS = ['vmax', 'vmax_late', 'dvmax', 'dvmax_late', 'kemax', 'kemax_late', 'pen', 'pen_late']
DESC = {'vmax': 'speed', 'dvmax': 'velocity jump per substep', 'kemax': 'kinetic-energy jump per substep (J/kg)', 'pen': 'penetration depth (m)'}
FIXED = {'vmax': [0.5, 1.0, 2.0], 'dvmax': [0.35, 0.75, 1.5], 'kemax': [0.1, 0.5, 1.0], 'pen': [0.005, 0.01, 0.02]}
def evaluate(flag, notion):
    tp = fp = fn = 0.0; raw = C.Counter()
    for it in lab_items:
        t, f, ww = truth(it, notion), flag(it), w(it)
        if f and t: tp += ww; raw['tp'] += 1
        elif f and not t: fp += ww; raw['fp'] += 1
        elif t: fn += ww; raw['fn'] += 1
        else: raw['tn'] += 1
    p = tp / (tp + fp) if tp + fp else float('nan'); r = tp / (tp + fn) if tp + fn else float('nan')
    return dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0, est_tp=tp, est_fp=fp, est_fn=fn, raw=dict(raw))
allk = [key(r) for r in EPIS if key(r) in st]
sg_rate = sum(k in ev for k in allk) / len(allk)
res = {'labeled': len(lab_items), 'missing_stats': miss_stats, 'episodes_with_stats': len(allk), 'sg_flag_rate': sg_rate, 'methods': {}}
def add(name, flag_k, notion, extra):
    flag = lambda it: flag_k(k_(it))
    e = evaluate(flag, notion); e['flag_rate'] = sum(flag_k(k) for k in allk) / len(allk); e.update(extra)
    res['methods'].setdefault(notion, []).append({'method': name, **e})
for notion in ('real', 'harmful'):
    add('SimuGuard contact_ejection (frozen)', lambda k: k in ev, notion, {})
    for s in STATS:
        vals = np.array([st[k][s] for k in allk]); base = s.replace('_late', '')
        th_a = float(np.quantile(vals, 1 - sg_rate))
        cand = [('same flag rate', th_a)] + [('fixed', t) for t in FIXED[base]]
        sweep = np.unique(np.quantile(vals, np.linspace(0.5, 0.999, 400)))
        best = max(sweep, key=lambda t: evaluate(lambda it: st[k_(it)][s] > t, notion)['f1'])
        cand.append(('best F1 on labels', float(best)))
        for how, th in cand:
            add(f'{s} > {th:.4g}', lambda k, s=s, th=th: st[k][s] > th, notion, {'stat': s, 'threshold': th, 'how': how})
json.dump(res, open('baseline/baseline_report.json', 'w'), indent=1)
f = lambda x: '—' if x != x else f'{100*x:.0f}'
L = ['# RoboLab: contact_ejection vs simple baselines on the labeled episodes', '',
     f"Labeled episodes used: {len(lab_items)} (38 flagged by SimuGuard, stratified; 100 unflagged). Statistics for {len(allk)} episodes.",
     'Precision / recall are population estimates (each labeled episode weighted by the part of the population it stands for). '
     '"late" = ignore the first second after reset. Raw counts on the labeled episodes in the last column.', '',
     f"SimuGuard flag rate on all episodes: {100*sg_rate:.1f}%.", '']
for notion in ('real', 'harmful'):
    L += [f'## truth = {notion}', '', '| method | threshold choice | flag rate | precision | recall | F1 | raw TP/FP/FN |', '|---|---|---|---|---|---|---|']
    for m in res['methods'][notion]:
        r = m['raw']; L.append(f"| {m['method']} | {m.get('how', '—')} | {100*m['flag_rate']:.1f} | {f(m['precision'])} | {f(m['recall'])} | {f(m['f1'])} | {r.get('tp',0)}/{r.get('fp',0)}/{r.get('fn',0)} |")
    L.append('')
L += ['Statistics: ' + '; '.join(f'{k} = max {v}' for k, v in DESC.items()) + '. Bodies: the same monitored free bodies as contact_ejection.']
open('baseline/baseline_report.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
