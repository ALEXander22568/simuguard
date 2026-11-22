"""Human labels of the RoboLab review set P -> detector precision per stratum, and what it does to the scores.

usage: python3 label_report.py review/labels_user_20261110.json  -> review/label_report.md / .json
P was drawn per policy and stratum (select_review.py), so precision is weighted by stratum size.
Per stratum, p_s ~ Beta(k+1, n-k+1); the true-artifact set is p_s of the flagged episodes in that stratum,
with the same success share as the flagged ones (episodes in a stratum are exchangeable).
Two notions: real = the flagged moment is a physics artifact; harmful = real and it may change the outcome.
"""
import json, sys, random, collections as C
from select_review import stratum, ev, key, P as EPIS   # same strata as the sampling
items = json.load(open('review_items.json'))
lab = json.load(open(sys.argv[1]))['labels']
rng = random.Random(0); B = 20000
out = {}; L = ['# RoboLab: human review of the flagged episodes (set P)', '']
for pol in ('pi05', 'gr00t'):
    rows = [r for r in EPIS if r['policy'] == pol]
    N = len(rows); S = sum(r['success_rl'] for r in rows)
    A = C.defaultdict(list)
    for r in rows:
        if key(r) in ev: A[stratum(ev[key(r)])].append(r)
    nA = sum(len(v) for v in A.values()); SA = sum(r['success_rl'] for v in A.values() for r in v)
    res = {'N': N, 'S': S, 'A': nA, 'SA': SA, 'strata': {}}
    for notion in ('real', 'harmful'):
        kn = {}
        for s in A:
            its = [i for i in items if i['set'] == 'P' and i['policy'] == pol and i['stratum'] == s]
            ok = lambda l: l.get('real') == 'yes' and (notion == 'real' or l.get('effect') == 'yes')
            kn[s] = (sum(ok(lab.get(i['id'], {})) for i in its), sum(i['id'] in lab for i in its))
        def est(ps):
            T = sum(ps[s] * len(A[s]) for s in A); TS = sum(ps[s] * sum(r['success_rl'] for r in A[s]) for s in A)
            return dict(prec=T / nA, T=T, lo=(S - TS) / N, hi=(S - TS + T) / N, excl=(S - TS) / (N - T))
        pt = est({s: (k / n if n else 0.0) for s, (k, n) in kn.items()})
        draws = [est({s: rng.betavariate(k + 1, n - k + 1) for s, (k, n) in kn.items()}) for _ in range(B)]
        ci = {m: (sorted(d[m] for d in draws)[int(.025 * B)], sorted(d[m] for d in draws)[int(.975 * B)]) for m in pt}
        res[notion] = {'point': pt, 'ci95': ci, 'k_n': kn}
    # set R: one unflagged episode per sampled task; weight by the task's unflagged count
    U = [r for r in rows if key(r) not in ev]; nU = len(U); SU = sum(r['success_rl'] for r in U)
    ucount = C.Counter(r['task'] for r in U)
    Ri = [i for i in items if i['set'] == 'R' and i['policy'] == pol and i['id'] in lab]
    kR = sum(lab[i['id']].get('miss') == 'yes' for i in Ri); nR = len(Ri)
    wsum = sum(ucount[i['task']] for i in Ri)
    q = sum(ucount[i['task']] * (lab[i['id']].get('miss') == 'yes') for i in Ri) / wsum if wsum else 0.0
    def full(pt, qq):
        F = qq * nU; FS = F * SU / nU
        lo = pt['lo'] - FS / N; hi = lo + (pt['T'] + F) / N
        return dict(F=F, lo=lo, hi=hi, excl=(N * pt['lo'] - FS) / (N - pt['T'] - F))
    qs = sorted(rng.betavariate(kR + 1, nR - kR + 1) for _ in range(B))
    res['R'] = {'k': kR, 'n': nR, 'miss_rate_weighted': q, 'miss_rate_ci95': (qs[int(.025 * B)], qs[int(.975 * B)]),
                'unflagged': nU, 'full': {n: full(res[n]['point'], q) for n in ('real', 'harmful')},
                'full_upper': {n: full(res[n]['point'], qs[int(.975 * B)]) for n in ('real', 'harmful')}}
    res['strata'] = {s: {'A': len(v), 'SA': sum(r['success_rl'] for r in v)} for s, v in A.items()}
    out[pol] = res
    f = lambda x: f'{100*x:.1f}'
    L += [f'## {pol}  (N={N}, reported {f(S/N)}, flagged {nA}, frozen exclusion {f((S-SA)/(N-nA))}, frozen interval [{f((S-SA)/N)}, {f((S-SA+nA)/N)}])', '',
          '| stratum | flagged (succ.) | real k/n | harmful k/n |', '|---|---|---|---|']
    for s in sorted(A, key=lambda s: -len(A[s])):
        L.append(f"| {s} | {len(A[s])} ({res['strata'][s]['SA']}) | {res['real']['k_n'][s][0]}/{res['real']['k_n'][s][1]} | {res['harmful']['k_n'][s][0]}/{res['harmful']['k_n'][s][1]} |")
    L += ['', '| notion | precision | est. true-artifact episodes | exclusion | interval |', '|---|---|---|---|---|']
    for notion in ('real', 'harmful'):
        p, c = res[notion]['point'], res[notion]['ci95']
        L.append(f"| {notion} | {f(p['prec'])} [{f(c['prec'][0])}, {f(c['prec'][1])}] | {p['T']:.0f} | {f(p['excl'])} [{f(c['excl'][0])}, {f(c['excl'][1])}] | [{f(p['lo'])}, {f(p['hi'])}] |")
    R = res['R']
    L += ['', f"Set R: {R['k']}/{R['n']} unflagged episodes have a missed artifact; weighted miss rate {f(R['miss_rate_weighted'])} "
          f"(95% upper {f(R['miss_rate_ci95'][1])}) of {R['unflagged']} unflagged -> about {R['full']['real']['F']:.0f} episodes; "
          "missed episodes given the unflagged success share.", '',
          '| notion | exclusion (P+R) | interval (P+R) | interval, miss rate at 95% upper |', '|---|---|---|---|']
    for n in ('real', 'harmful'):
        a, u = R['full'][n], R['full_upper'][n]
        L.append(f"| {n} | {f(a['excl'])} | [{f(a['lo'])}, {f(a['hi'])}] | [{f(u['lo'])}, {f(u['hi'])}] |")
    L.append('')
L += [f"Labeled: P {sum(k[0]=='P' for k in lab)}/38, R {sum(k[0]=='R' for k in lab)}/100. Brackets: 95% posterior, uniform prior per stratum."]
json.dump(out, open('review/label_report.json', 'w'), indent=1)
open('review/label_report.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
