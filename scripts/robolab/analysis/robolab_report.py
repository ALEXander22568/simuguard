"""RoboLab formal results -> report.md + per_task.csv (reported / exclusion / interval / artifact rate)."""
import json, math, random, collections as C, csv
E = json.load(open('events.json')); P = json.load(open('episodes.json'))
key = lambda x: (x['policy'], x['task'], x['run'], x['env'])
ev = C.defaultdict(list)
for x in E: ev[key(x)].append(x)
def fruit(x): return {'lemon_02', 'pomegranate01'} <= set(x['bodies']) and x['cstep'] <= 3
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
def scores(rows, flag):
    N = len(rows); S = sum(r['success_rl'] for r in rows); A = [r for r in rows if flag(r)]; a = len(A); SA = sum(r['success_rl'] for r in A)
    return dict(N=N, S=S, A=a, SA=SA, rep=S / N, excl=(S - SA) / (N - a), lo=(S - SA) / N, hi=(S - SA + a) / N, rate=a / N, ci=wilson(a, N))
L = ['# RoboLab formal run (SimuGuard) — summary', '',
     'Flag = episode has >=1 confirmed contact_ejection. Scores in %. interval = [(S-S_A)/N, (S-S_A+|A|)/N].', '']
defs = {'all flags': lambda r: key(r) in ev,
        'without the fruit-plate spawn overlap': lambda r: any(not fruit(x) for x in ev.get(key(r), []))}
L += ['| policy | flag set | N | S | reported | \\|A\\| | artifact rate (95% CI) | S_A | exclusion | interval |', '|---|---|---|---|---|---|---|---|---|---|']
for pol in ('pi05', 'gr00t'):
    rows = [r for r in P if r['policy'] == pol]
    for n, f in defs.items():
        s = scores(rows, f)
        L.append(f"| {pol} | {n} | {s['N']} | {s['S']} | {100*s['rep']:.1f} | {s['A']} | {100*s['rate']:.1f} ({100*s['ci'][0]:.1f}–{100*s['ci'][1]:.1f}) | {s['SA']} | {100*s['excl']:.1f} | [{100*s['lo']:.1f}, {100*s['hi']:.1f}] |")
L += ['']
# success rate inside vs outside A, and by attribute
for pol in ('pi05', 'gr00t'):
    rows = [r for r in P if r['policy'] == pol]
    A = [r for r in rows if key(r) in ev]; B = [r for r in rows if key(r) not in ev]
    tasks = C.Counter(r['task'] for r in A)
    L.append(f"- {pol}: success inside A {sum(r['success_rl'] for r in A)}/{len(A)} = {100*sum(r['success_rl'] for r in A)/len(A):.1f}%, outside {100*sum(r['success_rl'] for r in B)/len(B):.1f}%; "
             f"tasks with >=1 flag {len(tasks)}/{len({r['task'] for r in rows})}; tasks with >=15/20 flagged {sum(v>=15 for v in tasks.values())}.")
L += ['', '## Artifact rate by RoboLab task attribute', '', '| attribute | pi05 | gr00t |', '|---|---|---|']
attrs = sorted({a for r in P for a in r['attributes']})
for a in attrs:
    cells = []
    for pol in ('pi05', 'gr00t'):
        rows = [r for r in P if r['policy'] == pol and a in r['attributes']]
        k = sum(key(r) in ev for r in rows); cells.append(f"{100*k/len(rows):.1f}% ({k}/{len(rows)})" if rows else '—')
    L.append(f"| {a} | {cells[0]} | {cells[1]} |")
L += ['', '## Event types (confirmed contact_ejection events)', '']
for pol in ('pi05', 'gr00t'):
    xs = [x for x in E if x['policy'] == pol]
    c = C.Counter('fruit-plate spawn overlap' if fruit(x) else 'robot link involved' if any(b.startswith('robot/') for b in x['bodies'])
                  else 'shelf rack' if 'rack_l04' in x['bodies'] else 'first 1 s' if x['cstep'] <= 15 else 'object–object' for x in xs)
    L.append(f"- {pol}: {len(xs)} events; " + ', '.join(f"{k} {v}" for k, v in c.most_common()))
open('report.md', 'w').write('\n'.join(L) + '\n')
with open('per_task.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['policy', 'task', 'N', 'S', 'A', 'S_A', 'reported', 'exclusion'])
    for pol in ('pi05', 'gr00t'):
        for t in sorted({r['task'] for r in P if r['policy'] == pol}):
            rows = [r for r in P if r['policy'] == pol and r['task'] == t]; s = scores(rows, lambda r: key(r) in ev) if any(key(r) not in ev for r in rows) else None
            N = len(rows); S = sum(r['success_rl'] for r in rows); A = [r for r in rows if key(r) in ev]
            w.writerow([pol, t, N, S, len(A), sum(r['success_rl'] for r in A), round(S / N, 3), round((S - sum(r['success_rl'] for r in A)) / (N - len(A)), 3) if N > len(A) else ''])
print(open('report.md').read())
