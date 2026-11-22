"""Pick the review sets (seed fixed). P = flagged episodes for precision, stratified by event type;
R = unflagged episodes for recall, one per task, random tasks.  Writes review_sets.json."""
import json, random, collections as C
E = json.load(open('events.json')); P = json.load(open('episodes.json'))
key = lambda x: (x['policy'], x['task'], x['run'], x['env'])
ev = C.defaultdict(list)
for x in E: ev[key(x)].append(x)
def stratum(evs):
    if any(any(b.startswith('robot/') for b in x['bodies']) for x in evs): return 'robot'
    if any({'lemon_02', 'pomegranate01'} <= set(x['bodies']) for x in evs): return 'fruit_overlap'
    if any('rack_l04' in x['bodies'] for x in evs): return 'shelf_rack'
    if all(x['cstep'] <= 15 for x in evs): return 'first_1s'
    return 'object'
QUOTA = {'robot': 5, 'fruit_overlap': 3, 'shelf_rack': 4, 'first_1s': 3, 'object': 5}
rng = random.Random(20261110)
out = {'P': [], 'R': [], 'strata_sizes': {}}
for pol in ('pi05', 'gr00t'):
    rows = [r for r in P if r['policy'] == pol]
    flagged = [r for r in rows if key(r) in ev]
    by = C.defaultdict(list)
    for r in flagged: by[stratum(ev[key(r)])].append(r)
    out['strata_sizes'][pol] = {k: len(v) for k, v in by.items()}
    for s, q in QUOTA.items():
        pool = by[s][:]; rng.shuffle(pool); seen = set(); pick = []
        for r in pool:                      # spread over tasks first
            if r['task'] not in seen: pick.append(r); seen.add(r['task'])
            if len(pick) == q: break
        for r in pool:
            if len(pick) == q: break
            if r not in pick: pick.append(r)
        for r in pick: out['P'].append({**{k: r[k] for k in ('box', 'policy', 'task', 'run', 'env', 'success_rl', 'episode_length')}, 'stratum': s})
    un = [r for r in rows if key(r) not in ev]
    tasks = sorted({r['task'] for r in un}); rng.shuffle(tasks)
    for t in tasks[:50]:
        r = rng.choice([u for u in un if u['task'] == t])
        out['R'].append({**{k: r[k] for k in ('box', 'policy', 'task', 'run', 'env', 'success_rl', 'episode_length')}, 'stratum': 'unflagged'})
json.dump(out, open('review_sets.json', 'w'), indent=1)
print(out['strata_sizes'])
print('P', len(out['P']), C.Counter((x['policy'], x['box']) for x in out['P']))
print('R', len(out['R']), C.Counter((x['policy'], x['box']) for x in out['R']), 'minutes of video', sum(x['episode_length'] for x in out['R']) / 15 / 60)
