import json,collections as C
E=json.load(open('events.json')); P=json.load(open('episodes.json'))
def key(x): return (x['policy'],x['task'],x['run'],x['env'])
ev=C.defaultdict(list)
for x in E: ev[key(x)].append(x)
def cls(x):
    robot=any(b.startswith('robot/') for b in x['bodies'])
    if x['cstep']<=15: return 'spawn'          # first 1 s
    return 'robot' if robot else 'obj'
def score(rows,flag):
    N=len(rows);S=sum(r['success_rl'] for r in rows)
    A=[r for r in rows if flag(r)];a=len(A);SA=sum(r['success_rl'] for r in A)
    return dict(N=N,S=S,rep=S/N,A=a,rate=a/N,SA=SA,excl=(S-SA)/(N-a),lo=(S-SA)/N,hi=(S-SA+a)/N)
for pol in ['pi05','gr00t']:
    rows=[r for r in P if r['policy']==pol]
    print('==',pol)
    print(' event classes',C.Counter(cls(x) for x in E if x['policy']==pol))
    defs={'all':lambda r:key(r) in ev,
          'non-spawn':lambda r:any(cls(x)!='spawn' for x in ev.get(key(r),[])),
          'robot-contact':lambda r:any(cls(x)=='robot' for x in ev.get(key(r),[])),}
    for n,f in defs.items():
        s=score(rows,f); print(f' {n:14s} A={s["A"]:4d} rate={s["rate"]*100:5.1f} SA={s["SA"]:3d} rep={s["rep"]*100:5.1f} excl={s["excl"]*100:5.1f} int=[{s["lo"]*100:.1f},{s["hi"]*100:.1f}]')
    # per-task concentration
    t=C.Counter(r['task'] for r in rows if key(r) in ev)
    print(' tasks with >=15/20 flagged:',sum(v>=15 for v in t.values()),' flagged eps in those:',sum(v for v in t.values() if v>=15))
    # same body pair recurring per task
    pairs=C.Counter((x['task'],tuple(sorted(x['bodies']))) for x in E if x['policy']==pol)
    print(' top recurring',pairs.most_common(12))
