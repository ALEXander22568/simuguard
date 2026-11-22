"""HyVLA inputs for the narrow study: deduplicated inventory and flagged mapping.

Object basket shards s0-s4 overlap in seeds; merge rule (as in the paper's HyVLA numbers): one run per
seed, keep the shard with the lower start seed (s0 < s1 < ... < s4).  Flagged = physics-invalid event
in the frozen detector output (inventory.py), recordings complete.
"""
import csv, json, collections
rows = [json.loads(l) for l in open('inventory_raw_4090-B.jsonl')]
SH = {'s0': 0, 's1': 1, 's2': 2, 's3': 3, 's4': 4}
keep, dropped = {}, []
for r in rows:
    shard = r['root'].rsplit('/', 1)[1]
    k = (r['task'], r['seed'])
    rank = SH.get(shard, 0)
    if k in keep:
        old_rank = SH.get(keep[k]['root'].rsplit('/', 1)[1], 0)
        if rank < old_rank:
            dropped.append(keep[k]); keep[k] = r
        else:
            dropped.append(r)
    else:
        keep[k] = r
inv = sorted(keep.values(), key=lambda r: (r['task'], r['seed']))
with open('inventory_4090-B.jsonl', 'w') as fh:
    for r in inv: fh.write(json.dumps(r) + '\n')
print('inventory', len(rows), '->', len(inv), 'dropped duplicate-seed runs:',
      [(r['task'], r['seed'], r['root'].rsplit('/', 1)[1], 'S' if r['success'] else 'F') for r in dropped])
c = collections.Counter(); s = collections.Counter()
for r in inv: c[r['task']] += 1; s[r['task']] += r['success']
print({t: f"{s[t]}/{c[t]}" for t in sorted(c)})

paper = {}
for p in ('broad_paper_summary_containers.json', 'broad_paper_summary_object_basket.json'):
    for r in json.load(open(p))['rows']:
        paper[tuple(r['segment'].split('/')[-4::3])] = r  # (task dir, segment name); runs/ is a symlink
out = []
for r in inv:
    if not r['n_invalid']:
        continue
    f = r['files']
    p = paper.get(tuple(r['segment'].split('/')[-4::3]))
    out.append(dict(episode_id=f"{r['task']}/hyvla/{r['seed']}", task=r['task'], group='hyvla', seed=r['seed'],
        recorded='S' if r['success'] else 'F',
        paper_replayed='' if p is None else ('S' if p['counterfactual_success']['depen_1.0@event'] else 'F'),
        status='found' if (f['controls.npz'] and f['states.npz'] and f['trace.jsonl.gz']) else 'found_incomplete',
        host='4090-B', segment=r['segment'], invalid_stage=r['invalid_stage'], onset_substep=r['first_invalid_onset'],
        body=r['first_invalid_body'], event_id=r['first_invalid_event_id'], note=''))
with open('flagged_mapping_hyvla.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print('flagged', len(out), collections.Counter((o['task'], o['recorded'], o['paper_replayed'], o['status']) for o in out))
print('paper rows not in flagged:', [k for k in paper if k not in {tuple(o['segment'].split('/')[-4::3]) for o in out}])
