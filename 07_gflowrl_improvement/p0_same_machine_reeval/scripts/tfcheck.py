import json, glob, os
def recs(p):
    return [json.loads(l) for l in open(p)]
bad = {}   # (tag, bench) -> set of sample_id
for d in sorted(glob.glob('/workspace/parsed/*')):
    tag = os.path.basename(d)
    for f in glob.glob(d + '/*.jsonl'):
        bench = os.path.basename(f)[:-6]
        ids = []
        for r in recs(f):
            if r['tool_failures'] or r['hit_max_turns'] or not r['parse_ok']:
                ids.append((r['sample_id'], bool(r['tool_failures']), r['hit_max_turns'],
                            r['parse_ok'], r['correct']))
        if ids:
            print(f'{tag} [{bench}]')
            for i in ids:
                print(f'   sample {i[0]:4d}  tool_fail={i[1]} max_turns={i[2]} parse_ok={i[3]} correct={i[4]}')
            bad[(tag, bench)] = set(x[0] for x in ids)

# are those samples discordant?
def load(p, thr=0.5):
    return {json.loads(l)['index']: (1 if json.loads(l)['score'] >= thr else 0) for l in open(p)}
print('\n--- how these samples did across the three runs, plus the three-arm comparison ---')
for (tag, bench), ids in bad.items():
    ck = tag.split('_')[0]
    for sid in sorted(ids):
        row = []
        for t in ['sft', 'p4', 'cp']:
            v = [load(f'/workspace/exp/p0/{t}/run{i}/{bench}/0.jsonl').get(sid) for i in (1, 2, 3)]
            row.append(f'{t}={v}')
        print(f'{bench} #{sid:4d} (the one with the problem is {tag}):  ' + '  '.join(row))
