import json, itertools
from math import comb
from collections import defaultdict

TAGS = ['sft', 'p4', 'cp']
NAME = {'sft': 'SFT starting point', 'p4': 'P4 official GRPO', 'cp': "C' step85"}

def load(p, thr=0.5):
    d = {}
    for line in open(p):
        r = json.loads(line)
        d[r['index']] = 1 if r['score'] >= thr else 0
    return d

def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n) * 2
    return min(1.0, p)

for bench, runs_n in [('robospatial', 3), ('blinkdepth', 3)]:
    R = {t: [load(f'/workspace/exp/p0/{t}/run{i}/{bench}/0.jsonl') for i in range(1, runs_n + 1)] for t in TAGS}
    idx = sorted(R['sft'][0])
    n = len(idx)
    print(f'\n===== {bench}  n={n}  ({runs_n} runs each) =====')
    K = {t: {i: sum(r[i] for r in R[t]) for i in idx} for t in TAGS}
    print(f'{"":14s} {"runs":>18s}  {"mean":>6s}  {"always correct":>4s} {"always wrong":>4s} {"flip":>4s}')
    for t in TAGS:
        c = [sum(r.values()) for r in R[t]]
        a = sum(1 for i in idx if K[t][i] == runs_n)
        z = sum(1 for i in idx if K[t][i] == 0)
        f = n - a - z
        print(f'{NAME[t]:14s} {str(c):>18s}  {sum(c)/runs_n:6.1f}  {a:4d} {z:4d} {f:4d}')
    print('  --- per-sample comparison (only samples stable on both sides: 3/3 or 0/3) ---')
    for a, b in [('sft','p4'), ('sft','cp'), ('p4','cp')]:
        core = [i for i in idx if K[a][i] in (0, runs_n) and K[b][i] in (0, runs_n)]
        both = sum(1 for i in core if K[a][i] and K[b][i])
        onlyA = sum(1 for i in core if K[a][i] and not K[b][i])
        onlyB = sum(1 for i in core if not K[a][i] and K[b][i])
        neither = len(core) - both - onlyA - onlyB
        p = mcnemar(onlyA, onlyB)
        print(f'  {NAME[a]} -> {NAME[b]}: stable core {len(core)}/{n}  '
              f'both correct {both} both wrong {neither} | only former correct {onlyA} only latter correct {onlyB} '
              f'| net {onlyB-onlyA:+d}  McNemar p={p:.3f}')

# RefSpatial: 1 run each
print('\n===== RefSpatial three items (1 run each) =====')
for bench in ['reflocation', 'refplacement', 'refunseen']:
    row = []
    D = {}
    for t in TAGS:
        d = load(f'/workspace/exp/p0/{t}/ref1/{bench}/0.jsonl')
        D[t] = d
        row.append(f'{NAME[t]} {sum(d.values())}/{len(d)}')
    print(f'{bench:14s} ' + '   '.join(row))
    for a, b in [('sft','p4'), ('sft','cp'), ('p4','cp')]:
        dis = sum(1 for i in D[a] if D[a][i] != D[b][i])
        if dis:
            print(f'    {a}->{b} disagreements {dis}')
tot = {t: sum(sum(load(f'/workspace/exp/p0/{t}/ref1/{b}/0.jsonl').values()) for b in ['reflocation','refplacement','refunseen']) for t in TAGS}
print('RefSpatial three items total /277: ' + '  '.join(f'{NAME[t]} {tot[t]}' for t in TAGS))
