import json
from math import comb
TAGS=['sft','p4','cp']
NAME={'sft':'SFT','p4':'P4','cp':"C'"}
def load(p,thr=0.5):
    return {json.loads(l)['index']:(1 if json.loads(l)['score']>=thr else 0) for l in open(p)}
def mcn(b,c):
    n=b+c
    if n==0: return 1.0
    k=min(b,c)
    return min(1.0, sum(comb(n,i) for i in range(k+1))*(0.5**n)*2)
for bench in ['robospatial','blinkdepth']:
    R={t:[load(f'/workspace/exp/p0/{t}/run{i}/{bench}/0.jsonl') for i in (1,2,3)] for t in TAGS}
    idx=sorted(R['sft'][0]); n=len(idx)
    print(f'\n== {bench} n={n} · three-run paired per-sample (run i vs run i), disagreements pooled ==')
    for a,b in [('sft','p4'),('sft','cp'),('p4','cp')]:
        B=C=0
        per=[]
        for j in range(3):
            bb=sum(1 for i in idx if R[a][j][i] and not R[b][j][i])
            cc=sum(1 for i in idx if not R[a][j][i] and R[b][j][i])
            per.append((bb,cc)); B+=bb; C+=cc
        print(f'  {NAME[a]}->{NAME[b]}: per run (only former correct, only latter correct)={per}  pooled {B} vs {C}  net {C-B:+d}/3 runs  p={mcn(B,C):.4f}')
    # spread
    for t in TAGS:
        k={i:sum(r[i] for r in R[t]) for i in idx}
        flip=sum(1 for v in k.values() if 0<v<3)
        c=[sum(r.values()) for r in R[t]]
        sd=(flip/3)**0.5/2
        print(f'  {NAME[t]}: flip band {flip} -> sd of the three-run mean ≈ {sd:.2f} questions (2σ ≈ {2*sd:.1f} questions)')
