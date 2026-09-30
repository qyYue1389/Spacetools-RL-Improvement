#!/bin/bash
PY=/opt/conda-st/envs/spacetools-rl/bin/python
cd /workspace/p7repo
for t in sft p4 cp; do
  for i in 1 2 3; do
    d=/workspace/parsed/full_${t}_run${i}_rs
    [ -f "$d/robospatial.jsonl" ] || "$PY" parse_dump.py --emit "$d" /workspace/exp/p0/$t/run$i/robospatial/0.jsonl >/dev/null 2>&1
  done
done
"$PY" - <<'PYEOF'
import json, re
NUM = re.compile(r'\(\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\)')
TAGS=['sft','p4','cp']; NAME={'sft':'SFT 起点','p4':'P4 官方 GRPO','cp':"C' step85"}

def pts(s):
    return set((round(float(a),3), round(float(b),3)) for a,b in NUM.findall(s or ''))

def isvac(r):
    g=r['gt']
    return not (isinstance(g,str) and g.strip().lower() in ('yes','no'))

def classify(r):
    tool=set()
    for turn in r['trajectory']:
        for tr in turn.get('tool_responses') or []:
            tool |= pts(tr.get('text'))
    ans = pts(r['raw_answer'] if isinstance(r['raw_answer'],str) else str(r['raw_answer']))
    if not ans:            return 'no-answer', tool, ans
    if not tool:           return 'no-tool',   tool, ans
    if ans <= tool:        return 'passthrough', tool, ans
    return 'modified', tool, ans

def grid05(ans):
    return all(abs(v*20 - round(v*20)) < 1e-6 for p in ans for v in p) if ans else False

print(f'{"":14s} {"run":>4s} {"透传":>10s} {"改点":>10s} {"无工具":>7s} {"0.05网格":>8s}')
agg={}
for t in TAGS:
    rows=[]
    for i in (1,2,3):
        rs=[json.loads(l) for l in open(f'/workspace/parsed/full_{t}_run{i}_rs/robospatial.jsonl')]
        vac=[r for r in rs if isvac(r)]
        c={'passthrough':[0,0],'modified':[0,0],'no-tool':[0,0],'no-answer':[0,0]}
        g=0
        for r in vac:
            k,tool,ans = classify(r)
            c[k][0]+=1; c[k][1]+= 1 if r['correct'] else 0
            if k=='modified' and grid05(ans): g+=1
        def f(k): 
            n,ok=c[k]; return f'{n} ({100*ok/n:.1f}%)' if n else '0'
        print(f'{NAME[t]:14s} {i:>4d} {f("passthrough"):>14s} {f("modified"):>14s} {c["no-tool"][0]:>7d} {g:>8d}')
        rows.append(c)
    agg[t]=rows
print('\n--- 三次合计 ---')
for t in TAGS:
    tot={k:[sum(r[k][0] for r in agg[t]), sum(r[k][1] for r in agg[t])] for k in ('passthrough','modified','no-tool')}
    p,m = tot['passthrough'], tot['modified']
    print(f'{NAME[t]:14s} 透传 {p[0]/3:.1f}/次 正确率 {100*p[1]/max(p[0],1):.1f}%   '
          f'改点 {m[0]/3:.1f}/次 正确率 {100*m[1]/max(m[0],1):.1f}%   无工具 {tot["no-tool"][0]/3:.1f}')
PYEOF
