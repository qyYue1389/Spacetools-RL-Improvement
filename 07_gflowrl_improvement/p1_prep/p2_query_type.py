#!/usr/bin/env python3
"""P2 开机前调查:三臂在 robospatial Vacant 上给 roborefer 的查询写法(obj_name)。
用法: python p2_query_type.py [P0 dumps 目录,默认 ../p0_same_machine_reeval/dumps]
target = 查询里带了位置/空位描述("point close to and in front of the cup");object = 只问锚物体("cup")。"""
import sys, json, re, collections
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import p1_monitor as m
import os
DUMPS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'p0_same_machine_reeval', 'dumps')
CALL = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.S)
TGT = re.compile(r'\b(point|vacant|space|spot|area|place|placing|free|empty|left of|right of|in front of|behind|above|below|next to)\b', re.I)

def queries(out):
    qs = []
    for c in CALL.findall(out):
        try:
            qs.append(json.loads(c).get('arguments', {}).get('obj_name', '') or '')
        except Exception:
            pass
    return qs

D, Q = {}, {}
for arm in ('sft', 'p4', 'cp'):
    st = collections.defaultdict(lambda: [0, 0, 0, 0])
    for i in (1, 2, 3):
        for l in open(f'{DUMPS}/{arm}_run{i}_robospatial.jsonl'):
            r = json.loads(l)
            if m.category(r) is None:
                continue
            k, last, ans = m.classify(r['output'])
            qs = queries(r['output'])
            t = 'target' if any(TGT.search(q) for q in qs) else ('object' if qs else 'none')
            hit = bool(last) and m.inside(last[0], m.hull(m.pts(str(r['gts']))))
            s = st[t]; s[0] += 1; s[1] += hit; s[2] += (k == 'modified'); s[3] += float(r['score']) >= .5
            D.setdefault(r['index'], {})[(arm, i)] = hit
            Q.setdefault((r['index'], i), {})[arm] = tuple(qs)
    for t, v in sorted(st.items()):
        print(f'{arm:4s} {t:7s} 每次 {v[0]/3:5.1f} 题  工具点中 {100*v[1]/v[0]:3.0f}%  改点 {100*v[2]/v[0]:3.0f}%  答对 {100*v[3]/v[0]:3.0f}%')
n = len(Q)
print(f"查询逐字同 SFT:C′ {100*sum(v['cp']==v['sft'] for v in Q.values())/n:.1f}%  P4 {100*sum(v['p4']==v['sft'] for v in Q.values())/n:.1f}%")
H = lambda a, i: sum(D[i][(a, j)] for j in (1, 2, 3))
print('P4 工具点 3/3 中、C′ 0/3 的题:', sorted(i for i in D if H('p4', i) == 3 and H('cp', i) == 0))
