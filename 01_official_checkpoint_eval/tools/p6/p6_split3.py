import json, os, re, math, statistics
from collections import Counter
BASE = __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "p4", "parsed")
def load(b): return [json.loads(l) for l in open(f"{BASE}/{b}.jsonl",encoding="utf-8")]
DET = re.compile(r"Detected (\d+) instance\(s\)[^:]*:\s*(\[.*?\])")
PT  = re.compile(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)")

# ---------- 1. robospatial VQA 的题型 ----------
rs=load("robospatial")
vqa=[r for r in rs if str(r["gt"]).strip().lower() in ("yes","no")]
vac=[r for r in rs if str(r["gt"]).strip().startswith("[")]
pat=Counter()
for r in vqa:
    q=(r.get("question") or "").lower()
    if "fit" in q: pat["can X fit <rel> Y"]+=1
    elif "is there" in q: pat["is there ..."]+=1
    else: pat["其它"]+=1
print("### robospatial VQA 题型分布:", dict(pat))
rel=Counter()
for r in vqa:
    q=(r.get("question") or "").lower()
    for k in ["behind","in front of","left of","right of","on top of","under","above","below"]:
        if k in q: rel[k]+=1; break
print("    关系词:", dict(rel))
acc={}
for r in vqa:
    q=(r.get("question") or "").lower()
    k=next((k for k in ["behind","in front of","left of","right of","on top of","under","above","below"] if k in q),"?")
    d=acc.setdefault(k,[0,0]); d[0]+=1; d[1]+=bool(r["correct"])
print(f"\n    {'关系':<14}{'n':>5}{'正确率':>9}")
for k,(n,ok) in sorted(acc.items(), key=lambda kv:-kv[1][0]):
    print(f"    {k:<14}{n:5}{100*ok/n:8.1f}%")

# ---------- 2. Vacant:模型改点改好了还是改坏了 ----------
print("\n### robospatial Vacant:模型改动点位时,改好还是改坏?")
def gtpts(s): return [(float(a),float(b)) for a,b in PT.findall(s)]
def mind(p, G): return min(math.hypot(p[0]-g[0], p[1]-g[1]) for g in G) if G else None
mod=[]
for r in vac:
    det=None
    for t in r["trajectory"]:
        for x in t["tool_responses"]:
            m=DET.search(x["text"] or "")
            if m and det is None:
                p=PT.findall(m.group(2))
                if p: det=(round(float(p[0][0]),3), round(float(p[0][1]),3))
    p=PT.findall(r["raw_answer"] or "")
    if det is None or not p: continue
    a=(round(float(p[0][0]),3), round(float(p[0][1]),3))
    if a==det: continue
    G=gtpts(str(r["gt"]))
    if not G: continue
    mod.append((mind(det,G), mind(a,G), r["correct"]))
better=sum(1 for d0,d1,_ in mod if d1<d0); worse=sum(1 for d0,d1,_ in mod if d1>d0)
print(f"   模型改动的样本 {len(mod)} 个")
print(f"   改后离 GT 更近(改好) {better}   改后更远(改坏) {worse}")
print(f"   roborefer 原点到 GT 的中位距离 {statistics.median(d0 for d0,_,_ in mod):.4f}")
print(f"   模型改后    到 GT 的中位距离 {statistics.median(d1 for _,d1,_ in mod):.4f}")
print(f"   改动样本正确率 {100*sum(1 for *_ ,ok in mod if ok)/len(mod):.1f}%   (透传样本 {100*(101-45)/101:.1f}%)")

# ---------- 3. cvb2drelation:关系规则一致性 ----------
print("\n### cvb2drelation:模型是否遵守坐标比较规则?")
c2=load("cvb2drelation")
CH = re.compile(r"\(A\)\s*(.+?)\s*\(B\)\s*(.+?)\s*$", re.S)
follow=viol=skip=0; fw=vw=0
for r in c2:
    q=(r.get("question") or "")
    m=CH.search(q.replace("\n"," "))
    dets=[]
    for t in r["trajectory"]:
        for c,x in zip(t["tool_calls"], t["tool_responses"]):
            if c["name"].endswith("detect_one"):
                mm=DET.search(x["text"] or "")
                if mm:
                    p=PT.findall(mm.group(2))
                    if p: dets.append((float(p[0][0]), float(p[0][1])))
    ans=(r["raw_answer"] or "").strip().upper()
    ans=next((ch for ch in ans if ch in "AB"), "")
    if not m or len(dets)<2 or ans not in "AB": skip+=1; continue
    chA,chB=m.group(1).lower(), m.group(2).lower()
    (x1,y1),(x2,y2)=dets[0],dets[1]     # 第一个是主体,第二个是参照
    def truth(ch):
        if "left"  in ch: return x1 <  x2
        if "right" in ch: return x1 >  x2
        if "above" in ch or "top" in ch: return y1 < y2   # 图像 y 轴向下
        if "below" in ch or "under" in ch: return y1 > y2
        return None
    tA,tB=truth(chA),truth(chB)
    if tA is None or tB is None or tA==tB: skip+=1; continue
    rule = "A" if tA else "B"
    if ans==rule: follow+=1; fw += (not r["correct"])
    else:         viol+=1;   vw += (not r["correct"])
n=follow+viol
print(f"   可判定 {n}/{len(c2)}(跳过 {skip})")
print(f"   遵守规则 {follow:4} ({100*follow/n:5.1f}%)  其中答错 {fw:3}  ← 工具错(点错了)")
print(f"   违反规则 {viol:4} ({100*viol/n:5.1f}%)  其中答错 {vw:3}  ← 推理错")
