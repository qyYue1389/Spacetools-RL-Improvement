import json, os, re
from collections import Counter
BASE = __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "p4", "parsed")
def load(b): return [json.loads(l) for l in open(f"{BASE}/{b}.jsonl",encoding="utf-8")]
DET = re.compile(r"Detected (\d+) instance\(s\)[^:]*:\s*(\[.*?\])")
PT  = re.compile(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)")

# ---------- robospatial Vacant:透传检验 ----------
rs=load("robospatial")
vac=[r for r in rs if str(r["gt"]).strip().startswith("[")]
vqa=[r for r in rs if str(r["gt"]).strip().lower() in ("yes","no")]
print(f"robospatial 拆分: VQA {len(vqa)} / Vacant {len(vac)}")
same=diff=skip=0; same_wrong=diff_wrong=0
for r in vac:
    det=None
    for t in r["trajectory"]:
        for x in t["tool_responses"]:
            m=DET.search(x["text"] or "")
            if m and det is None:
                p=PT.findall(m.group(2))
                if p: det=(round(float(p[0][0]),3), round(float(p[0][1]),3))
    p=PT.findall(r["raw_answer"] or "")
    if det is None or not p: skip+=1; continue
    a=(round(float(p[0][0]),3), round(float(p[0][1]),3))
    if a==det:
        same+=1; same_wrong += (not r["correct"])
    else:
        diff+=1; diff_wrong += (not r["correct"])
n=same+diff
print(f"\n### robospatial Vacant 透传检验  可判定 {n}/{len(vac)}(跳过 {skip})")
print(f"   原样透传 {same:4} ({100*same/n:5.1f}%)  其中答错 {same_wrong:3}  ← 工具错")
print(f"   自行改动 {diff:4} ({100*diff/n:5.1f}%)  其中答错 {diff_wrong:3}")

# ---------- cvb2drelation 题目形态 ----------
print("\n### cvb2drelation 题目样例")
c2=load("cvb2drelation")
for r in c2[:3]:
    print(f"  [{'✓' if r['correct'] else '✗'}] {(r.get('question') or '')[:170].replace(chr(10),' ')}")
    print(f"      GT={r['gt']}  A={str(r['raw_answer'])[:40]}")
qs=[(r.get("question") or "") for r in c2]
kinds=Counter()
for q in qs:
    ql=q.lower()
    for k in ["left","right","above","below","under","on top","closer","behind","front"]:
        if k in ql: kinds[k]+=1
print("  关键词出现次数:", dict(kinds))

print("\n### robospatial VQA 题目样例")
for r in vqa[:3]:
    print(f"  [{'✓' if r['correct'] else '✗'}] GT={r['gt']:>3}  {(r.get('question') or '')[:170].replace(chr(10),' ')}")
