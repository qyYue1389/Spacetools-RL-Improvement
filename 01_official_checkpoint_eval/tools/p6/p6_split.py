import json, os, re
from collections import Counter
BASE = __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "p4", "parsed")
def load(b): return [json.loads(l) for l in open(f"{BASE}/{b}.jsonl",encoding="utf-8")]
PIX = re.compile(r"Pixel value at \(([\d.]+), ([\d.]+)\) is ([\d.eE+-]+)")
DET = re.compile(r"Detected (\d+) instance\(s\)[^:]*:\s*(\[.*?\])")
PT  = re.compile(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)")

print("### 检验 1:深度题 —— 模型是否总是选「测得深度更小」的那个?\n")
for b in ["blinkdepth","cvb3ddepth"]:
    rs=load(b); follow=viol=skip=0; foll_wrong=viol_wrong=0
    for r in rs:
        dets=[]; pix={}
        for t in r["trajectory"]:
            for c,x in zip(t["tool_calls"], t["tool_responses"]+[None]*99):
                if x is None: break
                if c["name"] in ("roborefer.detect_one","vlm.detect_one"):
                    m=DET.search(x["text"] or "")
                    if m:
                        p=PT.findall(m.group(2))
                        if p: dets.append((round(float(p[0][0]),3), round(float(p[0][1]),3)))
                if c["name"]=="vision_ops.index_at":
                    m=PIX.search(x["text"] or "")
                    if m: pix[(round(float(m.group(1)),3), round(float(m.group(2)),3))]=float(m.group(3))
        ans=(r["raw_answer"] or "").strip().upper()
        ans=ans[1] if ans.startswith("(") and len(ans)>1 else (ans[0] if ans else "")
        if len(dets)<2 or len(pix)<2 or ans not in "AB": skip+=1; continue
        dA=pix.get(dets[0]); dB=pix.get(dets[1])
        if dA is None or dB is None: skip+=1; continue
        rule = "A" if dA < dB else "B"
        if ans==rule:
            follow+=1
            if not r["correct"]: foll_wrong+=1
        else:
            viol+=1
            if not r["correct"]: viol_wrong+=1
    n=follow+viol
    print(f"{b:14} 可判定 {n}/{len(rs)}(跳过 {skip})")
    print(f"   遵守规则  {follow:4} ({100*follow/n:5.1f}%)  其中答错 {foll_wrong:3}  ← 工具错")
    print(f"   违反规则  {viol:4} ({100*viol/n:5.1f}%)  其中答错 {viol_wrong:3}  ← 推理错")
    print()

print("### 检验 2:pointing 题 —— 模型的答案是不是 roborefer 的原样透传?\n")
for b in ["reflocation","refplacement","refunseen"]:
    rs=load(b); same=diff=skip=0; same_wrong=diff_wrong=0
    for r in rs:
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
            same+=1
            if not r["correct"]: same_wrong+=1
        else:
            diff+=1
            if not r["correct"]: diff_wrong+=1
    n=same+diff
    print(f"{b:14} 可判定 {n}/{len(rs)}(跳过 {skip})")
    print(f"   原样透传  {same:4} ({100*same/n:5.1f}%)  其中答错 {same_wrong:3}  ← 工具错")
    print(f"   自行改动  {diff:4} ({100*diff/n:5.1f}%)  其中答错 {diff_wrong:3}")
    print()

print("### 检验 3:robospatial 的 yes/no 题 —— roborefer 找到东西了吗?\n")
rs=load("robospatial")
vqa=[r for r in rs if str(r["gt"]).strip().lower() in ("yes","no")]
print(f"VQA 样本 {len(vqa)}")
rows=[]
for r in vqa:
    cnt=[]
    for t in r["trajectory"]:
        for x in t["tool_responses"]:
            m=DET.search(x["text"] or "")
            if m: cnt.append(int(m.group(1)))
    gt=str(r["gt"]).strip().lower()
    ans=(r["raw_answer"] or "").strip().lower()
    ans="yes" if "yes" in ans else ("no" if "no" in ans else "?")
    rows.append((gt, ans, tuple(cnt), r["correct"]))
print(f"{'GT':>4} {'检测数模式':>12} {'n':>5} {'答yes':>7} {'答no':>6} {'正确率':>8}")
agg={}
for gt,ans,cnt,ok in rows:
    key=(gt, "有0" if 0 in cnt else "全>0")
    d=agg.setdefault(key,[0,0,0,0]); d[0]+=1
    d[1]+= (ans=="yes"); d[2]+= (ans=="no"); d[3]+= bool(ok)
for (gt,pat),(n,y,no,ok) in sorted(agg.items()):
    print(f"{gt:>4} {pat:>12} {n:5} {y:7} {no:6} {100*ok/n:7.1f}%")
