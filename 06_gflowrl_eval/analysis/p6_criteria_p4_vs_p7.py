#!/usr/bin/env python3
"""把 P6 的三条判据 + fit/Vacant 分析,同时跑在 P4 与 P7 的 parsed 上。

P6 原脚本:tools/p6/p6_split.py(判据 A/B)、tools/p6/p6_split3.py(题型与 Vacant)。
这里合并成一份、参数化 parsed 目录,好做逐样本对照。
"""
import json, os, re, math, statistics
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root
from collections import Counter

DIRS = [("P4(官方 ckpt)", f"{ROOT}/01_official_checkpoint_eval/p4/parsed"),
        ("P7(C' step85)", f"{ROOT}/06_gflowrl_eval/parsed")]
DET = re.compile(r"Detected (\d+) instance\(s\)[^:]*:\s*(\[.*?\])")
PT  = re.compile(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)")
PIX = re.compile(r"Pixel value at \(([\d.]+), ([\d.]+)\) is ([\d.eE+-]+)")
load = lambda d, b: [json.loads(l) for l in open(f"{d}/{b}.jsonl", encoding="utf-8")]

def qtype(r):
    q = (r.get("question") or "").lower()
    if "fit" in q: return "fit(要自由空间)"
    if "in front of" in q or "behind" in q: return "front/behind(要深度序)"
    return "relation(2D 可判定)"

def yesno(r):
    t = str(r.get("raw_answer") or "").strip().lower()
    return "yes" if "yes" in t else ("no" if "no" in t else "?")

print("=" * 78)
print("P6 的判据重新跑在 P4 与 P7 上")
print("=" * 78)

# ---------- 判据 A:深度题是否遵守「选测得更近的那个」 ----------
print("\n### 判据 A —— 深度题:模型是否逐字遵守「选测得深度更小的那个」")
for b in ["blinkdepth", "cvb3ddepth"]:
    print(f"  {b}")
    for tag, d in DIRS:
        rs = load(d, b); follow = viol = skip = 0
        for r in rs:
            dets = []; pix = {}
            for t in r["trajectory"]:
                for c, x in zip(t.get("tool_calls", []), t.get("tool_responses", [])):
                    if x is None: continue
                    txt = x.get("text") or ""
                    if c["name"] in ("roborefer.detect_one", "vlm.detect_one"):
                        m = DET.search(txt)
                        if m:
                            p = PT.findall(m.group(2))
                            if p: dets.append((round(float(p[0][0]), 3), round(float(p[0][1]), 3)))
                    if c["name"] == "vision_ops.index_at":
                        m = PIX.search(txt)
                        if m: pix[(round(float(m.group(1)), 3), round(float(m.group(2)), 3))] = float(m.group(3))
            ans = (r["raw_answer"] or "").strip().upper()
            ans = ans[1] if ans.startswith("(") and len(ans) > 1 else (ans[0] if ans else "")
            if len(dets) < 2 or len(pix) < 2 or ans not in "AB": skip += 1; continue
            dA, dB = pix.get(dets[0]), pix.get(dets[1])
            if dA is None or dB is None: skip += 1; continue
            rule = "A" if dA < dB else "B"
            follow += (ans == rule); viol += (ans != rule)
        n = follow + viol
        print(f"    {tag:16s} 遵守 {follow:4d} 违反 {viol:3d} 无法判定 {skip:3d}"
              f"   遵守率 {100*follow/n if n else 0:.1f}%")

# ---------- 判据 B:pointing 是否原样透传工具输出 ----------
print("\n### 判据 B —— pointing 题:答案是不是工具输出的原样透传")
for b in ["reflocation", "refplacement", "refunseen"]:
    line = f"  {b:14s}"
    for tag, d in DIRS:
        rs = load(d, b); same = tot = 0
        for r in rs:
            det = None
            for t in r["trajectory"]:
                for x in t.get("tool_responses", []):
                    m = DET.search((x or {}).get("text") or "")
                    if m and det is None:
                        p = PT.findall(m.group(2))
                        if p: det = (round(float(p[0][0]), 3), round(float(p[0][1]), 3))
            p = PT.findall(r["raw_answer"] or "")
            if det is None or not p: continue
            tot += 1
            same += ((round(float(p[0][0]), 3), round(float(p[0][1]), 3)) == det)
        line += f"   {tag} 透传 {same}/{tot}"
    print(line)

# ---------- robospatial VQA 题型 ----------
print("\n### robospatial VQA 按题型(P6 §6.4 的三分法)")
print(f"  {'题型':24s} {'n':>4s}  {'P4 正确率':>10s} {'P7 正确率':>10s}   {'P4 答yes':>8s} {'P7 答yes':>8s}")
tabs = {}
for tag, d in DIRS:
    rs = load(d, "robospatial")
    vqa = [r for r in rs if str(r["gt"]).strip().lower() in ("yes", "no")]
    t = {}
    for r in vqa:
        k = qtype(r); e = t.setdefault(k, [0, 0, 0, 0])
        e[0] += 1; e[1] += bool(r["correct"]); e[2] += (yesno(r) == "yes")
        e[3] += (str(r["gt"]).strip().lower() == "yes")
    tabs[tag] = t
for k in sorted(tabs[DIRS[0][0]], key=lambda x: -tabs[DIRS[0][0]][x][0]):
    a = tabs[DIRS[0][0]][k]; c = tabs[DIRS[1][0]][k]
    print(f"  {k:24s} {a[0]:4d}  {100*a[1]/a[0]:9.1f}% {100*c[1]/c[0]:9.1f}%   "
          f"{a[2]:4d}/{a[0]:<4d} {c[2]:4d}/{c[0]:<4d}   (GT=yes {a[3]})")

# ---------- Vacant:透传 vs 改点 ----------
print("\n### robospatial Vacant:模型改动 roborefer 给的点时,改好还是改坏")
for tag, d in DIRS:
    rs = load(d, "robospatial")
    vac = [r for r in rs if str(r["gt"]).strip().startswith("[")]
    gtpts = lambda s: [(float(x), float(y)) for x, y in PT.findall(s)]
    mind = lambda p, G: min(math.hypot(p[0]-g[0], p[1]-g[1]) for g in G) if G else None
    mod = []; passthru = 0; passthru_ok = 0
    for r in vac:
        det = None
        for t in r["trajectory"]:
            for x in t.get("tool_responses", []):
                m = DET.search((x or {}).get("text") or "")
                if m and det is None:
                    p = PT.findall(m.group(2))
                    if p: det = (round(float(p[0][0]), 3), round(float(p[0][1]), 3))
        p = PT.findall(r["raw_answer"] or "")
        if det is None or not p: continue
        a = (round(float(p[0][0]), 3), round(float(p[0][1]), 3))
        if a == det:
            passthru += 1; passthru_ok += bool(r["correct"]); continue
        G = gtpts(str(r["gt"]))
        if not G: continue
        mod.append((mind(det, G), mind(a, G), r["correct"]))
    better = sum(1 for d0, d1, _ in mod if d1 < d0)
    worse  = sum(1 for d0, d1, _ in mod if d1 > d0)
    okmod  = sum(1 for *_, ok in mod if ok)
    print(f"  {tag:16s} 透传 {passthru:3d}(正确率 {100*passthru_ok/passthru if passthru else 0:.1f}%)  "
          f"改点 {len(mod):3d}(正确率 {100*okmod/len(mod) if mod else 0:.1f}%)  "
          f"改好 {better} 改坏 {worse}  "
          f"原点中位距 {statistics.median(d0 for d0,_,_ in mod):.4f} -> 改后 {statistics.median(d1 for _,d1,_ in mod):.4f}")

# ---------- 链路坍塌总览 ----------
print("\n### 工具编排坍塌度(P6 §7 第 1 条的那个量)")
print(f"  {'benchmark':14s} {'n':>4s}  {'P4 主链路':>9s} {'P7 主链路':>9s}  {'P4 种类':>7s} {'P7 种类':>7s}")
for b in ["robospatial", "reflocation", "refplacement", "refunseen",
          "blinkdepth", "cvb2drelation", "cvb3ddepth", "boppose", "bopgrasp"]:
    row = [b, None, None, None, None, None]
    for j, (tag, d) in enumerate(DIRS):
        rs = load(d, b)
        c = Counter(r["chain_signature"] for r in rs)
        row[1] = len(rs)
        row[2 + j] = 100.0 * c.most_common(1)[0][1] / len(rs)
        row[4 + j] = len(c)
    print(f"  {row[0]:14s} {row[1]:4d}  {row[2]:8.1f}% {row[3]:8.1f}%  {row[4]:7d} {row[5]:7d}")
