#!/usr/bin/env python3
"""P4 ↔ P7 每个 benchmark 的差异有没有统计意义。
二元判分:McNemar 精确检验(对不一致对做双侧二项检验)。
连续判分(boppose/bopgrasp):配对差的符号检验 + Wilcoxon(如果 scipy 在)。"""
import json, os, math
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root
from math import comb
L = lambda d,b: {r["sample_id"]: r for r in (json.loads(l) for l in open(f"{d}/{b}.jsonl"))}
P4 = f"{ROOT}/01_official_checkpoint_eval/p4/parsed"
P7 = f"{ROOT}/06_gflowrl_eval/parsed"
P7C = f"{ROOT}/06_gflowrl_eval/parsed_runC"
B = ["robospatial","reflocation","refplacement","refunseen","blinkdepth",
     "cvb2drelation","cvb3ddepth","boppose","bopgrasp"]

def binom_two_sided(b, c):
    """McNemar 精确:n=b+c 次投币,看到 min(b,c) 或更极端的双侧概率"""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k+1)) / 2**n
    return min(1.0, 2*p)

print(f"{'benchmark':14s} {'n':>4s} {'P4对':>5s} {'P7对':>5s} {'差':>4s} "
      f"{'仅P4':>5s} {'仅P7':>5s} {'McNemar p':>10s}  结论")
for b in B:
    a, c = L(P4,b), L(P7,b)
    ids = sorted(set(a) & set(c))
    n4 = sum(1 for i in ids if a[i]["correct"]); n7 = sum(1 for i in ids if c[i]["correct"])
    only4 = sum(1 for i in ids if a[i]["correct"] and not c[i]["correct"])
    only7 = sum(1 for i in ids if c[i]["correct"] and not a[i]["correct"])
    p = binom_two_sided(only4, only7)
    verdict = "显著" if p < 0.05 else ("完全一致" if only4==0 and only7==0 else "不显著")
    print(f"{b:14s} {len(ids):4d} {n4:5d} {n7:5d} {n7-n4:+4d} {only4:5d} {only7:5d} {p:10.3f}  {verdict}")

print()
print("robospatial 用 P7 两次运行压噪声(P7 两次都对 vs P4 对):")
a = L(P4,"robospatial"); c = L(P7,"robospatial"); c2 = L(P7C,"robospatial")
lost = sum(1 for i in a if a[i]["correct"] and not c[i]["correct"] and not c2[i]["correct"])
gain = sum(1 for i in a if not a[i]["correct"] and c[i]["correct"] and c2[i]["correct"])
print(f"  P4对/P7恒错 {lost} · P4错/P7恒对 {gain} · McNemar p = {binom_two_sided(lost,gain):.3f}")

print()
print("连续判分的配对比较(符号检验):")
for b in ["boppose","bopgrasp"]:
    a, c = L(P4,b), L(P7,b)
    ids = sorted(set(a) & set(c))
    d = [c[i]["score"] - a[i]["score"] for i in ids]
    pos = sum(1 for x in d if x > 1e-12); neg = sum(1 for x in d if x < -1e-12)
    tie = len(d) - pos - neg
    mean = sum(d)/len(d)
    med = sorted(x for x in d if abs(x) > 1e-12)
    med = med[len(med)//2] if med else 0.0
    print(f"  {b:9s} P4均值 {sum(a[i]['score'] for i in ids)/len(ids):.4f}  "
          f"P7均值 {sum(c[i]['score'] for i in ids)/len(ids):.4f}  差 {mean:+.4f}")
    print(f"            P7更高 {pos} · P4更高 {neg} · 逐位相同 {tie} · "
          f"符号检验 p = {binom_two_sided(neg,pos):.3f}")
