#!/usr/bin/env python3
"""Is the P4 ↔ P7 difference on each benchmark statistically significant.
Binary scoring: McNemar exact test (two-sided binomial test on the discordant pairs).
Continuous scoring (boppose/bopgrasp): sign test on paired differences + Wilcoxon (if scipy is available)."""
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
    """McNemar exact: n=b+c coin flips, two-sided probability of seeing min(b,c) or more extreme"""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k+1)) / 2**n
    return min(1.0, 2*p)

print(f"{'benchmark':14s} {'n':>4s} {'P4 ok':>5s} {'P7 ok':>5s} {'diff':>4s} "
      f"{'P4only':>5s} {'P7only':>5s} {'McNemar p':>10s}  verdict")
for b in B:
    a, c = L(P4,b), L(P7,b)
    ids = sorted(set(a) & set(c))
    n4 = sum(1 for i in ids if a[i]["correct"]); n7 = sum(1 for i in ids if c[i]["correct"])
    only4 = sum(1 for i in ids if a[i]["correct"] and not c[i]["correct"])
    only7 = sum(1 for i in ids if c[i]["correct"] and not a[i]["correct"])
    p = binom_two_sided(only4, only7)
    verdict = "significant" if p < 0.05 else ("identical" if only4==0 and only7==0 else "not significant")
    print(f"{b:14s} {len(ids):4d} {n4:5d} {n7:5d} {n7-n4:+4d} {only4:5d} {only7:5d} {p:10.3f}  {verdict}")

print()
print("robospatial, using the two P7 runs to suppress noise (P7 right in both runs vs P4 right):")
a = L(P4,"robospatial"); c = L(P7,"robospatial"); c2 = L(P7C,"robospatial")
lost = sum(1 for i in a if a[i]["correct"] and not c[i]["correct"] and not c2[i]["correct"])
gain = sum(1 for i in a if not a[i]["correct"] and c[i]["correct"] and c2[i]["correct"])
print(f"  P4 right/P7 always wrong {lost} · P4 wrong/P7 always right {gain} · McNemar p = {binom_two_sided(lost,gain):.3f}")

print()
print("Paired comparison of continuous scores (sign test):")
for b in ["boppose","bopgrasp"]:
    a, c = L(P4,b), L(P7,b)
    ids = sorted(set(a) & set(c))
    d = [c[i]["score"] - a[i]["score"] for i in ids]
    pos = sum(1 for x in d if x > 1e-12); neg = sum(1 for x in d if x < -1e-12)
    tie = len(d) - pos - neg
    mean = sum(d)/len(d)
    med = sorted(x for x in d if abs(x) > 1e-12)
    med = med[len(med)//2] if med else 0.0
    print(f"  {b:9s} P4 mean {sum(a[i]['score'] for i in ids)/len(ids):.4f}  "
          f"P7 mean {sum(c[i]['score'] for i in ids)/len(ids):.4f}  diff {mean:+.4f}")
    print(f"            P7 higher {pos} · P4 higher {neg} · bit-for-bit identical {tie} · "
          f"sign test p = {binom_two_sided(neg,pos):.3f}")
