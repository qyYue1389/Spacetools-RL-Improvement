#!/usr/bin/env python3
"""What the samples where P4 and P7 diverge on robospatial look like."""
import json, os, collections
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root
L = lambda p: {r["sample_id"]: r for r in (json.loads(l) for l in open(p))}
a  = L(f"{ROOT}/01_official_checkpoint_eval/p4/parsed/robospatial.jsonl")
c  = L(f"{ROOT}/06_gflowrl_eval/parsed/robospatial.jsonl")
c2 = L(f"{ROOT}/06_gflowrl_eval/parsed_runC/robospatial.jsonl")
vqa = lambda r: len(str(r.get("gt","")).strip()) < 8

p4ok  = {i for i in a if a[i]["correct"]}
p7ok  = {i for i in c if c[i]["correct"] and c2[i]["correct"]}      # correct in both runs
p7bad = {i for i in c if not c[i]["correct"] and not c2[i]["correct"]}  # wrong in both runs

lost = sorted(p4ok & p7bad)   # P4 correct, P7 wrong in both runs
gain = sorted(p7ok - p4ok)    # P7 correct in both runs, P4 wrong
print(f"P4 correct {len(p4ok)} · P7 always correct {len(p7ok)} · P7 always wrong {len(p7bad)}")
print(f"lost (P4 correct/P7 wrong in both runs) {len(lost)}   gained (P7 always correct/P4 wrong) {len(gain)}   net {len(gain)-len(lost)}")
for name, ids in (("lost", lost), ("gained", gain)):
    nv = sum(1 for i in ids if vqa(a[i]))
    print(f"  {name}: VQA {nv} · Vacant {len(ids)-nv}")
    sig = collections.Counter(c[i]["chain_signature"] for i in ids)
    print(f"        P7 chain: {dict(sig)}")

print()
print("=== Yes/No direction (on the diverging VQA samples, what each side answered)")
import re
def ans(r):
    t = str(r.get("raw_answer","")).strip().lower()
    return "yes" if "yes" in t else ("no" if "no" in t else "?")
for name, ids in (("lost", lost), ("gained", gain)):
    v = [i for i in ids if vqa(a[i])]
    print(f"  {name}: " + str(collections.Counter((str(a[i]['gt']).strip(), ans(a[i]), ans(c[i])) for i in v)))

print()
print("=== How far the Vacant wrong answers are from the convex hull (Vacant samples P7 always gets wrong vs those P4 gets right)")
print("  (only samples whose predicted point can be parsed)")
def pt(r):
    try:
        v = eval(str(r.get("raw_answer","")))
        if isinstance(v, list) and v and isinstance(v[0], (list, tuple)):
            return tuple(v[0])
    except Exception:
        pass
    return None
vac_lost = [i for i in lost if not vqa(a[i])]
print(f"  Vacant lost {len(vac_lost)}; of which P4/P7 predicted points are both parseable: "
      f"{sum(1 for i in vac_lost if pt(a[i]) and pt(c[i]))}")

print()
print("=== Tools never called (robospatial, whole run)")
for tag, d in (("P4", a), ("P7", c)):
    used = collections.Counter()
    for r in d.values():
        for t, n in (r.get("tool_stats") or {}).items(): used[t] += n
    print(f"  {tag}: {dict(used)}")
