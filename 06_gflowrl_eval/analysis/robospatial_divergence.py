#!/usr/bin/env python3
"""robospatial 上 P4 与 P7 的分歧样本长什么样。"""
import json, os, collections
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root
L = lambda p: {r["sample_id"]: r for r in (json.loads(l) for l in open(p))}
a  = L(f"{ROOT}/01_official_checkpoint_eval/p4/parsed/robospatial.jsonl")
c  = L(f"{ROOT}/06_gflowrl_eval/parsed/robospatial.jsonl")
c2 = L(f"{ROOT}/06_gflowrl_eval/parsed_runC/robospatial.jsonl")
vqa = lambda r: len(str(r.get("gt","")).strip()) < 8

p4ok  = {i for i in a if a[i]["correct"]}
p7ok  = {i for i in c if c[i]["correct"] and c2[i]["correct"]}      # 两次都对
p7bad = {i for i in c if not c[i]["correct"] and not c2[i]["correct"]}  # 两次都错

lost = sorted(p4ok & p7bad)   # P4 对、P7 两次都错
gain = sorted(p7ok - p4ok)    # P7 两次都对、P4 错
print(f"P4 对 {len(p4ok)} · P7 恒对 {len(p7ok)} · P7 恒错 {len(p7bad)}")
print(f"丢掉(P4对/P7两次都错) {len(lost)}   捡到(P7恒对/P4错) {len(gain)}   净 {len(gain)-len(lost)}")
for name, ids in (("丢掉", lost), ("捡到", gain)):
    nv = sum(1 for i in ids if vqa(a[i]))
    print(f"  {name}: VQA {nv} · Vacant {len(ids)-nv}")
    sig = collections.Counter(c[i]["chain_signature"] for i in ids)
    print(f"        P7 链路: {dict(sig)}")

print()
print("=== Yes/No 方向(VQA 分歧样本上,两边各答了什么)")
import re
def ans(r):
    t = str(r.get("raw_answer","")).strip().lower()
    return "yes" if "yes" in t else ("no" if "no" in t else "?")
for name, ids in (("丢掉", lost), ("捡到", gain)):
    v = [i for i in ids if vqa(a[i])]
    print(f"  {name}: " + str(collections.Counter((str(a[i]['gt']).strip(), ans(a[i]), ans(c[i])) for i in v)))

print()
print("=== Vacant 错题离凸包有多远(P7 恒错的 Vacant 样本 vs P4 对的)")
print("  (仅统计能解析出预测点的样本)")
def pt(r):
    try:
        v = eval(str(r.get("raw_answer","")))
        if isinstance(v, list) and v and isinstance(v[0], (list, tuple)):
            return tuple(v[0])
    except Exception:
        pass
    return None
vac_lost = [i for i in lost if not vqa(a[i])]
print(f"  Vacant 丢掉 {len(vac_lost)} 个;其中 P4/P7 预测点都可解析的 "
      f"{sum(1 for i in vac_lost if pt(a[i]) and pt(c[i]))}")

print()
print("=== 未被调用的工具(robospatial 全程)")
for tag, d in (("P4", a), ("P7", c)):
    used = collections.Counter()
    for r in d.values():
        for t, n in (r.get("tool_stats") or {}).items(): used[t] += n
    print(f"  {tag}: {dict(used)}")
