#!/bin/bash
# Material for the report: VQA confusion matrix (P5 found a systematic "should
# answer no" bias with the official ckpt -- does ours have it?), turn/call
# distributions, and full verbatim trajectories of representative samples.
exec >>/tmp/material.log 2>&1
echo "=== START $(date -Is)"
sudo -n env PYTHONPATH=/opt/spacetools/SpaceTools-RL \
  /opt/conda-st/envs/spacetools-rl/bin/python - <<'PY'
import json, re, collections
def load(p):
    return [json.loads(l) for l in open(p)]
RS = load("/workspace/eval_out/robospatial/0.jsonl")
def gts_pts(r):
    return re.findall(r"\(\s*(-?[0-9.]+)\s*,\s*(-?[0-9.]+)\s*\)", r.get("gts") or "")
def vac(r): return bool(gts_pts(r))
ANS = re.compile(r"<answer>(.*?)</answer>", re.S)
def ans(r):
    m = ANS.search(r.get("output") or "")
    return (m.group(1).strip() if m else "")

print("### VQA 混淆矩阵(gt × 预测)")
cm = collections.Counter()
for r in RS:
    if vac(r): continue
    g = (r.get("gts") or "").strip().lower()
    a = ans(r).lower()
    gt = "yes" if "yes" in g else ("no" if "no" in g else "?")
    pd = "yes" if "yes" in a else ("no" if "no" in a else "?")
    cm[(gt, pd)] += 1
tot_gt = collections.Counter()
for (g, p), n in cm.items(): tot_gt[g] += n
print("        pred_no  pred_yes   |  行准确率")
for g in ("no", "yes"):
    n_no, n_yes = cm[(g, "no")], cm[(g, "yes")]
    right = n_no if g == "no" else n_yes
    print("  gt_%-4s %6d %8d   |  %5.2f%%  (n=%d)" % (g, n_no, n_yes, 100*right/max(tot_gt[g],1), tot_gt[g]))
print("  预测分布  yes=%d no=%d   真值分布  yes=%d no=%d" % (
    cm[("yes","yes")]+cm[("no","yes")], cm[("yes","no")]+cm[("no","no")],
    tot_gt["yes"], tot_gt["no"]))

print()
print("### 每样本工具调用次数分布")
for b in ["robospatial","reflocation","refplacement","refunseen"]:
    rows = load(f"/workspace/eval_out/{b}/0.jsonl")
    c = collections.Counter((r.get("output") or "").count("<tool_call>") for r in rows)
    tot = sum(k*v for k,v in c.items())
    print("  %-13s %s   平均 %.3f 次/样本" % (b, dict(sorted(c.items())), tot/len(rows)))

def dump(tag, r, limit=1500):
    print()
    print("#"*8, tag, "index=%s acc=%s" % (r.get("index"), r.get("acc")))
    print("--- gts:", " ".join((r.get("gts") or "").split())[:260])
    o = r.get("output") or ""
    o = o.replace("<|im_end|>", "").replace("<|im_start|>", "\n[").replace("\n\n", "\n")
    print("--- output:")
    print(o[:limit])

print()
print("### 代表性轨迹")
RL_ = load("/workspace/eval_out/reflocation/0.jsonl")
dump("RefSpatial-Location 答对(单工具单轮)", RL_[0])
for r in RS:
    if not vac(r) and (r.get("acc") or 0) >= 1 and (r.get("output") or "").count("<tool_call>") == 2:
        dump("RoboSpatial-VQA 答对(两次 roborefer)", r); break
for r in RS:
    if not vac(r) and (r.get("acc") or 0) < 1:
        dump("RoboSpatial-VQA 答错", r); break
for r in RS:
    if vac(r) and (r.get("acc") or 0) < 1:
        dump("RoboSpatial-Vacant 答错", r); break
for r in RS:
    if (r.get("output") or "").count("<tool_call>") >= 5:
        dump("RoboSpatial 最长链(5 次调用,跨三个 conda 环境)", r, 2000); break
PY
echo "=== DONE $(date -Is)"
