#!/usr/bin/env python3
"""P4(官方 ckpt, GRPO 之后) 与 P7(自训 SFT + GFlowRL C') 的逐样本对比。

只吃 parsed 记录,不需要 GPU。P4 用 run1(九个 benchmark 齐全),P7 用有效运行
(runA 的五个 key + runB 的四个)。样本按 sample_id 配对。
"""
import json, os, sys, collections
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root

P4 = f"{ROOT}/01_official_checkpoint_eval/p4/parsed"
P7 = f"{ROOT}/06_gflowrl_eval/parsed"
P7C = f"{ROOT}/06_gflowrl_eval/parsed_runC"
BENCH = ["robospatial", "reflocation", "refplacement", "refunseen",
         "blinkdepth", "cvb2drelation", "cvb3ddepth", "boppose", "bopgrasp"]

def load(d, b):
    p = f"{d}/{b}.jsonl"
    if not os.path.exists(p):
        return None
    return {r["sample_id"]: r for r in (json.loads(l) for l in open(p))}

def is_vqa(r):            # robospatial: Yes/No 题 vs Vacant 点题
    return len(str(r.get("gt", "")).strip()) < 8

def blame(r):
    """错题归因:按可验证的痕迹分类,顺序即优先级。"""
    if r.get("oom"):                       return "OOM"
    if r.get("tool_failures"):             return "工具失败"
    if r.get("truncated_generation"):      return "生成截断"
    if r.get("truncated_tool_response"):   return "工具响应截断"
    if r.get("hit_max_turns"):             return "顶轮数"
    if not r.get("parse_ok", True):        return "答案解析失败"
    if r.get("malformed_tool_calls"):      return "畸形 tool_call"
    if r.get("n_tool_calls", 0) == 0:      return "未调用工具"
    return "干净(推理/工具精度)"

def pct(a, b):
    return f"{100.0*a/b:6.2f}%" if b else "   n/a"

def topsig(rows, k=4):
    c = collections.Counter(r["chain_signature"] for r in rows.values())
    n = len(rows)
    return [(s, v, 100.0*v/n) for s, v in c.most_common(k)]

def toolcalls(rows):
    c = collections.Counter()
    for r in rows.values():
        for t, v in (r.get("tool_stats") or {}).items():
            c[t] += v
    return dict(c.most_common())

def varstats(rows):
    exposed = used = unused = phantom = 0
    n_with_vars = 0
    for r in rows.values():
        e = len(r.get("vars_exposed") or [])
        u = len(r.get("vars_used") or [])
        un = len(r.get("vars_unused") or [])
        ph = len(r.get("vars_phantom") or [])
        exposed += e; used += u; unused += un; phantom += ph
        if e: n_with_vars += 1
    return exposed, used, unused, phantom, n_with_vars

out = []
def P(s=""):
    out.append(s); print(s)

P("=" * 78)
P("P4(官方 ckpt · GRPO 之后)  ↔  P7(自训 SFT · GFlowRL C' step85)  逐样本对比")
P("P4 = p4/parsed(run1) · P7 = 有效运行的 parsed · 按 sample_id 配对")
P("=" * 78)

summary = []
for b in BENCH:
    a, c = load(P4, b), load(P7, b)
    if a is None or c is None:
        P(f"\n### {b}: 缺少 parsed({'P4' if a is None else 'P7'}),跳过"); continue
    common = sorted(set(a) & set(c))
    P("")
    P(f"### {b}   n(P4)={len(a)}  n(P7)={len(c)}  共同样本={len(common)}")

    # --- 正确率与配对
    ca = sum(1 for i in common if a[i]["correct"])
    cc = sum(1 for i in common if c[i]["correct"])
    both = sum(1 for i in common if a[i]["correct"] and c[i]["correct"])
    neither = sum(1 for i in common if not a[i]["correct"] and not c[i]["correct"])
    only4 = sum(1 for i in common if a[i]["correct"] and not c[i]["correct"])
    only7 = sum(1 for i in common if c[i]["correct"] and not a[i]["correct"])
    P(f"  正确率      P4 {ca:4d} {pct(ca,len(common))}   P7 {cc:4d} {pct(cc,len(common))}")
    P(f"  配对        都对 {both}  都错 {neither}  只有 P4 对 {only4}  只有 P7 对 {only7}")
    # 连续分(boppose/bopgrasp)另报均值
    sa = sum(a[i]["score"] for i in common) / len(common)
    sc = sum(c[i]["score"] for i in common) / len(common)
    P(f"  平均 score  P4 {sa:.4f}   P7 {sc:.4f}")

    # --- 工具链路签名
    P(f"  链路签名(P4)                                  链路签名(P7)")
    ta, tc = topsig(a), topsig(c)
    for k in range(max(len(ta), len(tc))):
        L = f"{ta[k][0][:34]:34s} {ta[k][1]:4d} {ta[k][2]:5.1f}%" if k < len(ta) else " " * 46
        Rr = f"{tc[k][0][:34]:34s} {tc[k][1]:4d} {tc[k][2]:5.1f}%" if k < len(tc) else ""
        P(f"    {L}  {Rr}")
    P(f"  主链路覆盖率  P4 {ta[0][2]:5.1f}%   P7 {tc[0][2]:5.1f}%"
      f"   签名种类 P4 {len(set(r['chain_signature'] for r in a.values()))}"
      f" P7 {len(set(r['chain_signature'] for r in c.values()))}")

    # --- 工具调用
    P(f"  工具调用    P4 {toolcalls(a)}")
    P(f"              P7 {toolcalls(c)}")
    na = sum(r["n_tool_calls"] for r in a.values()) / len(a)
    nc = sum(r["n_tool_calls"] for r in c.values()) / len(c)
    P(f"  调用次数/样本  P4 {na:.3f}   P7 {nc:.3f}")

    # --- 变量复用
    ea, ua, una, pha, nwa = varstats(a)
    ec, uc, unc, phc, nwc = varstats(c)
    P(f"  变量        P4 暴露 {ea:4d} 使用 {ua:4d} 未用 {una:4d} 幻觉 {pha:3d}  有变量的样本 {nwa}")
    P(f"              P7 暴露 {ec:4d} 使用 {uc:4d} 未用 {unc:4d} 幻觉 {phc:3d}  有变量的样本 {nwc}")

    # --- 错题归因
    wa = collections.Counter(blame(a[i]) for i in common if not a[i]["correct"])
    wc = collections.Counter(blame(c[i]) for i in common if not c[i]["correct"])
    keys = sorted(set(wa) | set(wc))
    P(f"  错题归因                       P4      P7")
    for k in keys:
        P(f"    {k:26s} {wa.get(k,0):5d}   {wc.get(k,0):5d}")

    summary.append((b, len(common), ca, cc, only4, only7, ta[0][2], tc[0][2], na, nc))

# --- robospatial 的 VQA/Vacant 细分 + P7 两次运行
P("")
P("=" * 78)
P("robospatial 细分")
a, c = load(P4, "robospatial"), load(P7, "robospatial")
c2 = load(P7C, "robospatial")
for name, sel in (("VQA", is_vqa), ("Vacant", lambda r: not is_vqa(r))):
    ids = [i for i in a if sel(a[i])]
    ca = sum(1 for i in ids if a[i]["correct"])
    cc = sum(1 for i in ids if c[i]["correct"])
    cc2 = sum(1 for i in ids if c2[i]["correct"]) if c2 else 0
    P(f"  {name:7s} n={len(ids):3d}   P4 {ca:3d} {pct(ca,len(ids))}   "
      f"P7-A {cc:3d} {pct(cc,len(ids))}   P7-C {cc2:3d} {pct(cc2,len(ids))}")
if c2:
    flip = [i for i in c if c[i]["correct"] != c2[i]["correct"]]
    stable_ok = [i for i in c if c[i]["correct"] and c2[i]["correct"]]
    P(f"  P7 两次:恒对 {len(stable_ok)} · 翻转 {len(flip)} · "
      f"其中 VQA {sum(1 for i in flip if is_vqa(c[i]))} / Vacant {sum(1 for i in flip if not is_vqa(c[i]))}")
    # P7 恒对集合 vs P4 正确集合
    p4ok = {i for i in a if a[i]["correct"]}
    P(f"  P7 恒对 ∩ P4 对 = {len(set(stable_ok) & p4ok)} · P7 恒对但 P4 错 = {len(set(stable_ok) - p4ok)} · "
      f"P4 对但 P7 两次都错 = {len(p4ok - {i for i in c if c[i]['correct'] or c2[i]['correct']})}")

# --- 总表
P("")
P("=" * 78)
P("总表")
P(f"{'benchmark':14s} {'n':>4s} {'P4对':>5s} {'P7对':>5s} {'仅P4':>5s} {'仅P7':>5s} "
  f"{'主链路P4':>8s} {'主链路P7':>8s} {'调用P4':>7s} {'调用P7':>7s}")
for b, n, ca, cc, o4, o7, ma, mc, na, nc in summary:
    P(f"{b:14s} {n:4d} {ca:5d} {cc:5d} {o4:5d} {o7:5d} {ma:7.1f}% {mc:7.1f}% {na:7.3f} {nc:7.3f}")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_p4_p7.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
