#!/usr/bin/env python3
"""Per-sample comparison of P4 (official checkpoint, after GRPO) and P7 (self-trained SFT + GFlowRL C').

Consumes parsed records only, no GPU needed. P4 uses run1 (all nine benchmarks), P7 uses the valid runs
(the five keys of runA + the four of runB). Samples are paired by sample_id.
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

def is_vqa(r):            # robospatial: Yes/No questions vs Vacant point questions
    return len(str(r.get("gt", "")).strip()) < 8

def blame(r):
    """Error attribution: classify by verifiable traces; the order is the priority."""
    if r.get("oom"):                       return "OOM"
    if r.get("tool_failures"):             return "tool failure"
    if r.get("truncated_generation"):      return "generation truncated"
    if r.get("truncated_tool_response"):   return "tool response truncated"
    if r.get("hit_max_turns"):             return "hit max turns"
    if not r.get("parse_ok", True):        return "answer parse failed"
    if r.get("malformed_tool_calls"):      return "malformed tool_call"
    if r.get("n_tool_calls", 0) == 0:      return "no tool call"
    return "clean (reasoning/tool precision)"

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
P("P4 (official checkpoint · after GRPO)  ↔  P7 (self-trained SFT · GFlowRL C' step85)  per-sample comparison")
P("P4 = p4/parsed (run1) · P7 = parsed of the valid runs · paired by sample_id")
P("=" * 78)

summary = []
for b in BENCH:
    a, c = load(P4, b), load(P7, b)
    if a is None or c is None:
        P(f"\n### {b}: missing parsed ({'P4' if a is None else 'P7'}), skipping"); continue
    common = sorted(set(a) & set(c))
    P("")
    P(f"### {b}   n(P4)={len(a)}  n(P7)={len(c)}  common samples={len(common)}")

    # --- accuracy and pairing
    ca = sum(1 for i in common if a[i]["correct"])
    cc = sum(1 for i in common if c[i]["correct"])
    both = sum(1 for i in common if a[i]["correct"] and c[i]["correct"])
    neither = sum(1 for i in common if not a[i]["correct"] and not c[i]["correct"])
    only4 = sum(1 for i in common if a[i]["correct"] and not c[i]["correct"])
    only7 = sum(1 for i in common if c[i]["correct"] and not a[i]["correct"])
    P(f"  accuracy    P4 {ca:4d} {pct(ca,len(common))}   P7 {cc:4d} {pct(cc,len(common))}")
    P(f"  paired      both right {both}  both wrong {neither}  only P4 right {only4}  only P7 right {only7}")
    # continuous scores (boppose/bopgrasp): also report the mean
    sa = sum(a[i]["score"] for i in common) / len(common)
    sc = sum(c[i]["score"] for i in common) / len(common)
    P(f"  mean score  P4 {sa:.4f}   P7 {sc:.4f}")

    # --- tool chain signatures
    P(f"  chain signature (P4)                            chain signature (P7)")
    ta, tc = topsig(a), topsig(c)
    for k in range(max(len(ta), len(tc))):
        L = f"{ta[k][0][:34]:34s} {ta[k][1]:4d} {ta[k][2]:5.1f}%" if k < len(ta) else " " * 46
        Rr = f"{tc[k][0][:34]:34s} {tc[k][1]:4d} {tc[k][2]:5.1f}%" if k < len(tc) else ""
        P(f"    {L}  {Rr}")
    P(f"  main-chain coverage  P4 {ta[0][2]:5.1f}%   P7 {tc[0][2]:5.1f}%"
      f"   signature kinds P4 {len(set(r['chain_signature'] for r in a.values()))}"
      f" P7 {len(set(r['chain_signature'] for r in c.values()))}")

    # --- tool calls
    P(f"  tool calls  P4 {toolcalls(a)}")
    P(f"              P7 {toolcalls(c)}")
    na = sum(r["n_tool_calls"] for r in a.values()) / len(a)
    nc = sum(r["n_tool_calls"] for r in c.values()) / len(c)
    P(f"  calls/sample  P4 {na:.3f}   P7 {nc:.3f}")

    # --- variable reuse
    ea, ua, una, pha, nwa = varstats(a)
    ec, uc, unc, phc, nwc = varstats(c)
    P(f"  variables   P4 exposed {ea:4d} used {ua:4d} unused {una:4d} hallucinated {pha:3d}  samples with variables {nwa}")
    P(f"              P7 exposed {ec:4d} used {uc:4d} unused {unc:4d} hallucinated {phc:3d}  samples with variables {nwc}")

    # --- error attribution
    wa = collections.Counter(blame(a[i]) for i in common if not a[i]["correct"])
    wc = collections.Counter(blame(c[i]) for i in common if not c[i]["correct"])
    keys = sorted(set(wa) | set(wc))
    P(f"  error attribution              P4      P7")
    for k in keys:
        P(f"    {k:26s} {wa.get(k,0):5d}   {wc.get(k,0):5d}")

    summary.append((b, len(common), ca, cc, only4, only7, ta[0][2], tc[0][2], na, nc))

# --- robospatial VQA/Vacant breakdown + the two P7 runs
P("")
P("=" * 78)
P("robospatial breakdown")
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
    P(f"  P7 two runs: always right {len(stable_ok)} · flipped {len(flip)} · "
      f"of which VQA {sum(1 for i in flip if is_vqa(c[i]))} / Vacant {sum(1 for i in flip if not is_vqa(c[i]))}")
    # P7 always-right set vs P4 correct set
    p4ok = {i for i in a if a[i]["correct"]}
    P(f"  P7 always right ∩ P4 right = {len(set(stable_ok) & p4ok)} · P7 always right but P4 wrong = {len(set(stable_ok) - p4ok)} · "
      f"P4 right but P7 wrong in both runs = {len(p4ok - {i for i in c if c[i]['correct'] or c2[i]['correct']})}")

# --- summary table
P("")
P("=" * 78)
P("Summary table")
P(f"{'benchmark':14s} {'n':>4s} {'P4 ok':>5s} {'P7 ok':>5s} {'P4only':>5s} {'P7only':>5s} "
  f"{'mainP4':>8s} {'mainP7':>8s} {'callsP4':>7s} {'callsP7':>7s}")
for b, n, ca, cc, o4, o7, ma, mc, na, nc in summary:
    P(f"{b:14s} {n:4d} {ca:5d} {cc:5d} {o4:5d} {o7:5d} {ma:7.1f}% {mc:7.1f}% {na:7.3f} {nc:7.3f}")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_p4_p7.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
