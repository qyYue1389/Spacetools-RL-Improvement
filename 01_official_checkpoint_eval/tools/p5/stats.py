import json, os, statistics
from collections import Counter
BASE = __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "p4", "parsed")
BM = ["blinkdepth","cvb2drelation","cvb3ddepth","reflocation","refplacement",
      "refunseen","robospatial","boppose","bopgrasp"]
print(f"{'benchmark':16} {'n':>5} {'assist turns':>8} {'user turns':>7} {'tools/sample':>9} "
      f"{'tool err':>6} {'phantom var':>7} {'unused var':>7} {'turn-count match':>8}")
print("-"*82)
tot=0
allstats={}
for b in BM:
    rs=[json.loads(l) for l in open(f"{BASE}/{b}.jsonl",encoding="utf-8")]
    tot+=len(rs)
    at=statistics.mean(r["num_turns_derived"] for r in rs)
    ut=statistics.mean(r["num_user_turns_derived"] for r in rs)
    tc=statistics.mean(r["n_tool_calls"] for r in rs)
    tf=sum(1 for r in rs if r["tool_failures"])
    ph=sum(1 for r in rs if r.get("vars_phantom"))
    vu=sum(1 for r in rs if r.get("vars_unused"))
    ag=sum(1 for r in rs if r.get("num_turns_agree") is True)
    agn=sum(1 for r in rs if r.get("num_turns_agree") is not None)
    print(f"{b:16} {len(rs):5} {at:8.2f} {ut:7.2f} {tc:9.2f} {tf:6} {ph:7} {vu:7} {ag:4}/{agn:<4}")
    allstats[b]=rs
print("-"*82)
print(f"{'total':16} {tot:5}")
print()
print("=== health metrics (all should be 0) ===")
print(f"{'benchmark':16} {'OOM':>5} {'truncated':>5} {'hit max turns':>7} {'no answer':>9} {'malformed call':>9}")
for b in BM:
    rs=allstats[b]
    print(f"{b:16} {sum(r['oom'] for r in rs):5} {sum(r['truncated_tool_response'] for r in rs):5} "
          f"{sum(r['hit_max_turns'] for r in rs):7} {sum(1 for r in rs if not r['parse_ok']):9} "
          f"{sum(r['malformed_tool_calls'] for r in rs):9}")
print()
print("=== dominant chain (most frequent chain_signature) ===")
for b in BM:
    rs=allstats[b]
    c=Counter(r["chain_signature"] for r in rs)
    top,n=c.most_common(1)[0]
    print(f"{b:16} {n:4}/{len(rs):<4} ({100*n/len(rs):5.1f}%)  {top}   [{len(c)} kinds total]")
