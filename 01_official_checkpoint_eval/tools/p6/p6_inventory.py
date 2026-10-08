import json, os, statistics
from collections import Counter
BASE = __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "p4", "parsed")
ACC = ["blinkdepth","cvb2drelation","cvb3ddepth","reflocation","refplacement","refunseen","robospatial"]
CONT = ["boppose","bopgrasp"]   # score is not accuracy, handled separately

def load(b): return [json.loads(l) for l in open(f"{BASE}/{b}.jsonl",encoding="utf-8")]

print("=== accuracy benchmarks: error list ===")
print(f"{'benchmark':16} {'n':>5} {'right':>5} {'wrong':>5} {'accuracy':>8} | {'tool err':>6} {'OOM':>4} {'trunc':>5} {'maxturn':>5} {'no ans':>6} {'phantom':>5} || {'clean wrong':>8}")
print("-"*112)
tot_w=tot_c=0
for b in ACC:
    rs=load(b)
    wrong=[r for r in rs if not r["correct"]]
    infra=[r for r in wrong if r["tool_failures"] or r["oom"] or r["truncated_tool_response"]
           or r["hit_max_turns"] or not r["parse_ok"] or r.get("vars_phantom")]
    clean=[r for r in wrong if r not in infra]
    tot_w+=len(wrong); tot_c+=len(clean)
    print(f"{b:16} {len(rs):5} {len(rs)-len(wrong):5} {len(wrong):5} {100*(1-len(wrong)/len(rs)):7.2f}% | "
          f"{sum(1 for r in wrong if r['tool_failures']):6} {sum(1 for r in wrong if r['oom']):4} "
          f"{sum(1 for r in wrong if r['truncated_tool_response']):5} {sum(1 for r in wrong if r['hit_max_turns']):5} "
          f"{sum(1 for r in wrong if not r['parse_ok']):6} {sum(1 for r in wrong if r.get('vars_phantom')):5} || {len(clean):8}")
print("-"*112)
print(f"{'total':16} {sum(len(load(b)) for b in ACC):5} {'':5} {tot_w:5} {'':8} | {'':6} {'':4} {'':5} {'':5} {'':6} {'':5} || {tot_c:8}")
print()
print(f"** total wrong answers {tot_w}, of which clean (everything normal but answered wrong) {tot_c} = {100*tot_c/tot_w:.1f}% **")
print()

print("=== continuous-metric benchmarks ===")
for b in CONT:
    rs=load(b)
    sc=[r["score"] for r in rs]
    tf=sum(1 for r in rs if r["tool_failures"])
    print(f"{b:12} n={len(rs)}  tool-error samples {tf}  score mean {statistics.mean(sc):.4f}  "
          f"(the bopgrasp score is RL NCE, lower is better)")
print()

print("=== chain distribution of wrong answers (clean wrong answers, by benchmark) ===")
for b in ACC:
    rs=load(b)
    wrong=[r for r in rs if not r["correct"]]
    c=Counter(r["chain_signature"] for r in wrong)
    allc=Counter(r["chain_signature"] for r in rs)
    print(f"\n{b}  ({len(wrong)} wrong)")
    for chain,n in c.most_common(4):
        tot=allc[chain]
        print(f"   {n:4}/{tot:<4} ({100*n/tot:5.1f}% wrong)  {chain}")
