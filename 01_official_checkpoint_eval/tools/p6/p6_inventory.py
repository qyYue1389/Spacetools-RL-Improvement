import json, os, statistics
from collections import Counter
BASE = __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..", "..", "p4", "parsed")
ACC = ["blinkdepth","cvb2drelation","cvb3ddepth","reflocation","refplacement","refunseen","robospatial"]
CONT = ["boppose","bopgrasp"]   # score 不是准确率,单独处理

def load(b): return [json.loads(l) for l in open(f"{BASE}/{b}.jsonl",encoding="utf-8")]

print("=== 准确率类 benchmark:错误清单 ===")
print(f"{'benchmark':16} {'n':>5} {'对':>5} {'错':>5} {'正确率':>8} | {'工具错':>6} {'OOM':>4} {'截断':>5} {'顶轮':>5} {'无ans':>6} {'幻影':>5} || {'clean错':>8}")
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
print(f"{'合计':16} {sum(len(load(b)) for b in ACC):5} {'':5} {tot_w:5} {'':8} | {'':6} {'':4} {'':5} {'':5} {'':6} {'':5} || {tot_c:8}")
print()
print(f"** 错题总数 {tot_w},其中 clean(一切正常却答错){tot_c} 个 = {100*tot_c/tot_w:.1f}% **")
print()

print("=== 连续指标 benchmark ===")
for b in CONT:
    rs=load(b)
    sc=[r["score"] for r in rs]
    tf=sum(1 for r in rs if r["tool_failures"])
    print(f"{b:12} n={len(rs)}  工具错样本 {tf}  score mean {statistics.mean(sc):.4f}  "
          f"(bopgrasp 的 score 是 RL NCE,越低越好)")
print()

print("=== 错题的链路分布(clean 错题,按 benchmark)===")
for b in ACC:
    rs=load(b)
    wrong=[r for r in rs if not r["correct"]]
    c=Counter(r["chain_signature"] for r in wrong)
    allc=Counter(r["chain_signature"] for r in rs)
    print(f"\n{b}  ({len(wrong)} 错)")
    for chain,n in c.most_common(4):
        tot=allc[chain]
        print(f"   {n:4}/{tot:<4} ({100*n/tot:5.1f}% 错)  {chain}")
