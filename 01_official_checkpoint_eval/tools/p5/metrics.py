import json, math, re, statistics, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
RL_DIR = os.environ.get("SPACETOOLS_RL", os.path.join(HERE, "..", "..", "..", "..", "SpaceTools-RL"))  # full SpaceTools-RL checkout
from typing import List, Tuple, Sequence, Any, Optional, Dict, Union

REPO = RL_DIR
RV = open(f"{REPO}/verl/utils/reward_score/robos_vision.py", encoding="utf-8").read()
BB = open(f"{REPO}/verl/utils/reward_score/bop_ask_bench.py", encoding="utf-8").read()

ns = {"math": math, "re": re, "List": List, "Tuple": Tuple, "Sequence": Sequence,
      "Any": Any, "Optional": Optional, "Dict": Dict, "Union": Union}
def pull(src, name):
    m = re.search(rf'^def {re.escape(name)}\(.*?(?=^def |\Z)', src, re.M | re.S)
    assert m, name
    exec(m.group(0), ns)
for n in ["_cross", "_polygon_area", "_convex_hull", "_convex_polygon_intersection", "_calculate_iou"]:
    pull(RV, n)
# regexes bop_ask_bench needs
ns["_PT_RE"] = re.compile(r"\(\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\)")
ns["_PT_SQ_RE"] = re.compile(r"\[\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\]")
for n in ["_extract_points", "_chamfer_score"]:
    pull(BB, n)

extract_points = ns["_extract_points"]; calc_iou = ns["_calculate_iou"]
chamfer = ns["_chamfer_score"]; hull = ns["_convex_hull"]; parea = ns["_polygon_area"]

rows = [json.loads(l) for l in open(HERE + "/../../p4/parsed/boppose.jsonl", encoding="utf-8")]
print(f"samples: {len(rows)}")

def ordered_mean_dist(p, g):
    return sum(math.hypot(px-gx, py-gy) for (px,py),(gx,gy) in zip(p,g)) / len(p)

recs = []
for r in rows:
    gt = extract_points(r["gt"]); pr = extract_points(r["raw_answer"] or "")
    ok = (len(gt) == 8 and len(pr) == 8)
    d = {"id": r["sample_id"], "ok": ok, "dumped": r.get("score")}
    if ok:
        d["hull_iou"]  = calc_iou(gt, pr)
        d["chamfer"]   = chamfer(pr, gt)                       # repo default threshold 0.15
        d["cham_dist"] = 0.15 * (1 - chamfer(pr, gt)) if chamfer(pr,gt) > 0 else None
        d["ord_dist"]  = ordered_mean_dist(pr, gt)
        # true symmetric chamfer distance (un-scored)
        def nn(a,b): return sum(min(math.hypot(ax-bx,ay-by) for bx,by in b) for ax,ay in a)/len(a)
        d["cham_raw"]  = (nn(pr,gt) + nn(gt,pr)) / 2
        d["hull_area_gt"] = parea(hull(gt)); d["hull_area_pr"] = parea(hull(pr))
    recs.append(d)

good = [r for r in recs if r["ok"]]
print(f"format-valid (8,8): {len(good)}/{len(recs)}")

# sanity: does hull_iou reproduce the dumped score?
diffs = [abs(r["hull_iou"] - r["dumped"]) for r in good if r["dumped"] is not None]
print(f"hull_iou vs dumped score: max abs diff {max(diffs):.3e}  (n={len(diffs)})")

def m(key): return statistics.mean(r[key] for r in good) * 100
print()
print("=== 候选指标 (mean x 100) ===")
print(f"  hull IoU        (现行)          {m('hull_iou'):8.2f}     <- P4 报的 53.36")
print(f"  chamfer score   (repo, t=0.15)  {m('chamfer'):8.2f}")
print()
print("=== 原始距离量 (归一化图像坐标) ===")
print(f"  symmetric chamfer distance  mean {statistics.mean(r['cham_raw'] for r in good):.4f}"
      f"  median {statistics.median(r['cham_raw'] for r in good):.4f}")
print(f"  ordered per-corner distance mean {statistics.mean(r['ord_dist'] for r in good):.4f}"
      f"  median {statistics.median(r['ord_dist'] for r in good):.4f}")
print()
print("=== 阈值扫描:max(0, 1 - d/t) x 100 ===")
print(f"{'t':>7} | {'chamfer(顺序无关)':>20} | {'ordered(尊重对应)':>20}")
for t in [0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30,0.40,0.50]:
    c = statistics.mean(max(0.0, 1 - r["cham_raw"]/t) for r in good)*100
    o = statistics.mean(max(0.0, 1 - r["ord_dist"]/t) for r in good)*100
    mark = "  <== 34.37" if abs(c-34.37) < 0.6 or abs(o-34.37) < 0.6 else ""
    print(f"{t:7.2f} | {c:20.2f} | {o:20.2f}{mark}")
