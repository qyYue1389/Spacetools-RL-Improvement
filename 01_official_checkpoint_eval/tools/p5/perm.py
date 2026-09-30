import json, math, re, statistics, os, itertools
from typing import List, Tuple, Sequence, Any, Optional, Dict, Union
HERE = os.path.dirname(os.path.abspath(__file__))
RL_DIR = os.environ.get("SPACETOOLS_RL", os.path.join(HERE, "..", "..", "..", "..", "SpaceTools-RL"))  # full SpaceTools-RL checkout
RV = open(RL_DIR+"/verl/utils/reward_score/robos_vision.py", encoding="utf-8").read()
BB = open(RL_DIR+"/verl/utils/reward_score/bop_ask_bench.py", encoding="utf-8").read()
ns = {"math":math,"re":re,"List":List,"Tuple":Tuple,"Sequence":Sequence,"Any":Any,"Optional":Optional,"Dict":Dict,"Union":Union}
def pull(src,name):
    m=re.search(rf'^def {re.escape(name)}\(.*?(?=^def |\Z)',src,re.M|re.S); exec(m.group(0),ns)
ns["_PT_RE"]=re.compile(r"\(\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\)")
ns["_PT_SQ_RE"]=re.compile(r"\[\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\]")
for n in ["_cross","_polygon_area","_convex_hull","_convex_polygon_intersection","_calculate_iou"]: pull(RV,n)
for n in ["_extract_points"]: pull(BB,n)
ep=ns["_extract_points"]; area=ns["_polygon_area"]

rows=[json.loads(l) for l in open(HERE+"/../../p4/parsed/boppose.jsonl",encoding="utf-8")]
PERMS=list(itertools.permutations(range(8)))

def mc_iou(A,B,n=40000):
    """IoU of two polygons taken IN GIVEN ORDER (even-odd rule), Monte Carlo."""
    xs=[p[0] for p in A+B]; ys=[p[1] for p in A+B]
    x0,x1,y0,y1=min(xs),max(xs),min(ys),max(ys)
    if x1<=x0 or y1<=y0: return 0.0
    import random; random.seed(0)
    def inside(poly,x,y):
        c=False; j=len(poly)-1
        for i in range(len(poly)):
            xi,yi=poly[i]; xj,yj=poly[j]
            if ((yi>y)!=(yj>y)) and (x < (xj-xi)*(y-yi)/(yj-yi+1e-18)+xi): c=not c
            j=i
        return c
    ia=ib=ii=0
    for _ in range(n):
        x=random.uniform(x0,x1); y=random.uniform(y0,y1)
        a=inside(A,x,y); b=inside(B,x,y)
        ia+=a; ib+=b; ii+=(a and b)
    u=ia+ib-ii
    return ii/u if u else 0.0

ident=[];best=[];bestperm=[];ordiou=[];hulliou=[]
for r in rows:
    gt=ep(r["gt"]); pr=ep(r["raw_answer"] or "")
    if len(gt)!=8 or len(pr)!=8: continue
    D=[[math.hypot(pr[i][0]-gt[j][0], pr[i][1]-gt[j][1]) for j in range(8)] for i in range(8)]
    ident.append(sum(D[i][i] for i in range(8))/8)
    bp=None;bv=1e9
    for p in PERMS:
        s=D[0][p[0]]+D[1][p[1]]+D[2][p[2]]+D[3][p[3]]+D[4][p[4]]+D[5][p[5]]+D[6][p[6]]+D[7][p[7]]
        if s<bv: bv=s; bp=p
    best.append(bv/8); bestperm.append(bp)
    hulliou.append(ns["_calculate_iou"](gt,pr))
    ordiou.append(mc_iou(pr,gt))

print(f"n = {len(ident)}")
print()
print("=== 角点距离(归一化图像坐标)===")
print(f"  按给出顺序 identity      mean {statistics.mean(ident):.4f}   median {statistics.median(ident):.4f}")
print(f"  最优一一匹配 assignment  mean {statistics.mean(best):.4f}   median {statistics.median(best):.4f}")
print(f"  → 顺序造成的额外误差      mean {statistics.mean(ident)-statistics.mean(best):.4f}")
same=sum(1 for p in bestperm if p==tuple(range(8)))
print(f"  最优匹配就是原顺序的样本: {same}/{len(bestperm)}")
from collections import Counter
c=Counter(bestperm)
print(f"  最常见的最优置换 (top 3): {[(p,n) for p,n in c.most_common(3)]}")
print()
print("=== 面积型指标 (mean x 100) ===")
print(f"  凸包 IoU(现行,顺序无关)        {statistics.mean(hulliou)*100:8.2f}")
print(f"  有序多边形 IoU(尊重顺序,MC)    {statistics.mean(ordiou)*100:8.2f}   <== 与 34.37 比")
print()
print("=== 最优匹配距离的阈值扫描 max(0,1-d/t) x 100 ===")
for t in [0.05,0.10,0.15,0.20,0.25,0.30]:
    v=statistics.mean(max(0.0,1-d/t) for d in best)*100
    print(f"   t={t:.2f}  {v:6.2f}")
