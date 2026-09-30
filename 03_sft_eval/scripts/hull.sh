#!/bin/bash
# Redo the Vacant analysis with the REAL criterion, using the repo's own
# geometry functions (robos_all._convex_hull / _is_point_in_convex_polygon)
# rather than reimplementing them -- same approach P5 used for the pose metric.
# Retracts my earlier "distance to nearest GT point" numbers.
exec >>/tmp/hull.log 2>&1
echo "=== START $(date -Is)"
sudo -n env PYTHONPATH=/opt/spacetools/SpaceTools-RL \
  /opt/conda-st/envs/spacetools-rl/bin/python - <<'PY'
import json, re, math, sys
from verl.utils.reward_score.robos_all import _convex_hull, _is_point_in_convex_polygon
def load(p):
    d={}
    for line in open(p):
        r=json.loads(line); d[r["index"]]=r
    return d
r1=load("/workspace/run1_robospatial.jsonl")
r2=load("/workspace/eval_out/robospatial/0.jsonl")
ANS=re.compile(r"<answer>(.*?)</answer>", re.S)
def pt(r):
    m=ANS.search(r.get("output") or "")
    if not m: return None
    f=re.findall(r"\(\s*(-?[0-9.]+)\s*,\s*(-?[0-9.]+)\s*\)", m.group(1))
    return (float(f[0][0]), float(f[0][1])) if f else None
def gts(r):
    return [(float(a),float(b)) for a,b in
            re.findall(r"\(\s*(-?[0-9.]+)\s*,\s*(-?[0-9.]+)\s*\)", r.get("gts") or "")]
def vac(r): return bool(gts(r))

def seg_dist(p, a, b):
    px,py=p; ax,ay=a; bx,by=b
    dx,dy=bx-ax,by-ay
    L=dx*dx+dy*dy
    t = 0.0 if L==0 else max(0.0, min(1.0, ((px-ax)*dx+(py-ay)*dy)/L))
    return math.hypot(px-(ax+t*dx), py-(ay+t*dy))
def dist_to_hull_edge(p, hull):
    return min(seg_dist(p, hull[i], hull[(i+1)%len(hull)]) for i in range(len(hull)))

# sanity: does my recomputation agree with the stored acc?
agree=dis=0
ds=[]
for i in sorted(r1):
    r=r1[i]
    if not vac(r): continue
    p=pt(r); g=gts(r)
    if not p or len(g)<3: continue
    hull=_convex_hull(g)
    inside=_is_point_in_convex_polygon(p, hull)
    stored = (r["acc"]>=1.0)
    if inside==stored: agree+=1
    else: dis+=1
    if not inside:
        ds.append(dist_to_hull_edge(p, hull))
print("  recompute vs stored acc (run1 Vacant): agree %d, disagree %d" % (agree, dis))
ds.sort(); n=len(ds)
print()
print("  CORRECTED: distance from the predicted point to the hull BOUNDARY,")
print("  over the wrong Vacant samples (n=%d):" % n)
print("     median %.4f   p25 %.4f   p75 %.4f   min %.4f   max %.4f" %
      (ds[n//2], ds[n//4], ds[3*n//4], ds[0], ds[-1]))
for t in (0.02, 0.05, 0.10):
    k=sum(1 for d in ds if d<=t)
    print("     within %.2f of the hull edge: %d (%.0f%%)" % (t, k, 100*k/n))

print()
print("  the 8 flipping Vacant samples, against the hull:")
flips=[i for i in sorted(set(r1)&set(r2)) if vac(r1[i]) and (r1[i]["acc"]>=1)!=(r2[i]["acc"]>=1)]
for i in flips:
    g=gts(r1[i]); hull=_convex_hull(g)
    p1,p2=pt(r1[i]),pt(r2[i])
    if not (p1 and p2): continue
    d1 = 0.0 if _is_point_in_convex_polygon(p1,hull) else dist_to_hull_edge(p1,hull)
    d2 = 0.0 if _is_point_in_convex_polygon(p2,hull) else dist_to_hull_edge(p2,hull)
    print("   %4d  run1 in=%-5s  run2 in=%-5s  run2 dist outside hull %.4f" %
          (i, _is_point_in_convex_polygon(p1,hull), _is_point_in_convex_polygon(p2,hull), d2))
PY
echo "=== DONE $(date -Is)"
