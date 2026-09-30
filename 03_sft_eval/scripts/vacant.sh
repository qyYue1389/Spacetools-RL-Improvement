#!/bin/bash
# The 2 wrong answers I eyeballed both handed the WHOLE spatial predicate to
# roborefer.detect_one as obj_name. Two examples is an anecdote -- quantify it,
# split by right/wrong, so it is a measurement instead of a hunch.
exec >>/tmp/vacant.log 2>&1
echo "=== START $(date -Is)"
sudo -n /opt/conda-st/envs/spacetools-rl/bin/python - <<'PY'
import json, re
rows=[]
for line in open("/workspace/eval_out/robospatial/0.jsonl"):
    r=json.loads(line); g=(r.get("gts") or ""); out=r.get("output") or ""
    is_vacant = bool(re.search(r"\(\s*-?[0-9]*\.?[0-9]+\s*,", g))
    names = re.findall(r'"obj_name":\s*"([^"]*)"', out)
    rows.append((is_vacant, (r.get("acc") or 0.0), names, out, g))

def stat(sel, label):
    sub=[r for r in rows if sel(r)]
    if not sub: return
    L=[len(n) for r in sub for n in r[2]]
    long_ones=[r for r in sub if any(len(n)>30 for n in r[2])]
    print("  %-22s n=%-4d mean_objname_len=%5.1f  samples_with_a_>30char_objname=%d (%.0f%%)"
          % (label, len(sub), (sum(L)/len(L) if L else 0), len(long_ones),
             100*len(long_ones)/len(sub)))

print("### robospatial, split by task type and correctness")
stat(lambda r: r[0] and r[1]>=1.0, "Vacant  correct")
stat(lambda r: r[0] and r[1]<1.0,  "Vacant  wrong")
stat(lambda r: (not r[0]) and r[1]>=1.0, "VQA     correct")
stat(lambda r: (not r[0]) and r[1]<1.0,  "VQA     wrong")

print()
print("### Vacant: how close are the misses? (min distance to any GT point)")
import math
ds=[]
for is_vac, acc, names, out, g in rows:
    if not is_vac or acc>=1.0: continue
    ans = re.search(r"<answer>(.*?)</answer>", out, re.S)
    if not ans: continue
    p = re.findall(r"\(\s*(-?[0-9.]+)\s*,\s*(-?[0-9.]+)\s*\)", ans.group(1))
    gt = re.findall(r"\(\s*(-?[0-9.]+)\s*,\s*(-?[0-9.]+)\s*\)", g)
    if not p or not gt: continue
    px,py=float(p[0][0]),float(p[0][1])
    d=min(math.hypot(px-float(a),py-float(b)) for a,b in gt)
    ds.append(d)
ds.sort()
if ds:
    print("  wrong Vacant samples with a parseable point: %d" % len(ds))
    print("  min dist to nearest acceptable point:")
    print("     median %.3f   p25 %.3f   p75 %.3f   min %.3f   max %.3f"
          % (ds[len(ds)//2], ds[len(ds)//4], ds[3*len(ds)//4], ds[0], ds[-1]))
    for t in (0.05, 0.10, 0.20):
        print("     within %.2f of an acceptable point: %d (%.0f%%)"
              % (t, sum(1 for d in ds if d<=t), 100*sum(1 for d in ds if d<=t)/len(ds)))
PY
echo "=== DONE $(date -Is)"
