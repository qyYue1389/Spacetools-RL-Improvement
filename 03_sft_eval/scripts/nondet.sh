#!/bin/bash
# Split the run1/run2 divergence into policy-side vs tool-side.
# Each rollout's `output` holds both the model's generations and the
# <tool_response> blocks, so the attribution is already on disk:
#   A same tool_call, same tool_response, different answer -> policy after the tool
#   B same tool_call, DIFFERENT tool_response              -> roborefer is non-deterministic
#   C different tool_call                                  -> policy diverged before the tool
exec >>/tmp/nondet.log 2>&1
echo "=== START $(date -Is)"
sudo -n /opt/conda-st/envs/spacetools-rl/bin/python - <<'PY'
import json, re, math
def load(p):
    d={}
    for line in open(p):
        r=json.loads(line); d[r["index"]]=r
    return d
r1=load("/workspace/run1_robospatial.jsonl")
r2=load("/workspace/eval_out/robospatial/0.jsonl")
CALL=re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
RESP=re.compile(r"<tool_response>(.*?)</tool_response>", re.S)
ANS =re.compile(r"<answer>(.*?)</answer>", re.S)
def norm(s): return " ".join(s.split())
def parts(r):
    o=r.get("output") or ""
    return ([norm(x) for x in CALL.findall(o)],
            [norm(x) for x in RESP.findall(o)],
            norm(ANS.search(o).group(1)) if ANS.search(o) else None)

A=B=C=same=0; Bsamp=[]; coord_deltas=[]
for i in sorted(set(r1)&set(r2)):
    c1,p1,a1 = parts(r1[i]); c2,p2,a2 = parts(r2[i])
    if c1==c2 and p1==p2 and a1==a2: same+=1; continue
    if c1!=c2: C+=1
    elif p1!=p2:
        B+=1; Bsamp.append(i)
        # if both responses carry coordinates, how far apart are they?
        f1=re.findall(r"\(([0-9.]+),\s*([0-9.]+)\)", " ".join(p1))
        f2=re.findall(r"\(([0-9.]+),\s*([0-9.]+)\)", " ".join(p2))
        if f1 and f2 and len(f1)==len(f2):
            for (x1,y1),(x2,y2) in zip(f1,f2):
                coord_deltas.append(math.hypot(float(x1)-float(x2), float(y1)-float(y2)))
    else: A+=1
tot=len(set(r1)&set(r2))
print("  n=%d" % tot)
print("  fully identical (call+response+answer) : %d" % same)
print("  C different tool_call    (policy, pre-tool)  : %d" % C)
print("  B same call, diff response (roborefer side)  : %d" % B)
print("  A same call+response, diff answer (policy)   : %d" % A)
if coord_deltas:
    coord_deltas.sort()
    n=len(coord_deltas)
    print("  roborefer coordinate drift on B, n=%d pairs:" % n)
    print("     median %.4f  p90 %.4f  max %.4f  exact-zero %d" % (
        coord_deltas[n//2], coord_deltas[int(.9*n)], coord_deltas[-1],
        sum(1 for d in coord_deltas if d==0)))
print("  B sample indices (first 10):", Bsamp[:10])
PY
echo "=== DONE $(date -Is)"
