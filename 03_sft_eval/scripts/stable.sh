#!/bin/bash
# P5 handled blinkdepth's run-to-run spread by decomposing into
# always-right / always-wrong / flipping, and reporting the stable core plus a
# band. Same treatment here, overall and split VQA / Vacant.
# The point: is the >=60 gate (210/350) inside the flip band or outside it?
exec >>/tmp/stable.log 2>&1
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
def vac(r): return bool(re.search(r"\(\s*-?[0-9]*\.?[0-9]+\s*,", r.get("gts") or ""))
idx=sorted(set(r1)&set(r2))

def block(sel, label, gate_frac=None):
    ii=[i for i in idx if sel(r1[i])]
    both=sum(1 for i in ii if r1[i]["acc"]>=1 and r2[i]["acc"]>=1)
    neither=sum(1 for i in ii if r1[i]["acc"]<1 and r2[i]["acc"]<1)
    flip=[i for i in ii if (r1[i]["acc"]>=1)!=(r2[i]["acc"]>=1)]
    n=len(ii)
    f1=sum(1 for i in flip if r1[i]["acc"]>=1); f2=sum(1 for i in flip if r2[i]["acc"]>=1)
    print("  %s  n=%d" % (label, n))
    print("     always-right %d   always-wrong %d   flipping %d" % (both, neither, len(flip)))
    print("     floor %.2f%% (%d)   ceiling %.2f%% (%d)" %
          (100*both/n, both, 100*(both+len(flip))/n, both+len(flip)))
    print("     run1 %d of the %d flips right · run2 %d  -> p_flip ~ %.2f" %
          (f1, len(flip), f2, (f1+f2)/(2*len(flip)) if flip else 0))
    if flip:
        p=(f1+f2)/(2*len(flip)); k=len(flip)
        mean=both+p*k; sd=math.sqrt(k*p*(1-p))
        print("     expected %.2f%% (%.1f/%d)  sd %.1f samples (%.2f pp)" %
              (100*mean/n, mean, n, sd, 100*sd/n))
        if gate_frac is not None:
            need=math.ceil(gate_frac*n)-both      # flips needed to clear the gate
            # exact binomial tail: P(X < need)
            from math import comb
            pr=sum(comb(k,x)*p**x*(1-p)**(k-x) for x in range(0, max(need,0)))
            print("     gate %.0f%% = %d correct -> needs %d of %d flips" %
                  (100*gate_frac, math.ceil(gate_frac*n), need, k))
            print("     P(a single run reads BELOW the gate) = %.1f%%" % (100*pr))
print("### RoboSpatial overall")
block(lambda r: True, "overall", 0.60)
print()
print("### split")
block(lambda r: not vac(r), "VQA   ")
block(vac, "Vacant")
PY
echo "=== DONE $(date -Is)"
