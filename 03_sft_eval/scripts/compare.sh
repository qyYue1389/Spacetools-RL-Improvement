#!/bin/bash
# Wait for run 2 to finish, then do the whole comparison in one go so the
# result is ready rather than needing another round-trip.
exec 9>/tmp/compare.lock
flock -n 9 || exit 0
exec >>/tmp/compare.log 2>&1
echo "=== START $(date -Is)"

for i in $(seq 1 40); do
    if ! pgrep -f eval3.sh >/dev/null; then break; fi
    sleep 30
done
echo "--- eval3 finished (or waited out) at $(date -Is)"

echo
echo "--- run 2 gates"
printf "  OOM                    %s\n" "$(grep -c OutOfMemoryError /workspace/eval_run2.log)"
printf "  Error:|ERROR:toolshed  %s\n" "$(grep -cE 'Error:|ERROR:toolshed' /workspace/eval_run2.log)"
grep -oE "val-core/[A-Za-z0-9/-]+/acc/mean@1:[0-9.]+" /workspace/eval_run2.log | sort -u | sed 's/^/  /'

echo
echo "--- per-sample comparison run1 vs run2"
sudo -n /opt/conda-st/envs/spacetools-rl/bin/python - <<'PY'
import json, re, os
def load(p):
    d={}
    for line in open(p):
        r=json.loads(line); d[r["index"]]=r
    return d
p2="/workspace/eval_out/robospatial/0.jsonl"
p1="/workspace/run1_robospatial.jsonl"
if not os.path.exists(p2): print("  run2 jsonl missing"); raise SystemExit
r1, r2 = load(p1), load(p2)
common = sorted(set(r1) & set(r2))
print("  run1 n=%d  run2 n=%d  common=%d" % (len(r1), len(r2), len(common)))

def vac(r): return bool(re.search(r"\(\s*-?[0-9]*\.?[0-9]+\s*,", r.get("gts") or ""))
agg={}
for tag, d in (("run1",r1),("run2",r2)):
    v=[x for x in d.values() if vac(x)]; q=[x for x in d.values() if not vac(x)]
    agg[tag]=(sum(x["acc"] for x in q), len(q), sum(x["acc"] for x in v), len(v))
    print("  %s  VQA %5.2f%% (%d/%d)   Vacant %5.2f%% (%d/%d)   Overall %5.2f%% (%d/%d)" % (
        tag, 100*agg[tag][0]/agg[tag][1], round(agg[tag][0]), agg[tag][1],
        100*agg[tag][2]/agg[tag][3], round(agg[tag][2]), agg[tag][3],
        100*(agg[tag][0]+agg[tag][2])/(agg[tag][1]+agg[tag][3]),
        round(agg[tag][0]+agg[tag][2]), agg[tag][1]+agg[tag][3]))

flips=[i for i in common if (r1[i]["acc"]>=1.0) != (r2[i]["acc"]>=1.0)]
w2r=[i for i in flips if r2[i]["acc"]>=1.0]; r2w=[i for i in flips if r1[i]["acc"]>=1.0]
print("  flipped: %d / %d   (wrong->right %d, right->wrong %d)" % (len(flips), len(common), len(w2r), len(r2w)))
same_out=sum(1 for i in common if (r1[i].get("output") or "")==(r2[i].get("output") or ""))
print("  byte-identical generations: %d / %d" % (same_out, len(common)))
if flips[:8]: print("  flipped indices (first 8):", flips[:8])
PY
echo "=== DONE $(date -Is)"
