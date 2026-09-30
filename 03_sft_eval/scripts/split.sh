#!/bin/bash
# Split robospatial into VQA / Vacant the way the P5 report does -- by GT shape:
#   Yes/No  -> VQA      (P5 got n=228)
#   [(x,y)] -> Vacant   (P5 got n=122)
# so the numbers line up row-for-row with paper Table 2.
exec >>/tmp/split.log 2>&1
echo "=== START $(date -Is)"
sudo -n /opt/conda-st/envs/spacetools-rl/bin/python - <<'PY'
import json, re
vqa = vac = other = 0
vqa_c = vac_c = 0.0
conf = {}   # (gt, pred) for VQA
for line in open("/workspace/eval_out/robospatial/0.jsonl"):
    r = json.loads(line)
    g = (r.get("gts") or "").strip()
    acc = r.get("acc") or 0.0
    gl = g.lower()
    is_vqa = ("yes" in gl[:400] or "no" in gl[:400]) and "(" not in g[:200]
    if re.search(r"\(\s*-?[0-9]*\.?[0-9]+\s*,", g):
        vac += 1; vac_c += acc
    elif is_vqa:
        vqa += 1; vqa_c += acc
    else:
        other += 1
print("VQA     n=%-4d acc=%.2f%%  (%d correct)" % (vqa, 100*vqa_c/max(vqa,1), round(vqa_c)))
print("Vacant  n=%-4d acc=%.2f%%  (%d correct)" % (vac, 100*vac_c/max(vac,1), round(vac_c)))
print("other   n=%d" % other)
tot = vqa+vac+other
print("total   n=%d  overall=%.2f%%" % (tot, 100*(vqa_c+vac_c)/tot))
PY
echo "=== DONE $(date -Is)"
