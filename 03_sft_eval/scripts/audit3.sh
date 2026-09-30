#!/bin/bash
# Last gap: the only 2 calls that touched the two Python-3.11.0 envs
# (depth_estimator, vision_ops). Print their tool_response verbatim -- a failed
# actor could return wording my regex does not cover.
exec >>/tmp/audit3.log 2>&1
echo "=== START $(date -Is)"
sudo -n /opt/conda-st/envs/spacetools-rl/bin/python - <<'PY'
import json, re
pat = re.compile(r"<tool_response>(.*?)</tool_response>", re.S)
for line in open("/workspace/eval_out/robospatial/0.jsonl"):
    r = json.loads(line); out = r.get("output") or ""
    if "depth_estimator." in out or "vision_ops." in out:
        print("### index=%s acc=%s" % (r.get("index"), r.get("acc")))
        for c in re.findall(r'\{"name": "([a-z_]+\.[a-z_0-9]+)"', out):
            print("   call:", c)
        for m in pat.findall(out):
            print("   response:", " ".join(m.split())[:300])
        print()
PY
echo "--- sample-weighted RefSpatial"
sudo -n /opt/conda-st/envs/spacetools-rl/bin/python - <<'PY'
n = {"reflocation":100, "refplacement":100, "refunseen":77}
a = {"reflocation":0.54, "refplacement":0.57, "refunseen":0.4675324675324675}
num = sum(a[k]*n[k] for k in n); den = sum(n.values())
print("  simple mean    : %.2f" % (100*sum(a.values())/3))
print("  sample-weighted: %.2f  (%d correct / %d)" % (100*num/den, round(num), den))
PY
echo "=== DONE $(date -Is)"
