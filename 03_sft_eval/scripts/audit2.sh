#!/bin/bash
# v2: audit the OUTPUT field only. v1 scanned the whole row, but `input` carries
# the tool SCHEMA (every tool name appears there for every sample), so it says
# nothing about what was called. `output` is the model's actual generation:
# its tool calls and the tool responses it got back.
exec >>/tmp/audit2.log 2>&1
echo "=== START $(date -Is)"
sudo -n /opt/conda-st/envs/spacetools-rl/bin/python - <<'PY'
import json, re, collections
TOOLS = ["roborefer","vlm","sam2","depth_estimator","bounding_box","vision_ops","grasp_generator"]
ERR = re.compile(r"Error:|ERROR:toolshed|Traceback|Version mismatch|not available|Unknown tool")
for b in ["robospatial","reflocation","refplacement","refunseen"]:
    calls = collections.Counter(); errs = collections.Counter()
    rows = 0; err_rows = 0; no_call_rows = 0
    for line in open(f"/workspace/eval_out/{b}/0.jsonl"):
        r = json.loads(line); out = r.get("output") or ""
        rows += 1
        hit = [t for t in TOOLS if t+"." in out]
        if not hit: no_call_rows += 1
        for t in hit: calls[t] += 1
        if ERR.search(out):
            err_rows += 1
            for t in hit: errs[t] += 1
    print("### %-13s rows=%-4d rows_with_error_text=%-3d rows_with_no_tool_call=%d"
          % (b, rows, err_rows, no_call_rows))
    for t in TOOLS:
        if calls[t]: print("      %-16s rows_calling=%-5d of_which_error=%d" % (t, calls[t], errs[t]))
PY
echo
echo "--- one sample output, to eyeball what a tool response looks like"
sudo -n /opt/conda-st/envs/spacetools-rl/bin/python -c "
import json
r=json.loads(open('/workspace/eval_out/reflocation/0.jsonl').readline())
print(r['output'][:900])"
echo "=== DONE $(date -Is)"
