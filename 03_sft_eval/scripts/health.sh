#!/bin/bash
# P5 §5.1 lists five health indicators. I checked OOM, error-text and
# tool-call-presence. The remaining three decide whether the wrong answers are
# the MODEL's mistakes or pipeline damage:
#   - missing <answer>
#   - turn exhaustion (the 8-turn cap)
#   - truncated tool responses / malformed tool calls
exec >>/tmp/health.log 2>&1
echo "=== START $(date -Is)"
sudo -n /opt/conda-st/envs/spacetools-rl/bin/python - <<'PY'
import json, re, collections
for b in ["robospatial","reflocation","refplacement","refunseen"]:
    n=0; no_ans=0; multi_ans=0; exhaust=0; trunc=0; malformed=0
    turns=collections.Counter(); wrong=[]
    for line in open(f"/workspace/eval_out/{b}/0.jsonl"):
        r=json.loads(line); out=r.get("output") or ""; n+=1
        na = out.count("<answer>")
        if na==0: no_ans+=1
        if na>1: multi_ans+=1
        t = out.count("<tool_call>")
        turns[t]+=1
        if t>=8: exhaust+=1
        if "truncated" in out.lower() or "max_tool_response" in out: trunc+=1
        # a tool_call block whose JSON will not parse
        for m in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", out, re.S):
            try: json.loads(m)
            except Exception: malformed+=1
        if (r.get("acc") or 0.0) < 1.0 and len(wrong)<2:
            wrong.append(r)
    print("### %-13s n=%-4d no_answer=%-3d multi_answer=%-3d turn_exhausted=%-3d truncated=%-3d malformed_call=%d"
          % (b,n,no_ans,multi_ans,exhaust,trunc,malformed))
    print("      tool_calls per sample:", dict(sorted(turns.items())))
PY

echo
echo "--- two wrong answers from robospatial, verbatim, to see what the errors look like"
sudo -n /opt/conda-st/envs/spacetools-rl/bin/python - <<'PY'
import json
shown=0
for line in open("/workspace/eval_out/robospatial/0.jsonl"):
    r=json.loads(line)
    if (r.get("acc") or 0.0) < 1.0:
        print("=== index=%s acc=%s" % (r.get("index"), r.get("acc")))
        print("--- gt:", " ".join((r.get("gts") or "").split())[:200])
        print("--- output:", " ".join((r.get("output") or "").split())[:700])
        print()
        shown+=1
        if shown==2: break
PY
echo "=== DONE $(date -Is)"
