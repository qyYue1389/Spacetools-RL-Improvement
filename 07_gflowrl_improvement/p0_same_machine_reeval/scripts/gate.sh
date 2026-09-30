#!/bin/bash
PY=/opt/conda-st/envs/spacetools-rl/bin/python
cd /workspace/p7repo
pass=0; fail=0
for d in /workspace/exp/p0/*/*/*/0.jsonl; do
  if "$PY" parse_dump.py --strict "$d" >/tmp/pd.out 2>&1; then
    pass=$((pass+1))
  else
    fail=$((fail+1)); echo "FAIL $d"; tail -20 /tmp/pd.out
  fi
done
echo "STRICT pass=$pass fail=$fail"
echo "--- aggregate health across all dumps ---"
for d in /workspace/exp/p0/*/*/*/0.jsonl; do
  "$PY" parse_dump.py --strict "$d" 2>&1 | grep -E "OOM samples|cut-off|truncated|hit max turns|no <answer>|malformed|tool failures"
done | awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+$/){s[$1" "$2]+=$i; break}} END{for(k in s) print k, s[k]}'
