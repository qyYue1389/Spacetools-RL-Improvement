#!/usr/bin/env bash
set -u
mkdir -p /root/logs
exec >> /root/logs/eval.log 2>&1
echo "==== start $(date -u +'%F %T') UTC"
# Get the token first, then turn on set -x — otherwise the trace writes it into the log
export HF_TOKEN="$(tr -d '\r\n' < /root/.hf_token)"
[ -n "$HF_TOKEN" ] || { echo "token empty, stopping"; exit 1; }
set -x
export HF_HOME=/workspace/hf
export PATH=/opt/conda-st/bin:$PATH
cd /root
hf download qyYue1389/spacetools-p7-gflowrl-cprime-8xa40 \
    --include "EVAL_FROM_SCRATCH.sh" --include "parse_dump.py" --local-dir /root || exit 1
grep -c BASH_ENV /root/EVAL_FROM_SCRATCH.sh
grep -cE "2b. (修|Fix) BENCHMARKS" /root/EVAL_FROM_SCRATCH.sh
echo "EVAL_RUN" > /root/logs/status_eval
P7_STEP=85 bash /root/EVAL_FROM_SCRATCH.sh
RC=$?
echo "EVAL_FROM_SCRATCH.sh exit code = $RC"
if [ "$RC" -eq 0 ]; then echo "EVAL_DONE_PASS" > /root/logs/status_eval; else echo "EVAL_DONE_FAIL" > /root/logs/status_eval; fi
echo "==== end $(date -u +'%F %T') UTC"
