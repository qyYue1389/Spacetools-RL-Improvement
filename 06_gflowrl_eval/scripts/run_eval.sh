#!/usr/bin/env bash
set -u
mkdir -p /root/logs
exec >> /root/logs/eval.log 2>&1
echo "==== start $(date -u +'%F %T') UTC"
# token 先取,再开 set -x —— 否则 trace 会把它写进日志
export HF_TOKEN="$(tr -d '\r\n' < /root/.hf_token)"
[ -n "$HF_TOKEN" ] || { echo "token 空,停"; exit 1; }
set -x
export HF_HOME=/workspace/hf
export PATH=/opt/conda-st/bin:$PATH
cd /root
hf download qzpm55555/spacetools-p7-gflowrl-cprime-8xa40 \
    --include "EVAL_FROM_SCRATCH.sh" --include "parse_dump.py" --local-dir /root || exit 1
grep -c BASH_ENV /root/EVAL_FROM_SCRATCH.sh
grep -c "2b. 修 BENCHMARKS" /root/EVAL_FROM_SCRATCH.sh
echo "EVAL_RUN" > /root/logs/status_eval
P7_STEP=85 bash /root/EVAL_FROM_SCRATCH.sh
RC=$?
echo "EVAL_FROM_SCRATCH.sh 退出码 = $RC"
if [ "$RC" -eq 0 ]; then echo "EVAL_DONE_PASS" > /root/logs/status_eval; else echo "EVAL_DONE_FAIL" > /root/logs/status_eval; fi
echo "==== end $(date -u +'%F %T') UTC"
