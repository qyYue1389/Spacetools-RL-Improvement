#!/bin/bash
# P7 · 第二组 5 次采样 (passk2).  一个 benchmark 一次调用。
#   usage: bash tools/p7/p7_passk2_run.sh <benchmark> <run>
#
# 与第一组 (p6/passk/, 80 GB @ gmu=0.25) 的可比性靠 KV 池大小对齐:
#   80 GB x 0.25 = 20 GB      <- 第一组
#   40 GB x 0.50 = 20 GB      <- 本机, run_eval.sh 的原值, 不要改
# 见偏离 [23]。照抄第一组日志里的 0.25 会得到 10 GB 池, 两组不可比。
#
# 与 run_eval.sh 的差异只有 run_eval_passk2.sh 里那 7 行 (采样 4 行 + rollout.n
# + calculate_log_probs + dump_token_diagnostics)。
set -eo pipefail
# A crashed sglang worker (31 GiB RSS) wrote a 50 GB core into
# /var/lib/vastai_kaalia/data/ and filled the 50 GB CONTAINER disk mid-run.
# The volume had room; the container disk did not.  Inherited by all children.
ulimit -c 0
B="${1:?usage: p7_passk2_run.sh <benchmark> <run>}"
RUN="${2:?usage: p7_passk2_run.sh <benchmark> <run>}"
if ! [[ "$RUN" =~ ^run[0-9]+$ ]]; then
    echo "REFUSING: run label '$RUN' is not of the form run<N>."
    exit 1
fi

source /workspace/env.sh
OUT=/workspace/experiments/passk2/$RUN
mkdir -p "$OUT" /workspace/logs
if [ -f "$OUT/$B/0.jsonl" ]; then
    echo "REFUSING: $OUT/$B/0.jsonl already exists."
    exit 1
fi

TRACE=/workspace/logs/passk2-$RUN-$B-gpu.txt
LOG=/workspace/logs/passk2-$RUN-$B.log
nohup /workspace/spacetools-repro/tools/gputrace.sh > "$TRACE" 2>&1 &
TPID=$!
trap 'kill $TPID 2>/dev/null' EXIT

echo "=== $B ($RUN) · passk2 · n=5 T=1.0 · gmu=0.5 (20 GB KV pool) ==="
echo "  dump  -> $OUT/$B/0.jsonl"
echo "  log   -> $LOG"
date +"start %H:%M:%S"
cd /workspace/SpaceTools/SpaceTools-RL
DATA_DIR=/workspace/eval-benchmarks OUTPUT_DIR="$OUT" \
    bash examples/toolshed/run_eval_passk2.sh "$SPACETOOLS_CKPT" "$B" > "$LOG" 2>&1 || true
date +"end   %H:%M:%S"

if grep -q "EVALUATION COMPLETE" "$LOG"; then
    echo "eval finished. NOW RUN:"
    echo "    python3 tools/parse_dump.py $OUT/$B --strict"
else
    echo "*** eval did NOT complete. Look at $LOG ***"
    grep -nE "RayTaskError|OutOfMemoryError|^Traceback" "$LOG" | head -5
    exit 1
fi
