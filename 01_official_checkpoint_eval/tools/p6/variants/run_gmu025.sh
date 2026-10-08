#!/bin/bash
# Decisive experiment: gpu_memory_utilization 0.25 (reproduces 40GB's 20GB KV pool on 80GB)
# Identical to tools/p4_run.sh except for this parameter.
set -eo pipefail
# A crashed sglang worker (31 GiB RSS) wrote a 50 GB core into
# /var/lib/vastai_kaalia/data/ and filled the 50 GB CONTAINER disk mid-run.
# The volume had room; the container disk did not.  Inherited by all children.
ulimit -c 0
B="${1:?usage: run_gmu025.sh <benchmark> <run>}"
RUN="${2:?usage: run_gmu025.sh <benchmark> <run>}"
if ! [[ "$RUN" =~ ^run[0-9]+$ ]]; then echo "REFUSING: bad run label '$RUN'"; exit 1; fi

source /workspace/env.sh
OUT=/workspace/experiments/p4/$RUN
mkdir -p "$OUT" /workspace/logs
if [ -f "$OUT/$B/0.jsonl" ]; then
    echo "REFUSING: $OUT/$B/0.jsonl already exists."; exit 1
fi

TRACE=/workspace/logs/p4-$RUN-$B-gpu.txt
LOG=/workspace/logs/p4-$RUN-$B.log
nohup /workspace/spacetools-repro/tools/gputrace.sh > "$TRACE" 2>&1 &
TPID=$!
trap 'kill $TPID 2>/dev/null' EXIT

echo "=== $B ($RUN) · gpu_memory_utilization=0.25 ==="
echo "  dump  -> $OUT/$B/0.jsonl"
date +"start %H:%M:%S"
cd /workspace/SpaceTools/SpaceTools-RL
DATA_DIR=/workspace/eval-benchmarks OUTPUT_DIR="$OUT" \
    bash examples/toolshed/run_eval_gmu025.sh "$SPACETOOLS_CKPT" "$B" > "$LOG" 2>&1 || true
date +"end   %H:%M:%S"

if grep -q "EVALUATION COMPLETE" "$LOG"; then
    echo "eval finished."
else
    echo "*** eval did NOT complete. Look at $LOG ***"
    grep -nE "RayTaskError|OutOfMemoryError|^Traceback" "$LOG" | head -5
    exit 1
fi
