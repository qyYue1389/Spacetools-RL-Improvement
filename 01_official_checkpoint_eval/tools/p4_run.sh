#!/bin/bash
# P4: run one benchmark, with a GPU trace alongside it.
#   usage: bash /workspace/p4_run.sh <benchmark>
set -eo pipefail
# A crashed sglang worker (31 GiB RSS) wrote a 50 GB core into
# /var/lib/vastai_kaalia/data/ and filled the 50 GB CONTAINER disk mid-run.
# The volume had room; the container disk did not.  Inherited by all children.
ulimit -c 0
B="${1:?usage: p4_run.sh <benchmark>}"
RUN="${2:-run1}"                       # run1/run2/run3 for the repeat benchmarks

# A pasted-together command line once turned the label into "run2bash", which
# quietly created a whole parallel results tree. Labels are run<N>, nothing else.
if ! [[ "$RUN" =~ ^run[0-9]+$ ]]; then
    echo "REFUSING: run label '"'"'$RUN'"'"' is not of the form run<N>."
    echo "Did the command line get pasted together?"
    exit 1
fi

source /workspace/env.sh
OUT=/workspace/experiments/p4/$RUN
mkdir -p "$OUT" /workspace/logs

if [ -f "$OUT/$B/0.jsonl" ]; then
    echo "REFUSING: $OUT/$B/0.jsonl already exists."
    echo "Delete it first if you really mean to re-run, or pass a different run label."
    exit 1
fi

TRACE=/workspace/logs/p4-$RUN-$B-gpu.txt
LOG=/workspace/logs/p4-$RUN-$B.log
nohup "$(dirname "$0")/gputrace.sh" > "$TRACE" 2>&1 &
TPID=$!
trap 'kill $TPID 2>/dev/null' EXIT

echo "=== $B ($RUN) ==="
echo "  dump  -> $OUT/$B/0.jsonl"
echo "  log   -> $LOG"
echo "  trace -> $TRACE"
echo
date +"start %H:%M:%S"
cd /workspace/SpaceTools/SpaceTools-RL
DATA_DIR=/workspace/eval-benchmarks OUTPUT_DIR="$OUT" \
    bash examples/toolshed/run_eval.sh "$SPACETOOLS_CKPT" "$B" > "$LOG" 2>&1 || true
date +"end   %H:%M:%S"

if grep -q "EVALUATION COMPLETE" "$LOG"; then
    echo "eval finished. NOW RUN THE CHECK:"
    echo "    bash tools/p4_check.sh $B $RUN"
else
    echo "*** eval did NOT complete. Look at $LOG ***"
    grep -nE "RayTaskError|OutOfMemoryError|^Traceback" "$LOG" | head -5
    exit 1
fi
