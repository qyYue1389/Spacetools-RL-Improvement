#!/bin/bash
# P4 gate: a non-zero exit here means this benchmark has NO RESULT.
#   usage: bash tools/p4_check.sh <benchmark> [run<N>]
set -eo pipefail
B="${1:?usage: p4_check.sh <benchmark> [run]}"
RUN="${2:-run1}"
source /workspace/env.sh
OUT=/workspace/experiments/p4/$RUN
DUMP="$OUT/$B/0.jsonl"
LOG=/workspace/logs/p4-$RUN-$B.log
TRACE=/workspace/logs/p4-$RUN-$B-gpu.txt

[ -f "$DUMP" ] || { echo "no dump at $DUMP"; exit 1; }

echo "=== rows ==="; wc -l < "$DUMP"
echo
echo "=== log must be free of these ==="
for pat in Traceback RayTaskError OutOfMemoryError; do
    printf "  %-18s %s\n" "$pat" "$(grep -c "$pat" "$LOG" || true)"
done
echo
echo "=== GPU peaks ==="
awk '{split($0,p,"|"); for(g=1;g<=4;g++){split(p[g],a,", "); if(a[2]>m[g])m[g]=a[2]}}
     END{for(g=1;g<=4;g++) printf "  GPU%d  %6.1f GB  (%.0f%%)\n", g-1, m[g]/1024, m[g]*100/40960}' "$TRACE"
echo
echo "=== wall clock ==="
# the tracer stamps each line with epoch seconds, so its FIRST line is the start.
# (the file's own mtime is the LAST sample, which is why the naive version of
#  this went negative.)
T0=$(head -1 "$TRACE" | cut -d" " -f1)
T1=$(stat -c %Y "$DUMP")
if [ -n "$T0" ] && [ "$T0" -gt 0 ] 2>/dev/null; then
    echo "  $(( (T1-T0)/60 ))m $(( (T1-T0)%60 ))s   ($((T1-T0)) s, launch -> dump written)"
else
    echo "  (no trace timestamps)"
fi
echo
echo "=== parse_dump --strict ==="
conda run -n spacetools-rl python $(dirname "$0")/parse_dump.py "$DUMP" -v --strict
rc=$?
echo
if [ $rc -eq 0 ]; then
    mkdir -p /workspace/p4-dumps/$RUN/$B
    cp "$DUMP" /workspace/p4-dumps/$RUN/$B/
    cp "$LOG" "$TRACE" /workspace/p4-dumps/$RUN/$B/ 2>/dev/null || true
    echo "PASS. dump copied to /workspace/p4-dumps/$RUN/$B/"
else
    echo "*** STRICT FAILED -- this benchmark has NO RESULT. Do not continue. ***"
fi
exit $rc
