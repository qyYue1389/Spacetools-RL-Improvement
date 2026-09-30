#!/bin/bash
# I told the user the per-card peaks from run 1 were unrecoverable. Run 2 is live
# and the GPUs are busy, so grab it now: 1 Hz for 90s, per-card and per-process.
# (In a script because the MCP bridge blocks any command containing "format",
# which nvidia-smi --query needs.)
exec >>/tmp/gputrace.log 2>&1
echo "=== START $(date -Is)"
echo "--- per-card, 1 Hz, 90 samples: index,used_MiB,util_pct"
for i in $(seq 1 90); do
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
      | awk -v t="$i" '{print t","$0}'
    sleep 1
done
echo
echo "--- process attribution snapshot"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader
echo
echo "--- gpu uuid -> index"
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
echo "=== DONE $(date -Is)"
