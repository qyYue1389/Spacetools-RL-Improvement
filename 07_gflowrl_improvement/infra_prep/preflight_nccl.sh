#!/usr/bin/env bash
# Pre-flight for a freshly rented training machine: is GPU peer-to-peer usable, and how fast?
# P7 lost time to a container where P2P looked fine in `nvidia-smi topo -p2p r` but every NCCL
# collective hung (training report §7.1); the workaround NCCL_P2P_DISABLE=1 routes all-reduce
# through host memory and slows update_actor.  Run this BEFORE starting training.
#
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash preflight_nccl.sh [python]     # the TRAINING GPUs
#
# Verdict:  P2P_OK        -> train with P2P on (do not set NCCL_P2P_DISABLE)
#           P2P_BROKEN    -> only works with NCCL_P2P_DISABLE=1; expect slower update_actor,
#                            consider another machine
#           NCCL_BROKEN   -> neither works; do not start training
set -uo pipefail
PY="${1:-/opt/conda-st/envs/spacetools-rl/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N=$(echo "${CUDA_VISIBLE_DEVICES:-0,1,2,3}" | tr ',' '\n' | grep -c .)
echo "== topology (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}, n=$N)"
nvidia-smi topo -m 2>/dev/null | head -12
run() {  # $1 = label, rest = env assignments
  local label="$1"; shift
  local out
  out=$(env NCCL_IB_DISABLE=1 "$@" timeout 100 "$PY" "$HERE/nccl_bw_probe.py" "$N" 2>&1 | grep -aE "PROBE_|Timeout|timed out|Error" | head -3)
  echo "  $label: ${out:-no output (hung or crashed)}"
  [[ "$out" == PROBE_OK* ]]
}
echo "== all-reduce probe"
run "P2P on " NCCL_P2P_DISABLE=0; p2p=$?
run "P2P off" NCCL_P2P_DISABLE=1; nop2p=$?
if [ $p2p -eq 0 ]; then echo "VERDICT: P2P_OK"; exit 0; fi
if [ $nop2p -eq 0 ]; then echo "VERDICT: P2P_BROKEN  (set NCCL_P2P_DISABLE=1, or rent another machine)"; exit 2; fi
echo "VERDICT: NCCL_BROKEN"; exit 1
