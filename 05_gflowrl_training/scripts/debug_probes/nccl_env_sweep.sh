#!/usr/bin/env bash
# 4-GPU NCCL all-reduce probe: for an NCCL hang where the very first collective times out,
# replace a 15-minute training launch with a minimal reproduction with a 40s timeout, and try env combinations one by one.
# Clue: /sys/class/infiniband has RoCE devices (rocep177s0f0/f1),
# but /dev/infiniband does not exist — NCCL recognizes the IB HCA from sysfs, and the verbs path cannot open the device.
set -uo pipefail
mkdir -p /root/logs
exec >> /root/logs/nccl_probe.log 2>&1
mark(){ echo "$1" > /root/logs/status_nccl; }
say(){ echo "[$(date +%H:%M:%S)] $*"; }
PY=/opt/conda-st/envs/spacetools-rl/bin/python
run(){
  local name="$1"; shift
  say "──────── $name ────────"
  say "env: $*"
  ( export CUDA_VISIBLE_DEVICES=4,5,6,7 NCCL_DEBUG=WARN PROBE_PORT=$((29500 + RANDOM % 400))
    for kv in "$@"; do export "$kv"; done
    timeout 100 "$PY" /root/ncclprobe.py 4 ) 2>&1 | grep -aE "ALLREDUCE_OK|NCCL WARN|Timeout|timed out|Error|error|Aborted" | head -6
  say "$name done"
}
mark NCCL_PROBE
say "════════ NCCL probe ════════"
run "A default" DUMMY=1
run "B IB off" NCCL_IB_DISABLE=1
run "C IB off + eth0 specified" NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0
run "D IB off + P2P off" NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
run "E IB off + SHM off" NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1
mark NCCL_PROBE_DONE
say "════════ probe finished ════════"
