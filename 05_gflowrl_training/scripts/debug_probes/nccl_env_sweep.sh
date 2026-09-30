#!/usr/bin/env bash
# 4 卡 NCCL all-reduce 探针:一次首个 collective 就超时的 NCCL 挂死,
# 用 40s 超时的最小复现替代 15 分钟的训练启动,逐个试 env 组合。
# 线索:/sys/class/infiniband 里有 RoCE 设备(rocep177s0f0/f1),
# 但 /dev/infiniband 不存在 —— NCCL 从 sysfs 认出 IB HCA,走 verbs 路径打不开设备。
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
  say "$name 结束"
}
mark NCCL_PROBE
say "════════ NCCL 探针 ════════"
run "A 默认" DUMMY=1
run "B IB 关" NCCL_IB_DISABLE=1
run "C IB 关 + 指定 eth0" NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0
run "D IB 关 + P2P 关" NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
run "E IB 关 + SHM 关" NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1
mark NCCL_PROBE_DONE
say "════════ 探针结束 ════════"
