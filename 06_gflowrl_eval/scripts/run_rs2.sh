#!/usr/bin/env bash
# RoboSpatial 第二次独立运行,用来把「61.43% 没涨」从单次读数升级成区间结论。
# 条件与第一次一致:gmu=0.545 · NUM_GPUS=4 EVAL_GPUS=1 · 同一个 ckpt。
# 唯一差别是 vlm 的 num_gpus 声明 0.6 -> 1.0(只改 Ray 的逻辑预留)。robospatial
# 全程只调 roborefer,vlm 一次都不会被调用,所以这个差别不进入分数,只影响摆放。
# 这台机器 GPU0/1 是 49140 MiB(ECC off)、GPU2/3 是 46068 MiB(ECC on),而 gmu 是
# 整卡比例 —— 策略落在哪张卡上 KV 池就差 1.7 GB。所以顺带 1 Hz/60s 记一份显存轨迹,
# 万一两次读数差得多,先看是不是落卡不同造成的。
set -u
mkdir /root/.eval_rs2.lock || { echo "已有实例在跑,退出"; exit 0; }
mkdir -p /root/logs
exec >>/root/logs/eval_rs2.log 2>&1
echo "==== start $(date -u +'%F %T') UTC  pid $$"
export HF_TOKEN="$(tr -d '\r\n' </root/.hf_token)"
[ -n "$HF_TOKEN" ] || { echo "token 空,停"; exit 1; }
set -x
export HF_HOME=/workspace/hf
export CONDA_ROOT=/opt/conda-st
export PATH=/opt/conda-st/bin:$PATH
export BASH_ENV=/opt/conda-st/etc/profile.d/conda.sh
export ROBOREFER_MODEL=/workspace/checkpoints/RoboRefer-8B-SFT
export DEPTH_CHECKPOINT=/workspace/checkpoints/depth_pro.pt
export NUM_GPUS=4
export EVAL_GPUS=1
export OUTPUT_DIR=/workspace/exp/p7_eval_step85_rs2
MODEL=/workspace/checkpoints/p7-step85
RUN_EVAL=/opt/spacetools/SpaceTools-RL/examples/toolshed/run_eval.sh

export PATH=/opt/conda-st/envs/spacetools-rl/bin:$PATH
ray stop --force || true
sleep 5

nohup bash -c 'while true; do date -u +"%F %T"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; sleep 60; done' >>/root/logs/gpu_rs2.log 2>&1 &
SAMPLER=$!

echo "EVAL_RS2_RUN" >/root/logs/status_eval_rs2
cd /opt/spacetools/SpaceTools-RL || exit 1
bash "$RUN_EVAL" "$MODEL" robospatial
RC=$?
set +x
kill $SAMPLER
echo "run_eval.sh 退出码 = $RC"
MARK=$(grep -c 'EVALUATION COMPLETE' /root/logs/eval_rs2.log)
NOROUTER=$(grep -c 'Could not find ToolRouterActor' /root/logs/eval_rs2.log)
OOM=$(grep -c OutOfMemoryError "$OUTPUT_DIR/robospatial/eval.log")
N=$(wc -l < "$OUTPUT_DIR/robospatial/0.jsonl")
echo "COMPLETE=$MARK · router失联=$NOROUTER · OOM=$OOM · 样本=$N"
if [ "$MARK" -ge 1 ] && [ "$OOM" -eq 0 ] && [ "$NOROUTER" -eq 0 ] && [ "$N" -eq 350 ]; then
    echo "EVAL_RS2_PASS" >/root/logs/status_eval_rs2
else
    echo "EVAL_RS2_FAIL" >/root/logs/status_eval_rs2
fi
rmdir /root/.eval_rs2.lock
echo "==== end $(date -u +'%F %T') UTC"
