#!/usr/bin/env bash
# Second independent RoboSpatial run, to upgrade "61.43%, no gain" from a single reading to an interval conclusion.
# Same conditions as the first run: gmu=0.545 · NUM_GPUS=4 EVAL_GPUS=1 · same ckpt.
# The only difference is vlm's num_gpus declaration 0.6 -> 1.0 (only changes Ray's logical reservation). robospatial
# only calls roborefer throughout and never calls vlm once, so this difference doesn't enter the score, it only affects placement.
# On this machine GPU0/1 are 49140 MiB (ECC off) and GPU2/3 are 46068 MiB (ECC on), while gmu is a
# fraction of the whole GPU — the KV pool differs by 1.7 GB depending on which GPU the policy lands on. So also record a GPU memory trace at 1 Hz/60s,
# in case the two readings differ a lot, first check whether landing on different GPUs caused it.
set -u
mkdir /root/.eval_rs2.lock || { echo "an instance is already running, exiting"; exit 0; }
mkdir -p /root/logs
exec >>/root/logs/eval_rs2.log 2>&1
echo "==== start $(date -u +'%F %T') UTC  pid $$"
export HF_TOKEN="$(tr -d '\r\n' </root/.hf_token)"
[ -n "$HF_TOKEN" ] || { echo "token empty, stopping"; exit 1; }
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
echo "run_eval.sh exit code = $RC"
MARK=$(grep -c 'EVALUATION COMPLETE' /root/logs/eval_rs2.log)
NOROUTER=$(grep -c 'Could not find ToolRouterActor' /root/logs/eval_rs2.log)
OOM=$(grep -c OutOfMemoryError "$OUTPUT_DIR/robospatial/eval.log")
N=$(wc -l < "$OUTPUT_DIR/robospatial/0.jsonl")
echo "COMPLETE=$MARK · router_lost=$NOROUTER · OOM=$OOM · samples=$N"
if [ "$MARK" -ge 1 ] && [ "$OOM" -eq 0 ] && [ "$NOROUTER" -eq 0 ] && [ "$N" -eq 350 ]; then
    echo "EVAL_RS2_PASS" >/root/logs/status_eval_rs2
else
    echo "EVAL_RS2_FAIL" >/root/logs/status_eval_rs2
fi
rmdir /root/.eval_rs2.lock
echo "==== end $(date -u +'%F %T') UTC"
