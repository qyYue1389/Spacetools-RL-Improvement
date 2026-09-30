#!/usr/bin/env bash
# 重跑被 OOM 污染的四个 benchmark。
# 病因: vlm(Molmo,实测 30.2 GiB,上游 vlm.py 按 fp32 加载)只声明 num_gpus=0.6,
# 被 Ray 和 depth_estimator/sam2 挤在同一张卡 -> 47.3/47.4 GiB -> 工具再要 576 MiB
# 就 OOM -> Toolshed 静默返回错误 -> 掉分。已把 vlm 声明改成 1.0(独占一张卡)。
# 工具需求 2.7 + 策略整卡 1.0 = 3.7 <= 4.0。gmu 仍是 0.545,对照口径不变。
#
# mkdir 是原子的:重复启动(比如调用侧超时重试)会直接退出,不会出现两个实例
# 互相 ray stop --force 把对方的 toolshed 打死。
set -u
mkdir /root/.eval_fix.lock || { echo "已有实例在跑(锁 /root/.eval_fix.lock),退出"; exit 0; }
mkdir -p /root/logs
exec >>/root/logs/eval_fix2.log 2>&1
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
export OUTPUT_DIR=/workspace/exp/p7_eval_step85_fix2
MODEL=/workspace/checkpoints/p7-step85
RUN_EVAL=/opt/spacetools/SpaceTools-RL/examples/toolshed/run_eval.sh

grep -n "'vlm':" "$RUN_EVAL" | head -2
grep -n 'gpu_memory_utilization' "$RUN_EVAL" | head -2

export PATH=/opt/conda-st/envs/spacetools-rl/bin:$PATH
ray stop --force || true
sleep 5

echo "EVAL_FIX2_RUN" >/root/logs/status_eval_fix
cd /opt/spacetools/SpaceTools-RL || exit 1
bash "$RUN_EVAL" "$MODEL" blinkdepth cvb3ddepth boppose bopgrasp
RC=$?
set +x
echo "run_eval.sh 退出码 = $RC"
# 判据不是退出码: run_eval.sh 的 cleanup 陷阱杀后台 toolshed 会把 SIGTERM(15) 带出来,
# 那是正常收尾。真正的判据是「跑完标记 + 样本数 + OOM 数 + 没有 router 失联」。
MARK=$(grep -c 'EVALUATION COMPLETE' /root/logs/eval_fix2.log)
NOROUTER=$(grep -c 'Could not find ToolRouterActor' /root/logs/eval_fix2.log)
echo "EVALUATION COMPLETE = $MARK 次 · router 失联 = $NOROUTER 次"
wc -l "$OUTPUT_DIR"/*/0.jsonl
OOM=0
for B in blinkdepth cvb3ddepth boppose bopgrasp; do
    N=$(grep -c OutOfMemoryError "$OUTPUT_DIR/$B/eval.log")
    echo "OOM $B = $N"
    OOM=$((OOM + N))
done
echo "OOM 合计 = $OOM"
if [ "$MARK" -ge 1 ] && [ "$OOM" -eq 0 ] && [ "$NOROUTER" -eq 0 ]; then
    echo "EVAL_FIX2_PASS" >/root/logs/status_eval_fix
else
    echo "EVAL_FIX2_FAIL" >/root/logs/status_eval_fix
fi
rmdir /root/.eval_fix.lock
echo "==== end $(date -u +'%F %T') UTC"
