#!/usr/bin/env bash
# Rerun the four benchmarks contaminated by OOM.
# Cause: vlm (Molmo, measured 30.2 GiB, upstream vlm.py loads in fp32) only declares num_gpus=0.6,
# so Ray squeezes it onto the same GPU as depth_estimator/sam2 -> 47.3/47.4 GiB -> a tool asking for another 576 MiB
# OOMs -> Toolshed silently returns an error -> points lost. The vlm declaration has been changed to 1.0 (a GPU to itself).
# Tool demand 2.7 + policy whole GPU 1.0 = 3.7 <= 4.0. gmu is still 0.545; comparison definition unchanged.
#
# mkdir is atomic: a duplicate launch (e.g. a timeout retry on the caller side) exits immediately, so two instances never
# ray stop --force each other and kill each other's toolshed.
set -u
mkdir /root/.eval_fix.lock || { echo "an instance is already running (lock /root/.eval_fix.lock), exiting"; exit 0; }
mkdir -p /root/logs
exec >>/root/logs/eval_fix2.log 2>&1
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
echo "run_eval.sh exit code = $RC"
# The criterion is not the exit code: run_eval.sh's cleanup trap killing the background toolshed carries out SIGTERM (15),
# which is a normal wrap-up. The real criterion is "completion marker + sample count + OOM count + no lost router".
MARK=$(grep -c 'EVALUATION COMPLETE' /root/logs/eval_fix2.log)
NOROUTER=$(grep -c 'Could not find ToolRouterActor' /root/logs/eval_fix2.log)
echo "EVALUATION COMPLETE = $MARK times · router lost = $NOROUTER times"
wc -l "$OUTPUT_DIR"/*/0.jsonl
OOM=0
for B in blinkdepth cvb3ddepth boppose bopgrasp; do
    N=$(grep -c OutOfMemoryError "$OUTPUT_DIR/$B/eval.log")
    echo "OOM $B = $N"
    OOM=$((OOM + N))
done
echo "OOM total = $OOM"
if [ "$MARK" -ge 1 ] && [ "$OOM" -eq 0 ] && [ "$NOROUTER" -eq 0 ]; then
    echo "EVAL_FIX2_PASS" >/root/logs/status_eval_fix
else
    echo "EVAL_FIX2_FAIL" >/root/logs/status_eval_fix
fi
rmdir /root/.eval_fix.lock
echo "==== end $(date -u +'%F %T') UTC"
