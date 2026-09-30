#!/bin/bash
# usage: bash /root/p0_go.sh          -> launches itself in background (idempotent via flock)
#        bash /root/p0_go.sh run      -> does the work
if [ "$1" != "run" ]; then
  nohup setsid flock -n /workspace/p0_setup.lock bash /root/p0_go.sh run >>/workspace/p0_setup.log 2>&1 </dev/null &
  sleep 2; echo "launched"; exit 0
fi
export HF_TOKEN=$(cat /root/.hf_token)
export HF_HOME=/workspace/hf
export MY_SFT_REPO=qzpm55555/spacetools-sft-v1-4xa6000
export MY_SFT_REV=91fd4bdf6d2ffe3b355cc650e71925c6e91a15c5
export P7_STEP=85
export KV_POOL_GB=24
cd /workspace/p7repo || exit 1
sed -n '1,/^step "8\./p' EVAL_FROM_SCRATCH.sh | head -n -1 >/workspace/p7repo/setup_only.sh
date
bash /workspace/p7repo/setup_only.sh
echo "SETUP_EXIT=$?"
date
