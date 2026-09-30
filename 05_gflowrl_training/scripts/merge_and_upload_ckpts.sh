#!/usr/bin/env bash
# 训练跑完后:merge 三个 ckpt -> HF 格式 -> 上传到 HF 私有 repo。
# 幂等:已 merge 的跳过;上传失败可重跑。
set -u
mkdir -p /root/logs
exec >> /root/logs/save_ckpts.log 2>&1
echo "=== start $(date -u +'%F %T')"

export PATH=/opt/conda-st/envs/spacetools-rl/bin:$PATH
eval "$(grep '^export HF_TOKEN=' /root/.bashrc)"
echo "HF_TOKEN len=${#HF_TOKEN}"

OUT=/workspace/exp/p7_cprime/rl_output
SNAP=/workspace/exp/p7_cprime/eval_ckpts
MERGED=/workspace/merged
PROV=$MERGED/provenance
REPO=qzpm55555/spacetools-p7-gflowrl-cprime-8xa40
VERL_DIR=/opt/spacetools/SpaceTools-RL

echo "WAIT_TRAIN" > /root/logs/status_save
while ps -eo cmd | grep -q '[f]ull_train.sh'; do sleep 60; done
echo "训练进程已退出 $(date -u +'%F %T'),status_full=$(cat /root/logs/status_full)"
sleep 120

NEWEST=$(ls -d "$OUT"/global_step_* | sed 's#.*global_step_##' | sort -n | tail -1)
echo "最后一个完整 ckpt: global_step_$NEWEST"

echo "MERGE" > /root/logs/status_save
mkdir -p "$MERGED"
cd "$VERL_DIR"
for PAIR in "$NEWEST:$OUT/global_step_$NEWEST/actor" "60:$SNAP/global_step_60/actor" "30:$SNAP/global_step_30/actor"; do
    STEP=${PAIR%%:*}
    SRC=${PAIR#*:}
    DST=$MERGED/global_step_$STEP
    if [ -f "$DST/config.json" ]; then echo "step $STEP 已 merge,跳过"; continue; fi
    if [ ! -d "$SRC" ]; then echo "step $STEP 源不存在:$SRC,跳过"; continue; fi
    echo "--- merge step $STEP <- $SRC  $(date -u +'%T')"
    python -m verl.model_merger merge --backend fsdp --local_dir "$SRC" --target_dir "$DST"
    echo "merge step $STEP 退出码 $?  $(du -sh "$DST" | cut -f1)"
done

mkdir -p "$PROV"
cp /root/logs/p7_metrics.csv /root/full_train.sh /root/logs/ckpt_janitor.log "$PROV"/
cp /root/logs/status_full "$PROV"/status_full.txt
git -C "$VERL_DIR" rev-parse HEAD > "$PROV/SpaceTools-RL.commit"
git -C "$VERL_DIR" status --porcelain > "$PROV/SpaceTools-RL.dirty"
gzip -c /root/logs/full_train.log > "$PROV/full_train.log.gz"
ls -l "$PROV"

echo "UPLOAD" > /root/logs/status_save
for STEP in $NEWEST 60 30; do
    D=$MERGED/global_step_$STEP
    [ -f "$D/config.json" ] || { echo "step $STEP 没 merge 成功,不上传"; continue; }
    echo "--- upload step $STEP $(date -u +'%T')"
    hf upload "$REPO" "$D" "global_step_$STEP" --repo-type model --private
    echo "upload step $STEP 退出码 $?"
done
hf upload "$REPO" "$PROV" provenance --repo-type model --private
echo "upload provenance 退出码 $?"

echo "=== 远端文件清单"
python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "qzpm55555/spacetools-p7-gflowrl-cprime-8xa40"
fs = sorted(api.list_repo_files(repo))
print(len(fs), "个文件")
for f in fs:
    print(" ", f)
PY
echo "DONE" > /root/logs/status_save
echo "=== end $(date -u +'%F %T')"
