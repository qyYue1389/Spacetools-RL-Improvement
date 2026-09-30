#!/bin/bash
# P0: 3 ckpts x (robospatial+blinkdepth) x3 runs + 3 ckpts x RefSpatial trio x1
if [ "$1" != "run" ]; then
  nohup setsid flock -n /workspace/p0_eval.lock bash /root/p0_eval.sh run >>/workspace/p0_eval.log 2>&1 </dev/null &
  sleep 2; echo launched; exit 0
fi
set -uo pipefail
ulimit -c 0                      # an OOM core dump once filled the container disk (P7 GPU_RESULTS:61)
export HF_TOKEN=$(cat /root/.hf_token)
export HF_HOME=/workspace/hf
CK=/workspace/checkpoints
RUN_EVAL=/opt/spacetools/SpaceTools-RL/examples/toolshed/run_eval.sh
export ROBOREFER_MODEL="$CK/RoboRefer-8B-SFT"
export DEPTH_CHECKPOINT="$CK/depth_pro.pt"
export CONDA_ROOT=/opt/conda-st
export PATH=/opt/conda-st/bin:/opt/conda-st/envs/spacetools-rl/bin:$PATH
export BASH_ENV=/opt/conda-st/etc/profile.d/conda.sh
export NUM_GPUS=4
export EVAL_GPUS=1
cd /opt/spacetools/SpaceTools-RL || exit 1

model_of() {
  case "$1" in
    sft) echo "$CK/sft-ckpt" ;;
    p4)  echo "$CK/p4-official" ;;
    cp)  echo "$CK/p7-step85" ;;
    *)   echo "BAD" ;;
  esac
}

one() {
  local tag=$1 run=$2; shift 2
  local out=/workspace/exp/p0/$tag/$run
  if [ -f "$out/.done" ]; then echo "== skip $tag/$run (already done)"; return 0; fi
  mkdir -p "$out"
  echo "== START $tag/$run : $* @ $(date -Is)"
  ray stop --force >/dev/null 2>&1
  rm -rf /root/tmp/ray
  sleep 5
  OUTPUT_DIR="$out" bash "$RUN_EVAL" "$(model_of "$tag")" "$@"
  local rc=$? ok=1 b n
  # run_eval.sh returns 15 even on success (it kills the backgrounded toolshed),
  # so "done" means: every requested dump exists AND its log has zero OOM.
  for b in "$@"; do
    if [ ! -s "$out/$b/0.jsonl" ]; then ok=0; echo "   $b: NO DUMP"; continue; fi
    n=$(grep -ac OutOfMemoryError "$out/$b/eval.log")
    echo "   $b: OOM=$n"
    [ "$n" != "0" ] && ok=0
  done
  echo "== END $tag/$run rc=$rc ok=$ok @ $(date -Is)"
  [ "$ok" -eq 1 ] && touch "$out/.done"
  return 0
}

for r in run1 run2 run3; do
  for t in sft p4 cp; do one "$t" "$r" robospatial blinkdepth; done
done
for t in sft p4 cp; do one "$t" ref1 reflocation refplacement refunseen; done
echo "ALL DONE $(date -Is)"
