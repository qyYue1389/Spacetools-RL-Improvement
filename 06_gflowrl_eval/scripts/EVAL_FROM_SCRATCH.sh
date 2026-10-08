#!/usr/bin/env bash
# On a bare GPU machine, evaluate P7's RL checkpoint on all nine benchmarks from scratch.
#
#   export HF_TOKEN=hf_xxx          # needed for the private repos
#   P7_STEP=86 bash EVAL_FROM_SCRATCH.sh
#
# Every step is idempotent; rerunning after an interruption skips what's already done. Each step's criterion is "success marker seen", not "exit code 0"
# — this project's failure modes are heavily concentrated in "looked like it succeeded".
set -uo pipefail

P7_REPO="${P7_REPO:-qyYue1389/spacetools-p7-gflowrl-cprime-8xa40}"
P7_STEP="${P7_STEP:-85}"
ENV_REPO="${ENV_REPO:-qyYue1389/spacetools-eval-env}"
PKG="${PKG:-/workspace/envpkg}"
CK="${CK:-/workspace/checkpoints}"
MODEL="$CK/p7-step$P7_STEP"
EVAL_OUT="${EVAL_OUT:-/workspace/exp/p7_eval_step$P7_STEP}"
RUN_EVAL=/opt/spacetools/SpaceTools-RL/examples/toolshed/run_eval.sh
# Target KV pool size (GB). 24 = the pool size when the SFT baseline (61.00 ± 0.77) was measured:
# 48 GB GPU × gpu_memory_utilization 0.5. This knob is a **fraction of the whole GPU**, not an absolute value;
# switch GPUs without changing it and the pool changes — pool size changes batch composition -> floating-point reduction order -> near-tie
# samples flip. To compare against the SFT starting point, the pool has to be the same size.
KV_POOL_GB="${KV_POOL_GB:-24}"
export HF_HOME="${HF_HOME:-/workspace/hf}"
export HF_HUB_ENABLE_HF_TRANSFER=1

die(){ echo; echo "✗ $*"; exit 1; }
step(){ echo; echo "==== $* ===="; }

step "0. Machine acceptance check"
command -v nvidia-smi >/dev/null 2>&1 || die "no nvidia-smi — driver not installed. The first check is nvidia-smi -L, not nvidia-smi"
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
NGPU=$(nvidia-smi -L | wc -l)
MEMMIN=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "$NGPU" -ge 4 ] || die "only $NGPU GPUs. Tool logical reservation 3.0 + policy 1.0 = 4.0, need at least 4"
case "$CC" in
    8.0|8.6|8.9) ;;
    *) die "compute capability $CC is not covered. The environment package's self-built extensions only have cubins for sm_80/86/89 and no +PTX; H100 (9.0) cannot run them" ;;
esac
[ "${DRV:-0}" -ge 550 ] || die "driver $DRV < 550"
# GPU memory floor: P4 ran all nine benchmarks with this set of fractions on A100-40GB with zero OOM, so 40 GB is enough.
# Measured peaks: Molmo 34.6 GB (upstream vlm.py:116 swallows the dtype config, loads in fp32),
# the GPU with DepthPro×2 at 25.7 GB, the policy GPU at gmu×whole GPU.
[ "$MEMMIN" -ge 38000 ] || die "smallest per-GPU memory ${MEMMIN} MiB < 40 GB. A single Molmo actor needs 34.6 GB; it won't fit.
    To run on smaller GPUs you must recompute the TOOL_CONFIGS fractions and possibly go 8-bit — that makes scores not comparable with the paper; don't force it:
    OOM won't crash the script, it just silently loses points (observed 88.3% -> 83.1%, without a sound)."
[ -n "${HF_TOKEN:-}" ] || die "no HF_TOKEN. The environment package and the ckpt are both private repos"
echo "  ${NGPU} GPUs · smallest GPU memory ${MEMMIN} MiB · cc $CC · driver $DRV · HF_TOKEN len ${#HF_TOKEN}"
AVAIL_ROOT=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
AVAIL_WS=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
echo "  container disk free ${AVAIL_ROOT}G (need ≥80) · /workspace free ${AVAIL_WS}G (need ≥120)"
NEED_ROOT=80; [ -e /opt/conda-st ] && NEED_ROOT=10   # if already restored, no need to keep room for unpacking
[ "${AVAIL_ROOT:-0}" -ge "$NEED_ROOT" ] || die "not enough container disk (need ${NEED_ROOT}G)"
[ "${AVAIL_WS:-0}" -ge 120 ] || die "not enough network volume"

step "1. Pull the environment package (22 GB)"
mkdir -p "$PKG" "$CK" "$HF_HOME"
command -v hf >/dev/null 2>&1 || pip install -q huggingface_hub hf-transfer
HF="$(command -v hf || command -v huggingface-cli)"
if [ ! -f "$PKG/RESTORE.sh" ]; then
    "$HF" download "$ENV_REPO" --local-dir "$PKG" || die "environment package download failed"
fi
[ -f "$PKG/RESTORE.sh" ] || die "no RESTORE.sh in the environment package"

step "2. Restore the environment to /opt"
if [ -e /opt/conda-st ]; then
    echo "  /opt/conda-st already exists, skipping restore"
else
    bash "$PKG/RESTORE.sh" "$PKG" || die "RESTORE.sh failed"
    bash "$PKG/POSTRESTORE.sh" || die "POSTRESTORE.sh failed"
fi
[ -f "$RUN_EVAL" ] || die "still no $RUN_EVAL after restore"

step "2b. Fix BENCHMARKS paths (upstream's hard-coded paths don't match the HF dataset's actual layout)"
# The script wants things like robospatial_home_multiturn/test.parquet, but the dataset actually has data/<key>.parquet.
# Without this fix it hits Missing: ... at line 179 and then exit 1. The SFT baseline also used data/<key>.parquet; same definition.
if grep -q 'robospatial_home_multiturn/test.parquet' "$RUN_EVAL"; then
    cp -n "$RUN_EVAL" "$RUN_EVAL.orig-benchmarks"
    python3 - "$RUN_EVAL" <<'BM_PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
keys = ["robospatial", "reflocation", "refplacement", "refunseen", "boppose",
        "bopgrasp", "blinkdepth", "cvb2drelation", "cvb3ddepth"]
new = "declare -A BENCHMARKS=(\n" + "".join('    [%s]="data/%s.parquet"\n' % (k, k) for k in keys) + ")\n"
s2 = re.sub(r"declare -A BENCHMARKS=\(.*?\n\)\n", new, s, count=1, flags=re.S)
assert s2 != s, "BENCHMARKS block did not match"
open(p, "w").write(s2)
print("  BENCHMARKS paths changed to data/<key>.parquet")
BM_PY
    [ $? -eq 0 ] || die "failed to fix BENCHMARKS paths"
else
    echo "  paths already fixed, skipping"
fi

step "3. Pull weights (at the pinned revisions, including 292 MB of eval data)"
bash "$PKG/FETCH_WEIGHTS.sh" || die "FETCH_WEIGHTS.sh failed"

step "4. Pull P7's RL checkpoint: global_step_$P7_STEP"
if [ ! -f "$MODEL/config.json" ]; then
    "$HF" download "$P7_REPO" --include "global_step_$P7_STEP/*" --local-dir "$CK/p7-dl" || die "P7 ckpt download failed"
    mkdir -p "$MODEL"
    cp -a "$CK/p7-dl/global_step_$P7_STEP/." "$MODEL/"
fi
[ -f "$MODEL/config.json" ] || die "no config.json in $MODEL"
grep -q Qwen2_5_VL "$MODEL/config.json" || die "$MODEL/config.json is not the Qwen2.5-VL architecture"
du -sh "$MODEL" | sed 's/^/  /'

step "5. Match the KV pool size (without this step, switching GPUs means it's not the same measurement)"
CARD_GB=$(( MEMMIN / 1024 ))
GMU=$(python3 -c "print(round(min(0.85, $KV_POOL_GB / $CARD_GB), 3))")
CUR=$(grep -oE 'gpu_memory_utilization=[0-9.]+' "$RUN_EVAL" | head -1 | cut -d= -f2)
echo "  whole GPU ${CARD_GB} GB · target pool ${KV_POOL_GB} GB · gpu_memory_utilization ${CUR} -> ${GMU}"
if [ "$CUR" != "$GMU" ]; then
    cp -n "$RUN_EVAL" "$RUN_EVAL.orig"
    sed -i "s/gpu_memory_utilization=$CUR/gpu_memory_utilization=$GMU/" "$RUN_EVAL"
    grep -n 'gpu_memory_utilization' "$RUN_EVAL" | sed 's/^/  /'
    echo "  ⚠️ changed — when reporting results you must state this value; it is not numerically neutral"
fi

step "6. Clear leftover Ray (if not cleared, VERIFY item 3 fails falsely)"
export PATH=/opt/conda-st/envs/spacetools-rl/bin:$PATH
ray stop --force || true
rm -rf /root/tmp/ray

step "7. Environment acceptance check (only proceed when all four items pass)"
V_OUT="$(CUDA_VISIBLE_DEVICES=0 bash "$PKG/VERIFY.sh" 2>&1)"; V_RC=$?
echo "$V_OUT" | tail -20
echo "$V_OUT" | grep -qE "四项全过|all four items pass" || die "VERIFY.sh did not print \"all four items pass\" (exit code $V_RC). Do not run eval in this state, the scores will be wrong"

step "8. Run the nine benchmarks (about 2 h 10 min)"
cd /opt/spacetools/SpaceTools-RL || die "repo is not at /opt/spacetools/SpaceTools-RL"
export ROBOREFER_MODEL="$CK/RoboRefer-8B-SFT"
export DEPTH_CHECKPOINT="$CK/depth_pro.pt"
export CONDA_ROOT=/opt/conda-st
export PATH=/opt/conda-st/bin:$PATH
# Non-interactive bash doesn't inherit shell functions, and conda activate is a function -> rely on BASH_ENV to source it
export BASH_ENV=/opt/conda-st/etc/profile.d/conda.sh
export NUM_GPUS=4
export EVAL_GPUS=1
export OUTPUT_DIR="$EVAL_OUT"
bash "$RUN_EVAL" "$MODEL" \
    robospatial reflocation refplacement refunseen \
    blinkdepth cvb2drelation cvb3ddepth boppose bopgrasp
EVAL_RC=$?
echo "  run_eval.sh exit code $EVAL_RC"
[ "$EVAL_RC" -eq 0 ] || die "run_eval.sh failed (exit code $EVAL_RC) — the nine benchmarks did not finish; ignore the gate and comparison numbers below"

step "9. Health gate (count OOMs and tool errors first, then look at scores)"
PD=""
for C in "$(dirname "$0")/parse_dump.py" "$PKG/parse_dump.py" /opt/spacetools/spacetools-repro/tools/parse_dump.py; do
    [ -f "$C" ] && PD="$C" && break
done
if [ -z "$PD" ]; then
    echo "  ⚠️ parse_dump.py not found, skipping the gate — then you must count by hand, and before looking at scores:"
    echo "     grep -c OutOfMemoryError and grep -cE 'Error:|ERROR:toolshed' must both be 0"
else
    for B in robospatial reflocation refplacement refunseen blinkdepth cvb2drelation cvb3ddepth boppose bopgrasp; do
        D="$EVAL_OUT/$B"
        [ -d "$D" ] || { echo "  skipping $B (no output directory)"; continue; }
        python "$PD" --strict "$D" || echo "  ✗ $B failed the gate — read it as \"this benchmark has no result yet\", not \"there's a warning\""
    done
fi

echo
echo "==== Done · output in $EVAL_OUT ===="
echo "When reporting results, also state: gpu_memory_utilization=$GMU · whole GPU ${CARD_GB} GB · NUM_GPUS=4 EVAL_GPUS=1"
echo "Comparison numbers:"
echo "  RoboSpatial Overall  SFT starting point 61.00 ± 0.77  ·  official checkpoint (P4 reproduction) 65.43–66.00  ·  paper 70.00"
echo "  RefSpatial three     SFT starting point 52.58 simple mean  ·  official checkpoint 53.35  ·  paper 53.07"
echo "  boppose / bopgrasp scores jitter; P4 ran them 3 times and reported an interval"
