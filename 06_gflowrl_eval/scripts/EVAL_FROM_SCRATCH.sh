#!/usr/bin/env bash
# 在一台裸 GPU 机器上,从零把 P7 的 RL checkpoint 评完九个 benchmark。
#
#   export HF_TOKEN=hf_xxx          # 私有 repo 要
#   P7_STEP=86 bash EVAL_FROM_SCRATCH.sh
#
# 每一步幂等,中断后重跑会跳过已完成的。每一步的判据是"看到成功标记",不是"退出码 0"
# —— 这个项目的失败模式高度集中在"看起来成功了"。
set -uo pipefail

P7_REPO="${P7_REPO:-qzpm55555/spacetools-p7-gflowrl-cprime-8xa40}"
P7_STEP="${P7_STEP:-85}"
ENV_REPO="${ENV_REPO:-qzpm55555/spacetools-eval-env}"
PKG="${PKG:-/workspace/envpkg}"
CK="${CK:-/workspace/checkpoints}"
MODEL="$CK/p7-step$P7_STEP"
EVAL_OUT="${EVAL_OUT:-/workspace/exp/p7_eval_step$P7_STEP}"
RUN_EVAL=/opt/spacetools/SpaceTools-RL/examples/toolshed/run_eval.sh
# KV 池的目标大小(GB)。24 = SFT 基线(61.00 ± 0.77)测出来时的池子大小:
# 48 GB 卡 × gpu_memory_utilization 0.5。这个旋钮是**整卡比例**不是绝对值,
# 换卡不改它,池子就变了 —— 池子大小改变 batch 组成 -> 浮点归约顺序 -> 接近平局的
# 样本翻转。要和 SFT 起点比,就得让池子一样大。
KV_POOL_GB="${KV_POOL_GB:-24}"
export HF_HOME="${HF_HOME:-/workspace/hf}"
export HF_HUB_ENABLE_HF_TRANSFER=1

die(){ echo; echo "✗ $*"; exit 1; }
step(){ echo; echo "==== $* ===="; }

step "0. 机器验收"
command -v nvidia-smi >/dev/null 2>&1 || die "没有 nvidia-smi —— 驱动没装。第一条检查是 nvidia-smi -L,不是 nvidia-smi"
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
NGPU=$(nvidia-smi -L | wc -l)
MEMMIN=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "$NGPU" -ge 4 ] || die "只有 $NGPU 张卡。工具逻辑预留 3.0 + 策略 1.0 = 4.0,至少 4 张"
case "$CC" in
    8.0|8.6|8.9) ;;
    *) die "compute capability $CC 不在覆盖范围。环境包的自编扩展只有 sm_80/86/89 的 cubin、没带 +PTX,H100(9.0)跑不了" ;;
esac
[ "${DRV:-0}" -ge 550 ] || die "驱动 $DRV < 550"
# 显存下限:P4 在 A100-40GB 上用这套分数跑完九个 benchmark、零 OOM,所以 40 GB 够。
# 实测峰值:Molmo 34.6 GB(上游 vlm.py:116 吃掉 dtype 配置,fp32 加载)、
# DepthPro×2 那张 25.7 GB、策略卡 gmu×整卡。
[ "$MEMMIN" -ge 38000 ] || die "最小单卡显存 ${MEMMIN} MiB < 40 GB。Molmo 一个 actor 就要 34.6 GB,装不下。
    要在更小的卡上跑必须重算 TOOL_CONFIGS 的分数、可能还要上 8-bit —— 那会让分数与论文不可比,不要硬跑:
    OOM 不会让脚本崩,只会静默掉分(实测 88.3% -> 83.1%,一声不响)。"
[ -n "${HF_TOKEN:-}" ] || die "没有 HF_TOKEN。环境包和 ckpt 都是私有 repo"
echo "  GPU ${NGPU} 张 · 最小显存 ${MEMMIN} MiB · cc $CC · 驱动 $DRV · HF_TOKEN len ${#HF_TOKEN}"
AVAIL_ROOT=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
AVAIL_WS=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
echo "  容器盘可用 ${AVAIL_ROOT}G(要 ≥80)· /workspace 可用 ${AVAIL_WS}G(要 ≥120)"
NEED_ROOT=80; [ -e /opt/conda-st ] && NEED_ROOT=10   # 已还原就不用再留解包空间
[ "${AVAIL_ROOT:-0}" -ge "$NEED_ROOT" ] || die "容器盘不够(需要 ${NEED_ROOT}G)"
[ "${AVAIL_WS:-0}" -ge 120 ] || die "网络卷不够"

step "1. 拉环境包(22 GB)"
mkdir -p "$PKG" "$CK" "$HF_HOME"
command -v hf >/dev/null 2>&1 || pip install -q huggingface_hub hf-transfer
HF="$(command -v hf || command -v huggingface-cli)"
if [ ! -f "$PKG/RESTORE.sh" ]; then
    "$HF" download "$ENV_REPO" --local-dir "$PKG" || die "环境包下载失败"
fi
[ -f "$PKG/RESTORE.sh" ] || die "环境包里没有 RESTORE.sh"

step "2. 还原环境到 /opt"
if [ -e /opt/conda-st ]; then
    echo "  /opt/conda-st 已存在,跳过还原"
else
    bash "$PKG/RESTORE.sh" "$PKG" || die "RESTORE.sh 失败"
    bash "$PKG/POSTRESTORE.sh" || die "POSTRESTORE.sh 失败"
fi
[ -f "$RUN_EVAL" ] || die "还原完了还是没有 $RUN_EVAL"

step "2b. 修 BENCHMARKS 路径(上游写死的路径与 HF 数据集实际布局不符)"
# 脚本要的是 robospatial_home_multiturn/test.parquet 之类,数据集里其实是 data/<key>.parquet。
# 不改就在第 179 行 Missing: ... 然后 exit 1。SFT 基线用的也是 data/<key>.parquet,口径一致。
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
assert s2 != s, "BENCHMARKS 块没匹配上"
open(p, "w").write(s2)
print("  BENCHMARKS 路径已改为 data/<key>.parquet")
BM_PY
    [ $? -eq 0 ] || die "改 BENCHMARKS 路径失败"
else
    echo "  已是修好的路径,跳过"
fi

step "3. 拉权重(按钉死的 revision,含 292 MB eval 数据)"
bash "$PKG/FETCH_WEIGHTS.sh" || die "FETCH_WEIGHTS.sh 失败"

step "4. 拉 P7 的 RL checkpoint: global_step_$P7_STEP"
if [ ! -f "$MODEL/config.json" ]; then
    "$HF" download "$P7_REPO" --include "global_step_$P7_STEP/*" --local-dir "$CK/p7-dl" || die "P7 ckpt 下载失败"
    mkdir -p "$MODEL"
    cp -a "$CK/p7-dl/global_step_$P7_STEP/." "$MODEL/"
fi
[ -f "$MODEL/config.json" ] || die "$MODEL 里没有 config.json"
grep -q Qwen2_5_VL "$MODEL/config.json" || die "$MODEL/config.json 不是 Qwen2.5-VL 架构"
du -sh "$MODEL" | sed 's/^/  /'

step "5. 对齐 KV 池大小(不做这一步,换卡就不是同一个测量)"
CARD_GB=$(( MEMMIN / 1024 ))
GMU=$(python3 -c "print(round(min(0.85, $KV_POOL_GB / $CARD_GB), 3))")
CUR=$(grep -oE 'gpu_memory_utilization=[0-9.]+' "$RUN_EVAL" | head -1 | cut -d= -f2)
echo "  整卡 ${CARD_GB} GB · 目标池 ${KV_POOL_GB} GB · gpu_memory_utilization ${CUR} -> ${GMU}"
if [ "$CUR" != "$GMU" ]; then
    cp -n "$RUN_EVAL" "$RUN_EVAL.orig"
    sed -i "s/gpu_memory_utilization=$CUR/gpu_memory_utilization=$GMU/" "$RUN_EVAL"
    grep -n 'gpu_memory_utilization' "$RUN_EVAL" | sed 's/^/  /'
    echo "  ⚠️ 改过了 —— 报结果时必须写上这个取值,它不是数值中性的"
fi

step "6. 清掉残留的 Ray(不清会让 VERIFY 第 3 项假失败)"
export PATH=/opt/conda-st/envs/spacetools-rl/bin:$PATH
ray stop --force || true
rm -rf /root/tmp/ray

step "7. 环境验收(四项全过才往下走)"
V_OUT="$(CUDA_VISIBLE_DEVICES=0 bash "$PKG/VERIFY.sh" 2>&1)"; V_RC=$?
echo "$V_OUT" | tail -20
echo "$V_OUT" | grep -q "四项全过" || die "VERIFY.sh 没有打出「四项全过」(退出码 $V_RC)。不要在这个状态下跑 eval,分数会是错的"

step "8. 跑九个 benchmark(约 2 小时 10 分)"
cd /opt/spacetools/SpaceTools-RL || die "仓库不在 /opt/spacetools/SpaceTools-RL"
export ROBOREFER_MODEL="$CK/RoboRefer-8B-SFT"
export DEPTH_CHECKPOINT="$CK/depth_pro.pt"
export CONDA_ROOT=/opt/conda-st
export PATH=/opt/conda-st/bin:$PATH
# 非交互 bash 继承不到 shell 函数,conda activate 是函数 -> 靠 BASH_ENV 自己 source
export BASH_ENV=/opt/conda-st/etc/profile.d/conda.sh
export NUM_GPUS=4
export EVAL_GPUS=1
export OUTPUT_DIR="$EVAL_OUT"
bash "$RUN_EVAL" "$MODEL" \
    robospatial reflocation refplacement refunseen \
    blinkdepth cvb2drelation cvb3ddepth boppose bopgrasp
EVAL_RC=$?
echo "  run_eval.sh 退出码 $EVAL_RC"
[ "$EVAL_RC" -eq 0 ] || die "run_eval.sh 失败(退出码 $EVAL_RC)——九个 benchmark 没有跑完,下面的门禁和对照口径都不要看"

step "9. 健康门禁(先数 OOM 和工具错误,再看分数)"
PD=""
for C in "$(dirname "$0")/parse_dump.py" "$PKG/parse_dump.py" /opt/spacetools/spacetools-repro/tools/parse_dump.py; do
    [ -f "$C" ] && PD="$C" && break
done
if [ -z "$PD" ]; then
    echo "  ⚠️ 找不到 parse_dump.py,跳过门禁 —— 那就必须手工数,而且要在看分数之前:"
    echo "     grep -c OutOfMemoryError 和 grep -cE 'Error:|ERROR:toolshed' 都必须是 0"
else
    for B in robospatial reflocation refplacement refunseen blinkdepth cvb2drelation cvb3ddepth boppose bopgrasp; do
        D="$EVAL_OUT/$B"
        [ -d "$D" ] || { echo "  跳过 $B(没有输出目录)"; continue; }
        python "$PD" --strict "$D" || echo "  ✗ $B 没过门禁 —— 读成「这个 benchmark 还没有结果」,不是「有个警告」"
    done
fi

echo
echo "==== 完成 · 输出在 $EVAL_OUT ===="
echo "报结果时要一起写上:gpu_memory_utilization=$GMU · 整卡 ${CARD_GB} GB · NUM_GPUS=4 EVAL_GPUS=1"
echo "对照口径:"
echo "  RoboSpatial Overall  SFT 起点 61.00 ± 0.77  ·  官方 ckpt(P4 复现)65.43–66.00  ·  论文 70.00"
echo "  RefSpatial 三项      SFT 起点 52.58 简单平均  ·  官方 ckpt 53.35  ·  论文 53.07"
echo "  boppose / bopgrasp 分数有抖动,P4 是跑 3 次报区间的"
