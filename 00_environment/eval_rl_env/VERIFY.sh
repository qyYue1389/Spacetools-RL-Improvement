#!/bin/bash
# 还原之后的真实验收。四项全过才算这套环境可用。
# 判据一律看实际状态,不看退出码 —— 这个项目栽过太多次假绿灯。
#
# 2026-09-12 修订(依据 03_sft_eval/sft_eval_results.md):
#   + 第 0 项前置检查。原来的三项**抓不到**两个真实故障:
#       - 五个环境 Python 版本不一致(3.11.0 vs 3.11.16)时 Ray 拒绝那些环境的
#         actor 入集群,三项全绿但 eval/RL 会静默缺席 8 个(RL 23 个)actor
#       - /workspace/logs 不存在时第 3 项的 grep 全读空却仍 exit 0
#   + 第 3 项不再相信 28_chain.sh 的退出码,改为在输出里找成功标记。
#     实测过它打印「✗ 链没通」的同时 exit 0。
#   + 第 2 项自动补 CUDA_HOME。04_smoke.sh 直连 env python、不激活 conda,
#     roborefer 的 deepspeed 会因此抛 MissingCUDAException。
set -uo pipefail
S=/root/pkgstage/scripts
[ -d "$S" ] || S=/root
[ -f "$S/03_verify.sh" ] || { echo "✗ 找不到验收脚本,先解开 scripts 包"; exit 1; }
CD="${CONDA_DIR:-/opt/conda-st}"
cd /

echo "==== 0. 前置检查(这些不过,后面三项的绿灯不可信)===="
RC0=0
p0(){ echo "  OK  $*"; }
b0(){ echo "  BAD $*"; RC0=1; }

# 0a 五环境 Python 一致性,以及不一致时 Ray 补丁是否到位
HEAD=""; MIS=""
for e in spacetools-rl spacetools-tool-vlm spacetools-tool-roborefer \
         spacetools-tool-bbox spacetools-tool-graspgen; do
    px="$CD/envs/$e/bin/python"
    [ -x "$px" ] || { b0 "$e 环境不存在"; continue; }
    v="$("$px" -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])')"
    [ "$e" = spacetools-rl ] && HEAD="$v"
    printf "      %-28s %s\n" "$e" "$v"
done
for e in spacetools-tool-vlm spacetools-tool-roborefer spacetools-tool-bbox spacetools-tool-graspgen; do
    px="$CD/envs/$e/bin/python"
    [ -x "$px" ] || continue
    v="$("$px" -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])')"
    [ "$v" = "$HEAD" ] && continue
    MIS="$MIS $e"
    if "$px" -c "
import inspect, sys, ray._private.node as n
sys.exit(0 if 'minor' in inspect.getsource(n.Node.check_version_info) else 1)" 2>/dev/null; then
        p0 "$e Python($v) != 头节点($HEAD),但 Ray minor 档补丁在位"
    else
        b0 "$e Python($v) != 头节点($HEAD) 且没有 Ray 补丁 —— 这个环境的 actor 入不了集群,先跑 POSTRESTORE.sh"
    fi
done
[ -n "$MIS" ] || p0 "五环境 Python 版本一致($HEAD)"

# 0b roborefer 的 CUDA_HOME(deepspeed 在 import 时就要读)
RR="$CD/envs/spacetools-tool-roborefer"
if [ -n "${CUDA_HOME:-}" ]; then
    p0 "CUDA_HOME 已由调用方设置($CUDA_HOME)"
elif [ -x "$RR/bin/nvcc" ]; then
    export CUDA_HOME="$RR"
    p0 "CUDA_HOME 未设,自动指向 $RR"
else
    b0 "roborefer 环境里没有 nvcc,且 CUDA_HOME 未设 —— 第 2 项会在 roborefer 上失败"
fi

# 0c 运行期目录
for d in /workspace/logs /workspace/smoke /workspace/checkpoints /workspace/hf; do
    if [ -d "$d" ]; then p0 "$d 存在"; else b0 "$d 缺失(POSTRESTORE.sh 建前两个,权重目录见 MANIFEST)"; fi
done

# 0d 不能有活着的 Ray 集群,也不能有残留的地址文件
if pgrep -x gcs_server >/dev/null 2>&1 || pgrep -x raylet >/dev/null 2>&1; then
    b0 "有 Ray 进程在跑 —— 第 3 项的 ray.init(num_cpus=...) 会被拒。先 ray stop --force"
else
    STALE=""
    for f in /root/tmp/ray/ray_current_cluster /tmp/ray/ray_current_cluster; do
        [ -f "$f" ] && STALE="$STALE $f"
    done
    if [ -n "$STALE" ]; then
        b0 "残留的集群地址文件:$STALE —— 第 3 项会以为集群还在。删掉它们再跑"
    else
        p0 "没有活着的 Ray 集群,也没有残留地址文件"
    fi
fi
[ "$RC0" -eq 0 ] || { echo; echo "✗ 前置检查不过 —— 停在这里,不要看后面三项的结果"; exit 1; }

echo
echo "==== 1. 五环境 import 闸门 + .so 架构实扫 ===="
bash "$S/03_verify.sh"; RC1=$?
echo
echo "==== 2. 七工具冒烟(真加载权重、真出结果)===="
bash "$S/04_smoke.sh"; RC2=$?
echo
echo "==== 3. Ray 跨环境工具链(变量跨 conda 环境传递)===="
CHAIN_OUT="$(bash "$S/28_chain.sh" 2>&1)"; RC3RAW=$?
echo "$CHAIN_OUT"
# 不相信退出码:必须在输出里看到成功标记,且没有失败标记
if echo "$CHAIN_OUT" | grep -q "跨环境工具链通了" && ! echo "$CHAIN_OUT" | grep -q "链没通"; then
    RC3=0
else
    RC3=1
    echo "  (28_chain.sh 退出码是 $RC3RAW,但输出里没有成功标记 —— 按失败计)"
fi

echo
echo "==== 汇总 ===="
echo "  前置检查   退出码 $RC0  (0=Python 一致或补丁在位、目录齐、无残留 Ray)"
echo "  环境验收   退出码 $RC1  (0=五环境 import 全通 且 自编扩展都含 sm_8x)"
echo "  七工具冒烟 退出码 $RC2  (0=七个都真出了结果,且结果里没有 Error:)"
echo "  跨环境链   退出码 $RC3  (0=输出里确认 ndarray 跨 conda 环境双向传递正确)"
if [ "$RC0" -eq 0 ] && [ "$RC1" -eq 0 ] && [ "$RC2" -eq 0 ] && [ "$RC3" -eq 0 ]; then
    echo "✓ 四项全过,环境可用"
else
    echo "✗ 有项目不过 —— 不要在这个状态下跑 eval,分数会是错的"
    exit 1
fi
