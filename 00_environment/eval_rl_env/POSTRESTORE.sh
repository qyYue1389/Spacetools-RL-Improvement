#!/bin/bash
# 还原之后、验收之前跑这个。
#   bash RESTORE.sh  →  bash POSTRESTORE.sh  →  bash VERIFY.sh
#
# 它补三件打包时没有、但每台新机器都需要的东西。幂等,可重复跑。
# 判据一律看实际状态:每一处补完立刻回读断言,不生效就非零退出。
#
# 为什么这些不在 tar 里:它们是 2026-09-11 那次 eval 才发现的,而重打 22 GB
# 只为三行改动不划算。将来重建环境包时若把五个环境的 Python 统一,第 ① 项
# 会自动跳过,那时这个脚本可以删掉。
set -uo pipefail
CD="${CONDA_DIR:-/opt/conda-st}"
ENVS="spacetools-rl spacetools-tool-vlm spacetools-tool-roborefer spacetools-tool-bbox spacetools-tool-graspgen"
BAD=0
step(){ echo; echo "==== $* ===="; }
ok(){   echo "  OK  $*"; }
bad(){  echo "  BAD $*"; BAD=$((BAD+1)); }

step "0. 五个环境的 Python 版本"
HEAD=""      # 集群头节点用 spacetools-rl,以它为基准
declare -A PYV
for e in $ENVS; do
    p="$CD/envs/$e/bin/python"
    [ -x "$p" ] || { bad "$e 不存在"; continue; }
    v="$("$p" -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])')"
    PYV[$e]="$v"
    printf "  %-28s %s\n" "$e" "$v"
    [ "$e" = spacetools-rl ] && HEAD="$v"
done
[ -n "$HEAD" ] || { echo "✗ 读不到 spacetools-rl 的 Python 版本"; exit 1; }

step "1. Ray 的 Python 补丁版本检查(仅对与头节点不一致的环境)"
# Ray 的 check_version_info 默认比完整版本串(patch 档),3.11.0 != 3.11.16 会
# 直接 RuntimeError,那个环境里的 actor 全部入不了集群 —— 而且 VERIFY 的三项
# 都抓不到,只在 eval / RL 规模下暴露(实测 8 个 actor 静默缺席)。
# Ray 自己支持 minor 档(utils.py: python_version_match_level="minor"),此时
# 只 logger.warning。node.py 的调用点没传这个参数,这里补上。
# 安全性:两侧 bytecode magic 与 pickle 协议相同,code object / pickle 互通,
# 这正是 minor 档的设计场景。该改动只放宽一个版本检查,不改变任何数值。
NEED=""
for e in $ENVS; do
    [ -n "${PYV[$e]:-}" ] || continue
    [ "${PYV[$e]}" = "$HEAD" ] || NEED="$NEED $e"
done
if [ -z "$NEED" ]; then
    ok "五个环境 Python 版本一致($HEAD),不需要这个补丁"
else
    echo "  与头节点($HEAD)不一致:$NEED"
    for e in $NEED; do
        f="$CD/envs/$e/lib/python3.11/site-packages/ray/_private/node.py"
        [ -f "$f" ] || { bad "$e 找不到 ray/_private/node.py"; continue; }
        "$CD/envs/spacetools-rl/bin/python" - "$f" <<'PY'
import os, shutil, sys
f = sys.argv[1]
OLD = '''        ray._private.utils.check_version_info(
            cluster_metadata, f"node {node_ip_address}"
        )'''
NEW = '''        ray._private.utils.check_version_info(
            cluster_metadata, f"node {node_ip_address}",
            python_version_match_level="minor"
        )'''
src = open(f).read()
if 'python_version_match_level="minor"' in src:
    print("    已打过,跳过"); sys.exit(0)
if OLD not in src:
    print("    锚点找不到 —— 不改动。ray 版本可能不同,需人工确认"); sys.exit(3)
if not os.path.exists(f + ".orig"):
    shutil.copy2(f, f + ".orig")
open(f, "w").write(src.replace(OLD, NEW, 1))
pc = os.path.join(os.path.dirname(f), "__pycache__", "node.cpython-311.pyc")
if os.path.exists(pc): os.remove(pc)
print("    已打补丁(原文件备份为 node.py.orig)")
PY
        rc=$?
        [ "$rc" = 3 ] && { bad "$e 锚点不匹配"; continue; }
        # 判据:在那个环境里 import ray,把源码读回来确认
        if "$CD/envs/$e/bin/python" -c "
import inspect, sys, ray._private.node as n
sys.exit(0 if 'minor' in inspect.getsource(n.Node.check_version_info) else 1)" 2>/dev/null; then
            ok "$e minor 档生效(回读确认)"
        else
            bad "$e 补丁没生效"
        fi
    done
fi

step "2. roborefer 环境的 CUDA_HOME"
# llava 的推理路径模块级硬依赖 deepspeed(见环境 README 洞 #17),deepspeed 在
# import 时要读 CUDA_HOME。打包机上有系统 /usr/local/cuda-12.8,命中 torch 的
# 第三条 fallback;只装驱动的机器上没有,于是 04_smoke.sh(直连 env python、
# 不激活 conda)会抛 MissingCUDAException。
# 真实 eval 走 Ray 的 runtime_env={"conda":...},会激活 conda,本来就能解析;
# 这个 activate.d 是给「激活」路径加的保险,顺带让冒烟脚本也能过。
RR="$CD/envs/spacetools-tool-roborefer"
if [ -x "$RR/bin/nvcc" ]; then
    A="$RR/etc/conda/activate.d"
    mkdir -p "$A"
    printf 'export CUDA_HOME="$CONDA_PREFIX"\n' > "$A/zz_cuda_home.sh"
    if grep -q 'CUDA_HOME' "$A/zz_cuda_home.sh"; then
        ok "activate.d/zz_cuda_home.sh 就位($("$RR/bin/nvcc" --version | tail -2 | head -1 | tr -s ' '))"
    else
        bad "activate.d 写入失败"
    fi
else
    bad "$RR/bin/nvcc 不存在 —— 这个环境本该自带 nvcc,先查环境还原是否完整"
fi

step "3. 运行期需要的目录"
# 28_chain.sh 把原始输出写 /workspace/logs/chain_raw.log,04_smoke.sh 写
# /workspace/smoke。目录不存在时 28_chain.sh 的 grep 全部读空,却仍然 exit 0
# —— 一个假绿灯。MANIFEST 只提了 checkpoints 和 hf,没提这两个。
for d in /workspace/logs /workspace/smoke; do
    mkdir -p "$d" 2>/dev/null || sudo -n mkdir -p "$d"
    if [ -w "$d" ]; then ok "$d 可写"; else bad "$d 不可写"; fi
done

step "汇总"
if [ "$BAD" -eq 0 ]; then
    echo "✓ 全部就位。接着跑 VERIFY.sh"
else
    echo "✗ $BAD 项不过 —— 不要接着跑 eval"
    exit 1
fi
