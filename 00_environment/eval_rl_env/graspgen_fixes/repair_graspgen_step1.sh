#!/bin/bash
# =============================================================================
# 修 spacetools-tool-graspgen —— 它是唯一没装成的工具环境
#
# 三个原因:
#  ① install_graspgen.sh 有两个交互式 read(GRASPGEN_DIR / MODELS_DIR),
#     非交互下 EOF + set -e 直接退出,日志停在 "Enter directory ..." 那一行。
#  ② PIP_CONSTRAINT 被导出成全局,把 torch 钉死在 2.9.1;而 GraspGen 的
#     pyproject 被 install_graspgen.sh 改成 torch>=2.3.0,<2.4 —— 必然冲突。
#     **约束只该管 spacetools-rl,工具环境本来就各有各的 torch。**
#  ③ 原来的成功判据只看环境是否存在,装了个空壳也算"✓ 建好了"。
# =============================================================================
set -uo pipefail
CONDA_DIR=/opt/conda-st
REPO_DIR=/opt/spacetools
export PATH="$CONDA_DIR/bin:$PATH"
eval "$(conda shell.bash hook)"
die() { echo "✗ $*"; exit 1; }

# ⚠️ 关键:工具环境不能继承 rl 的约束
unset PIP_CONSTRAINT

# ⚠️ 两个 read 的答案(不设就会 EOF 退出)
export GRASPGEN_DIR="$REPO_DIR"
export MODELS_DIR=/workspace/checkpoints
export TMPDIR=/root/tmp
export PIP_CACHE_DIR=/root/.cache/pip
export HF_HOME=/workspace/hf
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
export MAX_JOBS=6            # cgroup 上限 46.6 GiB,别再被 OOM 杀
mkdir -p "$MODELS_DIR" "$TMPDIR"

echo "=== 删掉空壳环境重建 ==="
conda env remove -n spacetools-tool-graspgen -y >/dev/null 2>&1
conda env list | grep -q "^spacetools-tool-graspgen " && die "旧环境没删掉"

set +u; conda activate spacetools-rl; set -u   # setup_tool_env.sh 从这里探测 python+ray
[ "${CONDA_DEFAULT_ENV:-}" = spacetools-rl ] || die "激活 spacetools-rl 失败"
echo "PIP_CONSTRAINT=${PIP_CONSTRAINT:-<已清空>}  GRASPGEN_DIR=$GRASPGEN_DIR  MODELS_DIR=$MODELS_DIR"

cd "$REPO_DIR/SpaceTools-Toolshed"
( set +u; source install_tools/setup_tool_env.sh spacetools-tool-graspgen graspgen )
echo "setup_tool_env.sh 退出码: $?"

# =============================================================================
echo; echo "=== 真实验收(不看退出码,看实际状态)==="
# =============================================================================
E="$CONDA_DIR/envs/spacetools-tool-graspgen"
fail=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "  ✓ $1"; else echo "  ✗ $1"; fail=$((fail+1)); fi; }

chk "环境存在"        "[ -d $E ]"
chk "ray 装了"        "$E/bin/python -c 'import ray'"
chk "toolshed 装了"   "$E/bin/python -c 'import toolshed'"
chk "torch 装了"      "$E/bin/python -c 'import torch'"
chk "pointnet2_ops"   "$E/bin/python -c 'import pointnet2_ops'"
chk "grasp_gen 包"    "$E/bin/python -c 'import grasp_gen'"
chk "GraspGen 权重"   "[ -d $MODELS_DIR/GraspGenModels ]"

echo "  --- 版本 ---"
$E/bin/python -c "import torch,ray;print('  torch',torch.__version__,'cuda',torch.version.cuda,'| ray',ray.__version__)" 2>&1 | tail -1

echo
[ "$fail" -eq 0 ] && echo "✓ graspgen 环境修好了" || echo "✗ 还有 $fail 项不过"
exit "$fail"
