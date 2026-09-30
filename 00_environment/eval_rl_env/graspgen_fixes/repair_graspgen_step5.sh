#!/bin/bash
# =============================================================================
# 修 graspgen 第五轮 —— 编译器版本
#
# 前四轮已解决:pickle5(3.11 编不过)、GRASPGEN_DIR/MODELS_DIR 的交互式 read、
# PIP_CONSTRAINT 不该传染到工具环境、cuda-version 才是真正的版本闸门。
# 现在:nvcc 12.1 ✓,torch 2.3.1+cu121 ✓,grasp_gen ✓,只差 pointnet2_ops。
#
# 本轮根因:
#   crt/host_config.h:132: #error -- unsupported GNU version!
#                          gcc versions later than 12 are not supported!
#   nvcc 12.1 最高支持 gcc 12,而 Ubuntu 24.04 的系统 gcc 是 13.3。
#   install_graspgen.sh 假设了更老的发行版。
#
# 处方:给 nvcc 配一个 gcc 12(conda-forge),而不是用 -allow-unsupported-compiler
#      强行绕过 —— 那个 flag 自己的警告里就写着"可能导致运行时行为错误"。
# =============================================================================
set -uo pipefail
CONDA_DIR=/opt/conda-st
G=/opt/spacetools/GraspGen
export PATH="$CONDA_DIR/bin:$PATH"
eval "$(conda shell.bash hook)"
die() { echo "✗ $*"; exit 1; }

unset PIP_CONSTRAINT
export TMPDIR=/root/tmp PIP_CACHE_DIR=/root/.cache/pip
export MODELS_DIR=/workspace/checkpoints
export MAX_JOBS=4
mkdir -p "$MODELS_DIR" "$TMPDIR"

set +u; conda activate spacetools-tool-graspgen; set -u
[ "${CONDA_DEFAULT_ENV:-}" = spacetools-tool-graspgen ] || die "激活失败"
echo "✓ $CONDA_DEFAULT_ENV"
echo "  nvcc  $(nvcc --version 2>/dev/null|grep -oP 'release \K[0-9.]+')"
python -c "import torch;print('  torch',torch.__version__,'cuda',torch.version.cuda)"
echo "  系统 gcc $(gcc --version|head -1|grep -oP '\) \K[0-9.]+')  ← nvcc 12.1 只支持 ≤12"

echo; echo "=== 1. 装 gcc 12 工具链 ==="
conda install -y -c conda-forge "gcc_linux-64=12" "gxx_linux-64=12" 2>&1 | tail -4
CC_BIN="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
CXX_BIN="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
[ -x "$CC_BIN" ] && [ -x "$CXX_BIN" ] || die "conda gcc 没装上"
GV="$("$CC_BIN" --version | head -1 | grep -oP '\) \K[0-9]+')"
echo "  conda gcc: $("$CC_BIN" --version|head -1)"
[ "$GV" = "12" ] || die "conda gcc 主版本是 $GV,不是 12"

echo; echo "=== 2. 编译 pointnet2_ops(nvcc 12.1 + gcc 12)==="
cd "$G/pointnet2_ops"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CC="$CC_BIN" CXX="$CXX_BIN"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$CUDA_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/stubs:${LIBRARY_PATH:-}"
export NVCC_PREPEND_FLAGS="-ccbin $CC_BIN"
rm -rf build
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" pip install --no-build-isolation . > /workspace/pn2_build.log 2>&1
RC=$?
echo "  pip 退出码 $RC(完整日志 /workspace/pn2_build.log)"
if [ "$RC" -ne 0 ]; then
    echo "  --- 报错摘录 ---"
    grep -nE "error:|fatal error|FAILED|unsupported" /workspace/pn2_build.log | head -8 | sed 's/^/    /'
    die "pointnet2_ops 编译失败"
fi
python -c "import pointnet2_ops._ext; print('  ✓ pointnet2_ops._ext 可导入')" || die "编译成功但导入失败"
cd "$G"

echo; echo "=== 3. numpy 回到 2.x ==="
pip install "numpy>=2.0" --force-reinstall --no-deps 2>&1 | tail -2

echo; echo "=== 4. GraspGen 权重 ==="
cd "$MODELS_DIR"
[ -d GraspGenModels ] || git clone --depth 1 https://huggingface.co/adithyamurali/GraspGenModels
du -sh "$MODELS_DIR/GraspGenModels" 2>/dev/null

# =============================================================================
echo; echo "=== 验收 ==="
# =============================================================================
E="$CONDA_DIR/envs/spacetools-tool-graspgen"
fail=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "  ✓ $1"; else echo "  ✗ $1"; fail=$((fail+1)); fi; }
chk "ray"                  "$E/bin/python -c 'import ray'"
chk "toolshed"             "$E/bin/python -c 'import toolshed'"
chk "torch"                "$E/bin/python -c 'import torch'"
chk "pointnet2_ops"        "$E/bin/python -c 'import pointnet2_ops'"
chk "pointnet2 CUDA 算子"  "$E/bin/python -c 'import pointnet2_ops._ext'"
chk "grasp_gen"            "$E/bin/python -c 'import grasp_gen'"
chk "dataset(pickle 回退)"  "$E/bin/python -c 'from grasp_gen.dataset.dataset import collate'"
chk "GraspGenModels 权重"   "[ -d $MODELS_DIR/GraspGenModels ]"
echo "  --- 版本 ---"
$E/bin/python -c "import torch,ray,numpy;print('  torch',torch.__version__,'cuda',torch.version.cuda,'| ray',ray.__version__,'| numpy',numpy.__version__)" 2>&1|tail -1
echo "  --- pointnet2_ops 的 cubin(必须含 sm_80)---"
SO=$(find "$E" -name "_ext*.so" -path "*pointnet2*" 2>/dev/null|head -1)
if [ -n "$SO" ]; then "$E/bin/cuobjdump" --list-elf "$SO" 2>/dev/null | head -4 | sed 's/^/    /'
else echo "    (找不到 _ext .so)"; fi
echo
[ "$fail" -eq 0 ] && echo "✓ graspgen 修好了" || echo "✗ 还有 $fail 项不过"
exit "$fail"
