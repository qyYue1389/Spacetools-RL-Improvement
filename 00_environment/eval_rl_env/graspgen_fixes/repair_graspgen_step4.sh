#!/bin/bash
# =============================================================================
# 修 graspgen 第四轮 —— 只剩 pointnet2_ops 的 CUDA 工具链
#
# 已确认的事实:
#  · torch 2.3.1+cu121 已装好(GraspGen 要的孤岛版本)
#  · grasp_gen 本体已装好
#  · 环境里的 nvcc 是 13.3,因为 `conda install cuda-toolkit=12.1` 只钉住了
#    那个 1 KB 的空壳元包;真正控制版本的是 `cuda-version`,它是 13.3。
#  · torch/utils/cpp_extension.py 的 _check_cuda_version 只在 **major 不同**
#    时报错,minor 不同仅 warning:
#        if cuda_ver.major != torch_cuda_version.major: raise RuntimeError(...)
#        warnings.warn(...)
#    所以判据是 major==12,不必死磕 12.1。
#
# 策略:先按正规方式钉 cuda-version=12.1;不成就退回用 spacetools-rl 里
#       已有的 nvcc 12.8(major 12,torch 允许)。
# =============================================================================
set -uo pipefail
CONDA_DIR=/opt/conda-st
G=/opt/spacetools/GraspGen
export PATH="$CONDA_DIR/bin:$PATH"
eval "$(conda shell.bash hook)"
die() { echo "✗ $*"; exit 1; }
nvcc_major() { "$1/bin/nvcc" --version 2>/dev/null | grep -oP 'release \K[0-9]+' || echo 0; }
nvcc_ver()   { "$1/bin/nvcc" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo none; }

unset PIP_CONSTRAINT
export TMPDIR=/root/tmp PIP_CACHE_DIR=/root/.cache/pip
export MODELS_DIR=/workspace/checkpoints
export MAX_JOBS=6
mkdir -p "$MODELS_DIR" "$TMPDIR"

set +u; conda activate spacetools-tool-graspgen; set -u
[ "${CONDA_DEFAULT_ENV:-}" = spacetools-tool-graspgen ] || die "激活失败"
GENV="$CONDA_PREFIX"
RLENV="$CONDA_DIR/envs/spacetools-rl"
echo "✓ $CONDA_DEFAULT_ENV · $(python -V 2>&1)"
python -c "import torch;print('  torch',torch.__version__,'cuda',torch.version.cuda)"

echo; echo "=== 1. 尝试把 cuda-version 钉到 12.1 ==="
echo "  改之前 nvcc: $(nvcc_ver "$GENV")  ·  cuda-version: $(conda list 2>/dev/null|awk '/^cuda-version /{print $2}')"
conda install -y -c nvidia "cuda-version=12.1" "cuda-nvcc=12.1" 2>&1 | tail -4
echo "  改之后 nvcc: $(nvcc_ver "$GENV")"

# =============================================================================
echo; echo "=== 2. 选一个 major==12 的 nvcc ==="
# =============================================================================
if [ "$(nvcc_major "$GENV")" = "12" ]; then
    CUDA_ROOT="$GENV";  echo "  用 graspgen 环境自己的 nvcc $(nvcc_ver "$GENV")"
elif [ "$(nvcc_major "$RLENV")" = "12" ]; then
    CUDA_ROOT="$RLENV"
    echo "  ⚠️ graspgen 环境的 nvcc 是 $(nvcc_ver "$GENV")(major 13,torch 会硬报错)"
    echo "     退回用 spacetools-rl 的 nvcc $(nvcc_ver "$RLENV") —— major 12,torch 只 warning"
    echo "     偏离记录:pointnet2_ops 用 nvcc $(nvcc_ver "$RLENV") 编,torch 是 cu121"
else
    die "两个环境都没有 major==12 的 nvcc"
fi

echo; echo "=== 3. 编译 pointnet2_ops ==="
cd "$G/pointnet2_ops"
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$CUDA_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$GENV/lib:$GENV/lib/stubs:$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
echo "  CUDA_HOME=$CUDA_HOME  ·  nvcc=$(nvcc --version|grep -oP 'release \K[0-9.]+')"
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" pip install --no-build-isolation . 2>&1 | tail -12
python -c "import pointnet2_ops._ext" 2>/dev/null || die "pointnet2_ops._ext 装不上/导入失败"
cd "$G"

echo; echo "=== 4. numpy 回到 2.x ==="
pip install "numpy>=2.0" --force-reinstall --no-deps 2>&1 | tail -2

echo; echo "=== 5. GraspGen 权重 ==="
cd "$MODELS_DIR"
[ -d GraspGenModels ] || git clone --depth 1 https://huggingface.co/adithyamurali/GraspGenModels
du -sh "$MODELS_DIR/GraspGenModels" 2>/dev/null

# =============================================================================
echo; echo "=== 验收(看实际状态)==="
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
if [ -n "$SO" ]; then "$CUDA_ROOT/bin/cuobjdump" --list-elf "$SO" 2>/dev/null | head -4 | sed 's/^/    /'
else echo "    (找不到 _ext .so)"; fi
echo
[ "$fail" -eq 0 ] && echo "✓ graspgen 修好了" || echo "✗ 还有 $fail 项不过"
exit "$fail"
