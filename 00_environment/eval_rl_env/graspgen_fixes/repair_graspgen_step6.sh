#!/bin/bash
# =============================================================================
# 修 graspgen 第六轮 —— pointnet2_ops 硬编码的 GPU 架构列表
#
# 根因(第十三个上游洞,在 GraspGen 内嵌的 pointnet2_ops 里):
#   pointnet2_ops/pointnet2_utils.py:23
#       os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
#   它在 **import 时**直接覆盖环境变量(这个扩展是首次 import 时 JIT 编译的),
#   于是外面设的 8.0;8.6;8.9 完全无效,nvcc 拿到 compute_37:
#       nvcc fatal : Unsupported gpu architecture 'compute_37'
#   CUDA 12 已经不支持 sm_37。
#
# 处方:把赋值改成 setdefault —— 外部设了就用外部的,没设才用它的默认。
#      这是最小改动,且不改变"没人设时"的上游行为。
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
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"      # A100 / A6000 / L40S
mkdir -p "$MODELS_DIR" "$TMPDIR"

set +u; conda activate spacetools-tool-graspgen; set -u
[ "${CONDA_DEFAULT_ENV:-}" = spacetools-tool-graspgen ] || die "激活失败"
SP="$CONDA_PREFIX/lib/python3.11/site-packages"
CC_BIN="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
CXX_BIN="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDA_HOME="$CONDA_PREFIX" CC="$CC_BIN" CXX="$CXX_BIN"
export PATH="$CUDA_HOME/bin:$PATH"
export NVCC_PREPEND_FLAGS="-ccbin $CC_BIN"
echo "✓ $CONDA_DEFAULT_ENV · nvcc $(nvcc --version|grep -oP 'release \K[0-9.]+') · gcc $($CC_BIN --version|head -1|grep -oP '\) \K[0-9.]+')"

echo; echo "=== 1. 把硬编码的架构列表改成 setdefault ==="
PATCHED=0
for f in "$SP/pointnet2_ops/pointnet2_utils.py" \
         "$G/pointnet2_ops/pointnet2_ops/pointnet2_utils.py" \
         "$G/pointnet2_ops/build/lib.linux-x86_64-cpython-311/pointnet2_ops/pointnet2_utils.py"; do
    [ -f "$f" ] || continue
    grep -q 'os.environ\["TORCH_CUDA_ARCH_LIST"\] =' "$f" || { echo "  (已改过) $f"; continue; }
    sed -i 's|os\.environ\["TORCH_CUDA_ARCH_LIST"\] = |os.environ.setdefault("TORCH_CUDA_ARCH_LIST", |; s|"3\.7+PTX;5\.0;6\.0;6\.1;6\.2;7\.0;7\.5"$|"3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5")|' "$f"
    python -c "import ast,sys; ast.parse(open('$f').read())" || die "改坏了语法: $f"
    echo "  ✓ $f"; sed -n '23p' "$f" | sed 's/^/      /'
    PATCHED=$((PATCHED+1))
done
[ "$PATCHED" -gt 0 ] || echo "  (没有需要改的文件)"

echo; echo "=== 2. 清掉 JIT 缓存,重新触发编译 ==="
rm -rf ~/.cache/torch_extensions "$G/pointnet2_ops/build"
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
timeout 1200 python -c "
import os, pointnet2_ops
print('  生效的 arch list:', os.environ.get('TORCH_CUDA_ARCH_LIST'))
import pointnet2_ops._ext as E
print('  ✓ pointnet2_ops._ext 编译并导入成功:', E.__file__)
" 2>&1 | tail -25
python -c "import pointnet2_ops._ext" 2>/dev/null || die "pointnet2_ops._ext 仍然导入失败"

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
echo "  --- _ext 的 cubin(必须含 sm_80)---"
SO=$(find ~/.cache/torch_extensions "$E" -name "_ext*.so" 2>/dev/null|head -1)
if [ -n "$SO" ]; then echo "    $SO"; "$E/bin/cuobjdump" --list-elf "$SO" 2>/dev/null|head -4|sed 's/^/    /'
else echo "    (找不到 _ext .so)"; fi
echo
[ "$fail" -eq 0 ] && echo "✓ graspgen 修好了" || echo "✗ 还有 $fail 项不过"
exit "$fail"
