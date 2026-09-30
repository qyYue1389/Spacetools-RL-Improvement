#!/bin/bash
# =============================================================================
# 修 graspgen 第三轮 —— 只剩 CUDA 工具链版本这一个问题
#
# 上游 install_graspgen.sh 写的是:
#     conda install -c nvidia cuda-toolkit=12.1 -y
# 但 `cuda-toolkit` 是个 1 KB 的元包,它对组件的依赖是宽松的。实测结果:
#     cuda-toolkit  12.1.1     ← 元包版本号对
#     cuda-version  13.3       ← 组件全是 13.3
#     nvcc          13.3.73
# 于是编 pointnet2_ops 时:
#     RuntimeError: The detected CUDA version (13.3) mismatches PyTorch (12.1)
#
# 正确钉法:用带版本的 label channel,它把整套组件一起钉死。
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
export MAX_JOBS=6
mkdir -p "$MODELS_DIR" "$TMPDIR"

set +u; conda activate spacetools-tool-graspgen; set -u
[ "${CONDA_DEFAULT_ENV:-}" = spacetools-tool-graspgen ] || die "激活失败"
echo "✓ $CONDA_DEFAULT_ENV · $(python -V 2>&1)"
python -c "import torch;print('  torch',torch.__version__,'cuda',torch.version.cuda)"

echo; echo "=== 1. 把 CUDA 工具链降到 12.1(label channel 钉住整套)==="
echo "  当前 nvcc: $(nvcc --version 2>/dev/null|grep -oP 'release \K[0-9.]+' || echo 无)"
conda install -y -c "nvidia/label/cuda-12.1.1" --override-channels \
      cuda-toolkit 2>&1 | tail -5
NV="$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo none)"
echo "  安装后 nvcc: $NV"
[ "$NV" = "12.1" ] || die "nvcc 仍不是 12.1(是 $NV)—— 换 label channel 也没钉住,需要手动排查"
conda list 2>/dev/null | grep -E "^cuda-version" | sed 's/^/  /'

echo; echo "=== 2. 编译 pointnet2_ops ==="
cd "$G/pointnet2_ops"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include:${CPATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/stubs:${LIBRARY_PATH:-}"
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" pip install --no-build-isolation . \
    || die "pointnet2_ops 编译失败"
cd "$G"

echo; echo "=== 3. numpy 回到 2.x(上游最后一步)==="
pip install "numpy>=2.0" --force-reinstall --no-deps || die "numpy 装不回去"

echo; echo "=== 4. GraspGen 权重 ==="
cd "$MODELS_DIR"
[ -d GraspGenModels ] || git clone https://huggingface.co/adithyamurali/GraspGenModels
du -sh "$MODELS_DIR/GraspGenModels" 2>/dev/null

# =============================================================================
echo; echo "=== 验收(看实际状态,不看退出码)==="
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
echo "  --- pointnet2_ops 的 cubin 架构(必须含 sm_80)---"
SO=$(find "$E" -name "_ext*.so" -path "*pointnet2*" 2>/dev/null|head -1)
if [ -n "$SO" ]; then "$CONDA_PREFIX/bin/cuobjdump" --list-elf "$SO" 2>/dev/null | head -4 | sed 's/^/    /'
else echo "    (找不到 _ext .so)"; fi

echo
[ "$fail" -eq 0 ] && echo "✓ graspgen 修好了" || echo "✗ 还有 $fail 项不过"
exit "$fail"
