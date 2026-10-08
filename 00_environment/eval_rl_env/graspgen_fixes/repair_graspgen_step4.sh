#!/bin/bash
# =============================================================================
# Fix graspgen, round four — only the CUDA toolchain for pointnet2_ops is left
#
# Confirmed facts:
#  · torch 2.3.1+cu121 is installed (the island version GraspGen wants)
#  · grasp_gen itself is installed
#  · the nvcc in the environment is 13.3, because `conda install cuda-toolkit=12.1` only pinned
#    that 1 KB empty-shell metapackage; what really controls the version is `cuda-version`, which is 13.3.
#  · _check_cuda_version in torch/utils/cpp_extension.py only errors when the **major differs**;
#    a different minor is only a warning:
#        if cuda_ver.major != torch_cuda_version.major: raise RuntimeError(...)
#        warnings.warn(...)
#    So the criterion is major==12; no need to insist on 12.1.
#
# Strategy: first pin cuda-version=12.1 the proper way; if that fails, fall back to the nvcc 12.8
#       already in spacetools-rl (major 12, allowed by torch).
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
[ "${CONDA_DEFAULT_ENV:-}" = spacetools-tool-graspgen ] || die "activation failed"
GENV="$CONDA_PREFIX"
RLENV="$CONDA_DIR/envs/spacetools-rl"
echo "✓ $CONDA_DEFAULT_ENV · $(python -V 2>&1)"
python -c "import torch;print('  torch',torch.__version__,'cuda',torch.version.cuda)"

echo; echo "=== 1. Try to pin cuda-version to 12.1 ==="
echo "  nvcc before: $(nvcc_ver "$GENV")  ·  cuda-version: $(conda list 2>/dev/null|awk '/^cuda-version /{print $2}')"
conda install -y -c nvidia "cuda-version=12.1" "cuda-nvcc=12.1" 2>&1 | tail -4
echo "  nvcc after: $(nvcc_ver "$GENV")"

# =============================================================================
echo; echo "=== 2. Pick an nvcc with major==12 ==="
# =============================================================================
if [ "$(nvcc_major "$GENV")" = "12" ]; then
    CUDA_ROOT="$GENV";  echo "  using the graspgen environment's own nvcc $(nvcc_ver "$GENV")"
elif [ "$(nvcc_major "$RLENV")" = "12" ]; then
    CUDA_ROOT="$RLENV"
    echo "  ⚠️ the graspgen environment's nvcc is $(nvcc_ver "$GENV") (major 13, torch will hard-error)"
    echo "     falling back to spacetools-rl's nvcc $(nvcc_ver "$RLENV") — major 12, torch only warns"
    echo "     deviation note: pointnet2_ops compiled with nvcc $(nvcc_ver "$RLENV"), torch is cu121"
else
    die "neither environment has an nvcc with major==12"
fi

echo; echo "=== 3. Compile pointnet2_ops ==="
cd "$G/pointnet2_ops"
export CUDA_HOME="$CUDA_ROOT"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$CUDA_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$GENV/lib:$GENV/lib/stubs:$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
echo "  CUDA_HOME=$CUDA_HOME  ·  nvcc=$(nvcc --version|grep -oP 'release \K[0-9.]+')"
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" pip install --no-build-isolation . 2>&1 | tail -12
python -c "import pointnet2_ops._ext" 2>/dev/null || die "pointnet2_ops._ext failed to install/import"
cd "$G"

echo; echo "=== 4. numpy back to 2.x ==="
pip install "numpy>=2.0" --force-reinstall --no-deps 2>&1 | tail -2

echo; echo "=== 5. GraspGen weights ==="
cd "$MODELS_DIR"
[ -d GraspGenModels ] || git clone --depth 1 https://huggingface.co/adithyamurali/GraspGenModels
du -sh "$MODELS_DIR/GraspGenModels" 2>/dev/null

# =============================================================================
echo; echo "=== acceptance check (look at actual state) ==="
# =============================================================================
E="$CONDA_DIR/envs/spacetools-tool-graspgen"
fail=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "  ✓ $1"; else echo "  ✗ $1"; fail=$((fail+1)); fi; }
chk "ray"                  "$E/bin/python -c 'import ray'"
chk "toolshed"             "$E/bin/python -c 'import toolshed'"
chk "torch"                "$E/bin/python -c 'import torch'"
chk "pointnet2_ops"        "$E/bin/python -c 'import pointnet2_ops'"
chk "pointnet2 CUDA ops"  "$E/bin/python -c 'import pointnet2_ops._ext'"
chk "grasp_gen"            "$E/bin/python -c 'import grasp_gen'"
chk "dataset (pickle fallback)"  "$E/bin/python -c 'from grasp_gen.dataset.dataset import collate'"
chk "GraspGenModels weights"   "[ -d $MODELS_DIR/GraspGenModels ]"
echo "  --- versions ---"
$E/bin/python -c "import torch,ray,numpy;print('  torch',torch.__version__,'cuda',torch.version.cuda,'| ray',ray.__version__,'| numpy',numpy.__version__)" 2>&1|tail -1
echo "  --- cubin of pointnet2_ops (must contain sm_80) ---"
SO=$(find "$E" -name "_ext*.so" -path "*pointnet2*" 2>/dev/null|head -1)
if [ -n "$SO" ]; then "$CUDA_ROOT/bin/cuobjdump" --list-elf "$SO" 2>/dev/null | head -4 | sed 's/^/    /'
else echo "    (cannot find the _ext .so)"; fi
echo
[ "$fail" -eq 0 ] && echo "✓ graspgen fixed" || echo "✗ $fail item(s) still failing"
exit "$fail"
