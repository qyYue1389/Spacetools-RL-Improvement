#!/bin/bash
# =============================================================================
# Fix graspgen, round three — only one problem left, the CUDA toolchain version
#
# Upstream install_graspgen.sh has:
#     conda install -c nvidia cuda-toolkit=12.1 -y
# But `cuda-toolkit` is a 1 KB metapackage, and its dependencies on the components are loose. Measured result:
#     cuda-toolkit  12.1.1     ← metapackage version number is right
#     cuda-version  13.3       ← the components are all 13.3
#     nvcc          13.3.73
# So when compiling pointnet2_ops:
#     RuntimeError: The detected CUDA version (13.3) mismatches PyTorch (12.1)
#
# Correct way to pin: use the versioned label channel, which pins the whole set of components together.
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
[ "${CONDA_DEFAULT_ENV:-}" = spacetools-tool-graspgen ] || die "activation failed"
echo "✓ $CONDA_DEFAULT_ENV · $(python -V 2>&1)"
python -c "import torch;print('  torch',torch.__version__,'cuda',torch.version.cuda)"

echo; echo "=== 1. Downgrade the CUDA toolchain to 12.1 (label channel pins the whole set) ==="
echo "  current nvcc: $(nvcc --version 2>/dev/null|grep -oP 'release \K[0-9.]+' || echo none)"
conda install -y -c "nvidia/label/cuda-12.1.1" --override-channels \
      cuda-toolkit 2>&1 | tail -5
NV="$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo none)"
echo "  nvcc after install: $NV"
[ "$NV" = "12.1" ] || die "nvcc is still not 12.1 (it is $NV) — switching to the label channel did not pin it either, needs manual investigation"
conda list 2>/dev/null | grep -E "^cuda-version" | sed 's/^/  /'

echo; echo "=== 2. Compile pointnet2_ops ==="
cd "$G/pointnet2_ops"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include:${CPATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/stubs:${LIBRARY_PATH:-}"
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" pip install --no-build-isolation . \
    || die "pointnet2_ops compile failed"
cd "$G"

echo; echo "=== 3. numpy back to 2.x (the last upstream step) ==="
pip install "numpy>=2.0" --force-reinstall --no-deps || die "cannot reinstall numpy"

echo; echo "=== 4. GraspGen weights ==="
cd "$MODELS_DIR"
[ -d GraspGenModels ] || git clone https://huggingface.co/adithyamurali/GraspGenModels
du -sh "$MODELS_DIR/GraspGenModels" 2>/dev/null

# =============================================================================
echo; echo "=== acceptance check (look at actual state, not the exit code) ==="
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
echo "  --- cubin architectures of pointnet2_ops (must contain sm_80) ---"
SO=$(find "$E" -name "_ext*.so" -path "*pointnet2*" 2>/dev/null|head -1)
if [ -n "$SO" ]; then "$CONDA_PREFIX/bin/cuobjdump" --list-elf "$SO" 2>/dev/null | head -4 | sed 's/^/    /'
else echo "    (cannot find the _ext .so)"; fi

echo
[ "$fail" -eq 0 ] && echo "✓ graspgen fixed" || echo "✗ $fail item(s) still failing"
exit "$fail"
