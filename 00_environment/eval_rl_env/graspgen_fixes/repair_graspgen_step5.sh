#!/bin/bash
# =============================================================================
# Fix graspgen, round five — compiler version
#
# Already solved in the first four rounds: pickle5 (does not compile on 3.11), the interactive reads for GRASPGEN_DIR/MODELS_DIR,
# PIP_CONSTRAINT must not leak into tool environments, cuda-version is the real version gate.
# Now: nvcc 12.1 ✓, torch 2.3.1+cu121 ✓, grasp_gen ✓, only pointnet2_ops is missing.
#
# Root cause this round:
#   crt/host_config.h:132: #error -- unsupported GNU version!
#                          gcc versions later than 12 are not supported!
#   nvcc 12.1 supports at most gcc 12, while the system gcc on Ubuntu 24.04 is 13.3.
#   install_graspgen.sh assumed an older distribution.
#
# Fix: give nvcc a gcc 12 (conda-forge), instead of forcing past it with -allow-unsupported-compiler
#      — that flag's own warning says "may result in incorrect runtime behavior".
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
[ "${CONDA_DEFAULT_ENV:-}" = spacetools-tool-graspgen ] || die "activation failed"
echo "✓ $CONDA_DEFAULT_ENV"
echo "  nvcc  $(nvcc --version 2>/dev/null|grep -oP 'release \K[0-9.]+')"
python -c "import torch;print('  torch',torch.__version__,'cuda',torch.version.cuda)"
echo "  system gcc $(gcc --version|head -1|grep -oP '\) \K[0-9.]+')  ← nvcc 12.1 only supports ≤12"

echo; echo "=== 1. Install the gcc 12 toolchain ==="
conda install -y -c conda-forge "gcc_linux-64=12" "gxx_linux-64=12" 2>&1 | tail -4
CC_BIN="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
CXX_BIN="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
[ -x "$CC_BIN" ] && [ -x "$CXX_BIN" ] || die "conda gcc did not install"
GV="$("$CC_BIN" --version | head -1 | grep -oP '\) \K[0-9]+')"
echo "  conda gcc: $("$CC_BIN" --version|head -1)"
[ "$GV" = "12" ] || die "conda gcc major version is $GV, not 12"

echo; echo "=== 2. Compile pointnet2_ops (nvcc 12.1 + gcc 12) ==="
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
echo "  pip exit code $RC (full log /workspace/pn2_build.log)"
if [ "$RC" -ne 0 ]; then
    echo "  --- error excerpt ---"
    grep -nE "error:|fatal error|FAILED|unsupported" /workspace/pn2_build.log | head -8 | sed 's/^/    /'
    die "pointnet2_ops compile failed"
fi
python -c "import pointnet2_ops._ext; print('  ✓ pointnet2_ops._ext importable')" || die "compiled successfully but import failed"
cd "$G"

echo; echo "=== 3. numpy back to 2.x ==="
pip install "numpy>=2.0" --force-reinstall --no-deps 2>&1 | tail -2

echo; echo "=== 4. GraspGen weights ==="
cd "$MODELS_DIR"
[ -d GraspGenModels ] || git clone --depth 1 https://huggingface.co/adithyamurali/GraspGenModels
du -sh "$MODELS_DIR/GraspGenModels" 2>/dev/null

# =============================================================================
echo; echo "=== Acceptance check ==="
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
if [ -n "$SO" ]; then "$E/bin/cuobjdump" --list-elf "$SO" 2>/dev/null | head -4 | sed 's/^/    /'
else echo "    (cannot find the _ext .so)"; fi
echo
[ "$fail" -eq 0 ] && echo "✓ graspgen fixed" || echo "✗ $fail item(s) still failing"
exit "$fail"
