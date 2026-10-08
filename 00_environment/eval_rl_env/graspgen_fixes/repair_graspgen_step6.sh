#!/bin/bash
# =============================================================================
# Fix graspgen, round six — the GPU architecture list hard-coded in pointnet2_ops
#
# Root cause (the thirteenth upstream hole, in the pointnet2_ops embedded in GraspGen):
#   pointnet2_ops/pointnet2_utils.py:23
#       os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
#   it overwrites the environment variable directly **at import time** (this extension is JIT-compiled on first import),
#   so the 8.0;8.6;8.9 set outside has no effect at all, and nvcc gets compute_37:
#       nvcc fatal : Unsupported gpu architecture 'compute_37'
#   CUDA 12 no longer supports sm_37.
#
# Fix: change the assignment to setdefault — use the external value if one is set, and its default only if not.
#      This is the smallest change, and it does not change upstream behavior "when nobody sets it".
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
[ "${CONDA_DEFAULT_ENV:-}" = spacetools-tool-graspgen ] || die "activation failed"
SP="$CONDA_PREFIX/lib/python3.11/site-packages"
CC_BIN="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
CXX_BIN="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDA_HOME="$CONDA_PREFIX" CC="$CC_BIN" CXX="$CXX_BIN"
export PATH="$CUDA_HOME/bin:$PATH"
export NVCC_PREPEND_FLAGS="-ccbin $CC_BIN"
echo "✓ $CONDA_DEFAULT_ENV · nvcc $(nvcc --version|grep -oP 'release \K[0-9.]+') · gcc $($CC_BIN --version|head -1|grep -oP '\) \K[0-9.]+')"

echo; echo "=== 1. Change the hard-coded architecture list to setdefault ==="
PATCHED=0
for f in "$SP/pointnet2_ops/pointnet2_utils.py" \
         "$G/pointnet2_ops/pointnet2_ops/pointnet2_utils.py" \
         "$G/pointnet2_ops/build/lib.linux-x86_64-cpython-311/pointnet2_ops/pointnet2_utils.py"; do
    [ -f "$f" ] || continue
    grep -q 'os.environ\["TORCH_CUDA_ARCH_LIST"\] =' "$f" || { echo "  (already changed) $f"; continue; }
    sed -i 's|os\.environ\["TORCH_CUDA_ARCH_LIST"\] = |os.environ.setdefault("TORCH_CUDA_ARCH_LIST", |; s|"3\.7+PTX;5\.0;6\.0;6\.1;6\.2;7\.0;7\.5"$|"3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5")|' "$f"
    python -c "import ast,sys; ast.parse(open('$f').read())" || die "broke the syntax: $f"
    echo "  ✓ $f"; sed -n '23p' "$f" | sed 's/^/      /'
    PATCHED=$((PATCHED+1))
done
[ "$PATCHED" -gt 0 ] || echo "  (no files need changing)"

echo; echo "=== 2. Clear the JIT cache and trigger a recompile ==="
rm -rf ~/.cache/torch_extensions "$G/pointnet2_ops/build"
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
timeout 1200 python -c "
import os, pointnet2_ops
print('  arch list in effect:', os.environ.get('TORCH_CUDA_ARCH_LIST'))
import pointnet2_ops._ext as E
print('  ✓ pointnet2_ops._ext compiled and imported successfully:', E.__file__)
" 2>&1 | tail -25
python -c "import pointnet2_ops._ext" 2>/dev/null || die "pointnet2_ops._ext still fails to import"

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
echo "  --- cubin of _ext (must contain sm_80) ---"
SO=$(find ~/.cache/torch_extensions "$E" -name "_ext*.so" 2>/dev/null|head -1)
if [ -n "$SO" ]; then echo "    $SO"; "$E/bin/cuobjdump" --list-elf "$SO" 2>/dev/null|head -4|sed 's/^/    /'
else echo "    (cannot find the _ext .so)"; fi
echo
[ "$fail" -eq 0 ] && echo "✓ graspgen fixed" || echo "✗ $fail item(s) still failing"
exit "$fail"
