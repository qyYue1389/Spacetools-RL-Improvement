#!/bin/bash
# =============================================================================
# Fix graspgen, round two — continue from where install_graspgen.sh died
#
# Progress from the previous round: read prompts bypassed, GraspGen cloned, pyproject patched,
# cuda-toolkit=12.1 installed, grasp_gen itself also built successfully.
# The only cause of death:
#     ERROR: Failed building wheel for pickle5
#     Py_SIZE(self->stack) = len;   ← Py_SIZE is no longer an lvalue from Python 3.11 on
# pickle5 is a backport of protocol 5 for Python 3.5–3.7; the standard library has it from 3.8 on,
# and it does not compile on 3.11 at all. Its only use in GraspGen is
#     grasp_gen/dataset/dataset.py:27   import pickle5 as pickle
# and it is inside try/except ImportError — removing the dependency automatically falls back to the standard library pickle.
# =============================================================================
set -uo pipefail
CONDA_DIR=/opt/conda-st
G=/opt/spacetools/GraspGen
export PATH="$CONDA_DIR/bin:$PATH"
eval "$(conda shell.bash hook)"
die() { echo "✗ $*"; exit 1; }

unset PIP_CONSTRAINT          # the tool environment has its own torch (2.3 series), not bound by the rl constraints
export TMPDIR=/root/tmp PIP_CACHE_DIR=/root/.cache/pip
export MODELS_DIR=/workspace/checkpoints
export MAX_JOBS=6             # cgroup 46.6 GiB
mkdir -p "$MODELS_DIR" "$TMPDIR"

set +u; conda activate spacetools-tool-graspgen; set -u
[ "${CONDA_DEFAULT_ENV:-}" = spacetools-tool-graspgen ] || die "activation failed"
echo "✓ $CONDA_DEFAULT_ENV · $(python -V 2>&1) · $CONDA_PREFIX"

echo; echo "=== 1. Remove pickle5 (deprecated dependency, does not compile on 3.11) ==="
grep -n '"pickle5"' "$G/pyproject.toml" || echo "  (already gone)"
sed -i '/"pickle5",\?/d' "$G/pyproject.toml"
grep -q '"pickle5"' "$G/pyproject.toml" && die "pickle5 was not removed"
echo "  ✓ removed"
echo "  confirm the fallback branch in dataset.py:"; sed -n '25,31p' "$G/grasp_gen/dataset/dataset.py"

echo; echo "=== 2. pip install -e GraspGen ==="
cd "$G"
pip install -e . --find-links https://data.pyg.org/whl/torch-2.3.0+cu121.html \
    || die "grasp_gen install failed"

echo; echo "=== 3. Compile pointnet2_ops (CUDA extension) ==="
cd "$G/pointnet2_ops"
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include:${CPATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/stubs:${LIBRARY_PATH:-}"
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" pip install --no-build-isolation . \
    || die "pointnet2_ops compile failed"
cd "$G"

echo; echo "=== 4. numpy back to 2.x (the last step of upstream install_graspgen.sh) ==="
pip install "numpy>=2.0" --force-reinstall --no-deps || die "cannot reinstall numpy"

echo; echo "=== 5. GraspGen weights ==="
cd "$MODELS_DIR"
[ -d GraspGenModels ] || git clone https://huggingface.co/adithyamurali/GraspGenModels
du -sh "$MODELS_DIR/GraspGenModels" 2>/dev/null

# =============================================================================
echo; echo "=== acceptance check (look at actual state, not the exit code) ==="
# =============================================================================
E="$CONDA_DIR/envs/spacetools-tool-graspgen"
fail=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "  ✓ $1"; else echo "  ✗ $1"; fail=$((fail+1)); fi; }
chk "ray"                "$E/bin/python -c 'import ray'"
chk "toolshed"           "$E/bin/python -c 'import toolshed'"
chk "torch"              "$E/bin/python -c 'import torch'"
chk "pointnet2_ops"      "$E/bin/python -c 'import pointnet2_ops'"
chk "grasp_gen"          "$E/bin/python -c 'import grasp_gen'"
chk "dataset (pickle fallback)" "$E/bin/python -c 'from grasp_gen.dataset.dataset import collate'"
chk "GraspGenModels weights" "[ -d $MODELS_DIR/GraspGenModels ]"
echo "  --- versions ---"
$E/bin/python -c "import torch,ray,numpy;print('  torch',torch.__version__,'cuda',torch.version.cuda,'| ray',ray.__version__,'| numpy',numpy.__version__)" 2>&1|tail -1
echo "  --- cubin architectures of pointnet2_ops ---"
SO=$(find "$E" -name "pointnet2*.so" 2>/dev/null|head -1)
[ -n "$SO" ] && "$E/bin/../bin/cuobjdump" --list-elf "$SO" 2>/dev/null | head -3 || echo "  (cannot find the .so or cuobjdump)"

echo
[ "$fail" -eq 0 ] && echo "✓ graspgen fixed" || echo "✗ $fail item(s) still failing"
exit "$fail"
