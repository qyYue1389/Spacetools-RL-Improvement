#!/bin/bash
# Rebuild pointnet2_ops for sm_80;86;89 after the A6000 -> A100 migration.
# The A6000 build produced sm_86-only cubins with no PTX fallback, which fail on
# sm_80 with "no kernel image is available for execution on the device".
#
# Neutralises the three conda-injected build traps recorded in P1:
#   - NVCC_PREPEND_FLAGS pins nvcc's host compiler to conda's gcc 15.3 (nvcc 12.8
#     refuses > 14). We point it at the system g++-13 instead.
#   - CFLAGS/CXXFLAGS carry conda's CUDA 13.3 headers.
#   - LDFLAGS injects -L$CONDA_PREFIX/lib AHEAD of torch's -L/usr/local/cuda/lib64,
#     so the link resolved libcudart from the env's CUDA 13.3 rather than the
#     system 12.8. This is the real mechanism behind deviation [3]; the P1 note
#     blamed LIBRARY_PATH/stubs, but LIBRARY_PATH is only searched after explicit
#     -L. /usr/local/cuda/lib64 is on the system ldconfig path, so dropping
#     conda's -L costs nothing at runtime.
# We do NOT use -allow-unsupported-compiler: NVIDIA warns it can produce silently
# wrong results, unacceptable for operators that emit geometric poses.
set -eo pipefail

source /workspace/env.sh >/dev/null 2>&1
conda activate spacetools-tool-graspgen

SRC=/workspace/SpaceTools/SpaceTools-Toolshed/GraspGen/pointnet2_ops

# Stale sm_86 objects would be silently reused by setuptools.
rm -rf "$SRC/build" "$SRC/pointnet2_ops.egg-info"

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13
export NVCC_PREPEND_FLAGS="-ccbin=/usr/bin/g++-13"
unset CFLAGS CXXFLAGS CPATH
unset LDFLAGS
export LIBRARY_PATH="$CUDA_HOME/lib64"        # real 12.8 cudart, not the stubs
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
export MAX_JOBS=16
unset PIP_CONSTRAINT || true                   # graspgen pins torch<2.4

echo "nvcc:  $(nvcc --version | tail -2 | head -1)"
echo "host:  $($CXX --version | head -1)"
echo "archs: $TORCH_CUDA_ARCH_LIST"
echo "LIBRARY_PATH=$LIBRARY_PATH"
echo "LDFLAGS=${LDFLAGS:-<unset>}"
echo

cd "$SRC"
pip install --no-build-isolation --force-reinstall --no-deps -v . 2>&1 | \
    grep -E "gencode|arch=|error|Error|ERROR|warning: |Building wheel|Successfully|Installing" | head -60
