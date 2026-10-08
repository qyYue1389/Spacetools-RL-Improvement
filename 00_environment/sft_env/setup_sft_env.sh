#!/bin/bash
# SpaceTools SFT environment install — 2× A6000 / RunPod
#
# Based on setup_envs.sh + docs/SETUP.md from github.com/spacetools/SpaceTools,
# with two RunPod-specific additions (see ⚠️).
#
#   bash setup.sh /workspace          # argument = persistent volume path
#
# About 1–1.5 h, mostly flash-attn compilation. The GPU is not needed for compiling, but nvcc is.

set -euo pipefail

ROOT="${1:-/workspace}"
PYTHON_VERSION=3.11
TORCH_VERSION=2.9.1
TRANSFORMERS_VERSION=4.57.1

# ⚠️ Key: cubin architecture coverage. 8.0 is the floor (A100), forward-compatible within the generation to 8.6 (A6000) / 8.9.
# Without it, many setup.py files ask torch.cuda.get_device_capability() — they compile only for the current GPU, useless on a different GPU.
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"
export MAX_JOBS="${MAX_JOBS:-8}"

echo "=== 0. Hardware and driver ==="
nvidia-smi --query-gpu=index,name,driver_version,compute_cap,memory.total --format=csv
# Expected: 2 × RTX A6000 · driver ≥ 535 · compute_cap 8.6 · 49140 MiB

echo
echo "=== 1. conda (installed on the persistent volume, otherwise lost on shutdown) ==="
export CONDA_DIR="$ROOT/miniconda3"
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
    bash /tmp/mc.sh -b -p "$CONDA_DIR"
    rm /tmp/mc.sh
fi
export PATH="$CONDA_DIR/bin:$PATH"
eval "$(conda shell.bash hook)"

# Caches also point to the persistent volume
mkdir -p "$ROOT"/{hf,pip-cache,conda-pkgs,tmp,models,experiments}
export HF_HOME="$ROOT/hf"
export PIP_CACHE_DIR="$ROOT/pip-cache"
export CONDA_PKGS_DIRS="$ROOT/conda-pkgs"
export TMPDIR="$ROOT/tmp"          # used by the flash-attn compile, do not put it on the container disk
export HF_HUB_ENABLE_HF_TRANSFER=1

echo
echo "=== 2. Clone repos (GitHub is authoritative) ==="
cd "$ROOT"
[ -d SpaceTools ]     || git clone --depth 1 https://github.com/spacetools/SpaceTools.git
[ -d SpaceTools-SFT ] || git clone --depth 1 https://github.com/ChicyChen/SpaceTools-SFT.git

echo
echo "=== 3. spacetools-sft environment ==="
conda env list | grep -q "^spacetools-sft " || \
    conda create -n spacetools-sft python=="$PYTHON_VERSION" -y
conda activate spacetools-sft

echo "--- torch $TORCH_VERSION (cu128) ---"
pip install torch=="$TORCH_VERSION" torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
pip install transformers=="$TRANSFORMERS_VERSION"

echo "--- llamafactory ---"
cd "$ROOT/SpaceTools-SFT"
pip install -e ".[torch,metrics]" --no-deps 2>/dev/null || pip install -e "." --no-deps

echo "--- remaining dependencies (per setup_sft() in setup_envs.sh) ---"
pip install datasets accelerate "peft>=0.18.0" "trl>=0.18.0" torchdata \
    gradio matplotlib "tyro<0.9.0" \
    einops numpy pandas scipy \
    sentencepiece tiktoken modelscope hf-transfer safetensors \
    av fire omegaconf packaging protobuf pyyaml pydantic \
    uvicorn fastapi sse-starlette \
    deepspeed

echo
echo "=== 4. ⚠️ nvcc version — the easiest place to trip on the RunPod base image ==="
# The logic of install_flash_attn.sh is `if ! which nvcc; then conda install cuda-toolkit=12.8; fi`
# When the base image ships its own nvcc but with the wrong version, this if is skipped — so the wrong nvcc version compiles the cu128 flash-attn.
NVCC_VER="$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo none)"
echo "current nvcc: $NVCC_VER"
if [ "$NVCC_VER" != "12.8" ]; then
    echo "not 12.8, installing one in the conda environment ..."
    conda install -c nvidia cuda-toolkit=12.8 -y 2>&1 | tail -3
fi
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
echo "CUDA_HOME=$CUDA_HOME"
nvcc --version | tail -2
nvcc --version | grep -q "release 12.8" || { echo "✗ nvcc is still not 12.8, stopping"; exit 1; }

echo
echo "=== 5. flash-attn(20–45 min)==="
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
pip install flash-attn --no-build-isolation

echo
echo "=== 6. Acceptance check ==="
python - <<'PY'
import torch, flash_attn, deepspeed, llamafactory
print(f"torch        {torch.__version__}   cuda {torch.version.cuda}")
print(f"flash-attn   {flash_attn.__version__}")
print(f"deepspeed    {deepspeed.__version__}")
print(f"GPU visible  {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  sm_{p.major}{p.minor}  {p.total_memory/2**30:.1f} GiB")
assert torch.__version__.startswith("2.9.1"), "wrong torch version"
assert "12.8" in (torch.version.cuda or ""), "torch is not cu128"
print("✓ version check passed")
PY

echo
echo "=== 7. Architecture self-check ==="
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SELF_DIR/check_arch.sh" "$CONDA_DIR/envs" || {
    echo "⚠️ some .so are missing sm_80 and will not run on an A100. SFT still runs on A6000, but note it."; }

echo
echo "=========================================="
echo "environment ready. Next, see stage B (change config) in TASK.md."
echo ""
echo "in every new shell, remember:"
echo "  export PATH=$CONDA_DIR/bin:\$PATH && eval \"\$(conda shell.bash hook)\""
echo "  conda activate spacetools-sft"
echo "  export HF_HOME=$ROOT/hf TMPDIR=$ROOT/tmp"
echo "=========================================="
