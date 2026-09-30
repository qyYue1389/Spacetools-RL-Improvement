#!/bin/bash
# SpaceTools SFT 环境安装 —— 2× A6000 / RunPod
#
# 依据 github.com/spacetools/SpaceTools 的 setup_envs.sh + docs/SETUP.md,
# 加了两处 RunPod 特有的处理(见 ⚠️)。
#
#   bash setup.sh /workspace          # 参数 = 持久卷路径
#
# 约 1–1.5 h,主要是 flash-attn 编译。不需要 GPU 参与编译,但需要 nvcc。

set -euo pipefail

ROOT="${1:-/workspace}"
PYTHON_VERSION=3.11
TORCH_VERSION=2.9.1
TRANSFORMERS_VERSION=4.57.1

# ⚠️ 关键:cubin 架构覆盖。8.0 是下限(A100),同代向上兼容 8.6(A6000)/ 8.9。
# 不设它,很多 setup.py 会去问 torch.cuda.get_device_capability() —— 只编当前这张卡,换卡就废。
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"
export MAX_JOBS="${MAX_JOBS:-8}"

echo "=== 0. 硬件与驱动 ==="
nvidia-smi --query-gpu=index,name,driver_version,compute_cap,memory.total --format=csv
# 期望:2 × RTX A6000 · driver ≥ 535 · compute_cap 8.6 · 49140 MiB

echo
echo "=== 1. conda(装在持久卷上,否则停机即失)==="
export CONDA_DIR="$ROOT/miniconda3"
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
    bash /tmp/mc.sh -b -p "$CONDA_DIR"
    rm /tmp/mc.sh
fi
export PATH="$CONDA_DIR/bin:$PATH"
eval "$(conda shell.bash hook)"

# 缓存也指到持久卷
mkdir -p "$ROOT"/{hf,pip-cache,conda-pkgs,tmp,models,experiments}
export HF_HOME="$ROOT/hf"
export PIP_CACHE_DIR="$ROOT/pip-cache"
export CONDA_PKGS_DIRS="$ROOT/conda-pkgs"
export TMPDIR="$ROOT/tmp"          # flash-attn 编译会用,别放容器盘
export HF_HUB_ENABLE_HF_TRANSFER=1

echo
echo "=== 2. 克隆仓库(以 GitHub 为准)==="
cd "$ROOT"
[ -d SpaceTools ]     || git clone --depth 1 https://github.com/spacetools/SpaceTools.git
[ -d SpaceTools-SFT ] || git clone --depth 1 https://github.com/ChicyChen/SpaceTools-SFT.git

echo
echo "=== 3. spacetools-sft 环境 ==="
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

echo "--- 其余依赖(照 setup_envs.sh 的 setup_sft())---"
pip install datasets accelerate "peft>=0.18.0" "trl>=0.18.0" torchdata \
    gradio matplotlib "tyro<0.9.0" \
    einops numpy pandas scipy \
    sentencepiece tiktoken modelscope hf-transfer safetensors \
    av fire omegaconf packaging protobuf pyyaml pydantic \
    uvicorn fastapi sse-starlette \
    deepspeed

echo
echo "=== 4. ⚠️ nvcc 版本 —— RunPod 基础镜像最容易踩的地方 ==="
# install_flash_attn.sh 的逻辑是 `if ! which nvcc; then conda install cuda-toolkit=12.8; fi`
# 基础镜像自带 nvcc 但版本不对时,这个 if 会跳过 —— 于是拿错版本的 nvcc 编 cu128 的 flash-attn。
NVCC_VER="$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo none)"
echo "当前 nvcc: $NVCC_VER"
if [ "$NVCC_VER" != "12.8" ]; then
    echo "不是 12.8,在 conda 环境里装一份 ..."
    conda install -c nvidia cuda-toolkit=12.8 -y 2>&1 | tail -3
fi
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
echo "CUDA_HOME=$CUDA_HOME"
nvcc --version | tail -2
nvcc --version | grep -q "release 12.8" || { echo "✗ nvcc 仍然不是 12.8,停"; exit 1; }

echo
echo "=== 5. flash-attn(20–45 min)==="
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
pip install flash-attn --no-build-isolation

echo
echo "=== 6. 验收 ==="
python - <<'PY'
import torch, flash_attn, deepspeed, llamafactory
print(f"torch        {torch.__version__}   cuda {torch.version.cuda}")
print(f"flash-attn   {flash_attn.__version__}")
print(f"deepspeed    {deepspeed.__version__}")
print(f"GPU 可见     {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  sm_{p.major}{p.minor}  {p.total_memory/2**30:.1f} GiB")
assert torch.__version__.startswith("2.9.1"), "torch 版本不对"
assert "12.8" in (torch.version.cuda or ""), "torch 不是 cu128"
print("✓ 版本检查通过")
PY

echo
echo "=== 7. 架构自检 ==="
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SELF_DIR/check_arch.sh" "$CONDA_DIR/envs" || {
    echo "⚠️ 有 .so 缺 sm_80,换到 A100 会跑不了。SFT 在 A6000 上仍能跑,但记一笔。"; }

echo
echo "=========================================="
echo "环境就绪。下一步看 TASK.md 的阶段 B(改配置)。"
echo ""
echo "每次新 shell 记得:"
echo "  export PATH=$CONDA_DIR/bin:\$PATH && eval \"\$(conda shell.bash hook)\""
echo "  conda activate spacetools-sft"
echo "  export HF_HOME=$ROOT/hf TMPDIR=$ROOT/tmp"
echo "=========================================="
