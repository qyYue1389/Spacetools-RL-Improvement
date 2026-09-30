# SpaceTools reproduction — shared environment
# Usage: source /workspace/env.sh   (every fresh shell / after every instance start)
#
# Everything here points at /workspace, which is the persistent volume on this
# instance. The container filesystem (/, /root, /opt, /venv) is only 50 GB and
# the five conda envs alone need 40-60 GB, so envs MUST NOT land there.

export SPACETOOLS_ROOT=/workspace/SpaceTools

# ---------------------------------------------------------------------------
# conda (miniforge on the volume, not the image's /opt/miniforge3)
# ---------------------------------------------------------------------------
export CONDA_ROOT=/workspace/miniforge3
__conda_setup="$("$CONDA_ROOT/bin/conda" shell.bash hook 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    export PATH="$CONDA_ROOT/bin:$PATH"
fi
unset __conda_setup

# /root/.condarc sets `envs_dirs: /venv` at USER scope, which outranks any
# .condarc inside $CONDA_ROOT. Environment variables outrank the user condarc,
# so this is the only reliable way to move envs off the container disk.
export CONDA_ENVS_DIRS=/workspace/envs
export CONDA_PKGS_DIRS=/workspace/conda-pkgs

# ---------------------------------------------------------------------------
# caches and scratch — all on the volume
# ---------------------------------------------------------------------------
export HF_HOME=/workspace/hf
export PIP_CACHE_DIR=/workspace/pip-cache
export TORCH_HOME=/workspace/torch
export TMPDIR=/workspace/tmp          # flash-attn unpacks + compiles CUDA kernels here

# ---------------------------------------------------------------------------
# model weights (required by examples/toolshed/run_eval.sh)
# ---------------------------------------------------------------------------
export MODELS_DIR=/workspace/models
export ROBOREFER_MODEL="$MODELS_DIR/RoboRefer-8B-SFT"
export DEPTH_CHECKPOINT="$MODELS_DIR/depth_pro.pt"
export SPACETOOLS_CKPT="$MODELS_DIR/spacetools-ckpt"

# ---------------------------------------------------------------------------
# eval GPU budget (4x A6000). Defaults in run_eval.sh already match, these make
# the intent explicit and let you override for a 2-GPU downgrade run.
# ---------------------------------------------------------------------------
export NUM_GPUS=4
# The tools need three cards: Molmo fp32 is 33 GB, RoboRefer 18.6 GB and
# DepthPro 12.5 GB PER ACTOR (two actors). Squeezing them onto two cards left
# no headroom and OOMed 68 times in a 124-sample blinkdepth run.
export EVAL_GPUS=1

# ---------------------------------------------------------------------------
# st_activate <env> — activate a conda env AND set the LD_LIBRARY_PATH that
# sglang's spawned subprocesses need to find libcudart.so.12 / libcudnn.so.
# run_eval.sh does this itself (line ~85); this mirrors it for interactive use.
# ---------------------------------------------------------------------------
st_activate() {
    if [ -z "$1" ]; then echo "usage: st_activate <env-name>" >&2; return 1; fi
    conda activate "$1" || return 1
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
    echo "activated $1  (LD_LIBRARY_PATH set)"
}
