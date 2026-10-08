#!/bin/bash
# =============================================================================
# SpaceTools eval environment build — 1× A6000 / RunPod
#
# The backbone follows setup_envs.sh from github.com/spacetools/SpaceTools and
# install_tools/setup_tool_env.sh from github.com/NVlabs/SpaceTools-Toolshed.
# Every deviation plugs a hole in upstream; each is marked ⚠️ and will be written into the final package's README.
#
#   nohup bash build_eval_envs.sh > /workspace/build.log 2>&1 &
#
# Idempotent: existing environments are skipped, safe to re-run.
# =============================================================================
set -uo pipefail

# ---- Paths: conda and repos go on the container disk (local, fast); weights go on the network volume (kept across pods) ------
CONDA_DIR=/opt/conda-st
REPO_DIR=/opt/spacetools
export HF_HOME=/workspace/hf
export CHECKPOINT_DIR=/workspace/checkpoints     # ⚠️ see §3.3; without it things hang
export PIP_CACHE_DIR=/root/.cache/pip
export TMPDIR=/root/tmp
export HF_HUB_ENABLE_HF_TRANSFER=1

# ---- Build architectures: must be set explicitly, otherwise only the current GPU is compiled for -------------------------------
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"
export FLASH_ATTN_CUDA_ARCHS="80;86"    # ⚠️ flash-attn does not read the one above, it reads its own
export MAX_JOBS=32

PY=3.11; TRANSFORMERS=4.57.1; TORCH=2.9.1; RAY=2.47.1
# ⚠️ Upstream only pins torch, not torchvision/torchaudio — in the SFT round this pulled a torchaudio built for torch 2.11,
#    giving undefined symbol at import. Pinned in full here.
TORCHVISION=0.24.1; TORCHAUDIO=2.9.1

mkdir -p "$REPO_DIR" "$HF_HOME" "$CHECKPOINT_DIR" "$TMPDIR" "$PIP_CACHE_DIR"
step() { echo; echo "############ $* ############  $(date +%H:%M:%S)"; }
have_env() { "$CONDA_DIR/bin/conda" env list 2>/dev/null | grep -q "^$1 "; }
die()  { echo "✗ $*"; exit 1; }

# ⚠️ This assertion was missing from the first version of this script, and the cost was 16 GB installed into base:
#    conda create fails on ToS → activate fails → script does not stop → pip installs into base's python3.14
#    → only blows up when flash-attn looks for /opt/conda-st/bin/nvcc, by which time ten minutes have passed.
#    "Exit codes lie" — after every step, verify the actual state, do not look at the return value.
act () {   # $1 = environment name. Activate it and prove it is really activated
    set +u; conda activate "$1"; set -u
    [ "${CONDA_DEFAULT_ENV:-}" = "$1" ] || die "expected $1 to be activated, actually got '${CONDA_DEFAULT_ENV:-none}'"
    case "$(python -V 2>&1)" in
        *" $PY."*) ;;
        *) die "the python in $1 is not $PY: $(python -V 2>&1)" ;;
    esac
    echo "✓ activated $1 · $(python -V 2>&1) · $CONDA_PREFIX"
}
# Call once after every big pip step: whichever step changed torch is where we stop, instead of waiting until flash-attn
chk_torch () {
    python -c "import sys,torch; ok=torch.__version__.startswith('2.9.1') and torch.version.cuda=='12.8'; print('  [$1] torch',torch.__version__,'cuda',torch.version.cuda,'OK' if ok else 'BAD'); sys.exit(0 if ok else 1)" \
        || die "torch was changed by the '$1' step — see the actual version in the line above"
}
mk_env () {   # $1 = environment name
    conda create -n "$1" python==$PY -y || die "conda create $1 failed"
    have_env "$1" || die "conda create $1 reported success but the environment does not exist"
}

# =============================================================================
step "0. conda + system dependencies"
# =============================================================================
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$TMPDIR/mc.sh"
    bash "$TMPDIR/mc.sh" -b -p "$CONDA_DIR" && rm -f "$TMPDIR/mc.sh"
fi
export PATH="$CONDA_DIR/bin:$PATH"
eval "$(conda shell.bash hook)"

# ⚠️ conda ≥26 cannot create environments unless the ToS is accepted (CondaToSNonInteractiveError),
#    and it only reports this on stderr; if create's non-zero return is not checked, everything goes wrong from there on.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true
conda config --set always_yes true >/dev/null 2>&1 || true
# Self-check: can we really create an environment
conda create -n _tos_probe python==$PY -y >/dev/null 2>&1 \
    && conda env remove -n _tos_probe -y >/dev/null 2>&1 \
    || die "conda still cannot create environments (ToS or channel config). Run conda create once by hand and look at the error"
echo "✓ conda can create environments"

command -v wget >/dev/null || (apt-get update -qq && apt-get install -y -qq wget)
command -v git  >/dev/null || (apt-get update -qq && apt-get install -y -qq git)
conda --version; nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

# =============================================================================
step "1. Clone repos (GitHub is authoritative)"
# =============================================================================
cd "$REPO_DIR"
[ -d SpaceTools ]           || git clone --depth 1 https://github.com/spacetools/SpaceTools.git
[ -d SpaceTools-RL ]        || git clone --depth 1 https://github.com/ChicyChen/SpaceTools-RL.git
[ -d SpaceTools-Toolshed ]  || git clone --depth 1 https://github.com/NVlabs/SpaceTools-Toolshed.git
for r in SpaceTools SpaceTools-RL SpaceTools-Toolshed; do
    echo "$r  $(git -C $r rev-parse HEAD)"
done

# =============================================================================
step "2. spacetools-rl (per setup_rl in setup_envs.sh, with three additions)"
# =============================================================================
if have_env spacetools-rl; then
    echo "already exists, skipping"
else
mk_env spacetools-rl
act spacetools-rl

# 2.1 torch — ⚠️ deviation ① (two places):
#   a) upstream `pip install torch==2.9.1 torchvision torchaudio` does not pin the last two;
#      in the SFT round this pulled a torchaudio built for torch 2.11, giving undefined symbol at import.
#   b) upstream has no --index-url, so it gets the PyPI default wheel. When the paper was written, the default for torch 2.9.1 was
#      cu128; **the default has now become cu130** — so when nvcc 12.8 compiles flash-attn it reports
#      "detected CUDA 12.8 mismatches the version used to compile PyTorch 13.0".
#      Must pin cu128: the SFT checkpoint was trained with torch 2.9.1+cu128, and the P4/P5
#      eval environment is also cu128. Accommodating cu130 would split the definition across the whole chain, and without any error.
pip install torch==$TORCH torchvision==$TORCHVISION torchaudio==$TORCHAUDIO \
    --index-url https://download.pytorch.org/whl/cu128
pip install transformers==$TRANSFORMERS

# ⚠️ Deviation ①c: a constraints file, so that no later pip install can change these four.
#    Measured: the latest flashinfer_python requires torch > 2.9.1, and pip will **silently** upgrade torch
#    together with the whole nvidia-*-cu13 set to 2.14.0+cu130 (that is how build.fail2 happened).
#    With the constraints, a conflict becomes an explicit pip error instead of an environment whose definition quietly changed.
cat > /root/constraints.txt <<EOF
torch==$TORCH
torchvision==$TORCHVISION
torchaudio==$TORCHAUDIO
transformers==$TRANSFORMERS
EOF
export PIP_CONSTRAINT=/root/constraints.txt
echo "PIP_CONSTRAINT=$PIP_CONSTRAINT"; cat "$PIP_CONSTRAINT"

# 2.2 sglang — ⚠️ deviation ④: upstream has
#       pip install "sglang[srt,openai]==0.5.6" --no-deps
#       pip install <a long hand-written list of dependencies>
#     --no-deps is there to keep sglang from bumping torch/transformers. But that hand-written list is **incomplete**:
#       ModuleNotFoundError: No module named 'pybase64'
#     and the manually installed flashinfer_python (unpinned) bumps nvidia-cutlass-dsl to
#     4.8.0.dev0, while sglang 0.5.6 requires ==4.2.1.
#     We have PIP_CONSTRAINT guarding torch/transformers, so we can let pip resolve
#     sglang's real dependency closure normally — sglang decides the versions itself, more accurate than the hand-written list.
pip install "sglang[srt,openai]==0.5.6" || die "sglang dependency resolution failed (check whether it conflicts with torch 2.9.1)"
# Extras from upstream's hand-written list that are outside the sglang closure (excluding flashinfer_python — let sglang set its version)
pip install anthropic openai blobfile decord2 torchao \
    lm-format-enforcer cuda-python sglang_router

chk_torch "sglang deps"

pip install "ray[default]==$RAY"
chk_torch "ray"

# 2.3 verl
cd "$REPO_DIR/SpaceTools-RL" && pip install -e "." --no-deps
pip install accelerate codetiming datasets dill hydra-core \
    "numpy<2.0.0" pandas peft "pyarrow>=19.0.0" pybind11 pylatexenc \
    torchdata "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    wandb tensorboard packaging

chk_torch "verl deps"

# 2.4 toolshed
cd "$REPO_DIR/SpaceTools-Toolshed" && pip install -e . --no-deps
pip install docstring_parser aiohttp aiohttp-cors Pillow \
    fastapi uvicorn python-multipart openai botocore pyyaml \
    anthropic matplotlib scipy requests click uvloop

chk_torch "toolshed deps"

# 2.4b ⚠️ Guard: the long dependency list above includes packages that depend on torch, such as torchao / flashinfer_python,
#      and pip may re-resolve torch to a different build. Before compiling flash-attn we must confirm it is still cu128,
#      otherwise the error it reports (CUDA version mismatch) is far from the real cause.
chk_torch "final confirmation before compile"
# Whether sglang / flashinfer installed on this torch can still be imported — once torch is constrained,
# the real risk changes from "torch gets swapped" to "flashinfer is incompatible with 2.9.1", which needs its own check.
python -c "import sglang, flashinfer; print('  sglang', sglang.__version__, '· flashinfer OK')" \
    || die "sglang/flashinfer import failed — check whether the ModuleNotFoundError above is a missing package (incomplete dependency closure) or an ABI incompatibility (torch version)"
pip check 2>&1 | head -10 || true   # conflicts are reported, not blocking: upstream already has several loose constraints

# 2.5 nvcc — flash-attn needs to be compiled
if ! nvcc --version 2>/dev/null | grep -q "release 12.8"; then
    conda install -c nvidia cuda-toolkit=12.8 -y 2>&1 | tail -2
fi
export CUDA_HOME="$CONDA_PREFIX"; export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version | tail -1

# 2.6 flash-attn — ⚠️ deviation ②: upstream has
#       pip install flash-attn --no-build-isolation 2>/dev/null || echo "WARNING: ..."
#     Errors are swallowed and it continues on failure. Here a failure stops the script.
echo "compiling flash-attn (FLASH_ATTN_CUDA_ARCHS=$FLASH_ATTN_CUDA_ARCHS, 20-45 min)..."
pip install flash-attn --no-build-isolation || { echo "✗ flash-attn compile failed, stopping"; exit 1; }

pip install qwen_vl_utils

# 2.7 ⚠️ Deviation ③: second cudnn upgrade + cachetools.
#     setup_envs.sh does not have these two steps, but the manual path in docs/SETUP.md does,
#     and the P1 notes confirm they are required (flash-attn downgrades cudnn).
pip install cachetools "nvidia-cudnn-cu12==9.16.0.29"

python - <<'PY'
import torch, verl, toolshed, flash_attn, torchvision, torchaudio, transformers
print("torch       ", torch.__version__, "cuda", torch.version.cuda)
print("torchvision ", torchvision.__version__)
print("torchaudio  ", torchaudio.__version__)
print("transformers", transformers.__version__)
print("flash_attn  ", flash_attn.__version__)
PY
conda deactivate
fi

# =============================================================================
step "3. Four tool environments (Toolshed setup_tool_env.sh)"
# =============================================================================
act spacetools-rl        # setup_tool_env.sh detects Python + Ray from the currently activated environment
cd "$REPO_DIR/SpaceTools-Toolshed"

mk_tool_env () {   # $1 = env name, $2 = tool name
    if have_env "$1"; then echo "== $1 already exists, skipping"; return; fi
    step "3.x $1  ($2)"
    ( set +u; source install_tools/setup_tool_env.sh "$1" "$2" ) \
        || die "$1 build failed (setup_tool_env.sh exited non-zero)"
    have_env "$1" || die "$1 reported success but the environment does not exist"
    echo "✓ $1 built"
}

# Order: lightest first, most likely to break second — graspgen's pointnet2_ops has to compile a CUDA extension,
# and it is the only environment that "none of these four eval keys uses at all, yet must be installed" (the schema must have 11).
mk_tool_env spacetools-tool-bbox      bbox
mk_tool_env spacetools-tool-graspgen  graspgen
mk_tool_env spacetools-tool-vlm       vlm
mk_tool_env spacetools-tool-roborefer roborefer

# ⚠️ Deviation ④ (actually plugging a hole in docs/SETUP.md):
#    run_eval.sh also puts sam2 and depth_estimator in spacetools-tool-vlm
#      'sam2':            'conda_env': 'spacetools-tool-vlm'
#      'depth_estimator': 'conda_env': 'spacetools-tool-vlm'
#    but setup_tool_env.sh <env> vlm only installs tool-vlm.txt (Molmo's dependencies).
#    Following the docs gives an environment that cannot run sam2/depth, and the error only appears when the actor is called.
#    Note: the header comment of tool-depth.txt says "needs numpy<2, must be a separate environment" — stale;
#        the actual requirements do not pin numpy at all, and do not conflict with numpy>2 for vlm/sam2.
step "3.5 Add sam2 + depth dependencies to spacetools-tool-vlm"
act spacetools-tool-vlm
bash install_tools/tool_scripts/install_sam2.sh  || echo "✗ sam2 dependencies failed"
bash install_tools/tool_scripts/install_depth.sh || echo "✗ depth dependencies failed"
conda deactivate

# =============================================================================
step "4. Weights (on the network volume, kept across pods; large files write sequentially at 486 MB/s, unaffected by the small-file slowness)"
# =============================================================================
act spacetools-rl
pip install -q hf-transfer huggingface_hub
python - <<'PY'
import os
from huggingface_hub import snapshot_download
for repo in ["Zhoues/RoboRefer-8B-SFT", "allenai/Molmo-7B-D-0924",
             "Qwen/Qwen2.5-VL-3B-Instruct"]:
    print(f"--- {repo}", flush=True)
    p = snapshot_download(repo, max_workers=16)
    print("  ->", p, flush=True)
PY
# Eval set
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("siyich/spacetools-eval-benchmarks", repo_type="dataset", max_workers=16)
print("benchmarks ->", p)
PY
# DepthPro ckpt (skipped if install_depth.sh already downloaded it)
[ -f "$CHECKPOINT_DIR/depth_pro.pt" ] || \
  wget -q --show-progress https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -O "$CHECKPOINT_DIR/depth_pro.pt"
ls -lh "$CHECKPOINT_DIR"
du -sh "$HF_HOME"
conda deactivate

# =============================================================================
step "4b. Freeze versions (the repo only pins six packages; everything else is that day's latest)"
# =============================================================================
# Upstream setup_envs.sh writes only the package name for most dependencies (sglang_router / flashinfer_python /
# torchao / accelerate / peft / ...), which pip resolves to that day's latest. When the paper was written, when P4 built its environment,
# and today, all three got different versions — all three failures of this build came from this.
# The repo cannot be reproducible across time, but we can make "this one time" reproducible: freeze the versions actually installed,
# so that every eval for both arms of RL afterwards restores from the same copy, keeping things internally consistent.
FREEZE=/workspace/freeze
mkdir -p "$FREEZE"
for env in spacetools-rl spacetools-tool-vlm spacetools-tool-roborefer \
           spacetools-tool-bbox spacetools-tool-graspgen; do
    have_env "$env" || continue
    "$CONDA_DIR/envs/$env/bin/pip" freeze > "$FREEZE/$env.txt" 2>/dev/null
    echo "$env: $(wc -l < "$FREEZE/$env.txt") packages → $FREEZE/$env.txt"
done
# Overview of key versions, for comparison with P4's PROVENANCE
{
  echo "# key versions of this build  $(date -u +%FT%TZ)"
  echo "# driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  for env in spacetools-rl spacetools-tool-vlm spacetools-tool-roborefer \
             spacetools-tool-bbox spacetools-tool-graspgen; do
    [ -f "$FREEZE/$env.txt" ] || continue
    echo "## $env"
    grep -iE "^(torch|torchvision|torchaudio|transformers|ray|sglang|flash-attn|flashinfer|numpy|nvidia-cudnn-cu12|accelerate|bitsandbytes|timm)==" \
        "$FREEZE/$env.txt" | sed 's/^/  /'
  done
} > "$FREEZE/KEY_VERSIONS.txt"
cat "$FREEZE/KEY_VERSIONS.txt"

# =============================================================================
step "5. Architecture self-check: do all compiled .so files contain sm_80"
# =============================================================================
# ⚠️ The earlier check_arch only scanned "*.so" and missed things like libcublas.so.12. Both kinds are scanned here.
BAD=0
for env in spacetools-rl spacetools-tool-vlm spacetools-tool-roborefer \
           spacetools-tool-bbox spacetools-tool-graspgen; do
    d="$CONDA_DIR/envs/$env"
    [ -d "$d" ] || continue
    n=0; bad=0
    while IFS= read -r so; do
        out=$(cuobjdump --list-elf "$so" 2>/dev/null) || continue
        [ -z "$out" ] && continue
        n=$((n+1))
        echo "$out" | grep -q "sm_8[06]" || { echo "  ✗ no sm_80/86: $so"; bad=$((bad+1)); }
    done < <(find "$d" \( -name "*.so" -o -name "*.so.*" \) 2>/dev/null | head -400)
    echo "$env: scanned $n .so files containing cubin, $bad missing sm_80/86"
    BAD=$((BAD+bad))
done
echo "architecture self-check: $BAD problem(s)"

# =============================================================================
step "Build finished"
# =============================================================================
conda env list
df -h / /workspace | grep -Ev "tmpfs|Filesystem"
echo "next step: bash smoke_tools.sh — per-tool smoke test"
