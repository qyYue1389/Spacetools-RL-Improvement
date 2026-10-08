# SpaceTools Eval Environment — Build and Usage Guide

Build date 2026-09-10 · RTX A6000 48GB · Ubuntu 24.04.3 · driver 580.159.04 · system CUDA 12.8

Package location: `https://huggingface.co/qzpm55555/spacetools-eval-env` (private)

> **The GitHub source repos are authoritative, not any fork.**
> `spacetools/SpaceTools` · `ChicyChen/SpaceTools-RL` · `NVlabs/SpaceTools-Toolshed` · `NVlabs/GraspGen` · `Zhoues/RoboRefer`

---

## Table of contents

1. [Overall architecture: why five environments](#1-overall-architecture-why-five-environments)
2. [Versions and functions of the five environments](#2-versions-and-functions-of-the-five-environments)
3. [Building from scratch: how to build each environment](#3-building-from-scratch-how-to-build-each-environment)
4. [How to run](#4-how-to-run)
5. [The 20 upstream holes](#5-the-20-upstream-holes)
6. [Acceptance criteria](#6-acceptance-criteria)
7. [Known limitations and open items](#7-known-limitations-and-open-items)

---

## 1. Overall architecture: why five environments

SpaceTools is a **tool-augmented spatial reasoning** system. The policy model (Qwen2.5-VL-3B) does not answer the question directly; instead it writes code that calls tools, the tools return results, and the model continues reasoning from them. There are 11 available tools, backed by 7 real implementations.

The dependencies of these tools **conflict with each other** and cannot be installed into one environment:

| Tool | Conflict |
|---|---|
| GraspGen | needs `torch>=2.3,<2.4` + CUDA 12.1 toolchain (has to compile a CUDA extension on the spot) |
| RoboRefer (NVILA/llava) | needs `torch==2.5.1` + `transformers==4.49.0` + `numpy==1.26.4` |
| Molmo (VLM) | needs `transformers==4.53.2` (goes through `trust_remote_code`; the remote code is only compatible with specific versions) |
| verl + sglang (training/inference frameworks) | `sglang 0.5.6` pins exactly `torch==2.9.1`; verl pins `numpy<2.0.0` |

**Toolshed's solution**: each tool runs in its own conda environment; Ray starts the actors with `runtime_env={"conda": <env name>}`, with a router actor in between doing routing and load balancing.

```
                    ┌─────────────────────────────────────────┐
                    │  driver / agent   (spacetools-rl)       │
                    │  verl + sglang + policy model           │
                    │  numpy 1.26.4                           │
                    └───────────────┬─────────────────────────┘
                                    │ ToolkitClient.call_tool()
                    ┌───────────────▼─────────────────────────┐
                    │  ToolRouterActor  (toolshed)            │
                    │  routing / load balancing / timeouts    │
                    └──┬────────┬────────┬────────┬───────────┘
       runtime_env:conda│        │        │        │
        ┌───────────────▼┐ ┌─────▼──────┐ ┌▼──────────────┐ ┌▼───────────────┐
        │ tool-vlm       │ │ tool-      │ │ tool-bbox     │ │ tool-graspgen  │
        │ torch 2.9.1    │ │ roborefer  │ │ no torch      │ │ torch 2.3.1    │
        │ cu128          │ │ torch2.5.1 │ │ numpy 2.4.6   │ │ cu121          │
        │ numpy 1.26.4   │ │ cu124      │ │               │ │ numpy 2.4.6    │
        │                │ │ numpy1.26.4│ │               │ │                │
        │ Molmo          │ │ RoboRefer  │ │ vision_ops    │ │ grasp_generator│
        │ sam2           │ │ 8B         │ │ bounding_box  │ │                │
        │ depth_estimator│ │            │ │               │ │                │
        └────────────────┘ └────────────┘ └───────────────┘ └────────────────┘
```

**Two hard constraints that apply everywhere:**

**① `ray` must be exactly the same in all five environments (2.47.1).** When starting actors across environments Ray does a version handshake; a mismatch fails outright. `create_tool_env.sh` probes the Python and Ray versions from the currently active environment to align them — so `spacetools-rl` must be activated before building the tool environments.

**② The numpy major version must be compatible with the driver.** A tool's return value (`ToolResult`) may carry an `ndarray` (depth map, segmentation mask) that has to cross processes back to the driver. Arrays serialized by numpy 2.x reference `numpy._core`, and **numpy 1.x cannot deserialize them**; the other direction (1.x → 2.x) works because numpy 2 ships a compatibility shim.

The driver-side `verl` pins `numpy<2.0.0`, so:

| Environment | Does the return value carry an ndarray | numpy requirement |
|---|---|---|
| `tool-vlm` | **yes** (`$depth_map`, `$segmentation_mask`) | **must be 1.26.4** |
| `tool-roborefer` | no (returns only coordinate text) | 1.26.4 (pinned by RoboRefer itself) |
| `tool-bbox` | no (float list after `.tolist()`) | 2.4.6 is fine |
| `tool-graspgen` | no (`grasp_pose` is already `.tolist()`) | 2.4.6 is fine |

**Four torches and four CUDA runtimes (12.8 / 12.8 / 12.4 / 12.1) coexist without interfering**, because each conda environment carries its own set of `nvidia-*` wheels. This is not a compromise; it is by design.

---

## 2. Versions and functions of the five environments

### 2.1 `spacetools-rl` — driver / training and inference frameworks

**Function**: runs verl (RL training framework), sglang (rollout inference engine) and the policy model; it is also the Ray driver and the host of the toolshed router. Tool calls are issued from here and results come back here.

```
Python              3.11.16          229 packages total
torch               2.9.1+cu128      torch.version.cuda = 12.8
torchvision         0.24.1+cu128
torchaudio          2.9.1+cu128
flash-attn          2.8.3.post1      built from source, all 72 cubins are sm_80
sglang              0.5.6            ← pins exactly torch==2.9.1, cannot be changed
sgl-kernel          0.3.18.post2
flashinfer-python   0.5.3
verl                0.8.0.dev0       ← SpaceTools-RL/setup.py pins torch==2.9.1
transformers        4.57.1           tokenizers 0.22.2
huggingface-hub     0.36.2           accelerate 1.14.0
numpy               1.26.4           ← verl pins <2.0.0
ray                 2.47.1
nvidia-cudnn-cu12   9.16.0.29        ← must be upgraded a second time after flash-attn
qwen-vl-utils       0.0.14           cachetools 7.1.8
timm 1.0.16 · safetensors 0.8.0 · einops 0.8.2 · peft 0.20.0 · datasets 5.0.1
triton 3.5.1 · pillow 12.3.0 · toolshed 0.1.0
nvcc 12.8 inside the env · gcc 14.3.0
```

**torch 2.9.1+cu128 cannot be replaced.** Three upstream sources pin it at the same time: `sglang 0.5.6`'s `torch==2.9.1`, `verl`'s `torch==2.9.1; extra=="sglang"`, and `SpaceTools-RL/setup.py:57`'s `"torch==2.9.1"`. The SFT checkpoint was also trained under this definition.

### 2.2 `spacetools-tool-vlm` — Molmo + SAM2 + DepthPro

**Function**: three tools share one environment.
- `vlm`: Molmo-7B-D-0924 open-vocabulary object localization
- `sam2`: SAM 2.1 segmentation from a single point, returns a boolean mask
- `depth_estimator`: DepthPro monocular depth estimation, returns depth map + focal length

```
Python              3.11.0           166 packages total
torch               2.9.1            torch.version.cuda = 12.8
torchvision         0.24.1
transformers        4.53.2           ← Molmo's trust_remote_code is only compatible with this version
tokenizers          0.21.4           huggingface-hub 0.36.2
accelerate          1.14.0
numpy               1.26.4           ← must match the driver, see §1②
ray                 2.47.1
SAM-2               1.0              depth-pro 0.1      ← both pure Python, no compiled extensions
opencv-python       5.0.0.93         timm 1.0.29
bitsandbytes        0.50.2           nvidia-cudnn-cu12 9.10.2.21
einops 0.8.2 · safetensors 0.8.0 · pillow 12.0.0 · triton 3.5.1 · toolshed 0.1.0
no nvcc inside the env (nothing needs to be compiled)
```

**Measured GPU memory**: Molmo peak **32.11 GiB** (see the fp32 issue in §7), DepthPro 7.94 GiB, SAM2 0.61 GiB.

### 2.3 `spacetools-tool-roborefer` — RoboRefer-8B

**Function**: the `roborefer` tool, spatial referring localization. All RefSpatial reasoning chains depend on it, and it also dominates in RoboSpatial. Underneath it is the NVILA/llava architecture (package name `vila`).

```
Python              3.11.16          371 packages total
torch               2.5.1            torch.version.cuda = 12.4
torchvision         0.20.1           torchaudio 2.5.1
flash-attn          2.8.3.post1      ← official prebuilt wheel cu12torch2.5cxx11abiFALSE
deepspeed           0.15.4           ← hard dependency of the inference path, not optional, see hole #16
vila                2.0.0            ← this is llava's package name
s2wrapper           0.1              ← git dependency, not on PyPI, see hole #18
transformers        4.49.0           tokenizers 0.21.0
huggingface-hub     0.28.1           accelerate 1.3.0
numpy               1.26.4
ray                 2.47.1
xformers 0.0.28.post3 · einops 0.8.1 · einops-exts 0.0.4 · timm 1.0.29
opencv-python 4.11.0.86 · peft 0.20.0 · datasets 3.2.0 · safetensors 0.5.2
triton 3.1.0 · bitsandbytes 0.45.2 · pillow 11.1.0 · qwen-vl-utils 0.0.10
nvidia-cudnn-cu12 8.9.2.26 related components · nvcc 12.4 inside the env · toolshed 0.1.0
```

**This environment is built exactly per `RoboRefer/pyproject.toml`, including `torch==2.5.1`.** Those 296 pins are a whole freeze built around torch 2.5.1 and are internally consistent; switch to any other torch version and conflicts pop up one after another (`accelerate==1.3.0`, `sympy==1.13.1`, etc.; `sympy` is a dependency of torch itself).

**Side benefit**: after dropping back to torch 2.5.1, flash-attn has a **ready-made prebuilt wheel** (cp311 + torch2.5), so no source build is needed.

**Measured GPU memory**: peak **17.16 GiB** (4 shards, loading takes about 53 seconds).

### 2.4 `spacetools-tool-bbox` — pure compute tools

**Function**: `vision_ops` (indexing into the depth map, coordinate transforms, etc.) and `bounding_box` (oriented bounding box from point cloud + mask). Pure numpy, **no GPU, no torch installed**.

```
Python              3.11.0           150 packages total
numpy               2.4.6
ray                 2.47.1
opencv-python       5.0.0.93
pillow              12.3.0
toolshed            0.1.0
no torch — by design
```

### 2.5 `spacetools-tool-graspgen` — GraspGen grasp generation

**Function**: `grasp_generator`, generates 6-DoF grasp poses from point cloud + mask, including top-down filtering and collision checking.

**This is an island environment**: CUDA toolchain, compiler and torch are all different from the other environments, because `pointnet2_ops` has to compile a CUDA extension on the spot.

```
Python              3.11.16          236 packages total
torch               2.3.1            torch.version.cuda = 12.1
torchvision         0.18.1
pointnet2-ops       3.0.0            ← compiled on the spot, cubins contain sm_80/86/89
grasp-gen           1.0.0
spconv-cu121        2.3.8            trimesh 4.5.3
transformers        4.48.3           tokenizers 0.21.4 · huggingface-hub 0.25.2
numpy               2.4.6
ray                 2.47.1
timm 1.0.15 · safetensors 0.8.0 · triton 2.3.1 · pillow 12.3.0 · toolshed 0.1.0
nvidia-cudnn-cu12 8.9.2.26
nvcc 12.1 inside the env · gcc 12.4.0    ← nvcc 12.1 does not accept Ubuntu 24.04's bundled gcc 13.3
```

**Measured GPU memory**: peak 0.46 GiB, inference 0.88 seconds. The model is small.

---

## 3. Building from scratch: how to build each environment

> If a prebuilt package already exists, go through [§4.1 Restore from HF](#41-restore-from-hf-recommended); no need to build from scratch. This section exists so that people can understand it, modify it, and redo it on a new platform.

### 3.0 Global prerequisites

```bash
CONDA_DIR=/opt/conda-st          # conda on the container disk (fast)
REPO_DIR=/opt/spacetools         # repo clones
export HF_HOME=/workspace/hf                 # HF cache on the persistent volume
export CHECKPOINT_DIR=/workspace/checkpoints # weights on the persistent volume
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"
export FLASH_ATTN_CUDA_ARCHS="80;86"         # flash-attn does not read the variable above
export MAX_JOBS=8                            # tune to the cgroup memory limit, see hole #6
```

**The first thing after installing conda is to accept the ToS** (hole #1):

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
# self-check: can we actually create an environment
conda create -n _probe python==3.11 -y && conda env remove -n _probe -y
```

**Clone four repos** (the fifth, RoboRefer, is cloned into the Toolshed directory in §3.3):

```bash
cd $REPO_DIR
git clone --depth 1 https://github.com/spacetools/SpaceTools.git
git clone --depth 1 https://github.com/ChicyChen/SpaceTools-RL.git
git clone --depth 1 https://github.com/NVlabs/SpaceTools-Toolshed.git
git clone --depth 1 https://github.com/NVlabs/GraspGen.git
```

### 3.1 `spacetools-rl`

The backbone follows `setup_rl()` in `SpaceTools/setup_envs.sh`, with six places where we must deviate.

```bash
conda create -n spacetools-rl python==3.11 -y
conda activate spacetools-rl

# ① must specify --index-url to cu128, and pin torchvision/torchaudio (holes #2 #3)
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.57.1

# ② constraints file, for this environment only (hole #4)
cat > /root/constraints-rl.txt <<EOF
torch==2.9.1
torchvision==0.24.1
torchaudio==2.9.1
transformers==4.57.1
EOF
export PIP_CONSTRAINT=/root/constraints-rl.txt

# ③ let pip resolve sglang's dependency closure normally; do not use --no-deps + a hand-written list (hole #5)
pip install "sglang[srt,openai]==0.5.6"
pip install anthropic openai blobfile decord2 torchao \
    lm-format-enforcer cuda-python sglang_router
pip install "ray[default]==2.47.1"

# ④ verl + toolshed
cd $REPO_DIR/SpaceTools-RL && pip install -e . --no-deps
pip install accelerate codetiming datasets dill hydra-core "numpy<2.0.0" pandas \
    peft "pyarrow>=19.0.0" pybind11 pylatexenc torchdata \
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0" wandb tensorboard packaging
cd $REPO_DIR/SpaceTools-Toolshed && pip install -e . --no-deps
pip install docstring_parser aiohttp aiohttp-cors Pillow fastapi uvicorn \
    python-multipart openai botocore pyyaml anthropic matplotlib scipy \
    requests click uvloop

# ⑤ flash-attn — must stop on failure (hole #7)
#    first probe for an official prebuilt wheel, otherwise build from source. cp311+torch2.9 has no wheel yet, so it must be built.
#    use the system /usr/local/cuda-12.8 (standard layout), not conda's split layout.
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=4 NVCC_THREADS=1          # 72 .o files, each nvcc peaks at 5-7 GB
pip install "flash-attn==2.8.3.post1" --no-build-isolation    # about 46 minutes

# ⑥ second cudnn upgrade — flash-attn downgrades it (hole #8)
pip install qwen_vl_utils
pip install cachetools "nvidia-cudnn-cu12==9.16.0.29"

unset PIP_CONSTRAINT     # the constraints belong to this environment only; never carry them into tool environments
```

**About flash-attn's prebuilt wheels**: Dao-AILab's release assets are published per the 4-tuple `<cuda> × <torch minor> × <cxx11abi> × <python>`, and all four fields must **match exactly** — the `.so` links directly against libtorch's C++ symbols, and every torch minor version changes the mangled names. Installing the wrong combination shows up as `ImportError: undefined symbol: _ZNK3c106...`.

```
cp311 wheels only go up to torch2.8, no torch2.9    → the rl env can only build from source
cp311 + torch2.5 has a ready-made wheel             → the roborefer env can install it directly
torch2.9 wheels were only released for cp312
```

### 3.2 The three "standard" tool environments (bbox / vlm / roborefer skeleton)

Toolshed's `setup_tool_env.sh <env name> <tool name>` does three things: `create_tool_env.sh` creates an empty environment (aligning Python + Ray) → installs toolshed → runs `tool_scripts/install_<tool>.sh`.

**It must be called with `spacetools-rl` activated**, because the versions are probed from the current environment.

```bash
conda activate spacetools-rl
cd $REPO_DIR/SpaceTools-Toolshed
source install_tools/setup_tool_env.sh spacetools-tool-bbox      bbox
source install_tools/setup_tool_env.sh spacetools-tool-vlm       vlm
source install_tools/setup_tool_env.sh spacetools-tool-roborefer roborefer
```

**The vlm environment needs three additions** (holes #9, #10):

```bash
conda activate spacetools-tool-vlm
cd $REPO_DIR/SpaceTools-Toolshed
bash install_tools/tool_scripts/install_sam2.sh      # run_eval.sh schedules sam2 in the vlm env
bash install_tools/tool_scripts/install_depth.sh     # depth too
pip install "transformers==4.53.2"                   # the two steps above bump transformers along the way; pull it back afterwards
pip install "numpy==1.26.4"                          # match the driver, see §1②
```

After downgrading `numpy` to 1.26.4, re-verify that `opencv-python` / `timm` / `bitsandbytes` still work — they might have been compiled against the numpy 2 ABI (in this build no problem was observed).

### 3.3 The llava part of `spacetools-tool-roborefer`

`setup_tool_env.sh ... roborefer` only builds the skeleton; llava (`vila`) has to be installed separately, and the upstream scripts have three holes at this step (#13 #14 #15).

```bash
conda activate spacetools-tool-roborefer
cd $REPO_DIR/SpaceTools-Toolshed

# ① clone it yourself; do not count on install_roborefer.sh reaching that step (hole #13)
git clone --depth 1 https://github.com/Zhoues/RoboRefer.git

# ② get the weights on disk first
hf download Zhoues/RoboRefer-8B-SFT --local-dir $CHECKPOINT_DIR/RoboRefer-8B-SFT
ln -s $CHECKPOINT_DIR $REPO_DIR/SpaceTools-Toolshed/checkpoints

# ③ skip env_setup.sh, install directly per pyproject (holes #14 #15)
#    no --no-deps — those 296 pins are one self-consistent set, including torch==2.5.1
cd RoboRefer && pip install -e .

# ④ flash-attn: this combination has a ready-made wheel, 2 minutes
FA=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1
pip install $FA/flash_attn-2.8.3.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

`pip install -e .` automatically brings in `deepspeed==0.15.4` (hole #16) and `s2wrapper` (git dependency, hole #18).

### 3.4 `spacetools-tool-graspgen`

Order is critical — **four preparatory steps must be done before running the upstream `install_graspgen.sh`**.

```bash
# ① first edit GraspGen's pyproject and commit locally (holes #9 #10)
#    the upstream script does `git pull` when the directory already exists, and it is set -e, so a dirty worktree blows up
cd $REPO_DIR/GraspGen
sed -i '/"pickle5",\?/d' pyproject.toml          # does not build on py3.11; the code has a try/except fallback
git -c user.email=b@l -c user.name=b commit -am "drop pickle5"

# ② scan the whole repo and turn the hard-coded arch list into setdefault (hole #12)
#    two files: pointnet2_ops/pointnet2_ops/pointnet2_utils.py
#              grasp_gen/models/pointnet/pointnet2_utils.py
grep -rl 'os.environ\["TORCH_CUDA_ARCH_LIST"\]' . --include=*.py | while read f; do
  sed -i 's|os\.environ\["TORCH_CUDA_ARCH_LIST"\][[:space:]]*=[[:space:]]*\(".*"\)|os.environ.setdefault("TORCH_CUDA_ARCH_LIST", \1)|' "$f"
done
git commit -am "TORCH_CUDA_ARCH_LIST setdefault"

# ③ create the empty env (versions probed from rl), then install CUDA 12.1 **before upstream gets to it** (hole #11)
conda activate spacetools-rl
cd $REPO_DIR/SpaceTools-Toolshed
source install_tools/create_tool_env.sh spacetools-tool-graspgen
conda activate spacetools-tool-graspgen
printf 'cuda-version 12.1.*\n' > $CONDA_PREFIX/conda-meta/pinned
conda install -y -c nvidia/label/cuda-12.1.1 --override-channels \
      "cuda-toolkit=12.1.1" "cuda-nvcc=12.1"
#    criterion: nvcc must be 12.1
nvcc --version | grep "release 12.1" || echo "not pinned"

# ④ gcc 12 (hole #17)
conda install -y -c conda-forge "gcc_linux-64=12" "gxx_linux-64=12"
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export NVCC_PREPEND_FLAGS="-ccbin $CC"

# ⑤ only now run the upstream script. Feed the two interactive reads via environment variables (hole #10)
pip install -e .                                  # toolshed
export GRASPGEN_DIR=$REPO_DIR MODELS_DIR=$CHECKPOINT_DIR
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" MAX_JOBS=4
bash install_tools/tool_scripts/install_graspgen.sh
```

**Why CUDA must be installed "ahead of time"**: upstream `install_graspgen.sh:124` runs `conda install -c nvidia cuda-toolkit=12.1 -y`, and `cuda-toolkit 12.1` in the `nvidia` channel is an **empty metapackage that does not constrain `cuda-version`**, so the components resolve to 13.3.x (hole #11). Worse, conda's "already satisfied" short-circuit **compares only package name and version, not channel**, so adding a `conda install -c nvidia/label/cuda-12.1.1 ...` **after** it just returns "All requested packages already installed" and does nothing. The only effective approach is to make that line a real no-op.

---

## 4. How to run

### 4.1 Restore from HF (recommended)

```bash
hf download qzpm55555/spacetools-eval-env --local-dir ./pkg
cd pkg
cat spacetools-envs-*.tar.zst.part* > spacetools-envs-20260910-0959.tar.zst

bash FETCH_WEIGHTS.sh    # pull 79G of weights at the pinned revisions
bash RESTORE.sh          # unpack to /opt, with built-in pre-checks
bash VERIFY.sh           # three real acceptance checks; only touch eval when all pass
```

**Three things must be confirmed before restoring** (`RESTORE.sh` checks the first two):

| | |
|---|---|
| Restore path | **must be `/opt/conda-st` and `/opt/spacetools`** — absolute paths are baked into the conda environments |
| GPU architecture | **A100 (sm_80) / A6000 (sm_86) / L40S (sm_89) work**; H100 (sm_90) does not — built without `+PTX`, so there is no intermediate code that can JIT to Hopper |
| glibc | **must not be older than Ubuntu 24.04 (glibc 2.39)**. `flash_attn_2_cuda` and `pointnet2_ops/_ext` link against the system glibc, which is not in the conda environment. 22.04 (glibc 2.35) reports `GLIBC_2.38 not found` |
| Driver | ≥ 550 (torch is cu128; this build used 580.159.04) |

All weight versions are pinned in `WEIGHTS_PINS.txt`:

```
allenai/Molmo-7B-D-0924             cab33fb7f1a40091911f81165f8481920621948f   30 G
Zhoues/RoboRefer-8B-SFT             bd04070786084c624156194d89333c375b274b28   16 G
Qwen/Qwen2.5-VL-3B-Instruct         66285546d2b821cf421d4f5eb2576359d3770cd3  7.1 G
facebook/sam2.1-hiera-small         ee5bba1d82bb8749febdf90f45e84b687142ba03  180 M
siyich/spacetools-eval-benchmarks   1d539ac935872c7aa712c85a77bf4b0cb469c8e8  283 M   (dataset)
adithyamurali/GraspGenModels        commit ec1ccbb5eec0680db669246ac312a3636f16ee43  7.9 G  (git-lfs)
depth_pro.pt                        sha256 3eb35ca6…85c0ce · 1904446787 bytes   (Apple CDN)
```

`facebook/sam2.1-hiera-small` is not in any documented list; `install_sam2.sh` pulls it itself at runtime — on an offline machine, if it is not prepared in advance, sam2 only fails **on its first call**.

### 4.2 Running eval

```bash
cd $REPO_DIR/SpaceTools-RL/examples/toolshed
bash run_eval.sh <ckpt path> robospatial reflocation refplacement refunseen
```

What it does: `ray start --head` → start the router and each tool actor with `start_toolkit(TOOL_CONFIGS)` → start the policy model → run the benchmarks.

**GPU count requirement** (tool reservations in `run_eval.sh:132-142`):

```
roborefer        1 actor × 0.6 = 0.6
vlm              1 actor × 0.6 = 0.6
sam2             2 actors × 0.2 = 0.4
depth_estimator  2 actors × 0.2 = 0.4
bounding_box     2 actors × 0.1 = 0.2
vision_ops       1 actor × 0    = 0
grasp_generator  1 actor × 0.1 = 0.1
                             ─────
                             2.3 GPUs  + the policy model's own share → at least 3
```

Ray's `num_gpus` is a **logical reservation**: one GPU can only satisfy requests totaling 1.0, and the remaining actors **queue forever waiting for resources, with no error**. This is the hardest kind of failure to track down.

**GPU memory must also suffice**: fractional `num_gpus` does not partition GPU memory; actors on the same GPU share all of it. Measured resident peaks of Molmo 32.11 GiB + RoboRefer 17.16 GiB already exceed a single A6000's 48 GB.

### 4.3 Running RL

`run_rl.sh` has far more actors (6 roborefer; 5 each of sam2/depth/bbox/graspgen; 8 vision_ops), so the GPU requirement is correspondingly higher. Between SFT → RL, first use the eval in §4.2 to confirm the checkpoint is usable.

---

## 5. The 20 upstream holes

Ordered by build stage. Each one gives location, mechanism, and countermeasure.

### Stage A: `spacetools-rl`

**#1 The ToS in conda ≥26 silently blocks environment creation**
On new conda versions, when the Terms of Service have not been accepted, `conda create` fails outright (`CondaToSNonInteractiveError`), **and only reports it on stderr**. `setup_envs.sh` does not check the return value, so all subsequent `pip install`s go into the base environment, and it only blows up when compiling flash-attn looks for `$CONDA_PREFIX/nvcc` — by then more than ten minutes have passed.
→ Run `conda tos accept` before creating environments, and self-check with one `conda create` + `conda env remove`.

**#2 The torch install has no `--index-url`**
`setup_envs.sh:41` runs `pip install torch==$TORCH_VERSION torchvision torchaudio`, which takes the PyPI default wheel. When the paper was written the default for torch 2.9.1 was cu128, but the PyPI default has since become cu130, so compiling flash-attn with nvcc 12.8 reports `detected CUDA 12.8 mismatches the version used to compile PyTorch 13.0`.
→ Explicit `--index-url https://download.pytorch.org/whl/cu128`.

**#3 `torchvision` / `torchaudio` are not version-pinned**
The same line pins only torch. pip resolves to a torchaudio compiled for **a different torch version**, and `import` gives `undefined symbol`.
→ Pin all three together.

**#4 `flashinfer_python` is not version-pinned and silently replaces torch**
The latest `flashinfer_python`, installed by hand upstream, requires `torch > 2.9.1`; pip **silently** upgrades torch together with the whole `nvidia-*-cu13` set to 2.14.0+cu130. The whole environment's definition quietly changes, while every command returns 0.
→ Write a `PIP_CONSTRAINT` for this environment, pinning torch/torchvision/torchaudio/transformers. A conflict then becomes an explicit pip error instead of an environment whose definition silently changed. **The constraints file is per-environment and must not be exported globally** — each tool environment has its own torch, and that is by design.

**#5 sglang uses `--no-deps` + a hand-written dependency list, and the list is incomplete**
`setup_envs.sh` first runs `pip install "sglang[srt,openai]==0.5.6" --no-deps`, then lists a long string of dependencies by hand. That list is missing packages (at runtime `ModuleNotFoundError: No module named 'pybase64'`), and the hand-installed `flashinfer_python` pushes `nvidia-cutlass-dsl` up to 4.8.0.dev0, conflicting with the `==4.2.1` required by sglang 0.5.6.
→ Once `PIP_CONSTRAINT` guards torch, let pip resolve sglang's real dependency closure normally; sglang decides the versions itself, which is more accurate than a hand-written list.

**#6 The two extras in `sglang[srt,openai]` do not exist in 0.5.6**
Only a warning, no functional impact, but it shows this line was copied from a different version.

**#7 flash-attn build failures are swallowed**
`setup_envs.sh` runs `pip install flash-attn --no-build-isolation 2>/dev/null || echo "WARNING: ..."`. The error is redirected away and the script continues on failure. The result is an environment without flash-attn, while the script shows green all the way.
→ Make it stop on failure. In addition, the build itself has two pitfalls:
- **CUDA header path**: conda's cuda-toolkit uses a **split layout**, with headers in `$CONDA_PREFIX/targets/x86_64-linux/include/`, while torch's `cpp_extension` only looks in `$CUDA_HOME/include`. `nvcc --version` looks perfectly fine; what blows up is the host compiler compiling `flash_api.cpp`, reporting `fatal error: cuda_runtime_api.h: No such file or directory`. Using the system `/usr/local/cuda-12.8` (standard layout) avoids it.
- **Memory**: `free -g` shows the host's memory (503 GB); the container's real limit is in `/sys/fs/cgroup/memory/memory.limit_in_bytes` (46.6 GiB this time). Each nvcc for flash-attn's backward kernels peaks at 5–7 GB; a large `MAX_JOBS` gets killed by the OOM killer, and the log **only says `Killed`, with no compile error at all**. `MAX_JOBS=4` + `NVCC_THREADS=1` is safe under 46.6 GiB.

**#8 The second cudnn upgrade is only in the docs, not in the script**
Installing flash-attn downgrades `nvidia-cudnn-cu12`. The manual path in `docs/SETUP.md` has a second upgrade to 9.16.0.29, but `setup_envs.sh` misses this step.

### Stage B: common mechanisms of the tool environments

**#9 `create_tool_env.sh` is interactive when the environment already exists**
`create_tool_env.sh:58-65`: when the environment already exists, `read -r response` asks `continue (c) / recreate (r) / abort (a)`, and **empty input defaults to `c`**. In non-interactive runs stdin reads empty, so it always takes "keep using the existing environment" — **you cannot count on it to rebuild**; you must `conda env remove` it yourself first.

**#10 The two interactive `read`s in `install_graspgen.sh`**
`install_graspgen.sh:37` and `:52` ask for `GRASPGEN_DIR` and `MODELS_DIR` respectively. Non-interactively, EOF + `set -e` exits silently, with the log stopping at the `Enter directory ...` line.
→ Both can be skipped by presetting environment variables of the same name.

**#11 `cuda-toolkit=12.1` does not pin the CUDA version, and the remedy does not work**
`install_graspgen.sh:124`'s `conda install -c nvidia cuda-toolkit=12.1 -y`: `cuda-toolkit 12.1` in the `nvidia` channel is a 1 KB **empty metapackage with loose dependencies on its components**, and it resolves to `cuda-nvcc 13.3.73` / `cuda-version 13.3`. Compiling `pointnet2_ops` then reports `RuntimeError: The detected CUDA version (13.3) mismatches PyTorch (12.1)`.

**The real version gate is the `cuda-version` metapackage, not `cuda-toolkit`.** And the old-style label channel `nvidia/label/cuda-12.1.1` **does not provide `cuda-version` at all** (`conda search` returns No match), so the spec must not include it.

Worse: **conda's "already satisfied" short-circuit compares only package name and version, not channel**. Adding a `conda install -c nvidia/label/cuda-12.1.1 --override-channels cuda-toolkit` after the upstream line just returns `All requested packages already installed`; `--override-channels` has no effect at all.
→ The only effective approach: **before the upstream line**, install `cuda-toolkit=12.1.1` + `cuda-nvcc=12.1` from the label channel so that its line becomes a real no-op; then write `$CONDA_PREFIX/conda-meta/pinned` to pin `cuda-version 12.1.*` as a second safeguard.

**#12 `pointnet2_ops` hard-overrides the build architectures at import time**
`pointnet2_utils.py:23`:

```python
os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
```

This is **an assignment, not setdefault**, executed at import time, so the externally set `8.0;8.6;8.9` has no effect at all. CUDA 12 no longer supports sm_37, hence `nvcc fatal: Unsupported gpu architecture 'compute_37'`.

**This line appears twice in the GraspGen repo**: `pointnet2_ops/pointnet2_ops/pointnet2_utils.py` and `grasp_gen/models/pointnet/pointnet2_utils.py`. If only one is changed, whether it works depends on import order.
→ Scan every `.py` in the repo and change the assignment to `setdefault` (use the external value if one is set, otherwise its default; this does not change upstream's behavior "when nobody sets it").

**#13 `install_roborefer.sh` puts the clone after a download that can `exit 1`**
`install_roborefer.sh` first runs `hf download` for 16 GB of weights and does `exit 1` on failure; **cloning the RoboRefer repo (the only source of llava) comes after it**. If the download is interrupted, llava never gets installed, while Toolshed README line 100 explicitly says RoboRefer "requires cloning a fork of RoboRefer".
→ Clone it yourself first.

**#14 `RoboRefer/env_setup.sh` hard-codes a flash-attn wheel that cannot be installed**
It hard-codes `flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`, which on the py3.11 environment built by Toolshed reports `is not a supported wheel on this platform`. And this step comes **before** `pip install -e ".[train,eval]"` (**the only step that installs llava**); once `set -e` stops, llava is not installed.
→ Skip `env_setup.sh` and run `pip install -e .` directly. Its steps 4–6 (`triton==3.1.0`, `deepspeed_replace`, `protobuf`) are for the training path and are not needed for pure inference.

**#15 GraspGen depends on `pickle5`, which does not build on Python 3.11**
`pickle5` is a backport of protocol 5 for Python 3.5–3.7; the standard library has it from 3.8 on. Building on 3.11 reports `error: lvalue required as left operand of assignment` (`Py_SIZE` is no longer an lvalue), and **the whole pip transaction fails**, so neither torch 2.3.1 nor `grasp_gen` gets installed.

Its only use in GraspGen is `import pickle5 as pickle` at `grasp_gen/dataset/dataset.py:27`, and it is **wrapped in `try/except ImportError`** — removing the dependency automatically falls back to the standard-library pickle.
→ Remove it from `pyproject.toml` **before** creating the environment, and `git commit` locally (because the upstream script does `git pull` when the directory already exists, and a dirty worktree fails). Note that the upstream script itself also edits this file with `sed -i` and **verifies that the sed took effect, exiting 1 if not**, so do not touch its anchor.

**#16 nvcc 12.1 does not accept Ubuntu 24.04's gcc 13.3**
`crt/host_config.h:132: #error -- unsupported GNU version! gcc versions later than 12 are not supported!`
→ Install `gcc_linux-64=12` / `gxx_linux-64=12` from conda-forge and point to them with `NVCC_PREPEND_FLAGS="-ccbin ..."`. Do not force past it with `-allow-unsupported-compiler` — that flag's own warning says it "may lead to incorrect runtime behavior".

### Stage C: RoboRefer / llava

**#17 The inference path hard-depends on the `train` extra**

```
llava/model/language_model/llava_llama.py:28
    from llava.train.sequence_parallel.globals import get_pg_manager
llava/train/sequence_parallel/globals.py:22
    import deepspeed.comm as dist          ← module level, unconditional
```

But `RoboRefer/pyproject.toml:343` puts deepspeed under `train = ["deepspeed==0.15.4", "ninja", "wandb"]`, and the `eval` extra does not have it. **Anyone who installs only the eval extra cannot even import the model definition.**

**#18 `flash_attn` is also a hard dependency for loading the model**

```
llava/model/multimodal_encoder/intern/flash_attention.py:23,25
```

Both import paths are in the single `flash_attn` package (the `except` only bridges old and new API names), and it is **on the loading path of the InternViT vision tower**: `llava.load → model/__init__ → llava_llama → llava_arch → multimodal_encoder.builder → intern_encoder → modeling_intern_vit → flash_attention`. There is no way around it.

**#19 `s2wrapper` is not on PyPI**
`pyproject.toml:48` says `"s2wrapper@git+https://github.com/bfshi/scaling_on_scales"`. `pip install s2wrapper` by module name is bound to give `No matching distribution found`.
→ A direct `pip install -e .` (with dependencies) handles it automatically; when installing dependencies by hand, use the git URL.

**#20 `llava/data/datasets_mixture.py:60` is a real syntax error**

```python
2D_choice_qa = Dataset(
^
SyntaxError: invalid decimal literal
```

`2D_choice_qa` is not a valid Python identifier (cannot start with a digit), so this file **cannot be imported under any Python version**. It is only used on the training path, so upstream never noticed. It drags down 6 modules: `llava.data`, `llava.eval.egoschema`, `llava.eval.eventbench`, `llava.eval.model_vqa_video`, `llava.eval.rtl`, `llava.eval.vnbench`. Pure inference is unaffected, but it shows that whole blocks of code in the RoboRefer repo have never been executed.

### Stage D: cross-environment integration (the most hidden group)

**#21 Four dependency declarations contradict each other on numpy**

```
SpaceTools-RL/setup.py:32          numpy<2.0.0     ← driver must be numpy 1.x
toolshed/pyproject.toml:27         numpy>=2.0.0
tool-vlm.txt:22                    numpy>2.0.0
tool-sam2.txt:27                   numpy>=2.0.0
tool-depth.txt:4                   # IMPORTANT: This tool requires numpy<2.0.0
                                   #   which conflicts with the base toolshed
tool-depth.txt:27                  # numpy<2.0.0        ← the pin is commented out
```

One `numpy` wants `>2.0` and another wants `==1.26.4`; they cannot both be satisfied. And `setup_tool_env.sh` only installs the former.

**Consequence**: when a numpy 2.x actor returns a `ToolResult` carrying an `ndarray` to the numpy 1.x driver, deserialization reports `No module named 'numpy._core.numeric'`. `numpy._core` is numpy 2's internal module path, which does not exist in numpy 1.x.

Measured by shape (numpy 2.x actor → numpy 1.x driver):

```
bare ndarray          FAIL      ← Ray's zero-copy channel does not save it
dict{ndarray}         FAIL
tuple(ndarray)        FAIL
list[float]           OK
np.float32 scalar     OK
ToolResult(plain text/scalar) OK
reverse numpy1 → numpy2  OK      ← numpy 2 has a compatibility shim
```

Specifically, of the seven tools, **only the `ToolResult`s of `depth_estimator` and `sam2` carry `ndarray`** (`$depth_map`, `$segmentation_mask`), and both are in the `tool-vlm` environment. `bounding_box` and `grasp_generator` call `.tolist()` on every array in their return values (`grasp_generator.py:504,548`), so they are plain Python lists and unaffected.
→ **`tool-vlm` must use numpy 1.26.4**. Keeping `bbox` and `graspgen` at 2.4.6 is safe (and they should not be touched — `pointnet2_ops` in `graspgen` was compiled under numpy 2).

**#22 toolshed tools "return an error string" instead of "raising an exception"**
When a tool errors, it does not raise; it wraps the error message into a normal `ToolResult` and returns it; the router also swallows exceptions inside actors and wraps them into normal returns.

```
CALL_OK   depth_estimator -> ModuleNotFoundError: No module named 'numpy._core.numeric'
                             return type: ToolResult
```

Note that it is `CALL_OK`, not `CALL_FAIL`. **The caller gets no exception at all.**

This means that if the numpy issue is not fixed, eval **runs to completion without errors**, every `depth_estimator` call returns an error message as the depth map, the model keeps reasoning on garbage input, and finally produces a **normal-looking score** — and then people go suspect the checkpoint.
→ Every acceptance check must look for `Error:` / `ERROR:toolshed` in the body of the `ToolResult`, not just "did it crash".

**#23 `run_eval.sh` schedules sam2 and depth in the vlm environment, but the install script does not install them**
In `run_eval.sh:136,137` the `conda_env` of both `'sam2'` and `'depth_estimator'` is `spacetools-tool-vlm`, while `setup_tool_env.sh <env> vlm` only installs `tool-vlm.txt` (Molmo's dependencies). Following the docs gives an environment that cannot run sam2/depth, and the error only appears **when the actor is called**.
→ Additionally run `install_sam2.sh` and `install_depth.sh` in the vlm environment.

**#24 `vlm.py:116` hard-codes `torch_dtype="auto"`, swallowing the dtype passed by the caller**
The `run_eval.sh` config explicitly passes `'dtype': 'float16'`, but the code hard-codes `auto`, and Molmo actually loads in **fp32**. Measured peak **32.11 GiB** (fp16 should be about 17 GB). Two different machines and two independent builds gave the same number.
→ A 48 GB single GPU can take it, but it directly determines how many GPUs to rent. Fixing this one line halves Molmo's footprint.

**#25 Tool argument conventions are inconsistent, even within the same file**

```
sam2.segment_from_point(image, x, y)       x,y are **normalized** [0,1]
vision_ops.index_at(data, u, v)            u,v are **normalized** [0,1]
grasp_generator.compute_grasp(...)
    _resolve_pointcloud  accepts numpy        (grasp_generator.py:71)
    _resolve_mask        accepts numpy        (grasp_generator.py:89)
    _resolve_image       **accepts only str or PIL.Image**, not numpy   (grasp_generator.py:80)
image argument of sam2 / depth_estimator   accepts numpy
```

Within the same file the three `_resolve_*` treat numpy inconsistently, and the `image` argument convention differs across tools. **When wiring up tools, read the source for every signature; do not assume by analogy** — and passing the wrong thing shows up as the "returns an error string but the call succeeds" behavior described in #22.

---

## 6. Acceptance criteria

The failure modes of this environment are heavily concentrated in "**looks like it succeeded**". Each of the following criteria is designed against a specific failure mode.

### 6.1 Three-layer acceptance check

`VERIFY.sh` runs three items; it is only usable if all pass:

| Layer | Script | What it checks |
|---|---|---|
| 1 | `03_verify.sh` | key module imports in all five envs + architecture scan of the cubins in every `.so` + `pip freeze` snapshot |
| 2 | `04_smoke.sh` | each of the seven tools actually loads its weights and actually produces a result |
| 3 | `28_chain.sh` | Ray cross-conda-env tool chain: ndarray passed in both directions |

Results of this build:

```
five-env import gate              all pass
architecture scan                 4619 .so files, 108 contain cubins, self-built extensions missing sm_8x: 0
seven-tool smoke test             7/7
Ray cross-env actors              5/5
cross-env variable passing        ndarray correct in both directions
```

### 6.2 Specific criteria and the failure modes they target

**`import` checks**

- To check whether `flash_attn` is installed, look at **whether `flash_attn.__version__` can be read** + **whether `import flash_attn_2_cuda` works**; a bare `import flash_attn` is not enough. A failed build leaves behind a **`flash_attn/` directory without `__init__.py`** in `site-packages`; Python 3 treats it as a namespace package, so `import` succeeds while `__file__` is `None`.
- Before `import flash_attn_2_cuda` you **must first `import torch`**. This extension links libtorch, and a bare import gives `ImportError: libc10.so: cannot open shared object file`.
- Before verifying `llava`, **`cd /` first**. The RoboRefer repo root has a `llava/` package directory, and importing there hits the source tree rather than the installed package. Also print `llava.__file__` to confirm the source.
- `import llava` is too shallow — `llava/__init__.py` is thin, and the real dependencies are only imported layer by layer at `llava.load()`. The real criterion is **constructing `RoboreferTool`** (loading 16 GB of weights).

**Architecture checks**

- Use `cuobjdump --list-elf <so>` to check whether the cubins contain `sm_8x`.
- **First assert that `cuobjdump` exists** (tool environments do not necessarily have cuda-toolkit installed), and **print how many files were actually scanned**. If 0 were scanned, explicitly report "no conclusion", counting as neither pass nor fail.
- Distinguish **self-built extensions** from **vendor prebuilt libraries**. The architectures of `libcufft` / `libnppc` / `libcublas` / `libcusolver` / `libaccinj64` / `torch/lib/libtorch_cuda_linalg.so` / `sgl_kernel/flashmla_ops` (Hopper only) are decided by NVIDIA and should not count as failures; the ones to watch are `flash_attn_2_cuda` and `pointnet2_ops/_ext`.

**Tool-call checks**

- The criterion is **`SMOKE_OK` was printed, and the result body contains no `-> Error:` or `ERROR:toolshed`** (targets #22).
- Check that `hf_device_map` has no `cpu` / `disk` — accelerate silently offloads when GPU memory is insufficient, and such results do not count.
- Print `torch.cuda.max_memory_allocated()`. Peak GPU memory is circumstantial evidence of "whether the model actually ran": loading without computing versus actually running once differ by an order of magnitude.

**Cross-environment checks**

- Verifying that "the actor starts" is not enough; verify **whether a `ToolResult` carrying an `ndarray` can cross environments back to the driver**, and go through the **real router path** (`start_toolkit` + `ToolkitClient`), not a direct `ray.get()`.

**General**

- After every step, verify the **actual state**, not the return value. The upstream scripts make heavy use of `2>/dev/null || echo WARNING` and unchecked `set -e`; exit codes cannot be trusted.
- Every self-check must **print how many objects it actually checked**. A self-check that does not print a count cannot distinguish "all passed" from "checked nothing".
- Which interpreter the acceptance code itself runs in, and whether it has the dependencies it needs, **is part of the criterion**. A broken acceptance tool and a broken object under test look exactly the same.

### 6.3 Reading the eval

After running `run_eval.sh`:

1. **First `grep -c OutOfMemoryError`**, then look at the scores. OOM samples count in the denominator.
2. Criterion: RoboSpatial ≥ 60 **and** RefSpatial ≥ 48 → can proceed to RL; < 55 or < 40 → stop and investigate.
3. **Do not use `boppose` as a criterion** — its metric mapping has not been sorted out yet.

---

## 7. Known limitations and open items

### 7.1 Hardware compatibility range

| | |
|---|---|
| ✅ A100 (sm_80) / A6000 (sm_86) / L40S (sm_89) | cubins cover these three architectures |
| ❌ H100 / H200 (sm_90) | at build time `TORCH_CUDA_ARCH_LIST` only went up to 8.9 and **had no `+PTX`**, so there is no intermediate code that can JIT to Hopper. Going to Hopper requires rebuilding `flash-attn` and `pointnet2_ops` |
| ⚠️ glibc | must not be older than Ubuntu 24.04 (2.39) |
| ⚠️ Driver | ≥ 550 |
| ⚠️ Restore path | must be `/opt/conda-st` + `/opt/spacetools` |

### 7.2 Unhandled items

**① Molmo's fp32 issue (hole #24)**
`torch_dtype="auto"` at `vlm.py:116` is not fixed. Molmo resides at 32.11 GiB, which directly affects the GPU budget. Fixing this line is the key step toward "renting fewer GPUs".

**② The minimal tool set for eval has not been trimmed**
`run_eval.sh` unconditionally starts all 7 tools (a reservation of 2.3 GPUs). Yet the four keys robospatial / reflocation / refplacement / refunseen do not use `grasp_generator`, and the weight of `vlm` and `roborefer` varies a lot across benchmarks. Trimming the tool set per benchmark can lower the GPU requirement.

**③ The policy model checkpoint is not version-pinned**
`MY_SFT_REPO` in `FETCH_WEIGHTS.sh` is empty. `main` on HF can be overwritten at any time; it is recommended to pin the commit SHA as well:

```bash
MY_SFT_REPO=<username>/<repo name> MY_SFT_REV=<commit> bash FETCH_WEIGHTS.sh
```

**④ Synthetic-scene test of `graspgen`**
When testing `compute_grasp` with a random point cloud, the network forward pass completes normally (measured: 1704 object points → 200 grasps generated → 26 left after top-down filtering, confidence [0.678, 0.926], inference 0.88 seconds), but collision filtering removes all of them — because in the synthetic scene the "scene point cloud" includes the object's own points, so the gripper necessarily intersects them. This is a property of the test scene, not an environment problem. It does not happen on real data.

### 7.3 Package contents

```
https://huggingface.co/qzpm55555/spacetools-eval-env   (private)

spacetools-envs-*.tar.zst.part00..05        22.10 GB   five conda environments + four repos (with .git)
spacetools-scripts-*.tar.zst                  81 KB    build/fix/acceptance scripts + constraints + freeze
build-logs/build-logs-20260910.tar.zst        85 KB    76 build logs + seven-tool smoke test output
build-logs/README_LOGS.md                              log index
MANIFEST.txt                                           packaging machine / repo commits / key versions
WEIGHTS_PINS.txt                                       weight revision pin table
RESTORE.sh / VERIFY.sh / FETCH_WEIGHTS.sh              the restore trio
README.md / SHA256SUMS.remote
```

**Repo commits**

```
SpaceTools           17d585539b6cc32f2b3c068ea2591762b4583fd9
SpaceTools-RL        54270e82443d3d2a4c2a737c2d3b33314a991fcc
SpaceTools-Toolshed  4f0512d092f53abc1e6c5c934245bf83a211466a
GraspGen             9b3cfc1e5b664698e047ddd482832f6e7796380c   (including the two local commits)
RoboRefer            d97a995ad28376720a4c8beb64915c58ed16c844
```

GraspGen's two local commits are the change records for hole #15 (drop pickle5) and hole #12 (`TORCH_CUDA_ARCH_LIST` to setdefault), viewable with `git log`.
