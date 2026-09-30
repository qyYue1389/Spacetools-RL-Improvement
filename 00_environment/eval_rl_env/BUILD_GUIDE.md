# SpaceTools Eval 环境 —— 构建与使用说明

构建日期 2026-09-10 · RTX A6000 48GB · Ubuntu 24.04.3 · 驱动 580.159.04 · 系统 CUDA 12.8

包地址：`https://huggingface.co/qzpm55555/spacetools-eval-env`（private）

> **以 GitHub 源码库为准，不以任何 fork 为准。**
> `spacetools/SpaceTools` · `ChicyChen/SpaceTools-RL` · `NVlabs/SpaceTools-Toolshed` · `NVlabs/GraspGen` · `Zhoues/RoboRefer`

---

## 目录

1. [整体架构：为什么是五个环境](#1-整体架构为什么是五个环境)
2. [五个环境的版本与功能](#2-五个环境的版本与功能)
3. [从零搭建：每个环境怎么搭](#3-从零搭建每个环境怎么搭)
4. [怎么跑](#4-怎么跑)
5. [上游的 20 个洞](#5-上游的-20-个洞)
6. [验收判据](#6-验收判据)
7. [已知限制与未决事项](#7-已知限制与未决事项)

---

## 1. 整体架构：为什么是五个环境

SpaceTools 是一个**工具增强的空间推理**系统。策略模型（Qwen2.5-VL-3B）不直接回答问题，而是写出调用工具的代码，工具返回结果，模型据此继续推理。可用工具 11 个，背后是 7 个真实实现。

这些工具的依赖**互相冲突**，没法装进一个环境：

| 工具 | 冲突点 |
|---|---|
| GraspGen | 要 `torch>=2.3,<2.4` + CUDA 12.1 工具链（要现场编 CUDA 扩展） |
| RoboRefer (NVILA/llava) | 要 `torch==2.5.1` + `transformers==4.49.0` + `numpy==1.26.4` |
| Molmo (VLM) | 要 `transformers==4.53.2`（走 `trust_remote_code`，远程代码只兼容特定版本） |
| verl + sglang（训练/推理框架） | `sglang 0.5.6` 精确钉 `torch==2.9.1`；verl 钉 `numpy<2.0.0` |

**Toolshed 的解法**：每个工具跑在自己的 conda 环境里，由 Ray 用 `runtime_env={"conda": <环境名>}` 起 actor，中间架一个 router actor 做路由和负载均衡。

```
                    ┌─────────────────────────────────────────┐
                    │  driver / agent   (spacetools-rl)       │
                    │  verl + sglang + 策略模型                │
                    │  numpy 1.26.4                           │
                    └───────────────┬─────────────────────────┘
                                    │ ToolkitClient.call_tool()
                    ┌───────────────▼─────────────────────────┐
                    │  ToolRouterActor  (toolshed)            │
                    │  路由 / 负载均衡 / 超时                   │
                    └──┬────────┬────────┬────────┬───────────┘
       runtime_env:conda│        │        │        │
        ┌───────────────▼┐ ┌─────▼──────┐ ┌▼──────────────┐ ┌▼───────────────┐
        │ tool-vlm       │ │ tool-      │ │ tool-bbox     │ │ tool-graspgen  │
        │ torch 2.9.1    │ │ roborefer  │ │ 无 torch       │ │ torch 2.3.1    │
        │ cu128          │ │ torch2.5.1 │ │ numpy 2.4.6   │ │ cu121          │
        │ numpy 1.26.4   │ │ cu124      │ │               │ │ numpy 2.4.6    │
        │                │ │ numpy1.26.4│ │               │ │                │
        │ Molmo          │ │ RoboRefer  │ │ vision_ops    │ │ grasp_generator│
        │ sam2           │ │ 8B         │ │ bounding_box  │ │                │
        │ depth_estimator│ │            │ │               │ │                │
        └────────────────┘ └────────────┘ └───────────────┘ └────────────────┘
```

**两条贯穿全局的硬约束：**

**① `ray` 五个环境必须完全一致（2.47.1）。** 跨环境起 actor 时 Ray 要做版本握手，不一致直接失败。`create_tool_env.sh` 就是从当前激活环境探测 Python 和 Ray 版本来对齐的——所以建工具环境前必须先激活 `spacetools-rl`。

**② numpy 的大版本必须与 driver 兼容。** 工具的返回值（`ToolResult`）里可能带 `ndarray`（深度图、分割掩码），要跨进程回到 driver。numpy 2.x 序列化出的数组引用 `numpy._core`，**numpy 1.x 反序列化不了**；反方向（1.x → 2.x）因为 numpy 2 带兼容垫片是通的。

driver 端 `verl` 钉死 `numpy<2.0.0`，所以：

| 环境 | 返回值是否带 ndarray | numpy 要求 |
|---|---|---|
| `tool-vlm` | **是**（`$depth_map`、`$segmentation_mask`） | **必须 1.26.4** |
| `tool-roborefer` | 否（只返回坐标文本） | 1.26.4（RoboRefer 自己钉的） |
| `tool-bbox` | 否（`.tolist()` 后的 float 列表） | 2.4.6 可以 |
| `tool-graspgen` | 否（`grasp_pose` 已 `.tolist()`） | 2.4.6 可以 |

**四个 torch、四套 CUDA 运行时（12.8 / 12.8 / 12.4 / 12.1）并存且互不干扰**，因为每个 conda 环境自带自己那套 `nvidia-*` 轮子。这不是将就，是设计如此。

---

## 2. 五个环境的版本与功能

### 2.1 `spacetools-rl` —— driver / 训练与推理框架

**功能**：跑 verl（RL 训练框架）、sglang（rollout 推理引擎）、策略模型；同时是 Ray 的 driver 和 toolshed router 的宿主。工具调用从这里发出，结果回到这里。

```
Python              3.11.16          共 229 个包
torch               2.9.1+cu128      torch.version.cuda = 12.8
torchvision         0.24.1+cu128
torchaudio          2.9.1+cu128
flash-attn          2.8.3.post1      源码编译,72 个 cubin 全是 sm_80
sglang              0.5.6            ← 精确钉 torch==2.9.1,不可动
sgl-kernel          0.3.18.post2
flashinfer-python   0.5.3
verl                0.8.0.dev0       ← SpaceTools-RL/setup.py 钉 torch==2.9.1
transformers        4.57.1           tokenizers 0.22.2
huggingface-hub     0.36.2           accelerate 1.14.0
numpy               1.26.4           ← verl 钉 <2.0.0
ray                 2.47.1
nvidia-cudnn-cu12   9.16.0.29        ← 必须在 flash-attn 之后二次升级
qwen-vl-utils       0.0.14           cachetools 7.1.8
timm 1.0.16 · safetensors 0.8.0 · einops 0.8.2 · peft 0.20.0 · datasets 5.0.1
triton 3.5.1 · pillow 12.3.0 · toolshed 0.1.0
环境内 nvcc 12.8 · gcc 14.3.0
```

**torch 2.9.1+cu128 不可替换。** 三个上游来源同时钉死它：`sglang 0.5.6` 的 `torch==2.9.1`、`verl` 的 `torch==2.9.1; extra=="sglang"`、`SpaceTools-RL/setup.py:57` 的 `"torch==2.9.1"`。SFT checkpoint 也是这个口径训出来的。

### 2.2 `spacetools-tool-vlm` —— Molmo + SAM2 + DepthPro

**功能**：三个工具共用一个环境。
- `vlm`：Molmo-7B-D-0924 开放词汇目标定位
- `sam2`：SAM 2.1 从一个点做分割，返回布尔掩码
- `depth_estimator`：DepthPro 单目深度估计，返回深度图 + 焦距

```
Python              3.11.0           共 166 个包
torch               2.9.1            torch.version.cuda = 12.8
torchvision         0.24.1
transformers        4.53.2           ← Molmo 的 trust_remote_code 只兼容这个版本
tokenizers          0.21.4           huggingface-hub 0.36.2
accelerate          1.14.0
numpy               1.26.4           ← 必须与 driver 对齐,见 §1②
ray                 2.47.1
SAM-2               1.0              depth-pro 0.1      ← 两个都是纯 Python,无编译扩展
opencv-python       5.0.0.93         timm 1.0.29
bitsandbytes        0.50.2           nvidia-cudnn-cu12 9.10.2.21
einops 0.8.2 · safetensors 0.8.0 · pillow 12.0.0 · triton 3.5.1 · toolshed 0.1.0
环境内无 nvcc（不需要编译任何东西）
```

**显存实测**：Molmo 峰值 **32.11 GiB**（见 §7 的 fp32 问题）、DepthPro 7.94 GiB、SAM2 0.61 GiB。

### 2.3 `spacetools-tool-roborefer` —— RoboRefer-8B

**功能**：`roborefer` 工具，空间指代定位。RefSpatial 全部推理链依赖它，RoboSpatial 里也占主导。底层是 NVILA/llava 架构（包名 `vila`）。

```
Python              3.11.16          共 371 个包
torch               2.5.1            torch.version.cuda = 12.4
torchvision         0.20.1           torchaudio 2.5.1
flash-attn          2.8.3.post1      ← 官方预编译轮子 cu12torch2.5cxx11abiFALSE
deepspeed           0.15.4           ← 推理路径硬依赖,不是可选,见洞 #16
vila                2.0.0            ← 这就是 llava 的包名
s2wrapper           0.1              ← git 依赖,不在 PyPI,见洞 #18
transformers        4.49.0           tokenizers 0.21.0
huggingface-hub     0.28.1           accelerate 1.3.0
numpy               1.26.4
ray                 2.47.1
xformers 0.0.28.post3 · einops 0.8.1 · einops-exts 0.0.4 · timm 1.0.29
opencv-python 4.11.0.86 · peft 0.20.0 · datasets 3.2.0 · safetensors 0.5.2
triton 3.1.0 · bitsandbytes 0.45.2 · pillow 11.1.0 · qwen-vl-utils 0.0.10
nvidia-cudnn-cu12 8.9.2.26 相关组件 · 环境内 nvcc 12.4 · toolshed 0.1.0
```

**这个环境完全按 `RoboRefer/pyproject.toml` 建，包括 `torch==2.5.1`。** 那 296 条 pin 是围绕 torch 2.5.1 做出来的一整份 freeze，内部自洽；换任何一个 torch 版本，冲突会一条接一条冒出来（`accelerate==1.3.0`、`sympy==1.13.1` 等等，`sympy` 是 torch 自己的依赖）。

**副产品**：退到 torch 2.5.1 之后，flash-attn 有**现成的预编译轮子**可用（cp311 + torch2.5），不用源码编译。

**显存实测**：峰值 **17.16 GiB**（4 个 shard，加载约 53 秒）。

### 2.4 `spacetools-tool-bbox` —— 纯计算工具

**功能**：`vision_ops`（索引深度图、坐标变换等）和 `bounding_box`（从点云 + 掩码算有向包围盒）。纯 numpy，**不用 GPU、不装 torch**。

```
Python              3.11.0           共 150 个包
numpy               2.4.6
ray                 2.47.1
opencv-python       5.0.0.93
pillow              12.3.0
toolshed            0.1.0
没有 torch —— 设计如此
```

### 2.5 `spacetools-tool-graspgen` —— GraspGen 抓取生成

**功能**：`grasp_generator`，从点云 + 掩码生成 6-DoF 抓取位姿，含 top-down 过滤和碰撞检测。

**这是个孤岛环境**：CUDA 工具链、编译器、torch 全都和其他环境不同，因为 `pointnet2_ops` 要现场编译 CUDA 扩展。

```
Python              3.11.16          共 236 个包
torch               2.3.1            torch.version.cuda = 12.1
torchvision         0.18.1
pointnet2-ops       3.0.0            ← 现场编译,cubin 含 sm_80/86/89
grasp-gen           1.0.0
spconv-cu121        2.3.8            trimesh 4.5.3
transformers        4.48.3           tokenizers 0.21.4 · huggingface-hub 0.25.2
numpy               2.4.6
ray                 2.47.1
timm 1.0.15 · safetensors 0.8.0 · triton 2.3.1 · pillow 12.3.0 · toolshed 0.1.0
nvidia-cudnn-cu12 8.9.2.26
环境内 nvcc 12.1 · gcc 12.4.0    ← nvcc 12.1 不收 Ubuntu 24.04 自带的 gcc 13.3
```

**显存实测**：峰值 0.46 GiB，推理 0.88 秒。模型很小。

---

## 3. 从零搭建：每个环境怎么搭

> 已有打好的包时，走 [§4.1 从 HF 还原](#41-从-hf-还原推荐)，不必从零搭。本节是为了让人能看懂、能改、能在新平台上重来一遍。

### 3.0 全局前提

```bash
CONDA_DIR=/opt/conda-st          # conda 装在容器盘(快)
REPO_DIR=/opt/spacetools         # 仓库克隆
export HF_HOME=/workspace/hf                 # HF 缓存放持久卷
export CHECKPOINT_DIR=/workspace/checkpoints # 权重放持久卷
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"
export FLASH_ATTN_CUDA_ARCHS="80;86"         # flash-attn 不读上面那个变量
export MAX_JOBS=8                            # 按 cgroup 内存上限调,见洞 #6
```

**装 conda 之后第一件事是接受 ToS**（洞 #1）：

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
# 自检:能不能真的建环境
conda create -n _probe python==3.11 -y && conda env remove -n _probe -y
```

**克隆四个仓库**（第五个 RoboRefer 在 §3.3 里克隆到 Toolshed 目录下）：

```bash
cd $REPO_DIR
git clone --depth 1 https://github.com/spacetools/SpaceTools.git
git clone --depth 1 https://github.com/ChicyChen/SpaceTools-RL.git
git clone --depth 1 https://github.com/NVlabs/SpaceTools-Toolshed.git
git clone --depth 1 https://github.com/NVlabs/GraspGen.git
```

### 3.1 `spacetools-rl`

主干照 `SpaceTools/setup_envs.sh` 的 `setup_rl()`，有六处必须偏离。

```bash
conda create -n spacetools-rl python==3.11 -y
conda activate spacetools-rl

# ① 必须指定 --index-url 到 cu128,并且钉死 torchvision/torchaudio（洞 #2 #3）
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.57.1

# ② 约束文件,本环境专用（洞 #4）
cat > /root/constraints-rl.txt <<EOF
torch==2.9.1
torchvision==0.24.1
torchaudio==2.9.1
transformers==4.57.1
EOF
export PIP_CONSTRAINT=/root/constraints-rl.txt

# ③ 让 pip 正常解析 sglang 的依赖闭包,不要用 --no-deps + 手工列表（洞 #5）
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

# ⑤ flash-attn —— 必须让它失败就停（洞 #7）
#    先探官方预编译轮子,没有再源码编译。cp311+torch2.9 目前没有轮子,只能编。
#    用系统的 /usr/local/cuda-12.8(标准布局),不要用 conda 的 split 布局。
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=4 NVCC_THREADS=1          # 72 个 .o,每个 nvcc 峰值 5-7 GB
pip install "flash-attn==2.8.3.post1" --no-build-isolation    # 约 46 分钟

# ⑥ cudnn 二次升级 —— flash-attn 会把它降级（洞 #8）
pip install qwen_vl_utils
pip install cachetools "nvidia-cudnn-cu12==9.16.0.29"

unset PIP_CONSTRAINT     # 约束只属于本环境,绝不带进工具环境
```

**关于 flash-attn 的预编译轮子**：Dao-AILab 的 release 资产按 `<cuda> × <torch minor> × <cxx11abi> × <python>` 四元组发布，四个字段必须**严格对上**——`.so` 直接链接 libtorch 的 C++ 符号，torch 每个小版本都会改 mangled name。装错组合的表现是 `ImportError: undefined symbol: _ZNK3c106...`。

```
cp311 的轮子只到 torch2.8,没有 torch2.9    → rl 环境只能源码编译
cp311 + torch2.5 有现成轮子                → roborefer 环境可以直接装
torch2.9 的轮子只发了 cp312
```

### 3.2 三个"标准"工具环境（bbox / vlm / roborefer 骨架）

Toolshed 的 `setup_tool_env.sh <env名> <工具名>` 做三件事：`create_tool_env.sh` 建空环境（对齐 Python + Ray）→ 装 toolshed → 跑 `tool_scripts/install_<工具>.sh`。

**必须从 `spacetools-rl` 激活状态下调用**，因为版本是从当前环境探测的。

```bash
conda activate spacetools-rl
cd $REPO_DIR/SpaceTools-Toolshed
source install_tools/setup_tool_env.sh spacetools-tool-bbox      bbox
source install_tools/setup_tool_env.sh spacetools-tool-vlm       vlm
source install_tools/setup_tool_env.sh spacetools-tool-roborefer roborefer
```

**vlm 环境要补三样**（洞 #9、#10）：

```bash
conda activate spacetools-tool-vlm
cd $REPO_DIR/SpaceTools-Toolshed
bash install_tools/tool_scripts/install_sam2.sh      # run_eval.sh 把 sam2 排在 vlm 环境
bash install_tools/tool_scripts/install_depth.sh     # depth 也是
pip install "transformers==4.53.2"                   # 上面两步会顺手升 transformers,装完拉回来
pip install "numpy==1.26.4"                          # 与 driver 对齐,见 §1②
```

`numpy` 降到 1.26.4 之后要复验 `opencv-python` / `timm` / `bitsandbytes` 仍可用——它们有可能是按 numpy 2 的 ABI 编译的（本次构建实测没有问题）。

### 3.3 `spacetools-tool-roborefer` 的 llava 部分

`setup_tool_env.sh ... roborefer` 只建了骨架，llava（`vila`）要单独装，而且上游脚本在这一步有三个洞（#13 #14 #15）。

```bash
conda activate spacetools-tool-roborefer
cd $REPO_DIR/SpaceTools-Toolshed

# ① 自己克隆,不要指望 install_roborefer.sh 走到那一步（洞 #13）
git clone --depth 1 https://github.com/Zhoues/RoboRefer.git

# ② 权重先落地
hf download Zhoues/RoboRefer-8B-SFT --local-dir $CHECKPOINT_DIR/RoboRefer-8B-SFT
ln -s $CHECKPOINT_DIR $REPO_DIR/SpaceTools-Toolshed/checkpoints

# ③ 跳过 env_setup.sh,直接按 pyproject 装（洞 #14 #15）
#    不加 --no-deps —— 那 296 条 pin 是自洽的一整套,包括 torch==2.5.1
cd RoboRefer && pip install -e .

# ④ flash-attn:这个组合有现成轮子,2 分钟
FA=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1
pip install $FA/flash_attn-2.8.3.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

`pip install -e .` 会自动带上 `deepspeed==0.15.4`（洞 #16）和 `s2wrapper`（git 依赖，洞 #18）。

### 3.4 `spacetools-tool-graspgen`

顺序很关键——**四个准备动作必须在跑上游 `install_graspgen.sh` 之前完成**。

```bash
# ① 先改 GraspGen 的 pyproject 并本地 commit（洞 #9 #10）
#    上游脚本在目录已存在时会 `git pull`,而它 set -e,工作区脏就炸
cd $REPO_DIR/GraspGen
sed -i '/"pickle5",\?/d' pyproject.toml          # py3.11 编不过,代码里有 try/except 回退
git -c user.email=b@l -c user.name=b commit -am "drop pickle5"

# ② 扫全仓库,把硬编码的架构列表改成 setdefault（洞 #12）
#    有两个文件:pointnet2_ops/pointnet2_ops/pointnet2_utils.py
#              grasp_gen/models/pointnet/pointnet2_utils.py
grep -rl 'os.environ\["TORCH_CUDA_ARCH_LIST"\]' . --include=*.py | while read f; do
  sed -i 's|os\.environ\["TORCH_CUDA_ARCH_LIST"\][[:space:]]*=[[:space:]]*\(".*"\)|os.environ.setdefault("TORCH_CUDA_ARCH_LIST", \1)|' "$f"
done
git commit -am "TORCH_CUDA_ARCH_LIST setdefault"

# ③ 建空环境(从 rl 探测版本),然后**抢在上游之前**把 CUDA 12.1 装好（洞 #11）
conda activate spacetools-rl
cd $REPO_DIR/SpaceTools-Toolshed
source install_tools/create_tool_env.sh spacetools-tool-graspgen
conda activate spacetools-tool-graspgen
printf 'cuda-version 12.1.*\n' > $CONDA_PREFIX/conda-meta/pinned
conda install -y -c nvidia/label/cuda-12.1.1 --override-channels \
      "cuda-toolkit=12.1.1" "cuda-nvcc=12.1"
#    判据:nvcc 必须是 12.1
nvcc --version | grep "release 12.1" || echo "没钉住"

# ④ gcc 12（洞 #17）
conda install -y -c conda-forge "gcc_linux-64=12" "gxx_linux-64=12"
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export NVCC_PREPEND_FLAGS="-ccbin $CC"

# ⑤ 现在才跑上游脚本。两个交互式 read 用环境变量喂掉（洞 #10）
pip install -e .                                  # toolshed
export GRASPGEN_DIR=$REPO_DIR MODELS_DIR=$CHECKPOINT_DIR
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" MAX_JOBS=4
bash install_tools/tool_scripts/install_graspgen.sh
```

**为什么 CUDA 必须"抢在前面"装**：上游 `install_graspgen.sh:124` 写的是 `conda install -c nvidia cuda-toolkit=12.1 -y`，而 `nvidia` 频道的 `cuda-toolkit 12.1` 是**空壳元包，不约束 `cuda-version`**，组件会解析成 13.3.x（洞 #11）。更麻烦的是 conda 的"已满足"短路**只比包名和版本、不比频道**，所以在它**之后**再补一条 `conda install -c nvidia/label/cuda-12.1.1 ...` 会直接返回 "All requested packages already installed"，什么也不做。唯一有效的做法是让它那一句变成真正的 no-op。

---

## 4. 怎么跑

### 4.1 从 HF 还原（推荐）

```bash
hf download qzpm55555/spacetools-eval-env --local-dir ./pkg
cd pkg
cat spacetools-envs-*.tar.zst.part* > spacetools-envs-20260910-0959.tar.zst

bash FETCH_WEIGHTS.sh    # 按钉死的 revision 拉 79G 权重
bash RESTORE.sh          # 解到 /opt,自带前置检查
bash VERIFY.sh           # 三项真实验收,全过才动 eval
```

**还原前必须确认三件事**（`RESTORE.sh` 会检查前两件）：

| | |
|---|---|
| 还原路径 | **必须是 `/opt/conda-st` 和 `/opt/spacetools`** —— conda 环境里烧死了绝对路径 |
| GPU 架构 | **A100 (sm_80) / A6000 (sm_86) / L40S (sm_89) 可用**；H100 (sm_90) 不可用——编译时没带 `+PTX`，没有能 JIT 到 Hopper 的中间码 |
| glibc | **不能比 Ubuntu 24.04（glibc 2.39）老**。`flash_attn_2_cuda` 和 `pointnet2_ops/_ext` 链的是系统 glibc，不在 conda 环境里。22.04（glibc 2.35）会报 `GLIBC_2.38 not found` |
| 驱动 | ≥ 550（torch 是 cu128；本次构建用的是 580.159.04） |

权重的版本全部钉死在 `WEIGHTS_PINS.txt`：

```
allenai/Molmo-7B-D-0924             cab33fb7f1a40091911f81165f8481920621948f   30 G
Zhoues/RoboRefer-8B-SFT             bd04070786084c624156194d89333c375b274b28   16 G
Qwen/Qwen2.5-VL-3B-Instruct         66285546d2b821cf421d4f5eb2576359d3770cd3  7.1 G
facebook/sam2.1-hiera-small         ee5bba1d82bb8749febdf90f45e84b687142ba03  180 M
siyich/spacetools-eval-benchmarks   1d539ac935872c7aa712c85a77bf4b0cb469c8e8  283 M   (dataset)
adithyamurali/GraspGenModels        commit ec1ccbb5eec0680db669246ac312a3636f16ee43  7.9 G  (git-lfs)
depth_pro.pt                        sha256 3eb35ca6…85c0ce · 1904446787 字节   (Apple CDN)
```

`facebook/sam2.1-hiera-small` 不在任何文档的清单里，是 `install_sam2.sh` 运行时自己拉的——离线机器上如果不预先准备，sam2 会在**第一次调用时**才失败。

### 4.2 跑 eval

```bash
cd $REPO_DIR/SpaceTools-RL/examples/toolshed
bash run_eval.sh <ckpt路径> robospatial reflocation refplacement refunseen
```

它做的事：`ray start --head` → 用 `start_toolkit(TOOL_CONFIGS)` 起 router 和各工具 actor → 起策略模型 → 跑 benchmark。

**GPU 数量要求**（`run_eval.sh:132-142` 的工具预留）：

```
roborefer        1 actor × 0.6 = 0.6
vlm              1 actor × 0.6 = 0.6
sam2             2 actors × 0.2 = 0.4
depth_estimator  2 actors × 0.2 = 0.4
bounding_box     2 actors × 0.1 = 0.2
vision_ops       1 actor × 0    = 0
grasp_generator  1 actor × 0.1 = 0.1
                             ─────
                             2.3 张卡  + 策略模型自己的那份 → 至少 3 张
```

Ray 的 `num_gpus` 是**逻辑预留**：一张卡只能满足总量 1.0 的请求，剩下的 actor 会**永远排队等资源，不报错**。这是最难查的一种失败。

**显存也要够**：`num_gpus` 的分数不切分显存，同卡上的 actor 共享全部显存。实测常驻峰值 Molmo 32.11 GiB + RoboRefer 17.16 GiB 就已经超过单张 A6000 的 48 GB。

### 4.3 跑 RL

`run_rl.sh` 的 actor 数量大得多（roborefer 6 个、sam2/depth/bbox/graspgen 各 5 个、vision_ops 8 个），GPU 需求相应更高。SFT → RL 之间应当先用 §4.2 的 eval 确认 checkpoint 可用。

---

## 5. 上游的 20 个洞

按构建阶段排列。每一条都给出位置、机制、对策。

### 阶段 A：`spacetools-rl`

**#1 conda ≥26 的 ToS 会静默挡住建环境**
新版 conda 未接受 Terms of Service 时 `conda create` 直接失败（`CondaToSNonInteractiveError`），**而且只在 stderr 里报**。`setup_envs.sh` 不检查返回值，于是后续所有 `pip install` 全都装进了 base 环境，要到编译 flash-attn 去找 `$CONDA_PREFIX/nvcc` 时才炸——那时已经过了十几分钟。
→ 建环境前先 `conda tos accept`，并用一次 `conda create` + `conda env remove` 自检。

**#2 torch 安装没有 `--index-url`**
`setup_envs.sh:41` 写的是 `pip install torch==$TORCH_VERSION torchvision torchaudio`，走 PyPI 默认轮子。论文写作时 torch 2.9.1 的默认是 cu128，而 PyPI 上的默认已经变成 cu130，于是 nvcc 12.8 编 flash-attn 时报 `detected CUDA 12.8 mismatches the version used to compile PyTorch 13.0`。
→ 显式 `--index-url https://download.pytorch.org/whl/cu128`。

**#3 `torchvision` / `torchaudio` 不钉版本**
同一行里只钉了 torch。pip 会解析到给**别的 torch 版本**编译的 torchaudio，`import` 时 `undefined symbol`。
→ 三个一起钉。

**#4 `flashinfer_python` 不钉版本，会静默换掉 torch**
上游手工安装的 `flashinfer_python` 最新版要求 `torch > 2.9.1`，pip 会**不报错地**把 torch 连同整套 `nvidia-*-cu13` 升到 2.14.0+cu130。整个环境的口径悄悄变了，而所有命令都返回 0。
→ 给本环境写一份 `PIP_CONSTRAINT`，钉住 torch/torchvision/torchaudio/transformers。冲突会变成一条明确的 pip 报错，而不是一个静默改口径的环境。**约束文件是环境级的，不能导出成全局**——工具环境各有各的 torch，那是设计如此。

**#5 sglang 用 `--no-deps` + 手工依赖表，表不全**
`setup_envs.sh` 先 `pip install "sglang[srt,openai]==0.5.6" --no-deps`，再手工列一长串依赖。那份表缺包（运行时 `ModuleNotFoundError: No module named 'pybase64'`），而且手装的 `flashinfer_python` 会把 `nvidia-cutlass-dsl` 顶到 4.8.0.dev0，与 sglang 0.5.6 要求的 `==4.2.1` 冲突。
→ 有 `PIP_CONSTRAINT` 守住 torch 之后，让 pip 正常解析 sglang 的真实依赖闭包，版本由 sglang 自己决定，比手工表准。

**#6 `sglang[srt,openai]` 的两个 extra 在 0.5.6 里不存在**
只是 warning，不影响功能，但说明这一行是从别的版本抄来的。

**#7 flash-attn 的编译失败被吞掉**
`setup_envs.sh` 写的是 `pip install flash-attn --no-build-isolation 2>/dev/null || echo "WARNING: ..."`。错误被重定向丢弃、失败也继续。结果是环境里没有 flash-attn，而脚本一路绿灯。
→ 让它失败就停。另外编译本身有两个坑：
- **CUDA 头文件路径**：conda 的 cuda-toolkit 用 **split 布局**，头文件在 `$CONDA_PREFIX/targets/x86_64-linux/include/`，而 torch 的 `cpp_extension` 只往 `$CUDA_HOME/include` 找。`nvcc --version` 一切正常，炸的是 host 编译器编 `flash_api.cpp` 那一步，报 `fatal error: cuda_runtime_api.h: No such file or directory`。用系统的 `/usr/local/cuda-12.8`（标准布局）可以绕开。
- **内存**：`free -g` 显示的是宿主机内存（503 GB），容器真正的上限在 `/sys/fs/cgroup/memory/memory.limit_in_bytes`（本次是 46.6 GiB）。flash-attn 的 backward kernel 每个 nvcc 峰值 5–7 GB，`MAX_JOBS` 开大会被 OOM killer 杀掉，日志里**只有 `Killed`，没有任何编译错误**。`MAX_JOBS=4` + `NVCC_THREADS=1` 在 46.6 GiB 下安全。

**#8 cudnn 的二次升级只写在 docs，没写进脚本**
flash-attn 安装会把 `nvidia-cudnn-cu12` 降级。`docs/SETUP.md` 的手动路径里有二次升级到 9.16.0.29，但 `setup_envs.sh` 漏了这一步。

### 阶段 B：工具环境的通用机制

**#9 `create_tool_env.sh` 在环境已存在时是交互式的**
`create_tool_env.sh:58-65`：环境已存在时 `read -r response` 问 `continue (c) / recreate (r) / abort (a)`，**空输入默认 `c`**。非交互执行时 stdin 读到空，于是永远走"继续用现有环境"——**指望它重建是不行的**，必须自己先 `conda env remove`。

**#10 `install_graspgen.sh` 的两个交互式 `read`**
`install_graspgen.sh:37` 和 `:52` 分别问 `GRASPGEN_DIR` 和 `MODELS_DIR`。非交互下 EOF + `set -e` 直接静默退出，日志停在 `Enter directory ...` 那一行。
→ 两个都能用同名环境变量预设跳过。

**#11 `cuda-toolkit=12.1` 钉不住 CUDA 版本，而且补救无效**
`install_graspgen.sh:124` 的 `conda install -c nvidia cuda-toolkit=12.1 -y`：`nvidia` 频道的 `cuda-toolkit 12.1` 是 1 KB 的**空壳元包，对组件的依赖是宽松的**，解析结果是 `cuda-nvcc 13.3.73` / `cuda-version 13.3`。编译 `pointnet2_ops` 时报 `RuntimeError: The detected CUDA version (13.3) mismatches PyTorch (12.1)`。

**真正的版本闸门是 `cuda-version` 这个元包，不是 `cuda-toolkit`。** 而且 `nvidia/label/cuda-12.1.1` 这个旧式 label 频道**根本不提供 `cuda-version`**（`conda search` 返回 No match），所以 spec 里不能带它。

更麻烦的是：**conda 的"已满足"短路只比包名和版本，不比频道**。在上游那句之后再补一条 `conda install -c nvidia/label/cuda-12.1.1 --override-channels cuda-toolkit` 会直接返回 `All requested packages already installed`，`--override-channels` 完全不起作用。
→ 唯一有效的做法：**在上游那句之前**把 `cuda-toolkit=12.1.1` + `cuda-nvcc=12.1` 从 label 频道装好，让它那一句变成真正的 no-op；再写 `$CONDA_PREFIX/conda-meta/pinned` 钉住 `cuda-version 12.1.*` 作为第二道保险。

**#12 `pointnet2_ops` 在 import 时硬覆盖编译架构**
`pointnet2_utils.py:23`：

```python
os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
```

这是**赋值不是 setdefault**，在 import 时执行，外面设的 `8.0;8.6;8.9` 完全无效。CUDA 12 已不支持 sm_37，于是 `nvcc fatal: Unsupported gpu architecture 'compute_37'`。

**这一句在 GraspGen 仓库里有两处**：`pointnet2_ops/pointnet2_ops/pointnet2_utils.py` 和 `grasp_gen/models/pointnet/pointnet2_utils.py`。只改其中一个，能不能通取决于导入顺序。
→ 扫全仓库所有 `.py`，把赋值改成 `setdefault`（外部设了用外部的，没设才用它的默认，不改变上游在"没人设时"的行为）。

**#13 `install_roborefer.sh` 把克隆放在会 `exit 1` 的下载之后**
`install_roborefer.sh` 先 `hf download` 16 GB 权重，失败就 `exit 1`；而**克隆 RoboRefer 仓库（llava 的唯一来源）排在它后面**。下载一中断，llava 就永远装不上，而 Toolshed README 第 100 行明确写着 RoboRefer "requires cloning a fork of RoboRefer"。
→ 自己先克隆。

**#14 `RoboRefer/env_setup.sh` 硬编码了一个装不上的 flash-attn 轮子**
它写死 `flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`，在 Toolshed 建的 py3.11 环境上报 `is not a supported wheel on this platform`。而这一步排在 `pip install -e ".[train,eval]"`（**唯一装 llava 的一步**）**之前**，`set -e` 一停，llava 就没装成。
→ 跳过 `env_setup.sh`，直接 `pip install -e .`。它的第 4–6 步（`triton==3.1.0`、`deepspeed_replace`、`protobuf`）是训练路径用的，纯推理不需要。

**#15 GraspGen 依赖 `pickle5`，在 Python 3.11 上编不过**
`pickle5` 是 Python 3.5–3.7 的 protocol-5 后向移植，3.8 起标准库自带。在 3.11 上编译报 `error: lvalue required as left operand of assignment`（`Py_SIZE` 不再是左值），**整个 pip 事务失败**，于是 torch 2.3.1 和 `grasp_gen` 一个都没装上。

GraspGen 里唯一的用处是 `grasp_gen/dataset/dataset.py:27` 的 `import pickle5 as pickle`，而且**包在 `try/except ImportError` 里**——删掉依赖会自动回退到标准库 pickle。
→ 在建环境**之前**从 `pyproject.toml` 删掉，并本地 `git commit`（因为上游脚本在目录已存在时会 `git pull`，工作区脏会失败）。注意上游脚本自己也会 `sed -i` 改这个文件并**验证 sed 生效、不生效就 exit 1**，所以不要动它的锚点。

**#16 nvcc 12.1 不接受 Ubuntu 24.04 的 gcc 13.3**
`crt/host_config.h:132: #error -- unsupported GNU version! gcc versions later than 12 are not supported!`
→ 从 conda-forge 装 `gcc_linux-64=12` / `gxx_linux-64=12`，用 `NVCC_PREPEND_FLAGS="-ccbin ..."` 指过去。不要用 `-allow-unsupported-compiler` 硬绕——那个 flag 自己的警告里就写着"可能导致运行时行为错误"。

### 阶段 C：RoboRefer / llava

**#17 推理路径强依赖 `train` extra**

```
llava/model/language_model/llava_llama.py:28
    from llava.train.sequence_parallel.globals import get_pg_manager
llava/train/sequence_parallel/globals.py:22
    import deepspeed.comm as dist          ← 模块级,无条件
```

而 `RoboRefer/pyproject.toml:343` 把 deepspeed 归在 `train = ["deepspeed==0.15.4", "ninja", "wandb"]` 里，`eval` extra 里没有。**只装 eval extra 的人，连模型定义都 import 不进来。**

**#18 `flash_attn` 也是加载模型的硬依赖**

```
llava/model/multimodal_encoder/intern/flash_attention.py:23,25
```

两条 import 路径都在 `flash_attn` 这一个包里（`except` 只是兼容新旧 API 名），而它在 **InternViT 视觉塔的加载路径上**：`llava.load → model/__init__ → llava_llama → llava_arch → multimodal_encoder.builder → intern_encoder → modeling_intern_vit → flash_attention`。绕不开。

**#19 `s2wrapper` 不在 PyPI 上**
`pyproject.toml:48` 写的是 `"s2wrapper@git+https://github.com/bfshi/scaling_on_scales"`。按模块名 `pip install s2wrapper` 必然 `No matching distribution found`。
→ 直接 `pip install -e .`（带依赖）会自动处理；手工装依赖时要用 git URL。

**#20 `llava/data/datasets_mixture.py:60` 是真的语法错误**

```python
2D_choice_qa = Dataset(
^
SyntaxError: invalid decimal literal
```

`2D_choice_qa` 不是合法的 Python 标识符（不能以数字开头），这个文件在**任何 Python 版本下都 import 不了**。它只在训练路径用，所以上游一直没发现。连带影响 6 个模块：`llava.data`、`llava.eval.egoschema`、`llava.eval.eventbench`、`llava.eval.model_vqa_video`、`llava.eval.rtl`、`llava.eval.vnbench`。纯推理不受影响，但说明 RoboRefer 仓库里有整块代码从未被执行过。

### 阶段 D：跨环境集成（最隐蔽的一组）

**#21 四份依赖声明关于 numpy 互相矛盾**

```
SpaceTools-RL/setup.py:32          numpy<2.0.0     ← driver 必须 numpy 1.x
toolshed/pyproject.toml:27         numpy>=2.0.0
tool-vlm.txt:22                    numpy>2.0.0
tool-sam2.txt:27                   numpy>=2.0.0
tool-depth.txt:4                   # IMPORTANT: This tool requires numpy<2.0.0
                                   #   which conflicts with the base toolshed
tool-depth.txt:27                  # numpy<2.0.0        ← pin 被注释掉了
```

`numpy` 一个要 `>2.0`、一个要 `==1.26.4`，不可能同时满足。而 `setup_tool_env.sh` 只装前者。

**后果**：numpy 2.x 的 actor 返回带 `ndarray` 的 `ToolResult` 给 numpy 1.x 的 driver 时，反序列化报 `No module named 'numpy._core.numeric'`。`numpy._core` 是 numpy 2 的内部模块路径，numpy 1.x 里不存在。

分形状实测（numpy 2.x actor → numpy 1.x driver）：

```
bare ndarray          FAIL      ← Ray 的零拷贝通道救不了
dict{ndarray}         FAIL
tuple(ndarray)        FAIL
list[float]           OK
np.float32 标量        OK
ToolResult(纯文本/标量) OK
反方向 numpy1 → numpy2  OK      ← numpy 2 有兼容垫片
```

具体到七个工具：**只有 `depth_estimator` 和 `sam2` 的 `ToolResult` 带 `ndarray`**（`$depth_map`、`$segmentation_mask`），它们都在 `tool-vlm` 环境里。`bounding_box` 和 `grasp_generator` 的返回值里每个数组都调了 `.tolist()`（`grasp_generator.py:504,548`），是纯 Python 列表，不受影响。
→ **`tool-vlm` 必须用 numpy 1.26.4**。`bbox` 和 `graspgen` 保持 2.4.6 是安全的（也不该动——`graspgen` 里的 `pointnet2_ops` 是在 numpy 2 环境下编译的）。

**#22 toolshed 的工具用"返回错误字符串"代替"抛异常"**
工具出错时不抛异常，而是把错误信息包成正常的 `ToolResult` 返回；router 也会把 actor 里的异常吞掉，包成正常返回。

```
CALL_OK   depth_estimator -> ModuleNotFoundError: No module named 'numpy._core.numeric'
                             返回类型: ToolResult
```

注意那是 `CALL_OK`，不是 `CALL_FAIL`。**调用方拿不到任何异常。**

这意味着 numpy 那个问题如果不修，eval 会**一路跑完不报错**，每一次 `depth_estimator` 调用都返回一句报错文本当深度图，模型拿着垃圾输入继续推理，最后吐出一个**看起来正常的分数**——然后人会去怀疑 checkpoint。
→ 任何验收都必须检查 `ToolResult` 的正文里有没有 `Error:` / `ERROR:toolshed`，不能只看"有没有崩"。

**#23 `run_eval.sh` 把 sam2 和 depth 排在 vlm 环境，但安装脚本不装它们**
`run_eval.sh:136,137` 里 `'sam2'` 和 `'depth_estimator'` 的 `conda_env` 都是 `spacetools-tool-vlm`，而 `setup_tool_env.sh <env> vlm` 只装 `tool-vlm.txt`（Molmo 的依赖）。照文档做会得到一个跑不了 sam2/depth 的环境，而且要到 **actor 被调用时**才报错。
→ 在 vlm 环境里补跑 `install_sam2.sh` 和 `install_depth.sh`。

**#24 `vlm.py:116` 写死 `torch_dtype="auto"`，吃掉调用方传的 dtype**
`run_eval.sh` 的配置里明确传了 `'dtype': 'float16'`，但代码写死 `auto`，Molmo 实际以 **fp32** 加载。实测峰值 **32.11 GiB**（fp16 应当约 17 GB）。两台不同机器、两次独立构建得到同一个数字。
→ 48 GB 的单卡扛得住，但它直接决定要租几张卡。修这一行可以把 Molmo 的占用减半。

**#25 工具的参数口径不统一，且同一文件内也不一致**

```
sam2.segment_from_point(image, x, y)       x,y 是**归一化** [0,1]
vision_ops.index_at(data, u, v)            u,v 是**归一化** [0,1]
grasp_generator.compute_grasp(...)
    _resolve_pointcloud  接受 numpy        (grasp_generator.py:71)
    _resolve_mask        接受 numpy        (grasp_generator.py:89)
    _resolve_image       **只接受 str 或 PIL.Image**，不接受 numpy   (grasp_generator.py:80)
sam2 / depth_estimator 的 image 参数        接受 numpy
```

同一个文件里三个 `_resolve_*` 对 numpy 的态度不一致，不同工具的 `image` 参数口径也不一致。**接工具时每个签名都要读源码，不能按类比想当然**——而且传错的表现是 #22 描述的那种"返回错误字符串但调用成功"。

---

## 6. 验收判据

这套环境的失败模式高度集中在"**看起来成功了**"。以下判据都是针对具体失败模式设计的。

### 6.1 三层验收

`VERIFY.sh` 跑三项，全过才算可用：

| 层 | 脚本 | 判什么 |
|---|---|---|
| 1 | `03_verify.sh` | 五环境关键模块 import + 所有 `.so` 的 cubin 架构实扫 + `pip freeze` 冻结 |
| 2 | `04_smoke.sh` | 七个工具逐个真加载权重、真出结果 |
| 3 | `28_chain.sh` | Ray 跨 conda 环境的工具链：ndarray 双向传递 |

本次构建的结果：

```
五环境 import 闸门        全通
架构实扫                  4619 个 .so,108 个含 cubin,自编扩展缺 sm_8x 共 0 个
七工具冒烟                7/7
Ray 跨环境 actor          5/5
跨环境变量传递             ndarray 双向正确
```

### 6.2 具体判据与它们针对的失败模式

**`import` 类**

- 检查 `flash_attn` 装没装，要看 **`flash_attn.__version__` 取不取得到** + **`import flash_attn_2_cuda` 通不通**，不能用裸 `import flash_attn`。失败的编译会在 `site-packages` 留下一个**没有 `__init__.py` 的 `flash_attn/` 目录**，Python 3 把它当命名空间包，`import` 会成功而 `__file__` 是 `None`。
- `import flash_attn_2_cuda` 之前**必须先 `import torch`**。这个扩展链接 libtorch，裸 import 会 `ImportError: libc10.so: cannot open shared object file`。
- 验 `llava` 之前**先 `cd /`**。RoboRefer 仓库根目录下就有 `llava/` 包目录，在那里 import 命中的是源码树而不是装好的包。同时打印 `llava.__file__` 确认来源。
- `import llava` 太浅——`llava/__init__.py` 很薄，真正的依赖要到 `llava.load()` 才逐层 import 进来。真判据是**把 `RoboreferTool` 构起来**（加载 16 GB 权重）。

**架构类**

- 用 `cuobjdump --list-elf <so>` 检查 cubin 里有没有 `sm_8x`。
- **先断言 `cuobjdump` 存在**（工具环境里不一定装了 cuda-toolkit），并**打印实际扫了多少个文件**。扫到 0 个要明确报"没有结论"，既不算通过也不算失败。
- 区分**自编扩展**和**厂商预编译库**。`libcufft` / `libnppc` / `libcublas` / `libcusolver` / `libaccinj64` / `torch/lib/libtorch_cuda_linalg.so` / `sgl_kernel/flashmla_ops`（Hopper 专用）的架构由 NVIDIA 决定，不该计入失败；真正要盯的是 `flash_attn_2_cuda` 和 `pointnet2_ops/_ext`。

**工具调用类**

- 判据是 **`SMOKE_OK` 打印了，并且结果正文里不含 `-> Error:` 或 `ERROR:toolshed`**（针对 #22）。
- 检查 `hf_device_map` 里没有 `cpu` / `disk`——accelerate 在显存不够时会静默 offload，那样的结果不作数。
- 打印 `torch.cuda.max_memory_allocated()`。显存峰值是"模型到底跑没跑"的旁证：只加载不计算，和真的跑了一遍，数字差一个量级。

**跨环境类**

- 光验"actor 起得来"不够，要验**带 `ndarray` 的 `ToolResult` 能不能跨环境回到 driver**，而且要走**真实的 router 路径**（`start_toolkit` + `ToolkitClient`），不是直接 `ray.get()`。

**通用**

- 每一步之后验证**实际状态**，不看返回值。上游脚本里大量使用 `2>/dev/null || echo WARNING` 和无检查的 `set -e`，退出码不可信。
- 任何自检都要**打印实际检查了多少个对象**。不打印计数的自检，无法区分"全通过"和"一个都没检查"。
- 验收代码本身跑在哪个解释器里、有没有它需要的依赖，**是判据的一部分**。验收工具坏了和被验对象坏了，现象一模一样。

### 6.3 eval 的判读

跑 `run_eval.sh` 之后：

1. **先 `grep -c OutOfMemoryError`**，再看分数。OOM 的样本会算进分母。
2. 判据：RoboSpatial ≥ 60 **且** RefSpatial ≥ 48 → 可以进 RL；< 55 或 < 40 → 停下来查。
3. **不要用 `boppose` 当判据**——它的 metric 映射尚未厘清。

---

## 7. 已知限制与未决事项

### 7.1 硬件兼容范围

| | |
|---|---|
| ✅ A100 (sm_80) / A6000 (sm_86) / L40S (sm_89) | cubin 覆盖这三个架构 |
| ❌ H100 / H200 (sm_90) | 编译时 `TORCH_CUDA_ARCH_LIST` 只到 8.9 且**没带 `+PTX`**，没有能 JIT 到 Hopper 的中间码。上 Hopper 要重编 `flash-attn` 和 `pointnet2_ops` |
| ⚠️ glibc | 不能比 Ubuntu 24.04（2.39）老 |
| ⚠️ 驱动 | ≥ 550 |
| ⚠️ 还原路径 | 必须 `/opt/conda-st` + `/opt/spacetools` |

### 7.2 未处理的事项

**① Molmo 的 fp32 问题（洞 #24）**
`vlm.py:116` 的 `torch_dtype="auto"` 未修。Molmo 常驻 32.11 GiB，直接影响 GPU 预算。修这一行是"少租卡"的关键一步。

**② eval 的最小工具集未裁剪**
`run_eval.sh` 无条件起全部 7 个工具（2.3 张卡的预留）。而 robospatial / reflocation / refplacement / refunseen 这四个 key 用不到 `grasp_generator`，`vlm` 和 `roborefer` 在不同 benchmark 里的权重也差别很大。按 benchmark 裁工具集可以降低 GPU 需求。

**③ 策略模型的 checkpoint 未钉版本**
`FETCH_WEIGHTS.sh` 里的 `MY_SFT_REPO` 是空的。HF 上的 `main` 随时可能被覆盖，建议把 commit SHA 一起钉下来：

```bash
MY_SFT_REPO=<用户名>/<仓库名> MY_SFT_REV=<commit> bash FETCH_WEIGHTS.sh
```

**④ `graspgen` 的合成场景测试**
用随机点云测 `compute_grasp` 时，网络前向能正常跑完（实测 1704 个物体点 → 生成 200 个抓取 → top-down 过滤剩 26 个，置信度 [0.678, 0.926]，推理 0.88 秒），但碰撞过滤会把结果全滤掉——因为合成场景里"场景点云"包含了物体自身的点，夹爪必然与之相交。这是测试场景的性质，不是环境问题。真实数据上不会这样。

### 7.3 包内容

```
https://huggingface.co/qzpm55555/spacetools-eval-env   (private)

spacetools-envs-*.tar.zst.part00..05        22.10 GB   五个 conda 环境 + 四个仓库(带 .git)
spacetools-scripts-*.tar.zst                  81 KB    构建/修复/验收脚本 + constraints + freeze
build-logs/build-logs-20260910.tar.zst        85 KB    76 份构建日志 + 七工具冒烟输出
build-logs/README_LOGS.md                              日志索引
MANIFEST.txt                                           打包机器 / 仓库 commit / 关键版本
WEIGHTS_PINS.txt                                       权重 revision 钉死表
RESTORE.sh / VERIFY.sh / FETCH_WEIGHTS.sh              还原三件套
README.md / SHA256SUMS.remote
```

**仓库 commit**

```
SpaceTools           17d585539b6cc32f2b3c068ea2591762b4583fd9
SpaceTools-RL        54270e82443d3d2a4c2a737c2d3b33314a991fcc
SpaceTools-Toolshed  4f0512d092f53abc1e6c5c934245bf83a211466a
GraspGen             9b3cfc1e5b664698e047ddd482832f6e7796380c   (含本地的两个 commit)
RoboRefer            d97a995ad28376720a4c8beb64915c58ed16c844
```

GraspGen 的两个本地 commit 就是洞 #15（删 pickle5）和洞 #12（`TORCH_CUDA_ARCH_LIST` 改 setdefault）的改动记录，`git log` 可查。
