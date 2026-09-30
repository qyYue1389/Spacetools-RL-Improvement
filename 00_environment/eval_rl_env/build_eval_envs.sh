#!/bin/bash
# =============================================================================
# SpaceTools eval 环境构建 —— 1× A6000 / RunPod
#
# 主干照 github.com/spacetools/SpaceTools 的 setup_envs.sh 与
# github.com/NVlabs/SpaceTools-Toolshed 的 install_tools/setup_tool_env.sh。
# 每一处偏离都在补上游的漏,都标了 ⚠️,并会写进最终包的 README。
#
#   nohup bash build_eval_envs.sh > /workspace/build.log 2>&1 &
#
# 幂等:已存在的环境会跳过,可以重跑。
# =============================================================================
set -uo pipefail

# ---- 路径:conda 和仓库放容器盘(本地,快);权重放网络卷(跨 pod 保留)------
CONDA_DIR=/opt/conda-st
REPO_DIR=/opt/spacetools
export HF_HOME=/workspace/hf
export CHECKPOINT_DIR=/workspace/checkpoints     # ⚠️ 见 §3.3,不设会挂死
export PIP_CACHE_DIR=/root/.cache/pip
export TMPDIR=/root/tmp
export HF_HUB_ENABLE_HF_TRANSFER=1

# ---- 编译架构:必须显式设,否则只编当前这张卡 -------------------------------
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX"
export FLASH_ATTN_CUDA_ARCHS="80;86"    # ⚠️ flash-attn 不读上面那个,读它自己的
export MAX_JOBS=32

PY=3.11; TRANSFORMERS=4.57.1; TORCH=2.9.1; RAY=2.47.1
# ⚠️ 上游只钉 torch,torchvision/torchaudio 不钉 —— SFT 那轮因此拿到给 torch 2.11
#    编的 torchaudio,import 时 undefined symbol。这里补齐。
TORCHVISION=0.24.1; TORCHAUDIO=2.9.1

mkdir -p "$REPO_DIR" "$HF_HOME" "$CHECKPOINT_DIR" "$TMPDIR" "$PIP_CACHE_DIR"
step() { echo; echo "############ $* ############  $(date +%H:%M:%S)"; }
have_env() { "$CONDA_DIR/bin/conda" env list 2>/dev/null | grep -q "^$1 "; }
die()  { echo "✗ $*"; exit 1; }

# ⚠️ 这个断言是本脚本第一版缺的,代价是 16 GB 装进了 base:
#    conda create 因 ToS 失败 → activate 失败 → 脚本不停 → pip 装进 base 的 python3.14
#    → flash-attn 去找 /opt/conda-st/bin/nvcc 才炸,而那时已经跑了十分钟。
#    "退出码会骗人" —— 每一步之后验证实际状态,不看返回值。
act () {   # $1 = 环境名。激活并证明真的激活了
    set +u; conda activate "$1"; set -u
    [ "${CONDA_DEFAULT_ENV:-}" = "$1" ] || die "期望激活 $1,实际是 '${CONDA_DEFAULT_ENV:-无}'"
    case "$(python -V 2>&1)" in
        *" $PY."*) ;;
        *) die "$1 里的 python 不是 $PY:$(python -V 2>&1)" ;;
    esac
    echo "✓ 已激活 $1 · $(python -V 2>&1) · $CONDA_PREFIX"
}
# 每个大的 pip 步骤之后调一次:谁改了 torch 就停在谁那一步,而不是等到 flash-attn
chk_torch () {
    python -c "import sys,torch; ok=torch.__version__.startswith('2.9.1') and torch.version.cuda=='12.8'; print('  [$1] torch',torch.__version__,'cuda',torch.version.cuda,'OK' if ok else 'BAD'); sys.exit(0 if ok else 1)" \
        || die "torch 被「$1」那一步改掉了 —— 看上面这行的实际版本"
}
mk_env () {   # $1 = 环境名
    conda create -n "$1" python==$PY -y || die "conda create $1 失败"
    have_env "$1" || die "conda create $1 报成功但环境不存在"
}

# =============================================================================
step "0. conda + 系统依赖"
# =============================================================================
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$TMPDIR/mc.sh"
    bash "$TMPDIR/mc.sh" -b -p "$CONDA_DIR" && rm -f "$TMPDIR/mc.sh"
fi
export PATH="$CONDA_DIR/bin:$PATH"
eval "$(conda shell.bash hook)"

# ⚠️ conda ≥26 不接受 ToS 就无法建环境(CondaToSNonInteractiveError),
#    而且它只在 stderr 里报,create 返回非零后如果不检查就会一路错下去。
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true
conda config --set always_yes true >/dev/null 2>&1 || true
# 自检:能不能真的建环境
conda create -n _tos_probe python==$PY -y >/dev/null 2>&1 \
    && conda env remove -n _tos_probe -y >/dev/null 2>&1 \
    || die "conda 仍然建不了环境(ToS 或 channel 配置)。手动跑一遍 conda create 看报错"
echo "✓ conda 可以建环境"

command -v wget >/dev/null || (apt-get update -qq && apt-get install -y -qq wget)
command -v git  >/dev/null || (apt-get update -qq && apt-get install -y -qq git)
conda --version; nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

# =============================================================================
step "1. 克隆仓库(以 GitHub 为准)"
# =============================================================================
cd "$REPO_DIR"
[ -d SpaceTools ]           || git clone --depth 1 https://github.com/spacetools/SpaceTools.git
[ -d SpaceTools-RL ]        || git clone --depth 1 https://github.com/ChicyChen/SpaceTools-RL.git
[ -d SpaceTools-Toolshed ]  || git clone --depth 1 https://github.com/NVlabs/SpaceTools-Toolshed.git
for r in SpaceTools SpaceTools-RL SpaceTools-Toolshed; do
    echo "$r  $(git -C $r rev-parse HEAD)"
done

# =============================================================================
step "2. spacetools-rl(照 setup_envs.sh 的 setup_rl,补三处)"
# =============================================================================
if have_env spacetools-rl; then
    echo "已存在,跳过"
else
mk_env spacetools-rl
act spacetools-rl

# 2.1 torch —— ⚠️ 偏离①(两处):
#   a) 上游 `pip install torch==2.9.1 torchvision torchaudio` 不钉后两个,
#      SFT 那轮因此拿到给 torch 2.11 编的 torchaudio,import 时 undefined symbol。
#   b) 上游没有 --index-url,走 PyPI 默认 wheel。论文写作时 torch 2.9.1 的默认是
#      cu128,**现在默认已变成 cu130** —— 于是 nvcc 12.8 编 flash-attn 时报
#      "detected CUDA 12.8 mismatches the version used to compile PyTorch 13.0"。
#      必须钉 cu128:SFT checkpoint 是 torch 2.9.1+cu128 训出来的,P4/P5 的
#      eval 环境也是 cu128。迁就 cu130 会让整条链的口径分家,而且不报错。
pip install torch==$TORCH torchvision==$TORCHVISION torchaudio==$TORCHAUDIO \
    --index-url https://download.pytorch.org/whl/cu128
pip install transformers==$TRANSFORMERS

# ⚠️ 偏离①c:约束文件,让后续任何 pip install 都不能改动这四个。
#    实测:flashinfer_python 的最新版要求 torch > 2.9.1,pip 会**静默**把 torch
#    连同整套 nvidia-*-cu13 升到 2.14.0+cu130(build.fail2 就是这么来的)。
#    有了约束,冲突会变成一条明确的 pip 报错,而不是一个悄悄变了口径的环境。
cat > /root/constraints.txt <<EOF
torch==$TORCH
torchvision==$TORCHVISION
torchaudio==$TORCHAUDIO
transformers==$TRANSFORMERS
EOF
export PIP_CONSTRAINT=/root/constraints.txt
echo "PIP_CONSTRAINT=$PIP_CONSTRAINT"; cat "$PIP_CONSTRAINT"

# 2.2 sglang —— ⚠️ 偏离④:上游写的是
#       pip install "sglang[srt,openai]==0.5.6" --no-deps
#       pip install <手工列的一长串依赖>
#     用 --no-deps 是为了防止 sglang 顶掉 torch/transformers。但那份手工表**不全**:
#       ModuleNotFoundError: No module named 'pybase64'
#     而且手动装的 flashinfer_python(未钉版本)会把 nvidia-cutlass-dsl 顶到
#     4.8.0.dev0,而 sglang 0.5.6 要求 ==4.2.1。
#     我们有 PIP_CONSTRAINT 守 torch/transformers,所以可以让 pip 正常解析
#     sglang 的真实依赖闭包 —— 版本由 sglang 自己决定,比手工表准。
pip install "sglang[srt,openai]==0.5.6" || die "sglang 依赖解析失败(看是否与 torch 2.9.1 冲突)"
# 上游手工表里 sglang 闭包之外的补充项(不含 flashinfer_python —— 让 sglang 定它的版本)
pip install anthropic openai blobfile decord2 torchao \
    lm-format-enforcer cuda-python sglang_router

chk_torch "sglang依赖"

pip install "ray[default]==$RAY"
chk_torch "ray"

# 2.3 verl
cd "$REPO_DIR/SpaceTools-RL" && pip install -e "." --no-deps
pip install accelerate codetiming datasets dill hydra-core \
    "numpy<2.0.0" pandas peft "pyarrow>=19.0.0" pybind11 pylatexenc \
    torchdata "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
    wandb tensorboard packaging

chk_torch "verl依赖"

# 2.4 toolshed
cd "$REPO_DIR/SpaceTools-Toolshed" && pip install -e . --no-deps
pip install docstring_parser aiohttp aiohttp-cors Pillow \
    fastapi uvicorn python-multipart openai botocore pyyaml \
    anthropic matplotlib scipy requests click uvloop

chk_torch "toolshed依赖"

# 2.4b ⚠️ 守卫:上面那一长串依赖里有 torchao / flashinfer_python 等会依赖 torch 的包,
#      pip 可能把 torch 重新解析成别的 build。编译 flash-attn 之前必须确认还是 cu128,
#      否则报的错(CUDA version mismatch)离真正的原因很远。
chk_torch "编译前最终确认"
# sglang / flashinfer 装在这个 torch 上还能不能 import —— 约束住 torch 之后,
# 真正的风险从"torch 被换掉"变成"flashinfer 与 2.9.1 不兼容",要单独验。
python -c "import sglang, flashinfer; print('  sglang', sglang.__version__, '· flashinfer OK')" \
    || die "sglang/flashinfer import 失败 —— 看上面的 ModuleNotFoundError 是缺包(依赖闭包不全)还是 ABI 不兼容(torch 版本)"
pip check 2>&1 | head -10 || true   # 冲突只报告不阻断:上游本来就有若干宽松约束

# 2.5 nvcc —— flash-attn 要编译
if ! nvcc --version 2>/dev/null | grep -q "release 12.8"; then
    conda install -c nvidia cuda-toolkit=12.8 -y 2>&1 | tail -2
fi
export CUDA_HOME="$CONDA_PREFIX"; export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version | tail -1

# 2.6 flash-attn —— ⚠️ 偏离②:上游写的是
#       pip install flash-attn --no-build-isolation 2>/dev/null || echo "WARNING: ..."
#     错误被吞、失败也继续。这里让它失败就停。
echo "编译 flash-attn(FLASH_ATTN_CUDA_ARCHS=$FLASH_ATTN_CUDA_ARCHS,20-45 min)..."
pip install flash-attn --no-build-isolation || { echo "✗ flash-attn 编译失败,停"; exit 1; }

pip install qwen_vl_utils

# 2.7 ⚠️ 偏离③:cudnn 二次升级 + cachetools。
#     setup_envs.sh 没有这两步,但 docs/SETUP.md 的手动路径有,
#     且 P1 记录确认必需(flash-attn 会把 cudnn 降级)。
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
step "3. 四个工具环境(Toolshed setup_tool_env.sh)"
# =============================================================================
act spacetools-rl        # setup_tool_env.sh 从当前激活环境探测 Python + Ray
cd "$REPO_DIR/SpaceTools-Toolshed"

mk_tool_env () {   # $1 = env 名, $2 = tool 名
    if have_env "$1"; then echo "== $1 已存在,跳过"; return; fi
    step "3.x $1  ($2)"
    ( set +u; source install_tools/setup_tool_env.sh "$1" "$2" ) \
        || die "$1 建立失败(setup_tool_env.sh 非零退出)"
    have_env "$1" || die "$1 报成功但环境不存在"
    echo "✓ $1 建好了"
}

# 顺序:最轻的先,最容易翻车的第二 —— graspgen 的 pointnet2_ops 要编 CUDA 扩展,
# 而它是唯一"这四个 eval key 完全用不到、却必须装"的环境(schema 必须是 11 个)。
mk_tool_env spacetools-tool-bbox      bbox
mk_tool_env spacetools-tool-graspgen  graspgen
mk_tool_env spacetools-tool-vlm       vlm
mk_tool_env spacetools-tool-roborefer roborefer

# ⚠️ 偏离④(其实是补 docs/SETUP.md 的漏):
#    run_eval.sh 把 sam2 和 depth_estimator 也放在 spacetools-tool-vlm 里
#      'sam2':            'conda_env': 'spacetools-tool-vlm'
#      'depth_estimator': 'conda_env': 'spacetools-tool-vlm'
#    但 setup_tool_env.sh <env> vlm 只装 tool-vlm.txt(Molmo 的依赖)。
#    照文档做会得到一个跑不了 sam2/depth 的环境,而且要到 actor 调用时才报错。
#    注:tool-depth.txt 头部注释说"需要 numpy<2,必须单独环境" —— 陈旧,
#        实际 requirements 里根本没有 pin numpy,和 vlm/sam2 的 numpy>2 不冲突。
step "3.5 往 spacetools-tool-vlm 追加 sam2 + depth 依赖"
act spacetools-tool-vlm
bash install_tools/tool_scripts/install_sam2.sh  || echo "✗ sam2 依赖失败"
bash install_tools/tool_scripts/install_depth.sh || echo "✗ depth 依赖失败"
conda deactivate

# =============================================================================
step "4. 权重(放网络卷,跨 pod 保留;大文件顺序写 486 MB/s,不受小文件慢的影响)"
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
# 评测集
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("siyich/spacetools-eval-benchmarks", repo_type="dataset", max_workers=16)
print("benchmarks ->", p)
PY
# DepthPro 的 ckpt(install_depth.sh 已下过就跳过)
[ -f "$CHECKPOINT_DIR/depth_pro.pt" ] || \
  wget -q --show-progress https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -O "$CHECKPOINT_DIR/depth_pro.pt"
ls -lh "$CHECKPOINT_DIR"
du -sh "$HF_HOME"
conda deactivate

# =============================================================================
step "4b. 冻结版本(repo 只钉了六个包,其余都是当天最新版)"
# =============================================================================
# 上游 setup_envs.sh 对绝大多数依赖只写包名(sglang_router / flashinfer_python /
# torchao / accelerate / peft / ...),pip 解析成当天最新版。论文写作时、P4 建环境时、
# 今天,三者拿到的版本都不同 —— 本次构建的三次失败全部源于此。
# repo 做不到跨时间可复现,但我们可以让"这一次"可复现:把实际装到的版本冻下来,
# 之后两臂 RL 的每次 eval 都用同一份恢复,保证内部自洽。
FREEZE=/workspace/freeze
mkdir -p "$FREEZE"
for env in spacetools-rl spacetools-tool-vlm spacetools-tool-roborefer \
           spacetools-tool-bbox spacetools-tool-graspgen; do
    have_env "$env" || continue
    "$CONDA_DIR/envs/$env/bin/pip" freeze > "$FREEZE/$env.txt" 2>/dev/null
    echo "$env: $(wc -l < "$FREEZE/$env.txt") 个包 → $FREEZE/$env.txt"
done
# 关键版本一览,方便和 P4 的 PROVENANCE 对照
{
  echo "# 本次构建的关键版本  $(date -u +%FT%TZ)"
  echo "# 驱动: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
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
step "5. 架构自检:所有编译出来的 .so 是否含 sm_80"
# =============================================================================
# ⚠️ 之前那版 check_arch 只扫 "*.so",漏掉 libcublas.so.12 这类。这里两种都扫。
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
        echo "$out" | grep -q "sm_8[06]" || { echo "  ✗ 无 sm_80/86: $so"; bad=$((bad+1)); }
    done < <(find "$d" \( -name "*.so" -o -name "*.so.*" \) 2>/dev/null | head -400)
    echo "$env: 扫了 $n 个含 cubin 的 .so,$bad 个缺 sm_80/86"
    BAD=$((BAD+bad))
done
echo "架构自检:$BAD 个问题"

# =============================================================================
step "构建结束"
# =============================================================================
conda env list
df -h / /workspace | grep -Ev "tmpfs|Filesystem"
echo "下一步:bash smoke_tools.sh —— 逐工具冒烟"
