#!/bin/bash
# =============================================================================
# 修 graspgen 第二轮 —— 从 install_graspgen.sh 死掉的地方接着做
#
# 上一轮进展:read 提示绕过了、GraspGen 克隆了、pyproject 打好补丁了、
# cuda-toolkit=12.1 装了、grasp_gen 本体也 build 成功了。
# 唯一的死因:
#     ERROR: Failed building wheel for pickle5
#     Py_SIZE(self->stack) = len;   ← Py_SIZE 在 Python 3.11 起不再是左值
# pickle5 是 Python 3.5–3.7 的 protocol-5 后向移植,3.8 起标准库自带,
# 在 3.11 上根本编不过。GraspGen 里唯一的用处是
#     grasp_gen/dataset/dataset.py:27   import pickle5 as pickle
# 且它在 try/except ImportError 里 —— 删掉依赖会自动回退到标准库 pickle。
# =============================================================================
set -uo pipefail
CONDA_DIR=/opt/conda-st
G=/opt/spacetools/GraspGen
export PATH="$CONDA_DIR/bin:$PATH"
eval "$(conda shell.bash hook)"
die() { echo "✗ $*"; exit 1; }

unset PIP_CONSTRAINT          # 工具环境有自己的 torch(2.3 系),不受 rl 的约束
export TMPDIR=/root/tmp PIP_CACHE_DIR=/root/.cache/pip
export MODELS_DIR=/workspace/checkpoints
export MAX_JOBS=6             # cgroup 46.6 GiB
mkdir -p "$MODELS_DIR" "$TMPDIR"

set +u; conda activate spacetools-tool-graspgen; set -u
[ "${CONDA_DEFAULT_ENV:-}" = spacetools-tool-graspgen ] || die "激活失败"
echo "✓ $CONDA_DEFAULT_ENV · $(python -V 2>&1) · $CONDA_PREFIX"

echo; echo "=== 1. 去掉 pickle5(废弃依赖,3.11 编不过)==="
grep -n '"pickle5"' "$G/pyproject.toml" || echo "  (已经没有了)"
sed -i '/"pickle5",\?/d' "$G/pyproject.toml"
grep -q '"pickle5"' "$G/pyproject.toml" && die "pickle5 没删掉"
echo "  ✓ 已删除"
echo "  确认 dataset.py 的回退分支:"; sed -n '25,31p' "$G/grasp_gen/dataset/dataset.py"

echo; echo "=== 2. pip install -e GraspGen ==="
cd "$G"
pip install -e . --find-links https://data.pyg.org/whl/torch-2.3.0+cu121.html \
    || die "grasp_gen 安装失败"

echo; echo "=== 3. 编译 pointnet2_ops(CUDA 扩展)==="
cd "$G/pointnet2_ops"
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include:${CPATH:-}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/stubs:${LIBRARY_PATH:-}"
TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" pip install --no-build-isolation . \
    || die "pointnet2_ops 编译失败"
cd "$G"

echo; echo "=== 4. numpy 回到 2.x(上游 install_graspgen.sh 的最后一步)==="
pip install "numpy>=2.0" --force-reinstall --no-deps || die "numpy 装不回去"

echo; echo "=== 5. GraspGen 权重 ==="
cd "$MODELS_DIR"
[ -d GraspGenModels ] || git clone https://huggingface.co/adithyamurali/GraspGenModels
du -sh "$MODELS_DIR/GraspGenModels" 2>/dev/null

# =============================================================================
echo; echo "=== 验收(看实际状态,不看退出码)==="
# =============================================================================
E="$CONDA_DIR/envs/spacetools-tool-graspgen"
fail=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "  ✓ $1"; else echo "  ✗ $1"; fail=$((fail+1)); fi; }
chk "ray"                "$E/bin/python -c 'import ray'"
chk "toolshed"           "$E/bin/python -c 'import toolshed'"
chk "torch"              "$E/bin/python -c 'import torch'"
chk "pointnet2_ops"      "$E/bin/python -c 'import pointnet2_ops'"
chk "grasp_gen"          "$E/bin/python -c 'import grasp_gen'"
chk "dataset(pickle 回退)" "$E/bin/python -c 'from grasp_gen.dataset.dataset import collate'"
chk "GraspGenModels 权重" "[ -d $MODELS_DIR/GraspGenModels ]"
echo "  --- 版本 ---"
$E/bin/python -c "import torch,ray,numpy;print('  torch',torch.__version__,'cuda',torch.version.cuda,'| ray',ray.__version__,'| numpy',numpy.__version__)" 2>&1|tail -1
echo "  --- pointnet2_ops 的 cubin 架构 ---"
SO=$(find "$E" -name "pointnet2*.so" 2>/dev/null|head -1)
[ -n "$SO" ] && "$E/bin/../bin/cuobjdump" --list-elf "$SO" 2>/dev/null | head -3 || echo "  (找不到 .so 或 cuobjdump)"

echo
[ "$fail" -eq 0 ] && echo "✓ graspgen 修好了" || echo "✗ 还有 $fail 项不过"
exit "$fail"
