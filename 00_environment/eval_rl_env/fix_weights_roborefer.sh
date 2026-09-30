#!/bin/bash
# =============================================================================
# 修三件事,都是"看起来建好了、其实是空的"
#
# ① GraspGenModels 的 .pth 全是 134 字节的 git-lfs 指针
#    —— 克隆时没有 git-lfs,拿到的是指针不是权重。
# ② /workspace/checkpoints/RoboRefer-8B-SFT 只有一个 .cache 空目录
#    —— install_roborefer.sh 里 `hf download` 中断,而它紧跟着 `exit 1`,
#       于是**后面克隆 RoboRefer 仓库那一步根本没执行**,llava 也就没装。
#       真正的 16 GB 权重其实在 HF 缓存里(我的构建阶段 4 下过),只是没落到这里。
# ③ depth / grasp 两个工具默认去 SpaceTools-Toolshed/checkpoints 找权重,
#    而我们放在 /workspace/checkpoints —— 做个软链接让两边一致。
#
# 第 ② 条是第十四个上游洞:Toolshed README 第 100 行写着 RoboRefer
# "requires cloning a fork of RoboRefer",但那一步被放在一个会 exit 1 的下载之后。
# =============================================================================
set -uo pipefail
CONDA_DIR=/opt/conda-st
TOOLSHED=/opt/spacetools/SpaceTools-Toolshed
CKPT=/workspace/checkpoints
export PATH="$CONDA_DIR/bin:$PATH"; eval "$(conda shell.bash hook)"
export HF_HOME=/workspace/hf TMPDIR=/root/tmp PIP_CACHE_DIR=/root/.cache/pip
export CHECKPOINT_DIR="$CKPT"
unset PIP_CONSTRAINT
die(){ echo "✗ $*"; exit 1; }

echo "############ 0. git-lfs ############"
command -v git-lfs >/dev/null || (apt-get update -qq && apt-get install -y -qq git-lfs)
git lfs install --skip-repo || die "git-lfs 装不上"
git lfs version

echo; echo "############ 1. 重新拉 GraspGenModels(带 lfs)############"
cd "$CKPT"
if [ -d GraspGenModels ]; then
    ( cd GraspGenModels && git lfs pull 2>&1 | tail -3 )
fi
SZ=$(stat -c %s "$CKPT/GraspGenModels/checkpoints/graspgen_franka_panda_gen.pth" 2>/dev/null || echo 0)
if [ "$SZ" -lt 1000000 ]; then
    echo "  lfs pull 没生效($SZ 字节),整个重克隆"
    rm -rf GraspGenModels
    GIT_LFS_SKIP_SMUDGE=0 git clone https://huggingface.co/adithyamurali/GraspGenModels 2>&1|tail -2
    SZ=$(stat -c %s "$CKPT/GraspGenModels/checkpoints/graspgen_franka_panda_gen.pth" 2>/dev/null || echo 0)
fi
echo "  graspgen_franka_panda_gen.pth = $SZ 字节"
[ "$SZ" -gt 1000000 ] || die "GraspGen 权重仍是指针文件"
du -sh "$CKPT/GraspGenModels"

echo; echo "############ 2. 把 RoboRefer 权重落到 checkpoints ############"
conda activate spacetools-tool-roborefer
HFCLI=$(command -v hf || command -v huggingface-cli) || die "没有 hf CLI"
echo "  用 $HFCLI(HF 缓存里已有 16 GB,应该很快)"
$HFCLI download Zhoues/RoboRefer-8B-SFT --local-dir "$CKPT/RoboRefer-8B-SFT" 2>&1 | tail -3
[ -f "$CKPT/RoboRefer-8B-SFT/config.json" ] || die "RoboRefer 权重还是没落地"
du -sh "$CKPT/RoboRefer-8B-SFT"

echo; echo "############ 3. checkpoints 软链接 ############"
if [ ! -e "$TOOLSHED/checkpoints" ]; then ln -s "$CKPT" "$TOOLSHED/checkpoints"; fi
ls -ld "$TOOLSHED/checkpoints"
ls "$TOOLSHED/checkpoints/" | head -5

echo; echo "############ 4. 重跑 install_roborefer.sh(这次会跳过下载,直接克隆+装 llava)############"
cd "$TOOLSHED"
bash install_tools/tool_scripts/install_roborefer.sh 2>&1 | tail -20
echo "  退出码 $?"

echo; echo "############ 验收 ############"
E="$CONDA_DIR/envs/spacetools-tool-roborefer"
fail=0
chk(){ if eval "$2" >/dev/null 2>&1; then echo "  ✓ $1"; else echo "  ✗ $1"; fail=$((fail+1)); fi; }
chk "RoboRefer 仓库已克隆"  "[ -d $TOOLSHED/RoboRefer ]"
chk "llava 可导入"          "$E/bin/python -c 'import llava'"
chk "llava.media"           "$E/bin/python -c 'from llava.media import Image'"
chk "roborefer 工具可导入"  "$E/bin/python -c 'from toolshed.tools.roborefer import RoboreferTool'"
chk "RoboRefer 权重"        "[ -f $CKPT/RoboRefer-8B-SFT/config.json ]"
chk "GraspGen 权重(真)"    "[ \$(stat -c %s $CKPT/GraspGenModels/checkpoints/graspgen_franka_panda_gen.pth) -gt 1000000 ]"
chk "depth_pro.pt"          "[ -f $CKPT/depth_pro.pt ]"
chk "toolshed/checkpoints 链接" "[ -e $TOOLSHED/checkpoints/depth_pro.pt ]"
echo
[ "$fail" -eq 0 ] && echo "✓ 全部修好" || echo "✗ 还有 $fail 项不过"
exit "$fail"
