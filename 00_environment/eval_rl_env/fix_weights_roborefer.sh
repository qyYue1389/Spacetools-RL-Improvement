#!/bin/bash
# =============================================================================
# Fix three things, all of the "looks built, but is actually empty" kind
#
# ① GraspGenModels' .pth files are all 134-byte git-lfs pointers
#    — there was no git-lfs at clone time, so we got pointers, not weights.
# ② /workspace/checkpoints/RoboRefer-8B-SFT has only an empty .cache directory
#    — `hf download` in install_roborefer.sh was interrupted, and it is immediately followed by `exit 1`,
#       so **the later step that clones the RoboRefer repo never ran at all**, and llava was never installed.
#       The real 16 GB of weights are actually in the HF cache (downloaded in stage 4 of my build), they just never landed here.
# ③ The depth / grasp tools look for weights in SpaceTools-Toolshed/checkpoints by default,
#    while we keep them in /workspace/checkpoints — make a symlink so both sides agree.
#
# Item ② is the fourteenth upstream hole: line 100 of the Toolshed README says RoboRefer
# "requires cloning a fork of RoboRefer", but that step was placed after a download that can exit 1.
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
git lfs install --skip-repo || die "cannot install git-lfs"
git lfs version

echo; echo "############ 1. Re-fetch GraspGenModels (with lfs) ############"
cd "$CKPT"
if [ -d GraspGenModels ]; then
    ( cd GraspGenModels && git lfs pull 2>&1 | tail -3 )
fi
SZ=$(stat -c %s "$CKPT/GraspGenModels/checkpoints/graspgen_franka_panda_gen.pth" 2>/dev/null || echo 0)
if [ "$SZ" -lt 1000000 ]; then
    echo "  lfs pull did not take effect ($SZ bytes), re-cloning the whole thing"
    rm -rf GraspGenModels
    GIT_LFS_SKIP_SMUDGE=0 git clone https://huggingface.co/adithyamurali/GraspGenModels 2>&1|tail -2
    SZ=$(stat -c %s "$CKPT/GraspGenModels/checkpoints/graspgen_franka_panda_gen.pth" 2>/dev/null || echo 0)
fi
echo "  graspgen_franka_panda_gen.pth = $SZ bytes"
[ "$SZ" -gt 1000000 ] || die "GraspGen weights are still a pointer file"
du -sh "$CKPT/GraspGenModels"

echo; echo "############ 2. Put the RoboRefer weights into checkpoints ############"
conda activate spacetools-tool-roborefer
HFCLI=$(command -v hf || command -v huggingface-cli) || die "no hf CLI"
echo "  using $HFCLI (the HF cache already has 16 GB, should be fast)"
$HFCLI download Zhoues/RoboRefer-8B-SFT --local-dir "$CKPT/RoboRefer-8B-SFT" 2>&1 | tail -3
[ -f "$CKPT/RoboRefer-8B-SFT/config.json" ] || die "RoboRefer weights still did not land"
du -sh "$CKPT/RoboRefer-8B-SFT"

echo; echo "############ 3. checkpoints symlink ############"
if [ ! -e "$TOOLSHED/checkpoints" ]; then ln -s "$CKPT" "$TOOLSHED/checkpoints"; fi
ls -ld "$TOOLSHED/checkpoints"
ls "$TOOLSHED/checkpoints/" | head -5

echo; echo "############ 4. Rerun install_roborefer.sh (this time it skips the download and goes straight to clone + install llava) ############"
cd "$TOOLSHED"
bash install_tools/tool_scripts/install_roborefer.sh 2>&1 | tail -20
echo "  exit code $?"

echo; echo "############ Acceptance check ############"
E="$CONDA_DIR/envs/spacetools-tool-roborefer"
fail=0
chk(){ if eval "$2" >/dev/null 2>&1; then echo "  ✓ $1"; else echo "  ✗ $1"; fail=$((fail+1)); fi; }
chk "RoboRefer repo cloned"  "[ -d $TOOLSHED/RoboRefer ]"
chk "llava importable"          "$E/bin/python -c 'import llava'"
chk "llava.media"           "$E/bin/python -c 'from llava.media import Image'"
chk "roborefer tool importable"  "$E/bin/python -c 'from toolshed.tools.roborefer import RoboreferTool'"
chk "RoboRefer weights"        "[ -f $CKPT/RoboRefer-8B-SFT/config.json ]"
chk "GraspGen weights (real)"    "[ \$(stat -c %s $CKPT/GraspGenModels/checkpoints/graspgen_franka_panda_gen.pth) -gt 1000000 ]"
chk "depth_pro.pt"          "[ -f $CKPT/depth_pro.pt ]"
chk "toolshed/checkpoints link" "[ -e $TOOLSHED/checkpoints/depth_pro.pt ]"
echo
[ "$fail" -eq 0 ] && echo "✓ all fixed" || echo "✗ $fail item(s) still failing"
exit "$fail"
