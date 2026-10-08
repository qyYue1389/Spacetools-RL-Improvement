#!/bin/bash
# =============================================================================
# Fix spacetools-tool-graspgen — it is the only tool environment that did not install
#
# Three causes:
#  ① install_graspgen.sh has two interactive reads (GRASPGEN_DIR / MODELS_DIR);
#     non-interactively, EOF + set -e exits straight away, and the log stops at the "Enter directory ..." line.
#  ② PIP_CONSTRAINT is exported globally and pins torch to 2.9.1; while GraspGen's
#     pyproject is changed by install_graspgen.sh to torch>=2.3.0,<2.4 — a guaranteed conflict.
#     **The constraints should only govern spacetools-rl; tool environments each have their own torch anyway.**
#  ③ The old success criterion only checked whether the environment exists, so an empty shell also counted as "✓ built".
# =============================================================================
set -uo pipefail
CONDA_DIR=/opt/conda-st
REPO_DIR=/opt/spacetools
export PATH="$CONDA_DIR/bin:$PATH"
eval "$(conda shell.bash hook)"
die() { echo "✗ $*"; exit 1; }

# ⚠️ Key: tool environments must not inherit the rl constraints
unset PIP_CONSTRAINT

# ⚠️ Answers to the two reads (without them it exits on EOF)
export GRASPGEN_DIR="$REPO_DIR"
export MODELS_DIR=/workspace/checkpoints
export TMPDIR=/root/tmp
export PIP_CACHE_DIR=/root/.cache/pip
export HF_HOME=/workspace/hf
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
export MAX_JOBS=6            # cgroup limit is 46.6 GiB, do not get OOM-killed again
mkdir -p "$MODELS_DIR" "$TMPDIR"

echo "=== delete the empty-shell environment and rebuild ==="
conda env remove -n spacetools-tool-graspgen -y >/dev/null 2>&1
conda env list | grep -q "^spacetools-tool-graspgen " && die "old environment was not deleted"

set +u; conda activate spacetools-rl; set -u   # setup_tool_env.sh detects python+ray from here
[ "${CONDA_DEFAULT_ENV:-}" = spacetools-rl ] || die "failed to activate spacetools-rl"
echo "PIP_CONSTRAINT=${PIP_CONSTRAINT:-<cleared>}  GRASPGEN_DIR=$GRASPGEN_DIR  MODELS_DIR=$MODELS_DIR"

cd "$REPO_DIR/SpaceTools-Toolshed"
( set +u; source install_tools/setup_tool_env.sh spacetools-tool-graspgen graspgen )
echo "setup_tool_env.sh exit code: $?"

# =============================================================================
echo; echo "=== real acceptance check (look at actual state, not the exit code) ==="
# =============================================================================
E="$CONDA_DIR/envs/spacetools-tool-graspgen"
fail=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "  ✓ $1"; else echo "  ✗ $1"; fail=$((fail+1)); fi; }

chk "environment exists"        "[ -d $E ]"
chk "ray installed"        "$E/bin/python -c 'import ray'"
chk "toolshed installed"   "$E/bin/python -c 'import toolshed'"
chk "torch installed"      "$E/bin/python -c 'import torch'"
chk "pointnet2_ops"   "$E/bin/python -c 'import pointnet2_ops'"
chk "grasp_gen package"    "$E/bin/python -c 'import grasp_gen'"
chk "GraspGen weights"   "[ -d $MODELS_DIR/GraspGenModels ]"

echo "  --- versions ---"
$E/bin/python -c "import torch,ray;print('  torch',torch.__version__,'cuda',torch.version.cuda,'| ray',ray.__version__)" 2>&1 | tail -1

echo
[ "$fail" -eq 0 ] && echo "✓ graspgen environment fixed" || echo "✗ $fail item(s) still failing"
exit "$fail"
