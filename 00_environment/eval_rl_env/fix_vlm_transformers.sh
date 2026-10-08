#!/bin/bash
# =============================================================================
# Fix spacetools-tool-vlm missing transformers
#
# Root cause (my own design error, second occurrence):
#   PIP_CONSTRAINT pins transformers to 4.57.1 and is exported globally,
#   while tool-vlm.txt hard-pins transformers==4.53.2 — conflict → cannot install.
#   The first occurrence was graspgen's torch (it wants >=2.3,<2.4).
#   **The constraints should only govern spacetools-rl; each tool environment has its own versions, by design.**
#
# Upstream pins 4.53.2 for vlm for a reason: Molmo goes through trust_remote_code,
# and its remote code is usually only compatible with a specific transformers version. So install 4.53.2 as upstream does.
# =============================================================================
set -uo pipefail
CONDA_DIR=/opt/conda-st
export PATH="$CONDA_DIR/bin:$PATH"; eval "$(conda shell.bash hook)"
die(){ echo "✗ $*"; exit 1; }
unset PIP_CONSTRAINT
export TMPDIR=/root/tmp PIP_CACHE_DIR=/root/.cache/pip

set +u; conda activate spacetools-tool-vlm; set -u
[ "${CONDA_DEFAULT_ENV:-}" = spacetools-tool-vlm ] || die "activation failed"
echo "✓ $CONDA_DEFAULT_ENV · PIP_CONSTRAINT=${PIP_CONSTRAINT:-<cleared>}"

echo; echo "=== install transformers==4.53.2 per tool-vlm.txt ==="
pip install "transformers==4.53.2" 2>&1 | tail -4

echo; echo "=== acceptance check (re-check the whole environment, in case the downgrade broke something else) ==="
E="$CONDA_DIR/envs/spacetools-tool-vlm"
fail=0
chk(){ if $E/bin/python -c "import $1" >/dev/null 2>&1; then echo "  ✓ $1"; else echo "  ✗ $1"; fail=$((fail+1)); fi; }
for m in torch torchvision ray toolshed transformers sam2 depth_pro PIL cv2 einops accelerate; do chk "$m"; done
echo "  --- versions ---"
$E/bin/python -c "
import torch,transformers,ray,numpy
print('  torch',torch.__version__,'| transformers',transformers.__version__,'| ray',ray.__version__,'| numpy',numpy.__version__)"
echo "  --- can Molmo's processor be built (no weight download, only the code path) ---"
$E/bin/python - <<'PY' 2>&1 | tail -3
import os
os.environ.setdefault("HF_HOME","/workspace/hf")
try:
    from transformers import AutoProcessor
    p = AutoProcessor.from_pretrained("allenai/Molmo-7B-D-0924", trust_remote_code=True)
    print("  ✓ Molmo processor built successfully:", type(p).__name__)
except Exception as e:
    print("  ✗ Molmo processor failed:", type(e).__name__, str(e)[:160])
PY
echo
[ "$fail" -eq 0 ] && echo "✓ vlm environment fixed" || echo "✗ $fail item(s) still failing"
exit "$fail"
