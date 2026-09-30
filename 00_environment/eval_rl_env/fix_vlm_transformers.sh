#!/bin/bash
# =============================================================================
# 修 spacetools-tool-vlm 缺 transformers
#
# 根因(我自己的设计错误,第二次发作):
#   PIP_CONSTRAINT 把 transformers 钉死 4.57.1 并导出成全局,
#   而 tool-vlm.txt 硬钉 transformers==4.53.2 —— 冲突 → 装不上。
#   第一次发作是 graspgen 的 torch(它要 >=2.3,<2.4)。
#   **约束只该管 spacetools-rl,工具环境各有各的版本,这是设计如此。**
#
# 上游为 vlm 钉 4.53.2 是有理由的:Molmo 走 trust_remote_code,
# 它的远程代码通常只兼容特定 transformers 版本。所以照上游装 4.53.2。
# =============================================================================
set -uo pipefail
CONDA_DIR=/opt/conda-st
export PATH="$CONDA_DIR/bin:$PATH"; eval "$(conda shell.bash hook)"
die(){ echo "✗ $*"; exit 1; }
unset PIP_CONSTRAINT
export TMPDIR=/root/tmp PIP_CACHE_DIR=/root/.cache/pip

set +u; conda activate spacetools-tool-vlm; set -u
[ "${CONDA_DEFAULT_ENV:-}" = spacetools-tool-vlm ] || die "激活失败"
echo "✓ $CONDA_DEFAULT_ENV · PIP_CONSTRAINT=${PIP_CONSTRAINT:-<已清空>}"

echo; echo "=== 按 tool-vlm.txt 装 transformers==4.53.2 ==="
pip install "transformers==4.53.2" 2>&1 | tail -4

echo; echo "=== 验收(整个环境重验,防止降级打坏别的)==="
E="$CONDA_DIR/envs/spacetools-tool-vlm"
fail=0
chk(){ if $E/bin/python -c "import $1" >/dev/null 2>&1; then echo "  ✓ $1"; else echo "  ✗ $1"; fail=$((fail+1)); fi; }
for m in torch torchvision ray toolshed transformers sam2 depth_pro PIL cv2 einops accelerate; do chk "$m"; done
echo "  --- 版本 ---"
$E/bin/python -c "
import torch,transformers,ray,numpy
print('  torch',torch.__version__,'| transformers',transformers.__version__,'| ray',ray.__version__,'| numpy',numpy.__version__)"
echo "  --- Molmo 的 processor 能不能构起来(不下权重,只看代码路径)---"
$E/bin/python - <<'PY' 2>&1 | tail -3
import os
os.environ.setdefault("HF_HOME","/workspace/hf")
try:
    from transformers import AutoProcessor
    p = AutoProcessor.from_pretrained("allenai/Molmo-7B-D-0924", trust_remote_code=True)
    print("  ✓ Molmo processor 构建成功:", type(p).__name__)
except Exception as e:
    print("  ✗ Molmo processor 失败:", type(e).__name__, str(e)[:160])
PY
echo
[ "$fail" -eq 0 ] && echo "✓ vlm 环境修好了" || echo "✗ 还有 $fail 项不过"
exit "$fail"
