#!/bin/bash
# Run the GFlowRL self-checks against a full SpaceTools-RL checkout with our patches applied.
# 跑 GFlowRL 的六个自检。这些脚本用 ast / 正则从真实源文件里抠出函数再执行,
# 所以必须指向一份完整的、已打上 patches_spacetools_rl/ 的 SpaceTools-RL。
#
# Usage:  SPACETOOLS_RL=/path/to/SpaceTools-RL bash run_checks.sh
#         (default: ../../../SpaceTools-RL, i.e. cloned next to this repo)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export SPACETOOLS_RL="${SPACETOOLS_RL:-$(cd ../../.. && pwd)/SpaceTools-RL}"
[ -d "$SPACETOOLS_RL/verl" ] || { echo "SpaceTools-RL not found at $SPACETOOLS_RL; set SPACETOOLS_RL=..." >&2; exit 1; }
echo "SpaceTools-RL: $SPACETOOLS_RL"; echo

# 前四个是纯文本检查;后两个要 torch —— 没有就 SKIP,不是 FAIL。
HAS_TORCH=1; python3 -c "import torch" 2>/dev/null || HAS_TORCH=0
fail=0
for s in p7_config_check p7_guard_check p7_gpusplit_check p7_onpolicy_check p7_fixedpoint_check p7_degenerate_check; do
    echo "=============== $s ==============="
    if [ "$HAS_TORCH" = 0 ] && { [ $s = p7_fixedpoint_check ] || [ $s = p7_degenerate_check ]; }; then
        echo "SKIP (no torch). 改过 compute_gflowrl_flow_gap / compute_policy_loss_gflowrl 之后必须跑。"
    else
        python3 "$s.py" || { fail=1; echo ">>> $s FAILED"; }
    fi
    echo
done
exit $fail
