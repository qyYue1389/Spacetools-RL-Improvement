#!/bin/bash
# =============================================================================
# 五个环境的真实验收 + 版本冻结
#
# ⚠️ 重做架构自检:第一版用 `cuobjdump` 但它不在那个 shell 的 PATH 里,
#    所有调用都失败被跳过,于是"扫了 0 个 .so,0 个问题" —— 一个验证了空集的绿灯。
#    这一版:用每个环境自己的 cuobjdump,扫 *.so 和 *.so.*,并打印实际扫了多少个。
# =============================================================================
set -uo pipefail
CONDA_DIR=/opt/conda-st
FREEZE=/workspace/freeze
ENVS="spacetools-rl spacetools-tool-vlm spacetools-tool-roborefer spacetools-tool-bbox spacetools-tool-graspgen"
mkdir -p "$FREEZE"

echo "############ 1. 每个环境的关键 import ############"
declare -A NEED=(
  [spacetools-rl]="torch verl toolshed flash_attn sglang ray transformers"
  [spacetools-tool-vlm]="torch ray toolshed transformers sam2 depth_pro"
  [spacetools-tool-roborefer]="torch ray toolshed transformers"
  [spacetools-tool-bbox]="ray toolshed numpy cv2"
  [spacetools-tool-graspgen]="torch ray toolshed pointnet2_ops grasp_gen"
)
FAIL=0
for e in $ENVS; do
    P="$CONDA_DIR/envs/$e/bin/python"
    [ -x "$P" ] || { echo "  ✗ $e 不存在"; FAIL=$((FAIL+1)); continue; }
    line="  $e:"
    for m in ${NEED[$e]}; do
        if "$P" -c "import $m" >/dev/null 2>&1; then line="$line $m✓"; else line="$line $m✗"; FAIL=$((FAIL+1)); fi
    done
    echo "$line"
done

echo
echo "############ 2. 架构自检(实扫,不是空集)############"
TOTAL_SO=0; TOTAL_BAD=0
for e in $ENVS; do
    d="$CONDA_DIR/envs/$e"
    CUOBJ="$d/bin/cuobjdump"
    [ -x "$CUOBJ" ] || CUOBJ="$CONDA_DIR/envs/spacetools-rl/bin/cuobjdump"
    [ -x "$CUOBJ" ] || { echo "  $e: 找不到 cuobjdump —— 跳过(这一项没有结论)"; continue; }
    n=0; bad=0; withcubin=0
    while IFS= read -r so; do
        n=$((n+1))
        out=$("$CUOBJ" --list-elf "$so" 2>/dev/null) || continue
        [ -z "$out" ] && continue
        withcubin=$((withcubin+1))
        echo "$out" | grep -qE "sm_8[0-9]" || { echo "    ✗ 无 sm_8x: ${so#$d/}"; bad=$((bad+1)); }
    done < <(find "$d" \( -name "*.so" -o -name "*.so.*" \) -type f 2>/dev/null)
    echo "  $e: 扫了 $n 个 .so,其中 $withcubin 个含 cubin,$bad 个缺 sm_8x"
    TOTAL_SO=$((TOTAL_SO+n)); TOTAL_BAD=$((TOTAL_BAD+bad))
done
echo "  合计:扫了 $TOTAL_SO 个 .so,$TOTAL_BAD 个有问题"
[ "$TOTAL_SO" -gt 0 ] || { echo "  ⚠️ 扫到 0 个文件 —— 这一项没有结论,不要当成通过"; FAIL=$((FAIL+1)); }

echo
echo "############ 3. 冻结版本 ############"
for e in $ENVS; do
    "$CONDA_DIR/envs/$e/bin/pip" freeze > "$FREEZE/$e.txt" 2>/dev/null
    echo "  $e: $(wc -l < "$FREEZE/$e.txt") 个包"
done
{
  echo "# 关键版本  $(date -u +%FT%TZ)"
  echo "# 驱动 $(nvidia-smi --query-gpu=driver_version --format=csv,noheader|head -1) · GPU $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
  for e in $ENVS; do
    echo "## $e"
    grep -iE "^(torch|torchvision|torchaudio|transformers|ray|sglang|flash-attn|flashinfer-python|numpy|nvidia-cudnn-cu12|accelerate|timm|sam-2|depth-pro|pointnet2-ops|grasp-gen)==" \
        "$FREEZE/$e.txt" | sed 's/^/  /'
    NV="$("$CONDA_DIR/envs/$e/bin/nvcc" --version 2>/dev/null | grep -oP 'release \K[0-9.]+')"
    [ -n "$NV" ] && echo "  (nvcc $NV)"
  done
} > "$FREEZE/KEY_VERSIONS.txt"
cat "$FREEZE/KEY_VERSIONS.txt"

echo
echo "############ 结果 ############"
[ "$FAIL" -eq 0 ] && echo "✓ 五个环境全部通过" || echo "✗ $FAIL 项不过"
exit "$FAIL"
