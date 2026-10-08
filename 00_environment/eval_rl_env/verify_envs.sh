#!/bin/bash
# =============================================================================
# Real acceptance check of the five environments + version freeze
#
# ⚠️ Redo of the architecture self-check: the first version used `cuobjdump`, but it was not on that shell's PATH,
#    so every call failed and was skipped, giving "scanned 0 .so, 0 problems" — a green light that verified the empty set.
#    This version: use each environment's own cuobjdump, scan *.so and *.so.*, and print how many were actually scanned.
# =============================================================================
set -uo pipefail
CONDA_DIR=/opt/conda-st
FREEZE=/workspace/freeze
ENVS="spacetools-rl spacetools-tool-vlm spacetools-tool-roborefer spacetools-tool-bbox spacetools-tool-graspgen"
mkdir -p "$FREEZE"

echo "############ 1. Key imports in each environment ############"
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
    [ -x "$P" ] || { echo "  ✗ $e does not exist"; FAIL=$((FAIL+1)); continue; }
    line="  $e:"
    for m in ${NEED[$e]}; do
        if "$P" -c "import $m" >/dev/null 2>&1; then line="$line $m✓"; else line="$line $m✗"; FAIL=$((FAIL+1)); fi
    done
    echo "$line"
done

echo
echo "############ 2. Architecture self-check (real scan, not the empty set) ############"
TOTAL_SO=0; TOTAL_BAD=0
for e in $ENVS; do
    d="$CONDA_DIR/envs/$e"
    CUOBJ="$d/bin/cuobjdump"
    [ -x "$CUOBJ" ] || CUOBJ="$CONDA_DIR/envs/spacetools-rl/bin/cuobjdump"
    [ -x "$CUOBJ" ] || { echo "  $e: cannot find cuobjdump — skipping (no conclusion for this item)"; continue; }
    n=0; bad=0; withcubin=0
    while IFS= read -r so; do
        n=$((n+1))
        out=$("$CUOBJ" --list-elf "$so" 2>/dev/null) || continue
        [ -z "$out" ] && continue
        withcubin=$((withcubin+1))
        echo "$out" | grep -qE "sm_8[0-9]" || { echo "    ✗ no sm_8x: ${so#$d/}"; bad=$((bad+1)); }
    done < <(find "$d" \( -name "*.so" -o -name "*.so.*" \) -type f 2>/dev/null)
    echo "  $e: scanned $n .so, of which $withcubin contain cubin, $bad missing sm_8x"
    TOTAL_SO=$((TOTAL_SO+n)); TOTAL_BAD=$((TOTAL_BAD+bad))
done
echo "  total: scanned $TOTAL_SO .so, $TOTAL_BAD with problems"
[ "$TOTAL_SO" -gt 0 ] || { echo "  ⚠️ scanned 0 files — no conclusion for this item, do not treat it as a pass"; FAIL=$((FAIL+1)); }

echo
echo "############ 3. Freeze versions ############"
for e in $ENVS; do
    "$CONDA_DIR/envs/$e/bin/pip" freeze > "$FREEZE/$e.txt" 2>/dev/null
    echo "  $e: $(wc -l < "$FREEZE/$e.txt") packages"
done
{
  echo "# key versions  $(date -u +%FT%TZ)"
  echo "# driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader|head -1) · GPU $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
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
echo "############ Result ############"
[ "$FAIL" -eq 0 ] && echo "✓ all five environments pass" || echo "✗ $FAIL item(s) failed"
exit "$FAIL"
