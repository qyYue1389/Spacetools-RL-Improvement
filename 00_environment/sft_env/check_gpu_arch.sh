#!/bin/bash
# 构建期 gate:每个编译出来的 .so 都必须带目标架构的 cubin(或 PTX 回退)。
#
# 这条检查的由来:A100 期 pointnet2_ops 被建成只有 sm_86 的 cubin、没有 PTX 回退,
# 换到 sm_80 直接跑不了。同批扫描的另外 65 个 .so 都带 sm_80,所以活了下来 ——
# 那不是运气,是「cubin 在同一大版本内向上兼容」这条规则。
#
# 下限取 sm_80(A100)。为它编的能在 8.6(A6000/A4000)、8.9(L40S)上跑。

set -uo pipefail
TARGET="${TARGET_SM:-sm_80}"
ROOT="${1:-/opt/conda/envs}"

command -v cuobjdump >/dev/null || { echo "跳过:找不到 cuobjdump"; exit 0; }

bad=0; checked=0
while read -r so; do
    out=$(cuobjdump --list-elf "$so" 2>/dev/null) || continue
    [ -z "$out" ] && continue                       # 不含 CUDA,跳过
    checked=$((checked+1))
    if ! grep -q "$TARGET" <<<"$out"; then
        # 有 PTX 也算过 —— 可以 JIT
        if cuobjdump --list-ptx "$so" 2>/dev/null | grep -q "ptx"; then
            echo "  PTX 回退: $so"
        else
            echo "  ✗ 缺 $TARGET 且无 PTX: $so"
            bad=$((bad+1))
        fi
    fi
done < <(find "$ROOT" -name "*.so" 2>/dev/null)

echo "扫描了 $checked 个含 CUDA 的 .so,$bad 个不合格"
[ "$bad" -eq 0 ] || { echo "构建失败:上面那些换卡就跑不了"; exit 1; }
echo "✓ 全部带 $TARGET 或 PTX 回退"
