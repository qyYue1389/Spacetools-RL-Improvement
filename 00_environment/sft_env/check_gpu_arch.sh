#!/bin/bash
# Build-time gate: every compiled .so must carry a cubin for the target architecture (or a PTX fallback).
#
# Where this check comes from: in the A100 period, pointnet2_ops was built with only an sm_86 cubin and no PTX fallback,
# and on sm_80 it simply would not run. The other 65 .so in the same scan all carried sm_80, so they survived —
# that was not luck, it was the rule "cubins are forward-compatible within the same major version".
#
# The floor is sm_80 (A100). What is built for it runs on 8.6 (A6000/A4000) and 8.9 (L40S).

set -uo pipefail
TARGET="${TARGET_SM:-sm_80}"
ROOT="${1:-/opt/conda/envs}"

command -v cuobjdump >/dev/null || { echo "skipping: cannot find cuobjdump"; exit 0; }

bad=0; checked=0
while read -r so; do
    out=$(cuobjdump --list-elf "$so" 2>/dev/null) || continue
    [ -z "$out" ] && continue                       # no CUDA, skip
    checked=$((checked+1))
    if ! grep -q "$TARGET" <<<"$out"; then
        # PTX also counts as a pass — it can be JIT-compiled
        if cuobjdump --list-ptx "$so" 2>/dev/null | grep -q "ptx"; then
            echo "  PTX fallback: $so"
        else
            echo "  ✗ missing $TARGET and no PTX: $so"
            bad=$((bad+1))
        fi
    fi
done < <(find "$ROOT" -name "*.so" 2>/dev/null)

echo "scanned $checked .so containing CUDA, $bad failed"
[ "$bad" -eq 0 ] || { echo "build failed: the ones above will not run on a different GPU"; exit 1; }
echo "✓ all carry $TARGET or a PTX fallback"
