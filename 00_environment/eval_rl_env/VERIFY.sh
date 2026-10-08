#!/bin/bash
# The real acceptance check after restore. Only when all four items pass is this environment usable.
# Every criterion looks at actual state, not exit codes — this project has been burned by false green lights too many times.
#
# 2026-09-12 revision (based on 03_sft_eval/sft_eval_results.md):
#   + Item 0, a precondition check. The original three items **cannot catch** two real failures:
#       - when the five environments' Python versions differ (3.11.0 vs 3.11.16), Ray refuses to let those environments'
#         actors join the cluster; all three items are green but eval/RL silently misses 8 (23 for RL) actors
#       - when /workspace/logs does not exist, item 3's grep reads nothing yet still exits 0
#   + Item 3 no longer trusts the exit code of 28_chain.sh; it looks for a success marker in the output instead.
#     It has been observed printing "✗ chain not connected" while exiting 0.
#   + Item 2 sets CUDA_HOME automatically. 04_smoke.sh calls the env python directly without activating conda,
#     so roborefer's deepspeed throws MissingCUDAException.
set -uo pipefail
S=/root/pkgstage/scripts
[ -d "$S" ] || S=/root
[ -f "$S/03_verify.sh" ] || { echo "✗ acceptance check script not found; unpack the scripts package first"; exit 1; }
CD="${CONDA_DIR:-/opt/conda-st}"
cd /

echo "==== 0. Precondition checks (if these fail, the green lights of the next three items cannot be trusted) ===="
RC0=0
p0(){ echo "  OK  $*"; }
b0(){ echo "  BAD $*"; RC0=1; }

# 0a Python consistency across the five environments, and, if inconsistent, whether the Ray patch is in place
HEAD=""; MIS=""
for e in spacetools-rl spacetools-tool-vlm spacetools-tool-roborefer \
         spacetools-tool-bbox spacetools-tool-graspgen; do
    px="$CD/envs/$e/bin/python"
    [ -x "$px" ] || { b0 "$e environment does not exist"; continue; }
    v="$("$px" -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])')"
    [ "$e" = spacetools-rl ] && HEAD="$v"
    printf "      %-28s %s\n" "$e" "$v"
done
for e in spacetools-tool-vlm spacetools-tool-roborefer spacetools-tool-bbox spacetools-tool-graspgen; do
    px="$CD/envs/$e/bin/python"
    [ -x "$px" ] || continue
    v="$("$px" -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])')"
    [ "$v" = "$HEAD" ] && continue
    MIS="$MIS $e"
    if "$px" -c "
import inspect, sys, ray._private.node as n
sys.exit(0 if 'minor' in inspect.getsource(n.Node.check_version_info) else 1)" 2>/dev/null; then
        p0 "$e Python($v) != head node ($HEAD), but the Ray minor-level patch is in place"
    else
        b0 "$e Python($v) != head node ($HEAD) and no Ray patch — this environment's actors cannot join the cluster; run POSTRESTORE.sh first"
    fi
done
[ -n "$MIS" ] || p0 "Python versions of the five environments match ($HEAD)"

# 0b CUDA_HOME for roborefer (deepspeed reads it at import time)
RR="$CD/envs/spacetools-tool-roborefer"
if [ -n "${CUDA_HOME:-}" ]; then
    p0 "CUDA_HOME already set by the caller ($CUDA_HOME)"
elif [ -x "$RR/bin/nvcc" ]; then
    export CUDA_HOME="$RR"
    p0 "CUDA_HOME not set, pointing it to $RR automatically"
else
    b0 "no nvcc in the roborefer environment and CUDA_HOME not set — item 2 will fail on roborefer"
fi

# 0c Runtime directories
for d in /workspace/logs /workspace/smoke /workspace/checkpoints /workspace/hf; do
    if [ -d "$d" ]; then p0 "$d exists"; else b0 "$d missing (POSTRESTORE.sh creates the first two; for the weights directory see MANIFEST)"; fi
done

# 0d There must be no live Ray cluster, and no leftover address files
if pgrep -x gcs_server >/dev/null 2>&1 || pgrep -x raylet >/dev/null 2>&1; then
    b0 "Ray processes are running — item 3's ray.init(num_cpus=...) will be rejected. Run ray stop --force first"
else
    STALE=""
    for f in /root/tmp/ray/ray_current_cluster /tmp/ray/ray_current_cluster; do
        [ -f "$f" ] && STALE="$STALE $f"
    done
    if [ -n "$STALE" ]; then
        b0 "leftover cluster address files: $STALE — item 3 will think the cluster is still up. Delete them and rerun"
    else
        p0 "no live Ray cluster and no leftover address files"
    fi
fi
[ "$RC0" -eq 0 ] || { echo; echo "✗ precondition checks failed — stop here, do not look at the results of the next three items"; exit 1; }

echo
echo "==== 1. Five-environment import gate + .so architecture scan ===="
bash "$S/03_verify.sh"; RC1=$?
echo
echo "==== 2. Seven-tool smoke test (really loads weights, really produces results) ===="
bash "$S/04_smoke.sh"; RC2=$?
echo
echo "==== 3. Ray cross-environment tool chain (variables passed across conda environments) ===="
CHAIN_OUT="$(bash "$S/28_chain.sh" 2>&1)"; RC3RAW=$?
echo "$CHAIN_OUT"
# Do not trust the exit code: the success marker must appear in the output, and no failure marker (28_chain.sh prints these markers in Chinese, so they are kept verbatim)
if echo "$CHAIN_OUT" | grep -q "跨环境工具链通了" && ! echo "$CHAIN_OUT" | grep -q "链没通"; then
    RC3=0
else
    RC3=1
    echo "  (28_chain.sh exit code is $RC3RAW, but there is no success marker in the output — counted as a failure)"
fi

echo
echo "==== Summary ===="
echo "  Precondition checks   exit code $RC0  (0=Python consistent or patch in place, directories present, no leftover Ray)"
echo "  Environment check     exit code $RC1  (0=imports pass in all five environments and all self-built extensions contain sm_8x)"
echo "  Seven-tool smoke test exit code $RC2  (0=all seven really produced results, and no Error: in the results)"
echo "  Cross-environment chain exit code $RC3  (0=output confirms ndarrays passed correctly both ways across conda environments)"
if [ "$RC0" -eq 0 ] && [ "$RC1" -eq 0 ] && [ "$RC2" -eq 0 ] && [ "$RC3" -eq 0 ]; then
    echo "✓ all four items pass, environment usable"
else
    echo "✗ some items failed — do not run eval in this state, the scores will be wrong"
    exit 1
fi
