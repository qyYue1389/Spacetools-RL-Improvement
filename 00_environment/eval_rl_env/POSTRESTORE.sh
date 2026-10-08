#!/bin/bash
# Run this after restore and before the acceptance check.
#   bash RESTORE.sh  →  bash POSTRESTORE.sh  →  bash VERIFY.sh
#
# It adds three things the package does not contain but every new machine needs. Idempotent, safe to re-run.
# Every criterion looks at actual state: each fix is read back and asserted right away; if it did not take effect, exit non-zero.
#
# Why these are not in the tar: they were only found during the 2026-09-11 eval, and repacking 22 GB
# just for three lines of changes is not worth it. If the five environments' Python is unified when the environment package is rebuilt, item ①
# will be skipped automatically, and this script can then be deleted.
set -uo pipefail
CD="${CONDA_DIR:-/opt/conda-st}"
ENVS="spacetools-rl spacetools-tool-vlm spacetools-tool-roborefer spacetools-tool-bbox spacetools-tool-graspgen"
BAD=0
step(){ echo; echo "==== $* ===="; }
ok(){   echo "  OK  $*"; }
bad(){  echo "  BAD $*"; BAD=$((BAD+1)); }

step "0. Python versions of the five environments"
HEAD=""      # the cluster head node uses spacetools-rl; use it as the reference
declare -A PYV
for e in $ENVS; do
    p="$CD/envs/$e/bin/python"
    [ -x "$p" ] || { bad "$e does not exist"; continue; }
    v="$("$p" -c 'import sys;print("%d.%d.%d"%sys.version_info[:3])')"
    PYV[$e]="$v"
    printf "  %-28s %s\n" "$e" "$v"
    [ "$e" = spacetools-rl ] && HEAD="$v"
done
[ -n "$HEAD" ] || { echo "✗ cannot read the Python version of spacetools-rl"; exit 1; }

step "1. Ray's Python patch-version check (only for environments that differ from the head node)"
# Ray's check_version_info compares the full version string by default (patch level); 3.11.0 != 3.11.16 will
# raise RuntimeError directly, and none of the actors in that environment can join the cluster — and none of VERIFY's three items
# catch it; it only shows up at eval / RL scale (measured: 8 actors silently missing).
# Ray itself supports the minor level (utils.py: python_version_match_level="minor"), in which case it
# only does logger.warning. The call site in node.py does not pass this parameter; we add it here.
# Safety: both sides have the same bytecode magic and pickle protocol, so code objects / pickles are interchangeable;
# this is exactly the scenario the minor level is designed for. The change only relaxes one version check and changes no numerics.
NEED=""
for e in $ENVS; do
    [ -n "${PYV[$e]:-}" ] || continue
    [ "${PYV[$e]}" = "$HEAD" ] || NEED="$NEED $e"
done
if [ -z "$NEED" ]; then
    ok "Python versions of the five environments match ($HEAD); this patch is not needed"
else
    echo "  differs from the head node ($HEAD): $NEED"
    for e in $NEED; do
        f="$CD/envs/$e/lib/python3.11/site-packages/ray/_private/node.py"
        [ -f "$f" ] || { bad "$e cannot find ray/_private/node.py"; continue; }
        "$CD/envs/spacetools-rl/bin/python" - "$f" <<'PY'
import os, shutil, sys
f = sys.argv[1]
OLD = '''        ray._private.utils.check_version_info(
            cluster_metadata, f"node {node_ip_address}"
        )'''
NEW = '''        ray._private.utils.check_version_info(
            cluster_metadata, f"node {node_ip_address}",
            python_version_match_level="minor"
        )'''
src = open(f).read()
if 'python_version_match_level="minor"' in src:
    print("    already patched, skipping"); sys.exit(0)
if OLD not in src:
    print("    anchor not found — no change made. The ray version may differ; needs manual confirmation"); sys.exit(3)
if not os.path.exists(f + ".orig"):
    shutil.copy2(f, f + ".orig")
open(f, "w").write(src.replace(OLD, NEW, 1))
pc = os.path.join(os.path.dirname(f), "__pycache__", "node.cpython-311.pyc")
if os.path.exists(pc): os.remove(pc)
print("    patched (original file backed up as node.py.orig)")
PY
        rc=$?
        [ "$rc" = 3 ] && { bad "$e anchor does not match"; continue; }
        # Criterion: import ray in that environment and read the source back to confirm
        if "$CD/envs/$e/bin/python" -c "
import inspect, sys, ray._private.node as n
sys.exit(0 if 'minor' in inspect.getsource(n.Node.check_version_info) else 1)" 2>/dev/null; then
            ok "$e minor level in effect (confirmed by read-back)"
        else
            bad "$e patch did not take effect"
        fi
    done
fi

step "2. CUDA_HOME for the roborefer environment"
# llava's inference path has a module-level hard dependency on deepspeed (see environment README hole #17), and deepspeed
# reads CUDA_HOME at import time. The packaging machine has a system /usr/local/cuda-12.8, which hits torch's
# third fallback; machines with only the driver installed do not, so 04_smoke.sh (calls the env python directly,
# without activating conda) throws MissingCUDAException.
# The real eval goes through Ray's runtime_env={"conda":...}, which activates conda and resolves it anyway;
# this activate.d is insurance for the "activate" path, and also lets the smoke test script pass.
RR="$CD/envs/spacetools-tool-roborefer"
if [ -x "$RR/bin/nvcc" ]; then
    A="$RR/etc/conda/activate.d"
    mkdir -p "$A"
    printf 'export CUDA_HOME="$CONDA_PREFIX"\n' > "$A/zz_cuda_home.sh"
    if grep -q 'CUDA_HOME' "$A/zz_cuda_home.sh"; then
        ok "activate.d/zz_cuda_home.sh in place ($("$RR/bin/nvcc" --version | tail -2 | head -1 | tr -s ' '))"
    else
        bad "failed to write activate.d"
    fi
else
    bad "$RR/bin/nvcc does not exist — this environment should ship its own nvcc; first check whether the environment restore is complete"
fi

step "3. Directories needed at runtime"
# 28_chain.sh writes its raw output to /workspace/logs/chain_raw.log, and 04_smoke.sh writes to
# /workspace/smoke. If the directories do not exist, every grep in 28_chain.sh reads nothing, yet it still exits 0
# — a false green light. MANIFEST only mentions checkpoints and hf, not these two.
for d in /workspace/logs /workspace/smoke; do
    mkdir -p "$d" 2>/dev/null || sudo -n mkdir -p "$d"
    if [ -w "$d" ]; then ok "$d writable"; else bad "$d not writable"; fi
done

step "Summary"
if [ "$BAD" -eq 0 ]; then
    echo "✓ all in place. Next, run VERIFY.sh"
else
    echo "✗ $BAD item(s) failed — do not go on to run eval"
    exit 1
fi
