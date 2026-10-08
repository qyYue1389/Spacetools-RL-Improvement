#!/bin/bash
# Acceptance check after restore. If any item fails, the restore is not good — do not look at the exit code, look at every line.
set -uo pipefail
export PATH=/opt/conda-st/bin:$PATH
eval "$(conda shell.bash hook)"
conda activate spacetools-sft

fail=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "  ✓ $1"; else echo "  ✗ $1"; fail=$((fail+1)); fi; }

echo "=== 1. Paths ==="
chk "/opt/conda-st exists"             '[ -d /opt/conda-st ]'
chk "/workspace/SpaceTools-SFT exists" '[ -d /workspace/SpaceTools-SFT ]'
chk "editable pointer not broken"              '[ -f /workspace/SpaceTools-SFT/src/llamafactory/__init__.py ]'

echo "=== 2. Hardware ==="
nvidia-smi --query-gpu=index,name,driver_version,compute_cap,memory.total --format=csv 2>&1 | sed 's/^/  /'
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk 'NR==1{print $1}')
case "$CC" in
  8.0|8.6) echo "  ✓ compute_cap $CC — has an sm_80 cubin, can run directly" ;;
  *) echo "  ✗ compute_cap $CC — this package has only an sm_80 cubin and no PTX fallback, flash-attn cannot run"; fail=$((fail+1)) ;;
esac

echo "=== 3. Package versions ==="
# ⚠️ This section once missed torchaudio, and as a result a defective package went out.
# Cause: upstream setup_envs.sh only pins torch, and pip resolved torchaudio to 2.11.0 (built for torch 2.11),
# which gives "undefined symbol: torch_library_impl" at import.
# Just importing llamafactory does not catch it — it is lazy, and only imports torchaudio at mm_plugin.
python - <<'PYEOF'
import sys
exp = {"torch":"2.9.1+cu128", "torchvision":"0.24.1+cu128", "torchaudio":"2.9.1+cu128",
       "transformers":"4.57.1", "flash_attn":"2.8.3.post1"}
bad = 0
for k, v in exp.items():
    try:
        m = __import__(k)
    except Exception as e:
        print(f"  ✗ {k:14s} import failed: {type(e).__name__}: {str(e)[:70]}"); bad += 1; continue
    got = m.__version__
    ok = got == v
    print(f"  {'✓' if ok else '✗'} {k:14s} {got}" + ("" if ok else f"   (expected {v})"))
    bad += 0 if ok else 1
import torch, deepspeed, llamafactory
print(f"  ✓ deepspeed     {deepspeed.__version__}")
print(f"  ✓ llamafactory  {llamafactory.__version__}  ← from {llamafactory.__file__}")
print(f"  ✓ torch cuda    {torch.version.cuda}")
sys.exit(1 if bad else 0)
PYEOF
[ $? -eq 0 ] || fail=$((fail+1))

echo "=== 3b. Real training import chain (ABI mismatches only show up here) ==="
python - <<'PYEOF'
import sys, importlib
bad = 0
for path in ["llamafactory.data.mm_plugin",              # imports torchaudio internally
             "llamafactory.train.tuner",
             "llamafactory.model.loader",
             "llamafactory.train.sft.workflow",
             "deepspeed.runtime.zero.stage_1_and_2"]:    # we use ZeRO-2
    try:
        importlib.import_module(path); print(f"  ✓ {path}")
    except Exception as e:
        print(f"  ✗ {path}: {type(e).__name__}: {str(e)[:80]}"); bad += 1
sys.exit(1 if bad else 0)
PYEOF
[ $? -eq 0 ] || fail=$((fail+1))

echo "=== 4. flash-attn can really compute (not just import) ==="
python - <<'PYEOF'
import sys, torch
from flash_attn import flash_attn_func
q,k,v = (torch.randn(2,64,4,64, dtype=torch.bfloat16, device="cuda") for _ in range(3))
o = flash_attn_func(q,k,v, causal=True)
ok = tuple(o.shape)==(2,64,4,64) and torch.isfinite(o).all().item()
print(f"  {'✓' if ok else '✗'} kernel output {tuple(o.shape)} {o.dtype} finite={torch.isfinite(o).all().item()}")
print(f"    GPU: {torch.cuda.get_device_name(0)} sm_{''.join(map(str,torch.cuda.get_device_capability(0)))}")
sys.exit(0 if ok else 1)
PYEOF
[ $? -eq 0 ] || fail=$((fail+1))

echo "=== 5. Dependency consistency ==="
if pip check >/dev/null 2>&1; then echo "  ✓ pip check passed"; else echo "  ✗ pip check has conflicts:"; pip check 2>&1 | sed 's/^/    /'; fail=$((fail+1)); fi

echo
[ "$fail" -eq 0 ] && echo "✓ all passed, environment usable" || echo "✗ $fail item(s) did not pass"
exit "$fail"
