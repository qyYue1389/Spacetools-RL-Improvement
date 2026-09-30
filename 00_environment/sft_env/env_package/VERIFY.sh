#!/bin/bash
# 恢复后的验收。任何一项不过就是没恢复好 —— 不看退出码,看每一行。
set -uo pipefail
export PATH=/opt/conda-st/bin:$PATH
eval "$(conda shell.bash hook)"
conda activate spacetools-sft

fail=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "  ✓ $1"; else echo "  ✗ $1"; fail=$((fail+1)); fi; }

echo "=== 1. 路径 ==="
chk "/opt/conda-st 存在"             '[ -d /opt/conda-st ]'
chk "/workspace/SpaceTools-SFT 存在" '[ -d /workspace/SpaceTools-SFT ]'
chk "editable 指针没断"              '[ -f /workspace/SpaceTools-SFT/src/llamafactory/__init__.py ]'

echo "=== 2. 硬件 ==="
nvidia-smi --query-gpu=index,name,driver_version,compute_cap,memory.total --format=csv 2>&1 | sed 's/^/  /'
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk 'NR==1{print $1}')
case "$CC" in
  8.0|8.6) echo "  ✓ compute_cap $CC —— 有 sm_80 cubin,可直接跑" ;;
  *) echo "  ✗ compute_cap $CC —— 本包只含 sm_80 cubin 且无 PTX 回退,flash-attn 跑不了"; fail=$((fail+1)) ;;
esac

echo "=== 3. 包版本 ==="
# ⚠️ 这一节曾经漏掉 torchaudio,导致带缺陷的包流出去过。
# 成因:上游 setup_envs.sh 只钉 torch,pip 把 torchaudio 解析到 2.11.0(给 torch 2.11 编的),
# import 时报 "undefined symbol: torch_library_impl"。
# 光 import llamafactory 抓不到 —— 它是惰性的,要到 mm_plugin 才 import torchaudio。
python - <<'PYEOF'
import sys
exp = {"torch":"2.9.1+cu128", "torchvision":"0.24.1+cu128", "torchaudio":"2.9.1+cu128",
       "transformers":"4.57.1", "flash_attn":"2.8.3.post1"}
bad = 0
for k, v in exp.items():
    try:
        m = __import__(k)
    except Exception as e:
        print(f"  ✗ {k:14s} import 失败: {type(e).__name__}: {str(e)[:70]}"); bad += 1; continue
    got = m.__version__
    ok = got == v
    print(f"  {'✓' if ok else '✗'} {k:14s} {got}" + ("" if ok else f"   (期望 {v})"))
    bad += 0 if ok else 1
import torch, deepspeed, llamafactory
print(f"  ✓ deepspeed     {deepspeed.__version__}")
print(f"  ✓ llamafactory  {llamafactory.__version__}  ← 来自 {llamafactory.__file__}")
print(f"  ✓ torch cuda    {torch.version.cuda}")
sys.exit(1 if bad else 0)
PYEOF
[ $? -eq 0 ] || fail=$((fail+1))

echo "=== 3b. 训练真实导入链(ABI 错配只在这里暴露)==="
python - <<'PYEOF'
import sys, importlib
bad = 0
for path in ["llamafactory.data.mm_plugin",              # 内部 import torchaudio
             "llamafactory.train.tuner",
             "llamafactory.model.loader",
             "llamafactory.train.sft.workflow",
             "deepspeed.runtime.zero.stage_1_and_2"]:    # 我们用 ZeRO-2
    try:
        importlib.import_module(path); print(f"  ✓ {path}")
    except Exception as e:
        print(f"  ✗ {path}: {type(e).__name__}: {str(e)[:80]}"); bad += 1
sys.exit(1 if bad else 0)
PYEOF
[ $? -eq 0 ] || fail=$((fail+1))

echo "=== 4. flash-attn 真的能算(不只是 import)==="
python - <<'PYEOF'
import sys, torch
from flash_attn import flash_attn_func
q,k,v = (torch.randn(2,64,4,64, dtype=torch.bfloat16, device="cuda") for _ in range(3))
o = flash_attn_func(q,k,v, causal=True)
ok = tuple(o.shape)==(2,64,4,64) and torch.isfinite(o).all().item()
print(f"  {'✓' if ok else '✗'} kernel 输出 {tuple(o.shape)} {o.dtype} 有限值={torch.isfinite(o).all().item()}")
print(f"    卡: {torch.cuda.get_device_name(0)} sm_{''.join(map(str,torch.cuda.get_device_capability(0)))}")
sys.exit(0 if ok else 1)
PYEOF
[ $? -eq 0 ] || fail=$((fail+1))

echo "=== 5. 依赖一致性 ==="
if pip check >/dev/null 2>&1; then echo "  ✓ pip check 通过"; else echo "  ✗ pip check 有冲突:"; pip check 2>&1 | sed 's/^/    /'; fail=$((fail+1)); fi

echo
[ "$fail" -eq 0 ] && echo "✓ 全部通过,环境可用" || echo "✗ $fail 项未通过"
exit "$fail"
