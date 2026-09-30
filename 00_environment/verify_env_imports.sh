#!/usr/bin/env bash
# P1 acceptance check.
#
# Replaces the repo's test_envs.sh, which is useless here: it tests
# spacetools-sft (deliberately not built, eval does not need it) and dies on
# `set -e` at the first step, and it covers none of the four tool environments.
#
# This walks the conda_env mapping in run_eval.sh and, for each environment,
# imports the toolshed tool modules that environment actually hosts. Importing
# toolshed.tools.<name> pulls in that tool's third-party stack, so one import
# exercises the whole dependency chain.
source /workspace/env.sh

declare -A MODS=(
  [spacetools-tool-roborefer]="roborefer"
  [spacetools-tool-vlm]="vlm sam2 depth_estimator"
  [spacetools-tool-bbox]="bounding_box vision_ops"
  [spacetools-tool-graspgen]="grasp_generator"
)
# bbox hosts only numpy/ray tools; torch is correctly absent there.
declare -A NEEDS_TORCH=(
  [spacetools-tool-roborefer]=1
  [spacetools-tool-vlm]=1
  [spacetools-tool-bbox]=0
  [spacetools-tool-graspgen]=1
)

for env in spacetools-tool-roborefer spacetools-tool-vlm spacetools-tool-bbox spacetools-tool-graspgen; do
  P=/workspace/envs/$env
  echo "=============== $env ==============="
  if [ ! -x "$P/bin/python" ]; then echo "  MISSING env"; continue; fi
  LD_LIBRARY_PATH="$P/lib:$P/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$P/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}" \
  "$P/bin/python" - "${NEEDS_TORCH[$env]}" ${MODS[$env]} <<'PY' 2>&1 | grep -vE "FutureWarning|warnings.warn|OpenGL.acceleratesupport|absl::InitializeLog|cpu_feature_guard|rebuild TensorFlow|^\s*$|it \[00:00|Migrating your old cache|one-time only operation"
import sys, importlib
needs_torch, mods = sys.argv[1] == "1", sys.argv[2:]
if needs_torch:
    try:
        import torch, numpy
        print(f"  torch {torch.__version__} | numpy {numpy.__version__} | cuda {torch.cuda.is_available()} x{torch.cuda.device_count()}")
        if torch.cuda.is_available():
            x = torch.randn(64, 64, device="cuda"); _ = (x @ x.T).sum().item()
            print("  OK   CUDA compute")
    except Exception as e:
        print(f"  FAIL torch/numpy: {type(e).__name__}: {str(e)[:140]}")
else:
    import numpy
    print(f"  numpy {numpy.__version__} (torch not required by this env)")
for m in mods:
    try:
        importlib.import_module(f"toolshed.tools.{m}")
        print(f"  OK   toolshed.tools.{m}")
    except Exception as e:
        print(f"  FAIL toolshed.tools.{m}: {type(e).__name__}: {str(e)[:200]}")
PY
done
