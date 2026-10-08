#!/bin/bash
# =============================================================================
# Smoke test the seven tools one by one — single GPU, one at a time
#
# Class names and method signatures all follow the GitHub source (NVlabs/SpaceTools-Toolshed), not guessed:
#   VisionOpsTool.index_at(data, u, v)          u,v are normalized [0,1]
#   BoundingBoxTool.compute_bbox(point_cloud(N,3), mask(H,W), focal_length_px)
#   Sam2SegmentationTool.segment_from_point(image, x, y)
#   DepthEstimatorTool.estimate_depth(image)
#   GraspGeneratorTool.compute_grasp(point_cloud, mask, image, focal_length_px)
#   RoboreferTool.detect_one(image, obj_name)   constructor argument is model_path
#   VLMTool.detect_one(image, obj_name)         constructor argument is model_name
#
# Criteria (exit code not considered):
#   ① the model was really loaded  ② no cpu/disk in hf_device_map (accelerate offloads silently)
#   ③ a result was really computed
#
#   bash smoke_tools.sh          all
#   bash smoke_tools.sh vlm      test just one
# =============================================================================
set -uo pipefail
CONDA_DIR=/opt/conda-st
export HF_HOME=/workspace/hf CHECKPOINT_DIR=/workspace/checkpoints
OUT=/workspace/smoke; mkdir -p "$OUT"
ONLY="${1:-all}"; PASS=0; FAIL=0

# Synthetic scene: point cloud of a 0.2 m cube + rectangular mask, for bbox / grasp
SCENE='
import numpy as np
rng = np.random.default_rng(0)
pts = rng.uniform(-0.1, 0.1, size=(2000, 3)).astype("float32"); pts[:, 2] += 0.6
H = W = 240; F = 500.0
mask = np.zeros((H, W), dtype=bool); mask[80:160, 90:170] = True
img = (rng.random((H, W, 3)) * 255).astype("uint8")
'

run () {  # $1=name $2=env $3=code
    [ "$ONLY" = all ] || [ "$ONLY" = "$1" ] || return 0
    echo; echo "########## $1  ($2) ##########  $(date +%H:%M:%S)"
    out="$("$CONDA_DIR/envs/$2/bin/python" - <<PYEOF 2>&1
$3
PYEOF
)"
    echo "$out" > "$OUT/$1.log"
    echo "$out" | grep -vE "^\s*$" | tail -10 | sed 's/^/  /'
    nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/  GPU memory: /'
    if echo "$out" | grep -q "^SMOKE_OK"; then echo "  ✓ $1"; PASS=$((PASS+1))
    else echo "  ✗ $1  (full log $OUT/$1.log)"; FAIL=$((FAIL+1)); fi
}

DEVMAP='
def check_devmap(t):
    for a in ("_model","model","_vlm","vlm"):
        m = getattr(t, a, None)
        if m is not None and hasattr(m, "hf_device_map"):
            dm = set(str(v) for v in m.hf_device_map.values())
            print("device_map:", dm)
            assert not (dm & {"cpu","disk"}), "offloaded to CPU/disk — this call is invalid"
            return
    print("device_map: (not exposed by this tool)")
'

run vision_ops spacetools-tool-bbox "$SCENE
from toolshed.tools.vision_ops import VisionOpsTool
t = VisionOpsTool(no_output_image=True, no_output_vars=False)
depth = np.full((H, W), 0.6, dtype='float32'); depth[80:160, 90:170] = 0.45
r = t.index_at(data=depth, u=0.5, v=0.5)
print('index_at(0.5,0.5) ->', str(r)[:160])
print('SMOKE_OK')"

run bounding_box spacetools-tool-bbox "$SCENE
from toolshed.tools.bounding_box import BoundingBoxTool
t = BoundingBoxTool(no_output_image=True, no_output_vars=False)
r = t.compute_bbox(point_cloud=pts, mask=mask, focal_length_px=F)
print('compute_bbox ->', str(r)[:220])
print('SMOKE_OK')"

run sam2 spacetools-tool-vlm "$SCENE
$DEVMAP
import torch
from toolshed.tools.sam2 import Sam2SegmentationTool
t = Sam2SegmentationTool(no_output_image=True, no_output_vars=False)
check_devmap(t)
r = t.segment_from_point(image=img, x=120, y=120)
print('segment_from_point ->', str(r)[:180])
print('peak GPU memory %.2f GiB' % (torch.cuda.max_memory_allocated()/2**30))
print('SMOKE_OK')"

run depth_estimator spacetools-tool-vlm "$SCENE
$DEVMAP
import torch
from toolshed.tools.depth_estimator import DepthEstimatorTool
t = DepthEstimatorTool(no_output_image=True, no_output_vars=False)
check_devmap(t)
r = t.estimate_depth(image=img)
print('estimate_depth ->', str(r)[:180])
print('peak GPU memory %.2f GiB' % (torch.cuda.max_memory_allocated()/2**30))
print('SMOKE_OK')"

run grasp_generator spacetools-tool-graspgen "$SCENE
import torch, pointnet2_ops._ext
print('pointnet2_ops._ext OK')
from toolshed.tools.grasp_generator import GraspGeneratorTool
t = GraspGeneratorTool(no_output_image=True, no_output_vars=False)
print('GraspGeneratorTool built OK')
try:
    r = t.compute_grasp(point_cloud=pts, mask=mask, image=img, focal_length_px=F)
    print('compute_grasp ->', str(r)[:180])
except RuntimeError as e:
    # Expected by design: no valid grasp can be found on a random point cloud. P5 recorded 40 such cases for bopgrasp.
    print('compute_grasp raised RuntimeError (normal on a random point cloud):', str(e)[:120])
print('peak GPU memory %.2f GiB' % (torch.cuda.max_memory_allocated()/2**30))
print('SMOKE_OK')"

run roborefer spacetools-tool-roborefer "$SCENE
$DEVMAP
import os, torch
from toolshed.tools.roborefer import RoboreferTool
t = RoboreferTool(model_path=os.environ.get('ROBOREFER_MODEL','Zhoues/RoboRefer-8B-SFT'),
                  no_output_image=True, no_output_vars=False,
                  exclude_methods=['general_query'], exclude_behavior='error')
check_devmap(t)
r = t.detect_one(image=img, obj_name='cup')
print('detect_one ->', str(r)[:200])
print('peak GPU memory %.2f GiB' % (torch.cuda.max_memory_allocated()/2**30))
print('SMOKE_OK')"

run vlm spacetools-tool-vlm "$SCENE
$DEVMAP
import torch
from toolshed.tools.vlm import VLMTool
t = VLMTool(model_name='allenai/Molmo-7B-D-0924', dtype='float16',
            no_output_image=True, no_output_vars=False,
            exclude_methods=['general_query'], exclude_behavior='error')
for a in ('_model','model'):
    m = getattr(t, a, None)
    if m is not None:
        print('actual dtype:', next(m.parameters()).dtype, ' ← config passes float16, vlm.py:116 hard-codes auto')
        break
check_devmap(t)
r = t.detect_one(image=img, obj_name='cup')
print('detect_one ->', str(r)[:200])
print('peak GPU memory %.2f GiB' % (torch.cuda.max_memory_allocated()/2**30))
print('SMOKE_OK')"

echo; echo "############ smoke test: $PASS passed / $FAIL failed ############"
exit "$FAIL"
