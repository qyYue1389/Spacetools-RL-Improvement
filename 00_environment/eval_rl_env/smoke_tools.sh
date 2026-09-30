#!/bin/bash
# =============================================================================
# 七个工具逐个冒烟 —— 单卡,一次一个
#
# 类名和方法签名全部照 GitHub 源码(NVlabs/SpaceTools-Toolshed),不是猜的:
#   VisionOpsTool.index_at(data, u, v)          u,v 是归一化 [0,1]
#   BoundingBoxTool.compute_bbox(point_cloud(N,3), mask(H,W), focal_length_px)
#   Sam2SegmentationTool.segment_from_point(image, x, y)
#   DepthEstimatorTool.estimate_depth(image)
#   GraspGeneratorTool.compute_grasp(point_cloud, mask, image, focal_length_px)
#   RoboreferTool.detect_one(image, obj_name)   构造参数是 model_path
#   VLMTool.detect_one(image, obj_name)         构造参数是 model_name
#
# 判据(不看退出码):
#   ① 真的加载了模型  ② hf_device_map 里没有 cpu/disk(accelerate 会静默 offload)
#   ③ 真的算出了结果
#
#   bash smoke_tools.sh          全部
#   bash smoke_tools.sh vlm      只测一个
# =============================================================================
set -uo pipefail
CONDA_DIR=/opt/conda-st
export HF_HOME=/workspace/hf CHECKPOINT_DIR=/workspace/checkpoints
OUT=/workspace/smoke; mkdir -p "$OUT"
ONLY="${1:-all}"; PASS=0; FAIL=0

# 合成场景:0.2 m 的立方体点云 + 矩形掩码,给 bbox / grasp 用
SCENE='
import numpy as np
rng = np.random.default_rng(0)
pts = rng.uniform(-0.1, 0.1, size=(2000, 3)).astype("float32"); pts[:, 2] += 0.6
H = W = 240; F = 500.0
mask = np.zeros((H, W), dtype=bool); mask[80:160, 90:170] = True
img = (rng.random((H, W, 3)) * 255).astype("uint8")
'

run () {  # $1=名字 $2=env $3=代码
    [ "$ONLY" = all ] || [ "$ONLY" = "$1" ] || return 0
    echo; echo "########## $1  ($2) ##########  $(date +%H:%M:%S)"
    out="$("$CONDA_DIR/envs/$2/bin/python" - <<PYEOF 2>&1
$3
PYEOF
)"
    echo "$out" > "$OUT/$1.log"
    echo "$out" | grep -vE "^\s*$" | tail -10 | sed 's/^/  /'
    nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed 's/^/  显存: /'
    if echo "$out" | grep -q "^SMOKE_OK"; then echo "  ✓ $1"; PASS=$((PASS+1))
    else echo "  ✗ $1  (完整日志 $OUT/$1.log)"; FAIL=$((FAIL+1)); fi
}

DEVMAP='
def check_devmap(t):
    for a in ("_model","model","_vlm","vlm"):
        m = getattr(t, a, None)
        if m is not None and hasattr(m, "hf_device_map"):
            dm = set(str(v) for v in m.hf_device_map.values())
            print("device_map:", dm)
            assert not (dm & {"cpu","disk"}), "offload 到 CPU/disk —— 这次调用作废"
            return
    print("device_map: (该工具未暴露)")
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
print('峰值显存 %.2f GiB' % (torch.cuda.max_memory_allocated()/2**30))
print('SMOKE_OK')"

run depth_estimator spacetools-tool-vlm "$SCENE
$DEVMAP
import torch
from toolshed.tools.depth_estimator import DepthEstimatorTool
t = DepthEstimatorTool(no_output_image=True, no_output_vars=False)
check_devmap(t)
r = t.estimate_depth(image=img)
print('estimate_depth ->', str(r)[:180])
print('峰值显存 %.2f GiB' % (torch.cuda.max_memory_allocated()/2**30))
print('SMOKE_OK')"

run grasp_generator spacetools-tool-graspgen "$SCENE
import torch, pointnet2_ops._ext
print('pointnet2_ops._ext OK')
from toolshed.tools.grasp_generator import GraspGeneratorTool
t = GraspGeneratorTool(no_output_image=True, no_output_vars=False)
print('GraspGeneratorTool 构建 OK')
try:
    r = t.compute_grasp(point_cloud=pts, mask=mask, image=img, focal_length_px=F)
    print('compute_grasp ->', str(r)[:180])
except RuntimeError as e:
    # 设计内结果:随机点云上找不到有效抓取。P5 记录过 bopgrasp 有 40 例是这种。
    print('compute_grasp 抛 RuntimeError(随机点云上属正常):', str(e)[:120])
print('峰值显存 %.2f GiB' % (torch.cuda.max_memory_allocated()/2**30))
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
print('峰值显存 %.2f GiB' % (torch.cuda.max_memory_allocated()/2**30))
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
        print('实际 dtype:', next(m.parameters()).dtype, ' ← 配置传的是 float16,vlm.py:116 写死 auto')
        break
check_devmap(t)
r = t.detect_one(image=img, obj_name='cup')
print('detect_one ->', str(r)[:200])
print('峰值显存 %.2f GiB' % (torch.cuda.max_memory_allocated()/2**30))
print('SMOKE_OK')"

echo; echo "############ 冒烟:$PASS 通过 / $FAIL 失败 ############"
exit "$FAIL"
