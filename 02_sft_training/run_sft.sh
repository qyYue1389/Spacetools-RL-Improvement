#!/bin/bash
# =============================================================================
# SpaceTools SFT Training
#
# Downloads data from HuggingFace, injects tool schemas into system prompts,
# trains a full fine-tune on Qwen2.5-VL, and fixes the checkpoint for RL.
#
# Versions:
#   v1  — Reasoning only (11 tools, ~7000 samples, robot-tool samples filtered)
#   v2  — Full pipeline  (17 tools, ~7900 samples, includes robot tools)
#
# Required environment variables:
#   TOOL_CONFIG       — path to toolshed YAML config (v1 or v2)
#                       e.g. ../verl/examples/toolshed/toolshed_v1_config.yaml
#
# Optional environment variables:
#   BASE_MODEL        — base VLM (default: Qwen/Qwen2.5-VL-3B-Instruct)
#   OUTPUT_DIR        — experiment output dir (default: ../../experiments/sft_<version>_<timestamp>)
#   VERSION           — v1 or v2 (default: v1)
#   MAX_STEPS         — training steps (default: 3000)
#   NUM_GPUS          — GPUs for training (default: 8)
#
# Usage:
#   VERSION=v1 TOOL_CONFIG=../verl/examples/toolshed/toolshed_v1_config.yaml \
#       bash scripts/spacetools/run_sft.sh
#
# Output:
#   $OUTPUT_DIR/sft_checkpoint/   — ready for RL training
# =============================================================================

set -euxo pipefail

: "${TOOL_CONFIG:?Set TOOL_CONFIG to path of toolshed YAML config (v1 or v2)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$LLAMA_DIR/.." && pwd)"

VERSION="${VERSION:-v1}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
MAX_STEPS="${MAX_STEPS:-3000}"
NUM_GPUS="${NUM_GPUS:-8}"        # 上游遗留:声明了但从未被使用(见下)

# =============================================================================
# 全局 batch 必须 = 8(= per_device x ga x 卡数),论文口径,不可变。
# 这里按实际可见的 GPU 数推导,而不是写死 —— 写死会在换机器时静默偏离。
#
# ⚠️ 上游的 NUM_GPUS 是空操作:第 42 行声明,但配置和启动命令里都没引用它。
# FORCE_TORCHRUN=1 会用**全部可见 GPU**,所以真正决定卡数的是 CUDA_VISIBLE_DEVICES。
# 想跑 2 卡就 `CUDA_VISIBLE_DEVICES=0,1`,别指望 NUM_GPUS。
# =============================================================================
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c '[0-9]')
else
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | grep -c '[0-9]')
fi

GLOBAL_BATCH=8
if [ $(( GLOBAL_BATCH % GPU_COUNT )) -ne 0 ]; then
    echo "✗ $GPU_COUNT 卡:全局 batch $GLOBAL_BATCH 不能被卡数整除,无整数解。"
    echo "  可用卡数:1 / 2 / 4 / 8。3 卡做不到 —— 8÷3 不是整数,"
    echo "  只能得到 6 或 9,那就偏离了论文口径,而本 checkpoint 是 RL 两臂的共同起点。"
    exit 1
fi
# per_device 上限 2:实测 per_device=4 在 A6000(48 GB)上 OOM ——
# 稳态 39.8 GiB + logits 瞬时尖峰 8.4 GiB ≈ 48 GiB,两卡皆炸,不是 GPU1 少 3 GiB 的问题。
PER_DEVICE=$(( GLOBAL_BATCH / GPU_COUNT ))
[ "$PER_DEVICE" -gt 2 ] && PER_DEVICE=2
GRAD_ACCUM=$(( GLOBAL_BATCH / (GPU_COUNT * PER_DEVICE) ))

# 收尾断言:算错了就停,不要静默产出口径不同的 checkpoint
ACTUAL=$(( PER_DEVICE * GRAD_ACCUM * GPU_COUNT ))
[ "$ACTUAL" -eq "$GLOBAL_BATCH" ] || {
    echo "✗ 全局 batch 算成了 $ACTUAL,应为 $GLOBAL_BATCH"; exit 1; }

echo "GPU 数 $GPU_COUNT · per_device=$PER_DEVICE · ga=$GRAD_ACCUM · 全局 batch $ACTUAL ✓"
# 数据集只有 4 个分片(streaming 模式)。超过 4 卡会有 rank 分不到数据分片。
# 写成显式 if 而不是 `[ ] && echo`:后者在条件为假时整体返回 1,会污染退出码。
if [ "$GPU_COUNT" -gt 4 ]; then
    echo "⚠️ $GPU_COUNT 卡 > 数据集分片数 4,部分 rank 可能拿不到数据分片"
fi

EXPERIMENT_DIR="${OUTPUT_DIR:-$REPO_DIR/experiments/sft_${VERSION}_$(date +%Y%m%d_%H%M%S)}"
SFT_DATA_DIR="$EXPERIMENT_DIR/sft_data"
SFT_OUTPUT_DIR="$EXPERIMENT_DIR/sft_checkpoint"
HF_CACHE_DIR="$EXPERIMENT_DIR/hf_cache"

mkdir -p "$EXPERIMENT_DIR" "$SFT_DATA_DIR" "$HF_CACHE_DIR"
echo "=== SpaceTools SFT Training ==="
echo "Version:    $VERSION"
echo "Base model: $BASE_MODEL"
echo "Tool config: $TOOL_CONFIG"
echo "Output:     $EXPERIMENT_DIR"

cp "$TOOL_CONFIG" "$EXPERIMENT_DIR/tool_config.yaml"

# =============================================================================
# Phase 1: Download and prepare SFT data from HuggingFace
# =============================================================================
echo "=== PHASE 1: Prepare SFT data ==="

python3 - "$HF_CACHE_DIR" "$SFT_DATA_DIR" <<'DOWNLOAD_PY'
import sys, os, shutil
from huggingface_hub import snapshot_download

cache_dir, sft_data_dir = sys.argv[1], sys.argv[2]
sft_path = snapshot_download(
    "siyich/spacetools-sft", repo_type="dataset",
    cache_dir=cache_dir, local_dir=os.path.join(sft_data_dir, "hf_download"),
    # 默认 max_workers=8 实测只有 0.8 MB/s(7463 个文件,卡在每文件延迟上)。
    # 实测 32 worker 写网络卷 30 MB/s、22.6 文件/s。纯下载并发,内容由哈希校验,零影响。
    max_workers=32,
)
for subdir in ["data", "images"]:
    src = os.path.join(sft_path, subdir)
    dst = os.path.join(sft_data_dir, subdir)
    if os.path.exists(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
print(f"Downloaded SFT data to {sft_data_dir}")
DOWNLOAD_PY

SFT_JSON="$SFT_DATA_DIR/data/train.json"

# Rewrite system prompts with tool schemas; optionally filter robot-tool samples
python3 - "$SFT_JSON" "$TOOL_CONFIG" "$SFT_DATA_DIR" "$VERSION" <<'PREP_PY'
import json, yaml, sys, re, os

sft_json = sys.argv[1]
tool_config_path = sys.argv[2]
sft_dir = sys.argv[3]
version = sys.argv[4]

with open(tool_config_path) as f:
    tool_data = yaml.safe_load(f)
schemas = [item["tool_schema"] for item in tool_data["tools"] if "tool_schema" in item]

lines = [
    "\n\n# Tools\n",
    "You may call one or more functions to assist with the user query.\n",
    "You are provided with function signatures within <tools></tools> XML tags:",
    "<tools>",
]
for schema in schemas:
    lines.append(json.dumps(schema, ensure_ascii=False))
lines.append("</tools>\n")
lines.append(
    "For each function call, return a json object with function name and "
    "arguments within <tool_call></tool_call> XML tags:\n"
    '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
)
tool_text = "\n".join(lines)

DEFAULT_SYSTEM = (
    "You are an expert in 3D spatial reasoning for robotics. "
    "Given an image and a spatial reasoning question, follow this process:\n\n"
    "1. First, think about the reasoning process as an internal monologue the first time "
    "you receive the question, and every time you receive new information.\n"
    "Your reasoning process MUST be enclosed within <think> </think> tags.\n"
    "2. After thinking, if you need additional information to answer the question, "
    "such as specific object location in the image, call the appropriate vision tool exactly once.\n"
    "3. When you receive a tool response that contains reasonable information, "
    "use that information to continue your analysis on the question.\n"
    "4. Once no further visual analysis or tool calls are needed, you MUST provide your final answer inside "
    "<answer> and </answer> tags without detailed illustrations.\n\n"
    "Example answer format: <answer> <Your final answer here> </answer>."
)

with open(sft_json) as f:
    data = json.load(f)
print(f"Loaded {len(data)} samples")

# V1: filter robot-tool samples; V2: keep all
if version == "v1":
    robot_pattern = re.compile(r'"name":\s*"robot\.')
    filtered = []
    robot_count = 0
    for item in data:
        has_robot = False
        for conv in item.get("conversations", []):
            content = conv.get("content", "")
            if isinstance(content, str) and robot_pattern.search(content):
                has_robot = True
                break
        if has_robot:
            robot_count += 1
        else:
            filtered.append(item)
    print(f"V1: removed {robot_count} robot-tool samples, kept {len(filtered)}")
    data = filtered
else:
    print(f"V2: keeping all {len(data)} samples (including robot tools)")

for item in data:
    item["system"] = DEFAULT_SYSTEM + tool_text
    item.pop("tools", None)
    if "images" in item:
        item["images"] = [
            os.path.join(sft_dir, img) if not os.path.isabs(img) else img
            for img in item["images"]
        ]

with open(sft_json, "w") as f:
    json.dump(data, f)
print(f"Written {len(data)} samples with {len(schemas)} tools in system prompt")
PREP_PY

cp "$TOOL_CONFIG" "$SFT_DATA_DIR/toolshed_config.yaml"
echo "=== PHASE 1 COMPLETE ==="

# =============================================================================
# Phase 2: SFT training via LLaMA-Factory
# =============================================================================
echo "=== PHASE 2: SFT training ==="

cd "$LLAMA_DIR"

DATASET_NAME="spacetools_sft_${VERSION}"

python3 -c "
import json
info_path = '$LLAMA_DIR/data/dataset_info.json'
with open(info_path) as f:
    info = json.load(f)
info['$DATASET_NAME'] = {
    'file_name': '$SFT_JSON',
    'formatting': 'sharegpt',
    'columns': {'images': 'images', 'messages': 'conversations', 'system': 'system'},
    'tags': {'role_tag': 'role', 'content_tag': 'content', 'user_tag': 'user', 'assistant_tag': 'assistant'}
}
with open(info_path, 'w') as f:
    json.dump(info, f, indent=2)
print('Registered dataset: $DATASET_NAME')
"

# 3 epochs via dataset repetition
DATASET_SPEC="${DATASET_NAME},${DATASET_NAME},${DATASET_NAME}"

cat > "$EXPERIMENT_DIR/sft_config.yaml" <<YAML
### model
model_name_or_path: $BASE_MODEL
image_max_pixels: 262144
video_max_pixels: 16384
trust_remote_code: true
use_reentrant_gc: false

### method
stage: sft
do_train: true
finetuning_type: full
freeze_vision_tower: true
freeze_multi_modal_projector: true
freeze_language_model: false
deepspeed: examples/deepspeed/ds_z2_config.json

### dataset
dataset: $DATASET_SPEC
template: qwen2_vl
cutoff_len: 8192
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 8
streaming: true
accelerator_config:
  dispatch_batches: false
buffer_size: 8
preprocessing_batch_size: 8

### output
output_dir: $SFT_OUTPUT_DIR
logging_steps: 5
save_steps: 500
plot_loss: true
overwrite_output_dir: false
save_only_model: true
report_to: none

### train
# 这两个值由上面按可见 GPU 数推导,保证 per_device x ga x 卡数 = 8(论文口径)。
# per_device 上限锁在 2:实测 4 在 A6000(48 GB)上 OOM。
per_device_train_batch_size: $PER_DEVICE
gradient_accumulation_steps: $GRAD_ACCUM
learning_rate: 2e-5
max_steps: $MAX_STEPS
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### eval
val_size: 20
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
YAML

set +u; conda activate spacetools-sft; set -u
export ACCELERATE_SPLIT_BATCHES=true
export TRANSFORMERS_NO_TORCH_LOAD_CHECK=1

FORCE_TORCHRUN=1 python -m llamafactory.cli train "$EXPERIMENT_DIR/sft_config.yaml"

echo "=== PHASE 2 COMPLETE ==="

# =============================================================================
# Phase 3: Checkpoint fixes for RL compatibility
# =============================================================================
echo "=== PHASE 3: Checkpoint fixes ==="

# Fix config.json: remove text_config, set tie_word_embeddings
python3 -c "
import json, pathlib
p = pathlib.Path('$SFT_OUTPUT_DIR/config.json')
c = json.loads(p.read_text())
changed = False
if 'text_config' in c:
    del c['text_config']; changed = True; print('Removed text_config')
if c.get('tie_word_embeddings') is not True:
    c['tie_word_embeddings'] = True; changed = True; print('Set tie_word_embeddings=True')
if changed:
    p.write_text(json.dumps(c, indent=2))
else:
    print('No fixes needed')
"

# Fix preprocessor_config.json: copy from base model
if [[ -d "$BASE_MODEL" ]]; then
    cp "$BASE_MODEL/preprocessor_config.json" "$SFT_OUTPUT_DIR/preprocessor_config.json"
    echo "Copied preprocessor_config.json from local base model"
else
    python3 -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download('$BASE_MODEL', 'preprocessor_config.json')
shutil.copy(path, '$SFT_OUTPUT_DIR/preprocessor_config.json')
print('Downloaded preprocessor_config.json from HuggingFace')
"
fi

cp "$TOOL_CONFIG" "$SFT_OUTPUT_DIR/toolshed_config.yaml"

echo "=== PHASE 3 COMPLETE ==="
echo ""
echo "=== SFT TRAINING COMPLETE ==="
echo "Checkpoint: $SFT_OUTPUT_DIR"
echo ""
echo "Next step: run RL training with this checkpoint:"
echo "  cd ../../verl"
echo "  SFT_CHECKPOINT=$SFT_OUTPUT_DIR bash examples/toolshed/run_rl.sh"
