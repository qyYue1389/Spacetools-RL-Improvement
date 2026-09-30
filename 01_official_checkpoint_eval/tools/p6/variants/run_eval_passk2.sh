#!/bin/bash
# =============================================================================
# SpaceTools Evaluation
#
# Evaluates a model checkpoint on the paper's benchmarks using live Toolshed
# tool execution. Data is downloaded automatically from HuggingFace.
#
# Paper benchmarks (9 evals -> 7 reported metrics):
#   robospatial       -> RoboSpatial (VQA, Vacant, Overall)
#   reflocation       \
#   refplacement       -> RefSpatial (averaged)
#   refunseen         /
#   blinkdepth        -> BLINK Depth
#   cvb2drelation     -> CVBench 2D Rel.
#   cvb3ddepth        -> CVBench 3D Depth
#   boppose           -> BOP-ask Pose
#   bopgrasp          -> BOP-ask Grasp (MACE + SR)
#
# Required environment variables:
#   ROBOREFER_MODEL   — path to RoboRefer-8B-SFT model
#   DEPTH_CHECKPOINT  — path to depth_pro.pt checkpoint
#
# Optional environment variables:
#   VERSION           — v1 or v2 (default: v1); controls tool config
#   OUTPUT_DIR        — eval output dir
#   NUM_GPUS          — total GPUs (default: 4)
#   EVAL_GPUS         — GPUs for eval inference (default: 1)
#
# Usage:
#   ROBOREFER_MODEL=/path/to/model DEPTH_CHECKPOINT=/path/to/depth_pro.pt \
#       bash examples/toolshed/run_eval.sh /path/to/model [benchmark1 benchmark2 ...]
#
#   # Run specific benchmarks only:
#   bash examples/toolshed/run_eval.sh /path/to/model robospatial bopgrasp
# =============================================================================

set -euxo pipefail

# `conda activate` is a shell function, and a non-interactive `bash run_eval.sh`
# does not inherit shell functions from the caller -- only exported variables.
# Load conda's hook here so the script works when invoked as `bash ...` rather
# than sourced. CONDA_ROOT is honoured if set, otherwise fall back to the conda
# on PATH.
if ! declare -F conda >/dev/null 2>&1; then
    _conda_base="${CONDA_ROOT:-}"
    if [ -z "$_conda_base" ] && command -v conda >/dev/null 2>&1; then
        _conda_base="$(conda info --base)"
    fi
    if [ -n "$_conda_base" ] && [ -f "$_conda_base/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        . "$_conda_base/etc/profile.d/conda.sh"
    else
        echo "ERROR: cannot locate conda.sh; set CONDA_ROOT or run 'conda init'." >&2
        exit 1
    fi
fi

: "${ROBOREFER_MODEL:?Set ROBOREFER_MODEL to path of RoboRefer-8B-SFT model}"
: "${DEPTH_CHECKPOINT:?Set DEPTH_CHECKPOINT to path of depth_pro.pt}"

MODEL_PATH="${1:?Usage: $0 MODEL_PATH [benchmark ...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$VERL_DIR/.." && pwd)"

# actor.fsdp_config.model_dtype: fp32 -> bf16.  EVAL ONLY -- see the warning.
#
# On a 40 GB A100 the policy card OOMs during rollout_mode()'s weight push:
#   sglang_rollout.py:213  get_named_tensor_buckets -> tensor.clone()
#   "Tried to allocate 20.00 MiB ... 19.12 MiB is free.
#    Process A 31.42 GiB (FSDP worker)   Process B 8.00 GiB (sglang weights)"
# verl's FSDP engine defaults model_dtype to fp32, so the 4.066 B-parameter
# policy is held as a 16.3 GB fp32 master copy; bf16 is 8.1 GB, freeing 8.2 GB.
#
# This is numerically inert FOR EVALUATION, and the reason is specific:
#   - every tensor in siyich/spacetools-ckpt is stored BF16 (825/825 tensors,
#     8.132 GB / 4.066 B params = 2.00 bytes/param -- checked, not assumed);
#   - verl upcasts that to the fp32 master, then downcasts it again to hand to
#     sglang, which runs bf16 (rollout dtype: bfloat16). bf16 -> fp32 -> bf16 is
#     the identity, so the fp32 master round-trips the same bits it was given;
#   - trainer.val_only=true, so there is no optimizer step and no gradient
#     accumulation -- the two things an fp32 master exists for.
#
# WARNING: this reasoning does NOT extend to training. If P7 ever runs run_rl.sh,
# the fp32 master matters and this flag must not be copied across.
#
# A 48 GB A6000 held the fp32 master fine, which is why P0-P3 never saw this.
#
# Two things that are NOT the knob here, both measured:
#   - gpu_memory_utilization 0.5 -> 0.35 left the failure byte-identical.
#     TorchMemorySaver has already released the KV cache by the time the weight
#     push runs, so sglang is down to 8 GB of weights and the fraction is moot.
#   - actor.fsdp_config.param_offload=True moved 31.46 -> 31.42 GiB, i.e.
#     nothing: update_weights gathers the params back onto the GPU regardless.
#   - update_weights_bucket_megabytes 2048 -> 512 freed its 1.5 GB transient and
#     the run still filled the card (GPU3 peaked at 39.5 GB). The fp32 master is
#     the only allocation big enough to matter.
VERSION="${VERSION:-v1}"
NUM_GPUS="${NUM_GPUS:-4}"
EVAL_GPUS="${EVAL_GPUS:-1}"

if [[ "$VERSION" == "v1" ]]; then
    TOOL_CONFIG="$SCRIPT_DIR/toolshed_v1_config.yaml"
elif [[ "$VERSION" == "v2" ]]; then
    TOOL_CONFIG="$SCRIPT_DIR/toolshed_v2_config.yaml"
else
    echo "Unknown VERSION: $VERSION (use v1 or v2)"; exit 1
fi

ALL_BENCHMARKS=(robospatial reflocation refplacement refunseen boppose bopgrasp blinkdepth cvb2drelation cvb3ddepth)
if [[ $# -gt 0 ]]; then RUN_BENCHMARKS=("$@"); else RUN_BENCHMARKS=("${ALL_BENCHMARKS[@]}"); fi

MODEL_NAME="$(basename "$MODEL_PATH")"
EVAL_DIR="${OUTPUT_DIR:-$REPO_DIR/experiments/eval_${VERSION}_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)}"

echo "=== SpaceTools Evaluation ==="
echo "Version:    $VERSION"
echo "Model:      $MODEL_PATH"
echo "Output:     $EVAL_DIR"
echo "Benchmarks: ${RUN_BENCHMARKS[*]}"

TOOLSHED_PID=""
cleanup() {
    echo "Cleaning up..."
    [ -n "$TOOLSHED_PID" ] && kill $TOOLSHED_PID 2>/dev/null && wait $TOOLSHED_PID 2>/dev/null
    ray stop --force 2>/dev/null || true
}
trap cleanup SIGINT SIGTERM EXIT
mkdir -p "$EVAL_DIR"

# Download eval benchmarks
echo "=== Downloading eval benchmarks ==="
set +u; conda activate spacetools-rl; set -u
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export HYDRA_FULL_ERROR=1
# NOTE: do not set expandable_segments here. The OOM traces recommend it, but
# sglang's HYBRID rollout mode uses TorchMemorySaver to release and rebuild the
# KV cache, and that refuses to run under expandable_segments:
#   RuntimeError: TorchMemorySaver is disabled for the current process because
#   expandable_segments is not supported yet.
# The fragmentation it would save is marginal now that the tools have three
# cards; the real fix was EVAL_GPUS=1 plus the re-balanced fractions.

# DATA_DIR may be preset to point at an existing (or subset) benchmark tree.
# Without it every run re-downloads the full 292 MB into the timestamped EVAL_DIR,
# because snapshot_download is given a cache_dir inside EVAL_DIR as well.
# Presetting it also allows running a truncated parquet for smoke tests.
if [ -z "${DATA_DIR:-}" ]; then
python3 - "$EVAL_DIR" <<'DL_PY'
import sys, os
from huggingface_hub import snapshot_download
eval_dir = sys.argv[1]
path = snapshot_download("siyich/spacetools-eval-benchmarks", repo_type="dataset",
    cache_dir=os.path.join(eval_dir, "hf_cache"), local_dir=os.path.join(eval_dir, "benchmarks"))
print(f"Downloaded eval benchmarks to {path}")
DL_PY

DATA_DIR="$EVAL_DIR/benchmarks"
else
    echo "Using preset DATA_DIR=$DATA_DIR (skipping benchmark download)"
fi

declare -A BENCHMARKS=(
    [robospatial]="data/robospatial.parquet"
    [reflocation]="data/reflocation.parquet"
    [refplacement]="data/refplacement.parquet"
    [refunseen]="data/refunseen.parquet"
    [boppose]="data/boppose.parquet"
    [bopgrasp]="data/bopgrasp.parquet"
    [blinkdepth]="data/blinkdepth.parquet"
    [cvb2drelation]="data/cvb2drelation.parquet"
    [cvb3ddepth]="data/cvb3ddepth.parquet"
)

for bench in "${RUN_BENCHMARKS[@]}"; do
    [[ -z "${BENCHMARKS[$bench]+x}" ]] && echo "Unknown benchmark: $bench" && exit 1
    [[ ! -f "$DATA_DIR/${BENCHMARKS[$bench]}" ]] && echo "Missing: $DATA_DIR/${BENCHMARKS[$bench]}" && exit 1
done

# =============================================================================
# Start toolshed
# =============================================================================
echo "=== Starting toolshed ($VERSION) ==="

ray stop --force 2>/dev/null || true; sleep 5
ray start --head --num-gpus="$NUM_GPUS" --port=6379 2>&1 | tail -3
export RAY_ADDRESS="127.0.0.1:6379"

if [[ "$VERSION" == "v1" ]]; then
    python3 - "$ROBOREFER_MODEL" "$DEPTH_CHECKPOINT" <<'TOOLSHED_PY' &
import ray, time, sys
ray.init(address="auto")
from toolshed import start_toolkit
roborefer_model, depth_checkpoint = sys.argv[1], sys.argv[2]
TOOL_CONFIGS = {
    'roborefer': {'num_actors': 1, 'resources': {'num_gpus': 0.6}, 'conda_env': 'spacetools-tool-roborefer', 'timeout': 600,
        'args': {'model_path': roborefer_model, 'no_output_image': True, 'no_output_vars': False, 'exclude_methods': ['general_query'], 'exclude_behavior': 'error'}},
    'vlm': {'num_actors': 1, 'resources': {'num_gpus': 1.0}, 'conda_env': 'spacetools-tool-vlm', 'timeout': 600,
        'args': {'no_output_image': True, 'no_output_vars': False, 'model_name': 'allenai/Molmo-7B-D-0924', 'dtype': 'auto', 'exclude_methods': ['general_query'], 'exclude_behavior': 'error'}},
    'sam2': {'num_actors': 2, 'resources': {'num_gpus': 0.2}, 'conda_env': 'spacetools-tool-vlm', 'timeout': 600, 'args': {'no_output_image': True, 'no_output_vars': False}},
    'depth_estimator': {'num_actors': 2, 'resources': {'num_gpus': 0.4}, 'conda_env': 'spacetools-tool-vlm', 'timeout': 600,
        'args': {'checkpoint_path': depth_checkpoint, 'no_output_image': True, 'no_output_vars': False}},
    'bounding_box': {'num_actors': 2, 'resources': {'num_gpus': 0.05}, 'conda_env': 'spacetools-tool-bbox', 'timeout': 600, 'args': {'no_output_image': True, 'no_output_vars': False}},
    'vision_ops': {'num_actors': 1, 'resources': {'num_gpus': 0}, 'conda_env': 'spacetools-tool-bbox', 'timeout': 600,
        'args': {'no_output_image': True, 'no_output_vars': False, 'exclude_methods': ['mask_crop', 'point_crop', 'project_2d_points_to_3d', 'project_3d_points_to_2d', 'draw_polygon'], 'exclude_behavior': 'error'}},
    'grasp_generator': {'num_actors': 1, 'resources': {'num_gpus': 0.1}, 'conda_env': 'spacetools-tool-graspgen', 'timeout': 600, 'args': {'no_output_image': True, 'no_output_vars': False}},
}
router = start_toolkit(TOOL_CONFIGS, detached=False, dashboard=False)
while True: time.sleep(60)
TOOLSHED_PY
else
    python3 - "$ROBOREFER_MODEL" "$DEPTH_CHECKPOINT" <<'TOOLSHED_PY' &
import ray, time, sys
ray.init(address="auto")
from toolshed import start_toolkit
roborefer_model, depth_checkpoint = sys.argv[1], sys.argv[2]
TOOL_CONFIGS = {
    'roborefer': {'num_actors': 1, 'resources': {'num_gpus': 0.6}, 'conda_env': 'spacetools-tool-roborefer', 'timeout': 600,
        'args': {'model_path': roborefer_model, 'no_output_image': True, 'no_output_vars': False, 'exclude_methods': ['general_query'], 'exclude_behavior': 'error'}},
    'vlm': {'num_actors': 1, 'resources': {'num_gpus': 1.0}, 'conda_env': 'spacetools-tool-vlm', 'timeout': 600,
        'args': {'no_output_image': True, 'no_output_vars': False, 'model_name': 'allenai/Molmo-7B-D-0924', 'dtype': 'auto', 'exclude_methods': ['general_query'], 'exclude_behavior': 'error'}},
    'sam2': {'num_actors': 2, 'resources': {'num_gpus': 0.2}, 'conda_env': 'spacetools-tool-vlm', 'timeout': 600, 'args': {'no_output_image': True, 'no_output_vars': False}},
    'depth_estimator': {'num_actors': 2, 'resources': {'num_gpus': 0.4}, 'conda_env': 'spacetools-tool-vlm', 'timeout': 600,
        'args': {'checkpoint_path': depth_checkpoint, 'no_output_image': True, 'no_output_vars': False}},
    'bounding_box': {'num_actors': 2, 'resources': {'num_gpus': 0.05}, 'conda_env': 'spacetools-tool-bbox', 'timeout': 600, 'args': {'no_output_image': True, 'no_output_vars': False}},
    'vision_ops': {'num_actors': 1, 'resources': {'num_gpus': 0}, 'conda_env': 'spacetools-tool-bbox', 'timeout': 600,
        'args': {'no_output_image': True, 'no_output_vars': False, 'exclude_methods': ['mask_crop', 'point_crop', 'project_2d_points_to_3d', 'project_3d_points_to_2d', 'draw_polygon'], 'exclude_behavior': 'error'}},
    'grasp_generator': {'num_actors': 1, 'resources': {'num_gpus': 0.1}, 'conda_env': 'spacetools-tool-graspgen', 'timeout': 600, 'args': {'no_output_image': True, 'no_output_vars': False}},
    'mock_robot': {'num_actors': 1, 'resources': {'num_gpus': 0}, 'conda_env': 'spacetools-tool-bbox', 'timeout': 600, 'args': {'no_output_image': True, 'no_output_vars': False}},
}
router = start_toolkit(TOOL_CONFIGS, detached=False, dashboard=False)
while True: time.sleep(60)
TOOLSHED_PY
fi

TOOLSHED_PID=$!; sleep 120

TOOL_YAML="$EVAL_DIR/toolshed_config.yaml"
python "$SCRIPT_DIR/generate_toolshed_config.py" --output "$TOOL_YAML" --use-image-by-index

# =============================================================================
# Run evaluations
# =============================================================================
CONFIG_PATH="$VERL_DIR/examples/sglang_multiturn/config"
cd "$VERL_DIR"

for bench in "${RUN_BENCHMARKS[@]}"; do
    echo ""
    echo "=== Evaluating: $bench ==="
    PARQ="$DATA_DIR/${BENCHMARKS[$bench]}"
    OUT_DIR="$EVAL_DIR/$bench"; mkdir -p "$OUT_DIR"

    python3 -m verl.trainer.main_ppo \
        --config-path="$CONFIG_PATH" \
        --config-name='robos_multiturn_grpo' \
        algorithm.adv_estimator=grpo \
        actor_rollout_ref.model.freeze_vision_model=true \
        trainer.val_only=true \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
        data.train_batch_size=32 \
        data.max_prompt_length=8192 \
        data.max_response_length=4096 \
        data.filter_overlong_prompts=False \
        data.truncation='error' \
        data.return_raw_chat=True \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=5 \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.calculate_log_probs=True \
        +trainer.dump_token_diagnostics=True \
        actor_rollout_ref.rollout.n=5 \
        actor_rollout_ref.rollout.max_num_seqs=64 \
        actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
        actor_rollout_ref.rollout.multi_turn.enable=True \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8 \
        actor_rollout_ref.rollout.multi_turn.max_parallel_calls=8 \
        actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
        actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_YAML" \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console"]' \
        trainer.project_name="eval_${VERSION}_${bench}" \
        trainer.experiment_name="eval_${VERSION}" \
        trainer.n_gpus_per_node="$EVAL_GPUS" \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=1 \
        trainer.total_epochs=1 \
        trainer.allow_last_response_fallback=true \
        trainer.default_local_dir="$OUT_DIR/rl_tmp" \
        trainer.validation_data_dir="$OUT_DIR" \
        data.train_files="$PARQ" \
        data.val_files="$PARQ" \
        2>&1 | tee "$OUT_DIR/eval.log"

    echo "=== $bench complete ==="
done

echo ""
echo "=== EVALUATION COMPLETE ==="
echo "Results: $EVAL_DIR"
