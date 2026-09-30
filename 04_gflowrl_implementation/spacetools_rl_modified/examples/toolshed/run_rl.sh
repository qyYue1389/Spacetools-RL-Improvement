#!/bin/bash
# =============================================================================
# SpaceTools RL Training (GRPO with multi-turn tool interaction)
#
# Takes an SFT checkpoint and trains with live Toolshed tool execution.
# Data is downloaded automatically from HuggingFace.
#
# Architecture (2 nodes, 16 GPUs):
#   Node 1 (tool node):  8 GPUs dedicated to Toolshed tool actors
#   Node 2 (train node): 8 GPUs for FSDP training + sglang rollout
#
# The script starts a multi-node Ray cluster across all SLURM-allocated nodes,
# then uses a placement group to pack tool actors onto one node while verl
# training runs on the remaining node.
#
# Versions:
#   v1  — 7 tool types (no robot): roborefer, vlm, sam2, depth_estimator,
#          bounding_box, vision_ops, grasp_generator
#   v2  — 8 tool types (V1 + mock_robot)
#
# Required environment variables:
#   SFT_CHECKPOINT    — path to SFT checkpoint directory
#   ROBOREFER_MODEL   — path to RoboRefer-8B-SFT model
#   DEPTH_CHECKPOINT  — path to depth_pro.pt checkpoint
#
# Optional environment variables:
#   VERSION           — v1 or v2 (default: v1)
#   OUTPUT_DIR        — experiment output dir (reuse for resume across submissions)
#   GPUS_PER_NODE     — GPUs per node (default: 8)
#   SAVE_FREQ         — checkpoint save frequency in steps (default: 5, -1 to disable)
#
# SLURM usage (2 nodes):
#   sbatch --nodes=2 --gpus-per-node=8 --exclusive --time=4:00:00 your_wrapper.sh
#
# The wrapper should activate conda and then source this script.
# =============================================================================

set -euxo pipefail

: "${SFT_CHECKPOINT:?Set SFT_CHECKPOINT to path of SFT checkpoint directory}"
: "${ROBOREFER_MODEL:?Set ROBOREFER_MODEL to path of RoboRefer-8B-SFT model}"
: "${DEPTH_CHECKPOINT:?Set DEPTH_CHECKPOINT to path of depth_pro.pt}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_DIR="$(cd "$VERL_DIR/.." && pwd)"

VERSION="${VERSION:-v1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
SAVE_FREQ="${SAVE_FREQ:-5}"

if [[ "$VERSION" == "v1" ]]; then
    TOOL_CONFIG="$SCRIPT_DIR/toolshed_v1_config.yaml"
    HF_RL_DATASET="siyich/spacetools-rlfulltools"
elif [[ "$VERSION" == "v2" ]]; then
    TOOL_CONFIG="$SCRIPT_DIR/toolshed_v2_config.yaml"
    HF_RL_DATASET="siyich/spacetools-rlfulltools"
else
    echo "Unknown VERSION: $VERSION (use v1 or v2)"; exit 1
fi

EXPERIMENT_DIR="${OUTPUT_DIR:-$REPO_DIR/experiments/rl_${VERSION}_$(date +%Y%m%d_%H%M%S)}"
RL_OUTPUT_DIR="$EXPERIMENT_DIR/rl_output"
VAL_OUTPUT_DIR="$EXPERIMENT_DIR/val_outputs"
HF_CACHE_DIR="$EXPERIMENT_DIR/hf_cache"

set +u; conda activate spacetools-rl; set -u

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export HYDRA_FULL_ERROR=1

TOOLSHED_PID=""
GPU_KEEPALIVE_PID=""
WORKER_SRUN_PIDS=()
cleanup() {
    # set -e must not apply in here.  This function kills its own background
    # jobs and then waits on them, so wait returns 128+SIGTERM and set -e aborts
    # cleanup halfway -- before "ray stop --force" -- leaving a live Ray cluster
    # behind and handing a successful run a non-zero exit code.
    set +e
    echo "Cleaning up..."
    [ -n "$GPU_KEEPALIVE_PID" ] && kill $GPU_KEEPALIVE_PID 2>/dev/null
    [ -n "$TOOLSHED_PID" ] && kill $TOOLSHED_PID 2>/dev/null && wait $TOOLSHED_PID 2>/dev/null
    # Kill worker srun keepalive processes (this lets SLURM clean up the Ray workers)
    for pid in "${WORKER_SRUN_PIDS[@]}"; do
        kill $pid 2>/dev/null
    done
    ray stop --force 2>/dev/null || true
}
trap cleanup SIGINT SIGTERM EXIT

mkdir -p "$EXPERIMENT_DIR" "$RL_OUTPUT_DIR" "$VAL_OUTPUT_DIR" "$HF_CACHE_DIR"
echo "=== SpaceTools RL Training ==="
echo "Version:        $VERSION"
echo "SFT checkpoint: $SFT_CHECKPOINT"
echo "Output:         $EXPERIMENT_DIR"
echo "Save freq:      $SAVE_FREQ"
echo "GPUs per node:  $GPUS_PER_NODE"

# Check for existing checkpoints (resume across wall-time resubmissions)
if ls "$RL_OUTPUT_DIR"/global_step_* 1>/dev/null 2>&1; then
    LATEST=$(ls -d "$RL_OUTPUT_DIR"/global_step_* | sort -t_ -k3 -n | tail -1)
    STEP=$(echo "$LATEST" | grep -oP 'global_step_\K\d+')
    echo "=== RESUMING from step $STEP ==="
else
    echo "=== STARTING FROM SCRATCH ==="
fi

cp "$TOOL_CONFIG" "$EXPERIMENT_DIR/tool_config.yaml"

# =============================================================================
# Start multi-node Ray cluster
# =============================================================================
echo "=== Starting Ray cluster ==="

# The P7 token diagnostics in ray_trainer._validate() are switched on by this env
# var, and test_freq makes _validate run during training too.  It is set by hand for
# eval runs, so a shell that ran eval first would silently carry it into training,
# where it sleeps the rollout engine and adds a full compute_log_prob pass per
# validation.  Nothing needs it here.
unset VERL_DUMP_TOKEN_DIAGNOSTICS

ray stop --force 2>/dev/null || true; sleep 5

# Detect SLURM multi-node allocation
if [[ -n "${SLURM_NODELIST:-}" ]]; then
    NODES=( $(scontrol show hostnames "$SLURM_NODELIST") )
else
    NODES=( "$(hostname -s)" )
fi
NUM_NODES=${#NODES[@]}
HEAD_NODE="${NODES[0]}"
# The sbatch script runs on the head node, so get IP directly
HEAD_IP=$(hostname -I | awk '{print $1}')
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

# --- Toolshed / training GPU split -------------------------------------------
# The Toolshed placement group and the trainer used to be handed the SAME number
# (GPUS_PER_NODE).  On two nodes that is correct: the PG lands on node 1, the
# trainer takes node 2.  On ONE node it books the same cards twice -- the PG
# grabs everything and the trainer gets none, or the PG never becomes ready.
if (( NUM_NODES > 1 )); then
    TOOL_GPUS="${TOOL_GPUS:-$GPUS_PER_NODE}"
    TRAIN_GPUS="${TRAIN_GPUS:-$GPUS_PER_NODE}"
else
    TOOL_GPUS="${TOOL_GPUS:-2}"
    TRAIN_GPUS="${TRAIN_GPUS:-$((GPUS_PER_NODE - TOOL_GPUS))}"
    if (( TOOL_GPUS + TRAIN_GPUS > GPUS_PER_NODE )); then
        echo "ERROR: TOOL_GPUS($TOOL_GPUS) + TRAIN_GPUS($TRAIN_GPUS) > GPUS_PER_NODE($GPUS_PER_NODE)" >&2
        exit 1
    fi
    if (( TRAIN_GPUS < 1 )); then
        echo "ERROR: TRAIN_GPUS=$TRAIN_GPUS -- nothing left to train on" >&2
        exit 1
    fi
fi

echo "Nodes:          $NUM_NODES (${NODES[*]})"
echo "Head node:      $HEAD_NODE ($HEAD_IP)"
echo "Total GPUs:     $TOTAL_GPUS"
echo "GPU split:      tools $TOOL_GPUS / training $TRAIN_GPUS (per node)"

export RAY_TMPDIR="/tmp/${USER}/ray"

# Ray workers inherit the raylet cwd, not the driver cwd.  The dataset stores
# image paths relative ("images/xxx.png") and the symlink further down is made
# at $VERL_DIR/images, so unless the raylet itself starts in $VERL_DIR every
# rollout fails with FileNotFoundError on those paths -- and only after ~20
# minutes of model loading, which is an expensive way to find out.  On SLURM the
# sbatch script happened to start in the repo; do not rely on the caller cwd.
cd "$VERL_DIR"

# Clean stale Ray state on all nodes
CONDA_ROOT="${_CONDA_ROOT:-/lustre/fsw/portfolios/nvr/users/siyic/miniconda3}"
RAY_BIN="$CONDA_ROOT/envs/spacetools-rl/bin/ray"
rm -rf "$RAY_TMPDIR" 2>/dev/null || true
CLEANUP_PIDS=()
for ((i=1; i<NUM_NODES; i++)); do
    srun --nodes=1 --ntasks=1 --overlap --nodelist="${NODES[$i]}" \
        bash -c "$RAY_BIN stop --force 2>/dev/null; rm -rf '$RAY_TMPDIR' 2>/dev/null" &
    CLEANUP_PIDS+=($!)
done
for pid in "${CLEANUP_PIDS[@]}"; do wait $pid 2>/dev/null; done
sleep 2

# Start Ray head directly (NOT via srun — srun kills daemon processes when
# the job step exits, which tears down the Ray GCS immediately).
ray start --head \
    --temp-dir="$RAY_TMPDIR" \
    --num-gpus="$GPUS_PER_NODE" \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --include-dashboard=true \
    --node-ip-address="$HEAD_IP" \
    --port=6379
sleep 10

# Verify Ray head is alive before proceeding
if ! ray status 2>/dev/null; then
    echo "ERROR: Ray head failed to start"
    exit 1
fi
echo "Ray head verified OK"

# Start Ray workers on remaining nodes.
# Use srun with an infinite-sleep keepalive so the srun step (and its cgroup)
# stays alive — otherwise SLURM kills the Ray daemon when the step exits.
WORKER_SRUN_PIDS=()
for ((i=1; i<NUM_NODES; i++)); do
    WORK_NODE="${NODES[$i]}"
    echo "Starting Ray worker on $WORK_NODE"
    srun --nodes=1 --ntasks=1 --overlap --nodelist="$WORK_NODE" bash -c "
        eval \"\$($CONDA_ROOT/bin/conda shell.bash hook)\"
        conda activate spacetools-rl
        export RAY_TMPDIR='$RAY_TMPDIR'
        ray start --address='$HEAD_IP:6379' \
            --temp-dir='$RAY_TMPDIR' \
            --num-gpus=$GPUS_PER_NODE
        # Keep the srun step alive so SLURM doesn't kill the Ray worker
        while true; do sleep 3600; done
    " &
    WORKER_SRUN_PIDS+=($!)
done

# Wait for workers to register
echo "Waiting for $NUM_NODES nodes to join Ray cluster..."
for attempt in $(seq 1 30); do
    # Count active node entries (each line like " 1 node_<hex>")
    REGISTERED=$(ray status 2>/dev/null | grep -cP 'node_[a-f0-9]+' || echo 0)
    if [[ "$REGISTERED" -ge "$NUM_NODES" ]]; then
        echo "All $NUM_NODES nodes registered in Ray cluster"
        break
    fi
    if [[ "$attempt" -eq 30 ]]; then
        echo "WARNING: Only $REGISTERED/$NUM_NODES nodes after 150s, proceeding anyway"
    fi
    sleep 5
done

export RAY_ADDRESS="$HEAD_IP:6379"

# GPU keepalive: prevent cluster from killing job due to "idle" GPUs
python3 -c "
import torch, time
while True:
    try:
        for i in range(torch.cuda.device_count()):
            x = torch.ones(1, device=f'cuda:{i}')
            del x
    except: pass
    time.sleep(30)
" &
GPU_KEEPALIVE_PID=$!

# =============================================================================
# Start Toolshed (packed onto one node via placement group)
# =============================================================================
echo "=== Starting Toolshed ==="

if [[ "$VERSION" == "v1" ]]; then
    # V1: 7 tool types (no robot)
    python3 - "$ROBOREFER_MODEL" "$DEPTH_CHECKPOINT" "$TOOL_GPUS" <<'TOOLSHED_PY' &
import ray, time, sys
ray.init(address="auto")
from toolshed import start_toolkit
roborefer_model, depth_checkpoint = sys.argv[1], sys.argv[2]
tool_gpus = float(sys.argv[3])
TOOL_CONFIGS = {
    'roborefer': {'num_actors': 6, 'resources': {'num_gpus': 0.6}, 'conda_env': 'spacetools-tool-roborefer', 'timeout': 600,
        'args': {'model_path': roborefer_model, 'no_output_image': True, 'no_output_vars': False, 'exclude_methods': ['general_query'], 'exclude_behavior': 'error'}},
    'vlm': {'num_actors': 2, 'resources': {'num_gpus': 0.6}, 'conda_env': 'spacetools-tool-vlm', 'timeout': 600,
        'args': {'no_output_image': True, 'no_output_vars': False, 'model_name': 'allenai/Molmo-7B-D-0924', 'dtype': 'float16', 'exclude_methods': ['general_query'], 'exclude_behavior': 'error'}},
    'sam2': {'num_actors': 5, 'resources': {'num_gpus': 0.2}, 'conda_env': 'spacetools-tool-vlm', 'timeout': 600,
        'args': {'no_output_image': True, 'no_output_vars': False}},
    'depth_estimator': {'num_actors': 5, 'resources': {'num_gpus': 0.2}, 'conda_env': 'spacetools-tool-vlm', 'timeout': 600,
        'args': {'checkpoint_path': depth_checkpoint, 'no_output_image': True, 'no_output_vars': False}},
    'bounding_box': {'num_actors': 5, 'resources': {'num_gpus': 0.1}, 'conda_env': 'spacetools-tool-bbox', 'timeout': 600,
        'args': {'no_output_image': True, 'no_output_vars': False}},
    'vision_ops': {'num_actors': 8, 'resources': {'num_gpus': 0}, 'conda_env': 'spacetools-tool-bbox', 'timeout': 600,
        'args': {'no_output_image': True, 'no_output_vars': False, 'exclude_methods': ['mask_crop', 'point_crop', 'project_2d_points_to_3d', 'project_3d_points_to_2d', 'draw_polygon'], 'exclude_behavior': 'error'}},
    'grasp_generator': {'num_actors': 5, 'resources': {'num_gpus': 0.1}, 'conda_env': 'spacetools-tool-graspgen', 'timeout': 600,
        'args': {'no_output_image': True, 'no_output_vars': False}},
}
print(f"Launching {len(TOOL_CONFIGS)} tool types (V1, no robot)...", flush=True)
_demand = sum(c['num_actors'] * c['resources']['num_gpus'] for c in TOOL_CONFIGS.values())
# The num_actors above were sized for the paper's 2-node run, where the tools own
# a whole 8-GPU node (demand 7.8 <= 8).  A single-node run has to leave cards for
# the trainer, so tool_gpus is 4 at most and the assert below simply refused to
# start.  Scale the counts to fit instead of making the operator hand-edit this
# dict in two places (v1 and v2) and drift them apart.
#
# On the 2-node path tool_gpus=8, _demand=7.8, and this block is not entered --
# every count and fraction stays exactly as the paper had it.
if _demand > tool_gpus:
    # vlm's 0.6 was sized for 80 GB cards.  Measured single-instance peak on this
    # machine is 32.11 GiB, because vlm.py:116 pins dtype='auto' and Molmo's config
    # names no dtype, so it loads fp32 regardless of the 'float16' passed above.
    # That is 0.70 of a 46 GB A40.  At 0.6 Ray is free to put vlm + two
    # depth_estimators on one card (0.6+0.2+0.2 = 1.0 logical, 47.9 GiB physical)
    # and the last one OOMs.  0.7 makes the logical fraction match the real
    # footprint, which is what keeps Ray's packing honest.
    _vlm = TOOL_CONFIGS.get('vlm')
    if _vlm is not None:
        _vlm['resources']['num_gpus'] = max(_vlm['resources']['num_gpus'], 0.7)
        _demand = sum(c['num_actors'] * c['resources']['num_gpus'] for c in TOOL_CONFIGS.values())
    _scale = tool_gpus / _demand
    for _name, _c in TOOL_CONFIGS.items():
        _c['num_actors'] = max(1, int(_c['num_actors'] * _scale))
    _demand = sum(c['num_actors'] * c['resources']['num_gpus'] for c in TOOL_CONFIGS.values())
    print("scaled tool actors by %.3f to fit tool_gpus=%s: %s -> demand %.2f logical GPUs" % (
        _scale, tool_gpus,
        {k: v['num_actors'] for k, v in TOOL_CONFIGS.items()}, _demand), flush=True)
assert _demand <= tool_gpus, (
    f"tool actors need {_demand} logical GPUs but the placement group reserves only {tool_gpus}. "
    f"Raise TOOL_GPUS or cut num_actors -- otherwise the PG goes ready and the actors queue forever.")
pg = ray.util.placement_group([{"CPU": 16, "GPU": tool_gpus}], strategy="STRICT_PACK")
ray.get(pg.ready())
print("Placement group ready (tools packed on one node)", flush=True)
router = start_toolkit(TOOL_CONFIGS, detached=False, dashboard=False, placement_group=pg)
print(f"Toolshed started: {list(TOOL_CONFIGS)}", flush=True)
while True: time.sleep(60)
TOOLSHED_PY
else
    # V2: 8 tool types (V1 + mock_robot)
    python3 - "$ROBOREFER_MODEL" "$DEPTH_CHECKPOINT" "$TOOL_GPUS" <<'TOOLSHED_PY' &
import ray, time, sys
ray.init(address="auto")
from toolshed import start_toolkit
roborefer_model, depth_checkpoint = sys.argv[1], sys.argv[2]
tool_gpus = float(sys.argv[3])
TOOL_CONFIGS = {
    'roborefer': {'num_actors': 6, 'resources': {'num_gpus': 0.6}, 'conda_env': 'spacetools-tool-roborefer', 'timeout': 600,
        'args': {'model_path': roborefer_model, 'no_output_image': True, 'no_output_vars': False, 'exclude_methods': ['general_query'], 'exclude_behavior': 'error'}},
    'vlm': {'num_actors': 2, 'resources': {'num_gpus': 0.6}, 'conda_env': 'spacetools-tool-vlm', 'timeout': 600,
        'args': {'no_output_image': True, 'no_output_vars': False, 'model_name': 'allenai/Molmo-7B-D-0924', 'dtype': 'float16', 'exclude_methods': ['general_query'], 'exclude_behavior': 'error'}},
    'sam2': {'num_actors': 5, 'resources': {'num_gpus': 0.2}, 'conda_env': 'spacetools-tool-vlm', 'timeout': 600,
        'args': {'no_output_image': True, 'no_output_vars': False}},
    'depth_estimator': {'num_actors': 5, 'resources': {'num_gpus': 0.2}, 'conda_env': 'spacetools-tool-vlm', 'timeout': 600,
        'args': {'checkpoint_path': depth_checkpoint, 'no_output_image': True, 'no_output_vars': False}},
    'bounding_box': {'num_actors': 5, 'resources': {'num_gpus': 0.1}, 'conda_env': 'spacetools-tool-bbox', 'timeout': 600,
        'args': {'no_output_image': True, 'no_output_vars': False}},
    'vision_ops': {'num_actors': 8, 'resources': {'num_gpus': 0}, 'conda_env': 'spacetools-tool-bbox', 'timeout': 600,
        'args': {'no_output_image': True, 'no_output_vars': False, 'exclude_methods': ['mask_crop', 'point_crop', 'project_2d_points_to_3d', 'project_3d_points_to_2d', 'draw_polygon'], 'exclude_behavior': 'error'}},
    'grasp_generator': {'num_actors': 5, 'resources': {'num_gpus': 0.1}, 'conda_env': 'spacetools-tool-graspgen', 'timeout': 600,
        'args': {'no_output_image': True, 'no_output_vars': False}},
    'mock_robot': {'num_actors': 2, 'resources': {'num_gpus': 0}, 'conda_env': 'spacetools-tool-bbox', 'timeout': 600,
        'args': {'no_output_image': True, 'no_output_vars': False}},
}
print(f"Launching {len(TOOL_CONFIGS)} tool types (V2, with mock_robot)...", flush=True)
_demand = sum(c['num_actors'] * c['resources']['num_gpus'] for c in TOOL_CONFIGS.values())
# The num_actors above were sized for the paper's 2-node run, where the tools own
# a whole 8-GPU node (demand 7.8 <= 8).  A single-node run has to leave cards for
# the trainer, so tool_gpus is 4 at most and the assert below simply refused to
# start.  Scale the counts to fit instead of making the operator hand-edit this
# dict in two places (v1 and v2) and drift them apart.
#
# On the 2-node path tool_gpus=8, _demand=7.8, and this block is not entered --
# every count and fraction stays exactly as the paper had it.
if _demand > tool_gpus:
    # vlm's 0.6 was sized for 80 GB cards.  Measured single-instance peak on this
    # machine is 32.11 GiB, because vlm.py:116 pins dtype='auto' and Molmo's config
    # names no dtype, so it loads fp32 regardless of the 'float16' passed above.
    # That is 0.70 of a 46 GB A40.  At 0.6 Ray is free to put vlm + two
    # depth_estimators on one card (0.6+0.2+0.2 = 1.0 logical, 47.9 GiB physical)
    # and the last one OOMs.  0.7 makes the logical fraction match the real
    # footprint, which is what keeps Ray's packing honest.
    _vlm = TOOL_CONFIGS.get('vlm')
    if _vlm is not None:
        _vlm['resources']['num_gpus'] = max(_vlm['resources']['num_gpus'], 0.7)
        _demand = sum(c['num_actors'] * c['resources']['num_gpus'] for c in TOOL_CONFIGS.values())
    _scale = tool_gpus / _demand
    for _name, _c in TOOL_CONFIGS.items():
        _c['num_actors'] = max(1, int(_c['num_actors'] * _scale))
    _demand = sum(c['num_actors'] * c['resources']['num_gpus'] for c in TOOL_CONFIGS.values())
    print("scaled tool actors by %.3f to fit tool_gpus=%s: %s -> demand %.2f logical GPUs" % (
        _scale, tool_gpus,
        {k: v['num_actors'] for k, v in TOOL_CONFIGS.items()}, _demand), flush=True)
assert _demand <= tool_gpus, (
    f"tool actors need {_demand} logical GPUs but the placement group reserves only {tool_gpus}. "
    f"Raise TOOL_GPUS or cut num_actors -- otherwise the PG goes ready and the actors queue forever.")
pg = ray.util.placement_group([{"CPU": 16, "GPU": tool_gpus}], strategy="STRICT_PACK")
ray.get(pg.ready())
print("Placement group ready (tools packed on one node)", flush=True)
router = start_toolkit(TOOL_CONFIGS, detached=False, dashboard=False, placement_group=pg)
print(f"Toolshed started: {list(TOOL_CONFIGS)}", flush=True)
while True: time.sleep(60)
TOOLSHED_PY
fi

TOOLSHED_PID=$!
echo "Toolshed PID: $TOOLSHED_PID"
sleep 120

RL_TOOL_YAML="$EXPERIMENT_DIR/rl_toolshed_config.yaml"
python "$SCRIPT_DIR/generate_toolshed_config.py" \
    --output "$RL_TOOL_YAML" \
    --use-image-by-index

# =============================================================================
# Download data from HuggingFace (skip datasets already present)
# =============================================================================
RL_PARQUET="$EXPERIMENT_DIR/rl_data/data/train.parquet"
VAL_PARQ="$EXPERIMENT_DIR/eval_data/data/robospatial.parquet"

echo "=== Checking RL + eval data ==="
python3 - "$HF_CACHE_DIR" "$EXPERIMENT_DIR" "$HF_RL_DATASET" "$RL_PARQUET" "$VAL_PARQ" <<'DL_PY'
import sys, os, time
from huggingface_hub import snapshot_download
cache_dir, exp_dir, rl_dataset = sys.argv[1], sys.argv[2], sys.argv[3]
rl_parq, val_parq = sys.argv[4], sys.argv[5]

def download_with_retry(repo_id, local_dir, max_retries=3):
    for attempt in range(max_retries):
        try:
            snapshot_download(repo_id, repo_type="dataset",
                cache_dir=cache_dir, local_dir=local_dir)
            return
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                print(f"Download failed ({e}), retrying in {wait}s...", flush=True)
                time.sleep(wait)
            else:
                raise

if os.path.isfile(rl_parq):
    print(f"RL data already present: {rl_parq}")
else:
    print(f"Downloading RL data: {rl_dataset}")
    download_with_retry(rl_dataset, os.path.join(exp_dir, "rl_data"))

if os.path.isfile(val_parq):
    print(f"Eval data already present: {val_parq}")
else:
    print(f"Downloading eval data: siyich/spacetools-eval-benchmarks")
    download_with_retry("siyich/spacetools-eval-benchmarks", os.path.join(exp_dir, "eval_data"))

print("All data ready")
DL_PY
CONFIG_PATH="$VERL_DIR/examples/sglang_multiturn/config"

# =============================================================================
# RL training (training uses one full node = 8 GPUs)
# =============================================================================
echo "=== Starting RL training ==="
cd "$VERL_DIR"

# Symlink images directory so relative paths in parquet resolve correctly
# (the dataset uses relative paths like "images/xxx.png" from CWD)
if [[ -d "$EXPERIMENT_DIR/rl_data/images" ]]; then
    ln -sfn "$EXPERIMENT_DIR/rl_data/images" "$VERL_DIR/images"
    echo "Symlinked images: $VERL_DIR/images -> $EXPERIMENT_DIR/rl_data/images"
fi

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='robos_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.freeze_vision_model=true \
    actor_rollout_ref.model.path="$SFT_CHECKPOINT" \
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
    data.train_batch_size=64 \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=8 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$RL_TOOL_YAML" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name="spacetools_rl_${VERSION}" \
    trainer.experiment_name="rl_${VERSION}" \
    trainer.n_gpus_per_node="$TRAIN_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.allow_last_response_fallback=true \
    trainer.default_local_dir="$RL_OUTPUT_DIR" \
    trainer.validation_data_dir="$VAL_OUTPUT_DIR" \
    data.train_files="[$RL_PARQUET]" \
    data.val_files="$VAL_PARQ" \
    "$@"

echo ""
echo "=== RL TRAINING COMPLETE ==="
echo "Output:         $RL_OUTPUT_DIR"
echo "Val outputs:    $VAL_OUTPUT_DIR"
if ls "$RL_OUTPUT_DIR"/global_step_* 1>/dev/null 2>&1; then
    LATEST=$(ls -d "$RL_OUTPUT_DIR"/global_step_* | sort -t_ -k3 -n | tail -1)
    STEP=$(echo "$LATEST" | grep -oP 'global_step_\K\d+')
    echo "Latest ckpt:    step $STEP"
fi
echo ""
echo "Next step: evaluate with:"
echo "  bash examples/toolshed/run_eval.sh $RL_OUTPUT_DIR/global_step_XXX"
