#!/bin/bash
# =============================================================================
# SpaceTools Step 4 with GFlowRL instead of GRPO
#
# This is a THIN WRAPPER around run_rl.sh, not a copy of it.
#
# Why a wrapper: P7 compares two arms — GRPO and GFlowRL — and the comparison is
# only worth anything if EVERYTHING ELSE is identical.  Duplicating run_rl.sh
# would let the two arms drift apart silently over time.  Here the entire delta
# is the OVERRIDES block below, and you can read it in ten seconds.
#
# run_rl.sh ends its python invocation with "$@", so anything this script passes
# is appended to the hydra command line.  Hydra takes the LAST value for a
# repeated key (verified), so kl_loss_coef=0 here overrides the =0.01 in
# run_rl.sh.
#
# Required environment variables (same as run_rl.sh):
#   SFT_CHECKPOINT    — path to SFT checkpoint directory
#   ROBOREFER_MODEL   — path to RoboRefer-8B-SFT model
#   DEPTH_CHECKPOINT  — path to depth_pro.pt checkpoint
#
# Optional (same as run_rl.sh):
#   VERSION / OUTPUT_DIR / GPUS_PER_NODE / SAVE_FREQ
#
# Optional (GFlowRL only — defaults are the C-prime configuration):
#   GF_BETA       — beta in pi_ref * exp(beta*r)             (default 8.0)
#   GF_EPS_LOW    — Eq. 7 lower clip bound                   (default 0.2)
#   GF_EPS_HIGH   — Eq. 7 upper clip bound                   (default 0.28)
#   GF_EPS_IS     — Eq. 8 importance-sampling clip           (default 0.2)
#   GF_VARIANT    — cprime | paper | normalized              (default cprime)
#
# Usage (identical to run_rl.sh, 2 nodes / 16 GPUs):
#   sbatch --nodes=2 --gpus-per-node=8 --exclusive your_wrapper.sh
#     ... where the wrapper activates conda and then runs this script.
#
# See: records/P7_STEP_C12_RESULTS.md   (why cprime; fixed-point self-check)
#      records/P7_ROUTE_C_GATE.md       (the gate that chose cprime)
#      records/P7_GPU_ACCOUNTING.md     (why this needs 16 GPUs, not 8)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GF_BETA="${GF_BETA:-8.0}"
GF_EPS_LOW="${GF_EPS_LOW:-0.2}"
GF_EPS_HIGH="${GF_EPS_HIGH:-0.28}"
GF_EPS_IS="${GF_EPS_IS:-0.2}"
GF_VARIANT="${GF_VARIANT:-cprime}"

# Keep the two arms' outputs apart.  run_rl.sh defaults to experiments/rl_<ver>_<ts>;
# without this a GFlowRL run could resume from a GRPO checkpoint and nobody would notice.
VERSION="${VERSION:-v1}"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/experiments/rl_gflowrl_${GF_VARIANT}_${VERSION}_$(date +%Y%m%d_%H%M%S)}"

# -----------------------------------------------------------------------------
# THE ENTIRE DELTA FROM THE GRPO ARM IS THESE SEVEN LINES
# -----------------------------------------------------------------------------
#
#   loss_mode=gflowrl   switches BOTH halves at once:
#                         - ray_trainer.py  overwrites batch["advantages"] with g~
#                         - core_algos.py   uses Eq. 8 instead of the PPO clip loss
#
#   kl_loss_coef=0      MUST be zero.  d = log pi_ref - log pi_old is already the
#                       KL-like term inside Eq. 4/6; an explicit KL penalty on top
#                       double-counts it and changes the objective.
#
#   use_kl_loss stays True (set by run_rl.sh, NOT overridden here) because
#   dp_actor.py only appends ref_log_prob to the batch under that flag, and
#   GFlowRL needs it.  Looks contradictory, isn't: the switch is what computes
#   ref_log_prob, the coefficient is what penalizes with it.
#
#   Both mistakes are caught at runtime by the guard in ray_trainer.py — it
#   raises rather than training a wrong objective quietly.  Belt and braces.
#
GFLOWRL_OVERRIDES=(
  actor_rollout_ref.actor.policy_loss.loss_mode=gflowrl
  actor_rollout_ref.actor.kl_loss_coef=0
  +actor_rollout_ref.actor.policy_loss.gflowrl.beta="$GF_BETA"
  +actor_rollout_ref.actor.policy_loss.gflowrl.eps_low="$GF_EPS_LOW"
  +actor_rollout_ref.actor.policy_loss.gflowrl.eps_high="$GF_EPS_HIGH"
  +actor_rollout_ref.actor.policy_loss.gflowrl.eps_is="$GF_EPS_IS"
  +actor_rollout_ref.actor.policy_loss.gflowrl.variant="$GF_VARIANT"
)
# -----------------------------------------------------------------------------

cat <<BANNER

===============================================================================
  Step 4 — GFlowRL arm (arXiv:2607.13394 Eq. 4-8)
===============================================================================
  variant        $GF_VARIANT   $( [ "$GF_VARIANT" = cprime ] \
        && echo "(no length normalization — the only variant with the Prop. B.1 fixed point)" \
        || echo "*** NOT cprime — this variant has NO analytic fixed point on ragged lengths ***" )
  beta           $GF_BETA
  eps_low/high   $GF_EPS_LOW / $GF_EPS_HIGH   (asymmetric, per Eq. 7)
  eps_is         $GF_EPS_IS
  kl_loss_coef   0            (use_kl_loss stays True — see comments above)
  output         $OUTPUT_DIR

  Everything else is inherited from run_rl.sh unchanged, including
  rollout.n=5, train_batch_size=64, lr=1e-6, clip_ratio, and the tool config.
  rollout.n is deliberately NOT raised to the paper's G=16: it would make the
  two arms incomparable.  See P7_STEP_C12_RESULTS.md section 4.

  Watch these metrics from step 1:
    gflowrl/clip_saturation   measured 77.9% offline; if it sits near 1.0 the
                              flow gap is pinned and only its SIGN survives
    gflowrl/Z_t_mean          Var = O(1/G); at G=5 this is ~3.2x noisier than
                              the paper's G=16
    gflowrl/variant_is_cprime must be 1.0
===============================================================================

BANNER

if [ "$GF_VARIANT" != "cprime" ]; then
  echo "WARNING: variant='$GF_VARIANT' is a diagnostic arm, not the training arm." >&2
  echo "         Its fixed-point loss is ~4-5e-2, not 0. Continuing in 5s..." >&2
  sleep 5
fi

exec bash "$SCRIPT_DIR/run_rl.sh" "${GFLOWRL_OVERRIDES[@]}" "$@"
