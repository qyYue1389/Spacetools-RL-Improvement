# 04 · GFlowRL implementation (C′ variant) / GFlowRL 实现

GFlowRL ([arXiv:2607.13394](https://arxiv.org/abs/2607.13394)) plugged into [SpaceTools-RL](https://github.com/ChicyChen/SpaceTools-RL) (a verl fork) as a new policy loss. Everything is at the commit that was trained: SpaceTools-RL `c6fef78a` = upstream `54270e82` + the 19 patches here.
把 GFlowRL 作为新的 policy loss 接进 SpaceTools-RL。本目录对应实际训练用的 commit `c6fef78a`(= 上游 `54270e82` + 这里的 19 个 patch)。

## The objective, as implemented / 实现的公式

```
d_i   = masked_sum(log π_ref − log π_old)            summed over the whole response, not divided by |y|
Z_t   = mean over the uid group of (β·r_i + d_i)       Eq. 4
g_i   = Z_t − d_i − β·r_i                              Eq. 6   (flow gap)
g̃_i   = clip(g_i, −ε_low, +ε_high)                     Eq. 7
L     = mean_i  w_i · (g̃_i + masked_sum(log π_θ − log π_old))²      Eq. 8
w_i   = clamp(exp(sequence log-ratio).detach(), max = 1 + ε_is)
```

**C′** = no length normalization anywhere, so the fixed point is exactly π ∝ π_ref · exp(βr) (the paper's per-token normalization turns β into |y|·β ≈ 2700 on this data). Why C′: `design_notes/07_cprime_gate_length_normalization.md`.

## Where the code is / 代码位置

| File (under `spacetools_rl_modified/`) | What |
|---|---|
| `verl/trainer/ppo/ray_trainer.py` → `compute_gflowrl_flow_gap()` | No-grad half on the driver: Eq. 4 group log Z → Eq. 6 flow gap → Eq. 7 clip, written into `batch["advantages"]`; β-decomposition and degeneracy metrics; guards for silent misconfigurations |
| `verl/trainer/ppo/core_algos.py` → `compute_policy_loss_gflowrl()` | Gradient half: Eq. 8, registered as policy loss `"gflowrl"` |
| `verl/workers/config/actor.py`, `verl/trainer/config/actor/actor.yaml` | Declare `policy_loss.gflowrl.*` so Hydra accepts it |
| `examples/toolshed/run_rl_gflowrl.sh` | Thin wrapper over `run_rl.sh`: sets the loss, β, ε, and `kl_loss_coef=0` (`GF_VARIANT`, `GF_BETA`, `GF_EPS_LOW`, `GF_EPS_HIGH`, …) |
| `examples/toolshed/run_rl.sh` | Upstream RL launcher + single-node GPU split, tool-actor scaling, cleanup fixes |
| `examples/toolshed/run_eval.sh` | Upstream eval launcher + GPU budget, dtype, dump fields |

`patches_spacetools_rl/0001–0010` are eval / infrastructure fixes, `0011–0019` are GFlowRL. Apply with `git am` on a clone of upstream at `54270e82`.

## Self-checks / 自检

```bash
SPACETOOLS_RL=/path/to/patched/SpaceTools-RL bash checks/run_checks.sh
```

| Check | Guards |
|---|---|
| `p7_fixedpoint_check.py` | Loss and gradient are exactly 0 at π ∝ π_ref·exp(βr) for C′ (needs torch) |
| `p7_degenerate_check.py` | C′ still produces gradient on reward-degenerate groups, where GRPO gives exactly zero (needs torch) |
| `p7_config_check.py` | Every config key the code reads is passed by the wrapper, and nothing else changes between arms |
| `p7_guard_check.py` | The KL-loss guard fires on both silent misconfigurations |
| `p7_gpusplit_check.py` | Tool / training GPU split on 1 and 2 nodes |
| `p7_onpolicy_check.py` | The update is on-policy (IS weight ≡ 1) for the planned GPU counts |

Other scripts in `checks/` are the offline analyses behind `design_notes/` (Z_t decomposition, estimator comparison, length-normalization gate). They read `../../01_official_checkpoint_eval/p6/` (run `tools/unpack_dumps.sh` first).

## Design notes / 设计记录(Chinese)

`design_notes/01…09` follow the order the decisions were made: route decision → prior-art survey → batch-constant criterion → offline Z_t / synthetic fixed point → response mask and |y| → GPU resampling with log-probs → C′ gate → training plan → implementation and fixed-point check. `IMPLEMENTATION_NOTES.md` is the original walkthrough of the code (its `code/` and `diff/` paths now correspond to `spacetools_rl_modified/` and `patches_spacetools_rl/`).

`migration_acceptance/` holds the blinkdepth run used to accept the move to a new GPU machine before training.
