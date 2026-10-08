# GFlowRL — everything Step 4 training uses

> ⚠ **`code/` holds snapshots, not the canonical copy.** The canonical copy is in `../../SpaceTools-RL/`,
> and training runs the canonical copy. **Editing files here does not affect any training run.**
> Whether the snapshots are stale: `bash verify_copies.sh`

This directory answers one question: **to get GFlowRL (C′ arm) running, which files are involved, what they do,
and how to confirm they have not been broken.**

```
README.md            ← this file
MANIFEST.txt         sha256 of each file in code/ and its path in the canonical copy
verify_copies.sh     snapshot vs canonical copy, OK / DRIFT per file
run_checks.sh        runs the five self-checks (runs the canonical scripts in spacetools-repro)
code/                snapshots of the implementation and launch scripts
diff/                three patches: two feat commits + the current uncommitted fixes
docs/                why C′, the gate, the fixed-point self-check, the work plan
```

---

## 1. Implementation: two halves

| file | location | what it does |
|---|---|---|
| `verl/trainer/ppo/ray_trainer.py` | `compute_gflowrl_flow_gap()` L221 | **The half without gradients.** Eq.4 within-group log Z → Eq.6 flow gap → Eq.7 asymmetric clip; the result is broadcast into `batch["advantages"]` |
| | L1824 in `fit()` | call site + two guards against "silent mismatch" (`use_kl_loss=False` / `kl_loss_coef≠0` both raise) |
| `verl/trainer/ppo/core_algos.py` | `compute_policy_loss_gflowrl()` L2237 | **The half with gradients.** Eq.8, registered name `"gflowrl"` |
| `verl/workers/config/actor.py` | `PolicyLossConfig.gflowrl` | lets hydra set `policy_loss.gflowrl.*`. **Without this field, launch crashes** |
| `verl/trainer/config/actor/actor.yaml` | `policy_loss.gflowrl: {}` | same as above, yaml-side declaration |

```
d_i   = masked_sum(log π_ref − log π_old)          summed over the whole sequence, not divided by |y|
Z_t   = mean over uid group (β·r_i + d_i)          Eq.4
g_i   = Z_t − d_i − β·r_i                          Eq.6
g̃_i   = clip(g_i, −ε_low, +ε_high)                 Eq.7
L     = mean_i  w_i · (g̃_i + masked_sum(log π_θ − log π_old))²      Eq.8
w_i   = clamp(exp(whole-sequence log ratio).detach(), max = 1 + ε_is)
```

### A config-dependent fact: under the current config `w` is identically 1

`train_batch_size=64` × `rollout.n=5` = 320 sequences, and `ppo_mini_batch_size=64`, after
`fsdp_workers.py:249` multiplies by 5 and divides by the GPU count, equals exactly the number of sequences per GPU ⇒ **each rollout batch is cut into only
one mini-batch**; together with `ppo_epochs=1`, `on_policy` at `dp_actor.py:550` holds, so
`old_log_prob = log_prob.detach()`, and `log_ratio` is numerically identically 0 (the gradient is not 0).

    w = clamp(exp(0), max=1+eps_is) = 1      <- eps_is has no effect
    loss value = mean(g~^2)                   <- gradient = mean(2*g~_i * grad masked_sum(log pi_theta))

Not a defect: a single optimization step has no `pi_theta/pi_old` mismatch to correct in the first place. But **this is config-dependent** —
changing `ppo_mini_batch_size` (the easiest thing to touch to save GPU memory) or `ppo_epochs` flips it.
`p7_onpolicy_check.py` guards this.

**Why this split**: `g̃` depends only on `π_old`, `π_ref`, `r`, all of which are frozen during the actor update,
so computing it once per step on the driver is enough; and on the driver `old_log_probs` / `ref_log_prob` /
`token_level_scores` / `uid` are all readily available. **So the actor side does not need the raw `r`,
and not a single line of `dp_actor.py` was changed.**

### variant

| variant | d in Eq.4 | d in Eq.5/6 | purpose |
|---|---|---|---|
| **`cprime`** | **`d`** | **`d`** | **training arm, default** |
| `paper` | `d` | `d/L` | paper as is, diagnostic arm. Fixed-point loss 4.0e-2, clip 100% saturated |
| `normalized` | `d/L` | `d/L` | both sides normalized, diagnostic arm. 5.1e-2 / 84% |

---

## 2. Launch

```bash
# GRPO arm — as is, zero code changes (but add kl_loss_coef=0 as below)
bash examples/toolshed/run_rl.sh actor_rollout_ref.actor.kl_loss_coef=0

# C′ arm
bash examples/toolshed/run_rl_gflowrl.sh
```

`run_rl_gflowrl.sh` is a **thin wrapper, not a copy of `run_rl.sh`** — the two arms must be comparable, and copying 400 lines
would drift sooner or later. The entire delta is seven hydra overrides, passed through via the `"$@"` at the end of `run_rl.sh`.

Tunable environment variables: `GF_BETA` (default 8.0) · `GF_EPS_LOW` 0.2 · `GF_EPS_HIGH` 0.28 ·
`GF_EPS_IS` 0.2 · `GF_VARIANT` cprime.

A single machine also needs the GPU split set (see §5): `TOOL_GPUS` / `TRAIN_GPUS`.

---

## 3. Run the self-checks before running

```bash
bash run_checks.sh
```

| script | what it catches |
|---|---|
| `p7_config_check` | whether the `policy_loss.gflowrl.*` config path works; whether the keys the wrapper passes match the keys the code reads; **whether the delta between the two arms is only the objective** (overrides going out of bounds and rollout correction being turned on for one side only are caught here) |
| `p7_guard_check` | whether the two silent-mismatch guards actually raise |
| `p7_gpusplit_check` | single-machine GPU split; two-node behavior must be unchanged |
| `p7_fixedpoint_check` | **must be run after changing those two functions.** cprime's loss at the Prop. B.1 fixed point must be ≈ 0 |
| `p7_degenerate_check` | whether C′ gives a nonzero gradient on reward-degenerate groups |
| `p7_onpolicy_check` | whether the batch config still keeps `on_policy` true. If it holds, Eq.8's `w` is identically 1 and `eps_is` has no effect; if not, both become real issues |

These scripts use `ast` / regex to **extract code from the real source files and execute it**, so they test the copy in the repo.
So `run_checks.sh` just cds there and runs the canonical copy; no copies are kept here.

---

## 4. The three places most easily broken by an edit

1. **`masked_sum` must not be swapped for `masked_mean`.** That is the **entire** difference between C′ and `normalized`;
   swapping it raises no error, it just silently trains a different objective. → `p7_fixedpoint_check` jumps from 7.4e-13 to 5e-2.
2. **`w` must be `.detach()`ed.** It is an importance weight correcting the sampling distribution, not the objective being differentiated.
3. **`use_kl_loss` must stay True, `kl_loss_coef` must be 0.**
   `dp_actor.py:528` only puts `ref_log_prob` into the batch when the former is on; and `d` is itself
   the KL-type quantity in Eq.4/6, so adding an explicit KL double-counts it. **The switch decides whether it is computed; the coefficient decides whether it is penalized.**
   Both have guards as a backstop.

---

## 5. Metrics to watch from training step 1

```
gflowrl/clip_saturation        measured 77.9% offline. Staying near 1.0 for long ⇒ g̃ is pinned,
                               the magnitude information in the reward is lost and only the sign remains; then it is no longer GFlowRL
gflowrl/is_weight_min          under the current config these two should be identically 1.0 / 0.0,
gflowrl/is_weight_collapsed_frac   and log_ratio_abs_mean identically 0 — because one rollout batch
                               takes only one optimization step, on_policy holds, old_log_prob = log_prob.detach(),
                               so w = exp(0) = 1 and eps_is has no effect at all (see §5.1).
                               ⚠ They are not diagnostics, they are **assertions**: once they move, someone changed
                               ppo_mini_batch_size or ppo_epochs and on_policy flipped,
                               and then w becomes exp(sum of logprob differences over thousands of tokens), clipped only from above;
                               a per-token drift of −3e-3 is enough to push some sequence's w down to ~0.02 —
                               the sequence is still in the batch but contributes nothing, and the effective batch silently shrinks
gflowrl/Z_t_mean               Var = O(1/G); G=5 is 3.2× noisier than the paper's G=16
reward/degenerate_group_frac   fraction of groups whose r is all identical. **Logged in both arms**; the call site is before the gflowrl
reward/degenerate_rollout_frac branch — if only one arm has this number, it proves nothing.
reward/group_range_mean        the 56.9% / 75.8% P7 cited earlier are estimates on the eval distribution;
                               the real value on the training distribution can only be read from here
gflowrl/variant_is_cprime      must be identically 1.0
[GFlowRL] WARNING              response_mask looks like the attention-mask fallback.
                               Only warns, does not raise, and rollouts that never call tools trigger false positives —
                               do not ignore it, take a look at the num_turns distribution
```

---

## 6. GPU split (must read for single machine)

The original `run_rl.sh` gave **the same `GPUS_PER_NODE`** both to the Toolshed placement group
and to `trainer.n_gpus_per_node`. Two nodes is fine (PG lands on node 1, training on node 2);
**on a single machine the same set of GPUs is requested twice**, the PG grabs all of them, and the training side gets none.

Now split into `TOOL_GPUS` / `TRAIN_GPUS`, **two-node behavior verbatim unchanged**, single-machine default 2 / the rest.

⚠ **When reducing `num_actors`, `TOOL_GPUS` must be raised accordingly.** Ray's `num_gpus` is a **logical reservation**,
unrelated to GPU memory: the original config's seven tools total **7.8 logical GPUs**; cut down to 13 actors it is **3.0**.
If the PG reserves only 2 while the actors need 3.0, the PG becomes ready but the actors never get scheduled — the
assert in the heredoc is there to catch exactly this.

---

## 7. diff/

| file | content |
|---|---|
| `01-gflowrl-policy-loss.patch` | commit `33036a51`, both halves of the implementation + variant switch |
| `02-wrapper-and-guard.patch` | commit `0b5098ae`, wrapper + two guards |
| `03-uncommitted-fixes.patch` | **not yet committed**: `STRICT_PACK` split, `PolicyLossConfig.gflowrl` field, `actor/ppo_kl` definition, two IS weight metrics |
| `HEAD.txt` | HEAD and working tree state at generation time |

---

## 8. docs/

| file | why read it |
|---|---|
| `P7_STEP_C12_RESULTS.md` | **Implementation doc.** How the self-checks were done, results, four risks not ruled out |
| `P7_ROUTE_C_GATE.md` | why C′, the derivation of β's lower bound, the 77.9% saturation rate as a cost that must be reported alongside |
| `P7_ROUTE_C_PLAN.md` | work plan: all hyperparameters, statistical design and pre-registration. **§2 "Implementation: what to change" has been superseded by the actual implementation; do not follow it to make changes** |
| `P7_STEP23_RESULTS.md` | offline measurements of reward degeneracy rate, length distribution, clip saturation |

**One thing not to misread**: `P7_STEP_C12_RESULTS.md` §3 measured that C′ gives a nonzero gradient on reward-degenerate groups
(mean|g̃| 0.167 vs non-degenerate 0.247). That signal **comes entirely from the drift term, none of it from reward** —
it is doing distribution matching, not ranking. **This is a mechanism, not an expected gain.**
