# P7 Route C — steps 1&2 results: GFlowRL implementation and fixed-point self-check

Status: **passed**. Implementation landed, fixed point exactly 0, gradient direction correct.
Date: 2026-09-03
Code: `SpaceTools-RL` @ f945bb6b + 2 uncommitted changes
Scripts: `tools/p7/p7_fixedpoint_check.py`, `tools/p7/p7_degenerate_check.py`

---

## 1. Where the implementation lands: two places

Per the design in P7_ROUTE_C_PLAN, the loss is split into two halves: **the half without gradients is computed once on the driver, the half with gradients is computed in the actor.**

### 1.1 `verl/trainer/ppo/ray_trainer.py` — `compute_gflowrl_flow_gap()`

Runs once per training step on the driver, producing Eq. 4 → Eq. 6 → Eq. 7:

- `Z_t` = within-group `mean(β·r_i + d_i)` (in-batch MC estimate of Eq. 4, grouped by `uid`)
- `g_i = Z_t − d_i − β·r_i` (Eq. 6)
- `g̃_i = clip(g_i, −ε_low, +ε_high)` (Eq. 7, asymmetric)
- Written into `batch["advantages"]`, broadcast to the token dimension by `response_mask`

`variant` picks one of three; it is the core switch of this reproduction:

| variant | d in Eq. 4 | d in Eq. 5/6 | notes |
|---|---|---|---|
| `paper` | `d` (not normalized) | `d/L` | paper as is |
| `normalized` | `d/L` | `d/L` | both sides normalized |
| **`cprime`** | **`d`** | **`d`** | **neither side normalized ← this is what we train with** |

The call site hooks into `fit()` right after the existing `compute_advantage(...)`, and only fires when `policy_loss.loss_mode == "gflowrl"`; without this switch, the code path is verbatim identical to before.

### 1.2 `verl/trainer/ppo/core_algos.py` — `compute_policy_loss_gflowrl()`

Registered with `@register_policy_loss("gflowrl")`; the gradient-carrying part of Eq. 8:

```
L = mean_i  w_i · ( g̃_i + [log π_θ(y_i|x) − log π_old(y_i|x)] )²
w_i = min( π_θ/π_old , 1+ε_is )        # stop-gradient
```

Two implementation points:

- **`masked_sum`, not `masked_mean`.** This is the **entire** difference between C′ and the paper's formulation. Using `masked_mean` here would silently turn it into the `normalized` config, **with no error** — this is the line of the implementation most easily broken by an edit; before changing it, read the fixed-point check in §2.
- `w_i` has `.detach()`. It is an importance weight correcting the sampling distribution, not part of the objective being differentiated.

### 1.3 A foolproofing guard

`compute_gflowrl_flow_gap` includes a check on `response_mask`: if the fraction of samples in the whole batch with "policy tokens < non-pad tokens" is < 50%, it prints a WARNING.

It targets the silent fallback at `ray_trainer.py:157` — when `response_mask` is missing it degrades to the attention mask, `|y_i|` then **also counts the tokens returned by tools**, `d` and `L` are both wrong, and there is no error.

**Known false-positive risk:** a rollout that never calls a tool naturally satisfies `L == non-pad`. If most rollouts in some batch do not call tools, this WARNING fires falsely. It only warns and does not raise, so the worst case is noise; but **do not just ignore it when you see it**; take a look at the `num_turns` distribution.

---

## 2. Fixed-point self-check (step 2)

### 2.1 Method: test the code in the repo, not a copy of it

The script uses `ast.get_source_segment` to extract the source text of the two functions from the two **real source files**, then `exec`s them into a stubbed namespace (`masked_sum` / `masked_mean` / `DataProto` are stubs).

The reason for this detour: **if you retype the loss and test that, you are testing whether the retyped copy is correct, not whether the one in the repo is.** The latter is what will run on the GPU machine.

### 2.2 Construction

Prop. B.1 says the fixed point of Eq. 8 is `π_θ(y|x) = π_ref(y|x)·exp(β·r(x,y))/Z(x)`. At that point, for every sampled trajectory

```
log π_θ(y_i) − log π_ref(y_i) = β·r_i − log Z(x)
```

So we construct this π_θ **analytically**: given any `π_ref` and any `r`, set `log π_θ = log π_ref + (β·r − log Z)/|y_i|`, spread per token. Use Eq. 4's in-batch constant as `log Z`.

Setup: `G=8`, 4 prompts, sequence lengths 12–40, **random and unequal** (unequal is the key; equal lengths would let all three variants pass and detect nothing), with one group having all-identical rewards (degenerate group). Construction precision `max|err| = 3.8e-6`.

### 2.3 Results

| variant | `max\|g̃_i\|` | clip saturation | loss | fixed point |
|---|---|---|---|---|
| **`cprime`** | **2.4e-06** | **0.0000** | **7.4e-13** | **PASS** |
| `paper` | 2.0e-01 (pinned at ε_low) | 1.0000 | 4.0e-02 | FAIL |
| `normalized` | 2.8e-01 (pinned at ε_high) | 0.8438 | 5.1e-02 | FAIL |

**`cprime` is exactly 0** (7.4e-13 is fp32 noise), and the clip never activates — consistent with Remark B.3.

**The FAILs of the other two are not bugs; they are the first independent reproduction, through the real code path, of the conclusion from P7 steps 2–3.** Before, it was computed by an offline script; this time it was computed by functions extracted from the repo source, and the conclusion is the same:

- `paper` mixes `Z_t` (sequence-sum scale) and `d/L` (per-token scale); the two sides differ in scale by a factor of `L`, so `g` pins the clip outright, 100% saturation.
- `normalized` has consistent scales on both sides, but `g_i = mean_j(β·r_j + d_j/L_j) − (β·r_i + d_i/L_i)`, while what is constant at the fixed point is `β·r + d`, not `β·r + d/L` — `d/L` varies with `L`. **As long as lengths are unequal, the analytic fixed point does not exist.**

This is exactly why `paper` / `normalized` were rejected and C′ was chosen. Now that reason has an executable form: **if anyone changes these two functions, run `p7_fixedpoint_check.py`; if the loss is no longer 0, the change is wrong.**

### 2.4 Gradient check

Perturb θ away from the fixed point by 0.01:

- loss = 1.64e-03 > 0 ✓
- ‖∂L/∂log π‖ = 7.46e-02 > 0 ✓
- **gradient at padding positions = 0.000e+00** (exact) ✓
- IS weight truncation fraction = 0.0000 (should not truncate under a small perturbation) ✓

---

## 3. Side confirmation: C′ does not give zero gradient on reward-degenerate groups

P6 measured 56.3–79.0% of groups with all-identical rewards. GRPO's advantage is `(r − mean)/std`, so **the gradient on these groups is exactly 0** — more than half of the rollout budget is thrown away.

Measured with the same real code (`p7_degenerate_check.py`, 6 groups, 3 degenerate, `β=8`, with policy drift):

| | mean \|g̃\| | GRPO advantage |
|---|---|---|
| degenerate groups (50% of rollouts) | **0.1671** | **0 (discarded)** |
| non-degenerate groups | 0.2467 | 0.81–0.91 |
| fraction of rollouts with `\|g̃\| < 1e-6` | **0.0000** | 0.5000 |

The gradient C′ gives on degenerate groups is **0.68 times** that on non-degenerate groups — not zero, and not noise-level either.

**But we must be clear about what it is and what it is not.**

On degenerate groups `r` is constant, so `g_i = mean_j(d_j) − d_i`; **the signal comes entirely from the drift term `d = log π_ref − log π_old`, none of it from reward.** What it does is pull the drift of the G trajectories in the group toward the group mean, i.e. **drag π_θ toward the distribution `π_ref·exp(βr)/Z`**.

So: it is **not** telling apart "which answer is better" in these groups — the rewards are all identical, there is no better or worse to distinguish. It is the distribution-matching term, shaping, not ranking.

This is a **concrete, mechanism-level** reason GFlowRL might beat GRPO (it uses the half of the budget GRPO throws away), but there is still a step between it and "so accuracy will go up", and that step can only come out of a training run. **Do not write this into any conclusion as an expected gain.**

---

## 4. Risks this step could not rule out

Per the §3.3 discipline, report the upper bounds and costs alongside:

1. **High clip saturation.** 72.9% on synthetic data; the C′ gate measured 77.9% on real rollouts. When `g̃` is pinned at ±ε, the loss becomes `(±ε + log_ratio)²` — **the magnitude information in the reward is mostly lost; only the sign survives.** This is a known cost of C′ (P7_ROUTE_C_GATE.md), not something this step can solve, but during training `gflowrl/clip_saturation` must be logged every step. If it stays near 1.0 for long, we are effectively running a sign-only KL regularizer, not GFlowRL.
2. **`β` has only a lower bound.** The gate derived `drift/reward ∝ 1/β² ⇒ β ≥ 2`; there is no upper bound. Default is 8.0; **this is a value inherited from the paper, not one we measured.**
3. **Synthetic is not real.** Everything above is synthetic rollouts. In real data the distributions of `d` and `L` and the degeneracy rate will all differ. Step 2 proves **the code implements the math it claims to implement**, not that this math works on SpaceTools.
4. **`ref_log_prob` must actually be in the batch.** `dp_actor.py:528` only appends it when `use_kl_loss` is on. If this switch is forgotten when gflowrl is on, `compute_gflowrl_flow_gap` raises KeyError — this one crashes, which makes it a good error, but it is worth knowing in advance.

---

## 5. Next steps

Steps 1&2 done. From step 3 on, GPU and data are needed:

- Download SFT data (6.32 GB) + RL data (3.38 GB)
- Run SFT (4 GPUs about 8–12 h, no resident tools needed) → SFT baseline eval
- **Smoke test whether Step 4 fits into 4×A100-40GB** (this is currently the biggest unknown; tools occupying GPUs + 60.6 GB optimizer state)
- GFlowRL training → Table 2 nine-benchmark eval → compare against the SFT baseline and the P4 numbers

`model_dtype=bf16` (deviation [20]) **is never carried into training**.
