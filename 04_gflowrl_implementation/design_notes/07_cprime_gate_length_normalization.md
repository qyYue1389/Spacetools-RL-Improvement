# C′ gate: the three length normalization configs, compared once on real data

> Prerequisite gate for Route C. **Zero GPU.** Written 2026-09-02.
> Script `tools/p7/p7_cprime_gate.py` (stdlib only, can be rerun directly).
> Data: real `|y|` (tokens), real rewards, and real two-sided logprob differences from `p6/passk2/`.

---

## 0. Why this gate exists

A′'s conclusion: the paper as is (A) and "normalized `Eq.4`" (B) each have one illness, while
**"no normalization on either side" (C′) is the only one of the three healthy on both metrics**
(the three-config table in `P7_GPU_RESULTS.md` §6). But C′'s cost is exactly the problem the paper introduced length normalization
to solve in the first place — original text: *"Reasoning rollouts can span thousands of tokens, causing
instability as log-probabilities scale with length; we normalize by response length
to **prevent long sequences from dominating the loss**."*

**This doc measures that cost, on our own length distribution.**

The three configs (`d_i := log π_ref − log π_old`):

    A   paper as is      Z_t = mean_j(β·r_j + d_j)        g_i = Z_t − d_i/L_i − β·r_i
    B   normalized Eq.4  Z_t = mean_j(β·r_j + d_j/L_j)    g_i = Z_t − d_i/L_i − β·r_i
    C′  neither side     Z_t = mean_j(β·r_j + d_j)        g_i = Z_t − d_i     − β·r_i

**Length dominance happens in the update term of Eq. 8, not in the flow gap** — the flow gap is clamped by the clip,
the update term is not. This determines how this doc measures it.

> ⚠ **`d_i` is a proxy, not the true value.** No training happened in passk2 (`π_old = π_ref`), so the real drift is identically 0.
> Here it is replaced by the **logprob difference between rollout (sglang) and trainer (FSDP)**, comparable in units and magnitude
> (exactly the quantity the IS weight `w_i` was set up for), **but it is not training drift**. See §5.

---

## 1. Verdict: **gate passed, C′ can be the training arm for Route C**

Three items, in order: the cost, the benefit, and a cost that must be stated alongside.

---

## 2. Cost: length dominance is mild, because our trajectories are short

`|y|` (policy-generated tokens, real counts from `patches/rl/0009`):

| | median | mean | min | max | max/min |
|---|--:|--:|--:|--:|--:|
| `robospatial` | 334 | 350 | 45 | 900 | 20.0× |
| `blinkdepth` | 423 | 441 | 125 | 1240 | 9.9× |
| `boppose` | 305 | 317 | 294 | 973 | 3.3× |

Update term = `(1/L)·Σδ_t` (A, B) or `Σδ_t` (C′); loss ∝ the square of that term.
Both assumptions on `δ` are reported: **independent** across tokens (∝√L) and fully **aligned** (∝L).

**Share of the loss taken by the longest 10% of rollouts** (even share = 10%):

| | normalized · independent | normalized · aligned | unnormalized · independent | **unnormalized · aligned (worst)** |
|---|--:|--:|--:|--:|
| `robospatial` | 5.4% (0.54×) | 10.0% | 16.4% (1.64×) | **24.2% (2.42×)** |
| `blinkdepth` | 6.5% (0.65×) | 10.0% | 15.1% (1.51×) | **22.8% (2.28×)** |
| `boppose` | 7.9% (0.79×) | 10.0% | 13.7% (1.37×) | **20.7% (2.07×)** |

**Within group** (what actually affects the gradient direction for a given prompt is the relative weight within the group):
the longest rollout relative to the group mean is only **1.02–1.15× (independent) / 1.04–1.32× (aligned)**.

### 2.1 Why so mild: our length spread is small

Sensitivity of the same calculation on a log-uniform length distribution (reference only, **not our shape**):

    max/min        1×      3×     10×     20×    100×
    longest 10%   10.0%   22%     36%     45%     ~57–60%

**Our real distribution is much more concentrated than log-uniform** — median `|y|` is 305–423 tokens, **not the
"thousands of tokens" the paper worries about**. So even with max/min reaching 20×, the share is still only 24.2%.

> **An observation in the other direction: normalization is not neutral either.** With independent tokens, normalization pushes the longest 10%
> down to **0.54–0.79×** — it **over-weights short sequences**. So this is not a choice of "biased vs unbiased";
> it is a choice between two biases pointing in opposite directions.

---

## 3. Benefit: under C′ the reward signal is not drowned by drift

Within-group variance decomposition of `g` (`g_i = [Z − β·r_i] + [drift term]`, β=8):

| | groups | Var(reward term) | Var(drift term) A | Var(drift term) C′ | **drift/reward C′** |
|---|--:|--:|--:|--:|--:|
| `robospatial` non-degenerate | 153 | 10.24 | 2.9e-06 | 0.523 | **0.042** |
| `robospatial` degenerate | 197 | 0 | 2.7e-06 | 0.265 | — (reward variance is 0) |
| `blinkdepth` non-degenerate | 26 | 10.24 | 1.9e-06 | 0.504 | **0.041** |
| `blinkdepth` degenerate | 98 | 0 | 1.9e-06 | 0.368 | — |
| `boppose` non-degenerate | 25 | 0.038 | 1.8e-07 | 0.016 | 0.555 |

**Where there is a reward signal, reward dominates drift by about 24 : 1** (0.041–0.042 on both accuracy benchmarks).
`boppose`'s 0.555 is because its reward is continuous IoU with within-group variance of only 0.038 —
and per `P7_DECISION.md` §0.2 it is excluded from P7's scope anyway.

**On degenerate groups with no reward signal** (56–79% of prompts), the contrast is sharp:

    A   drift variance 2.7e-06  —  effectively nothing, and a **common shift**
                                -> 30.0% (robospatial) / 49.2% (blinkdepth) of groups clipped to the same value
    C′  drift variance 0.26     —  discriminative, **whole-group same-value rate 0.0%**

The latter is exactly what TB should do: even when the rewards are identical, the deviation of `π_old` from `π_ref` should still be corrected.
**A does nothing on these groups; C′ does.**

### 3.1 One thing to state honestly: C′ has the highest saturation rate

| config | saturation rate (robospatial / blinkdepth / boppose) | whole group same value |
|---|---|---|
| A paper as is | 73.7% / 70.8% / 21.7% | **30.0% / 49.2% / 5.0%** |
| B normalized Eq.4 | 43.7% / 21.0% / 16.3% | 0.0% |
| **C′ normalize neither** | **77.9% / 77.1% / 28.0%** | **0.0%** |

Once the drift term enters `g` unnormalized, its magnitude (median of the whole-sequence sum 0.42 / 0.50 nats) already exceeds the clip half-width 0.28,
so C′ has the **most** clipped rollouts.

> **The trade-off is explicit: C′ trades "more rollouts clipped to ±ε" for "no group has its reward erased entirely".**
> And where there is reward, reward still dominates drift 24:1. **This trade is worth it.**

---

## 4. ⚠ A coupling: β cannot be chosen independently, and the earlier recommendation does not hold

`Var(reward term) = β²·Var(r)`, while `Var(drift term)` does not depend on β. So

    drift/reward ∝ 1/β²        measured 0.042 at β=8 (robospatial)

Extrapolating:

    β = 8    0.042      reward dominates 24×
    β = 4    0.168      reward dominates 6×
    β = 2    0.672      reward dominates 1.5×
    β = 1.64 1.00       **critical: drift and reward equal**
    β = 1    2.69       **drift dominates reward 2.7×**

**So the item in `P7_STEP23_RESULTS.md` §1.4, "β should start near 1, not 8", does not hold under C′.**
That item was derived from clip saturation at a **zero-drift** starting point; under C′ saturation is mainly driven by drift,
β has a much smaller effect on saturation, and too small a β lets **drift drown the reward**.

**β recommendation under C′: no lower than ~2, still start with the paper's 8.** This conflicts with step 2's recommendation;
**this doc takes precedence** (that item's precondition is config A with zero drift).

---

## 5. Limitations

- **`d` is the rollout/trainer framework difference, not training drift.** Real training drift can only be larger:
  C′'s saturation rate will be higher, and A's "whole group same value" will be more severe. **Does not change the verdict's direction; magnitude unknown.**
- **The β extrapolation in §4 relies on `Var(drift term)` not depending on β** (this holds),
  but it uses the variance of the framework difference; real drift has a different variance, so **the critical β will move**.
- **Only the starting point was measured** (`π_θ = π_old`, update term is 0). The length dominance in §2 is derived under two assumptions on `δ`,
  **not measured gradients**. Measuring it requires actually training.
- **`boppose` numbers are listed but do not take part in the verdict** (already excluded by §0.2).
- Rewards and lengths come from a **single sampling run** (passk2), same caveat as the previous docs.

---

## 6. Net effect on Route C

| item | disposition |
|---|---|
| **training arm** | **set to C′** (no normalization on either side). Gate passed |
| cost of length dominance | worst case 2.1–2.4× (longest 10%), only 1.04–1.32× within group. **Acceptable** |
| is the reward drowned | no, reward dominates 24× on non-degenerate groups |
| degenerate groups | C′ discriminates (0% whole group same value), A does not |
| **β** | **no lower than 2, start with 8**. Step 2's "β≈1" is void under C′ (§4) |
| still unresolved | machine feasibility for C (Step 4 does not fit on the current 4×A100-40GB) and statistical power |

**The gate's job is to rule out, not to approve.** It rules out the possibility that "all three configs fail",
and does **not** answer whether C is worth running — that depends on the machine and statistical power, see the Route C plan.
