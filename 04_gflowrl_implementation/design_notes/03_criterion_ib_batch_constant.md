# P7 criterion i-b: comparing three within-batch constants (synthetic half)

> `P7_DECISION.md` criterion i-b. **Zero GPU.** Written 2026-09-01.
> Script `tools/p7/p7_estimators.py` (numpy + stdlib, can be rerun directly).
> Rewards use the **real G=5 groups** from `p6/passk`; the `log π_ref − log π_old` term **is synthetic**
> — it is unavailable before the forward pass, so it is generated as "per-token drift × `|y|`", with a scannable magnitude.
> **After the forward pass this script must be rerun with the true values.**

---

## Summary

**Headline: the inconsistent normalization between `Eq. 4` and `Eq. 5/6` erases the within-group reward differences entirely
as soon as `π_old` leaves `π_ref` — and the threshold is tiny, only about `5e-4` nat per token.**

    g_i = C − d_i/|y_i| − β·r_i ,   C = mean_j( β·r_j + d_j )
          ^^^^^^^^ normalized          ^^^^^^^ unnormalized (Eq. 4)

The drift term is at **sum scale** in `C` and at **per-token scale** in the subtracted term, so within a group it is approximately a
**common shift**. Once the common shift exceeds the clip half-width, the whole group is pushed to the same side, `g̃_i` takes the same value across the group,
and **the reward's contribution is completely erased**.

| Per-token drift | Drift contribution to `C` (median) | **Whole group `g̃` same value** |
|---|--:|--:|
| 0 | 0.0000 | 56.9% (i.e. step 2's reward-degenerate groups, numbers match) |
| 3e-4 | 0.118 | 10.6% |
| 1e-3 | 0.369 | **36.0%** |
| 3e-3 | 1.228 | **56.0%** |
| 1e-2 | 3.776 | **78.3%** |

The threshold can be computed directly: sd of the drift contribution to `C` ≈ per-token drift × `|y|` / `√G`;
setting it equal to `ε_high = 0.28` gives **per-token drift ≈ 5.0e-4 nat** (`robospatial`, `|y|≈1258`)
/ **7.5e-4** (`boppose`, `|y|≈839`).

**And this one is fixable: also normalize the logprob term of `Eq. 4` by `|y|`, and the phenomenon disappears completely** (§6).
So the source of the mechanism is **the inconsistent normalization itself, not GFlowRL's idea of within-batch estimation**.

---

## 1. ⚠ First, correct a statement: "three estimation methods" is too loose

TB residual `Δ_i = C + (1/|y_i|)·log(π/π_ref) − β·r_i`; the relationship between the three is actually:

| | How the constant is set | Which π | Gradient flows or not |
|---|---|---|---|
| **GFlowRL** (Eq. 4) | Arithmetic mean | `π_old` | **`sg[]`, no** |
| **VarGrad** | Minimizes `Σ Δ_i²` = arithmetic mean | `π_θ` | **Yes** |
| **DevGrad** (f-TB) | Minimizes `(1/B) Σ L_f(Δ_i + C)` | — | — |

**With squared loss + `π_θ = π_old`, the three are numerically identical.** There are only two differences:
**(a)** whether gradient flows through the constant (only shows up once `π_θ` drifts away within a minibatch, §5);
**(b)** the choice of `f` — squared gives the mean, absolute gives the median, Huber gives a robust constant (§4).

What I said last round, "three estimates of the same constant", was not accurate enough; corrected accordingly.

---

## 2. 2D scan of β × drift (`robospatial`)

Format: **saturation rate / sign disagreement rate** (mean vs huber, after clip)

| Per-token drift | β=0.5 | β=1 | β=2 | β=8 |
|---|---|---|---|---|
| **0** | 12.3% / 0.0% | 23.5% / 0.0% | 43.1% / 0.0% | 43.1% / 0.0% |
| 3e-4 | 25.7% / 10.9% | 35.8% / 10.2% | 47.5% / 11.0% | 50.0% / 11.3% |
| 1e-3 | 69.0% / 16.2% | 71.3% / 14.8% | 74.4% / 15.5% | 80.8% / 16.4% |
| 3e-3 | 86.2% / 20.1% | 87.5% / 20.1% | 88.7% / 19.3% | 89.8% / 18.0% |
| 1e-2 | 97.0% / 18.1% | 96.6% / 18.1% | 95.9% / 17.4% | 97.0% / 17.4% |

**Both are the opposite of what was guessed before writing the script:**

1. **Once drift is nonzero, the saturation rate rushes toward 100% almost independently of β.**
   β only has the say in the `drift = 0` row — that is, **only at the first training step**.
   Step 2's recommendation "β should start around 1" therefore needs a qualifier: **it only holds for the first step**;
   after that, β's effect on saturation is drowned out by the drift term.
2. **Saturation ≠ the choice of constant does not matter.** Clip cuts off the **magnitude**; what survives is the **sign**,
   and the sign still depends on `C`: mean and huber give **opposite update directions** on **11%–20%** of rollouts.
   (The verdict I first wrote into the script last round was "after saturation the constant does not matter"; **the data disagrees, changed.**)

---

## 3. The whole group's `g̃` clipped to the same value — this section is the headline

See the table in the summary. Three points worth recording separately:

- **The 56.9% in the `drift = 0` row is exactly step 2's reward-degenerate group fraction**, matching to one decimal place.
  This is an unplanned internal consistency check: two scripts, two paths, the same number.
- **The curve is non-monotonic**: a small drift (3e-4) actually breaks the ties of the degenerate groups, so the same-value rate first drops to 10.6%,
  then climbs past the original level as drift increases. So "low same-value rate" does not equal "good signal".
- **The threshold is tiny.** The clip half-width is only 0.2–0.28, while `C` holds **the sum of logprobs over a thousand-plus tokens**.
  The units differ by three orders of magnitude, so naturally the tolerance is only on the order of `5e-4` nat/token —
  ordinary RL fine-tuning crosses it within a few steps.

---

## 4. Sampling variance of the estimator itself: this is where i-b lands

Fix the true value, draw `G` samples repeatedly 20000 times:

| G | Noise distribution | Var(mean) | Var(median) | Var(huber) | Relative to mean |
|--:|---|--:|--:|--:|---|
| 5 | Gaussian | 0.1990 | 0.2867 | 0.2696 | median **1.44×** · huber 1.35× |
| 5 | Heavy-tailed t3 | 0.5836 | 0.4112 | 0.3962 | median **0.70×** · huber 0.68× |
| 5 | Binary + noise (bimodal) | 3.4161 | 11.9909 | 11.9013 | median **3.51×** · huber 3.48× |
| 16 | Gaussian | 0.0634 | 0.0913 | 0.0861 | median 1.44× · huber 1.36× |
| 16 | Heavy-tailed t3 | 0.1829 | 0.1121 | 0.1075 | median **0.61×** · huber 0.59× |
| 16 | Binary + noise (bimodal) | 1.0688 | 7.9345 | 7.9020 | median **7.42×** · huber 7.39× |

**Three distributions give three different orderings**, in opposite directions:

- **Gaussian**: mean is optimal; robust constants pay a 1.35–1.44× variance cost.
- **Heavy-tailed t3**: mean is worst; median/huber need only 0.59–0.70× of it.
- **Bimodal (binary reward + noise)**: mean is **clearly optimal**, median is 3.5–7.4× worse — it chases one of the peaks.

> **So criterion i-b cannot be closed on synthetic data.** Which constant to pick depends entirely on
> the **real distribution shape** of `log π_ref − log π_old`, and only the forward pass can give that.
> **This is exactly where i-b lands: after the forward pass gets the real drift, look at its distribution shape first, then judge the estimator.**
> One prior: our reward is close to binary on the two accuracy benchmarks — if the drift term is also close to bimodal,
> **the arithmetic mean is actually the best of the three**, and GFlowRL's choice holds up here.

---

## 5. Is `sg[]` worth it (GFlowRL freezes the constant, VarGrad does not)

How much the constant moves when `π_θ` keeps drifting away within a minibatch:

| Per-token drift within minibatch | Constant shift (mean) | Relative to `β·r` range (0–8) |
|--:|--:|--:|
| 0.001 | 0.494 | 0.062 |
| 0.003 | 1.505 | 0.188 |
| 0.010 | 5.244 | **0.656** |
| 0.030 | 13.555 | **1.694** |

**The shift is proportional to `|y|`** (same unit mismatch as §T6), so the constant frozen by `sg[]`
can drift **more than the entire range of `β·r`** within a single minibatch.
This is both the reason for `sg[]` (without freezing, the gradient would have to pass through a quantity this large)
and its cost (the frozen value can go stale quickly).

---

## 6. Candidate fix: length-normalize `Eq. 4` too

    C_norm = mean_i[ β·r_i + (log π_ref − log π_old)_i / |y_i| ]

| Per-token drift | Eq. 4 as is: saturated / whole group same value | **After normalization: saturated / whole group same value** |
|--:|---|---|
| 3e-4 | 54.3% / 11.1% | 43.1% / **0.0%** |
| 1e-3 | 79.1% / 36.0% | 43.1% / **0.0%** |
| 3e-3 | 89.4% / 53.7% | 43.1% / **0.0%** |
| 1e-2 | 96.3% / 78.3% | 43.1% / **0.0%** |
| 3e-2 | 99.4% / 92.6% | 43.1% / **0.0%** |

**After normalization, both metrics return to the `drift = 0` level, and no longer grow with drift.**

> So the "reward contribution erased" mechanism of §3 **comes from the inconsistent normalization between `Eq. 4` and `Eq. 5/6` itself,
> not from GFlowRL's idea of within-batch estimation.**
>
> ⚠ **The cost was measured on 2026-09-02; this candidate is not free.** See §6.1 below.
> (The original text said "needs to be measured separately with the `p7_fixedpoint.py` framework, not done this round" — it has now been done.)

### 6.1 Fixed-point cost: the fix restores the fixed point, but that fixed point is a point mass

`tools/p7/p7_fix_cost.py`. Reuses the small world of `p7_fixedpoint.py` (`|Y|=64`, can be normalized exactly).

**First write out the algebra.** Zero loss (`Eq. 5` with length normalization) requires, for every `y`:

    Z + (1/L_y)(log π_θ,y − log π_ref,y) − β·r_y = 0
    =>  log π_θ,y = log π_ref,y + L_y·(β·r_y − Z) − c(Z),   c(Z) = logsumexp(...)

Substituting back into the residual gives `residual_i = Z_t − Z − c(Z)/L_i`. When `L` varies, for it to be zero for all `i`,
**`c(Z)=0`** and **`Z_t=Z`** must hold simultaneously. The former is a monotone equation in one variable and **generically has a unique root** `Z*`;
the latter is the "estimator self-consistency" step:

    Eq.4 as is (unnormalized)   Z_t = E[β·r] − E[L·β·r] + Z·E[L]              -> in general ≠ Z*
    Eq.4 normalized (the fix)   Z_t = E[β·r + (log π_ref − log π_θ)/L] = Z    ✓ **identically**

**Numerical verification** (at the point constructed from `c(Z*)=0`):

| World | `Z*` | Fix `\|Z_t−Z*\|` | Paper as is `\|Z_t−Z*\|` | Fix loss | Paper as is loss |
|---|--:|--:|--:|--:|--:|
| `\|y\|` 331–3142 (median 1572) | 7.765 | **8.9e-16** | 6.355 | **6.9e-34** | 4.04e+01 |
| `\|y\|` 254–445 (median 366, measured magnitude) | 7.730 | **8.9e-16** | 4.935 | **0.0** | 2.44e+01 |

> Consistency self-check: at that point the residual is the constant `Z_t − Z*`, so the loss should equal `(Z_t−Z*)²`.
> Measured `4.935² = 24.35` matches `2.435e+01`. **Mechanically consistent.**

**So the existence problem is fixed. The cost is in the location of the fixed point:**

    Homogeneous length scan (same L, everything analytic)
       L        Z*      loss(fix)       entropy H  eff. support  max p
       1.0   5.34947    7.3e-31        3.0448    21.006   0.1388   <- bit-for-bit identical to Prop.B.1's beta-tilt
       2.0   6.24797    3.6e-30        2.2283     9.284   0.2285
       5.0   7.11816    3.2e-30        1.2468     3.479   0.4815
      10.0   7.50997    8.6e-36        0.6471     1.910   0.7500
      50.0   7.88858    7.9e-31        0.0038     1.004   0.9996
     334.0   7.97517    0.0            0.0000     1.000   1.0000   <- robospatial's measured median |y|
      π_ref                            3.8812    48.482   0.0570

**At `L = 334` (robospatial's measured median token count), the fixed point is a point mass: effective support 1.000, `max p = 1.000000`.**
**Distribution matching degenerates into reward maximization — exactly what GFlowRL is meant to avoid.**

The `L = 1` row is **bit-for-bit identical** to Prop. B.1's `π_ref·exp(β·r)` (H 3.0448 / support 21.006 / max p 0.138755);
this is a mechanical self-check: the fix must reduce to Prop. B.1 at `L=1`, and it does.

### 6.2 No contradiction with the item withdrawn in `P7_STEP23_RESULTS.md` §2.3

§2.3 withdrew "effective inverse temperature = `L·β`", on the grounds that **with the paper as is** the minimum falls at `c ≈ 0.93–1.15`, not `c = L`.
That **still holds**, because with the paper as is **there is no zero at all** — the minimum is a compromise between two incompatible conditions,
"self-consistency" and "zero residual", and naturally falls between them.

**The fix makes the zero actually exist, so Remark B.4's `L·β` reading comes back under the fix.**
Both statements hold, each for a different objective.

### 6.3 Net conclusion: this fix is not free, and both sides come from the same place

| Configuration | Whole group `g̃` same value | Self-consistent zero | Tilt of the fixed point |
|---|---|---|---|
| **A. Paper as is** (Eq.4 unnormalized / Eq.5-6 normalized) | 26.9%–46.0% (measured) | **Does not exist** | Minimum at `c ≈ 1` |
| **B. §6 fix** (both sides normalized) | **0.0%** (measured) | **Exists** | **`L·β` → argmax** |
| **C. Neither side normalized** (back to Prop. B.1) | — | Exists | `β`, entropy on the same order as `π_ref` |

**A's disease (reward contribution erased) and B's disease (fixed point collapses to argmax) are two sides of the same length normalization.**
The only one healthy on both metrics is **C** — and C's cost is exactly the problem the paper introduced length normalization to solve:
long sequences get more weight in the loss. **The paper chose A, and recorded this trade-off in Remark B.4, calling it
"mild length-dependent bias"; as measured, it is neither mild, nor just a bias.**

---

## 7. Net effect on P7

| Item | Disposition |
|---|---|
| The "three estimation methods" statement | **Corrected** to "the same constant, two differences: `sg` or not / choice of `f`" (§1) |
| Step 2's "start β around 1" | **Qualifier added**: only holds for the first training step; after that saturation is dominated by drift and almost independent of β (§2) |
| "After saturation the constant does not matter" | **Overturned**: the sign still depends on `C`; 11%–20% of rollouts get the opposite direction (§2) |
| Criterion i-b | **Cannot be closed on synthetic data**; it lands on looking at the drift's **distribution shape** after the forward pass (§4) |
| Criterion (i) `Var(Z_t)` | Gets a prior: **if the drift is close to bimodal, the arithmetic mean is the best of the three** — GFlowRL's choice holds up |
| **New candidate** | **Length-normalize `Eq. 4`**; on synthetic data it completely eliminates the §3 phenomenon; fixed-point cost needs to be verified separately (§6) |

**This one is constructive for A′**: if the forward pass confirms the real drift exceeds `5e-4`/token,
then A′'s answer will be "`Z_t` is not usable here, **but the cause is not variance, it is inconsistent normalization**" —
and that cause **has a concrete fix**. **The negative result is therefore not a dead end.**

---

## 8. Limitations

- **The drift term is synthetic.** Every number in this document involving `drift` depends on that assumption (independent Gaussian × `|y|`).
  The real `log π_ref − log π_old` may be correlated across rollouts, and may be asymmetric.
  **After the forward pass this must be rerun with the true values.**
- **Length is in characters, not tokens**, so the absolute value of the `5e-4 nat/token` threshold carries the same uncertainty;
  the order of magnitude (1e-4 ~ 1e-3) is credible.
- **Rewards come from a single sampling run (n=1)**, labeled the same as in the previous documents.
- The candidate fix in §6 **only measured the symptom disappearing, not the cost** (fixed point, convergence).
- This document is entirely at the **population/analytic level**; no real gradient steps were run.
