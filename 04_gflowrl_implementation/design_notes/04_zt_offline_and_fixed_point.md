# P7 steps 2–3 results: offline decomposition of Z_t and the synthetic fixed-point test

> Follows on from steps 2 and 3 in `records/P7_DECISION.md` §6. **Zero GPU, zero machine time, no real model touched.**
> Written 2026-09-01.
> Scripts: `tools/p7/p7_zt_offline.py` (real data) · `tools/p7/p7_fixedpoint.py` (synthetic)
> Both scripts use only the stdlib + numpy and can be rerun directly.

---

## Summary

**The most important item is an observation that relocates P7's question itself:**

> **Under the `β = 8`, `G = 5` configuration, on the two accuracy benchmarks the flow gap `g_i` takes only three values
> — `0`, `−ε_low`, `+ε_high`. Not a single rollout carries an unclipped flow gap.**
>
>     robospatial   degenerate groups 995 (g always 0) + clip-saturated 755 = 1750  (100%)
>     blinkdepth    degenerate groups 470               + clip-saturated 150 =  620  (100%)
>     boppose       degenerate groups 160               + clip-saturated  58 =  218 / 300 (72.7%)
>
> In other words: **at the training starting point, all numerical information in `Z_t` and `β·r` is clipped away, only the sign remains.**
> Only `boppose` (continuous IoU reward) keeps 82/300 rollouts with graded gradients.

This changes the timing of criterion (i): **`Var(Z_t)` does not matter at the starting point** (there `π_old = π_ref`,
`Z_t` reduces to `mean(β·r)`, which is then clipped away);
**the moment it starts to matter is exactly the moment the T6 unit mismatch starts to show** — both are the same thing
(the `log π_ref − log π_old` term) at work.

The other three:

| | Conclusion |
|---|---|
| **Definition of `|y|`** | Using the raw response length inflates `|y|` by **1.14× / 1.41× / 2.40×**, and **differs systematically by benchmark** |
| **Remark B.4** | With length normalization, **a self-consistent zero-loss fixed point does not exist** (not "moved"). But **the scary reading "the effective inverse temperature becomes L·β" does not hold** — withdrawn |
| **`G = 16 → 5`** | `Var(Z_t)` **3.17×**, `sd` **1.78×**, numerically confirming `O(1/G)` |

---

## 1. Step 2: real data (`p6/passk`, G=5)

    python3 tools/p7/p7_zt_offline.py

**Splitting reuses the `TURN_SPLIT` and `leading_is_assistant` conventions of `tools/parse_dump.py`,
not a rewrite of our own** (P3 got burned: a homemade splitter silently dropped the first assistant turn).
The script has a built-in regression check: **split turn count vs verl `num_turns`, 2670/2670 all match.**

### 1.1 A · Within-group spread of `β·r`

| | Within-group sd (mean) | Within-group range (mean) | Reward-degenerate groups |
|---|--:|--:|--:|
| `robospatial` | 1.514 | 3.451 | 199/350 = 56.9% |
| `blinkdepth` | 0.826 | 1.935 | 94/124 = 75.8% |
| `boppose` | 0.393 | 0.927 | 32/60 = 53.3% |

In all three the **median is 0.000** — more than half the groups have exactly identical rewards.

### 1.2 B · Decomposition of `|y|`: assistant-generated vs `<tool_response>`

⚠ **The unit is characters, not tokens** (no tokenizer on this machine, `models/` is on the GPU machine).
Ratios and the magnitude of the spread are credible, absolute values are not; exact token counts come for free in the forward-pass step, and this should be rerun then.

| | assistant `|y|` (median) | Raw response (median) | **Ratio (median)** |
|---|--:|--:|--:|
| `robospatial` | 1258 | 1418 | **1.14×** |
| `blinkdepth` | 1561 | 2200 | **1.41×** |
| `boppose` | 839 | 2004 | **2.40×** |

> **This is the magnitude of the trap in item 2 of `P7_DECISION.md` §5.1.**
> If `|y_i|` mistakenly uses the raw response length, the normalization denominator is inflated 1.14–2.40×,
> **and the inflation factor differs systematically by benchmark** — `boppose`'s tool output (depth + point cloud + 8 corner points)
> is far more verbose than `robospatial`'s two coordinates.
> **The consequence is not added noise; it imposes on different tasks different effective temperatures set by tool verbosity.**

### 1.3 C · Within-group relative range of `|y|`, `(max−min)/mean`

| | assistant length | Raw response |
|---|--:|--:|
| `robospatial` | mean 0.41 · median 0.33 | 0.37 · 0.30 |
| `blinkdepth` | mean 0.34 · median 0.28 | 0.30 · 0.21 |
| `boppose` | mean 0.28 · median 0.05 | 0.15 · 0.03 |

The premise of the paper's Remark B.4 is "response lengths vary only modestly within rollout groups".
**Measured within-group relative range is 0.28–0.41 (up to 1.78).** This premise holds more weakly for us than on math problems,
but **not catastrophically weakly** — see the withdrawal in §2.3.

### 1.4 D · Clip saturation of the flow gap (see summary)

    Starting point π_old = π_ref  =>  g_i = β·(r̄ − r_i)
    |g| leaving the clip interval [−0.2, +0.28] only needs |r̄ − r_i| > 0.035
    binary reward + G=5: in a mixed group |r̄ − r_i| >= 1/5 = 0.2  ->  necessarily saturated

| | All rollouts | **Non-degenerate groups only** |
|---|--:|--:|
| `robospatial` | 755/1750 = 43.1% | **755/755 = 100.0%** |
| `blinkdepth` | 150/620 = 24.2% | **150/150 = 100.0%** |
| `boppose` | 58/300 = 19.3% | 58/140 = 41.4% |

**On both binary-reward benchmarks, every single non-degenerate rollout is saturated, none left.**

> **Both readings have to be reported.**
> **① This is not necessarily a bug.** Clip is a trust region by design; the paper's ablation shows removing it loses 3.9 points
> and raises the mean gradient by 6.3×; **clip is not optional**.
> **② But it determines what `β` does.** After saturation `β` no longer sets the size of the tilt, only the sign threshold;
> the distribution-matching property is then carried almost entirely by the `(1/|y_i|)·log(π_θ/π_old)` term,
> which is structurally very close to a **clipped, sign-driven policy gradient**.
> **This also gives a testable explanation for the paper's own sentence "β is insensitive within [1,10]": most of β is clipped away.**

**This yields an actionable recommendation that is the opposite of the handoff document:**
with binary reward + `G=5`, keeping the flow gap from always saturating needs `β ≲ 0.28/0.2 ≈ 1.4`.
~~**`β` should start around 1, not 8.**~~ Note that this reason is completely unrelated to the one withdrawn in §2.3 (`L·β`).

> **⚠ 2026-09-02 · The scope of this recommendation has been narrowed; it is void under the C′ configuration.**
> See `records/P7_ROUTE_C_GATE.md` §4. The derivation above assumes **configuration A with zero drift** (the training starting point).
> Once on C′ (no length normalization on either side, i.e. the training arm chosen after A′), saturation is mainly driven by the **drift term**
> and has little to do with β; and `Var(reward term) = β²·Var(r)` while `Var(drift term)` is independent of β, so
> **drift/reward ∝ 1/β²** — measured 0.042 at β=8, **at β=1 drift instead dominates reward 2.7×**.
> **Under C′ β is no lower than 2; still start at the paper's 8.**

---

## 2. Step 3: synthetic fixed-point test

    python3 tools/p7/p7_fixedpoint.py      # all PASS

**The purpose is to take "is our loss written correctly" out of P7's question** (`P7_DECISION.md` §0).
The paper's own claims are verified on an enumerable discrete `y` space; after this, anomalies on real data can no longer be blamed on the implementation.

### 2.1 T1 / T2 · Prop. B.1 and Remark B.3, reproduced item by item

    Z_t == log Z(x)              6.155376796568 == 6.155376796568
    residual for each y           max|Δ| = 1.8e-15
    loss                          7.9e-31
    counter-check: π_θ = π_ref    loss = 0.0560          (indeed > 0 away from the fixed point)
    flow gap at the fixed point   max|g| = 1.8e-15  <  ε_low = 0.2   -> clip naturally inactive

### 2.2 T3(b) · **With length normalization, a self-consistent zero-loss fixed point does not exist**

Zero loss requires `log π = log π_ref + L·(β·r − Z_t)`, and normalization pins `Z_t` to `logZ_L / L`;
while Eq. 4's self-consistency requires `Z_t = (1−L)·E[β·r] + L·Z_t`, which for `L ≠ 1` gives `Z_t = E[β·r]`.
Both holding at once requires `(1/L)·logZ_L == E_{p_L}[β·r]` — **a measure-zero coincidence.**
At `L = 1` (i.e. no length normalization) `(1−L) = 0`, self-consistency is automatically satisfied, which is exactly the Prop. B.1 case.

Numerical scan (candidate family `π_c ∝ π_ref·exp(c·β·r)`, `Z_t` recomputed self-consistently per Eq. 4 each time, **no clip**):

    L      argmin c    min loss
    1.0     1.0000     7.9e-31      <- exact zero
    2.0     1.2000     3.9e-01
    3.0     1.1400     7.4e-01
    5.0     1.0700     1.1e+00
   10.0     1.0300     1.4e+00

Six random worlds repeated (L=3): min loss 0.71–1.66, argmin c 1.00–1.15, **none is zero.**

> **Remark B.4 only solves the residual equation; it does not redo the third step of Prop. B.1 (estimator self-consistency).**
> So "mild length-dependent bias", in the self-consistent sense, is not "the fixed point moved",
> but "**the fixed point does not exist**": the loss has a positive floor that grows with `L`.

### 2.3 ⚠ One withdrawal: "effective inverse temperature = `L·β`" does not hold

`P7_DECISION.md` §5.1, based on Remark B.4, once wrote: the effective inverse temperature is `|y|·β`, and with `|y|` in the thousands
`exp(L·β·r)` pushes the mass onto the argmax, **so distribution matching degenerates into reward maximization**.

**The numerical test overturns it.** The minimum always falls at **`c ≈ 0.93–1.15`, not `c = L`**.
The `c = L` form is a zero only **when `Z_t` is treated as a free constant**; once `Z_t` is required to be self-consistent per Eq. 4,
it is no longer the minimum. **The tilt does not inflate with `L`.**

**So consequence 1 of §5.1 ("β cannot be transferred directly, transfer `L·β`") is void.**
`β` indeed should not be taken directly as 8, but the reason is the clip saturation in §1.4, not length.

> Self-check per the §3.3 prescription: this was written into the document **inferred from one sentence of the paper, without running any numbers**.
> Overturning it took only thirty lines of code. **The cost was that it existed in the document for half a day;
> had it gone into the implementation first, the cost would have been a β scan built on the error.**

### 2.4 T4 · `Var(Z_t) = O(1/G)`, and 16 → 5

Measured at `π_old = π_ref` (an honest analogue of the training starting point; at the fixed point every trajectory's TB target is identical,
`Var(Z_t) = 0` independent of `G`, so nothing can be measured there):

    G     Var(Z_t)   G*Var    sd
     2     2.3336    4.667   1.528
     4     1.1652    4.661   1.079
     5     0.9289    4.644   0.964   <- SpaceTools run_rl.sh
     8     0.5871    4.697   0.766
    16     0.2913    4.661   0.540   <- paper Table 9
    32     0.1449    4.638   0.381
    64     0.0734    4.700   0.271

`G*Var` stays between 4.64–4.70; **`O(1/G)` holds numerically**.
**16 → 5: variance 3.17×, sd 1.78×.**

### 2.5 T5 / T6 · The signal on degenerate groups, and a unit mismatch

Using logprobs at real magnitudes (`|y| ≈ 1258–1610`, per-token logprob ≈ −0.42):

| | `π_old = π_ref` (starting point) | `π_old` has drifted |
|---|---|---|
| `g_i` on a degenerate group | All `0`, **within-group range 0** | `[11.66, 11.63, 11.68, 11.61, 11.70]`, range 0.09 |
| `Z_t` (Eq. 4, unnormalized) | 4.800 | **16.460** |
| `(1/|y|)·log(π_old/π_ref)` | `[0, 0]` | `[−0.050, +0.040]` |
| `β·r` | `[0, 8]` | `[0, 8]` |

**Two points:**

1. **The statement in `P7_DECISION.md` §3.2 has to be pulled back once more.** On degenerate groups the Eq. 8 residual is **always 0 at the starting point**
   (not "nonzero"), and only becomes nonzero after drift — and that bit of discrimination **does not come from the reward**,
   it comes from the inconsistent normalization between Eq. 4 and Eq. 5. **Whether it is a useful signal is exactly what criterion (iii) has to answer.**
2. **The unit mismatch is real, and only shows up mid-training.** `Z_t` (Eq. 4) grows with **the sum over `|y`**,
   while the term it is meant to center is a **per-token mean**. At the starting point `π_old = π_ref` makes that term 0,
   `Z_t` reduces to `mean(β·r)`, the three terms are on the same order, and **the problem is completely invisible**;
   once there is drift, `|y|` in the thousands means the two differ by about three orders of magnitude.

---

## 3. Net effect on `P7_DECISION.md`

| Item | Disposition |
|---|---|
| §5.1 consequence 1 "β cannot be transferred, transfer `L·β`" | **Withdrawn** (§2.3). The conclusion "β should not be taken as 8" is kept, with the reason changed to clip saturation |
| §5.1 consequence 2 "`|y|` must use the sum of `response_mask`" | **Kept and quantified**: 1.14× / 1.41× / **2.40×**, and differs systematically by benchmark |
| §3.2 "on degenerate groups GFlowRL produces gradient while GRPO does not" | **Pulled back once more**: always 0 at the starting point; the discrimination after drift comes from inconsistent normalization |
| Criterion (i) `Var(Z_t)` | **Timing rewritten**: does not matter at the starting point (clipped away), only matters once `π_old` starts drifting |
| Criterion (ii) length normalization | **Upgraded**: not "the fixed point moved" but "the fixed point does not exist"; the floor grows with `L` |
| Criterion (iii) degenerate-group residual | **Made concrete**: the control to compare against is now clear — 0 at the starting point / after drift it comes from the unit mismatch |
| New | **Clip saturation rate** should go into the formal report: it determines how `β` is chosen, and it is measurable purely offline |

---

## 4. Limitations

- **All real-data numbers come from a single sampling run (n=1), never repeated.** Same label as `P7_DECISION.md` §3.
  The second independent set of 5 samples now has a third use: also measuring a spread for the clip saturation rate.
- **The length unit is characters, not tokens.** Ratios are credible, absolute values are not; exact values come for free at the forward pass.
- **`p6/passk` is an eval benchmark, not the training distribution `spacetools-rlfulltools`.**
  Degeneracy rate, saturation rate and length distribution may all differ on the training data, and cannot be extrapolated from the existing data.
- **§2.2 and §2.3 are this document's numerical + algebraic conclusions, not the paper's words.**
  The paper only says length normalization is an engineering trade-off with mild bias;
  "no self-consistent zero exists" is our addition, and **should be independently re-checked before being cited externally**.
- The synthetic world is a discrete space with `|Y| = 64`, `log π = O(1)`; on a real LLM `log π(y) = O(|y|)`.
  T1–T3 only need a small normalizable world; T5/T6 switch to real magnitudes — do not read the two together.
