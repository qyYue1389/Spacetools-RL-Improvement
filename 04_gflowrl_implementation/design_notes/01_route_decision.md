# P7 route decision: the one sentence

> Follows on from `records/P7_HANDOFF.md`. **This document does exactly one thing: fix the question P7 answers, and write down the criteria.**
> Written 2026-09-01. When the decision was made, **no** data had been downloaded, no machine rented, no training script written,
> and `run_rl.sh` had not been opened — this is the output of step 1 in §5 of the handoff document.
>
> Based on: `P7_HANDOFF.md` (full text) · `P6_REPORT.md` (full text) · `P6_GPU_HANDOFF.md` §0 void list ·
> project doc "Notes on the two papers" · the new measurements in §3 of this document.

---

## 0. Decision

**Pick A (algorithm-side contribution), but rewrite its one sentence.**

> **On SpaceTools' multi-turn tool trajectories, is GFlowRL's within-batch Monte Carlo estimator `Z_t`
> still a usable log-partition estimate?**
> **If not — are the two other estimates of the same within-batch constant (VarGrad's variance minimizer, DevGrad's optimal constant)?**
>
> Split into three questions:
> 1. What is the magnitude of the **within-group variance** of `Z_t` relative to `β·r`;
> 2. Is the bias that **length normalization** introduces on heterogeneous 2–16-turn trajectories large enough to move the fixed point;
> 3. On the groups with **degenerate rewards** (§3), is the Eq. 8 residual a meaningful gradient, or noise.

> **The second half of the sentence was added on 2026-09-01 after searching for existing implementations** (see `records/P7_PRIOR_ART.md`).
> Why add it: when the original question answers "not usable", it **has nowhere to go**; with the two other estimators added,
> **the answer is actionable in both directions**. Cost unchanged — the three are different estimates of the same constant,
> the framework in `tools/p7/p7_fixedpoint.py` runs directly on the real groups in `p6/passk`, **still zero GPU**.
> **Scope is pinned to offline estimator diagnostics (variance, saturation rate, units), no training** —
> three estimators slide very easily into a "which one scores higher" score-chasing experiment, which is exactly what the ⚠ paragraph of the handoff document warned against.

**Why this sentence: the answer is attributable in both directions.** It asks about properties of the data, the reward and the reference policy,
not whether our Eq. 8 is written correctly. The latter is guarded **separately** by the synthetic fixed-point test:
construct `π_θ = π_ref·exp(β·r)/Z` and check loss ≈ 0; check that clip fails in the neighborhood of the fixed point (paper Remark B.3).
**These two responsibilities must be kept separate**, otherwise we are back to a non-attributable negative result (§1).

### 0.1 Pre-registered criteria (written down first, to avoid picking after the fact)

| # | Criterion | What counts as pass / fail |
|---|---|---|
| i | Report the **distribution** of `Var(Z_t)/(β·r)²`, not a single number | Per the §3.3 discipline, **spread on both sides is required** — a second independent set of 5 samples is a **precondition for it, not optional** |
| i-b | **Report the three ways of taking the same within-batch constant side by side** | ~~three estimates~~ **wording corrected**: with squared loss + `π_θ=π_old` the three are numerically identical, and differ only in **`sg` or not** (GFlowRL vs VarGrad) and **the choice of `f`** (squared/absolute/Huber). **The synthetic half is done, see `records/P7_CRITERION_IB.md`; the real half has to wait for the forward pass** — which constant to pick depends entirely on the **shape of the distribution** of the drift (Gaussian → mean optimal; heavy-tailed → median 0.6× better; bimodal → mean 3.5–7.4× better) |
| ii | Run length normalization on/off once each | Compare **fixed-point location**, not scores |
| iii | Residual on degenerate groups | Must be compared against "the zero-information control with `π_old` replaced by `π_ref`", **not just checked for being nonzero** |

> **2026-09-01 · The synthetic half of criterion i-b added one more candidate fix** (`P7_CRITERION_IB.md` §6):
> **also normalize the logprob term of `Eq. 4` by `|y|`**; on synthetic data this pushes "the whole group's `g̃` clipped to the same value"
> from 36%–93% back to **0.0%**, and it does not grow with drift. **Cost (fixed point) not measured.**
> This turns A′'s negative answer into a constructive one: "`Z_t` is not usable, but the cause is inconsistent normalization, and that has a fix".

> **2026-09-01 · All three criteria were rewritten once by steps 2/3, see `records/P7_STEP23_RESULTS.md` §3 for details.**
> The most important one: **at the starting point the flow gap takes only three values `{−ε_low, 0, +ε_high}`**
> (on both binary benchmarks the non-degenerate rollouts are **100% saturated**),
> so **`Var(Z_t)` does not matter at the starting point; it starts to matter at the same time as the T6 unit mismatch** —
> both are driven by the `log π_ref − log π_old` term. What criterion (i) has to measure is **after drift**.

### 0.2 Explicitly excluded from P7

**`bopgrasp` / `boppose`.** P6 §4.2: `r` (NCE) is insensitive to gripper orientation —
a fallback answer with a median orientation error of **63.9°** gets NCE **1.07**, better than the **1.38** of a grasp actually computed by the tool.

> Where the reward is misspecified, any algorithm that targets `r` has nothing to say.
> And distribution matching is one level worse than reward maximization: GRPO only keeps the single bad mode with the highest `r`,
> **distribution matching keeps that bad mode alive too, in proportion to `exp(β·r)`.**
> **This is a scope statement, not a risk note** — P7 claims nothing about these two benchmarks.

**Swapping tools (2026-09-02 · user decision).** No second depth model installed, no stronger pointing model installed.

The two experiments in `P6_NOTES.md` that were "blocked by installation, suggest scheduling separately" are now **decided against**.
The reason is scope, not cost: **P7 tests swapping the objective; swapping tools is a separate, orthogonal path**,
and with both mixed together no result is attributable. All measured headroom numbers are kept
(tool error : reasoning error ≈ 13:1 · RoboRefer accounts for 53.4% · 17 near-ties on depth ·
pointing oracle +10.83 pp), there are just no experiments scheduled to claim them any more.

> **Also keep P6's conclusion in mind, otherwise P7's success or failure will be misread**:
> **swapping the objective cannot touch the 273/322 (84.7%) tool errors and tool-set gaps.**
> Whatever P7's result, it should not be expected to move this headroom — it is outside P7's scope.

### 0.3 ⚠ "Improve accuracy" and A′ are not the same goal — this has to be written out, not left vague

The P7 goal the user stated on 2026-09-02 is "**try applying GFlowRL and see whether it improves model accuracy**".
**That statement corresponds to Route C (full same-starting-point comparison), not to the A′ chosen in this document.** The difference is not wording:

| | What it asks | Cost | How this document handles it |
|---|---|---|---|
| **A′** (chosen) | Is the within-batch estimator `Z_t` still usable on multi-turn tool trajectories | Small, mostly offline | §0 |
| **C** (excluded) | How many points GFlowRL gains over GRPO | Rerun SFT + both arms × multiple seeds, 3–4 days of machine time | §4.1 |

C is excluded on **numbers**, not preference (§4.1): the upper bound on the movable share is **33–44/2001 = 1.65–2.20 pp**,
and that is an "eat all of it, miss none" upper bound; while the spread of a single run aggregated to 2001 samples is already **0.5–0.8 pp**.
**The expected effect is the same order of magnitude as the noise**; resolving it would need multiple seeds for both arms, on top of us having no SFT checkpoint
and the 4-GPU machine being 8× short on throughput (experiment D).

> **⚠ 2026-09-02 update: the gate for C has been run, and passed.** See `records/P7_ROUTE_C_GATE.md`.
> After A′ judged `Z_t` unusable as in the paper, the first question for Route C is no longer "should we spend 3–4 days",
> but "**which loss to train**" — each of the three configurations has its own disease. The gate compared them once on real data:
> **the training arm is set to C′ (no length normalization on either side, `Eq.4` or `Eq.5/6`).**
> · The cost of length dominance is mild: the longest 10% account for 20.7–24.2% of the loss (even share would be 10%), **only 1.04–1.32× within a group** —
>   because our median `|y|` is only 305–423 tokens, not the thousands the paper worries about.
> · The reward is not drowned by drift: on non-degenerate groups **reward dominates drift 24×** (drift/reward = 0.041–0.042).
> · On degenerate groups C′ discriminates (whole group same value **0.0%**), while A is 30–49%.
> · **The price is that C′ has the highest clip saturation rate (77.9%)** — trading "more clipped to ±ε" for "no group has its reward erased".
> · **Knock-on: `P7_STEP23_RESULTS.md` §1.4's "start β at 1" is void under C′** —
>   drift/reward ∝ 1/β², and at β=1 drift dominates reward 2.7×. **Under C′ β is no lower than 2; still start at 8.**
>
> **The gate's job is to exclude, not to approve**: it ruled out "all three configurations fail",
> it did not answer whether C is worth running — that depends on machine feasibility and statistical power.
>
> **The full Route C plan is written up: `records/P7_ROUTE_C_PLAN.md`.** Two decisions that must be made by the user:
> **① Rent 2 nodes × 8 GPUs or not** — the current 4×A100-40GB **cannot run Step 4** (tools take 3 GPUs,
> 1 is left for training, optimizer state is 60.6 GB; `model_dtype=bf16` cannot rescue it, the fp32 master is exactly what the optimizer needs).
> **② Accept or not "the primary metric reports only `robospatial`"** — with the total over nine benchmarks as the primary metric,
> **it cannot be detected even at maximal effect** (needs a net difference ≥48, ceiling 44); while P6 has already located the movable share to
> **about 42 of the 44 in `robospatial`**; narrowed to n=350 it needs ≥20 with a ceiling of 42, **and power is sufficient**.

**Disposition: scope unchanged, still going with A′.** This document does not switch to C on its own — that is a scheduling decision for 3–4 days of machine time,
and it must be made explicitly by the user after seeing the numbers above.

**But A′'s output is related to "can it improve accuracy", not unrelated**: if A′ answers "`Z_t` is not usable
here", then C will not win however it is run, **so doing A′ first is exactly the cheapest path to that question**;
if A′ answers "usable, and the normalization inconsistency has a fix" (`P7_CRITERION_IB.md` §6),
then there is a basis for discussing whether to spend 3–4 days measuring scores.

---

## 1. Why not use the original A sentence from the handoff document

Original sentence: "Can GFlowRL train stably on multi-turn tool trajectories?" **It has two flaws at once.**

**① A "yes" is almost free.** The meaning of "stable" comes from GFlowRL's comparison against FlowRL
(in 421 FlowRL steps, 55 had gradient norm ≥ 1e6), and the root cause the paper itself diagnoses is **the learning-schedule mismatch between a randomly initialized
`Z_φ` and the pretrained policy**. What GFlowRL does is delete `Z_φ`.
**Without `Z_φ`, that failure mode does not exist** — normal gradient norms show nothing.

**② A "no" is not attributable.** `microsoft/gflowrl` is a 404; the loss has to be written ourselves from Eq. 4–8.
So is "training unstable" because the algorithm does not generalize to multi-turn tool trajectories, or because our Eq. 8 is wrong?
To separate them, we would first have to reproduce the gradient statistics the paper reports in its own setting (7B math) — that is the budget of another project.

> The handoff document says "A failing is also a result". **That only holds when the result is attributable.**
> Without a reference implementation, "stable / unstable" is exactly the least attributable kind of criterion.

### 1.1 ⚠ 2026-09-01 re-check: FlowRL has code now — should we fall back to the original A?

**No.** The main reason for rewriting A into A′ was **attribution** (② above): without a reference implementation,
"training unstable" cannot be told apart between the algorithm not generalizing and us writing it wrong. `Xuekai-Zhu/FlowRL` is built on verl,
Apache-2.0, and seems to fill exactly that gap.

**But what it fills is precisely the part outside A′.** The two share the TB residual, `response_mask`, length normalization and
the verl pipeline; the differences are exactly **`Z_φ` vs `Z_t`, Eq. 7's asymmetric clip, and the IS part** —
**that is, all of GFlowRL's contribution, with not a single line of reference.**

> So the borrowed code makes "the pipeline is wrong" less likely, and does **not** make "does the estimator work on multi-turn tool trajectories"
> attributable. **A′ asks about exactly that residual. This search does not weaken A′; it clears away the noise around it.**

The same search also externally confirmed A′'s gap: **no implementation of a GFlowNet-style objective on multi-turn tool trajectories was found**,
consistent with the paper's own Appendix A sentence "unclear whether ... extends to broader agentic or multimodal
RL settings". (A search is not a proof of non-existence, see the note at the top of `P7_PRIOR_ART.md`.)

---

## 2. Half of the "orchestration collapse" motivation does not hold

The ⚠ paragraph of `P7_HANDOFF.md` and `P6_REPORT.md` §7.1 both treat
`all three RefSpatial 277/277 use the same chain` as measured evidence of collapse. **Most of this evidence is spurious.**

RefSpatial questions are "point to where that thing is"; the correct chain is just one `detect_one` call;
P6 §2.2 itself says the policy's entire contribution on these 277 is the wording of `obj_name`.
**There is no second equally valid chain to preserve — this is not collapse, the task just has one path.**
The same goes for `boppose`'s 60/60 pass-through and the 95.6% / 99.8% rule compliance on depth questions.
P6 §7's sentence "the optimal policy is a deterministic rule to begin with" **rules out the use case for distribution matching along with it**.

**There is only one place where collapse really costs something, and P6 has already located it cleanly:**

- **2a should-have-called-but-didn't, 11 cases** = 3 hand-classified (`#246` / `#272` / `#301`) + 8 `front/behind`
- `front/behind`: 29 questions ask about depth order, **`depth_estimator` is in that benchmark's tool list,
  and was called 0 times in 29/29** (all 350 `robospatial` samples, 0 calls)

**This is the only place where "a known valid alternative chain exists, the policy never takes it, and the cost is quantifiable (+8 samples)".**
This kind of suppressed mode is exactly what distribution matching claims to preserve.

### 2.1 Correction to the movable share

| Definition | Count | Share of 322 | Notes |
|---|--:|--:|---|
| Handoff lower bound (reasoning errors only) | 19 | 5.9% | |
| Handoff upper bound (+3b) | 33 | 10.2% | Attribution of 3b is disputed; P6 §8 reports both readings |
| **This document's correction (+2a)** | **44** | **13.7%** | **The 11 in 2a are the only one of the three blocks whose mechanism is "distribution-matching shaped"** |

**The direction is unchanged, but the three blocks can now be bet on separately.** 2a is an orchestration problem; the 19 reasoning errors and the 14 in 3b are not.

---

## 3. New measurement: `p6/passk` itself is a set of real G=5 rollout groups

Those three files were run for pass@k (`n=5, T=1.0`), and the conclusion was null.
**But they happen to be the only kind of data P7 really lacks** — five rollouts per prompt, with reward, turn count and full trajectory.

    p6/passk/robospatial/0.jsonl   1750 lines = 350 prompts × 5
    p6/passk/blinkdepth/0.jsonl     620 lines = 124 × 5
    p6/passk/boppose/0.jsonl        300 lines =  60 × 5

### 3.1 Length heterogeneity (`num_turns` is the **verl definition** = user + assistant + 1)

| | Within-group mean | Overall range | **Within-group range** (mean / max) | Output-length relative range (median) |
|---|--:|--:|--:|--:|
| `robospatial` | 4.00 | 2–10 | 0.09 / 6 | 0.29 |
| `blinkdepth` | 6.70 | **2–16** | **2.24 / 10** | 0.18 |
| `boppose` | 10.00 | 6–16 | 0.90 / 6 | 0.03 |

**Read the two kinds of heterogeneity separately:** for `robospatial` / `boppose` the heterogeneity is mainly **between prompts**
(turn count is nearly constant within a group); `blinkdepth` differs by 2.24 turns **within a group**, max difference 10.
The paper's Remark B.4 "negligible when rollouts have similar lengths" **is directly violated on `blinkdepth`**.

### 3.2 Within-group reward degeneracy — the one most worth betting on

    Fraction of prompts whose reward is identical across all G=5
      robospatial   199/350 = 56.9%       within-group reward range: mean 0.431  median 0.000
      blinkdepth     94/124 = 75.8%       mean 0.242  median 0.000
      boppose        32/60  = 53.3%       mean 0.116  median 0.000

**For half to three quarters of the prompts, all five rollouts get the same reward.**

- For **GRPO**: after advantage normalization it is 0, **the whole prompt produces no gradient**.
- For **GFlowRL**: Eq. 8 is a per-trajectory squared residual, and `g_i` also contains the length-normalized
  `log(π_old/π_ref)` term — **even if `r` is all equal, the residual is generally nonzero, and it still produces a gradient**.

> **So on SpaceTools, on more than half the prompts, the difference between GFlowRL and GRPO
> is not one of degree but of presence vs absence.** This is more concrete than the paper's diversity score (3.93 vs 1.21),
> and sturdier than the "orchestration collapse" motivation already weakened by §2,
> **and it is about properties of the data and reward, not about whether our code is right.**

> **⚠ Labeled per the §3.3 discipline: all of the above comes from a single sampling run, n=1, never repeated.**
> For now it is **an observation pending re-check, not a conclusion**.
> Re-checking it needs exactly the "second independent set of 5 samples" already listed in the handoff document —
> **that task now has a second reason, and this reason matters more than the original one (measuring spread for majority voting).**
> Also: the training data is `spacetools-rlfulltools`, not these three eval benchmarks;
> **the degeneracy rate may differ on the training distribution, and cannot be extrapolated from the existing data.**

> **⚠ Correction 2026-09-01 (after going back to the PDF).** The statement above, "GFlowRL still produces gradient on degenerate groups, GRPO does not",
> **still holds mathematically** (Eq. 8 is a per-trajectory squared residual, not a within-group contrast),
> **but it is not a property of "GFlowRL's published configuration"**: the rollout settings in the paper's Table 9 say
> `Filter groups: Accuracy-based` — **groups with identical rewards are dropped whole**,
> so under their recipe the problem never comes up. For us it is a dilemma:
> copying the filter would drop **53%–76%** of prompts at `G=5` (on top of the existing 8× throughput deficit);
> not filtering is **an active deviation from the paper's recipe, which we have to justify ourselves**.
> **This is downgraded from "one of our advantages" to "a design decision that must be made and must be justified", and goes under criterion (iii).**

### 3.3 Recompute

    cd p6/passk && python3 - <<'PY'
    import json, statistics as st
    from collections import defaultdict
    for k in ['robospatial','blinkdepth','boppose']:
        g = defaultdict(list)
        for line in open(f'{k}/0.jsonl'):
            d = json.loads(line); g[d['index']].append(d)
        degen = sum(1 for v in g.values()
                    if max(x['score'] for x in v) - min(x['score'] for x in v) < 1e-9)
        print(k, len(g), '%.1f%%' % (100*degen/len(g)))
    PY

---

## 4. Disposition of B and C

### 4.1 C (full same-starting-point comparison): excluded, for harder reasons than the handoff document gives

The upper bound on the movable share is **33/2001 = 1.65 pp** (handoff definition) or **44/2001 = 2.20 pp** (§2.1 corrected definition),
**and that is an upper bound for eating all of it, missing none**. Meanwhile the spread of a single run:

    robospatial VQA four runs 161 / 167 / 168 / 169   =  8 samples = 3.5 pp
    aggregated to 2001 samples                       ≈  0.5–0.8 pp

> **⚠ 2026-09-02 · One correction (checked on HuggingFace directly, not second-hand).**
> "No SFT checkpoint" **holds**: there are only 2 model repos under `siyich`,
> `spacetools-ckpt` has only one branch `main`, no tags, only one set of weights,
> its README marks `base_model: Qwen/Qwen2.5-VL-3B-Instruct` + `reinforcement-learning`,
> i.e. the post-RL one; searching the whole site for `spacetools` finds only that.
>
> **But the bar for "rerun SFT" was previously overestimated: the SFT data is public.**
> `siyich/spacetools-sft` (**7463 files / 6.32 GB**, `data/train.json` + images)
> never appeared in any record. Likewise `siyich/spacetools-rlpointtools` (4 files / 1.00 GB) was never mentioned.
> Add `ChicyChen/SpaceTools-SFT`, code Apache 2.0 (commit `b7ebbf32`):
> **the blocker for rerunning Step 3 is compute, not materials.**
>
> **This does not change the conclusion that C is excluded** — that rests on the movable share of 1.65–2.20 pp being the same order as noise,
> and on the 4-GPU machine's 8× throughput deficit. But the weight of "we would also have to rerun SFT ourselves to get the same starting point" drops a notch:
> it is an executable training run, not an expedition that first has to solve a data problem.

**The expected effect is the same order of magnitude as the noise.** Resolution would require per-sample pairing + **multiple seeds on both sides**
(a direct requirement of item 3 in §3.3), both arms × multiple seeds × 3–4 days, and we would first have to rerun SFT ourselves to get the same starting point.

> **It is not that C is not worth it; the cost of reaching that resolution is unaffordable.**

### 4.2 B: agree to put it downstream of A, but change the target

The handoff document says "recover those 14 samples in `robospatial`". P6 §6.4 has already broken those 14 down:

- **The 105 `fit` questions are a tool-set gap** — no tool returns a free-space extent, and accuracy on the `no` class is **4/18 = 22%**.
  **No objective swap can move it: what the model lacks is information, not distribution.**
- **The 29 `front/behind` questions** are clean 2a: a known valid alternative chain exists, and the upper bound is clear (**+8 samples**).

**So B should be rewritten as: "Will distribution matching get `depth_estimator` called again on these 29 questions?"**

---

## 5. Verified: length normalization in Eq. 4 and Eq. 6 (2026-09-01, back to the original `2607.13394v1.pdf`)

**The notes did not misreport it — this asymmetry is in the paper as is, and it matters much more than "a typo".**

    Eq. 4   Z_t(x) := (1/G) Σ_i ( β·r + log π_ref(y_i|x) − log π_old(y_i|x) )       no 1/|y|
    Eq. 5   Δ_i    = sg[Z_t] + (1/|y_i|)·log( π_θ(y_i|x)   / π_ref(y_i|x) ) − β·r    has 1/|y|
    Eq. 6   g_i    = sg[Z_t] + (1/|y_i|)·log( π_old(y_i|x) / π_ref(y_i|x) ) − β·r    has 1/|y|
    Eq. 8   L      = (1/G) Σ_i w_i · ( g̃_i + (1/|y_i|)·log( π_θ / π_old ) )²

Appendix B's Eq. 10 (the population form of `Z_t`) likewise has **no** length normalization, confirming this is not a typesetting issue.
**Prop. B.1 is proved under the premise of "no length normalization"**; the consequences of length normalization are put separately in Remark B.4.

### 5.1 The original text of Remark B.4 is more serious than the "slight bias" in the notes

The paper itself writes out the fixed point after length normalization:

    π_θ*(y|x) = π_ref(y|x) · exp( |y| · ( β·r(x,y) − Z_t(x) ) )

**The effective inverse temperature is `|y|·β`, not `β`.** Original text: "in the idealized setting where all rollouts share
a common length |y| = L, the fixed point recovers the reward-tilted distribution at inverse
temperature **Lβ**; equivalently, Proposition B.1 applies with **β ↦ Lβ**."
When lengths differ, `exp(−|y|·Z_t)` **is no longer a normalizing constant shared across sequences**, and the fixed point is distorted by length.

**Two direct consequences:**

1. ~~**Handoff document §2④ "start β at the paper's default of 8" is underdetermined. The transferable quantity is `L·β`.**~~
   **⚠ 2026-09-01 withdrawn, see `records/P7_STEP23_RESULTS.md` §2.3.**
   The numerical test shows: the `c = L` form is a zero only **when `Z_t` is treated as a free constant**;
   once `Z_t` is required to be self-consistent per Eq. 4, the minimum falls at **`c ≈ 0.93–1.15`, not `c = L`**.
   **The tilt does not inflate with `L`; this consequence is void.**
   **The conclusion "β should not be taken directly as 8" still holds, but for a different reason**: with binary reward + `G=5`,
   the flow gap is **100% clip-saturated for non-degenerate rollouts**; avoiding saturation needs `β ≲ 1.4`.
   (This was inferred from one sentence of the paper and written into the document without running any numbers. Overturning it took thirty lines of code.)
   In passing, reading the paper itself with the §3.3 discipline: the 37.9 / 37.2 / 38.2 / 37.7 / 36.4 for β ∈ {1,5,8,10,15}
   **are one number per β** — the claim "insensitive within [1,10]" is itself a single point.

2. **A new implementation trap, not the same place as handoff document §2①.**
   `|y_i|` must be **the number of tokens generated by the policy** (i.e. the sum of `response_mask`), **not the raw response length**.
   The response of a multi-turn trajectory contains `<tool_response>`, each capped at `max_tool_response_length = 2048`,
   and one sample can have 4–5 calls.
   **If normalized by raw length, the effective temperature `L·β` is set by how verbose the tool output is, not by reasoning length.**
   §2① says the **numerator** of `log π` must be masked; this one says **the denominator must be masked too**;
   both use the same `response_mask`, but missing either one gives completely different symptoms —
   the former computes the wrong probability, the latter silently changes the temperature of the target distribution.
   **Quantified on 2026-09-01** (`P7_STEP23_RESULTS.md` §1.2): raw/assistant length ratio
   `robospatial` **1.14×** · `blinkdepth` **1.41×** · `boppose` **2.40×** —
   **and it differs systematically by benchmark**, which amounts to imposing on different tasks different temperatures set by tool verbosity.

### 5.2 Two more things checked in passing, both bearing directly on criterion (i)

- **`G = 16` (paper Table 9), while we have `rollout.n = 5`.**
  Remark B.2 states the estimator variance is **O(1/G)**; the first sentence of the limitations in Appendix A is
  "in-batch MC estimate of log Z can have higher variance in principle,
  **especially when the group size is small**".
  **16 → 5 is 3.2× the variance; we are standing right on the side least favorable for variance.** This is exactly what criterion (i) has to measure.
- **The second limitation sentence in Appendix A confirms A′'s gap verbatim**: "it remains unclear whether the same
  estimator-centric design principle extends to broader **agentic or multimodal RL** settings".

### 5.3 The remaining hyperparameters transcribed (paper Table 9, for alignment during implementation)

    G = 16 · T = 1.0 · top-p = 1.0 · lr 1e-6 · AdamW · warmup 10 · weight decay 0.1
    grad clip 1.0 · KL 0.0 · entropy 0.0 · loss aggregation token-mean
    ε_low 0.2 / ε_high 0.28 (asymmetric flow-gap clipping)
    importance sampling sequence-level · IS threshold 2.0 · ratio scaling geometric · RS threshold 1.01 / 0.99
    Filter groups: Accuracy-based        <- see the correction in §3.2

    Ablation removing flow-gap clipping: average score 40.92 -> 37.02; gradient norm mean 0.095 -> 0.601 (6.3×),
    max 6.184 -> 16.57. **Clipping is not optional.**

---

## 6. Order of next steps (replaces handoff document §5)

The order in handoff document §5 is loss → mask → gradient norm. **Under A′ it should become:**

1. ~~**Verify Eq. 4 / Eq. 6 in §5** (back to the PDF, zero cost)~~ — **completed 2026-09-01, see §5.**
   Three outputs: β cannot be transferred directly (transfer `L·β`), `|y_i|` must use the sum of `response_mask`, `G=16→5` is 3.2× the variance
2. ~~**Measure the three terms of `Z_t` separately on `p6/passk`**~~ — **completed 2026-09-01, see `records/P7_STEP23_RESULTS.md` §1.**
   `tools/p7/p7_zt_offline.py`. What remains, `log π_ref` / `log π_old`, needs one
   **forward pass, no tools attached, no sglang pool started** — **a much smaller GPU request than training, scheduled separately**
3. ~~**Synthetic fixed-point test**~~ — **completed 2026-09-01, all PASS, see §2.**
   `tools/p7/p7_fixedpoint.py` (T1–T6). **It has already taken "is the loss written correctly" out of P7's question**
3b. ~~**Synthetic half of criterion i-b**~~ — **completed 2026-09-01, see `records/P7_CRITERION_IB.md`.**
   `tools/p7/p7_estimators.py`. **Headline**: `Eq.4` unnormalized / `Eq.5-6` normalized makes the drift a
   **common shift** within the group; beyond the clip half-width it clips the whole group's `g̃` to the same value — **the reward contribution is erased**,
   with a threshold of only **≈5e-4 nat/token**. **To do**: use `p7_fixedpoint.py` to measure the fixed-point cost of the §6 fix
4. ~~**Wire up `response_mask`**~~ — **completed 2026-09-01, see `records/P7_STEP4_RESULTS.md`.**
   `|y_i| = response_mask.sum()`, confirmed line by line from the verl source. **Two new findings:**
   (a) the name `response_mask` has **two opposite semantics** — the agent loop gives "policy tokens",
   while the fallback `compute_response_mask()` at `ray_trainer.py:157` is `attention_mask[:,-L:]`,
   **tool tokens are all 1**, and the fallback is **silent** -> a runtime guard must be added (§2 gives how to write it);
   (b) see the new precondition under step 5 below
5. **A second independent set of 5 samples** — simultaneously the precondition for criterion (i) and the re-check of §3.2.
   **⚠ Correction 2026-09-01: the goal "measure the drift distribution with a forward pass" does not hold.**
   **No training ever happened** in the P4/P6 data, `π_old = π_ref = the same ckpt`,
   so `log π_ref − log π_old` is **identically 0** on this data, however many forward passes are run.
   **Change it to measuring the logprob difference between rollout (sglang) and trainer (FSDP)** —
   which is exactly the reason GFlowRL's IS weight `w_i` exists (the paper cites Yao et al. 2025);
   **it exists even with zero training**, and is the first real proxy for drift.
   Good news: `calculate_log_probs` is **one switch that controls both sides**,
   so the forward pass is **merged into the sampling run**, no separate GPU session needed.
   The full GPU-side task spec is in **`records/P7_GPU_HANDOFF.md`**.
   **⚠ New precondition: the dump patch must be applied before the GPU session.**
   `p4/dumps/` and `p6/passk/` store text only, without token ids or masks,
   so a forward pass over recorded trajectories would have to **re-derive the mask from text** — exactly what §2① forbids.
   Following the shape of `patches/rl/0008`, additionally write
   `response_mask_rle` (run-length encoded) and `n_policy_tokens` in `_dump_generations()`.
   **Apply the patch before running, otherwise it has to be rerun after.**
6. Only after all of the above pass, talk about getting on GPUs and downloading the 3.38 GB of training data

> **2026-09-01 · Landing cost revised down, but one more risk** (based on `records/P7_PRIOR_ART.md`):
> · **No need to modify `fsdp_workers.py` and `fsdp_vllm.py`** — FlowRL modifies those two files only to place and shard
>   `ProjZModule`, and that is exactly what GFlowRL deletes. **Our change is smaller than FlowRL's.**
> · In our fork (`SpaceTools-RL @ f0742338`, verl 0.8.0.dev) **`register_policy_loss`
>   and `ref_log_prob` are both already in place**; the minimal path is about ten lines: `use_kl_loss=True` + `kl_loss_coef=0`
>   gets ref logprob for free; add `token_level_scores` to `select_keys` (`advantages` is group-normalized,
>   GFlowRL needs raw `r`); register `"gflowrl"`; pass two more quantities at the call site at `dp_actor.py:615`.
> · ⚠ **FlowRL's reproduction path is verl 0.4.0, ours is 0.8.0.dev — the pattern transfers, the API may not.**
>   **Do not assume it can be copied as is.**
> · The already registered `bypass_mode` (`core_algos.py:2236`) already handles IS weights and the rejection mask,
>   the same kind of thing as GFlowRL's `w_i = min(π_θ/π_old, 1+ε)`; **read it once before writing our own**.

**Prohibitions that still apply**: see `P7_HANDOFF.md` §4, and
`model_dtype=bf16` must never be carried into training (deviation `[20]`),
and any number quoted from an 80 GB run must state `gmu` (deviation `[23]`).
