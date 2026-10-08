# SpaceTools reproduction: Accuracy report

> **Scope**: reproduction results for the paper's Table 2, attribution of the deviations, reproducibility notes.
> Error attribution (which kind of error, which tool is responsible) is P6 material and is written up separately — this report only
> cites its conclusions when explaining the gaps.
>
> **Subject**: SpaceTools (CVPR 2026, arXiv:2512.04069), official checkpoint
> `siyich/spacetools-ckpt`, **no retraining**.
> **Written** 2026-08-30 · records in `records/P5_RESULTS.md` · raw data in `p4/`

---

## Summary

Using the officially released checkpoint and eval sets, we ran all nine eval
keys of the paper's Table 2 on a self-built environment, 2121 samples, and reassembled ten numbers (of which **nine are independent**).

Converting each item's deviation into **number of samples** (n differs tenfold across benchmarks, so pp are not directly comparable):
**eight of the ten items deviate by at most 4 samples, most by only 1–3, in both directions, with no systematic bias.**
Two items are far beyond that magnitude, and in opposite directions:

| | Difference | Status |
|---|--:|---|
| RoboSpatial · VQA | **−6.13 pp** | Unexplained. Two candidates ruled out |
| BOP-ASK · Pose | **+18.99 pp** | Unexplained. One candidate ruled out |

The run side left nothing suspicious: **all five health metrics are zero across the 2121 samples**
(OOM, truncated responses, turns exhausted, missing `<answer>`, malformed tool call),
and the turn-count reconciliation passes per-sample for 2121/2121. **So the two gaps are real problems, not run accidents.**

The deviation attribution went item by item through the 23 deviations in `PROVENANCE.txt`; after review only two cannot be ruled out
(cudnn 9.16, numpy 2.x), and both are hard constraints that cannot be avoided. Stronger evidence is
**consistency across three hardware generations**: on A6000 / A100-40GB / A100-80GB, the
stable core of blinkdepth and the success/failure split of bopgrasp do not move.

**Conclusion: the reproduction holds, the definitions are comparable, and the two gaps are quantified and bounded.**

---

## 1. Definitions used in the reproduction

Before reporting any number, first write down clearly "what exactly we measured". Each of the following affects how to read the numbers.

### 1.1 Decoding is greedy, and that is grounded

verl's `val_kwargs` defaults to `temperature 0 / top_p 1.0 / n 1 / do_sample False`,
and the released repo's `run_eval.sh` **does not override it**. The `rollout.n=5` in the script is the
**training group size** and does not apply to eval.

**So the policy side is deterministic, and sampling randomness can be ruled out of the deviation attribution directly.**
What decoding the paper used is unknown, but "run with the released repo's defaults" is the only grounded definition.

> The later sampling experiments (`n=5, T=1.0`) are **a different inference strategy**, not the same measurement,
> and their results do not go into Table 2. See §3.2 and the P6 report.

### 1.2 Only nine of the ten numbers are independent

`RoboSpatial · Overall` is the **sample-weighted average** of VQA and Vacant, not an independent measurement.
Back-computing from the paper's own numbers:

    (228 × 79.38 + 122 × 52.46) / 350 = 69.996      paper reports 70.00

Off by 0.004. The VQA / Vacant split is given naturally by the GT shape (`Yes`/`No` is VQA,
a `[(x, y), …]` point list is Vacant), giving 228 / 122.

**Readers should not treat the ten numbers as ten independent pieces of evidence.**

### 1.3 Three ways of reporting, and why

| Case | How reported | Reason |
|---|---|---|
| Small benchmarks (n ≤ 124) | **Report an interval, not a single-run point value** | The pipeline has two sources of non-determinism (see §4.3); a single-run point value invites a wrong reading |
| `blinkdepth` | **Report the stable upper bound 112/124 + single-run band 106–109** | A single run anywhere in the band would be read as "3 points below the paper", and multiple runs have already falsified that reading |
| RefSpatial | **Give both averages** | The paper gives only one number and does not state the aggregation; the two conventions differ by 0.44 pp |

> **⚠ An important limitation on the "stable upper bound".**
> Reporting the stable upper bound is correct **as a way of reporting** (it is more robust than a single-run point value),
> **but you cannot infer "the paper also reported an upper bound" from "the upper bound exactly equals the paper's number".**
> The upper bound is a monotonically increasing function of the number of samples drawn; any target value will be reached sooner or later,
> and equality only shows that we happened to run exactly that many times. This was falsified once by measurement in §3.1.

---

## 2. Results: Table 2 comparison

| Table 2 row | Source key | n | This reproduction | Paper | Difference | **≈ samples** |
|---|---|--:|--:|--:|--:|--:|
| RoboSpatial · VQA | `robospatial` (split by GT shape) | 228 | 73.25–73.68% (2 runs) | 79.38 | **−5.70 ~ −6.13** | **−13 ~ −14** |
| RoboSpatial · Vacant | same as above | 122 | 50.82–51.64% (2 runs) | 52.46 | −0.82 ~ −1.64 | −1 ~ −2 |
| RoboSpatial · Overall | **weighted average, not independent** | 350 | 65.43–66.00% | 70.00 | −4.00 ~ −4.57 | (−14 ~ −16)‡ |
| BLINK · Relative Depth | `blinkdepth` | 124 | stable upper bound **112/124 = 90.32%**, single-run band 106–109 | 90.32 | upper bound equal † | 0 † |
| RefSpatial (average of three) | `reflocation`+`refplacement`+`refunseen` | 277 | 53.35 simple / 53.79 weighted | 53.07 | +0.28 / +0.72 | +1 ~ +2 |
| CVBench · 2D Relation | `cvb2drelation` | 650 | 94.62% | 94.92 | −0.30 | −2 |
| CVBench · 3D Depth | `cvb3ddepth` | 600 | 96.50% | 96.00 | +0.50 | +3 |
| BOP-ASK · Pose | `boppose` | 60 | **53.36 mean IoU** (3 runs bit-for-bit identical) | 34.37 | **+18.99** | **equivalent to +18** § |
| BOP-ASK · Grasp MACE | `bopgrasp` | 60 | 43.07–46.17 (3 runs) | 43.06 | paper falls on our lower bound | 0 |
| BOP-ASK · Grasp SR | same as above | 60 | 55.00–56.67% (3 runs) | 50.00 | +5.00 / +6.67 | +3 ~ +4 |

† **An equal upper bound is not evidence that "the paper reported an upper bound"**, see §1.3 and §3.1.
‡ Overall is a weighted average, not an independent measurement, and is not counted among the "eight items".
§ Pose is a continuous mean IoU with no right/wrong; here it is converted into "equivalent to how many extra samples scoring 0", derivation in §3.2.

**Converting to number of samples is necessary**: n ranges from 60 to 650, a tenfold difference, so the same pp on different benchmarks
represents sample counts that differ by an order of magnitude. By sample count, apart from VQA and Pose,
**the largest deviation among the other eight items is 4 samples**.

### The two RefSpatial aggregations

| | reflocation | refplacement | refunseen | Average |
|---|---:|---:|---:|---:|
| This reproduction | 54.00 (54/100) | 58.00 (58/100) | 48.05 (37/77) | |
| Simple average of three | | | | **53.35** (+0.28) |
| Sample-weighted (n=277) | | | | **53.79** (+0.72) |
| Paper | | | | 53.07 |

At n=277 one sample is 0.36 pp; the two conventions differ by less than half a percentage point.
**The report uses the simple average (53.35) and also gives the weighted value.**

### BOP-ASK Pose metric mapping (once a gate)

The `score` of `boppose` **is not accuracy**. `verl/utils/reward_score/bop_ask_bench.py:172`
defines it for `question_type == "pose"` as: take 8 points each from GT and prediction (the 2D
projections of the 3D box corners); if either side has fewer than eight, return 0.0, otherwise return **the IoU of the two convex hulls** — a continuous value in [0,1].

Early on, based on "the metric is completely insensitive to corner order", we inferred that the paper did **not** use this metric.
**After checking the paper text, that inference was overturned** — the appendix's reward definition says:

> "Predicted and ground-truth poses are converted to eight 2D projected corners.
> The reward is the **IoU between convex hulls** of predicted (Ĉ) and ground
> truth (C) corner sets. R_IoU = IoU(C, Ĉ) when both sets are valid
> (|Ĉ| = |C| = 8), and 0 otherwise."

§5.1 also says "normalized Intersection-over-Union (IoU) in range [0, 100] (%)".
This corresponds verbatim to `bop_ask_bench.py:172`. **Order insensitivity is not a defect, it is the spec.**

Recomputing from `raw_answer` with the repo's own geometry functions matches the scores in the dump **60/60 bit-for-bit**
(max absolute difference 0.000e+00). **So this cell can be filled, and it must be filled with 53.36** — the metric is
confirmed comparable, and leaving it empty would amount to concealment.

---

## 3. The two gaps

Eight items fall within noise. The remaining two go in opposite directions and are explained separately.
What they share: **every candidate that can be measured has been measured**; what remains either cannot be tested from outside,
or belongs to P6's manual classification.

### 3.1 RoboSpatial · VQA: −6.13 pp

**The gap is stable, not sampling.** Four independent runs:

    single runs   161 / 167 / 168 / 169       mean 166.2/228 = 72.92%
    paper                                           181/228 = 79.38%
    difference                                 14.8 samples

The best run (169) is still 12 samples short of the paper. Merging two runs, of the 350 samples
218 are always right, 108 always wrong, 24 flip — **the stable core itself is below the paper.**

#### Candidates ruled out

| Candidate | How it was ruled out |
|---|---|
| **The paper reports the upper bound over multiple samples** | With three runs the upper bound = VQA 181 / Vacant 64 / Overall 245, **exactly equal to the paper in three places**; after a fourth run the VQA upper bound rose to **188**, 7 samples past the paper. And picking any three of the four gives 181/183/186 — that "exact hit" was itself a coincidence of which combination was chosen. **The upper bound is not saturated, so the criterion does not hold.** |
| **The tool found nothing, so the model guessed** | `roborefer` returned non-zero detections on all 228 samples |

#### Three leads that still stand

1. **The `no` class collapses to chance level.** The 165 items with GT `yes` are stable at 83.03% (**identical** across the two
   runs); the 63 items with GT `no` get only 47.62% / 49.21%. This is not a naive
   yes preference — the marginal distribution of outputs (yes 170/169, no 58/59) is very close to GT (165/63)
   — but something systematic does push it toward yes on hard questions.
2. **Structural mismatch between question and tool output.** Of the 228, **105 are "Can X fit ⟨rel⟩ Y?"**
   — they ask "does it fit", while the tool chain gives the **point positions** of two objects, not the **extent** of free space.
   Split by relation word, accuracy is fairly uniform (66.7–77.1%); **the gap is not concentrated in one kind of
   spatial relation**.
3. **Instability at the decision layer (new).** Under five samples, 100/228 (43.9%) of VQA questions swing
   between samples, and for **46 of them the five tool-call sequences are verbatim identical** — same image, same tool returns,
   opposite yes/no answers. These are not tool errors, and this points to the same place as lead 2:
   the middle step is not pinned down by any tool output.

> **But do not say the gap has been explained away.** Majority vote@5 gets 176/228, above all four greedy runs
> (161–169), but per-sample paired tests (McNemar) are significant only against the lowest run (p=0.014),
> against the other three p = 0.163 / 0.230 / 0.281, and all four comparisons share the same vote arm.
> ~~**The claim "self-consistency recovers about 60%" has been downgraded to "same direction, not significant".**~~
> ~~Settling it needs a second independent set of 5 samples.~~
>
> **⚠ 2026-09-02 · The second sampling set is done; downgraded one more level: majority vote@5 has no effect.**
> Set 2's vote arm gets **166/228** (set 1 got 176/228) — **the vote arm's own spread is 10 samples,
> the same magnitude as the spread of the four greedy runs.** Redoing the paired tests against the same four greedy runs:
> **0/12 comparisons are significant, and the VQA direction flips sign between the two sets**
> (set 2 · VQA net −1 / −2 / +5 / −3, p = 1.000 / 0.868 / 0.500 / 0.720).
> Set 1's `p=0.014` is exactly the kind of artifact described in row 3 of the §7 table.
> **The lead-3 phenomenon still holds (pure decision flips 46 / 62), but it does not amount to a recoverable gain.**
> Source: `records/P7_GPU_RESULTS.md` §3.4.

**Conclusion: −6.13 pp is a real gap and is, as of now, entirely unexplained.**
**(2026-09-02 update: with majority vote@5 downgraded to no effect, it has one fewer candidate explanation than when this report was written.)**

### 3.2 BOP-ASK · Pose: +18.99 pp

The metric is confirmed to match the paper (§2), so this is **a real difference, and we are the higher one**.
First, quantify the gap in a testable form:

    ours    60 × 0.5336 = 32.02
    paper   60 × 0.3437 = 20.62
    gap                   11.40

There are 51 non-zero samples with mean 0.628, so going from our distribution to the paper's number
**is roughly equivalent to 18 extra samples scoring 0**.

And the caption of the paper's Table 2 says:

> Values of 0 indicate the model either fails to produce valid responses,
> **outputs answers in wrong formats**, or produces entirely incorrect predictions.

In the Pose column almost every baseline is 0.00 (Qwen2.5-VL-3B, Molmo, RoboRefer, RoboBrain,
SpaceLLaVA, RoboPoint all 0.00), which shows that **producing eight valid coordinates is itself hard**.
This reproduction is **60/60 format-valid, zero format failures**.

| Candidate | Status |
|---|---|
| **Format failures on the paper's side** (about 18 samples scoring 0) | **Most self-consistent**, consistent with the caption and the column of 0.00s. **Cannot be tested from outside** |
| ~~Different decoding~~ | **Ruled out by measurement**, see below |
| Different eval set | The released `boppose.parquet` has 60 items; the paper does not give the BOP-ASK split size. Needs comparison against the original BOP-ASK benchmark |
| Different checkpoint | The other eight items match, which weakens this candidate |

#### Ruling out the "decoding" candidate

If the paper reported avg@1 with temperature 1.0 sampling, this candidate predicts a lower score and more format errors.
Rerunning with an explicit override `val_kwargs.do_sample=True / n=5 / temperature=1.0`, **neither holds**:

    sampled avg@1   53.60         vs greedy 53.36 (0.24 higher)      paper 34.37
    zero-score rate 42/300 = 14%  vs greedy  9/60 = 15% (1 pp lower)

The standard error of the mean over 300 samples is about 2 points; the paper's value is 19 points away.

> **Only `avg@1` can be used here, not `best@5`.** Taking the max of 5 has selection bias —
> even if the scores were pure noise, the max would be above the mean (measured `best@5 = 57.23`). That is not a gain.

**Conclusion: three candidates remain, and the most self-consistent one cannot be tested from outside.**

---

## 4. Deviation attribution

`PROVENANCE.txt` records 23 deviations from upstream. After going through them one by one:

### 4.1 Ruled out

| Deviation | Grounds for ruling out |
|---|---|
| `preprocessor_config.json` replaced | Byte-for-byte identical to the Hub (sha256 `f2058c71…`), and Qwen has not changed that file since 2025-02, so the paper necessarily got the same copy |
| Sampling randomness | eval is greedy (§1.1) |
| sm_80 rebuild of `pointnet2_ops` | An architecture mismatch would `exit(-1)` and kill the process outright rather than fail silently, and the full run had zero crashes |
| The other 18 | Hard blockers, verified no-ops, or pure scheduling |

### 4.2 Could still move the numbers

| Deviation | Why it might | Why it cannot be avoided |
|---|---|---|
| **cudnn 9.16.0.29** | Differing convolution results are exactly what pytorch#168167 discusses | sglang reads `torch.backends.cudnn.version()` at startup and refuses to run below 9.15 |
| **numpy 2.x** (`spacetools-rl`) | No known effect; the pin is stale metadata | Ray cannot deserialize numpy-2 arrays from the tool environment into a numpy-1 driver |
| **`model_dtype=bf16`** (pending) | See §4.4 | If it holds, it is a consequence of GPU memory footprint, not optional |

### 4.3 Non-determinism has two sources, both isolated

| Source | Symptom | Scope |
|---|---|---|
| sglang floating-point reduction order | Near-tie samples flip between runs | All benchmarks; blinkdepth had 13 flips over six runs |
| GraspGen diffusion sampling | The tool itself samples | Only `bopgrasp`: across two runs only 3/60 output texts are identical, but for 55 of the 60 the outcome is determined by scene geometry, and only 5 flip |

**For small benchmarks, reporting an interval is enough.**

### 4.4 A retracted ruling-out

`model_dtype=bf16` was set to fit the policy on a 40 GB GPU (verl's FSDP defaults to fp32
master; 4.066 B parameters is 16.3 GB; bf16 needs only 8.1 GB).

**The argument on the weight path is still conclusive**: all 825 tensors in the checkpoint are stored as BF16,
`bf16 → fp32 → bf16` is the identity, a bit-for-bit comparison gives 0 differing elements and max absolute difference 0.000e+00;
and under `val_only` the FSDP model does no forward pass at all.

**But the conclusion "ruled out end to end" has been retracted.** It was declared ruled out on the basis of a single fp32 run (107/124,
"falls in the 107–109 band"), and two more runs the same day overturned it:

    fp32 @ gmu=.25   107 · 105 · 105     range 105–107
    bf16 @ gmu=.25   108 · 109           range 108–109      the two ranges do not overlap

That 107 is **the top edge of fp32's own range**, and was read as "falls in the band".

**The conclusion in the opposite direction does not hold either**: if the five values were identically distributed, the probability that the two bf16 runs happen to take the top two places is
`1/C(5,2)` = 0.1 (strongly suggestive, not significant); structurally only **1 sample** (#49) is
"bf16 always right, fp32 always wrong", and 0 the other way; fp32's own three runs also differ by 4–6 samples.

**The correct statement is "evidence for ruling out is insufficient, retracted", not a conclusion in either direction.**

> What was actually shaken is the **implicit premise** of the weight argument — "the only path by which `model_dtype` affects the numbers is
> the weights". Measurement hints at a second path (**unverified**): fp32 master takes about 8 GB more
> (peak 33.8 vs 30.4 GB), different GPU memory allocation → different sglang batch composition → near-tie flips.
> **This is the same kind of mechanism as §4.5, not a weight-level effect.**

### 4.5 New deviation: `gpu_memory_utilization` is a fraction of the whole GPU

After moving to A100-80GB, the GPU memory peaks of the three tool GPUs are **byte-for-byte identical** to the old machine
(Molmo 34.6, DepthPro 25.7 GB); only the policy GPU went from 25.6 to 46.8 GB.
Reason: this knob is a static fraction of the **whole GPU** — the same 0.5 is a 20 GB KV pool on 40 GB
and a 40 GB pool on 80 GB.

Pool size changes batch composition → floating-point reduction order → near-tie samples flip,
**exactly the mechanism identified in §4.3**. So **80 GB is not a numerically neutral substitution on the policy side.**

**Handling: policy runs on 80 GB always use `gpu_memory_utilization=0.25`**
(reproducing P4's pool size), and **every number quoted from an 80 GB policy run notes its value**.
All numbers in §2 of this report come from the original 40 GB / 0.5 runs and are not affected.

### 4.6 Cross-hardware consistency: stronger evidence than item-by-item checking

| Observation | A6000 (sm_86) | A100-40GB | A100-80GB |
|---|---|---|---|
| blinkdepth stable upper bound | 112/124 | 112/124 | 111/124 (4 runs) · six runs merged 112 |
| blinkdepth always wrong | 12 | 12 | 13 (intersection with 40GB: 12) |
| bopgrasp success/failure | 19 / 41 | 20 / 40 | — |
| bopgrasp MACE / SR | 44.46 / 55.00 | 43.07–46.17 / 55.00–56.67 | — |

Across the two platforms the "always right" intersection is 99 and the "always wrong" intersection is 12.

**The whole accumulated set of deviations — bf16, sm_80, cudnn 9.16, numpy 2.x, rebuilt CUDA extensions —
taken together did not move the stable core of any benchmark.** This is stronger than any item-by-item argument about single deviations,
because it is end to end.

> **But read it by number of runs, not by platform.** The 112/124 in the table above was only reached by merging six runs;
> 80 GB alone over four runs is 111/124, and the two existing 40 GB runs are also 111/124.
> This is the same thing as the limitation in §1.3.

---

## 5. Run health: why the numbers above can be trusted

This section draws no conclusions; it determines whether the preceding sections can be trusted.

### 5.1 All five health metrics are zero

Nine benchmarks, 2121 samples:

| Metric | Hits |
|---|--:|
| OOM | **0** |
| Tool response truncated | **0** |
| Turns exhausted (8-turn limit) | **0** |
| Missing `<answer>` | **0** |
| Malformed tool call | **0** |

The three non-zero "tool error" counts have each been traced and are not contamination: the 1 in `blinkdepth` and the 1 in `boppose`
are the model asking a tool for a variable that was never produced (the tool refuses, the model recovers on its own);
the 40 in `bopgrasp` are the grasp tool genuinely failing on those scenes, by design.

### 5.2 Turn-count reconciliation 2121/2121

The parser's own transcript decomposition agrees with the `num_turns` that verl writes to disk **on every single sample**.
**This is the foundation of the whole report's credibility** — chain signatures, tool histograms and turn statistics are all built on that
decomposition.

> This guard is not decoration. It caught a real parsing bug: the splitter used to silently drop the assistant's
> first turn, which made `depth_estimator` disappear entirely from the tool histogram of the depth benchmark
> (actually 122 calls), and reported the mean turn count as 2.13 when the true value is 3.13.

### 5.3 Turn counts and tool usage

| benchmark | n | assistant turns | tool calls/sample | dominant chain | coverage |
|---|--:|--:|--:|---|--:|
| `blinkdepth` | 124 | 3.13 | 5.16 | `depth_estimator×1+roborefer×2+vision_ops×2@3t` | 82.3% |
| `cvb2drelation` | 650 | 2.06 | 2.03 | `roborefer×2@2t` | 94.6% |
| `cvb3ddepth` | 600 | 3.04 | 5.01 | `depth_estimator×1+roborefer×2+vision_ops×2@3t` | 96.2% |
| `reflocation` | 100 | 2.00 | 1.00 | `roborefer×1@2t` | **100.0%** |
| `refplacement` | 100 | 2.00 | 1.00 | `roborefer×1@2t` | **100.0%** |
| `refunseen` | 77 | 2.00 | 1.00 | `roborefer×1@2t` | **100.0%** |
| `robospatial` | 350 | 2.00 | 1.63 | `roborefer×2@2t` | 62.3% |
| `boppose` | 60 | 5.03 | 4.03 | `bounding_box+depth_estimator+roborefer+sam2@5t` | 96.7% |
| `bopgrasp` | 60 | 4.95 | 4.00 | `depth_estimator+grasp_generator+roborefer+sam2@5t` | 95.0% |

**The three RefSpatial benchmarks each have only one chain, 277/277 with zero variation.**
This matters for reading the scores: on these benchmarks the policy is a thin wrapper around RoboRefer.
In the paper's Table 2, RoboRefer-8B-SFT alone gets 48.37 on RefSpatial, and SpaceTools-3B gets 53.07.

> `vars_unused` is 600/600 on `cvb3ddepth`, which looks like widespread wasted work,
> **but is actually a constant background**: `estimate_depth` publishes both `$depth_map` and
> `$focal_length_px`, and these tasks really do not need the focal length. It must be subtracted out before being used as a signal.

---

## 6. Methodology: three errors of the same type

Over the course of this reproduction **the same error** occurred three times, and each was overturned by later checks. It is written here because
it affects **how to read every interval in this report**.

> **Common shape: take a number that has not been repeated, compare it against a reference that has spread, then treat it as a conclusion.**

| # | What happened | How it was overturned |
|---|---|---|
| 1 | "The upper bound over multiple samples exactly equals the paper's number ⇒ the paper reported an upper bound" | The upper bound is a monotonically increasing function of the number of samples; **without requiring saturation it does not hold**. The fourth run went 7 samples past the paper |
| 2 | "A single fp32 run falls in the expected band ⇒ deviation `[20]` is ruled out" | That was the **top edge** of fp32's own range; two more runs overturned it |
| 3 | "Majority vote@5 is above the greedy interval ⇒ the gain is real" | The vote was run as only one set, giving **one value**; switching to per-sample paired tests, only one of four was significant. **After the second set was completed on 2026-09-02: 0/12 significant, direction flipped sign, downgraded to "no effect" — of the three rows in this table, this is the only one that later data zeroed out completely** |

**Prescription (now written into the process): before announcing, first ask — how many times have I run this number? How many times was the reference run?
Has the spread on both sides been measured?**

Two of the three checks were only added after someone pressed for them, which shows this step cannot rely on remembering it on the spot.

---

## 7. Reproducibility appendix

### 7.1 Model weights (pinned by commit)

| Weights | Version |
|---|---|
| `siyich/spacetools-ckpt` | `f953b1a130a74d5fca78d779dc1a686c892b239f` |
| `Zhoues/RoboRefer-8B-SFT` | `bd04070786084c624156194d89333c375b274b28` |
| `allenai/Molmo-7B-D-0924` | `cab33fb7f1a40091911f81165f8481920621948f` |
| `facebook/sam2.1-hiera-small` | `ee5bba1d82bb8749febdf90f45e84b687142ba03` |
| `GraspGenModels` | `ec1ccbb5eec0680db669246ac312a3636f16ee43` |
| `depth_pro.pt` | sha256 `3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce` (plain wget, the checksum is the only identifier) |

**Re-downloads must use `--revision <commit>`**, otherwise weight drift becomes a new source of deviation
— and that is exactly the prime suspect this report sets out to rule out.

### 7.2 Environment matrix (five conda environments, mutually incompatible)

| Environment | torch | numpy | transformers |
|---|---|---|---|
| `spacetools-rl` | 2.9.1 | 2.4.6 | 4.57.1 |
| `spacetools-tool-roborefer` | 2.5.1 | 1.26.4 | 4.49.0 |
| `spacetools-tool-vlm` | 2.9.1 | 2.4.6 | 4.53.2 |
| `spacetools-tool-bbox` | — | 2.4.6 | — |
| `spacetools-tool-graspgen` | 2.3.1 | 2.4.6 | 4.48.3 |

**Five environments are required, not laziness.** The dependencies between tools genuinely conflict
(RoboRefer's ABI after VILA downgrades it, GraspGen's three layers of dependencies);
they communicate across environments via Ray, which is also exactly where the numpy 2.x deviation comes from.

### 7.3 Hardware

| | Numbers in this report | Other platforms used |
|---|---|---|
| GPU | **4× A100-SXM4-40GB**, cc 8.0, driver 580.159.03 | A6000 (cc 8.6) · A100-SXM4-80GB (cc 8.0) |
| CUDA | 12.8 (V12.8.93) | same |
| gcc | 13.3.0 | same |

**The GPU must be one of sm_80 / 8.6 / 8.9**: `pointnet2_ops` ships cubins only for these three
and **has no PTX fallback**; switching to H100 (sm_90) or sm_120 fails directly with
`no kernel image is available for execution on the device`.

### 7.4 Code versions and our patches

| Repo | commit |
|---|---|
| SpaceTools-RL | `8d193ec3` (branch `repro-4xa6000`) |
| SpaceTools-SFT | `b7ebbf32` |
| SpaceTools-Toolshed | `712e557a` |

All of our changes to upstream are recorded in `records/CHANGES.md` and listed one by one among the 23 deviations in `PROVENANCE.txt`.
The ones directly relevant to the numbers in this report are:

- `8d193ec3` — additionally writes `uid` and `num_turns` into the dump. **This is purely additive instrumentation**;
  it does not change generation, but it makes the turn-count reconciliation in §5.2 possible.
- `model_dtype=bf16` — see §4.4. **EVAL ONLY, must never be carried into training.**
- Dataset path patch `f7560170` — all nine parquets verified to resolve.

### 7.5 How to rerun

    source /workspace/env.sh
    bash tools/p4_run.sh <benchmark> <run label>      # the label must be run<N>
    python3 tools/parse_dump.py --strict <dump dir>  # gatekeeper: health metrics must all be zero
    python3 tools/parse_dump.py --compare  ...        # report intervals over multiple runs

`p4_run.sh` has two guards, both paid for by real incidents: **refuse to overwrite an existing dump**,
**refuse labels that are not `run<N>`** (a run-together command line once turned the label into `run2bash`,
quietly creating an entire parallel results tree).

### 7.6 Raw data

| Content | Path |
|---|---|
| 13 raw dumps | `p4/dumps/` |
| 9 enriched per-sample records | `p4/parsed/` (all statistics in §5 of this report come from here) |
| eval logs and 1 Hz GPU memory traces | `p4/logs/` |
| dumps of the sampling experiment (§3.2) | `p6/passk/` |
| fp32 control (§4.4), three runs | `p6/fp32/` (**with README: looking only at run7 gives the retracted conclusion**) |
| control for deviation `[23]` (§4.5) | `p6/gmu025/` (**with README: comparability depends on KV pool size, not the knob value**) |
| all deviations, weights, environments, hardware | `records/PROVENANCE.txt` |

**Every number can be recomputed directly from the repo copy**, with no GPU and no rerun of the eval.

---

## 8. Conclusions and limitations

### Conclusions

1. **The reproduction holds.** Eight of the ten items deviate by at most 4 samples (most by 1–3),
   in both directions, with no systematic bias.
2. **The definitions are comparable.** The BOP-ASK Pose metric mapping has been confirmed against the paper text,
   and corresponds verbatim to the repo implementation; the fact that only nine of the ten numbers are independent is stated explicitly.
3. **The run side is clean.** All five health metrics are zero across 2121 samples, and the turn-count reconciliation passes per-sample.
   **The two gaps are real problems, not run accidents.**
4. **The deviations are attributed.** Of the 23 deviations only two cannot be ruled out, and both are hard constraints;
   stronger evidence is the consistency across three hardware generations.

### Limitations

- **The two gaps are unexplained.** RoboSpatial VQA −6.13 pp, BOP-ASK Pose +18.99 pp.
  Every measurable candidate has been measured; the most self-consistent remaining explanation (format failures on the paper's side) **cannot be tested from outside**.
- **The "stable upper bound" has not been shown to be saturated.** It is robust as a way of reporting, but cannot be used as evidence about the paper's methodology.
- **The end-to-end ruling-out of `model_dtype=bf16` has been retracted.** The weight path is conclusive, the overall evidence is insufficient,
  and no conclusion can be drawn in either direction.
- **This report covers only accuracy and deviation attribution.** Error attribution (which kind of error, which tool is responsible,
  how much swapping a tool would gain) is P6 material.

### One judgment left for later work

The value of a reproduction is not only in the numbers that match. **The two items that do not match, and the three retracted conclusions,
are the most informative part of this work** — each comes with a testable criterion and an
explicit overturning. Recording them as **bounded open questions** is more useful than smoothing them over.

---

*Records: `records/P5_RESULTS.md` (process and finer-grained data) · `records/P6_NOTES.md` (error attribution)
· `records/P6_GPU_RESULTS.md` (GPU experiments) · `records/PROVENANCE.txt` (deviations and versions)*
