# P4 / P5 / P6 ↔ P7 per-sample comparison analysis

Subjects: `p4/parsed/` (official checkpoint, after GRPO, run1) and P7's `parsed/`
(self-trained SFT + GFlowRL **C′** step85, the valid run with zero OOM), paired by `sample_id`.
Method: everything uses only the parsed records, zero GPU. Scripts are in `analysis/`, output in `analysis/compare_p4_p7.txt`.

Sources: `records/P4_RESULTS.md`, `records/P5_REPORT.md`, `records/P6_REPORT.md`,
`records/P6_NOTES.md`, `06_gflowrl_eval/gflowrl_eval_report.md`, `03_sft_eval/sft_eval_report.md`.

---

## 0. First: what this comparison can and cannot answer

**Can answer:** "the official post-RL checkpoint and the checkpoint we trained with C′, on the same
2121 samples, where do they differ **per-sample** in behavior". Tool chains, variable reuse, error attribution, pass-through rate are all
verifiable traces that do not depend on anyone's interpretation.

**Cannot answer:** "is C′ better or worse than GRPO". The two checkpoints have **different bases** — the official one is the paper's
own SFT starting point, ours is a self-trained SFT ckpt. This is not a controlled A/B.

**The truly controlled comparison is SFT starting point ↔ C′**, and on that side **there are only summary numbers, no dump**
(the SFT eval ran on a machine that has since been released, and the trajectories were not kept). So any conclusion about
"what C′ changed relative to its own starting point" below will be marked as relying on summary quantities recorded in the SFT report,
one notch lower in evidence strength. **This is the biggest limitation of this analysis; next time a checkpoint is evaluated, the dump must be saved too.**

---

## 1. Summary table: per-sample pairing

| benchmark | n | P4 correct | P7 correct | Only P4 correct | Only P7 correct | Main chain P4 | Main chain P7 | Calls/sample P4 | P7 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| robospatial | 350 | 229 | 215 | **37** | **23** | 62.3% | 64.9% | 1.629 | 1.646 |
| reflocation | 100 | 54 | 52 | 3 | 1 | 100.0% | 98.0% | 1.000 | 1.020 |
| refplacement | 100 | 58 | 58 | **0** | **0** | 100.0% | 99.0% | 1.000 | 1.010 |
| refunseen | 77 | 37 | 37 | **0** | **0** | 100.0% | 100.0% | 1.000 | 1.000 |
| blinkdepth | 124 | 107 | 109 | 4 | 6 | 82.3% | 85.5% | 5.161 | 5.306 |
| cvb2drelation | 650 | 615 | 615 | 6 | 6 | 94.6% | 98.3% | 2.031 | 2.009 |
| cvb3ddepth | 600 | 579 | 579 | **0** | **0** | 96.2% | 99.7% | 5.010 | 5.013 |
| boppose | 60 | 38 | 39 | 1 | 2 | 96.7% | 98.3% | 4.033 | 4.017 |
| bopgrasp | 60 | 54 | 56 | 0 | 2 | 95.0% | 90.0% | 4.000 | 4.000 |

P4 uses run1 (single run); the P4 report itself ran robospatial / blinkdepth / bopgrasp multiple times and reported intervals,
so the P4 column is one end of the interval, not "P4's true value".

### 1.1 First structural finding: on three benchmarks the two checkpoints are identical per-sample

`refplacement` (100), `refunseen` (77), `cvb3ddepth` (600) — **777 samples, not a single divergence**.
Not the same score — **the same right/wrong on every single question**.

This is not a coincidence; `p6` already gave the mechanism: on these benchmarks "reasoning" is a rule that can be written as code,
and the model executes it verbatim. The rule's input is the tool output, and the tools are deterministic, so **no matter which policy model is swapped in,
the result is the same**.

> **Implication: these three benchmarks cannot detect any difference between RL algorithms.** What they measure is RoboRefer and
> DepthPro, not the policy. Putting them in a "does RL work" comparison table only dilutes the signal.

`reflocation` has 4 divergences, `cvb2drelation` 12 (symmetric 6/6), `boppose` 3 —
likewise close to zero. **Of the nine benchmarks, the only ones with real policy freedom are `robospatial`
(60 divergences) and `blinkdepth` (10).**

### 1.2 Second structural finding: none of the nine benchmarks is significant, in either direction

McNemar exact test (two-sided binomial) on the discordant pairs of each benchmark:

| benchmark | n | P4 correct | P7 correct | Diff | Only P4 | Only P7 | McNemar p | |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| robospatial | 350 | 229 | 215 | −14 | 37 | 23 | 0.092 | Not significant |
| reflocation | 100 | 54 | 52 | −2 | 3 | 1 | 0.625 | Not significant |
| refplacement | 100 | 58 | 58 | 0 | 0 | 0 | 1.000 | **Identical per-sample** |
| refunseen | 77 | 37 | 37 | 0 | 0 | 0 | 1.000 | **Identical per-sample** |
| blinkdepth | 124 | 107 | 109 | +2 | 4 | 6 | 0.754 | Not significant |
| cvb2drelation | 650 | 615 | 615 | 0 | 6 | 6 | 1.000 | Not significant |
| cvb3ddepth | 600 | 579 | 579 | 0 | 0 | 0 | 1.000 | **Identical per-sample** |
| boppose | 60 | 38 | 39 | +1 | 1 | 2 | 1.000 | Not significant |
| bopgrasp | 60 | 54 | 56 | +2 | 0 | 2 | 0.500 | Not significant |

After using P7's two runs to suppress single-run noise, robospatial is still not significant:
P4 correct while P7 wrong both times 29, P4 wrong while P7 correct both times 17, **p = 0.104**.

For the two continuously scored ones, a paired sign test gives the same conclusion of zero, and moreover **the mean and the per-sample direction contradict each other**:

```
boppose    P4 mean 0.5336   P7 mean 0.5573   diff +0.0237
           but per-sample P7 higher 19 · P4 higher 26 · bit-for-bit identical 15    sign test p = 0.371
bopgrasp   P4 mean 1.9354   P7 mean 1.8576   diff −0.0778
           but per-sample P7 higher 25 · P4 higher 31 · bit-for-bit identical  4    sign test p = 0.504
```

`boppose`'s mean rose by 0.0237, while **most samples actually went down** — the mean was pulled up by a few samples whose IoU changed by a lot.
`bopgrasp`'s threshold count (+2) and mean (−0.078) go in opposite directions. **Neither of these numbers can
be taken as "got better".**

> **So to the question "did any benchmark get better", the answer is: three are nominally a bit higher
> (blinkdepth +2, boppose +1, bopgrasp +2), but none passes the test;
> the two nominally lower (robospatial −14, reflocation −2) are equally not significant.**
>
> Note this does not mean "the two checkpoints are the same" — on 777 samples they are **literally identical**,
> while the behavioral difference on robospatial **can be traced to a mechanism** (§6).
> The accurate statement is: **the resolution of this measurement is insufficient to decide either direction on any single benchmark.**
> To decide robospatial's 4 pp, both sides need multiple runs.

---

## 2. Directly answering the motivation P6 left for P7

P6 report §7 item 1:

> The existing policy's tool orchestration has already heavily collapsed: on the three RefSpatial, 277/277 use exactly the same chain,
> `cvb2drelation` 615 of 650 the same. **This is exactly what GRPO would do, and exactly what a distribution-matching objective claims to
> avoid.** P7's motivation went from "should in theory" to "already observed".

C′ is GFlowRL's distribution-matching objective. **It did not reduce collapse; on seven of the nine benchmarks it is actually more collapsed.**

| benchmark | n | Main chain coverage P4 → P7 | Chain signature kinds P4 → P7 |
|---|--:|---|---|
| cvb2drelation | 650 | 94.6% → **98.3%** | 10 → **6** |
| cvb3ddepth | 600 | 96.2% → **99.7%** | 4 → **3** |
| blinkdepth | 124 | 82.3% → **85.5%** | 11 → **9** |
| boppose | 60 | 96.7% → **98.3%** | 3 → **2** |
| robospatial | 350 | 62.3% → **64.9%** | 3 → 3 |
| refunseen | 77 | 100.0% → 100.0% | 1 → 1 |
| reflocation | 100 | 100.0% → 98.0% | 1 → **3** |
| refplacement | 100 | 100.0% → 99.0% | 1 → **2** |
| bopgrasp | 60 | 95.0% → **90.0%** | 2 → 2 |

On the three RefSpatial C′ did add a few chains (3 of 277 samples took a different path), and `bopgrasp`'s
main chain also loosened by 5 pp. But on **the four with large sample sizes** (cvb2d 650, cvb3d 600, blink 124,
robospatial 350), the direction is consistently more concentrated, with fewer signature kinds.

> **Conclusion: the hypothesis "a distribution-matching objective can avoid tool-orchestration collapse" was not supported in this measurement.**
>
> Caveats that must go with it: (a) this is a cross-base comparison, not a controlled A/B; (b) C′ trained only 85 steps, 1 epoch;
> (c) the training side already measured **63–80% of groups as reward-degenerate**; on those groups the gradient comes entirely from the drift term,
> doing flow matching toward `π_ref·exp(βr)/Z` rather than learning the reward — **an objective that receives almost no reward signal
> has no reason to make orchestration more diverse.** Taken together, this looks more like "not enough training for the objective
> to express itself" than "C′'s properties have been falsified".

P6 §7 item 2 also wrote: the inference-side room to exploit **has been measured, and the result is zero** (majority vote@5 significant in 0 of 12
cells). So P7's motivation was already down to just "orchestration diversity", and that one is now also a negative reading.

---

## 3. Variable reuse: P6 taxonomy 2d, the two checkpoints are bit-for-bit identical

| benchmark | | Exposed | Used | **Unused** | Phantom |
|---|---|--:|--:|--:|--:|
| blinkdepth | P4 | 269 | 125 | **146** | 2 |
| | P7 | 275 | 122 | **154** | 1 |
| cvb3ddepth | P4 | 1203 | 600 | **603** | 0 |
| | P7 | 1200 | 600 | **600** | 0 |
| boppose | P4 | 240 | 180 | **60** | 0 |
| | P7 | 240 | 180 | **60** | 0 |
| bopgrasp | P4 | 240 | 180 | **60** | 0 |
| | P7 | 240 | 180 | **60** | 0 |

On `cvb3ddepth`, **1200 variables exposed, exactly 600 used, exactly 600 left** — on every sample
`estimate_depth` exposes two variables, `$depth_map` and `$focal_length_px`, and the model always uses only the former.
On `boppose`/`bopgrasp`, 240/180/60 is bit-for-bit identical on both sides.

> P6 warned that `vars_unused` must first subtract the constant background (things like `$focal_length_px` that nobody uses anyway).
> After subtracting it, **real "got it but didn't use it" is nearly zero**, and there is no difference between the two checkpoints.
> **C′ changed nothing on the variable-reuse dimension.**

---

## 4. Error attribution: the conclusion that 322 wrong answers are 100% clean holds equally on P7

P6's most important structural finding was "322 wrong answers are 100% clean — not a single wrong answer has an infrastructure excuse".
On the P7 side, rerunning the same attribution:

```
                    P4 error attribution     P7 error attribution
robospatial         clean 121                clean 134 · no tool called 1
reflocation         clean 46                 clean 48
refplacement        clean 42                 clean 42
refunseen           clean 40                 clean 40
blinkdepth          clean 17                 clean 15
cvb2drelation       clean 35                 clean 35
cvb3ddepth          clean 21                 clean 21
boppose             clean 22                 clean 21
bopgrasp            tool failure 5 · clean 1 tool failure 4
```

OOM, generation truncation, tool-response truncation, hitting the turn limit, malformed tool_call, answer-parse failure — **all 0 on both sides**.
The only non-clean ones are `bopgrasp`'s `grasp_generator` domain failures (P4 5, P7 4), and one sample on P7's
`robospatial` that answered without calling any tool at all.

> This one matters, because it **pins** the question "why is C′'s score low" **to policy behavior**: not the environment, not GPU memory,
> not truncation. The OOM=0 under three independent definitions in the P7 eval report is the other half of the evidence for the same thing.

---

## 5. Rerunning P6's three criteria

### 5.1 Criterion A — do depth questions verbatim follow "pick the one measured closer"

```
blinkdepth    P4 follows 109 violates 5 undecidable 10   compliance 95.6%
              P7 follows 109 violates 5 undecidable 10   compliance 95.6%     <- bit-for-bit identical
cvb3ddepth    P4 follows 598 violates 1 undecidable 1    compliance 99.8%
              P7 follows 598 violates 2 undecidable 0    compliance 99.7%
```

**P6's core insight holds unchanged on P7**: on depth questions, "reasoning" is a rule the model executes verbatim,
and errors can only come from the numbers the tools give. C′ did not move it by one percentage point.

### 5.2 Criterion B — are pointing questions verbatim pass-through of tool output

```
              P4               P7
reflocation   99/99            94/99
refplacement  100/100          98/100
refunseen     77/77            76/77
total         276/276 = 100%   268/276 = 97.1%
```

The "276/276 verbatim pass-through" recorded in P6 reproduced on P4. **C′ made the model stop passing through verbatim on 8 samples.**
This looks like "the policy started participating", but the next section shows that on this task, participating means getting worse.

### 5.3 Criterion C and question-type breakdown — robospatial VQA

```
question type                      n     P4 accuracy   P7 accuracy    P4 answers yes   P7 answers yes   GT=yes
fit (needs free space)             105     69.5%      69.5%      83/105     77/105      87
relation (2D-decidable)             94     77.7%      77.7%      69/94      73/94       60
front/behind (needs depth order)    29     72.4%      62.1%      18/29      13/29       18
```

- **On `fit` and `relation` the two checkpoints' accuracy is bit-for-bit identical.** The biggest gap P6 located
  — "`fit` questions almost always answered yes, 83 times out of 105" — C′ lowered yes from 83 to 77,
  **accuracy did not budge (69.5% → 69.5%)**. The 6 fewer yes answers did not win a single question.
- **The only thing that moved is `front/behind`: 72.4% → 62.1%, 3 fewer correct on n=29.** P6 judged this category as
  **clean 2a (should have called, didn't)** — the question is exactly depth order, which the tool gives directly, yet 29/29 never called
  `depth_estimator`. C′ did not make it call it; it just made it answer no a few more times in this category.

**Across all 350 `robospatial` samples, `depth_estimator` is called 0 times — 0 for both P4 and P7.**
(The SFT starting point is 1.) This structural gap recorded in P6 §6.4 **was not fixed by GRPO, and not by C′ either.**

---

## 6. Where exactly the RoboSpatial gap is: a causal chain that can be fully accounted for

`robospatial` is the only one of the nine with real policy freedom; P4 ↔ P7 differ by 14 questions. Broken down:

### 6.1 The whole gap is in Vacant; VQA is noise

```
           n     P4          P7 1st run     P7 2nd run
VQA       228   167 73.25%   164 71.93%    166 72.81%
Vacant    122    62 50.82%    51 41.80%     54 44.26%
```

Direction of divergent samples (P7 wrong both times vs P4 correct = lost; P7 correct both times vs P4 wrong = gained):

```
lost 29:   VQA 16 · Vacant 13
gained 17: VQA 14 · Vacant  3
net:       VQA −2 · Vacant −10
```

The answer directions of those 30 divergent VQA samples are **symmetric**: of the 16 lost, 12 are `gt=Yes` flipped from yes to
no, 4 are `gt=No` flipped from no to yes; of the 14 gained, 10 are `gt=Yes` flipped from no to yes,
4 are `gt=No` flipped from yes to no. **No systematic direction; it is just flip noise.**

### 6.2 Vacant's −10 questions can be accounted for completely

P6 recorded one thing: `robospatial` Vacant **is not pure pass-through, and whenever the model changes the point it gets worse**. Running this criterion
on both sides:

```
              pass-through                point override                fixed/broken   median orig-point dist -> after
P4       101 (accuracy 55.4%)   21 (accuracy 28.6%)   10 / 11     0.0716 -> 0.0483
P7        79 (accuracy 57.0%)   42 (accuracy 14.3%)   23 / 19     0.0942 -> 0.0855
```

**C′ doubled the samples that "modify the point roborefer gave" from 21 to 42, and the accuracy of point overrides is only 14.3%
— pass-through is 57.0%.**

Check (each side's per-category accuracy × each side's per-category sample count):

```
P4  101×0.554 + 21×0.286 = 56 + 6 = 62 = 50.82%   ✓
P7   79×0.570 + 42×0.143 = 45 + 6 = 51 = 41.80%   ✓
```

Counterfactual: **if P7 kept P4's pass-through/point-override ratio (101/21), computed with its own per-category accuracy,
it would be 101×0.570 + 21×0.143 = 57.6 + 3.0 ≈ 61 questions = 50.0% — right back at P4's level.**

> **So the entire RoboSpatial gap can be attributed, with nothing left over, to one behavioral change: C′ makes the policy more often
> "improve" the point the tool gave, and P6 already quantified this behavior as harmful.**
>
> This and the drop in pass-through rate in §5.2 (276/276 → 268/276) are the same thing showing up on two benchmarks:
> **C′ did make the policy "more proactive", but on this tool chain, proactive means worse** — because the task's information
> bottleneck is in the tools, and the policy has no information the tools did not give it.

---

## 7. Relation to P5's conclusions

The several definition conclusions in the P5 report all still hold on P7 and do not need to be argued again:

- **Of the ten Table 2 numbers only nine are independent** (Overall is the sample-weighted VQA/Vacant).
- **Decoding is greedy**; `rollout.n=5` is the group size used for training and does not apply to evaluation.
- **`gpu_memory_utilization` is a fraction of the whole GPU** — P5 §4.5 listed it as a "new deviation"; this time P7 back-computed it from
  GPU capacity as 0.545 to align the KV pool back to 24 GB, consistent definition.
- **blinkdepth should be reported with its stable upper bound plus an interval** (P4 three runs 86.29–87.90, upper bound 112/124 = 90.32).
  P7's single run 87.90 **is exactly the top end of P4's interval**, not "higher than P4".
- **The BOP-ASK Pose metric mapping is still unresolved** (the paper's 34.37 is not a proportion at n=60). The difference between P7's 55.73
  mean IoU and P4's 53.36 **cannot be taken as a capability difference**.

P5 §3.1 recorded "RoboSpatial VQA −6.13 pp is the only gap beyond noise" — P7 differs from P4 on VQA
by only 1–3 questions (see §6.1), **so P7 neither widened nor narrowed this gap; it is inherited.**

---

## 8. Conclusion

**0. None of the nine benchmarks is significant, in either direction.** The smallest McNemar exact-test p is
robospatial's 0.092 (0.104 after using P7's two runs to suppress noise). The three nominally a bit higher
(blinkdepth +2, boppose +1, bopgrasp +2) are all not significant, and `boppose`'s mean is opposite to its per-sample
direction, `bopgrasp`'s threshold count is opposite to its mean. **The statement "P7 is better than P4 on some benchmark"
currently has no evidence supporting it.**

**1. Of the nine benchmarks, only two can detect policy differences.** On 777 samples (refplacement /
refunseen / cvb3ddepth) the two checkpoints are identical per-sample; reflocation, cvb2drelation,
boppose, bopgrasp also have single-digit divergences. **The real measurement surface is robospatial (350) and
blinkdepth (124).** Future RL comparison experiments should allocate compute only to these two.

**2. The motivation P6 left for P7 — "a distribution-matching objective can avoid the tool-orchestration collapse caused by GRPO" —
is a negative reading in this measurement.** On the four benchmarks with large sample sizes C′ is consistently more concentrated, with fewer signature kinds.
But this reading is weakened by three things: cross-base, 85 steps, and 63–80% of groups reward-degenerate. **What it falsifies is
"C′ can achieve it at this amount of training", not "C′'s theoretical properties do not hold".**

**3. The only measurable behavioral change from C′ is making the policy modify tool output more often — and this change is harmful.**
pointing pass-through 276/276 → 268/276; Vacant point overrides 21 → 42, point-override accuracy 14.3% vs pass-through 57.0%.
RoboSpatial's 14-question gap can be attributed to this behavior with nothing left over.

**4. Of the three structural gaps P6 located, C′ moved not one.**
- `fit` questions (105, need free-space extent) accuracy 69.5% → 69.5%, bit-for-bit identical;
- `robospatial` has 0 `depth_estimator` calls throughout, 0 for both P4 and P7;
- rule compliance on depth questions 95.6% / 99.8% → 95.6% / 99.7%, bit-for-bit identical.

P6 judged all three as **tool-set gaps or should-have-called-but-didn't**, not problems reward shaping can solve.
**That changing the objective cannot move them is consistent with P6's expectation.**

**5. Error attribution is 100% clean on both sides.** Not a single wrong answer has an infrastructure excuse — this makes the causal explanations
of the four items above clean.

---

## 9. Limitations

1. **Cross-base, not a controlled A/B.** P4 is the paper's own base + GRPO, P7 is self-trained SFT + C′.
   Every statement "C′ caused X" can strictly only be stated as "this P7 checkpoint exhibits X".
2. **No dump for the SFT starting point.** The truly controlled SFT ↔ C′ per-sample comparison cannot be done; only summary numbers can be compared
   (robospatial main chain SFT 64.3% → C′ 64.9%, calls 1.649 → 1.646 — **these two numbers show
   the collapse was already like this at the SFT starting point, and C′ did not change it**; but these are summary quantities, not per-sample evidence).
   **Next time a checkpoint is evaluated, the dump must be archived too.**
3. **P4 uses the single run1.** P4 ran robospatial / blinkdepth / bopgrasp multiple times and reported intervals;
   the P4 column in this document is one end of the interval. P7 has two runs only for robospatial. The "lost/gained" counts from per-sample pairing
   therefore include single-run noise — §6.1 uses "P7 wrong both times / correct both times" to suppress this noise, and what remains on VQA
   is still symmetric flips.
4. **bopgrasp's "correct" is a threshold count of score ≥ 0.5**, opposite in direction to the mean score (P7 correct count 56 > P4 54,
   but mean score 1.8576 < 1.9354). The two readings of this benchmark contradict each other; **do not interpret it in isolation**.
5. **The attribution criteria reuse P6's implementation**, including the boundaries P6 itself recorded (attribution of 3b is disputed, the 31 manual ones
   have only one round of labeling, no inter-annotator agreement measure).

---

## Appendix: scripts and artifacts used in this document

```
analysis/compare_p4_p7.py            per-sample pairing: accuracy, chain signatures, tool calls, variables, error attribution
analysis/compare_p4_p7.txt           full output of the above
analysis/robospatial_divergence.py   direction and question type of robospatial divergent samples
analysis/p6_criteria_p4_vs_p7.py     P6 criteria A/B, VQA three-way question-type split, Vacant pass-through vs point override, collapse degree
analysis/significance_p4_p7.py       McNemar exact test + paired sign test for continuous scores
parsed/<benchmark>.jsonl             structured records of P7's nine benchmarks (same structure as p4/parsed)
parsed_runC/robospatial.jsonl        robospatial second run
```

`parsed/` is generated from `dumps/` by `spacetools-repro/tools/parse_dump.py --emit`;
fields are identical one-for-one with `p4/parsed/` (incl. `chain_signature`, `vars_exposed/used/unused/phantom`,
`trajectory`, `num_turns_verl_convention`), **needs no GPU, can be recomputed at any time.**
