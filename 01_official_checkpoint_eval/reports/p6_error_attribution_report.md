# SpaceTools reproduction: error attribution and headroom report

> **Scope**: where every error among the 2121 samples belongs, and "how much can the score rise at most by replacing/adding a given stage".
> Accuracy and deviation attribution belong to P5; see `records/P5_REPORT.md`.
>
> **Subject**: SpaceTools (CVPR 2026, arXiv:2512.04069), official checkpoint, no retraining.
> **Written** 2026-08-30 · process notes `records/P6_NOTES.md` · GPU experiments `records/P6_GPU_RESULTS.md`

---

## Summary

The seven accuracy-type benchmarks have 2001 samples in total, **322 wrong answers**; the two continuous-metric ones
(`boppose` / `bopgrasp`) have another 120 samples, handled separately.

**322/322 fully attributed**, of which 291 were settled automatically by three executable criteria (zero GPU, zero manual work),
and the last 31 were classified manually, looking at each image.

| Category | n | share of 322 |
|---|--:|--:|
| **1 tool error** (wrong localization / detection degeneration) | **241** | **74.8%** |
| **toolset gap** (`fit`: no tool gives free space) | **32** | 9.9% |
| 3 reasoning error | 19 | 5.9% |
| 3b coordinate frame / semantic mismatch | 14 | 4.3% |
| 2a should have called, didn't | 11 | 3.4% |
| 2D projection not separable · 2c wrong argument · 4 format error · 6 annotation | 2 / 1 / 1 / 1 | 1.6% |

**Tool error : reasoning error = 241 : 19 ≈ 12.7 : 1.** The paper's Appendix Table 16 gives
23 : 7 out of 30 cases on grasp; **our ratio leans further toward the tool side, and on ten times as many samples**.

> **Each of the top five rows of the table above has one real sample, see §3.4**; examples for the two continuous-metric benchmarks are in
> §4.1 and §4.2. Category labels are easy to read as abstract concepts; those sections lay out, item by item, "what the tool actually returned,
> what the model actually did, what the GT is".

Three supporting structural findings:

1. **100% of the 322 wrong answers are clean.** OOM, response truncation, turns exhausted, missing `<answer>`,
   malformed tool call — 0 hits among the wrong answers for all of them. **Not a single wrong answer has an infrastructure excuse.**
2. **On most benchmarks, "reasoning" is a rule that can be written as code, and the model executes it verbatim.**
   On pointing, 276/276 pure pass-through; on `boppose`, 60/60; on depth questions, 95.6% / 99.8% follow the rule.
3. **The only exception is `robospatial`**, and it happens to be the only benchmark with a real gap.

**The direction of headroom was rewritten once by measurement**: offline attribution said "RoboRefer alone accounts for 53.4% of wrong answers,
replace it first"; measurement says **replacing it is worse (−14.61 pp), but per-sample complementarity gives +10.83 pp** —
the action is **add**, not replace.

---

## 1. Method: criteria before conclusions

The credibility of this attribution does not come from the conclusions; it comes from **every criterion passing two checks before it was used**.

### 1.1 Core insight

**On most benchmarks of this system, "reasoning" is a rule that can be written as code.**
If the model follows that rule verbatim, its reasoning is right, and the error can only be in the numbers the tool gave.
This turns "attribution" from reading comprehension into arithmetic.

### 1.2 Two checks, both required

| Check | What it asks | What happens if it fails |
|---|---|---|
| **Self-consistency rate** | On samples the model **got right**, does the rule's answer equal the model's answer? | The model is not using this rule; the criterion is invalid |
| **Rule vs GT** | Is the rule itself correct? | The model may be **faithfully executing a wrong rule**; then "followed but wrong" is not a tool error |

The second was only added in this round, and it immediately caught a case that would have led to a wrong conclusion (§2.3).

> **A lesson written into the process: an automatic criterion that is not accurate enough is worse than none.**
> The first version of the `cvb2drelation` criterion had a self-consistency rate of only 91.8%, while that benchmark has only 35 wrong answers in total —
> not enough signal-to-noise; taking its 8.2% directly as the reasoning-error rate would have built the conclusion on our own parsing bug.
> **The right decision at the time was to fall back to manual.** Later it turned out it was not "impossible to turn into a criterion", just that the criterion was written wrong:
> after fixing it, the self-consistency rate was **99.83%**. **Inaccurate criterion ≠ cannot be turned into a criterion; the difference is whether you go back and fix it.**

---

## 2. Three automatic criteria

### 2.1 Criterion A — depth questions: does it always pick the one with "smaller measured depth"

`blinkdepth` and `cvb3ddepth` go through the same chain
(`depth_estimator×1 + roborefer×2 + vision_ops×2`); the question asks which is closer.

| benchmark | decidable | follows rule | follows but wrong → tool error | violates and wrong → reasoning error |
|---|--:|--:|--:|--:|
| `blinkdepth` | 114/124 | 109 (95.6%) | **14** | **1** |
| `cvb3ddepth` | 599/600 | 598 (**99.8%**) | **21** | **0** |

**100% of the 21 wrong answers in `cvb3ddepth` are DepthPro's errors; zero errors on the reasoning side.**

The two benchmarks share the chain and the rule, with reasoning fidelity close to 100% on both, yet the scores differ by ten points —
**those ten points are purely DepthPro's accuracy difference on different images**, unrelated to orchestration or the policy.
**This pair of benchmarks is therefore a natural control group.**

### 2.2 Criterion B — pointing questions: is the answer a pure pass-through of the tool output

| benchmark | decidable | pure pass-through | modified by model | pass-through and wrong → tool error |
|---|--:|--:|--:|--:|
| `reflocation` | 99/100 | **99 (100%)** | 0 | 45 |
| `refplacement` | 100/100 | **100 (100%)** | 0 | 42 |
| `refunseen` | 77/77 | **77 (100%)** | 0 | 40 |

**276 of 276 samples are pure pass-through, not a single modification.**

> **A sharp corollary.** On these 277 samples, the policy's entire contribution relative to "calling RoboRefer directly"
> is **how it phrases that `obj_name`** — it does no post-processing at all.
> In the paper's Table 2, RoboRefer-8B-SFT alone scores 48.37 on RefSpatial, SpaceTools-3B scores 53.07.
> **That +4.7 comes entirely from query construction, not a single point from reasoning.**

### 2.3 Criterion C — relation questions: the order of two points in the image plane

The same rule (compare x or compare y), applied to two benchmarks, **gives two different conclusions**:

| | decidable | self-consistency rate | **rule vs GT** | model vs GT |
|---|--:|--:|--:|--:|
| `cvb2drelation` | 627/650 | **99.83%** | **95.4%** | 94.7% |
| `robospatial` VQA relation questions | 87/87 | **98.55%** | **79.3%** | 79.3% |

**Self-consistency** is high on both — the model is executing this rule in both places.
But **the rule's accuracy against GT differs by 16 points**, and that decides how "followed the rule but wrong" is judged:

- **`cvb2drelation`: the image plane is its correct semantics** (CVBench 2D relation is defined
  in the image plane to begin with). So "followed but wrong" = detection point inaccurate = **tool error**.
  The gap distribution supports this: correct group median 0.410, followed-but-wrong group 0.192.
- **`robospatial` VQA: the image plane is not its semantics.** "Followed but wrong" **cannot** be judged a tool error;
  the images must be inspected. §3 shows the vast majority are coordinate-frame/semantic problems.

> **A very hard conclusion.** On `robospatial` relation questions:
> rule vs GT **69/87**, model vs GT **69/87**, 68/69 are the same samples.
> **Not close, identical — on this question type the model is just a single two-point coordinate comparison,
> and its ceiling is that rule's ceiling, 79.3%.**

---

## 3. Manual classification: the last 31

The 31 items that still needed image inspection after the automatic criteria were reviewed one by one (original image + the **points the tool actually returned** + question /
GT / model answer / criterion / gap). Per-item verdicts and evidence are in `p6/manual/verdicts.jsonl`;
review cards are in `p6/manual/cards/`.

### 3.1 The biggest chunk: "above/below" in `robospatial` is not image up/down

Of the 21 `robospatial` items, **14 are 3b (coordinate frame/semantics)**, in two subtypes.

**① Image y ≠ 3D above (11 items).** Both detection points are **correct**, the model faithfully compared image y,
but the GT means "above" in the 3D sense:

    #283  lamp / bed        floor lamp stands **beside** the bed; higher in the image only because it is farther     GT=no
    #294  bag / trash bin   bag is on the floor **next to** the trash bin                                            GT=no
    #316  boards / fridge   cutting boards on the counter right of the stove, fridge on the left; side by side, not above/below   GT=no
    #329  microwave/bottle  side by side on the same counter                                                         GT=no

**Image y encodes both "higher" and "farther", and the tool only returns 2D points; this ambiguity cannot be resolved from the data.**

**② Camera frame ≠ the benchmark's reference frame (3 items).** Systematic reversal on left/right questions:

    #347  toilet / shelf      camera frame: toilet on the left (0.215 vs 0.801)      GT=yes (right)  gap 0.586
    #295  microwave / fridge  camera frame: microwave on the left (0.331 vs 0.898)   GT=no           gap 0.567
    #310  mouse / keyboard    camera frame: mouse on the right (0.911 vs 0.511)      GT=yes (left)   gap 0.400

**A gap of 0.40–0.59 cannot be detection error**, and all three are opposite to the camera frame.
**This explains the 79.3% vs 95.4% in §2.3.**

### 3.2 The other 17

| Category | n | Example |
|---|--:|---|
| 1a tool error | 9 | `#273`/`#274`: same image, same `'bottle'`→hand soap misdetection; **one bad detection causes two wrong answers** |
| 2a should have called, didn't | 3 | `#246`/`#272`/`#301` only detected the subject; answered **without ever getting the reference object's position** |
| 2D projection not separable | 2 | The two points differ by 0.004–0.006 on the decision axis; the 2D projection lacks the information needed to decide |
| 2c wrong argument | 1 | Query string written as `'fluentice tube'` (misspelled fluorescent); the tool still returns a point, the error propagates silently |
| 3 reasoning error | 1 | Got a distinguishable `'building in red box'` but answered using another point |
| 4 format error | 1 | See §3.3 |
| 6 annotation/reference ambiguity | 1 | Both points correct, the relation in the image is clear, GT is the opposite |

### 3.3 Two actionable findings

#### `--strict` misses "answer not in the option set"

`blinkdepth` #69 detected three children, probed three depths, then answered **`C`** — while the options are only `(A)/(B)`.
`parse_ok` is true (`<answer>` tag intact, parseable), so **`--strict` did not catch it**,
and "format errors = 0" was under-reported as a result.

    full scan of 1374 questions with options: blinkdepth 1/124 · cvb2drelation 0/650 · cvb3ddepth 0/600

**Fixed**: `parse_dump.py` adds `answer_off_options`; when two or more `(A)(B)…` can be parsed from the question,
it checks the answer's first character; free-answer benchmarks return `None` and are not judged.
**Made it a warning, not a contamination flag** — "answering outside the options" is the model's error, not a broken pipeline,
and changing the contamination criteria would retroactively change the status of runs that already passed.
Measured: full run of all nine benchmarks of P4 run1, `--strict` still `exit=0`.

#### A recurring named tool failure: RoboRefer detection degeneration

**Returns the same point for different queries.** Occurs 5 times across three benchmarks:

    cvb2d #180   'brand name' and 'sign' same point; 3 more phrasings, still the same point
    cvb2d #621   **six different queries all return (0.558, 0.594)**
    cvb2d #234   'building' and 'tall skyscraper building' same point
    blink #86    'red dot under label A' and 'label B' same point, depth readings bit-for-bit identical
    robo  #278   'table' and 'desk' same point (the same nightstand)

**What is notable is how the model responds: it actively tries 3–6 phrasings to work around it.**
This and criterion B's conclusion corroborate each other — phrasing is its only lever, and this failure mode is exactly the one phrasing cannot break through.

### 3.4 Sample instances: one for each of the top five categories

The category table is easy to read as abstract labels. Below, each category gets one real sample, laying out **what the tool actually returned**,
**what the model actually did**, and **what the GT is**. All numbers can be recomputed directly from `p4/parsed/`.

#### ① 1 tool error (241 items, 74.8%) — `cvb3ddepth #86`

> *Which object is closer to the camera, the lamp (red box) or the refrigerator (blue box)?*

    detect_one('lamp (highlighted by a red box)')          -> (0.368, 0.144)   ✓ point is correct
    detect_one('refrigerator (highlighted by a blue box)') -> (0.600, 0.469)   ✓ point is correct
    index_at($depth_map, 0.368, 0.144)  ->  **4.536 m**    (lamp)
    index_at($depth_map, 0.600, 0.469)  ->  **2.378 m**    (refrigerator)

    rule "pick the smaller" -> B       model answers B       GT = A

**All the model did was compare 4.536 and 2.378, and it compared them correctly.**
**Nothing to fault on the reasoning side; the blame lies entirely with the tool.**

> **Both points really are correct — this one was verified by looking at the rendered review card**, not inferred from "the detection point falls inside the box
> annotated in the question". The red box encloses a **ceiling lamp** at the top of the frame, and `(0.368, 0.144)` is inside the box;
> the blue box encloses the fridge in the mid-ground, and `(0.600, 0.469)` is also inside the box.

The lamp is at the top of the frame and projects small, but it is a **ceiling lamp directly overhead**, physically very close to the camera.
GT says it is closer, meaning the actual depth is < 2.378 m, while DepthPro reported 4.536 m —
**a relative gap of 90.7%: not slightly off, but off by nearly 2× and in the wrong direction**.
Monocular depth reading "projects small" as "far" on texture-poor ceiling fixtures is a known failure mode.

#### But the "tool error" label is broader than "the depth model is incompetent"

Three mechanisms all land in this category, **and should not all be read as the model estimating badly**:

1. The depth model really did estimate wrong (`#86`)
2. The point is on the object, but the depth at that pixel belongs to something else (thin objects, edge bleed-through)
3. The two objects are nearly equally far to begin with; **there is no margin at all**

Splitting the 36 decidable depth wrong answers by relative gap `|dA−dB| / min`:

| Relative gap | Count | Reading |
|---|--:|---|
| < 10% (near-tie) | **11** | More like "no margin" than "estimated wrong" |
| 10 – 50% | 17 | |
| > 50% (confidently reversed) | **8** | `#86` is here |

The comparison is more convincing: **median relative gap is 83.4% for correct samples and 23.1% for wrong ones.**
**Most depth wrong answers occur where the two objects are close to begin with.**
This is exactly the basis for how §6.3 bounds "swapping the depth tool" — a slightly stronger tool can flip back the near-tie part;
eating the confidently-wrong part needs a significantly stronger model.

#### ② Toolset gap (32 items, 9.9%) — `robospatial #128`

> *Can the cup fit in front of the speaker?*  GT = **No**, model answers **Yes**

    chain roborefer×2@2t — this is the **entire** tool output for this sample:
      detect_one('cup')     -> (0.780, 0.494)
      detect_one('speaker') -> (0.154, 0.491)

Two **point locations**. **No tool ever told it how much free space there is in front of the speaker, or how big the cup is.**
The question is "does it fit", what it got is "where is it" — this is not the model failing to think it through;
**no tool returns the quantity it needs at all**.

> This is exactly the difference from ⑤: those 8 `front/behind` items **had a tool but didn't call it**;
> these 105 `fit` questions **have no tool to call**. The former is the model's error, the latter the toolset's.

#### ③ 3 reasoning error (19 items, 5.9%) — `robospatial` Vacant `#11`

**Full question:**

> *In the image, there is a cup. Pinpoint one point within the **vacant space** suitable
> for placing an object **close to and to the right of** the cup. Your answer should be
> formatted as a list of a tuple, i.e. `[(x1, y1)]` … The coordinates should be between
> 0 and 1, indicating the normalized pixel locations.*

**The whole sample has only one tool call** (chain `roborefer×1@2t`):

    roborefer.detect_one('cup')  ->  (0.271, 0.870)      ✓ lands precisely on the blue mug on the table

**The GT is a region made of 23 acceptable points** — the narrow strip of tabletop to the right of the cup:

    [(0.334, 0.886), (0.329, 0.982), (0.418, 0.987), (0.530, 0.989), (0.551, 0.988),
     (0.558, 0.878), (0.423, 0.880), (0.373, 0.878), (0.487, 0.878), (0.488, 0.919), …]

    x range 0.327 – 0.558        y range 0.878 – 0.989

**Model answer: `[(0.50, 0.85)]`**

**It did not copy the tool's point; it did it by hand.** And look closely at how it went wrong:
**x = 0.50 falls within the GT range (0.327–0.558), direction entirely correct; it is y that is off** —
all GT y values are ≥ 0.878, the model gave 0.85, **about 0.03 too high, just outside the top edge of that narrow strip**, a near miss.
**It knew to go right; it didn't get how far forward.**

This is one instance of the statistic on Vacant: **101 pure pass-through items at 55.4% accuracy,
21 items where the model overrode the point itself at 28.6%; overriding halves it**. 15 of the 19 reasoning errors come from here.

#### And "overriding halves it" has a testable mechanism: it is eyeballing

Note the answer is `(0.50, 0.85)` — both coordinates are round numbers. Checking all of Vacant:

| | n | accuracy | both coordinates multiples of 0.05 |
|---|--:|--:|--:|
| pure pass-through | 104 | 54.8% | **2 items (2%)** |
| model overrides point itself | 18 | 27.8% | **6 items (33%)** |

**A 16× difference.** When passing through, the coordinates carry the tool's three decimals (`0.271, 0.870`);
once the model does it itself, they become `(0.40, 0.70)`, `(0.30, 0.40)`, `(0.50, 0.85)` —
**eyeballed round numbers**.

> **So the concrete meaning of "reasoning error" on Vacant is: it isn't computing, it's eyeballing.**
> And Vacant's GT is a narrow strip a few dozen pixels wide; eyeballing at 0.05 granularity is not fine enough.
> Recompute: `do_vacant()` in `tools/p6/p6_relations.py`.

#### ④ 3b coordinate frame/semantics (14 items, 4.3%) — `robospatial #283`

> *Is the lamp above the bed?*  GT = **No**, model answers **Yes**

    detect_one('lamp') -> (0.561, 0.234)   ✓ lands on the floor lamp
    detect_one('bed')  -> (0.703, 0.775)   ✓ lands on the bed
    image y: 0.234 < 0.775  ⇒ "lamp is above bed"

**Both points are correct and the rule was executed fine, but that rule is not this question's semantics.**
Looking at the image makes it clear: the floor lamp **stands on the floor beside the bed**, not above the bed; it is higher in the image
**only because it is farther from the camera**.

**The root is: the single number image y encodes both "higher" and "farther"**, and the tool only returns 2D points
— this ambiguity cannot be resolved from the data the model gets. It is neither a tool error (the points are correct)
nor an ordinary reasoning error (the rule was not violated), so it gets its own category.

#### ⑤ 2a should have called, didn't (11 items, 3.4%) — `robospatial #246`

> *Is the paper towel above the counter?*  GT = **Yes**, model answers **No**

    this sample has **only one tool call in total**:
      detect_one('paper towel') -> (0.293, 0.481)
    then answers directly.

**The position of the reference object `counter` was never obtained.** Comparing the above/below relation of two objects
with the coordinates of only one of them — the other half is a guess. `#272` (bottle / water) and
`#301` (game controller / sofa) have the same shape.

Another subtype is the 8 `front/behind` items: 29 questions ask about depth order,
and **`depth_estimator.estimate_depth` is right there in the tool list, yet it was not called once in 29**.

---

## 4. The two continuous-metric benchmarks

`boppose` / `bopgrasp` have no "correct/wrong", so criteria A and C do not apply.
**But criterion B does, and more cleanly** — both tools put the final answer directly in 2D form in the returned text.

### 4.1 `boppose`: 60/60 pass-through, zero involvement from the reasoning side

    bit-for-bit same order            56/60
    same set (reordering allowed)     **60/60**       reorder-only samples #0 #10 #25 #35

**The model never modifies the corner values.** Those 4 are just reorderings — and the metric is convex-hull IoU, insensitive to order,
so reordering doesn't change the score, and it also shows that **it is trying to satisfy the question's "bottom face first,
counter-clockwise" requirement and failing** (P5: samples where the optimal matching equals the model's given order, 0/60).

**So all of `boppose`'s error is `bounding_box`'s error.**

Zero scores are not random either; they follow **point-cloud fitting degeneration**:

| | n | OBB shortest/longest edge (median) | ratio < 0.20 |
|---|--:|--:|--:|
| zero score | 9 | **0.161** | **6/9** |
| nonzero | 51 | 0.306 | 12/51 |

Fisher exact **p = 0.0164**. **When the box is fitted flat and thin, the projected convex hull is useless.**

#### Sample instance: `#0` (zero score) vs `#28` (0.952)

> *What are the eight normalized (x, y) image coordinates of the cuboid corners for the rightmost instance of the cylindrical can of sliced mushrooms?*

    #0   IoU = 0.000
      compute_bbox: Extent [0.016, 0.017, **0.001**]  shortest/longest = 0.059  uses **524** points
      tool 2D corners  [(0.786,0.803), (0.797,0.793), (0.777,0.781), …]
      model answer     [(0.786,0.803), (0.797,0.793), (0.777,0.781), …]
      bit-for-bit same order ✗   same set ✓      <- the model swapped two of the points

    #28  IoU = 0.952
      compute_bbox: Extent [0.214, 0.158, 0.046]      shortest/longest = 0.215  uses **36801** points
      bit-for-bit same order ✓

**The "cuboid" of `#0` is 1.6 cm × 1.7 cm × 1 mm — a sheet of paper.** Only 524 points went into the fit,
the OBB collapsed into a plane, and its projection is a thin sliver with zero overlap with the GT convex hull.

And what the model did on these two samples is **exactly the same**: copy down the eight points the tool gave.
In `#0` it did one more thing — **swapped two points**, probably trying to satisfy the question's
"bottom face first, counter-clockwise" ordering requirement.
**That reorder had no effect on the score at all** (convex-hull IoU is insensitive to order),
but it is the only trace of the model "actively doing something", and doing it didn't help.

### 4.2 `bopgrasp`: tool fails 40/60, the model has a fully deterministic fallback

    20 items where compute_grasp gives 5 points -> 19 bit-for-bit pass-through
    40 items taking the failure exit -> model puts the grasp center on the roborefer detection point **40/40 (±0.02)**

**When the tool fails, the model does not abstain; it executes a deterministic rule: grasp the object's center.**

#### Sample instance: `#0` (tool succeeds) vs `#1` (tool fails → fallback)

    #0  tool succeeds
      detect_one('carton of orange juice')  -> (0.381, 0.428)
      compute_grasp -> "Projected 2D gripper points: [(0.375, 0.190), …]"
      model answer center (0.375, 0.190)      <- bit-for-bit pass-through of compute_grasp
      GT center           (0.358, 0.257)       center distance 0.069
      gripper axis vs GT angle  **17.1°**      NCE = **1.31**

    #1  tool fails
      detect_one(...)  -> (0.730, 0.594)
      compute_grasp -> "Error: RuntimeError: No collision-free grasps found.
                        All grasps collide with other objects."
      model answer center (0.730, 0.594)      <- **0.0000 from the detection point**
      GT center           (0.730, 0.518)       center distance 0.076
      gripper axis vs GT angle  **54.9°**      NCE = **1.23**

**This pair compresses the whole contradiction into two samples.** In `#1` the tool explicitly reports "all grasps collide",
so the model puts the gripper center unchanged on the object detection point — **displacement 0.0000, not approximate, copied**.
Its orientation is off by **54.9°** (nearly perpendicular to the correct grasp direction; it would not pick anything up in reality),
**yet NCE gives it 1.23, lower than `#0` (1.31), where the tool actually computed a grasp.**

All 40 failure samples have this shape, `40/40`. Whole-group statistics: the fallback group's median center distance 0.092
is **closer** than the tool-success group's 0.116, while the median angle 63.9° vs 18.3° is **much worse**.

#### A contradiction that must be recorded: NCE and SR give opposite rankings

| | tool succeeds (n=20) | fallback (n=40) |
|---|--:|--:|
| distance from grasp center to GT | 0.116 | **0.092** |
| angle between gripper axis and GT (median) | **18.3°** | 63.9° |
| angle < 30° | **16/20** | 7/40 |
| NCE (lower is better) | 1.38 | **1.07** |
| SR (already computed in P4) | **90%** | ≈40% |

**The fallback wins on position and loses on orientation**, which is exactly why the two metrics conflict:
**NCE is driven mainly by position, SR by orientation.**

> **One reading is usable, one is not.**
> It **cannot** be read as "the fallback is better than the tool" — the two groups are different scenes; which scenes the tool fails on is not random,
> so this is a confounded comparison.
> What it **can** be read as: **this RL reward (NCE) is insensitive to gripper orientation.**
> An answer whose orientation is off by a median of 64° scores lower on NCE than a real grasp generated by the tool.
> This is an observation about reward design, **and it is directly relevant to P7: before swapping the objective function,
> first confirm what the current objective function is rewarding.**

> **Trap: the `correct` field in `p4/parsed/bopgrasp.jsonl` is meaningless.**
> It is the generic `score >= 0.5` criterion, while bopgrasp's `score` is NCE, where lower is better.

---

## 5. A fourth error category: consistency

The sampling experiment (`n=5, T=1.0`) shows **there is one more category not in the original six**: the model itself cannot give a stable answer to the same question.
The key cut is splitting once more by "are the five `<tool_call>`s verbatim identical" —
**identical calls ⇒ all five faced the same evidence, so a flip can only happen at the decision layer**:

| | n | split | of which tool calls **verbatim identical** | calls differ |
|---|--:|--:|--:|--:|
| RoboSpatial VQA | 228 | 100 (43.9%) | **46** | 54 |
| RoboSpatial Vacant | 122 | 51 (41.8%) | **0** | 51 |
| `blinkdepth` | 124 | 30 (24.2%) | **3** | 27 |

- **The 46 in VQA are pure decision flips**: same image, same `roborefer` returns, opposite yes/no.
  Example (#122, GT=Yes, five runs `1,0,0,0,0`): the first run decides there is a gap between the chair back and the wall → Yes,
  the second decides it is occupied by the bookshelf → No. **The tool answers "where is it", the question asks "does it fit",
  and no tool output can pin down the step in between.**
- **Vacant has none**: all 51 flips come with a different `obj_name` — that is not "casting five votes",
  it is "asking five different questions".
- **`blinkdepth` has only 3**, which **confirms criterion A from the other side**: given the same two depth readings,
  "pick the smaller" is almost never violated.

### Don't count it as a gain yet

Majority vote@5 gets 176/228 on VQA, higher than all four greedy runs (161–169). **But the vote was only run
once, a single value.** Doing a per-sample paired test (McNemar) instead:

    vs run1 (167) net +9  p=0.163      vs run2 (168) net +8  p=0.230
    vs run3 (161) net +15 p=0.014 *    vs run8 (169) net +7  p=0.281

**Only significant against the lowest run**, and the four comparisons share the same vote arm.
~~**"Self-consistency can close about 60% of the gap" has been downgraded to "same direction, not significant".**~~

> **⚠ 2026-09-02 · Downgraded once more: majority vote@5 has no effect.**
> After a second independent set of 5 samples (`p6/passk2/`) was run, the conclusion of this whole section above changed.
> **Set 1's vote arm is 176/228, set 2's vote arm is 166/228 — the vote arm's own spread is 10 samples,
> the same order of magnitude as the spread of the four greedy runs (161/167/168/169, range 8).**
> Redoing the paired test against the same four greedy runs:
>
>     set 2 · VQA        net −1 / −2 / +5 / −3     p = 1.000 / 0.868 / 0.500 / 0.720
>     set 2 · Vacant     net −2 / −3 / −1 / −2     p all ≥ 0.607
>     set 2 · blinkdepth net +2 / +3 / +1 / +0     p all ≥ 0.453
>
> **0/12 comparisons significant, and the VQA direction flipped sign between the two sets.**
> Set 1's `p=0.014` is exactly the kind of artifact this project's §3.3 warning no. 3 is about:
> **the four comparisons share the same vote arm, and that arm had no spread at the time.** Now it has one, and the effect is gone.
>
> **Correct statement: majority vote@5 has no effect on these three benchmarks.**
> The −6.46 pp gap on `robospatial` VQA therefore **has one fewer candidate explanation and remains entirely unexplained**.
> Basis: `records/P7_GPU_RESULTS.md` §3.4.
>
> **The phenomenon is still there, the gain is not**: pure decision flips in VQA are 46 / 62 in the two sets,
> Vacant is 0 in both (0/51, 0/47), `blinkdepth` is single digits in both (3 / 4).
> The model really does waver on the same evidence, but **the wavering is symmetric, and voting cannot recover it**.

Two other boundaries: **pass@k is unusable on binary tasks** (pure random `pass@5` for A/B and yes/no = 96.88%;
measured blinkdepth 94.35%, VQA 89.04%, **both below blind guessing**);
**majority vote cannot go into Table 2** — the released repo's `val_kwargs` default is greedy, which is the only definition with grounding.

---

## 6. Headroom

### 6.1 Responsibility share of individual tools

| Tool | wrong answers directly caused | share of 322 |
|---|--:|--:|
| **RoboRefer** | 127 (three RefSpatial benchmarks) + 45 (Vacant pass-through) = **172** | **53.4%** |
| **DepthPro** | 21 (`cvb3ddepth`) + 14 (`blinkdepth`) = **35** | 10.9% |
| `bounding_box` | all of `boppose`'s error (counted separately, not in the 322) | — |
| `grasp_generator` | 40/60 direct failures (counted separately) | — |

### 6.2 Measured: swapping the pointing tool — "add", not "replace"

On the 397 probes, `roborefer` replaced with Molmo-7B-D, rest of the flow unchanged:

| | n | RoboRefer | Molmo | diff |
|---|--:|--:|--:|--:|
| total | 397 | 211 (53.15%) | 152 (38.29%) | **−14.61 pp** |

**Replacing it wholesale is significantly worse** (McNemar p=1.05e-06). But the two tools are highly complementary:

    both correct 109 · only RoboRefer correct 102 · only Molmo correct 43 · both wrong 143
    per-sample oracle tool choice 254/397 = 63.98%   vs RoboRefer **+10.83 pp**

And this is not a "two draws beat one" artifact: running RoboRefer twice in the same environment, the 397 items have **scores and returned points
bit-for-bit identical**; the self-union adds not a single sample.

> **So the "tool errors are 53.4%" headroom cannot be eaten by replacing a single tool.**
> Either build a router / ensemble, or bring in a significantly stronger third tool.
> **The oracle upper bound is not an achievable gain.**

### 6.3 Bounding: swapping the depth tool

Splitting the 42 depth wrong answers by relative gap `|dA−dB| / min`: **17 are near-ties** (a slightly stronger tool can flip them),
**25 confidently gave the reversed order** (need a significantly stronger model).
A "slightly better" depth model recovers 17 at most. **Pointing still comes first in priority.**

### 6.4 Decomposition of the `robospatial` VQA gap

    this run VQA 167/228 = 73.2%      paper 181/228 = 79.38%      gap 14 samples

| Question type | n | accuracy | depth not called | GT=`no` accuracy | gain if made 100% |
|---|--:|--:|--:|--:|--:|
| relation questions (2D decidable) | 87 | 79.3% | — | 18/31 = 58% | +18 → 81.1% |
| `front/behind` (needs depth order) | 29 | 72.4% | **29/29** | 7/11 = 64% | +8 → 76.8% |
| **`fit` (needs free-space extent)** | 105 | 69.5% | **105/105** | **4/18 = 22%** | **+32 → 87.3%** |
| detection mismatch | 7 | 57.1% | — | — | +3 → 74.6% |

**Of all 350 `robospatial` samples, 0 ever called `depth_estimator`** —
and it is right there in that benchmark's tool list. Among the 228 VQA questions, **134** have answers that depend on depth or free space,
and the model answered all of them using only two 2D points.

The two kinds of gap are different in nature: **`front/behind` is clean 2a (should have called, didn't)** — it asks exactly for depth order, and a tool gives it directly;
**`fit` is a toolset gap** — no single tool returns free-space extent.

> **The `no`-class collapse now has a location.** P5 recorded that "questions with GT `no` sit at chance level";
> broken down, **it is concentrated in fit questions**: 22% vs 58% for relation questions and 64% for front/behind.
> **On "does it fit" the model almost always answers yes** (83 yes answers on 105 questions).
> **As long as fit questions stop being answered yes across the board, the gap is filled** — and this path points to **adding tools or adding compositional ability**,
> not tuning the prompt, not swapping the pointing tool.

---

## 7. Four points for P7

1. **The current policy's tool orchestration has already collapsed heavily**: on the three RefSpatial benchmarks 277/277 use exactly the same chain,
   and in `cvb2drelation` 615 of 650 are the same. This is exactly what GRPO would do, and exactly what a distribution-matching objective
   claims to avoid. **P7's motivation goes from "should in theory" to "already observed".**
2. **But the prerequisite measurement of "headroom on the reasoning side" is currently empty.** pass@k is unusable on binary tasks,
   and the substitute, majority vote@5, did not reach significance. ~~If P7's motivation depends on it, **a second set of samples is needed first**.~~
   **⚠ 2026-09-02: the second set of samples is done (`p6/passk2/`); the conclusion is that majority vote@5 has **no effect**
   (0/12 significant, VQA direction flipped sign). **So this prerequisite measurement is not "empty", it is "measured, result zero"** —
   P7's motivation **cannot** be built on reasoning-side headroom. See the update block in §5 and `P7_GPU_RESULTS.md` §3.4.**
3. **Before swapping the objective function, first confirm what the current objective function is rewarding.** §4.2: NCE is insensitive to gripper orientation;
   a fallback answer with orientation off by 64° scores lower than a real grasp.
4. **Training with the paper's configuration on a four-GPU machine is not realistic** (resident tools take 2 GPUs, training at most 2, throughput short by about 8×
   → 64–96 hours). Ways out: add GPUs, fall back to LoRA, or only do algorithm-side validation. Details in the P5 report and
   `P6_GPU_RESULTS.md` §7.

---

## 8. Limitations

- **Classification had only one round of annotation; κ cannot be computed.** Those 31 items are single-annotated with no second annotator,
  **so there is no annotation agreement measure**. The mitigation is publishing per-item evidence in `p6/manual/verdicts.jsonl`
  — each item states "on what grounds" — so others can audit the conclusions without having to trust the annotator.
- **The assignment of 3b is debatable.** In the taxonomy 3b belongs to reasoning errors; counted that way,
  tool error : reasoning error = 241 : 33 ≈ 7.3 : 1 instead of 12.7 : 1. **Both readings are reported** —
  the model reasons in the image plane because the tool only gives it two 2D points;
  which side gets the blame depends on whether you think the policy should fill in depth on its own.
- **One bad detection can contaminate multiple samples** (`#273`/`#274`, same image, same misdetection).
  Counting by sample counts the same tool defect more than once, so the "tool responsibility share" is counted **by wrong answer**, not
  **by defect**.
- **The two-group comparison in `bopgrasp` is confounded** (which scenes the tool fails on is not random); see §4.2.
- **This report does not cover accuracy and deviation attribution**; that belongs to P5.

---

## Appendix: scripts and data

| Script | What it does |
|---|---|
| `tools/p6/p6_inventory.py` | Error inventory; clean vs has an infrastructure cause |
| `tools/p6/p6_split{,2,3}.py` | Criterion A / criterion B / question type and relation split |
| `tools/p6/p6_relations.py` | **Criterion C**, with both the self-consistency and semantics checks built in |
| `tools/p6/p6_continuous.py` | Pass-through criterion and position-orientation decomposition for `boppose` / `bopgrasp` |
| `tools/p6/p6_consistency.py` | Fourth error category (consistency), with paired test built in |
| `tools/p6/make_cards.py` · `verdicts.py` | Review cards and per-item verdicts for manual classification |
| `tools/p6/gpu_{pointing,depth}_swap.py` | Tool swap, **needs no policy, needs no sglang** |

| Data | Path |
|---|---|
| Per-sample records (source of all statistics) | `p4/parsed/` |
| Manual classification: pending / verdicts / review cards | `p6/manual/` |
| Tool-swap probes and outputs | `p6/probes/` · `p6/swap/` |
| Sampling experiments | `p6/passk/` |

**Every number can be recomputed directly from the repo copies**, with no GPU and no rerun of the evaluation
— except the review cards, which need the `eval-benchmarks` parquet.
