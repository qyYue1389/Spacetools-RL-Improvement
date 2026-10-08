# P6 error attribution (full version)

> **Scope**: the **2121 samples** of the SpaceTools official checkpoint in the P4 full evaluation.
> The seven accuracy-type benchmarks have **2001 samples / 322 wrong answers** in total, each sample assigned to exactly one class;
> the continuously scored `boppose` / `bopgrasp` add another 120 samples, handled separately.
>
> **This document does not cover P7 / GFlowRL.** The subject is the checkpoint released with the paper; nothing is retrained.
>
> **Base draft**: `01_official_checkpoint_eval/reports/p6_error_attribution_report.md`. The differences in this document: **every wrong answer is explicitly assigned to one class**
> (the cross-tab can be recomputed), **each class gets two real samples**, and all numbers were rerun on the spot from `p4/parsed/`.
> The recomputation scripts are in the Appendix.

---

## 0. First, three things that affect how to read this table

**① 100% of the 322 wrong answers are clean.** OOM, truncated tool response, turns exhausted, missing `<answer>`,
malformed tool call, phantom variable — the hit count on wrong answers is **0 for all of them**:

```
benchmark          n  correct  wrong  accuracy | toolfail OOM trunc maxturn no-ans phantom || clean wrong
blinkdepth       124     107     17   86.29%  |     0    0   0    0    0    0  ||   17
cvb2drelation    650     615     35   94.62%  |     0    0   0    0    0    0  ||   35
cvb3ddepth       600     579     21   96.50%  |     0    0   0    0    0    0  ||   21
reflocation      100      54     46   54.00%  |     0    0   0    0    0    0  ||   46
refplacement     100      58     42   58.00%  |     0    0   0    0    0    0  ||   42
refunseen         77      37     40   48.05%  |     0    0   0    0    0    0  ||   40
robospatial      350     229    121   65.43%  |     0    0   0    0    0    0  ||  121
total           2001            322                                            ||  322
```

**Not a single wrong answer has an infrastructure excuse.** This determines the nature of the whole table below: it attributes capability, not accidents.

**② The label "tool error" is broader than "the model is incompetent".** It covers three mechanisms and should not all be read as "the depth/detection model estimates badly":
a genuinely wrong estimate, a point that lands on the object but on a pixel that belongs to something else, and **two objects that simply have almost no margin to begin with**.
§2 splits it apart and measures it.

**③ Each class gets the full trajectory of two samples.** The quoting format is uniform:

```
turn N  THINK  the original text of the model's <think> for this turn (truncated with … when too long, not rewritten)
        CALL   tool_name({original arguments})
        RESP   original tool response (truncated with … when too long)
        ANSWER the content of <answer>
```

All taken from the `trajectory` field of `p4/parsed/<benchmark>.jsonl`, **quoted verbatim, not translated, not polished**.
Only two kinds of processing: truncation where too long (marked `…`), and our own annotations added at the end of lines (marked `←`; written in Chinese in the original version of this document).
**Trajectories often contain things the count table cannot see** — whether the model noticed the tool was broken, what it translated "behind" into,
whether it wrote down that it needed depth and then did not fetch it. The analysis of each class below points out that sentence.

**④ The assignment of 3b is disputed, so both readings are reported.** If "coordinate frame / semantic mismatch" is counted as reasoning error,
tool error : reasoning error = **241 : 33 ≈ 7.3 : 1**; if it is listed separately (what this document does), it is **241 : 19 ≈ 12.7 : 1**.
Which side gets the blame depends on whether you think the policy should fill in depth on its own.

---

## 1. Summary table: error cause × number of questions × benchmark

**322 / 322 all classified, zero uncovered, zero duplicates.**

| # | Error cause | Code | n | Share of 322 | Distribution (by benchmark) |
|---|---|---|--:|--:|---|
| 1 | **Tool error — detection localization inaccurate** | `1` | **197** | 61.2% | robospatial 45 · reflocation 44 · refplacement 42 · refunseen 40 · cvb2drelation 26 |
| 2 | **Tool error — depth estimate inaccurate** | `1` | **35** | 10.9% | cvb3ddepth 21 · blinkdepth 14 |
| 3 | **Toolset gap** (`fit`: no tool gives free space) | `5` | **32** | 9.9% | robospatial 32 |
| 4 | **Reasoning error** | `3` | **19** | 5.9% | robospatial 15 · cvb2drelation 3 · blinkdepth 1 |
| 5 | **Coordinate frame / semantic mismatch** | `3b` | **14** | 4.3% | robospatial 14 |
| 6 | **Tool error — degenerate detection** (different queries return the same point) | `1a` | **9** | 2.8% | robospatial 4 · cvb2drelation 2 · reflocation 2 · blinkdepth 1 |
| 7 | **Should have called, did not** | `2a` | **11** | 3.4% | robospatial 11 (of which front/behind without calling depth 8 · only detected the subject 3) |
| 8 | **Not separable in the 2D projection** (insufficient information) | `2D` | **2** | 0.6% | cvb2drelation 2 |
| 9 | **Argument error** (query string written wrong) | `2c` | **1** | 0.3% | cvb2drelation 1 |
| 10 | **Annotation / reference ambiguity** | `6` | **1** | 0.3% | cvb2drelation 1 |
| 11 | **Format error** (answer not in the option set) | `4` | **1** | 0.3% | blinkdepth 1 |

**The three large classes together are 241 + 32 + 19 = 292, i.e. 90.7%.**

```
tool error total (1 + 1a)  197 + 35 + 9 = 241   74.8%
reasoning error (3)                     19    5.9%
tool error : reasoning error = 12.7 : 1
```

> The paper's Appendix Table 16 gives 23 : 7 out of 30 cases on grasp.
> **Our ratio leans further toward the tool side, and on a sample ten times larger.**

**Another cut — by benchmark, each class as a share of that benchmark's own wrong answers:**

| benchmark | Wrong answers | Tool error | Toolset gap | Reasoning error | 3b | 2a | Other |
|---|--:|--:|--:|--:|--:|--:|--:|
| reflocation | 46 | **46** | — | — | — | — | — |
| refplacement | 42 | **42** | — | — | — | — | — |
| refunseen | 40 | **40** | — | — | — | — | — |
| cvb3ddepth | 21 | **21** | — | — | — | — | — |
| cvb2drelation | 35 | **28** | — | 3 | — | — | 4 |
| blinkdepth | 17 | **15** | — | 1 | — | — | 1 |
| robospatial | 121 | **49** | **32** | 15 | **14** | **11** | — |

> **Except for `robospatial`, the wrong answers of every benchmark are dominated by tool error (80–100%).**
> Of all five non-zero "reasoning error / 3b / 2a / toolset gap" counts, **72/77 fall in `robospatial` alone**.
> This is not a coincidence: `robospatial` is the only benchmark where the rule "compare two points in the image plane" does not hold (§5).

---

## 2. Error cause 1: tool error — detection localization inaccurate (197 items, 61.2%)

### 2.1 How it was judged

Two criteria, neither requires looking at the image:

- **pointing questions** (`reflocation` / `refplacement` / `refunseen` / `robospatial` Vacant):
  Is the model's answer a **verbatim pass-through** of the point returned by RoboRefer? Yes → the model did no processing at all, so the error can only be the tool's.
  Measured: **276/276 verbatim pass-through, not a single change** (the three RefSpatial); 104/122 pass-through on Vacant.
- **relation questions** (`cvb2drelation`): is the model executing the rule "compare x / compare y"?
  Follows the rule but answers wrong → the detection point is inaccurate. The self-consistency rate of this criterion (whether the rule agrees with the model on the samples answered **correctly**) is **99.83%**,
  and the rule's accuracy against GT is **95.4%** — the image plane is the correct semantics for CVBench 2D relation, so this criterion is usable.

### 2.2 Analysis

**This tier is the bulk of all wrong answers, and its bulk is RoboRefer.** 126 items on the three RefSpatial,
45 on `robospatial` Vacant, 26 on `cvb2drelation`; all 197 together are determined by the position of the detection point.

There is a sharp corollary: on the 277 RefSpatial samples, **the policy's entire contribution relative to "just call RoboRefer directly"
is how it words that `obj_name`** — it does no post-processing. In the paper's Table 2, RoboRefer-8B-SFT alone on
RefSpatial is 48.37 and SpaceTools-3B is 53.07; **that +4.7 comes entirely from query construction, not one point from reasoning.**

The distance distribution also supports this criterion: on `cvb2drelation`, the median distance along the deciding axis is **0.410** in the correct group
and **0.192** in the follows-rule-but-wrong group — the points in the wrong batch were crowded together to begin with.

### 2.3 Sample A — `cvb2drelation #10`: the rule was executed correctly, the point was wrong

> *Considering the relative positions of the **rug** (annotated by the red box) and the **pillow** …
> where is the rug located with respect to the pillow?*   `(A) above  (B) below`   **GT = A**

**Full trajectory** (chain `roborefer×2@2t`):

```
turn 1  THINK  Looking at this image, I need to identify the positions of the rug (marked by
               the red box) and the pillow to determine their spatial relationship. Let me
               detect the locations of these objects:
        CALL   roborefer.detect_one({"obj_name": "rug in red box", "image_index": 0})
        RESP   Detected 1 instance(s) of 'rug in red box': [(0.377, 0.794)]
        CALL   roborefer.detect_one({"obj_name": "pillow", "image_index": 0})
        RESP   Detected 1 instance(s) of 'pillow': [(0.552, 0.42)]

turn 2  THINK  Based on the detection results: - The rug (in the red box) is located at
               coordinates (0.377, 0.794) - The pillow is located at coordinates (0.552, 0.42)
               The vertical coordinate (v-value) indicates the vertical position in the image:
               - Rug: v = 0.794 (lower in the image) - Pillow: v = 0.42 (higher in the image)
               Since the rug has a higher v-value (0.794) compared to the pillow (0.42),
               the rug is positioned lower in the …
        ANSWER B
```

**The model compared the two y values, and compared them correctly** — the reasoning in that part of the trajectory is textbook, not one step wrong.
Yet the GT says the rug is **above** the pillow — under image-plane semantics this can only mean
that at least one of the two detection points is not where it should be (the rug is a large area; RoboRefer gives one point within it).
**The reasoning side is beyond reproach.**

### 2.4 Sample B — `refplacement #0`: the model is just a pipe

> *Please point out the **free space on the top of the wooden shelf and beside the picture frame**.*

**Full trajectory** (chain `roborefer×1@2t`, `score = 0.0`):

```
turn 1  THINK  I need to find the free space on the top of the wooden shelf and beside the
               picture frame. The image shows a wooden shelf on the right side and a picture
               frame on the wall. I will use the roborefer.detect_one tool to locate these
               objects and then find the free space around them.
        CALL   roborefer.detect_one({"obj_name": "free space on the top of the wooden shelf
                                     and beside the picture frame", "image_index": 0})
        RESP   Detected 1 instance(s) of 'free space on the top of the wooden shelf and
               beside the picture frame': [(0.816, 0.485)]

turn 2  THINK  I have found the free space on the top of the wooden shelf and beside the
               picture frame at (0.816, 0.485). This point is within the normalized pixel
               space of the image. Now, I can format this point as a list of tuples as
               requested.
        ANSWER [(0.816, 0.485)]
```

**Note the sentence "and then find the free space around them" in the turn 1 THINK — it originally intended to do a second step,
and then in turn 2 it just copied the return value.** The only thing the model did was stuff the whole question sentence into `obj_name`,
and then copy the return value verbatim into `<answer>`.
Chain `roborefer×1@2t`: no second call, no verification, no post-processing.
**Every single one** of these 42 `refplacement` wrong answers has this shape.

> So the headroom for this class is not in the policy. P6 §6.2 measured: replacing RoboRefer wholesale with Molmo-7B-D is
> **worse by −14.61 pp**, but per-sample oracle tool selection gives **+10.83 pp** — the move is to **add** (router / ensemble),
> not to replace.

---

## 3. Error cause 2: tool error — depth estimate inaccurate (35 items, 10.9%)

### 3.1 How it was judged

`blinkdepth` and `cvb3ddepth` take the same chain (`depth_estimator×1 + roborefer×2 + vision_ops×2`),
and the question asks which one is closer. The rule: **pick the one with the smaller measured depth**.

| benchmark | Decidable | Follows rule | Follows but wrong → tool error | Violates and wrong → reasoning error |
|---|--:|--:|--:|--:|
| `blinkdepth` | 114/124 | 109 (95.6%) | **14** | **1** |
| `cvb3ddepth` | 599/600 | 598 (**99.8%**) | **21** | **0** |

**100% of the 21 wrong answers on `cvb3ddepth` are DepthPro's errors; zero errors on the reasoning side.**

### 3.2 Analysis: half of these 35 have no margin at all

Splitting these 35 by relative gap `|d_A − d_B| / min(d_A, d_B)`:

| Relative gap | Count | Reading |
|---|--:|---|
| **< 10% (near tie)** | **10** | More like "no margin" than "estimated wrong" |
| 10 – 50% | 17 | |
| **> 50% (confidently reversed)** | **8** | |

The comparison is more convincing: **the median relative gap is 83.3% on correct samples and 25.3% on these 35 wrong answers.**
**Most depth wrong answers happen where the two objects were very close to begin with.**

> This directly bounds "swap the depth tool": **a slightly stronger tool could flip back the 10 near ties;
> eating the 8 confident errors needs a significantly stronger model.**

There is also a natural control: the two benchmarks share the same chain, the same rule, and reasoning fidelity close to 100% on both, yet the scores differ by ten points —
**those ten points are purely DepthPro's accuracy difference across different images**, unrelated to orchestration or policy.

### 3.3 Sample A — `cvb3ddepth #86`: confidently reversed (relative gap 91%)

> *Which object is closer to the camera, the **lamp** (red box) or the **refrigerator** (blue box)?*
> `(A) lamp  (B) refrigerator`   **GT = A**

**Full trajectory** (chain `depth_estimator×1 + roborefer×2 + vision_ops×2@3t`):

```
turn 1  THINK  I need to determine which object is closer to the camera by analyzing the depth
               information. Let me follow the systematic process: First, I'll estimate the depth
               map of the image, then locate both objects and compare their depth values.
        CALL   depth_estimator.estimate_depth({"image_index": 0})
        RESP   Estimated depth map for 1036x756 image. Focal length: 920.9px.
               Depth range: 1.84m to 16.34m (mean: 5.86m). Use $depth_map …
        CALL   roborefer.detect_one({"obj_name": "lamp (highlighted by a red box)", …})
        RESP   Detected 1 instance(s) of 'lamp (highlighted by a red box)': [(0.368, 0.144)]
        CALL   roborefer.detect_one({"obj_name": "refrigerator (highlighted by a blue box)", …})
        RESP   Detected 1 instance(s) of 'refrigerator …': [(0.6, 0.469)]

turn 2  THINK  Now I'll get the depth values at both detected locations:
        CALL   vision_ops.index_at({"data": "$depth_map", "u": 0.368, "v": 0.144})
        RESP   Pixel value at (0.368, 0.144) is 4.53564453125
        CALL   vision_ops.index_at({"data": "$depth_map", "u": 0.6, "v": 0.469})
        RESP   Pixel value at (0.600, 0.469) is 2.378054141998291

turn 3  THINK  Based on the depth analysis: - Lamp (red box): depth = 4.54 meters
               - Refrigerator (blue box): depth = 2.38 meters The refrigerator has a smaller
               depth value (2.38m) compared to the lamp (4.54m), which means the refrigerator
               is closer to the camera.
        ANSWER B
```

**This trajectory is the standard shape of this system when it "works correctly"** — three turns, five calls, one comparison.
Both detection points are also correct (verified against the image on the review card: the red box is the ceiling light at the top of the frame, and `(0.368, 0.144)` lands inside the box).

**All the model did was compare 4.536 with 2.378, and it compared them correctly.**
The red box encloses a **ceiling light** at the top of the frame, physically very close to the camera; GT says it is closer, meaning the actual depth is < 2.378 m,
while DepthPro reported 4.536 m — **a relative gap of 91%; not slightly off, but off by nearly 2× and in the wrong direction.**
Monocular depth reading "projects small" as "far" on texture-poor ceiling fixtures is a known failure mode of it.

### 3.4 Sample B — `blinkdepth #5`: near tie, anyone would have to flip a coin

> *Which point is closer to the camera?*   `(A) A is closer  (B) B is closer`   **GT = A**

**Full trajectory** (chain `depth_estimator×1 + roborefer×2 + vision_ops×2 + vlm×2@4t`):

```
turn 1  THINK  Looking at the image, I can see: - Point A is labeled near the top center,
               appearing to be on an orange in the background - Point B is labeled in the upper
               right area, also on an orange in the background … Both points appear to be on
               background oranges, but it's not immediately obvious which is closer. I need to
               use tools to determine t…
        CALL   depth_estimator.estimate_depth({"image_index": 0})
        RESP   … Depth range: 0.45m to 1.16m (mean: 0.66m) …
        CALL   roborefer.detect_one({"obj_name": "a", …})   RESP  [(0.315, 0.045)]
        CALL   roborefer.detect_one({"obj_name": "b", …})   RESP  [(0.635, 0.106)]

turn 2  THINK  Now let me get the depth values at these two points:
        CALL   vision_ops.index_at({"data": "$depth_map", "u": 0.315, "v": 0.045})
        RESP   Pixel value at (0.315, 0.045) is 0.8774694800376892
        CALL   vision_ops.index_at({"data": "$depth_map", "u": 0.635, "v": 0.106})
        RESP   Pixel value at (0.635, 0.106) is 0.8658632040023804

turn 3  THINK  The depth values are very close: - Point A: 0.877m - Point B: 0.866m
               The difference is only about 0.01m, which is within the margin where depth
               estimation might not be reliable. According to the guidelines, when depth values
               are too close (within 0.5m), I should reconsider the visual appearance and use
               the VLM detector for verification. Let me try the VLM detector:
        CALL   vlm.detect_one({"obj_name": "a", …})   RESP  [(0.378, 0.055)]
        CALL   vlm.detect_one({"obj_name": "b", …})   RESP  [(0.638, 0.052)]

turn 4  THINK  … Point A appears to be on an orange that is slightly more forward in the scene,
               while Point B is on an orange that is very close to the background. The depth
               values being so close (within 0.5m) makes …
        ANSWER B
```

**The depth difference between the two points is 1.16 cm, while the depth span of the whole image is only 71 cm (relative gap 1.3%).**
This is not "DepthPro estimated wrong"; it is that **this question is undecidable at this tool's resolution**.

**And the trajectory gives a fact the count table cannot see: the model noticed this itself.**
In turn 3 it explicitly writes "the difference is only about 0.01 m, which is within the range where depth estimation may be unreliable",
and, following the rule in the system prompt, switches to `vlm` to re-check, finally falling back to a visual judgment — **it did every action it could, and was still wrong.**

> Putting it under the same "tool error" label as `#86` is a coarseness of the label, not an error in the conclusion —
> **so the gap table in §3.2 must be cited together with the counts.** Incidentally, the criterion counts this item as "follows the rule"
> because its final answer (B) agrees with "pick the smaller one"; **but the trajectory shows it did not get B through that rule.**
> This is the known error direction of the criterion (the 4.4% of the 95.6% self-consistency rate), and here it caught one instance.

---

## 4. Error cause 3: toolset gap — `fit` questions (32 items, 9.9%)

### 4.1 How it was judged

Not by a criterion, but by the **tool histogram**: the quantity the question needs, the model never went to fetch, and **no single tool can return it**.

```
robospatial VQA fit questions   n = 105   accuracy 73/105 = 69.5%
                                **depth_estimator not called 105/105**
                                accuracy when GT = no  4/18 = 22%
                                model answers yes 83/105
```

### 4.2 Analysis

The question asks "does it fit", while the tools return "where is it". **`bounding_box` can give object size, `depth_estimator`
can give depth, but no tool returns "how much free space there is in some direction"** — that quantity has to be computed by combining several tools,
and the model never did that.

The consequence is very clean: **on fit questions the model almost always answers yes** (83 times out of 105 questions),
so it guesses most GT = yes questions right, and GT = no questions collapse to 22%.
What P5 recorded as "questions with GT `no` sit at chance level" **is, when split apart, concentrated here**
(fit 22% vs relation questions 58% vs front/behind 64%).

> **This path points to adding tools or adding composition ability, not tuning the prompt and not swapping the pointing tool.**
> As soon as fit questions stop being answered yes across the board, the 6.46 pp gap of `robospatial` VQA relative to the paper is closed.

### 4.3 Sample A — `robospatial #128`: the typical shape (should answer no, answered yes)

> *Can the **cup** fit **in front of** the **speaker**?*   **GT = No**, model answers **Yes**

**Full trajectory** (chain `roborefer×2@2t`):

```
turn 1  THINK  I need to analyze whether the cup can fit in front of the speaker. To do this,
               I should: 1. Locate the cup in the image 2. Locate the speaker in the image
               3. Assess the spatial relationship and available space
               Let me detect both objects first.
        CALL   roborefer.detect_one({"obj_name": "cup", "image_index": 0})
        RESP   Detected 1 instance(s) of 'cup': [(0.78, 0.494)]
        CALL   roborefer.detect_one({"obj_name": "speaker", "image_index": 0})
        RESP   Detected 1 instance(s) of 'speaker': [(0.154, 0.491)]

turn 2  THINK  Now I have the positions: - Cup is at (0.78, 0.494) - this is on the right side
               of the image - Speaker is at (0.154, 0.491) - this is on the left side …
               The question asks if the cup can fit "in front o…
        ANSWER Yes
```

**The turn 1 THINK states the plan clearly: step 3 is "Assess the spatial relationship and
available space". Then it only got two point positions, step 3 had nothing to work with, and turn 2 answered directly.**

No tool ever told it how much open space there is in front of the speaker, or how big the cup is.
What it got was "where", what was asked was "does it fit" — **the quantity it needs is not returned by any tool at all.**

### 4.4 Sample B — `robospatial #122`: wrong in the opposite direction too, so it is not simply a yes bias

> *Can the **tissue box** fit **behind** the **chair**?*   **GT = Yes**, model answers **No**

**Full trajectory** (chain `roborefer×2@2t`):

```
turn 1  THINK  I need to analyze whether the tissue box can fit behind the chair. To do this,
               I should: 1. Locate the tissue box 2. Locate the chair 3. Assess the spatial
               relationship and available space behind the chair
        CALL   roborefer.detect_one({"obj_name": "tissue box", …})  RESP  [(0.306, 0.531)]
        CALL   roborefer.detect_one({"obj_name": "chair", …})       RESP  [(0.477, 0.517)]

turn 2  THINK  Now I have the positions: - Tissue box: (0.306, 0.531) … - Chair: (0.477, 0.517)
               … "Behind the chair" from the camera's perspective would mean further away from
               the c…
        ANSWER No
```

**A verbatim-identical plan (even the THINK wording is almost the same), again only two point positions; this time it answered no, and was just as wrong.**

Only putting these two side by side is accurate: **it is not "the model has a yes bias, so it is wrong", but "it is guessing blindly,
and the prior of the guess leans yes"** — the prior helps it guess quite a few of the 87 GT=yes questions right, and collapses to 4/18 on the 18 GT=no questions.
**In both directions it has no quantity to base the answer on.**

---

## 5. Error cause 4: reasoning error (19 items, 5.9%)

### 5.1 How it was judged

Using the two earlier criteria in reverse: **violates the rule** (depth question did not pick the smaller one, relation question did not compare by coordinates)
or **modifies the tool's output on its own** (pointing question not passed through), and answers wrong.

Distribution: `robospatial` Vacant 15 · `cvb2drelation` 3 · `blinkdepth` 1.
**15/19 concentrated in Vacant alone.**

### 5.2 Analysis: it is not computing, it is eyeballing

There is an extremely clean statistic on Vacant:

| | n | Accuracy | Both coordinates are multiples of 0.05 |
|---|--:|--:|--:|
| Verbatim pass-through | 104 | **54.8%** | 2 items (**2%**) |
| Point override by the model | 18 | **27.8%** | 6 items (**33%**) |

**Once it overrides, accuracy halves, and the share of multiples of 0.05 differs by 16×.**

When passing through, the coordinates carry the tool's three decimal places (`0.271, 0.870`); as soon as the model does it itself, they become
**eyeballed round numbers** like `(0.40, 0.70)`, `(0.50, 0.85)`.
**So the concrete meaning of "reasoning error" on Vacant is: it is not computing, it is eyeballing.**
And Vacant's GT is a narrow strip a few tens of pixels wide; eyeballing at 0.05 granularity is not fine enough.

### 5.3 Sample A — `robospatial #11`: right direction, distance not nailed

> *In the image, there is a **cup**. Pinpoint one point within the **vacant space** suitable for
> placing an object **close to and to the right of** the cup.*

**Full trajectory** (chain `roborefer×1@2t`, **only one tool call in the entire sample**):

```
turn 1  THINK  I need to find a vacant space that is close to and to the right of the cup.
               First, I should locate the cup in the image to understand its position, then
               identify an appropriate vacant space to its right. Looking at the image, I can
               see a white mug/cup on the desk in the lower left portion of the image.
               Let me detect its exact location first.
        CALL   roborefer.detect_one({"obj_name": "cup", "image_index": 0})
        RESP   Detected 1 instance(s) of 'cup': [(0.271, 0.87)]   ✓ lands precisely on the cup on the desk

turn 2  THINK  The cup is located at approximately (0.271, 0.87), which is in the lower left
               area of the desk. Now I need to find a vacant space that is: 1. Close to the cup
               2. To the right of the cup (higher x-coordinate) Looking at the desk surface,
               to the right of the cup (x > 0.271), there appears to be vacant desk space.
               The desk extends to the right with some items like a keyboard and mouse further
               right. A suitable vacant sp…
        ANSWER [(0.50, 0.85)]
```

```
GT is an area made of 23 acceptable points — the narrow desk strip to the right of the cup:
  x range 0.327 – 0.558      y range 0.878 – 0.989
```

**There is not a single formula in the turn 2 THINK.** It says "look at the desk, there seems to be space to the right of the cup",
then gives a number — **this is what "eyeballing" looks like in a trajectory.**

**It did not copy the tool's point; it did it itself, and you can see clearly how it went wrong:**
`x = 0.50` lands inside the GT range (0.327–0.558); **the direction is entirely correct; what is off is y** —
all GT y values are ≥ 0.878, the model gave 0.85, **about 0.03 too high, just outside the upper edge of that narrow strip, a near miss.**
**It knew to go right, but did not nail how far forward.** Both coordinates are multiples of 0.05.

### 5.4 Sample B — `robospatial #3`: the same shape, further off

> *In the image, there is a **speaker**. Pinpoint one point within the vacant space suitable for
> placing an object **close to and behind** the speaker.*

**Full trajectory** (chain `roborefer×1@2t`):

```
turn 1  THINK  I need to find a vacant space that is close to and behind the speaker. First, I
               should detect where the speaker is located … Looking at the image, I can see a
               white speaker with a mesh/grill design on the left side of the desk. There's
               also a monitor, some papers, a mouse, and a cup visible …
        CALL   roborefer.detect_one({"obj_name": "speaker", "image_index": 0})
        RESP   Detected 1 instance(s) of 'speaker': [(0.154, 0.491)]

turn 2  THINK  The speaker is detected at position (0.154, 0.491) … Behind the speaker (to the
               right in the image) would be the area where the monitor and other desk items are
               located. To find a vacant space that is: - Close to the speaker - Behind it
               (to the right in the image) I should look a…
        ANSWER [(0.40, 0.70)]
```

```
GT area (10 points): x 0.251 – 0.400   y 0.416 – 0.455
```

**Note the definition turn 2 writes for itself: "behind = to the right in the image".**
It translated "behind" into the right side of the image — so x = 0.40 sits exactly on the right boundary of the GT, **the direction was judged correctly**;
but for y it gave 0.70, while the largest GT y is only 0.455, **off by 0.25**.

Again two multiples of 0.05.

**Putting the two side by side shows the mechanism: the model judges "which direction" fairly well, but "how far" is pure estimation,
and Vacant's scoring is exactly sensitive to the latter.**

---

## 6. Error cause 5: coordinate frame / semantic mismatch (14 items, 4.3%)

### 6.1 How it was judged

**The criteria cannot judge this class; the image has to be looked at.** The trigger condition is: both detection points are correct and the rule was executed correctly,
but **that rule is not the semantics of this question**.

Criterion C gives hard evidence that this class exists:

| | Decidable | Self-consistency rate | **Rule vs GT** | Model vs GT |
|---|--:|--:|--:|--:|
| `cvb2drelation` | 627/650 | 99.83% | **95.4%** | 94.7% |
| `robospatial` VQA relation questions | 87/87 | 98.55% | **79.3%** | 79.3% |

On both sides the model is executing the same rule (self-consistency ≈99% on both), but **the rule's accuracy against GT differs by 16 points**.
On `robospatial` the rule is right on GT **69/87**, the model is right on GT **69/87**, and **68/69 are the same samples** —
**not close, identical: on this question type the model is just one two-point coordinate comparison, and its ceiling is that rule's ceiling of 79.3%.**

### 6.2 Analysis: two subtypes

**① Image y ≠ 3D above (11 items).** The single number image y **encodes both "higher" and "farther"**,
and the tool only returns a 2D point — this ambiguity is unsolvable within the data the model gets.

**② Camera frame ≠ the benchmark's reference frame (3 items).** Systematic reversals appear on left/right questions,
with gaps of 0.40–0.59, **which cannot be detection error**.

### 6.3 Sample A — `robospatial #283` (subtype ①)

> *Is the **lamp** above the **bed**?*   **GT = No**, model answers **Yes**

**Full trajectory** (chain `roborefer×2@2t`):

```
turn 1  THINK  I need to determine if the lamp is positioned above the bed in the image …
               From my initial observation: - The lamp appears to be a tall floor lamp with a
               rectangular lampshade, positioned in the center-right area of the image
               - The bed is visible with bedding on the right side of the image
        CALL   roborefer.detect_one({"obj_name": "lamp", …})  RESP  [(0.561, 0.234)]  ✓ lands on the floor lamp
        CALL   roborefer.detect_one({"obj_name": "bed",  …})  RESP  [(0.703, 0.775)]  ✓ lands on the bed

turn 2  THINK  Now I have the coordinates: - Lamp: (0.561, 0.234) … - Bed: (0.703, 0.775) …
               In image coordinates, a lower y-value means higher up in the image … Since the
               lamp has y-coordinate 0.234 and the bed has y-coordinate 0.775, the lamp is
               indeed positioned above the bed **in the image** - it app…
        ANSWER Yes
```

**In the turn 1 THINK it says "a tall floor lamp" itself — it saw that it is a floor lamp.**
Then in turn 2 it swapped the question for "whose y is smaller in the image", and that swap is where it went wrong.

**Both points are correct and the rule was executed without problems, but that rule is not the semantics of this question.**
The floor lamp **stands on the floor next to the bed**, not above the bed; it is higher in the image **only because it is farther from the camera**.

It is neither a tool error (the points are correct) nor an ordinary reasoning error (the rule was not violated), so it is listed separately.

### 6.4 Sample B — `robospatial #347` (subtype ②)

> *Is the **toilet** right of the **shelf**?*   **GT = Yes**, model answers **No**

**Full trajectory** (chain `roborefer×2@2t`):

```
turn 1  THINK  I need to determine if the toilet is to the right of the shelf in this bathroom
               image … From my initial observation: - I can see a toilet in the lower left
               portion of the image - There appears to be a wooden shelf unit on the right side
        CALL   roborefer.detect_one({"obj_name": "toilet", …})  RESP  [(0.215, 0.766)]
        CALL   roborefer.detect_one({"obj_name": "shelf",  …})  RESP  [(0.801, 0.709)]

turn 2  THINK  Now I have the coordinates: - Toilet: (0.215, 0.766) - x-coordinate is 0.215
               - Shelf: (0.801, 0.709) - x-coordinate is 0.801 In normalized coordinates, the
               x-axis goes from 0 (left) to 1 (right). Since the toilet's x-coordinate (0.215)
               is less than the shelf's x-coordinate (0.801), the toilet is to the LEFT of the
               shelf, not to the right. Therefore, the answer … is No.
        ANSWER No
```

**Nothing wrong can be found in this reasoning — in turn 1 it even visually confirmed "the toilet is in the lower left of the frame, the shelf on the right",
consistent with the detection points.** The gap along the deciding axis is 0.586.

**A gap of 0.586 spans most of the image — it cannot be detection error.**
GT says the toilet is on the right, meaning "left/right" in this question is not left/right as seen from the camera
(`#295` microwave/fridge gap 0.567, `#310` mouse/keyboard gap 0.400,
**all three directions are opposite to the camera frame**).

**These three are the direct source of the 79.3% vs 95.4% in §6.1.**

---

## 7. Error cause 6: tool error — degenerate detection (9 items, 2.8%)

### 7.1 Definition

**RoboRefer returns the same point for different queries.** It differs from "localization inaccurate": inaccurate means giving a wrong answer,
degenerate means **not answering that query at all**. It appears across three benchmarks, 9 items in total.

### 7.2 Analysis

Worth noting is **the model's response: it actively switches among 3–6 phrasings to work around it**. This and the conclusion "phrasing is its only lever"
corroborate each other — and this failure mode happens to be **one that phrasing cannot break through**.

There is also an amplification effect: **one bad detection can contaminate multiple samples** (Sample B below).
So this document's "tool share of blame" is counted **per wrong answer**, not **per defect** — the same tool defect gets counted repeatedly.

### 7.3 Sample A — `cvb2drelation #621`: six queries, the same point

> *…where is the **bottle** (red box) located with respect to the **vase**?*  **GT = A**, model answers **B**

**Full trajectory** (chain `roborefer×6@6t` — six turns, all the same thing):

```
turn 1  THINK  Looking at this bathroom image, I need to identify the bottle (in the red box)
               and the vase, then determine their relative positions.
        CALL   roborefer.detect_one({"obj_name": "bottle in red box"})  RESP  [(0.558, 0.594)]
        CALL   roborefer.detect_one({"obj_name": "vase"})               RESP  [(0.558, 0.594)]

turn 2  THINK  The detection results show both objects at the same location (0.558, 0.594),
               which seems unusual. Let me look more carefully at the image … Let me try
               detecting the bottle more specifically:
        CALL   roborefer.detect_one({"obj_name": "red bottle on counter"}) RESP [(0.558, 0.594)]

turn 3  THINK  Let me try detecting the soap or other items on the counter that might be in the
               red box area:
        CALL   roborefer.detect_one({"obj_name": "soap in red box"})    RESP  [(0.558, 0.594)]

turn 4  THINK  Looking at the image more carefully … Let me detect the items on the counter
               more specifically:
        CALL   roborefer.detect_all({"obj_name": "bottles on counter"}) RESP  [(0.558, 0.594)]

turn 5  THINK  Let me try detecting the small items that might be in the red box area:
        CALL   roborefer.detect_all({"obj_name": "small items on counter"})
                                                                RESP  [(0.654, 0.626)]

turn 6  THINK  Based on the visual analysis of the image, the red box is located on the bathroom
               counter in the center-left area … the most direct comparison is to the …
        ANSWER B
```

**The first sentence of turn 2 is the evidence: "both objects at the same location (0.558, 0.594), which seems
unusual"** — **the model explicitly recognized that the tool was broken**, then spent four turns and four phrasings working around it, getting exactly the same coordinate the first five times.
It **knew** the result was off and kept trying, **but had no means to make the tool change its answer**.
Finally in turn 6 it gave up on verification and answered by force with "visual analysis".

### 7.4 Sample B — `robospatial #273` + `#274`: one bad detection, two wrong answers in a row

> `#273` *Is the **orange** left of the **bottle**?*    **GT = Yes**, model answers **No**
> `#274` *Is the **water pitcher** left of the **bottle**?*  **GT = Yes**, model answers **No**

**The two trajectories side by side** (both `roborefer×2@2t`):

```
#273
turn 1  THINK  I need to determine if the orange is to the left of the bottle … There's an
               orange on the counter - There's a water bottle (gray/silver colored) on the
               counter - Various other items like a paper towel holder, utensils …
        CALL   roborefer.detect_one({"obj_name": "orange"})  RESP  [(0.477, 0.552)]
        CALL   roborefer.detect_one({"obj_name": "bottle"})  RESP  [(0.082, 0.498)]  ← wrong detection
turn 2  THINK  … Since the orange's x-coordinate (0.477) is greater than the bottle's
               x-coordinate (0.082), the orange is to the RIGHT of the bottle …
        ANSWER No

#274
turn 1  THINK  I need to determine if the water pitcher is to the left of the bottle … 1. There's
               a water pitcher/jug on the left side of the counter … 2. There's a gray water
               bottle (Hydro Flask) in the center-right area …
        CALL   roborefer.detect_one({"obj_name": "water pitcher"})  RESP  [(0.266, 0.509)]
        CALL   roborefer.detect_one({"obj_name": "bottle"})         RESP  [(0.082, 0.498)]  ← same point
turn 2  THINK  … Since 0.266 > 0.082, the water pitcher is actually to the RIGHT of the bottle …
        ANSWER No
```

**In the `#274` THINK the model itself wrote "gray water bottle (Hydro Flask) in the center-right area"
— the bottle it saw is center-right in the frame, while the tool returned x = 0.082 (far left).**
The two perceptions directly contradict each other; the model did not notice and computed on from the coordinates.

`'bottle'` was detected at x = 0.082 (the far left of the frame; manual review identified it as hand soap),
so **every question on this image of the form "is X left of the bottle" will be answered no.**
Two questions, one defect.

---

## 8. Error cause 7: should have called, did not (11 items, 3.4%)

### 8.1 Definition

**The tool list has a tool that directly provides the quantity needed for the answer, and the model never called it.**
This is exactly the difference from "toolset gap" (§4): those 105 `fit` questions have **no tool to call**,
these 11 items **have a tool but did not call it**. **The former is the toolset's fault, the latter is the model's.**

Two subtypes: **front/behind without calling depth, 8 items** · **only detected the subject, the reference object never fetched, 3 items**.

### 8.2 Analysis

The most glaring number: **across all 350 `robospatial` samples, 0 called `depth_estimator`** —
and it is right there in that benchmark's tool list. The 29 `front/behind` questions ask precisely about depth order, **and not one of the 29 called it**,
accuracy 72.4%.

This class is **the only place where "a known effective alternative chain exists, yet the policy never takes it"**,
and is therefore the most direct target for reward shaping (upper bound about +8 questions).

### 8.3 Sample A — `robospatial #246`: only got half of the comparison

> *Is the **paper towel** above the **counter**?*   **GT = Yes**, model answers **No**

**Full trajectory** (chain `roborefer×1@2t`, **only one tool call in total**):

```
turn 1  THINK  I need to determine if the paper towel is above the counter. The paper towel is
               located on the counter, and I need to check if it is positioned above the counter
               surface. Let me use the detection tool to locate the paper towel.
        CALL   roborefer.detect_one({"obj_name": "paper towel", "image_index": 0})
        RESP   Detected 1 instance(s) of 'paper towel': [(0.293, 0.481)]

turn 2  THINK  The detection shows that the paper towel is at coordinates (0.293, 0.481), which
               means it is positioned in the upper-left area of the counter. However, the
               question asks if the paper towel is "above" the counter. Looking at the image,
               the paper towel is sitting on top of the counter surface. The paper towel appears
               to be at a similar or slightly elevated position above the counter surface, but
               it's not actually "abov…
        ANSWER No
```

**In the turn 1 THINK it already locked in the plan: "locate the paper towel" —
from start to finish it never intended to fetch the position of the counter.** Turn 2 made an impossible comparison with one point,
and finally wrapped up by eyeballing with "Looking at the image".

**The position of the reference object `counter` was never fetched.** It compares the above/below relation of two objects
but only got the coordinates of one of them — **the other half is guessed.**
`#272` (bottle / water) and `#301` (game controller / sofa) have the same shape.

### 8.4 Sample B — `robospatial #230`: asks about depth order, but only fetches two 2D points

> *Is the **speaker** in front of the **cup**?*   **GT = Yes**, model answers **No**

**Full trajectory** (chain `roborefer×2@2t` — **`depth_estimator` not called**):

```
turn 1  THINK  I need to determine if the speaker is in front of the cup. To do this, I should:
               1. Locate the speaker in the image 2. Locate the cup in the image
               3. **Compare their positions to determine which is closer to the camera**
        CALL   roborefer.detect_one({"obj_name": "speaker"})  RESP  [(0.154, 0.491)]
        CALL   roborefer.detect_one({"obj_name": "cup"})      RESP  [(0.78, 0.494)]

turn 2  THINK  Now I have the positions: - Speaker: (0.154, 0.491) - This is on the left side of
               the image - Cup: (0.78, 0.494) - This is on the right side of the image …
               **To determine which is "in front of" the other, I need to consider depth.**
               The speaker appears to be sitti…
        ANSWER No
```

**It writes down twice that it needs depth** (turn 1 step 3, turn 2 "I need to consider depth"),
**and `depth_estimator.estimate_depth` is right there in its tool list, yet it never called it.**
In the end it answered with two x coordinates and an "appears to be".

"in front of" asks about **depth order**, and two 2D points contain no depth.
`depth_estimator.estimate_depth` is in the tool list; one call would get it
(all 600 questions on `cvb3ddepth` do exactly this, with accuracy 96.5%).
**Here it did not call it, so it could only answer by force from the x coordinates.**

> Compare with the `cvb3ddepth` chain `depth_estimator×1 + roborefer×2 + vision_ops×2` in §3 —
> **the same model with the same tools lays out these three steps correctly on another benchmark.**
> So this is not a missing capability; it is **the orchestration on `robospatial` collapsing into the single chain `roborefer×2`**.

---

## 9. Error cause 8: not separable in the 2D projection (2 items)

**Both are listed in full.** Definition: both detection points are **correct**, but their difference along the deciding axis is so small that the 2D projection
**does not contain the information needed for the decision** — the difference from 3b is that no semantic misalignment is involved here; there is simply not enough information.

### 9.1 `cvb2drelation #412`

> *…where is the **person** (red box) located with respect to the **cell phone**?*  **GT = B**, model answers **A**

```
turn 1  CALL   roborefer.detect_one({"obj_name": "person in red box"})  RESP  [(0.58, 0.545)]
        CALL   roborefer.detect_one({"obj_name": "cell phone"})         RESP  [(0.253, 0.539)]
turn 2  THINK  Comparing the vertical positions (v-coordinates): - Person in red box: v = 0.545
               - Cell phone: v = 0.539 Since the v-coordinate increases downward in image
               coordinates, a higher v-value means a lower position in the image …
        ANSWER A
```

**A difference of 0.006 along the deciding axis (y)** — the model treated 0.545 vs 0.539 as a decidable difference.

### 9.2 `cvb2drelation #478`

> *…where is the **person** (red box) located with respect to the **bottle**?*  **GT = A**, model answers **B**

```
turn 1  CALL   roborefer.detect_one({"obj_name": "person in red box"})  RESP  [(0.507, 0.431)]
        CALL   roborefer.detect_one({"obj_name": "bottle"})             RESP  [(0.503, 0.85)]
turn 2  THINK  Comparing the x-coordinates (horizontal position): - Person in red box: x = 0.507
               - Bottle: x = 0.503 The person in the red box has a slightly higher x-coordinate
               (0.507 > 0.503), meaning they are positioned **slightly** to the right …
        ANSWER B
```

**A difference of 0.004 along the deciding axis (x), while their y differs by 0.42 — the person and the bottle are stacked vertically.**
The model itself wrote "slightly", and still made the judgment from that difference.

**A gap of 0.004–0.006 on an image normalized to [0,1] is 2–3 pixels.**
These two can be blamed neither on detection (the points are correct) nor on reasoning (the rule was executed);
**they are the boundary of the approach "answer this question with 2D points" itself.**

---

## 10. Error causes 9–11: 1 item each (all listed)

### 10.1 `2c` argument error — `cvb2drelation #169`

> *…where is the **fluorescent tube** located with respect to the **tank**?*  **GT = A**, model answers **B**

```
turn 1  THINK  Looking at this image, I need to identify the positions of the fluorescent tube
               and the tank … Let me detect both objects to get their precise locations:
        CALL   roborefer.detect_one({"obj_name": "fluentice tube"})   ← fluorescent spelled as fluentice
        RESP   Detected 1 instance(s) of 'fluentice tube': [(0.84, 0.631)]
        CALL   roborefer.detect_one({"obj_name": "tank"})   RESP  [(0.532, 0.461)]

turn 2  THINK  Based on the detection results: - The fluorescent tube is located at coordinates
               (0.84, 0.631) - The tank is located at (0.532, 0.461) … the tube (0.631) is
               positioned lower/below the tank (0.461) in the image.
        ANSWER B
```

**In THINK it wrote "fluorescent tube", in `obj_name` it typed `fluentice tube`
— the error happened in the step from thinking to calling, and afterwards it kept believing it had queried a fluorescent tube.**

**The model wrote the query string wrong, and RoboRefer returned a point anyway.**
The tool gives no "I don't recognize this word" signal — so **the error propagates silently**,
and the subsequent coordinate comparison is done on an unrelated point.

> This is the only **pure argument error** in the whole attribution, but the interface problem it points to is general:
> the detection tool does not report an error on meaningless queries (compare `reflocation #6`, where the tool explicitly returned
> `Detected 0 instance(s)` and the model still produced a point on its own — that item was assigned to `1a`).

### 10.2 `6` annotation / reference ambiguity — `cvb2drelation #263`

> *…where is the **plant** (annotated by the red box) located with respect to the **plants**?*
> **GT = B**, model answers **A**

```
turn 1  THINK  … I need to identify the plant in the red box and compare its position to other
               plants in the scene.
        CALL   roborefer.detect_all({"obj_name": "plant"})          RESP  [(0.855, 0.674)]
turn 2  THINK  Let me also detect the plant more specifically by looking at the red box area:
        CALL   roborefer.detect_one({"obj_name": "plant in red box"}) RESP [(0.845, 0.18)]
turn 3  THINK  … - The plant in the red box is at (0.845, 0.18) - in the upper right portion
               - The other plant detected is at (0.855, 0.674) - in the middle-right portion
               Comparing the x-coordinates … Since the red box plant has a smaller x-coordinate
               (0.845) compared to the other plan…
        ANSWER A
```

The question's options are `(A) left` / `(B) right`. Both points are correct (the one in the red box is above, the one in the foreground below),
and their **x differs by 0.010** — from this the model judged "the one in the red box is slightly to the left" and answered A; GT says B.
Which plant `the plants` (plural) in the question refers to is doubtful, so it is assigned to annotation/reference ambiguity,
**not counted as a fault of the model or the tool.**

> ⚠ **During re-check, the evidence string of the manual verdict was found to use the wrong axis.** In `p6/manual/verdicts.jsonl` this item says
> "in the image the one in the red box is higher ⇒ rule gives above, GT=below", **but this question asks left/right, not above/below**
> (original `question` field: `(A) left\n(B) right`).
>
> The classification itself is unaffected (both points correct, GT opposite, reference doubtful — still `6`),
> **but judging by the x gap of 0.010, it also falls inside criterion C's "rule degenerates" interval (threshold 0.02)** —
> that is, its shape is actually closer to the two `2D-indistinguishable` items in §9.
> **This document counts by the archived classification and does not re-adjudicate; but this evidence needs fixing.**

### 10.3 `4` format error — `blinkdepth #69`

> *Which point is closer to the camera?*   `(A) A is closer  (B) B is closer`   **GT = B**

**Full trajectory** (chain `depth_estimator×1 + roborefer×3 + vision_ops×3@3t`):

```
turn 1  THINK  Looking at the image, I can see three children playing with a bicycle on a
               sidewalk. The children appear to be at roughly similar distances from the camera,
               so I need to use depth estimation to determine which is closer …
        CALL   depth_estimator.estimate_depth({"image_index": 0})
        RESP   … Depth range: 0.89m to 8.40m (mean: 3.28m) …
        CALL   roborefer.detect_one({"obj_name": "child on the left"})    RESP [(0.216, 0.339)]
        CALL   roborefer.detect_one({"obj_name": "child in red jacket"})  RESP [(0.597, 0.506)]
        CALL   roborefer.detect_one({"obj_name": "child on the right"})   RESP [(0.92, 0.652)]

turn 2  THINK  Now let me get the depth values at these three locations:
        CALL   vision_ops.index_at(… 0.216, 0.339)  RESP  3.118241310119629
        CALL   vision_ops.index_at(… 0.597, 0.506)  RESP  2.0428996086120605
        CALL   vision_ops.index_at(… 0.920, 0.652)  RESP  1.346768856048584

turn 3  THINK  Based on the depth analysis: - Child on the left (A): depth = 3.12m
               - Child in red jacket (B): depth = 2.04m - Child on the right (C): depth = 1.35m
               The child on the right has the smallest depth value (1.35m vs 3.12m for child on
               the left), indicating they are closest to the camera.
        ANSWER C      <- **the only options are (A)/(B)**
```

**Turn 3 makes it clear how it arrived at `C`: it invented its own A/B/C labels for the three children**,
then faithfully reported "C is closest" — **while the question's (A)/(B) refer to the two annotated points on the image, not to three children.**
The whole depth chain was executed without any problem; the error is that it redefined the option set.

**The model detected three children, probed three depths, and then answered with an option that does not exist.**

This item also surfaces a pipeline issue: `parse_ok` is true (the `<answer>` tag is intact and parseable),
**so `--strict` did not catch it**, and "format error = 0" was therefore under-reported.
A full scan of the 1374 questions with options: `blinkdepth` 1/124 · `cvb2drelation` 0/650 · `cvb3ddepth` 0/600.

> **Fixed**: `parse_dump.py` gained an `answer_off_options` field, made a warning rather than a contamination flag —
> "answering outside the options" is the model's fault, not a broken pipeline.

---

## 11. The two continuously scored benchmarks (another 120 samples)

`boppose` / `bopgrasp` have no "right/wrong"; criteria A and C do not apply.
**But the pass-through criterion does apply, and more cleanly** — both tools put the final answer directly in 2D form in the returned text.

### 11.1 `boppose`: 60/60 pass-through, all error belongs to `bounding_box`

```
bit-for-bit, same order        56/60
same set (reordering allowed)  60/60        samples with reordering only: #0 #10 #25 #35
```

**The model never modifies corner values.** Those 4 cases are only reorderings — and the metric is convex-hull IoU, insensitive to order,
so reordering neither changes the score, and it also shows precisely that **it is trying to satisfy the question's requirement "bottom face first, counter-clockwise"
and cannot**.

The zero scores are not random either; they follow **point-cloud fitting degeneracy**:

| | n | OBB shortest/longest edge (median) | Ratio < 0.20 |
|---|--:|--:|--:|
| Zero score | 9 | **0.161** | **6/9** |
| Non-zero | 51 | 0.306 | 12/51 |

Fisher exact **p = 0.0164**. **When the box is fitted flat and thin, the projected convex hull is useless.**

#### Sample A — `boppose #0` (IoU = 0.000)

**Full trajectory** (chain `bounding_box×1 + depth_estimator×1 + roborefer×1 + sam2×1@5t`):

```
turn 1  THINK  I'll help you find the 8 normalized corner coordinates of the bounding box for
               the can of sliced mushrooms. Let me process this step by step.
        CALL   roborefer.detect_one({"obj_name": "can of sliced mushrooms"})
        RESP   Detected 1 instance(s): [(0.779, 0.787)]

turn 2  CALL   sam2.segment_from_point({"x": 0.779, "y": 0.787, "image_index": 0})
        RESP   Segmented object … with **IoU score: 0.640**. Use $segmentation_mask …

turn 3  CALL   depth_estimator.estimate_depth_with_pointcloud({"image_index": 0})
        RESP   Generated 1932x1092 depth map and 2,109,744 3D points …

turn 4  CALL   bounding_box.compute_bbox({"point_cloud": "$point_cloud",
                                          "mask": "$segmentation_mask", …})
        RESP   Oriented bounding box computed from **524/2109744 points**.
               Corners in 3D: [[0.326, 0.195, 0.899], [0.34, 0.189, 0.904], …]
               Extent [0.016, 0.017, **0.001**]   shortest/longest = 0.059

turn 5  ANSWER [(0.786, 0.803), (0.797, 0.793), (0.777, 0.781), (0.787, 0.803),
                (0.788, 0.77), (0.798, 0.792), (0.778, 0.781), (0.788, 0.771)]
```

**There is not a single THINK from turn 2 to turn 5.** On this chain the model is a pure pipe:
detect → segment → point cloud → fit → copy the answer. The only "active action" is swapping two of the eight points the tool gave
(same set, different order).

**This "cuboid" is 1.6 cm × 1.7 cm × 1 mm — a sheet of paper.**
Only 524 points took part in the fit, the OBB collapsed into a plane, the projection is a thin slit, and it has zero overlap with the GT convex hull.

#### Sample B — `boppose #28` (IoU = 0.952)

**Full trajectory** (the same chain, compared step by step):

```
turn 1  THINK  I'll help you find the 8 normalized corner coordinates of the oatmeal-raisin
               cookies box. Let me process this step by step.          <- verbatim isomorphic to #0
        CALL   roborefer.detect_one({"obj_name": "oatmeal-raisin cookies"})
        RESP   [(0.479, 0.417)]

turn 2  CALL   sam2.segment_from_point({"x": 0.479, "y": 0.417, …})
        RESP   Segmented object … with **IoU score: 0.971**          <- #0 is 0.640

turn 3  CALL   depth_estimator.estimate_depth_with_pointcloud({"image_index": 0})
        RESP   Generated … 2,109,744 3D points …                      <- identical to #0

turn 4  CALL   bounding_box.compute_bbox(…)
        RESP   Oriented bounding box computed from **36801/2109744 points**
               Extent [0.214, 0.158, 0.046]   shortest/longest = 0.215        <- #0 is 524 points / 0.059

turn 5  ANSWER [(0.428, 0.351), (0.445, 0.544), (0.504, 0.276), …]     bit-for-bit, same order ✓
```

**The model does exactly the same thing on these two samples: copy down the eight points the tool gave.**
The difference lies entirely in `sam2`'s mask quality (0.640 vs 0.971) and the number of points entering the fit (524 vs 36,801).
In `#0` the model did one extra thing — swapped two points, probably trying to satisfy the question's ordering requirement —
**that reordering has no effect on the score at all, but it is the only trace of the model "actively doing something", and doing it did not help.**

### 11.2 `bopgrasp`: tool failure 40/60, the model has a fully deterministic fallback

```
compute_grasp gives 5 points on 20 items -> bit-for-bit pass-through 19
40 items take the failure exit -> model places the grasp center on the roborefer detection point **40/40 (±0.02)**
```

**When the tool fails, the model does not abstain; it executes a deterministic rule: grasp the object center.**

#### Sample A — `bopgrasp #0` (tool succeeds)

**Full trajectory** (chain `depth_estimator×1 + grasp_generator×1 + roborefer×1 + sam2×1@5t`):

```
turn 1  THINK  I need to find the grasp plane for the carton of orange juice. To do this, I'll
               need to: 1. Detect the orange juice carton location 2. Segment it to get a mask
               3. Estimate depth and generate a point cloud 4. Compute the grasp pose …
        CALL   roborefer.detect_one({"obj_name": "carton of orange juice"})  RESP [(0.381,0.428)]
turn 2  THINK  Good, I found the orange juice carton at (0.381, 0.428). Now I need to segment it …
        CALL   sam2.segment_from_point(…)   RESP  … **IoU score: 0.932** …
turn 3  THINK  Great! I have the segmentation mask. Now I need to estimate depth …
        CALL   depth_estimator.estimate_depth_with_pointcloud(…)  RESP  … 937,664 3D points …
turn 4  THINK  Perfect! Now I have: - $segmentation_mask - $point_cloud - $focal_length_px
               Now I can compute the grasp pose …
        CALL   grasp_generator.compute_grasp(…)
        RESP   **Generated collision-free grasp with confidence 0.953** from 13885/937664 masked
               points. Filtered 100 → 8 collision-free grasps.
               Projected 2D gripper points: [(0.375, 0.190), (0.422, 0.200), (0.323, 0.17…
turn 5  THINK  Excellent! The grasp generator has provided me with the five normalized (x, y)
               points … Let me format this according to the user's requested format.
        ANSWER Grasp center: [0.375, 0.190], Left finger base: [0.422, 0.200],
               Right finger base: [0.323, 0.178], Left finger tip: [0.407, 0.300],
               Right finger tip: [0.310, 0.291]          <- bit-for-bit pass-through of compute_grasp
```

```
GT center                    (0.358, 0.257)     center distance 0.069
gripper axis vs GT angle     17.1°              NCE = 1.31
```

#### Sample B — `bopgrasp #1` (tool fails → fallback)

**Full trajectory** (the first four turns are isomorphic to `#0`; the fifth turn diverges):

```
turn 1  THINK  I need to find the grasp plane points for the cylindrical Parmesan cheese
               container … 1. Detect 2. Segment 3. Estimate depth 4. Compute grasp 5. Extract …
        CALL   roborefer.detect_one({"obj_name": "grated Parmesan cheese container"})
        RESP   [(0.73, 0.594)]
turn 2  CALL   sam2.segment_from_point(…)  RESP  … **IoU score: 0.516** …   <- #0 is 0.932
turn 3  CALL   depth_estimator.estimate_depth_with_pointcloud(…)  RESP  … 937,664 3D points …
turn 4  THINK  Perfect! Now I have … Now I can compute the grasp pose for the Parmesan …
        CALL   grasp_generator.compute_grasp(…)
        RESP   **Error: RuntimeError: No collision-free grasps found.
               All grasps collide with other objects.**
turn 5  THINK  The grasp generator couldn't find a collision-free grasp, likely because the
               Parmesan cheese container is surrounded by other objects. **Since I was instructed
               not to call the grasp tool more than once if it fails, I need to estimate the
               grasp pose myself based on the image.** Looking at the Parmesan cheese container
               at position (0.73, 0.594), it appears to be a cylindrical container. For a
               typical parallel-jaw gripper g…
        ANSWER Grasp center: [0.73, 0.594], Left finger base: [0.70, 0.594],
               Right finger base: [0.76, 0.594], Left finger tip: [0.67, 0.594],
               Right finger tip: [0.79, 0.594]
```

**Turn 5 spells out that fallback rule plainly: "I was told not to call it a second time after the tool fails, so I can only estimate it myself".**
So it places the grasp center unchanged on the detection point (**displacement 0.0000**),
and the **y coordinates of all five points are 0.594** — a horizontal gripper opened at the object center.

```
GT center                    (0.730, 0.518)     center distance 0.076
gripper axis vs GT angle     54.9°              NCE = 1.23
```

**This pair compresses the whole contradiction into two samples.** `#1`'s orientation is off by **54.9°**
(almost perpendicular to the correct grasp direction; in reality it would not pick anything up),
**yet NCE gives it 1.23, lower than `#0` (1.31), where the tool actually computed a grasp.**
**Of one that can grasp and one that cannot, the reward function rates the latter better.**

Whole-group statistics:

| | Tool succeeds (n=20) | Fallback (n=40) |
|---|--:|--:|
| Distance from grasp center to GT (median) | 0.116 | **0.092** |
| Angle between gripper axis and GT (median) | **18.3°** | 63.9° |
| Angle < 30° | **16/20** | 7/40 |
| NCE (lower is better) | 1.38 | **1.07** |

**The fallback wins on position and loses on orientation** — which is exactly why the two metrics disagree:
**NCE is mostly determined by position, SR by orientation.**

> **One reading is valid, one is not.**
> It **cannot** be read as "the fallback is better than the tool" — the two groups are different scenes, and which scenes the tool fails on is not random; this is a confounded comparison.
> What it **can** be read as: **this RL reward (NCE) is insensitive to gripper orientation.**
> An answer whose orientation is off by a median of 64° scores lower on NCE than a real grasp generated by the tool.
>
> ⚠ **The `correct` field in `p4/parsed/bopgrasp.jsonl` is meaningless** —
> it is the generic criterion `score >= 0.5`, while `bopgrasp`'s `score` is NCE, **where lower is better**.

---

## 12. A fourth kind of error: consistency (not among the 11 classes above)

The sampling experiment (`n=5, T=1.0`) shows another class that **does not belong to the attribution of any single run**:
the model cannot give a stable answer to the same question by itself. The key cut is to split again by "whether the `<tool_call>` of the five runs are verbatim identical" —
**identical calls ⇒ the five runs face the same evidence, so a flip can only happen at the decision layer**:

| | n | Split | Of which tool calls **verbatim identical** | Calls differ |
|---|--:|--:|--:|--:|
| RoboSpatial VQA | 228 | 100 (43.9%) | **46** | 54 |
| RoboSpatial Vacant | 122 | 51 (41.8%) | **0** | 51 |
| `blinkdepth` | 124 | 30 (24.2%) | **3** | 27 |

- **The 46 on VQA are pure decision flips**: same image, same `roborefer` returns, opposite yes/no.
- **Vacant has none at all**: all 51 flips come with a different `obj_name` — that is not "casting five votes",
  it is "asking five different questions".
- **`blinkdepth` has only 3**, which in turn corroborates the depth criterion: given the same two depth readings,
  "pick the smaller one" is almost never violated.

> **⚠ The phenomenon is there, the gain is not.** After re-checking majority vote@5 on the second group of independent samples:
> **0/12 paired comparisons are significant, and the sign of the VQA direction flipped between the two groups.**
> **Correct statement: majority vote@5 has no effect on these three benchmarks.**
> The model does waver on the same evidence, but **the wavering is symmetric, and voting cannot recover it**.

---

## 13. Summary: three things this table yields

**① The share of blame is overwhelmingly on the tool side, and concentrated in one tool.**

| Tool | Wrong answers directly caused | Share of 322 |
|---|--:|--:|
| **RoboRefer** | 126 (the three RefSpatial) + 45 (Vacant pass-through) + 26 (cvb2d) + 9 (degenerate) = **206** | **64.0%** |
| **DepthPro** | 21 (`cvb3ddepth`) + 14 (`blinkdepth`) = **35** | 10.9% |
| `bounding_box` | all `boppose` error (counted separately) | — |
| `grasp_generator` | 40/60 direct failures (counted separately) | — |

> **Definition**: counted here by **per-sample classification**, so RoboRefer's share (64.0%) is higher than the
> 53.4% in the base draft's summary — the base draft only counted the three RefSpatial + Vacant pass-through, while this table additionally counts
> the 26 items of `cvb2drelation` (follows rule but wrong = detection point inaccurate) and the 9 degenerate detections.
> **The two numbers do not contradict each other; the counting scope differs. Always cite them with the definition.**

**But this headroom cannot be eaten by replacing a single tool**: measured, replacing it wholesale with Molmo is **worse by −14.61 pp**,
while the per-sample oracle is **+10.83 pp**. Either build a router / ensemble, or bring in a significantly stronger third tool.
**The oracle upper bound is not an achievable gain.**

**② Except on `robospatial`, "reasoning" is a rule that can be written as code, and the model executes it verbatim.**
276/276 verbatim pass-through on pointing; 60/60 on `boppose`; 95.6% / 99.8% rule-following on the depth questions.
**This means swapping in any policy model on these benchmarks gives the same result.**

**③ The only real policy gap is `robospatial`, and its gap has a clear shape:**

| Question type | n | Accuracy | Depth not called | Accuracy at GT=`no` | Reaching 100% would add |
|---|--:|--:|--:|--:|--:|
| Relation questions (2D-decidable) | 87 | 79.3% | — | 18/31 = 58% | +18 |
| `front/behind` (needs depth order) | 29 | 72.4% | **29/29** | 7/11 = 64% | **+8** (has a tool, did not call it) |
| **`fit` (needs free-space extent)** | 105 | 69.5% | **105/105** | **4/18 = 22%** | **+32** (no tool to call) |
| Detection does not match | 7 | 57.1% | — | — | +3 |

**There are only two actionable priorities:**
1. **The 8 items of `front/behind`** — the only place where "a known effective alternative chain exists, yet the policy never takes it";
   directly targetable with reward shaping, upper bound +8 questions.
2. **The 32 items of `fit`** — needs added tools or added composition ability, **not prompt tuning, not swapping the pointing tool**.

---

## 14. Limitations

- **Manual classification has only one round of annotation, so κ cannot be computed.** The 32 items are single-annotated with no second annotator,
  so there is **no inter-annotator agreement measure**. The mitigation is the per-item evidence published in `p6/manual/verdicts.jsonl`.
- **The assignment of `3b` is disputed** (see item ④ in §0); both readings are reported.
- **One piece of archived evidence needs fixing**: in `p6/manual/verdicts.jsonl`, the evidence string for `cvb2drelation #263`
  is written as above/below, while that question's options are left/right (see §10.2). The classification is unaffected.
- **The criterion and the trajectory occasionally disagree**: `blinkdepth #5` was judged "follows the rule", but the trajectory shows it ultimately reached the same answer through visual reasoning
  (see §3.4). This is one instance in the criterion's known error direction; the 95.6% self-consistency rate quantifies its upper bound.
- **One bad detection can contaminate multiple samples** (§7.4), so the "tool share of blame" is counted **per wrong answer**, not **per defect**.
- **"Tool error" is a coarse label**; the gap table in §3.2 must be cited together with the counts, otherwise "no margin" gets read as "estimates badly".
- **The two-group comparison on `bopgrasp` is confounded** (which scenes the tool fails on is not random).
- **This document does not cover accuracy or deviation attribution**, which is P5's content; **nor does it cover P7 / GFlowRL.**
- **One difference from the base draft**: `01_official_checkpoint_eval/reports/p6_error_attribution_report.md` records the manual batch as 31 items,
  while `p6/manual/verdicts.jsonl` actually has **32 items** (`pending31.json` also has 32 entries).
  This document counts 32; the total is still 322, and **every class count matches the base draft's summary table item by item**.

---

## Appendix: recomputation

### A.1 Existing scripts (run from the root of the `spacetools-repro` repo)

| Script | Output |
|---|---|
| `tools/p6/p6_inventory.py` | the clean table in §0 |
| `tools/p6/p6_split.py` | criterion A (depth) · criterion B (pointing) |
| `tools/p6/p6_relations.py` | criterion C (relation questions), with built-in self-consistency and semantics checks; Vacant pass-through vs point override |
| `tools/p6/p6_continuous.py` | the two continuous benchmarks in §11 |
| `tools/p6/p6_consistency.py` | §12 consistency |

> ⚠ `p6_inventory.py` / `p6_split*.py` hardcode the path as `$HOME/mnt/Agentic RL/spacetools-repro/p4/parsed`.
> On a different machine, work around it with a symlink; do not edit the archived scripts:
>
> ```bash
> mkdir -p "$HOME/p6shim/mnt" && ln -sfn "$HOME/mnt" "$HOME/p6shim/mnt/Agentic RL"
> HOME="$HOME/p6shim" python3 tools/p6/p6_inventory.py
> ```
>
> `p6_relations.py` / `p6_continuous.py` use relative paths; just run them from the repo root.

### A.2 New in this document: per-sample classification script

The cross-tab in §1 is generated by a new script that combines the criteria above, gives **manual verdicts priority**,
assigns each of the 322 wrong answers exactly one class, and self-checks for "uncovered / duplicates":

```
python3 attrib.py p4/parsed p6/manual/verdicts.jsonl
→ total wrong answers 322 · classified 322 · uncovered 0 · extra 0
```

The only implementation difference between the script and `p6_split.py` is one patch in the depth criterion:

```python
# Take the first two detection points "whose depth was actually probed" — when roborefer degenerates the model re-detects with vlm,
# in which case the first two detection points never entered index_at, so they must be aligned by the keys of pix
probed = [p for p in d if p in pix]
dA, dB = (pix[probed[0]], pix[probed[1]]) if len(probed) >= 2 else (pix.get(d[0]), pix.get(d[1]))
```

**Without this patch, `blinkdepth #119` falls into "criterion skipped" and cannot be classified**
(for this sample RoboRefer returns the same point `(0.55, 0.283)` for A/B, and the model only got usable coordinates after re-detecting with `vlm`).
With it applied, 322/322 are fully covered, and every class count matches the base draft's summary table item by item.

### A.3 Data

| Data | Path |
|---|---|
| Per-sample records (source of all statistics) | `spacetools-repro/p4/parsed/` |
| Manual classification: pending / verdicts / review cards | `spacetools-repro/p6/manual/` |
| Probes and outputs of the tool swap | `spacetools-repro/p6/probes/` · `p6/swap/` |
| Sampling experiment (two groups) | `spacetools-repro/p6/passk/` · `p6/passk2/` |

**Except for the review cards, every number can be recomputed directly from the repo copy, with no GPU and no rerun of the evaluation.**
