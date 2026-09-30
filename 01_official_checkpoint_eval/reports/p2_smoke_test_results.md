# P2 smoke test results

> **Corrections from Phase V1 (2026-08-27).** Every accuracy number below
> reproduces exactly under the fixed parser. Three diagnostic statistics do not,
> and the corrected values are the ones P5/P6 should use:
>
> - **Grasp tool split: 19 succeeded / 41 errored**, not 27/33. The "Leads for
>   P6" table below counted only `No collision-free grasps` as a failure and
>   scored the 8 `Top-down filtering removed all` samples as successes. Verified
>   against the raw dump text independently of the parser. The tool therefore
>   fails on **68%** of samples, not 55%.
> - **Failure causes are 33 vs 8, not 96 vs 16.** Those came from `grep -c` on
>   the eval log, where the tool prints several lines per call. The dump — what
>   the model actually received — has 33 and 8. Collision filtering still
>   dominates, by **4.1x** rather than 6x.
> - **"Mean turns" below follows verl's convention**
>   (`user + assistant + 1`, `tool_agent_loop.py:238`). `parse_dump.py` reports
>   assistant turns, which is half of it: blinkdepth 6.26 = 3.13 assistant +
>   2.13 tool-result + 1 prompt. Neither is wrong; they are different counts.
>
> See `CHANGES.md` §7.

All runs below are clean: zero OutOfMemoryError, zero Traceback, zero RayTaskError.
Any grasp-tool RuntimeErrors listed are the tool's designed "no valid grasp here"
exit, which the model receives as a tool response and recovers from.

## Comparison against the paper

| benchmark            |    n | ours          | paper  | direction |
|----------------------|-----:|---------------|--------|-----------|
| cvb2drelation (head) |   32 | 93.75%        | 94.92  | low       |
| blinkdepth (full)    |  124 | 86.29-89.52%  | 90.32  | see below |
| robospatial Vacant   |   32 | 56.25%        | 52.46  | high      |
| bopgrasp MACE (full) |   60 | 44.46         | 43.06  | high      |
| bopgrasp SR (full)   |   60 | 55.00%        | 50.00  | high      |

Three above, two below: no systematic bias.

Caveat on the subsets: the 32-sample runs use head(N), not random sampling.
robospatial's first 32 rows are 100% "vacant" (point-answer) questions, so they
must be compared against the paper's Vacant number (52.46), never its Overall
(70.00), which mixes in the yes/no questions.

## The pipeline is not deterministic

blinkdepth was run three times with an identical configuration:

| run  | correct | accuracy |
|------|--------:|---------:|
| run4 |     108 |   87.10% |
| run5 |     107 |   86.29% |
| run6 |     111 |   89.52% |

Mean 87.63%, range 3.23 pp. Decomposed across the three runs:

| group            |   n | note                                        |
|------------------|----:|---------------------------------------------|
| always correct   | 106 | stable core                                 |
| always wrong     |  12 | genuine DepthPro failures on hard scenes    |
| flipping         |   6 | near-ties decided by floating-point noise   |

The stable ceiling is 106 + 6 = 112 of 124 = 90.32%, which is exactly the number
the paper reports. Our stable core therefore matches the paper's; a single run
simply draws somewhere in the 106-112 band.

Mechanism: sglang's batch composition changes floating-point reduction order,
which shifts logits enough to flip low-confidence tokens. Measured between two
runs: 57 of 124 outputs differed textually, 14 of those probed different pixels,
and 1 changed the final answer. One flipped sample (idx 54) had byte-identical
probe points AND byte-identical depth values in both runs -- only the model's
conclusion differed, on a pair of depths 0.056 m apart out of ~1.08 m.

Consequence for P4 and P5: a single-run number carries roughly +/-2 samples of
nondeterminism on top of sampling error. Small benchmarks suffer most.

| benchmark          |   n | 1 sample | expected run-to-run band |
|--------------------|----:|---------:|--------------------------|
| bopgrasp, boppose  |  60 |  1.67 pp | ~ +/- 5 pp               |
| blinkdepth         | 124 |  0.81 pp | ~ +/- 1.6 pp             |
| cvb2drelation      | 650 |  0.15 pp | ~ +/- 1.2 pp             |

The bopgrasp MACE/SR figures above come from a single run and carry that band.
For P4, small benchmarks should be run 2-3 times and reported as a range.

## Why blinkdepth fails when it fails

Of the 16 wrong samples in run4, none were infrastructure failures: zero
truncated tool responses, zero max-turn exhaustions, one tool error, and every
sample produced an <answer>. Twelve used exactly the standard chain
(depth x1, roborefer x2, vision_ops x2) in 2 turns.

The model's reasoning rule is correct essentially always: it answers with the
point whose measured depth is smaller, and the question asks which point is
closer to the camera. Across all samples, 102 of 103 correct answers and 15 of
15 wrong ones picked the nearer measured point. So every failure is the depth
tool returning the wrong relative depth, not the model misreading it.

Two inspected cases, with the probe points overlaid on the images, confirm the
pointing tools land exactly on the pre-drawn A/B markers:

  idx 84 - foggy sea. Point A sits in empty haze, point B on a ship's rigging.
           DepthPro returned A=10.44 m and B=19.05 m, i.e. it placed distant fog
           nearer than the ship.
  idx 43 - mountain valley. Both points sit on similar dark hillsides at
           opposite frame edges, across a scene spanning foreground grass to
           distant peaks.

Both are classic monocular-depth failure modes: low texture and very large depth
range. In the plan's taxonomy these are 1b (depth inaccurate), not 1a (pointing
wrong) and not 3a (misreading tool output).

Implication: blinkdepth's score is set almost entirely by DepthPro. The policy
contributes nothing beyond calling the right tools, which is also why
RoboRefer-8B-SFT alone scores 88.71 and SpaceTools only 90.32.

## Wall-clock, for the P4 budget

Toolshed starts once for a whole run; verl/sglang restarts per benchmark.
Startup is ~240 s. Per-sample cost is remarkably flat across workloads:

| run           |   n | mean turns | image px | eval time | s/sample |
|---------------|----:|-----------:|---------:|----------:|---------:|
| cvb2drelation |  32 |       4.00 |  0.17 MP |     116 s |     3.62 |
| blinkdepth    | 124 |       6.26 |  0.19 MP |     432 s |     3.48 |
| robospatial   |  32 |       4.00 |  2.76 MP |     107 s |     3.34 |
| bopgrasp      |  32 |       9.88 |  0.92 MP |     118 s |     3.69 |

Turn count varies 2.5x and pixel count 16x, yet the per-sample cost stays within
3.3-3.7 s. The bottleneck is policy generation and the concurrency ceiling
(max_num_seqs=256), not the tools.

P4 estimate for all 2121 samples:
    2121 x 3.5 s          = ~7,400 s = ~2.1 h
    Toolshed startup once =    ~240 s
    9 x sglang restart    =  ~1,080 s
    total                 = ~2.5 h

Report this as 2.5-4 h. The largest measured run was 124 samples; cvb2drelation
(650) and cvb3ddepth (600) need several concurrency waves and may add overhead.

## Memory, measured

| component            | plan estimate | measured        |
|----------------------|--------------:|----------------:|
| Molmo (fp32)         |        ~15 GB |       33.0 GB   |
| RoboRefer-8B         |        ~17 GB |       18.6 GB   |
| depth_estimator x2   |   ~4 GB each  |  12.5 GB each   |
| sam2 x2              |   ~2 GB each  |   1.2 GB each   |
| bbox x2 + grasp      |         ~5 GB |         ~1 GB   |
| tools total          |        ~49 GB |        ~80 GB   |

Two plan assumptions were wrong: Ray PACKS fractional GPUs rather than spreading
them (roborefer 0.5 + vlm 0.5 = 1.0 landed both large models on one card), and
DepthPro is three times larger than estimated. Together they put ~80 GB on two
48 GB cards, i.e. 84% occupancy with no headroom, and concurrent tool calls
OOMed 68 times in a 124-sample run. Those samples scored 38.5% against 88.3%
for the clean ones, dragging blinkdepth from 88.3% to 83.1%.

Fix: EVAL_GPUS=1 so the tools get three cards, with the fractional budget
re-balanced around the measured sizes. All upstream replica counts preserved.

Stress test: robospatial at 2.76 MP (14x the blinkdepth baseline) ran with zero
OOM, so P4 is feasible on this hardware.

## Leads for P6

Grasp tool failure rate on real BOP-ASK data, split by whether the tool errored:

| group          |  n | MACE  | SR     |
|----------------|---:|------:|-------:|
| tool succeeded | 27 | 49.42 | 77.78% |
| tool errored   | 33 | 40.40 | 36.36% |
| all            | 60 | 44.46 | 55.00% |

The grasp tool fails outright on 55% of samples. If it never failed, SR would be
around 77.78%, i.e. roughly 23 points of headroom. This is a biased upper bound,
since scenes where the tool succeeds are probably easier; the plan's oracle-tool
ablation is the rigorous version. It does point at the right target though.

Failure causes across 60 samples: 96 "No collision-free grasps" versus 16
"Top-down filtering removed all". Collision filtering dominates by 6x, so the
hardcoded TOPDOWN_GRAVITY_VECTOR is NOT the main suspect. Look instead at
collision_threshold (0.0100 m) and the scene downsampling, which keeps only
8,192 of 1,359,917 points (0.6%).

## Methodological warning for P6

The model confabulates justifications. One sample reasoned "Since I was
instructed not to call the grasp tool more than once if it fails, I need to
estimate the grasp pose myself" -- but no such instruction appears anywhere in
its 10,170-character prompt. The fallback behaviour is real and effective
(learned during SFT), but the stated reason was invented.

P6 must therefore classify errors from verifiable traces only: which tools were
called, with what arguments, what they returned, and what the final answer was.
The model's stated motivation is not evidence.
