# P4 results: full evaluation on 4x A100-SXM4-40GB

**Complete.** Nine benchmark keys, 2121 samples, plus two extra runs each of the
two n=60 benchmarks -- 13 evaluations in all. Written as each benchmark landed
rather than at the end, because the instance can disappear.

Deliverables live in `p4/`: `dumps/` the raw verl output, `parsed/` the enriched
per-sample records for P6, `logs/` the eval logs and 1 Hz GPU traces, gzipped.

Reproduced with `tools/p4_run.sh <benchmark> [run<N>]` and
`tools/p4_check.sh <benchmark> [run<N>]`, both in this repo. `p4_run.sh` refuses
to overwrite an existing dump and refuses a run label that is not `run<N>`; both
guards exist because of mistakes they now prevent -- the second after a pasted
command line created a whole parallel results tree under `run2bash`.

The session's own transcript is archived at `/workspace/session-archive/` on the
persistent volume. It is **not** in this repo: it is 3 MB of full command history
and would be an outward-facing disclosure into a repository whose visibility
could not be verified from the instance. Everything needed to continue the work
is in these records by design, not in that transcript.

**Every one of the 13 runs passed `parse_dump.py --strict`. Across 2121 samples
the health indicators are zero without exception** -- no OOM, no truncated tool
response, no turn exhaustion, no missing `<answer>`, and `turn-count mismatch 0`
on every dump that carries verl's own count. Given that a silently OOM-poisoned
run is the failure mode this whole pipeline is built to catch, that is the single
most important line in this file.

Configuration is `run_eval.sh` at `f0742338` with `EVAL_GPUS=1` and the tool
budget from `910f6f5d`. The one deviation specific to this hardware is
`actor.fsdp_config.model_dtype=bf16`, which is bit-identical for evaluation
(`CHANGES.md` §8, `PROVENANCE.txt` [20]).

## Table 2 comparison

All nine benchmark keys complete, 2121 samples. Every run passed
`parse_dump.py --strict`; a benchmark that had not would not appear here.

| Table 2 row | n | ours | paper | delta |
|-------------|--:|-----:|------:|------:|
| RoboSpatial VQA | 228 | 73.25-73.68% (2 runs) | 79.38 | **-5.70 to -6.13** |
| RoboSpatial Vacant | 122 | 50.82-51.64% (2 runs) | 52.46 | -0.82 to -1.64 |
| RoboSpatial Overall | 350 | 65.43-66.00% (2 runs) | 70.00 | -4.00 to -4.57 |
| BLINK Relative Depth | 124 | 86.29-87.90% (3 runs) | 90.32 | **stable ceiling matches exactly** |
| RefSpatial (3-way mean) | 277 | 53.35 simple / 53.79 weighted | 53.07 | +0.28 / +0.72 |
| CVBench 2D Relation | 650 | 94.62% | 94.92 | -0.30 |
| CVBench 3D Depth | 600 | 96.50% | 96.00 | **+0.50** |
| BOP-ASK Pose | 60 | 53.36 mean IoU (3 runs, identical) | 34.37 | *metric mapping open* |
| BOP-ASK Grasp MACE | 60 | 43.07-46.17 (3 runs) | 43.06 | paper at our lower bound |
| BOP-ASK Grasp SR | 60 | 55.00-56.67% (3 runs) | 50.00 | +5.00 / +6.67 |

**Eight of the ten are within a couple of samples of the paper, in both
directions.** Two need saying out loud:

- **RoboSpatial VQA, -6.13 pp**, is the only shortfall larger than run-to-run
  noise. 14 samples at n=228.
- **BOP-ASK Pose cannot be compared at all yet** -- the eval's metric is a
  projected convex-hull IoU that is provably blind to corner order, and the
  paper's 34.37 is demonstrably not a rate over 60. See below.

`Overall` is not an independent measurement -- it is the sample-weighted mean of
VQA and Vacant, which the plan verified and this run reproduces. **Of Table 2's
ten numbers only nine are independent**, and P5 must say so.

### bopgrasp across three runs

| run | MACE | SR |
|---|--:|--:|
| run1 | 43.0653 | 56.67% (34/60) |
| run2 | 46.1745 | 56.67% (34/60) |
| run3 | 43.6386 | 55.00% (33/60) |
| **range** | **43.07-46.17** (3.11 pp) | **55.00-56.67%** (1.67 pp) |
| paper | 43.06 | 50.00 |

The paper's MACE sits **exactly at our lower bound**. SR is consistently 3 to 4
samples above it.

P2 measured 44.46 / 55.00 on the A6000; **both fall inside our range**, which is
the cleanest cross-hardware agreement in the whole run.

**This is the most nondeterministic benchmark of the nine, and for a different
reason than the others.** Between run1 and run2 only **3 of 60 outputs** were
textually identical and only 8 of 60 scores matched -- against boppose, where all
60 scores matched every time. The cause is not sglang: GraspGen is a diffusion
model, so the tool itself samples. P0 predicted exactly this ("the remaining
nondeterminism comes from the tools; GraspGen is diffusion").

What it moves and what it does not:

| | |
|---|---|
| always fails, all 3 runs | 37 samples |
| always succeeds, all 3 runs | 18 samples |
| flips | **5 samples** |

So scene geometry decides the grasp tool's outcome for 55 of 60 scenes, and the
diffusion sampling only reaches 5 borderline ones. The identical 56.67% in runs 1
and 2 is a coincidence of the threshold, not stability -- the underlying scores
differ on 52 of 60 samples.

### blinkdepth: both hardware platforms reach the paper's number

Three draws on the A100 against P2's three on the A6000, decomposed the same way:

| | draws | always correct | always wrong | flip | **stable ceiling** |
|---|---|--:|--:|--:|--:|
| **A100** (this work) | 109 / 107 / 109 | 105 | 12 | 7 | **112/124 = 90.32%** |
| **A6000** (P2) | 108 / 107 / 111 | 106 | 12 | 6 | **112/124 = 90.32%** |
| paper | | | | | **90.32** |

**Two independent hardware platforms produce the identical stable ceiling, and it
is exactly the number the paper reports.** The always-wrong count is 12 on both.
Of the always-correct samples, **103 are the same samples** on both platforms --
only 2 differ on the A100 side and 3 on the A6000 side.

This is stronger evidence than the bit-identity check on the weights. That showed
the `model_dtype=bf16` change cannot alter what sglang receives; this shows that
**the whole accumulated set of deviations -- bf16, sm_80, cudnn 9.16, numpy 2.x,
the rebuilt CUDA extension -- does not move the benchmark's stable core at all.**
What moves between runs is the same 6-7 near-tie samples P2 identified, decided by
floating-point reduction order.

The A100's 1.61 pp spread against the A6000's 3.23 pp is draw luck, not a
systematic difference: the flip counts (7 and 6) are effectively the same.

**P5 should report this benchmark as its stable ceiling with the band attached,
not as a single draw.** A single run lands anywhere in 107-111 and invites a
"we are 3 points below the paper" reading that three runs show to be wrong.

### RefSpatial: which mean?

The paper reports one number for RefSpatial and no breakdown, so the three
benchmarks can only be checked in aggregate -- and *how* they are aggregated is
not stated. Both conventions land close:

| | reflocation | refplacement | refunseen | mean |
|---|---:|---:|---:|---:|
| ours | 54.00 (54/100) | 58.00 (58/100) | 48.05 (37/77) | |
| simple mean of three | | | | **53.35** |
| sample-weighted (149/277) | | | | **53.79** |
| paper | | | | 53.07 |

**+0.28 pp** under a simple mean, **+0.72 pp** weighted -- 1 to 2 samples at
n=277, where one sample is 0.36 pp. The choice of convention moves the answer by
less than half a percentage point, so it does not change any conclusion, but P5
should state which it used and give both.

For `robospatial` the plan established that the paper's Overall *is* the
sample-weighted mean, by reproducing 70.00 from the VQA/Vacant split to within
0.004. No equivalent check is possible here.

`refunseen` came in lowest, which is what "unseen objects" should do. Worth
noting because it was predictable before the run: for our mean to land exactly
on the paper's, `refunseen` would have had to be 45-47%, and it came in at
48.05%.

### robospatial: the split, and the one real gap so far

The plan's split rule reproduces exactly. Ground truth divides on shape --
`Yes`/`No` is VQA, a `[(x, y), ...]` point list is Vacant -- giving **228 / 122**,
precisely the counts the plan predicted, and the plan's arithmetic check holds:

    (228 x 79.38 + 122 x 52.46) / 350 = 69.996   vs the paper's Overall of 70.00

So **the paper's Overall is the sample-weighted mean of the other two, not an
independent measurement.** Of Table 2's ten numbers only nine are independent,
and P5 must say so or a reader will count ten pieces of evidence.

Question type fully determines the tool pattern, with almost no exceptions:

| | chain |
|---|---|
| VQA (228) | `roboreferx2@2t` 218, `roboreferx1@2t` 9, `roboreferx3@2t` 1 |
| Vacant (122) | `roboreferx1@2t` **122/122** |

**VQA is the only real shortfall in the whole evaluation, and a second run
confirms it.** Each figure moved by exactly one sample between runs:

| | run1 | run2 | paper |
|---|--:|--:|--:|
| VQA | 73.25% (167) | 73.68% (168) | 79.38 |
| Vacant | 50.82% (62) | 51.64% (63) | 52.46 |
| Overall | 65.43% (229) | 66.00% (231) | 70.00 |

At n=228 one sample is 0.44 pp, so the ~6 pp VQA gap is about 13 samples and is
**not** single-run nondeterminism. Vacant is 2 samples off and needs no
explanation.

**The errors are strongly asymmetric, and the asymmetry reproduces almost
exactly:**

| ground truth | n | run1 | run2 |
|---|--:|--:|--:|
| `yes` | 165 | **83.03%** (137) | **83.03%** (137) |
| `no` | 63 | 47.62% (30) | 49.21% (31) |

The `yes` class is *identical* across runs; the `no` class moves by one sample.
**On questions whose answer is "no" the model sits at chance, run after run.**

It is not a naive yes-bias -- the marginal answer distribution (yes 170/169,
no 58/59) tracks the ground truth's (yes 165, no 63) closely -- but something
systematic pushes it toward "yes" on the hard cases. Closing the gap needs ~13
more correct; if they all came from the `no` class it would have to rise from
~48% to ~69%.

Across the two runs, 218 of 350 samples are always correct, 108 always wrong,
and 24 flip -- a 6.9% flip rate, which puts the stable core well below the
paper's number rather than the difference living in the noise.

**This is a P5/P6 lead, not a conclusion**, and it rests only on verifiable
traces: which tools were called and what the final answer was. It says nothing
yet about *why*. Note also that P2's robospatial Vacant figure of 56.25% came
from `head(32)`, which is not representative of the full 122 -- the full number
is 50.82%.

### Two depth benchmarks, one chain, ten points apart

`cvb3ddepth` and `blinkdepth` run **the same tool chain** --
`depth_estimatorx1 + roboreferx2 + vision_opsx2` -- on 96.2% and 82.3% of their
samples respectively, and score 96.50% and 86.29%.

P2 established that blinkdepth's score is set almost entirely by DepthPro, with
failures concentrated in low-texture, very-large-depth-range scenes (a foggy sea,
a mountain valley). `cvb3ddepth` puts the identical chain on easier imagery and
loses 21 of 600. **So the ten-point spread is scene difficulty, not the tool
chain** -- which makes this pair a natural control for P6's attribution work:
same tools, same orchestration, same policy, very different outcome.

Worth noting for P5: `cvb3ddepth` at 96.50% is also, to two decimals, the number
the paper's Table 2 gives for RoboRefer-8B-SFT alone on this benchmark (96.50).
Almost certainly a coincidence at n=600 -- but the plan already flags that on
several benchmarks a single tool outscores the whole system, and this is one of
the two where it does.

### boppose: the number is solid, the comparison is not

**Do not report `boppose` as 63.33%.** That is `parse_dump`'s generic
`score >= 0.50` accuracy, and `score` here is not an accuracy at all.

`verl/utils/reward_score/bop_ask_bench.py:172` scores `question_type == "pose"`
by extracting **8 points** from the ground truth and the prediction -- the
corners of a 3D box projected to 2D -- returning 0.0 if either side does not have
exactly eight, and otherwise the **IoU of the two convex hulls**. It is a
continuous value in [0, 1].

Our run, recomputed from `raw_answer` with the repo's own function and matching
the dumped scores **60/60 exactly**:

    mean IoU x 100     53.36
    median             68.46
    all 60 samples     (8, 8) points -- zero format failures
    9 zero-score       genuine zero overlap, not format rejections

**Which aggregation does the paper's 34.37 use?** A threshold sweep finds no
clean match (>=0.75 gives 38.33%, >=0.80 gives 23.33%). And 34.37 cannot be a
rate over n=60 at all: `60 x 0.3437 = 20.62`, which is not an integer. That test
is decisive because it works everywhere else --

| paper value | n | n x p | integer? |
|---|--:|--:|---|
| blinkdepth 90.32 | 124 | 111.997 | yes -> 112 |
| cvb2drelation 94.92 | 650 | 616.980 | yes -> 617 |
| cvb3ddepth 96.00 | 600 | 576.000 | yes -> 576 |
| VQA 79.38 | 228 | 180.986 | yes -> 181 |
| Vacant 52.46 | 122 | 64.001 | yes -> 64 |
| bopgrasp SR 50.00 | 60 | 30.000 | yes -> 30 |
| **boppose 34.37** | 60 | **20.622** | **no** |
| bopgrasp MACE 43.06 | 60 | 25.836 | no (known continuous) |

-- so the paper's Pose figure is a continuous mean, like MACE, and most likely
mean IoU x 100. On that reading we are **+18.99 pp above the paper**, which is
far outside anything the other eight numbers show.

**That gap is a P5 question, not a result.** The most likely explanation is that
the paper's pose metric is not this projected-hull IoU -- a 2D convex-hull
overlap is much more forgiving than a real 3D pose measure. This is the same
trap the plan already documented for `bopgrasp`, where the eval's `acc/mean@1`
is an RL training NCE and the paper's MACE/SR need
`analysis/analyze_grasp_result.py`. **The repo ships a post-processor for grasp
and for nothing else**, so `boppose` needs the mapping established before its
number can be placed in Table 2 at all.

**The metric is blind to corner order, and that is probably the explanation.**
Three independent runs of `boppose` produced **identical scores on all 60
samples** to 1e-12, while 3 to 5 samples' output text differed each time. One of those three, sample
10, emitted a genuinely different `<answer>`:

    run1  [(0.645, 0.58), (0.678, 0.572), (0.679, 0.574), (0.646, 0.582), ...]
    run2  [(0.645, 0.58), (0.648, 0.656), (0.678, 0.572), (0.679, 0.646), ...]

-- the **same eight points in a different order**, scoring exactly the same
because a convex hull discards ordering. Confirmed by measurement rather than
inference: shuffling all eight predicted corners and re-scoring changes the score
by **0.00 on all 60 samples**, maximum absolute change `0.00e+00`.

So the eval's pose score measures whether the projected *silhouette* overlaps,
and is completely insensitive to corner correspondence -- which is exactly where
a box's orientation lives. A pose metric that respects correspondence must score
lower, often much lower. **That is the most likely source of the +18.99 pp gap**,
and it is testable off-machine: re-score the dumps with a correspondence-aware
metric and see whether the number moves toward 34.37.

What is safe to report now: the metric definition, the full distribution, the
order-invariance result, and that both runs are clean and format-valid on every
sample.

### boppose is stable, and for a reason worth stating

| run | sha256 | mean IoU x100 | per-sample agreement |
|---|---|--:|--:|
| run1 | `b4c864a5ad65` | 53.3586 | -- |
| run2 | `0d3ddf3930bd` | 53.3586 | **60/60** to 1e-12 |
| run3 | `af001188f682` | 53.3586 | **60/60** to 1e-12 |

Three independent runs, three different sha256s, three different sets of `uid`s.
Text differs on 3 to 5 samples in each pairwise comparison (57/60, 55/60, 57/60
identical) and **not one of those differences reaches the score**.

The two dumps are *not* the same file -- different sha256, different `uid`s,
eleven minutes apart -- and 3 of 60 outputs differ textually. The pipeline's
nondeterminism is present; it just cannot reach this metric. Two of the three
differ only inside `<think>`, and the third differs only in corner ordering,
which the hull discards.

Contrast blinkdepth, where P2 measured 57 of 124 outputs differing and 1 flipping
the answer: a binary A/B choice decided by two near-equal depths is maximally
exposed to a logit nudge, while eight coordinates scored by area overlap are
almost immune. **The expected run-to-run band is not a property of sample size
alone -- it depends on how the metric couples to the output.**

### bopgrasp, and a classification bug in the repo's own analysis script

`parse_dump`'s `score` column is meaningless here -- `mean score 1.9354` is the
RL training NCE, lower-is-better, exactly as the plan warned. The paper's figures
come from `analysis/analyze_grasp_result.py`, the only post-processor the repo
ships:

    Overall mean centre-angle score: 43.0653 over 60 samples      (paper 43.06)
    Overall > 40 ratio: 56.67%                                    (paper 50.00)

**MACE lands on the paper's number essentially exactly.** SR is +6.67 pp, which
is 4 samples at n=60 -- inside the ~±5 pp band this benchmark carries.

#### The script under-counts tool failures

`analyze_grasp_result.py:248` classifies a sample as a tool error with a single
exact substring match:

    "has_error": "Error: RuntimeError: No collision-free grasps found." in obj["output"]

**It never looks for `Top-down filtering removed all grasps.`**, the grasp tool's
other failure exit. On this run that is 11 of 60 samples silently counted as
successes. This is almost certainly where P2's 27/33 split came from -- not a
hand-counting slip but the script's own blind spot, inherited.

Correcting it sharpens the P6 lead rather than blurring it:

| classification | errored | succeeded |
|---|---|---|
| script's (one failure mode) | n=29, MACE 40.20, SR 37.93% | n=31, MACE 45.75, SR 74.19% |
| **correct (both modes)** | n=40, MACE 40.66, SR 40.00% | **n=20, MACE 47.87, SR 90.00%** |

The 11 misclassified samples score MACE 41.89 / SR 45.45% on their own -- much
nearer the errored group than the succeeded one, which is why moving them raises
the succeeded group's SR from 74.19% to 90.00%.

**So the headroom lead is larger than P2 recorded**: if the grasp tool never
failed, SR would be around 90% against an actual 56.67%, roughly **33 points**
rather than P2's 23. The same caveat still applies and is now more important, not
less: scenes where the tool succeeds are probably easier, so this is a biased
upper bound. The plan's oracle-tool ablation is the rigorous version.

#### Cross-hardware agreement on the failure split

| | this run (A100) | P2 corrected (A6000) | P2 as originally recorded |
|---|---|---|---|
| succeeded / errored | **20 / 40** | 19 / 41 | ~~27 / 33~~ |
| `No collision-free grasps` | 29 samples | 33 samples | ~~96~~ (log lines) |
| `Top-down filtering removed all` | 11 samples | 8 samples | ~~16~~ (log lines) |

The log for this run contains 73 and 20 matching lines against 29 and 11 actual
samples -- the tool prints several lines per call, which is the trap P2 fell into
and V1 caught.

### Run-to-run bands

`blinkdepth`, three draws on this hardware:

    109/124  87.90%     (Phase V2)
    107/124  86.29%     (P4 run1)
    109/124  87.90%     (P4 run2)
    range    107-109      1.61 pp,  mean 108.33

See the decomposition above: both hardware platforms reach a stable ceiling of
112/124 = 90.32%, the paper's exact figure. **Report this one as a band, never as
a point value.**

## Run health

| benchmark | OOM | truncated | max turns | no answer | turn-count mismatch | tool failures |
|-----------|----:|----------:|----------:|----------:|--------------------:|--------------:|
| `blinkdepth` run1 | 0 | 0 | 0 | 0 | 0 | 1 sample (2 events) |
| `blinkdepth` run2 | 0 | 0 | 0 | 0 | 0 | 2 samples (3 events) |
| `cvb2drelation` | 0 | 0 | 0 | 0 | 0 | 0 |
| `reflocation` | 0 | 0 | 0 | 0 | 0 | 0 |
| `refplacement` | 0 | 0 | 0 | 0 | 0 | 0 |
| `refunseen` | 0 | 0 | 0 | 0 | 0 | 0 |
| `robospatial` run1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `robospatial` run2 | 0 | 0 | 0 | 0 | 0 | 0 |
| `cvb3ddepth` | 0 | 0 | 0 | 0 | 0 | 0 |
| `boppose` run1 | 0 | 0 | 0 | 0 | 0 | 1 sample (1 event) |
| `boppose` run2 | 0 | 0 | 0 | 0 | 0 | 1 sample (1 event) |
| `boppose` run3 | 0 | 0 | 0 | 0 | 0 | 1 sample (1 event) |
| `bopgrasp` run1 | 0 | 0 | 0 | 0 | 0 | 40 samples (by design, see below) |

`blinkdepth`'s tool failure is **not** contamination. Sample 89 asked
`bounding_box.compute_bbox` for `$point_cloud` and `$segmentation_mask` having
called neither `estimate_depth_with_pointcloud` nor `sam2`; the tool rejected the
call ("point_cloud must be Nx3 numpy array, got str"), and the model recovered
and answered correctly. A model planning error -- P6 taxonomy 2c -- and the
reason `vars_phantom` now exists.

**Sample 89 does it again in run2**, with the same two invented variables, and
run2 adds sample 119 inventing `$segmentation_mask`. So the behaviour is a
property of those samples rather than a random slip, which makes it a tractable
P6 target: two of 124 blinkdepth samples reliably ask a tool for something no
tool ever produced.

## Wall clock and memory

| benchmark | n | wall clock | peak GPU (4 cards) |
|-----------|--:|-----------:|--------------------|
| `blinkdepth` (run1) | 124 | 7m 28s | 21.1 / 34.6 / 25.7 / 25.6 GB |
| `blinkdepth` (run2) | 124 | 7m 34s | 20.5 / 34.6 / 25.7 / 25.5 GB |
| `cvb2drelation` | 650 | 22m 51s | 31.5 / 20.0 / 25.7 / 28.5 GB |
| `reflocation` | 100 | 5m 14s | 20.5 / 30.4 / 9.3 / 25.6 GB |
| `refplacement` | 100 | 5m 10s | 20.5 / 30.4 / 9.3 / 29.8 GB |
| `refunseen` | 77 | 4m 51s | 20.5 / 30.4 / 9.3 / 29.8 GB |
| `robospatial` (run1) | 350 | 15m 10s | 20.8 / 30.4 / 9.3 / 32.6 GB |
| `robospatial` (run2) | 350 | 15m 11s | 20.8 / 30.4 / 17.5 / 33.2 GB |
| `cvb3ddepth` | 600 | 23m 9s | 20.7 / 34.5 / 25.7 / 31.9 GB |
| `boppose` (run1) | 60 | 5m 29s | 31.1 / 21.0 / 25.8 / 28.3 GB |
| `boppose` (run2) | 60 | 5m 26s | 21.8 / 30.4 / 25.8 / 29.2 GB |
| `boppose` (run3) | 60 | 5m 26s | 21.8 / 30.4 / 25.8 / 29.8 GB |
| `bopgrasp` (run1) | 60 | 6m 13s | 21.7 / 30.4 / 34.4 / 26.5 GB |

**Card indices are not stable across runs.** Toolshed restarts per invocation and
Ray re-packs, so Molmo was on GPU1 for `blinkdepth` and GPU0 for
`cvb2drelation`. The grouping is what is fixed -- Molmo alone, RoboRefer + SAM2,
DepthPro x2 + bbox + grasp, policy -- not the index. Read the peak columns as a
multiset. No card has exceeded **34.6 GB of 40** (87%), which is Molmo's measured
ceiling.

### Total cost

    13 evaluations, 2121 samples + 240 repeat samples
    wall clock                        ~2h 10m
    marginal                          1.75 s/sample
    fixed                             230 s per invocation, paid 13 times (~50m)

The handoff budgeted 2.5-4 h for nine benchmarks run in one invocation. Running
them one at a time, with a `--strict` gate between each and three runs of the two
small benchmarks, came in under the low end.

### Cost model

Two measured points fit a straight line:

    marginal   1.75 s/sample
    fixed       230 s per run_eval.sh invocation (Toolshed + sglang startup)

P2 measured **3.3-3.7 s/sample** on the A6000. The A100 is running at roughly
half that, consistent with the handoff's prediction from memory bandwidth
(1555 vs 768 GB/s). Running one benchmark at a time pays the 230 s startup nine
times instead of once -- about 30 minutes of the total -- which buys a `--strict`
gate between every benchmark.

## Tool usage

| benchmark | calls |
|-----------|-------|
| `blinkdepth` | `vision_ops` 249, `roborefer` 244, `depth_estimator` 122, `vlm` 22, `bounding_box` 2, `sam2` 1 |
| `cvb2drelation` | `roborefer` 1311, `vision_ops` 6, `depth_estimator` 3 |
| `reflocation` | `roborefer` 100 -- nothing else |
| `refplacement` | `roborefer` 100 -- nothing else |
| `refunseen` | `roborefer` 77 -- nothing else |
| `robospatial` | `roborefer` 570 -- nothing else |
| `cvb3ddepth` | `roborefer` 1202, `vision_ops` 1200, `depth_estimator` 600, `vlm` 4 |
| `boppose` | `roborefer` 61, `bounding_box` 61, `sam2` 60, `depth_estimator` 60 |
| `bopgrasp` | `roborefer` 60, `sam2` 60, `depth_estimator` 60, `grasp_generator` 60 |

`cvb2drelation` is essentially a single-tool benchmark: 615 of 650 samples used
exactly `roboreferx2@2t` -- point at two things, compare. **`vlm` is never
called**, which is the same pattern V3 found on `robospatial`.

All three RefSpatial benchmarks are stronger still: **every sample, 277 of 277,
uses exactly `roboreferx1@2t`**, with no variation at all. One roborefer call, then
an answer. On these the policy is a thin wrapper around RoboRefer, which is
worth remembering when reading the scores -- the plan's Table 2 excerpt has
RoboRefer-8B-SFT alone at 48.37 on RefSpatial against SpaceTools-3B's 53.07.

**DepthPro loads on first call, not at actor init.** Its card sits at 9.3 GB
when a benchmark never calls depth (`reflocation`, `refplacement`,
`robospatial`) and 25.7 GB once it does -- even `cvb2drelation`'s three calls
are enough to bring both actors up. That is the whole variance in the third
column of the memory table, and 25.7 GB is its ceiling.
