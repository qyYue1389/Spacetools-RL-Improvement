# P7 GFlowRL (C′) checkpoint eval report

Subject: `global_step_85` of `qzpm55555/spacetools-p7-gflowrl-cprime-8xa40`
Executed: 2026-09-17 ~ 09-18 · RunPod 4× RTX A6000
Companions: `05_gflowrl_training/gflowrl_training_report_8xA40.md` (training side), `03_sft_eval/sft_eval_report.md` (starting point), `P4→P5 handoff document (not included)` (official checkpoint comparison)

---

## Summary

After replacing GRPO in SpaceTools Step 4 with GFlowRL's **C′** objective and training 85 steps, evaluated on the nine keys of the paper's Table 2:
**2121 samples, zero OOM, zero truncation, zero turn-limit hits, zero missing `<answer>`**.

Conclusion in one sentence: **wherever a before/after comparison is possible, nothing is better than the SFT starting point, and nothing is worse.**

```
RoboSpatial Overall   SFT 61.00 (two runs 216/211)  ->  C′ 62.14 (two runs 215/220)
                      4 questions apart, about 1.5 standard errors, not a statistically meaningful improvement
                      the gap to the official checkpoint 65.43–66.00 (about 11 questions) was not closed at all
RefSpatial three      52.58 -> 52.68 simple mean; weighted is 53.07 on both sides
other five benchmarks no SFT starting point to compare against (only four keys were evaluated back then),
                      on par with or slightly better than the official checkpoint; mainly proves the environment reproduction is correct
```

---

## 1. What is evaluated, and how

### 1.1 What is evaluated is a system, not a model

The policy model (Qwen2.5-VL-3B, this time C′'s step85) is served by sglang; the seven tools are Ray actors,
**each running in its own conda environment** (the dependencies of the five environments conflict: torch 2.3.1 / 2.5.1 / 2.9.1,
numpy 1.26.4 and 2.4.6 side by side). The model looks at an image + spatial reasoning question, `<think>` → calls tools →
`<tool_response>` injected back → continues reasoning → `<answer>`, at most 8 turns.

Decoding is **greedy** (verl's `val_kwargs` default `temperature 0 / top_p 1.0 / n 1 /
do_sample False`; `rollout.n=5` in the script is the group size used for training and does not apply to evaluation).

### 1.2 Nine keys

| key | n | Metric |
|---|--:|---|
| robospatial | 350 | Sample-weighted Yes/No accuracy (228) + point convex-hull membership (122) |
| reflocation / refplacement / refunseen | 100 / 100 / 77 | Point convex-hull membership |
| blinkdepth | 124 | Option accuracy |
| cvb2drelation / cvb3ddepth | 650 / 600 | Option accuracy |
| boppose | 60 | **IoU** of two convex hulls of 8 projected corner points; a continuous value, not accuracy |
| bopgrasp | 60 | Composite score (the repo script can split it into MACE / SR) |

### 1.3 Three runs

```
Run A  2026-09-17 23:40:25 -> 09-18 01:45:27  2 h 05 m   nine keys
       of which five (robospatial / reflocation / refplacement / refunseen / cvb2drelation)
       had zero OOM, results valid; the other four were contaminated by a GPU memory problem, results discarded (see §4.2), this report does not record their scores
Run B  09-18 03:06:46 -> 03:53:18             46 m       rerun of the four contaminated ones, zero OOM
Run C  09-18 04:16:51 -> 04:36:36             19 m 45 s  robospatial second run, zero OOM
```

The order of criteria is fixed by the project and **cannot be reversed**: **first count OOM → then check silent tool errors → only then look at scores**.
Reason in §4.2: OOM does not crash the script, it only silently lowers the score.

---

## 2. How the environment and GPUs were configured

### 2.1 Machine

```
GPU        4× NVIDIA RTX A6000 · sm_86 · driver 580.159.03
           GPU0 / GPU1   49140 MiB   ECC Disabled
           GPU2 / GPU3   46068 MiB   ECC Enabled      <- mixed ECC on the same machine, see §4.4
OS         Ubuntu 24.04.3 LTS · glibc 2.39
CPU/RAM    96 cores / 503 GB
Disk       container disk 120 G (62 G used) · /workspace network volume
```

### 2.2 Software stack (restored to /opt from the 22 GB environment package)

```
torch 2.9.1+cu128 (cuda 12.8) · transformers 4.57.1 · ray 2.47.1 · numpy 1.26.4
sglang 0.5.6 · verl 0.8.0.dev
conda envs: spacetools-rl / -tool-vlm / -tool-roborefer / -tool-bbox / -tool-graspgen
```

The environment package `qzpm55555/spacetools-eval-env` is unpacked to `/opt/conda-st` + `/opt/spacetools`
(absolute paths are baked into the conda environments, so the location cannot change); weights are pulled at the pinned revisions.

### 2.3 How the GPUs are split

`NUM_GPUS=4 EVAL_GPUS=1` — of the four GPUs, **one whole GPU goes to the policy, three to the tools**.

Logical reservations of the tools (Ray `num_gpus`, **not a GPU memory quota**):

```
roborefer        1 actor × 0.6      RoboRefer-8B
vlm              1 actor × 1.0      Molmo-7B-D        <- changed from 0.6 to 1.0 this time, see §3.3
sam2             2 actor × 0.2
depth_estimator  2 actor × 0.2      DepthPro
bounding_box     2 actor × 0.1
vision_ops       1 actor × 0
grasp_generator  1 actor × 0.1
                        total 2.7  +  policy whole GPU 1.0  =  3.7  ≤  4.0
```

Measured placement (peaks from Run C's 60-second sampling trace):

```
GPU0  20.5 GB   roborefer + several small actors
GPU1  30.9 GB   Molmo alone                 <- where the change took effect
GPU2   8.6 GB   rest of depth / sam2 / bbox / grasp
GPU3  41.7 GB   policy (verl FSDP + sglang)
```

**The policy side needs a complete GPU, not a fraction.** When it cannot get a whole GPU, verl throws
`ValueError: Total available GPUs X is less than total desired GPUs 1` in
`resource_pool_manager.create_resource_pool()`.

---

## 3. Which parameters were changed, and why

Three changes in total relative to upstream `run_eval.sh`, all with as-run backups (`run_eval.sh.orig*`).

### 3.1 The nine path lines of `BENCHMARKS`

```diff
-    [robospatial]="robospatial_home_multiturn/test.parquet"
-    [reflocation]="refspatial_bench/location.parquet"
-    ... nine lines in total
+    [robospatial]="data/robospatial.parquet"
+    [reflocation]="data/reflocation.parquet"
+    ... nine lines in total
```

**Why:** the upstream script expects a nested directory structure, while the dataset
`siyich/spacetools-eval-benchmarks`, whether at `main` or the pinned `1d539ac9`, **is flat**:
`data/<key>.parquet`. Upstream changed the dataset structure and the script did not follow. Without the change, every run gives
`Missing: .../robospatial_home_multiturn/test.parquet` + `exit 1`, and not a single sample runs.

**Effect on scores:** none. What changed is the file path; the parquet read is the same one.

### 3.2 `gpu_memory_utilization` 0.5 → 0.545

**Why:** this knob is **a fraction of the whole GPU, not an absolute value**. The SFT baseline (RoboSpatial 61.00 ± 0.77)
was measured with 48 GB GPU × 0.5 = **24 GB KV pool**. The smallest GPU on this machine is 46068 MiB
(the two with ECC on); to keep the pool at 24 GB it has to be back-computed: `24 / 44 = 0.545`.

**Why alignment is mandatory:** pool size changes batch composition → floating-point reduction order → **near-tie samples flip**.
Without alignment it is not the same measurement, and the difference against the SFT starting point would mix in a pure machine term.

**Effect on scores:** yes, and that is exactly the purpose — to bring it back to the same condition as the baseline. Results must be reported with this value.

### 3.3 `vlm`'s `num_gpus` 0.6 → 1.0

```diff
-    'vlm': {'num_actors': 1, 'resources': {'num_gpus': 0.6}, ...}
+    'vlm': {'num_actors': 1, 'resources': {'num_gpus': 1.0}, ...}
```

**Why:** the declared value badly mismatches actual GPU memory. `run_eval.sh` passes vlm
`'dtype': 'float16'`, but upstream `vlm.py:116` hardcodes `torch_dtype="auto"`, swallowing this config,
so Molmo-7B-D **loads in fp32 and was measured at 30.2 GiB** — while it declares only 0.6.

Ray's `num_gpus` only guarantees "the logical shares on the same GPU sum to no more than 1.0"; **it does not partition GPU memory**.
So Ray, completely legitimately, puts `vlm(0.6) + depth_estimator(0.2) + sam2(0.2) = 1.0` on the same GPU:

```
30.20 + 9.15 + 7.97 = 47.3 GiB / whole GPU 47.4 GiB       only 70 MiB left
tool asks for another 576 MiB -> torch.OutOfMemoryError
```

After changing to 1.0, Molmo has a GPU to itself; tool demand goes 2.3 → 2.7, plus the policy's 1.0 still ≤ 4.0.

**Effect on scores:** does not enter the scores. It only changes Ray's placement, not any model, scoring function or tool behavior;
the tools are deterministic for a given input (the SFT eval verified on 350 samples "zero cases of the same query returning a different result").
What it changes is **whether the run can finish without silently losing points**.

### 3.4 Two things not changed but that must be given explicitly

```
NUM_GPUS=4 EVAL_GPUS=1          upstream defaults are 8 / 4, written for 8-GPU nodes.
                                on a 4-GPU machine, without overriding EVAL_GPUS, tools 2.7 + the 4 verl wants = 6.7 > 4.0,
                                Ray reports no error, actors queue forever
BASH_ENV=/opt/conda-st/etc/profile.d/conda.sh
                                `conda activate` is a shell function and does not cross subprocesses. Non-interactive bash
                                automatically sources BASH_ENV, so the function exists in subshells. No script change needed
```

**Deliberately not changed:** the `vlm.py:116` fp32 problem itself. The paper's results were produced under that behavior;
changing it would make them incomparable. We only change Ray's reservation so it fits, not the precision it loads in.

---

## 4. Run health

### 4.1 Gate (only look at scores if all zero)

```
                      n    OOM  truncation  turn-limit  missing<answer>  tool-failure samples
robospatial(A)      350     0    0      0        0          0
robospatial(C)      350     0    0      0        0          0
reflocation         100     0    0      0        0          0
refplacement        100     0    0      0        0          0
refunseen            77     0    0      0        0          0
cvb2drelation       650     0    0      0        0          0
blinkdepth          124     0    0      0        0          1
cvb3ddepth          600     0    0      0        0          0
boppose              60     0    0      0        0          0
bopgrasp             60     0    0      0        0         43   <- domain result, see §4.5
```

OOM = 0 is corroborated by three independent definitions: main log `grep -c OutOfMemoryError` = 0,
each benchmark's own `eval.log` = 0, per-sample `OOM samples` from `parse_dump.py --strict` = 0;
even the literal string `CUDA out of memory` appears 0 times.

### 4.2 The batch of discarded results (only the method recorded, not the numbers)

In Run A, the four keys blinkdepth / cvb3ddepth / boppose / bopgrasp hit the GPU memory mismatch from §3.3.
**The symptom is not a crash, it is silently losing points**: after a tool OOMs, Toolshed wraps the error as a normal `ToolResult`
and returns it to the model (the caller gets no exception), the model keeps reasoning with the error string, the eval finishes, reports no error,
and gives a plausible-looking number.

Error attribution quantifies this very clearly:

```
cvb3ddepth   contaminated run 243 wrong -> 230 attributed to OOM, only 13 genuinely wrong
             rerun             21 wrong ->   0 attributed to OOM, all 21 genuinely wrong
blinkdepth   contaminated run  25 wrong ->  19 attributed to OOM
             rerun             15 wrong ->   0
```

One more downstream effect worth recording: the contaminated run's log has 590 lines of
`TypeError: Expected PIL Image, got <class 'str'>` — that is a **second-order consequence of OOM**:
the tool returns an error string → the model feeds that string as a `$variable` to the next tool → TypeError.
After the rerun this dropped to 4 lines (and those 4 are the model passing wrong arguments itself, concentrated in 1 sample, which still ended up answered correctly).

**All of these numbers are discarded and not cited in this report.** Their dumps are kept because they are a physical specimen of "what silent contamination looks like",
useful for diagnosing similar failures later.

### 4.3 Tool-call histogram

```
robospatial(A)   {roborefer: 576}                                      2.00 turns/sample
robospatial(C)   {roborefer: 575}                                      2.00
reflocation      {roborefer: 102}                                      2.01
refplacement     {roborefer: 101}                                      2.01
refunseen        {roborefer:  77}                                      2.00
cvb2drelation    {roborefer: 1306}                                     2.02
blinkdepth       {vision_ops 253, roborefer 247, depth 125, vlm 33}    3.16
cvb3ddepth       {roborefer 1206, vision_ops 1202, depth 600}          3.01
boppose          {roborefer 61, sam2 60, depth 60, bounding_box 60}    5.02
bopgrasp         {roborefer 60, sam2 60, depth 60, grasp_generator 60} 4.90
```

Two things can be read directly from this table:

1. **The five RoboSpatial / RefSpatial keys call only roborefer throughout**; `depth_estimator`
   is never called once. In the SFT eval back then it was called 1 time in 350 samples. In other words,
   **after C′ training, the behavior "go check depth when depth should be checked" was not amplified; instead it went to zero.**
2. Only the four blinkdepth / cvb3ddepth / boppose / bopgrasp really use the full tool set —
   which also explains why the §3.3 GPU memory mismatch only blew up on them, while the four SFT keys never even touch it.

### 4.4 An uncontrolled variable: mixed ECC

On this machine GPU0/1 are 49140 MiB (ECC off), GPU2/3 are 46068 MiB (ECC on), a difference of 3072 MiB.
And `gpu_memory_utilization` is a fraction of the whole GPU — **depending on which GPU the policy lands on, the KV pool differs by about 1.7 GB**.

Run C's GPU memory trace shows the policy landed on GPU3 (the 46068 MiB one, pool ≈ 25.1 GB); Run A did not record
a trace at the time, so which GPU it landed on is unknown. This is a real confound in the §5.2 conclusion and has to be written down.

**What to do next time:** start 1 Hz sampling during eval, and explicitly pin the policy to a GPU.

### 4.5 bopgrasp's 43/60

`grasp_generator` returns `No collision-free grasps found` (32) or
`Top-down filtering removed all grasps` (11) on 43 samples. **This is a domain result, not a failure** — the tool ran normally
and told the model that, given the depth and segmentation it received, no feasible grasp exists in this scene. P4's corrected number under the same definition is **41/60**;
the two sides agree, so it is not a problem of this run.

Incidentally reproduced an upstream defect P4 had recorded: `analysis/analyze_grasp_result.py` only matches
`No collision-free grasps`, **missing `Top-down filtering removed all`**, so it reports
32 error samples when the real count is 43.

---

## 5. Results

### 5.1 Nine keys (all from zero-OOM runs)

```
RoboSpatial   VQA      228   71.93 / 72.81   (two runs)
              Vacant   122   41.80 / 44.26
              Overall  350   61.43 / 62.86   mean 62.14
RefSpatial    Location 100   52.00
              Placement100   58.00
              Unseen    77   48.05
              three     277   52.68 simple mean / 53.07 weighted
BLINK Relative Depth   124   87.90
CV-Bench 2D Relation   650   94.62
CV-Bench 3D Depth      600   96.50
BOP-ASK Pose            60   55.73  mean IoU
BOP-ASK Grasp           60   MACE 44.79 · SR 55.00%
```

Definition: `gpu_memory_utilization=0.545` · `NUM_GPUS=4 EVAL_GPUS=1` · 4× A6000 ·
`vlm num_gpus 1.0` · greedy decoding.

### 5.2 Three-way comparison

| Table 2 row | n | SFT starting point | Official checkpoint (P4 reproduction) | **C′ step85** | Paper |
|---|--:|--:|--:|--:|--:|
| RoboSpatial VQA | 228 | 71.27 | 73.25–73.68 | **72.37** (mean of two runs) | 79.38 |
| RoboSpatial Vacant | 122 | 41.80 | 50.82–51.64 | **43.03** (mean of two runs) | 52.46 |
| **RoboSpatial Overall**‡ | 350 | **61.00 ± 0.77** | **65.43–66.00** | **62.14** (mean of two runs) | 70.00 |
| RefSpatial Location | 100 | 54.00 | — | 52.00 | — |
| RefSpatial Placement | 100 | 57.00 | — | 58.00 | — |
| RefSpatial Unseen | 77 | 46.75 | — | 48.05 | — |
| **RefSpatial three** | 277 | 52.58 / 53.07 | 53.35 / 53.79 | **52.68 / 53.07** | 53.07 |
| BLINK Relative Depth | 124 | Not evaluated | 86.29–87.90 (upper bound 90.32) | **87.90** | 90.32 |
| CV-Bench 2D Relation | 650 | Not evaluated | 94.62 | **94.62** | 94.92 |
| CV-Bench 3D Depth | 600 | Not evaluated | 96.50 | **96.50** | 96.00 |
| BOP-ASK Pose (IoU) | 60 | Not evaluated | 53.36 | **55.73** | 34.37※ |
| BOP-ASK Grasp MACE | 60 | Not evaluated | 43.07–46.17 | **44.79** | 43.06 |
| BOP-ASK Grasp SR | 60 | Not evaluated | 55.00–56.67 | **55.00** | 50.00 |

‡ Overall is the sample-weighted VQA/Vacant, not an independent measurement.
※ The paper's 34.37 is not a proportion at n=60 (60 × 0.3437 is not an integer); P4 already judged it as **metric mapping unresolved**;
do not read it as a gap.
The two RefSpatial numbers are "simple mean / sample-weighted". That SFT round evaluated only four keys, so the last five rows have no starting point.

**cvb2drelation 94.62 and cvb3ddepth 96.50 are bit-for-bit identical to P4's official checkpoint** — this is the strongest piece of evidence that this environment
reproduction is correct (same tools, same scoring function, same data).

### 5.3 RoboSpatial two runs

```
              n     Run A           Run C
  VQA       228   164  71.93%    166  72.81%
  Vacant    122    51  41.80%     54  44.26%
  Overall   350   215  61.43%    220  62.86%     mean 217.5 = 62.14%

  always correct 205 · always wrong 120 · flips 25       possible single-run range 205 ~ 230
  samples whose two generations differ verbatim 199/350
```

Compare the SFT baseline: `216 / 211`, always correct 199, upper bound 228, expected 213.5 (61.00%), 29 flips.

**4 questions apart ≈ 1.5 standard errors, not a statistically meaningful improvement:**

```
C′  flips 25 -> single-run sd ≈ 2.5 questions, two-run mean sd ≈ 1.8 questions
SFT flips 29 -> two-run mean sd ≈ 1.9 questions
pooled sd ≈ 2.6 questions, diff 4.0 questions -> about 1.5 σ, two-sided p ≈ 0.12
the two ranges 205–230 and 199–228 overlap almost completely
```

Moreover, **SFT's 61.00 was measured on a different machine**; SFT was not re-measured on this machine this time, and on top of the mixed ECC in §4.4,
for this 4-question difference **the machine-level explanation and the model-level explanation cannot currently be separated**. There is only one clean way
to pin it down: run robospatial with the SFT ckpt too, same machine, same session.

### 5.4 VQA confusion matrix: three models fail in the same place

```
                        gt_no accuracy     gt_yes accuracy
SFT starting point        49.21%           80.61%
official ckpt (after RL)  47.62%           83.03%
C′ step85  Run A          50.79%           80.00%
C′ step85  Run C          50.79%           81.21%      <- gt_no bit-for-bit identical in both runs 32/63
```

**On questions where the answer should be "no", all three models are flipping a coin.** The negative expectation the SFT report wrote down back then
("the coin-flip level on `gt_no` still exists after the paper's RL; do not list it as an expected gain of RL")
is confirmed here for the third time — **C′ did not fix it either**. The few points gained between the two runs all come from `gt_yes`
and Vacant samples near the boundary.

---

## 6. Conclusion

**1. C′ produced no measurable improvement wherever a before/after comparison is possible.** RoboSpatial differs by 4 questions (1.5 σ),
RefSpatial weighted is bit-for-bit identical, Vacant is exactly the same as the SFT starting point. Of the roughly 11 questions the official checkpoint leads by,
not a single one was caught up.

**2. This is self-consistent with the training-side metrics, not a surprise.** Three points from the training report:

- Reward-degenerate groups make up **63–80%** (`reward/degenerate_group_frac` median 0.703). When a whole group has the same score
  the reward term is always 0 and the gradient comes entirely from the drift term — at that point C′ is doing flow matching toward `π_ref·exp(βr)/Z`,
  **not learning the reward**.
- `actor/grad_norm` is **103–5229** throughout, while `grad_clip=1.0`: every update is clipped down to direction only,
  with no magnitude.
- 1 epoch / 85 steps.

**A policy that receives no reward signal on about 70% of samples, and whose every update keeps only the direction, being indistinguishable from its starting point after 85 steps,
is an expected result.**

**3. Tool-use behavior was not amplified; it actually narrowed.** The SFT starting point called `depth_estimator`
1 time in robospatial's 350 samples (and got that one right); this time it is **0 times**. The SFT report identified "not calling a tool when it should"
as the class of problem RL should fix and SFT cannot; on this item C′ gives a reading in the opposite direction.

**4. The other five benchmarks cannot say whether C′ is good or bad, because there is no starting point.** They are on par with or slightly better than the official checkpoint;
their value is in proving this eval environment reproduction is correct (the two CV-Bench numbers are bit-for-bit identical to P4).

---

## 7. Meaningful problems (related to this result, and that will come up again on another machine)

### 7.1 `num_gpus` is a logical reservation, not a GPU memory quota — and vlm's declaration was off by 5×

Detailed in §3.3. Repeating the key points, because it is the only problem this time that actually ruined results:

- Ray only guarantees logical shares on the same GPU ≤ 1.0; **it does not partition GPU memory**;
- `vlm.py:116`'s `torch_dtype="auto"` swallows `dtype: float16`, Molmo loads in fp32 using 30.2 GiB,
  yet declares only 0.6;
- so Ray legitimately crams 30.2 + 9.15 + 7.97 onto a 47.4 GiB GPU;
- **OOM does not crash, it only silently loses points.**

It only triggers on benchmarks that really use the full tool set, so the four SFT eval keys never touch it —
SFT report §6.2 had actually already recorded this dangerous placement (`GPU1 46.3 GiB / 48 GiB, util 0%`),
it is just that nobody on that GPU asked for more memory, so it sat there quietly full and nothing happened.

### 7.2 It finished, but the exit code is 15

After `run_eval.sh` prints `=== EVALUATION COMPLETE ===`, its own cleanup trap
`kill`s the background toolshed process, `wait` carries out the SIGTERM, and the script's exit code is 15.

**Judging death by exit code would discard a batch of completely valid results.** All three runs had exit code 15, and all three finished.
The correct criterion is **success marker + sample count + OOM count + router-lost count**, not `$?`.
(This and SFT report §6.6 "the acceptance-check script itself can also give a false green" are two sides of the same class of problem:
one fails yet exits 0, the other succeeds yet exits 15.)

### 7.3 `run_eval.sh` has no concurrency guard; two instances kill each other

The launch command was retried once by an upper layer, which started two `run_eval.sh` at the same time. Both run
`ray stop --force` at the start, so the later one kills the earlier one's toolshed together with all tool actors
(`start_toolkit(detached=False)`: when the parent process dies, all actors die).

**The symptom is extremely hidden**: the earlier eval **keeps running**, except that from then on every tool call gets
`Could not find ToolRouterActor 'toolshed_router'`, and it still produces scores.

**Countermeasure:** add an atomic lock at the start of the launch script (`mkdir /root/.lock || exit 0`), and add the count of
`Could not find ToolRouterActor` to the gate. Both have been added this time.

### 7.4 Mixed ECC on the same machine leaves the KV pool uncontrolled

§4.4. `gpu_memory_utilization` is a fraction of the whole GPU, while on this machine the usable GPU memory differs by 3072 MiB between pairs of GPUs;
depending on which GPU the policy lands on, the pool differs by about 1.7 GB — and pool size is exactly the knob that flips near-tie samples.
**To compare with a baseline from another machine, pin the policy to a GPU, or at least record which GPU it landed on.**

### 7.5 The BENCHMARKS mapping in upstream `run_eval.sh` does not match the dataset's actual layout

§3.1. This is the "first run on a new machine always exits 1" problem, unrelated to the checkpoint.

### 7.6 `conda activate` fails in non-interactive subprocesses

`run_eval.sh:84` calls `conda activate spacetools-rl`, but `conda` is a shell function that does not cross subprocesses,
reporting `CondaError: Run 'conda init' before 'conda activate'`.
The fix is `BASH_ENV=/opt/conda-st/etc/profile.d/conda.sh`; no script change needed.

### 7.7 Our own `parse_dump.py` takes files, not directories

The gate step of `EVAL_FROM_SCRATCH.sh` passes a directory, while `parse_dump.py` wants the
`0.jsonl` file path, so it prints `not a file` and reports `CONTAMINATED OR UNREADABLE`.
This time the gate was rerun by hand with file paths. **This is a defect that gives the illusion "the gate ran"; it needs fixing.**

### 7.8 `analyze_grasp_result.py` undercounts one class of grasp failure

§4.5. Upstream only matches `No collision-free grasps`, missing `Top-down filtering removed all`,
reporting 43 as 32. P4 had recorded it; reproduced this time.

---

## 8. Appendix: artifact list and what they can answer

```
dumps/<run>/<benchmark>/0.jsonl    one sample per line, correct and wrong both included. Fields identical one-for-one with the P4 batch:
                                   input (~10 KB full prompt), output (**full multi-turn trajectory,
                                   including real tool returns**), gts, answer, acc/score/reward/fmt, index
logs/<run>/<benchmark>/eval.log    tool-side view: actor pid, tool-internal retries, verbatim OOM text
logs/*.log                         main logs of the three runs (with set -x throughout)
gates/                             per-benchmark output of parse_dump.py --strict
scripts/                           as-run launch scripts + as-run and original versions of upstream run_eval.sh
gpu/                               Run C's 60-second GPU memory trace
```

**The dump and eval.log answer different questions** (the lesson of P4's "96 vs 16"): the log records the tool's internal
multiple attempts, the dump records what happened at the sample level. Keep both.

The `image` field is mostly null; to get the image go through `sample_id → parquet`; the benchmark parquet is not shipped with the package
(292 MB); a pinned version is on HF at `siyich/spacetools-eval-benchmarks @ 1d539ac9`.

**To generate P4's enriched `parsed/` records** (`vars_exposed` / `vars_used` /
`vars_unused` — directly corresponding to P6 taxonomy 2d "not reusing variables", the extracted `question`,
`num_turns_verl_convention`), just run `spacetools-repro`'s parser on these dumps;
**no GPU needed**.
