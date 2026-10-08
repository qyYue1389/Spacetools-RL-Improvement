# P6 / P7 GPU session results

> Machine: 4×A100-SXM4-80GB (sm_80), driver 570.195.03, CUDA 12.8, gcc 13.3.
> Data was copied over from the old 4×A100-SXM4-40GB; paths stay unchanged under `/workspace/`.
> Started 2026-08-29; the record is appended to as experiments progress.
> Handoff doc: `records/P6_GPU_HANDOFF.md`.

## Status overview

| Experiment | Status | Conclusion |
|---|---|---|
| go/no-go migration acceptance check | **Passed** | See §1. Step 4 was judged a pass after diagnosis, and it surfaced deviation `[23]` |
| A. pointing tool comparison | **Done** | Molmo is 14.61 pp worse overall; the real finding is complementarity, +10.83 pp (oracle) |
| B. depth tool comparison | **Half done** | Regression check passed; **the tool-swap half was not done** — there is only one depth tool |
| F. fp32 control | **Done, conclusion retracted** | A single run of 107 was once judged "ruled out"; two extra runs gave 105/105. See §4 |
| G. robospatial resampling | **Done, hypothesis overturned** | The upper bound is not saturated; "the paper reports the upper bound" does not hold. See §5b |
| C. pass@k | **Done** | Ruled out the "decoding method" candidate for Pose; 43.9% of VQA questions swing between samples. See §6 |
| D. P7 feasibility estimate | **Done (analytical estimate)** | Not enough GPUs, short by about 8×; the paper's configuration is not realistic. See §7 |

---

## 1. Migration acceptance check

The first three steps passed outright: architecture sm_80 (nothing needs recompiling), all five conda environments usable,
the sm_80 cubin of `pointnet2_ops` **actually ran successfully** on this GPU (stronger evidence than `cuobjdump`),
seven-tool end-to-end `12 checks: 12 passed, 0 warned, 0 failed`.

Weights complete: Molmo-7B-D 30 GB (fp32), all 7 shards present, no broken symlinks, no `.incomplete`
— the most common failure point when copying an HF cache across machines did not happen. Volume usage 141 GB, matching the handoff doc.

### Step 4: 106/124, outside the 107–109 band in the doc

    run1 40G bf16 gmu=.5   107      run5 80G bf16 gmu=.25  108
    run2 40G bf16 gmu=.5   109      run6 80G bf16 gmu=.25  109
    run4 80G bf16 gmu=.5   106      run7 80G FP32 gmu=.25  107

`--strict` itself passed, all six health metrics zero, mean turns 3.13, same as recorded in P4.

**Judged a pass, based on the stable upper bound rather than a single run landing in the band.** Six runs merged: always correct 99 · always wrong 12 ·
flip 13 · **upper bound 112/124 = 90.32%**, exactly the same as the paper and as A6000 / A100-40G.
The 12 "always wrong" hard core samples reproduce in every single run.

> **Read the upper bound by number of samples drawn, not by platform.** 112 only appears when six runs across platforms are merged;
> the four 80 GB runs alone give 111/124, and the two 40 GB runs still available locally also give 111/124.
> The narrow 107–109 band in the doc comes from n=3 and is a fragile criterion — the actual range over five bf16 runs is 106–109.

### Deviation [23], surfaced by this

The GPU memory peaks of the three tool GPUs are **byte-for-byte identical** to the old machine (Molmo 34.6, DepthPro GPU 25.7);
only the policy GPU changed: **25.6 → 46.8 GB**. `gpu_memory_utilization` is a static fraction of the **whole GPU**;
0.5 is a 20 GB pool on 40 GB and a 40 GB pool on 80 GB.

Pool size changes batch composition → floating-point reduction order → near-tie samples flip;
this is exactly the one mechanism of run-to-run variation this project has already established. **So 80 GB is not a numerically neutral substitution on the policy side.**
Handoff doc §1.5 says the knob is "just more relaxed" — correct for "does it fit", but it misses the numerical consequences.

Decisive experiment (gmu 0.5→0.25, policy GPU peak back to 30.4 GB): two runs gave 108 and 109, both in the band;
the three samples run4 broke (32, 72, 100) all came back. **But causality is not established**: n=1 on the 0.5 side.
A harder configuration-dependent signal is that **sample 69 hits the 8-turn cap in both 0.25 runs and does not under 0.5**
— a score difference of one or two can be sampling, but the same sample repeatedly hitting the turn cap under the same configuration is systematic.

This also shows that **0.25 cannot restore 80 GB to 40 GB**: the full P4 run of 2121 samples had `hit_max_turns=0`,
while under 0.25 the blinkdepth benchmark alone has 1–2. Bit-for-bit reproduction never existed in the first place; what reproduces is the upper bound.

**Decision: policy runs on 80 GB use `gpu_memory_utilization=0.25`**, the reason being to reproduce P4's
pool size, not "0.5 has been shown to be worse". `EVAL_GPUS` stays unchanged (the warning in doc §1.5 holds).

---

## 2. Experiment A: pointing tool comparison

### Regression check (must pass first, otherwise no difference is interpreted)

    benchmark        n   roborefer now    P4 baseline    diff
    reflocation     98   55 (56.12%)     53 (54.08%)    +2
    refplacement   100   59 (59.00%)     58 (58.00%)    +1
    refunseen       77   37 (48.05%)     37 (48.05%)     0
    robospatial    122   60 (49.18%)     62 (50.82%)    -2
    total          397  211 (53.15%)    210 (52.90%)    +1

Max per-benchmark difference is 2 samples, in both directions, within the doc's tolerance of "a difference of one or two is normal". **Passed.**
(**But the reason it passed is not the one the doc gives** — see "But this overturns the reason the regression check passed" below.)

> The baseline is the per-item correctness of **our own P4 run** (the probes' `baseline_correct`),
> not the paper. The regression check has to answer "does the offline path reproduce the full pipeline",
> not "does it reproduce the paper"; using the paper as baseline would mix path bugs with the reproduction gap.
>
> The baseline table in doc §3 writes `reflocation` as 55.10%, which is inconsistent with its own total of 52.90% (210/397):
> 53+58+37+62 = 210, so that row should be 53/98 = 54.08%. **The probe file is correct;
> that row of the doc is a typo.**

The only `tool failure 1` (reflocation#6) is not a failure: RoboRefer legitimately returned
`Detected 0 instance(s)`, and that sample's baseline is `None`/incorrect anyway, consistent with P4.

### Molmo comparison

    benchmark        n   Molmo         RoboRefer      diff
    reflocation     98   57 (58.16%)   53 (54.08%)    +4.08
    refplacement   100   37 (37.00%)   58 (58.00%)   -21.00
    refunseen       77   28 (36.36%)   37 (48.05%)   -11.69
    robospatial    122   30 (24.59%)   62 (50.82%)   -26.23
    total          397  152 (38.29%)  211 (53.15%)   -14.61

Tool failures 0 — Molmo gave a point for every item, no refusals.

**The +4.08 on `reflocation` is noise, not a finding**: McNemar discordant pairs 18 vs 16, p=0.86.
The other three are significantly worse (refplacement p=0.0002, robospatial p<0.0001, total p=1.05e-06).

> Not comparable with the paper: in the paper's Table 2, Molmo-7B alone scores 0.00 on RefSpatial; here it is 36–58%.
> The paper measures end-to-end on the full task; here only the pointing step is replaced and the rest of the flow is unchanged.
> That is exactly the value of this offline path — it separates "is the tool good" from "can the policy use the tool".

### The real finding: the two tools are highly complementary

    accuracy  RoboRefer 53.15%  Molmo 38.29%  intersection 27.46%
    union (per-sample oracle tool choice) 63.98%   vs RoboRefer +10.83 pp (43 samples)

    only Molmo correct / only RoboRefer correct
      reflocation   18 / 16      <- the two tools disagree on over a third of samples; it just happens to cancel out
      refplacement   6 / 28
      refunseen      9 / 18
      robospatial   10 / 40

**Even though Molmo is 14.61 pp worse overall, it still got 43 samples right that RoboRefer got wrong.**
This changes the direction of P6's offline attribution recommendation: the original conclusion was "RoboRefer alone accounts for 53.4% of wrong answers, swap the pointing tool first";
measured, **swapping it out wholesale is worse**, but **per-sample selection gives +10.83 pp**. The action should be "add" rather than "replace"
— router/ensemble, or bring in a stronger third tool.

**This is an oracle upper bound**: it assumes we know which tool is right (which requires the answer). Same kind of caveat as the doc's warning about the 33 pp
for grasp: a biased upper bound, not an achievable gain. But a magnitude of 43 samples says "choosing between tools" deserves its own experiment.

### Follow-up: +10.83 pp is real complementarity, not "two draws beat one"

**Union and upper bound are the same structure** (the lesson from experiment G). The union above was computed from **one run each** of the two tools,
so the 43 "only Molmo correct" could include noise from "two draws beat one anyway".
One extra RoboRefer run, with its self-union as the reference:

    RoboRefer run1        211/397        RoboRefer self-union   211  <- equals a single run
    RoboRefer run2        211/397        RoboRefer ∪ Molmo  254
    samples with different scores across runs      0
    samples with different returned points         0

    self-union gain (pure noise)  0 samples  +0.00 pp
    Molmo's true margin          43 samples  +10.83 pp   <- same as the old definition

**RoboRefer is bit-for-bit deterministic on this path**; the self-union adds nothing, so +10.83 pp stands as is,
and it is now measured rather than assumed.

### But this overturns the reason the regression check passed

Earlier, the regression check's +2/+1/0/−2 was attributed to "RoboRefer's own slight non-determinism"
(handoff doc §3 says the same: "RoboRefer itself has slight non-determinism, a difference of one or two samples is normal").
**Two bit-for-bit identical runs prove that explanation wrong.**

The real cause is **cross-hardware drift**:

    of 397 items, returned point differs from P4 baseline  226 (57.1%)
    displacement (normalized coords)  median 0.0040 · mean 0.0127 · max 0.3506
                       < 0.005: 137/226 · < 0.02: 207/226

    per benchmark  reflocation 49/98 · refplacement 73/100 · refunseen 52/77 · robospatial 52/122

Three pieces of evidence pin down the cause: the `obj_name` of the five differing samples is **verbatim identical** to the query string the policy actually issued in the P4 dump;
this session's two runs agree bit-for-bit on 397/397; and compared with P4, **57% of the points moved slightly**.
That is, **RoboRefer is deterministic on a given hardware + software stack; on different hardware its output shifts slightly across the board** —
the same kind of mechanism as the policy-side KV pool (deviation `[23]`), only happening inside the tool.

More than half the samples' points changed; the vast majority of displacements stay on the same object and do not change correctness;
**only 11 happen to cross the decision boundary**. They are not special cases; they are the visible tip of the 226 drifted items.

> **The regression check still passes, but the reason has to change**: not "noise within tolerance", but
> "of 226 systematic small shifts, 11 crossed the decision boundary, and after the directions cancel the net difference is 1 sample".
> The doc's threshold "a difference of ten means the path is wrong" is still valid (a net difference of 1 is far within it),
> but it hides the 57% drift underneath. **Systematic differences do not go away by running more times.**

### Three more: same point, opposite verdict

The returned points for `robospatial` #9, #22, #70 are **bit-for-bit identical** to the baseline, yet the correctness is reversed. Unrelated to drift;
the cause is the baseline definition — verified item by item that none of the three is **pure pass-through**:

    #9   tool (0.767, 0.743) → policy answer (0.87, 0.77)    baseline correct · offline wrong
    #22  tool (0.149, 0.583) → policy answer (0.05, 0.58)    baseline correct · offline wrong
    #70  tool (0.5,   0.908) → policy answer (0.483, 0.644)  baseline wrong · offline correct

The probe file's `baseline_correct` records the correctness of the **policy's final answer**, while the offline path uses the
**raw tool point**. P6 offline attribution recorded that robospatial Vacant is only 82.8% pure pass-through
(21 of 122 items where the model changed the point itself); these three fall within those 21.

> RefSpatial does not have this problem (276/276 pure pass-through), so the +3 there comes entirely from drift.
> But it means **the "baseline" in the robospatial column and the offline path do not measure the same quantity**:
> the former includes the policy's point-override behavior, the latter does not. This column must be flagged separately in tool comparisons.

---

## 3. Experiment B: depth tool

### Regression check: passed, replay column matches exactly

    benchmark      n   DepthPro rerun    baseline replay   diff
    blinkdepth   115   96  (83.48%)     97  (84.35%)   -1
    cvb3ddepth   599  576  (96.16%)    577  (96.33%)   -1
    total        714  672  (94.12%)    674  (94.40%)   -2

The replay column (pure arithmetic, no tool calls) = 97/115, 577/599, 674/714, verbatim identical to the handoff doc,
which shows the probes were not touched. As the doc requires, the replay value 674 is the reference, not the actual baseline of 677.

### The "swap tool" half was not done

Toolshed v1 has only one depth tool, `depth_estimator` (DepthPro); there is no second one to swap in.
Doc §7 also already ruled out a generic oracle ablation (blinkdepth's GT is A/B, not a depth map),
so there is neither a second tool nor depth GT to use as an oracle. **Completing B requires installing a new depth model first.**

### Substitute analysis: are the wrong answers "close" or "way off"

Relative gap `|dA-dB| / min(dA,dB)` (the criterion is "pick the smaller one", so the gap is the margin):

    blinkdepth  correct median 0.618  wrong 0.103      cvb3ddepth  correct 0.875  wrong 0.146

    wrong answers with gap < 10% (near-tie, a slightly stronger tool could flip it)  blinkdepth 9/19 · cvb3ddepth 8/23
    wrong answers with gap >= 10% (confidently gave the reversed order)              blinkdepth 10/19 · cvb3ddepth 15/23

**Most depth wrong answers are not near-ties.** A "slightly better" depth tool would recover about 17 samples at most;
eating all 42 wrong answers needs a significantly stronger model. Compared with experiment A, where pointing has an upper bound of 43 samples
**and needs nothing new installed**, pointing still comes first in priority.

---

## 4. Experiment F: fp32 end-to-end control

**Result 107/124 = 86.29%, inside the expected band 107–109. `--strict` passed, health metrics all zero.**

Configuration: removed `model_dtype=bf16` (falls back to the default `fp32` at `engine/fsdp.yaml:33`),
`gpu_memory_utilization=0.25`. **The second one is required** — the criterion is that both arms differ only in dtype,
and the bf16 reference arm (run5/run6 = {108, 109}) is at 0.25.

Confirmed this run really is fp32: resolved config `'model_dtype': 'fp32'`; run7 log contains `bf16` **0** times
(run5 has 1); FlashAttention warning `current dype ... is torch.float32`;
GPU3 peak 33.8 GB vs 30.4 GB for bf16 with the same config.

Per-sample: the 14 "always wrong" from the two bf16 runs with the same config **all 14/14 reproduce** under fp32;
the 12 hard core samples from all five bf16 runs are also all wrong under fp32; 2 "always correct" broke (samples 18, 49),
same order of magnitude as run4's 3 and cross-hardware's 2–3.

**Verdict: deviation `[20]` can be struck off end to end** (`PROVENANCE.txt` and `P5_RESULTS.md` §4 updated accordingly).
Theory already predicted no difference: the ckpt is stored in BF16, `bf16→fp32→bf16` is an identity transform
(825/825 tensors verified bit-for-bit), so sglang receives the same weights under both configurations.

> **⚠ The conclusion above was retracted the same day.** See below.

### ⚠ After two extra runs: the ruling-out conclusion does not hold

    fp32 @ gmu=.25   107 (run7) · 105 (run9) · 105 (run10)    range 105–107
    bf16 @ gmu=.25   108 (run5) · 109 (run6)                  range 108–109

**The two ranges do not overlap.** run7's 107 is the **top edge** of fp32's own range;
I read it as "inside the 107–109 band" and on that basis wrote "deviation [20] ruled out end to end".
Two additional runs overturned that verdict.

**But the conclusion in the opposite direction does not hold either**; the strength of evidence, honestly stated:

- Five values {105,105,107,108,109}; if identically distributed, the probability that the two bf16 runs take exactly the top two places is
  `1/C(5,2)` = **0.1**. Strongly suggestive, **not significant**.
- Structurally weak: **only 1 sample** (#49) is "bf16 always correct, fp32 always wrong"; 0 in the reverse direction.
- fp32 itself is non-deterministic too: the three runs differ by 4–6 samples
  (three runs: always correct 102 · always wrong 14 · flip 8; bf16 two runs: 107 · 14 · 3).
- The "always wrong" hard core sets of the two intersect in 13 samples, essentially consistent.

**The correct statement is "evidence for ruling out is insufficient, retracted", not a conclusion in either direction.**

### The weight argument still holds, but its implicit assumption is now in question

The ckpt is BF16, `bf16→fp32→bf16` is the identity, 825/825 tensors verified bit-for-bit —
**sglang really does receive the same weights**; that part is not shaken.

What is shaken is its implicit premise: **the only path through which `model_dtype` affects the numbers is the weights**.
The measurements hint at a second path, **unverified**: fp32 master takes about 8 GB more
(GPU3 peak 33.8 vs 30.4), so GPU memory allocation and fragmentation differ → sglang batch composition differs →
near-tie samples flip. **Same kind of mechanism as deviation `[23]`, not a weight-level effect.**

> **Methodology: this is the second occurrence of the same mistake in this GPU session.**
> Experiment G was "concluding before the upper bound saturated"; this time it was "concluding from a single point at the edge of its own range".
> Both were overturned by "one more run", and both criteria came from this project's own docs.
> The difference is that for G I ran the extra check myself before announcing; this time I did not — it was only added after the user asked "is there anything that needs rerunning".

---

## 5. Harness defects fixed (during experiments A/B)

These scripts were newly written for this GPU session; the image-loading and scoring paths had never been called for real before,
so the bugs clustered at the first real call — **the regression check exists precisely to catch this, and it did.**

| Location | Consequence | Fix |
|---|---|---|
| `tools/p6/gpu_pointing_swap.py` `load_image` | pandas reads the parquet `list<struct>` as an ndarray with `dtype=object`; the function only accepted `list`/`tuple`, so it fell straight through to the TypeError at the end on the first hop. **397/397 image loads failed; the tool was never called once** | Added an ndarray branch |
| `tools/p6/gpu_depth_swap.py` `load_image` | Same bug | Fixed together |
| `verl/utils/reward_score/__init__.py` | In the scoring branches for `RoboSpatial/*` and `BLINK/Spatial_Relation`, `format_score_val` is only assigned inside the `if training_args` block (the other six defaults in the same block are all initialized **before** the `if`), so an offline call raises UnboundLocalError. **All 122 robospatial items failed to score** | **Upstream not changed**; instead the script passes a `training_args` that replicates the eval config |

The two values for the third one are pinned down by evidence: `format_score = 0.0` (`ppo_trainer.yaml:230`),
`allow_last_response_fallback = True` (`run_eval.sh:297` overrides the config default `false`).

> **Also measured that both are inert on this path**: the answer string the harness builds always has the `<answer>` tag,
> so the fallback never triggers; every combination of the two values gives 60/122.
> So the −2 on robospatial is not caused by the stub; it has the same origin as reflocation's +2.

Two other doc-level issues: the label `run_fp32` would be rejected by the `^run[0-9]+$` guard in `p4_run.sh`;
the GPU percentage in `tools/p4_check.sh` is hard-coded to divide by 40960, which is distorted on 80 GB (GPU3 reports 117%, actually 57%).

---

## 5b. Experiment G: robospatial additional sampling — hypothesis overturned

**Conclusion: "the paper reports the stable upper bound" does not hold. The −6.13 pp on RoboSpatial VQA remains unexplained.**

### What happened

The criterion in doc §5.5 was fixed in advance: if after the third sample the VQA upper bound climbs to ≈181/228 = 79.38%,
that means the paper reports the upper bound rather than a typical sample value, the same structure as blinkdepth.

After run3 finished, the criterion was **hit literally, and exactly**:

    3-run upper bound   VQA 181/228 · Vacant 64/122 · Overall 245/350
    paper               VQA 181     · Vacant 64     · Overall 245        diff 0 samples

Three exact matches; it looks like a neat conclusion. **But the upper bound is a monotonically increasing function of the number of samples drawn**,
so before announcing, a saturation check was added — one more run (run8).

### The fourth run overturned it

    VQA upper bound (paper 181)
      1 run:  161, 167, 168, 169
      2 runs: 174, 176, 178, 181, 183
      3 runs: 181, 183, 186          <- the four 3-run combinations; we happened to use the one that gives 181
      4 runs: 188

    Vacant upper bound (paper 64)
      3 runs: 63, 64, 65             <- 64 is just the middle of three possible values
      4 runs: 65

    Overall 4-run upper bound 253/350 = 72.29%   paper 70.00 = 245/350

The upper bound did not saturate; at four runs it already exceeds the paper by 7 samples (VQA). The 3-run "exact hit" was not only a coincidence,
**it was a coincidence of which combination was picked**: the same three runs in a different combination give 183 or 186; and among 2-run combinations one already reaches 181.

### The criterion itself is broken

The upper bound rises monotonically, **so any target value will be reached sooner or later**; "upper bound equals the paper's number" therefore cannot serve as
evidence that "the paper reports the upper bound" — it only says we ran exactly that many times. When the doc wrote this criterion
it did not require saturation, and saturation is a necessary condition, not a bonus.

> **This equally weakens the blinkdepth conclusion** (`P5_RESULTS.md` §1, §4).
> The only difference is growth speed: blinkdepth has few flip samples, upper bound 111 at 2 runs, 111 at 4, 112 at 6, so it climbs slowly;
> robospatial VQA has 43 flips, going 178→181→188 across 2→3→4 runs.
> **blinkdepth's 112/124 is likewise not proven saturated**, only closer. A note has been added in `P5_RESULTS.md`.

### The gap is back to unexplained, and more concrete

    VQA single run   161 / 167 / 168 / 169     mean 166.2/228 = 72.92%
    paper                                           181/228 = 79.38%
    diff                                       14.8 samples

    wrong in all four (hard core) 40 · correct in all four 145 · flip 43

The best of the four runs (run8, 169) is still 12 samples short of the paper. **−6.13 pp is a real gap**,
handed back to P6 manual classification. The lead recorded in `P5_RESULTS.md` §2① is still valid:
the `no` class collapses to chance level, and 105/228 are "Can X fit ⟨rel⟩ Y?"-type questions with a structural mismatch with the tool output.

### Single-run scores of the four runs

    run1 (40G, gmu=.5)   Overall 229/350  VQA 167  Vacant 62
    run2 (40G, gmu=.5)   Overall 231/350  VQA 168  Vacant 63
    run3 (80G, gmu=.25)  Overall 222/350  VQA 161  Vacant 61
    run8 (80G, gmu=.25)  Overall 231/350  VQA 169  Vacant 62

All four `--strict` passed, health metrics all zero. run3 is the lowest run,
yet the 3-run upper bound is pushed up by it — it got right samples others got wrong, not because it is stronger.

---

## 6. Experiment C: pass@k

Three benchmarks, `val_kwargs.do_sample=True / n=5 / temperature=1.0 / top_p=1.0`,
everything else as in P4 (including `model_dtype=bf16`), `gpu_memory_utilization=0.25` (deviation `[23]`).
Variant script `tools/p6/variants/run_eval_passk.sh`, five lines different from `run_eval.sh`.

The overrides really took effect: resolved config is `do_sample: True / n: 5 / temperature: 1.0`,
dump row counts 300 / 620 / 1750 = number of samples × 5, and **for no sample are all 5 output texts identical**
(boppose 0/60, blinkdepth, robospatial 0/350).

### ⚠ pass@k is unusable on binary tasks

`blinkdepth` picks A/B and `RoboSpatial VQA` answers yes/no; both are two-way choices. With 5 draws,
**pure random guessing gives pass@5 = 1 − 0.5⁵ = 96.88%**. Measured: blinkdepth 94.35%,
VQA 89.04%, **both below blind guessing** — pass@k here is almost entirely determined by the fact of "drawing 5 times"
and carries no information about model ability, **so it cannot be used to measure the headroom on the inference side**.

The meaningful substitute is **majority vote@5** (≥3/5 correct), because it is an achievable strategy rather than an oracle,
and on binary tasks its random baseline is 50% rather than 96.88%.

`RoboSpatial Vacant` is point prediction, not binary; random baseline ≈0, so **its pass@5 is meaningful**.

### Results

    benchmark            avg@1      majority vote@5   greedy reference          paper
    blinkdepth          85.00%      110/124 88.71%   107,108,109,109      112/124
    RoboSpatial VQA     71.40%      176/228 77.19%   161,167,168,169      181/228
    RoboSpatial Vacant  49.02%      —                61,62,62,63           64/122
    boppose (IoU)       53.60       —                53.36 (3 runs bit-for-bit identical)   34.37

- **blinkdepth's majority-vote gain is noise-level**: 110 vs greedy's 107–109, only 1–3 samples higher,
  while greedy's own spread is 3 samples wide. **Not a conclusion.**
- **The VQA gain is real**: 176 vs 161–169, 7–15 samples higher, clearly outside the greedy interval.
  Self-consistency pushes VQA from the greedy mean of 166.2 to 176, **closing about 60% of the −6.46 pp gap,
  with about 2.2 pp left that it cannot close**.
- **Vacant pass@5 = 70.49%** vs avg@1 49.02%, +21.48 pp; valid because it is not binary.
- **boppose uses avg@1, not best@5**: taking the max of 5 runs has selection bias; even if the score were pure noise,
  the max would exceed the mean, so `best@5 = 57.23` cannot count as a gain.

### ① BOP-ASK Pose "decoding method" candidate: ruled out

One of the four candidates in `P5_RESULTS.md §2②` is "if the paper sampled with temperature 1.0 and reported avg@1,
the score would be lower and formatting errors more likely". **Neither part holds**:

    avg@1 53.60   vs greedy 53.36 (0.24 higher)      paper 34.37
    zero-score samples 42/300 = 14%   vs greedy 9/60 = 15%

Getting to the paper's 34.37 would require roughly 18 more samples scoring 0, while sampling **lowered** the zero-score rate
by 1 percentage point. The standard error of the mean over 300 samples is about 2 points; the paper's value is 19 points away.
**Three candidates remain; "format failures on the paper's side" is still the most self-consistent.**

### ② A fourth error category for VQA: consistency

    how many of 5 correct   all correct   all wrong   split (1–4)
    blinkdepth            87      7       30  (24.2%)
    RoboSpatial VQA      103     25      100  (43.9%)
    RoboSpatial Vacant    35     36       51  (41.8%)

**43.9% of VQA questions swing between the 5 samples.** This is neither a tool error nor a reasoning error;
it is a calibration/consistency problem — **P6's existing three categories (tool error / reasoning error / annotation issue) have no slot for it**.
It is two sides of the same phenomenon as the `no` class collapsing to chance level recorded in `P5_RESULTS.md §2①`.

### ③ Experiment C does not change the accuracy to report

The paper's decoding is unknown, but the released repo's `val_kwargs` default is greedy, and P4 followed it —
this is the only definition with grounding. **Majority vote@5 is a different inference strategy, not the same measurement**;
putting 77.19% into Table 2 would compare the paper's number against a strategy the paper never used.
C's value is in explanation, not in scores.

### ④ `--strict`'s malformed criterion does not apply to sampling runs

Among robospatial's 1750 samples, 1 has a malformed `tool_call` (one draw of sample 167:
the model first emitted an isolated garbage character `槛`, then gave a well-formed call), which made `--strict` exit with 1.

Checked item by item, **it is not contamination**: all other health metrics are zero (OOM / truncation / turn reconciliation / tool failure /
turn cap), and T=1.0 sampling by definition occasionally draws low-probability tokens — the full P4 run of 2121 greedy
samples had 0 malformed, precisely because greedy does not draw low-probability tokens. Impact is zero: the other four draws of that sample are all correct,
and pass@5 is True with or without it.

`tools/parse_dump.py` is fixed: when it detects duplicate indices in a dump (i.e. n>1 sampling),
it downgrades malformed calls to a **warning** instead of contamination; the other contamination criteria are unchanged. Verified that the criteria for greedy runs were not weakened.

---

## 7. Experiment D: P7 training feasibility estimate

**Conclusion: reproducing P7 training on this machine with the paper's configuration is not realistic; the bottleneck is GPU count, not per-GPU capacity.**

Per handoff doc §6, "measure only three things, don't start training". Measured s/step was not done; reason and basis at the end.

### Parameters and GPU memory (parameter count read from safetensors headers)

    parameter count 4.066 B

    training state (FSDP sharded across training GPUs)
      fp32 master  15.1 GB
      fp32 grads   15.1 GB
      Adam m       15.1 GB
      Adam v       15.1 GB
      subtotal     60.6 GB
      reference model 7.6 GB (bf16; run_rl.sh sets ref.param_offload=True, can be offloaded to CPU)

    resident tools (measured this GPU session, not estimated)
      Molmo GPU 34.6 + DepthPro GPU 25.7 + RoboRefer GPU 20.5 = 80.8 GB

Key differences between `run_rl.sh` and eval: `gpu_memory_utilization` 0.5 → **0.7**,
`actor.fsdp_config.param_offload=False`, `optimizer_offload=False`,
`use_kl_loss=True` (needs the reference model), `max_response_length` 4096 → 8192,
`rollout.n=5`, `GPUS_PER_NODE` defaults to **8**.

### GPU count is the hard constraint

Tools total 80.8 GB, taking **at least 2** 80 GB GPUs (Molmo 34.6 + DepthPro 25.7 = 60.3
on one, RoboRefer 20.5 on another). Of the 4 GPUs, **at most 2 are left for training**.

    FSDP over 2 GPUs       60.6 / 2 = 30.3 GB/GPU
    sglang pool @ gmu=0.7  56 GB/GPU
    total                  86.3 GB > 80        <- gmu=0.7 does not fit

`gmu` needs to drop to about 0.45–0.5. But verl's hybrid engine frees GPU memory back and forth between rollout and update
(the eval side achieves this with TorchMemorySaver), so the peak is not a simple sum —
**this item can only be settled by measurement; an analytical estimate cannot give it**.

### Throughput: short by about 8×

The paper's Step 4 is **2 nodes × 8 A100-80G GPUs, 8–12 hours**. We have at most **2 training GPUs**,
8× fewer; linear extrapolation gives **64–96 hours (3–4 days)**, not yet counting efficiency loss
(with a smaller world size, gradient sync takes a larger share, and batches have to be split finer).

**A conclusion of this magnitude does not depend on measurement**: even if s/step were twice as good as the extrapolation, it would still be a day and a half or more.

### Two prerequisites missing on this machine

1. **RL training data is not here**: `siyich/spacetools-rlfulltools`, 3.38 GB / **5425 files**
   (mostly small images). Has to be downloaded now.
2. **No SFT checkpoint**: upstream only releases the final ckpt after RL (already noted in handoff doc §6).
   For throughput measurement the existing ckpt can stand in (same architecture and parameter count, no effect on the machine estimate),
   but it is not the real training starting point; **a same-starting-point comparison still requires rerunning SFT ourselves**.

So measuring s/step was judged not worth it: it can only make "64–96 hours" precise without changing the direction,
and the cost is download + startup + tuning `gmu`, far beyond this item's 30-minute budget.

### Three points for P7

- **Do not schedule training on this machine with the paper's configuration.** Either get more GPUs (separate tools and training onto different nodes,
  as the paper does), or fall back to LoRA — the latter cuts both the fp32 gradients and the Adam states, those 45.4 GB.
- **`model_dtype=bf16` must never be carried into training** (deviation `[20]`). fp32 master is exactly what the optimizer step
  needs; the argument that it is inert for eval does not extend to training.
- **If measuring**, first get both prerequisites in place and lower `gmu` from 0.7;
  the first number to look at is whether GPU memory really is freed back and forth between rollout and update.

---

## 8. Where the data is

| Content | Path |
|---|---|
| Experiment A / B outputs | `p6/swap/*.jsonl` |
| Experiment F (run7) dump and logs | `p6/fp32/blinkdepth/` |
| Deviation `[23]` evidence (run5 / run6) | `p6/gmu025/run{5,6}/blinkdepth/` |
| Experiment G's third and fourth samples | `p6/gmu025/run{3,8}/robospatial/`. **Note**: their KV pool is 20 GB, the same as P4's run1/run2; they are samples under the same configuration, not a different configuration — see that directory's README |
| Experiment C dumps | `p6/passk/{boppose,blinkdepth,robospatial}/` |
| Experiment F's other two runs (run9 / run10) | `p6/fp32/run{9,10}/blinkdepth/`, with a README warning that looking only at run7 leads to the wrong conclusion |
| Migration acceptance check (run4, P4 config) | `p4/dumps/run4/blinkdepth/` + `p4/logs/p4-run4-*` |
| **run2/robospatial late submission** | `p4/dumps/run2/robospatial/` (run in P4 but never committed to the repo; 231/350 = 66.00%, consistent with `P4_RESULTS.md`) |
| Variant and resident scripts | `tools/p6/variants/` (includes a README explaining each difference) |

The sha256 of every dump matches the original on the volume; key numbers can be recomputed directly from the repo copies.

---

## 9. Not done and pending

All GPU tasks for this GPU session are finished. The items below are split by "does it still need a GPU".

### Needs GPU, blocked by installation

- **Experiment B's tool swap** — the only undelivered output of handoff doc §9 (`p6/swap/depth_<tool>.jsonl`).
  Toolshed v1 has only one depth tool; a second depth model must be installed first.
  Gain is bounded: 17 near-tie wrong answers (a slightly stronger tool suffices), 25 confidently wrong (needs a significantly stronger one).
- **A stronger pointing tool** — likewise requires installing a new model. The Molmo run gives a lower bound and path validation;
  the measured complementarity (oracle +10.83 pp / 43 samples) says the direction should be "add" rather than "replace".

The cost of both is mainly download and environment setup, not running.

> **⚠ 2026-09-02 · The user decided not to install new tools; these two items are now "decided not to do".**
> Reason is scope: P7 tests swapping the objective function; swapping tools is an orthogonal path, and doing both at once cannot be attributed.
> The measured headroom (pointing oracle +10.83 pp / 43 samples, depth near-ties 17)
> **is kept as quantified open items**; no further experiments are scheduled. Details in `records/P7_DECISION.md` §0.2.

### Needs GPU, but does not affect any decision

- **Settling `[20]` fp32** — currently fp32 n=3 {107,105,105} vs bf16 n=2 {108,109},
  p≈0.1, not significant. 3–4 more runs per arm would settle it. P4 used bf16 throughout and is internally consistent, so this only concerns
  "does bf16 introduce a bias relative to the paper", and the paper's dtype is unknown.
- **Causality of `[23]`** — the 80 GB / gmu=0.5 side has only n=1. The decision is based on "reproduce P4's pool size",
  not on "0.5 is worse", so extra points only answer a scientific question.
- **blinkdepth majority vote@5** — 110 vs greedy's 107–109 is noise-level; turning it into a conclusion needs more runs.
  But blinkdepth is not where the gap is.
- **Measured s/step for experiment D** — needs the RL data (3.38 GB / 5425 files) and SFT ckpt in place first.
  It can only make "64–96 hours" precise, without changing the direction.

### Does not need GPU

- **P6 manual classification** — of 322 wrong answers, 223 are already settled automatically; **99 remain**.
  Stratified sampling, two-round annotation agreement (κ), representative cases, all offline.
  **New lead from this session: 43.9% of VQA questions swing between samples**; this is a fourth category outside the existing three
  (tool error / reasoning error / annotation issue), and how to place it should be decided before classification.
- **RoboSpatial VQA's −6.46 pp** — the last big gap in the whole reproduction.
  Experiment G ruled out "the paper reports the upper bound"; experiment C shows self-consistency can close about 60%, leaving about 2.2 pp.
  The two leads in `P5_RESULTS.md` §2① are still valid.
- **pose / grasp overlays** (paper Fig. 10–12 style) — `bopgrasp`'s GT is 5 2D keypoints.
- **Write up the formal P5 / P6 reports.**
