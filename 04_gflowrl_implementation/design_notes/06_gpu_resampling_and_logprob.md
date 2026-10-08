# P7 GPU session results: migration acceptance check · dump patch · second batch of 5 samples

> Continues from `records/P7_GPU_HANDOFF.md`. Machine: **4×A100-SXM4-40GB · sm_80 · CUDA 12.8 ·
> driver 580.159.03**. Date 2026-09-02.
> **No training this session**, no RL data downloaded, no training scripts written (handoff doc ⚠③).
>
> **A qualification that applies to the whole document**: the first batch `p6/passk/` ran on an **80 GB** machine; this machine is **40 GB**.
> The KV pools of the two batches are aligned to 20 GB (see §0), but **cross-hardware drift** still exists
> (P6 measured: of 397 RoboRefer calls, 226 returned points with tiny displacements, median 0.0040).
> **So the spread measured on the second batch = sampling noise + cross-hardware drift; it is an upper bound on the sampling spread, not the pure sampling spread.**

---

## 0. Config: `gpu_memory_utilization = 0.5`, not 0.25

    P4 full run            4×A100-40GB   gmu=0.50  ->  KV pool 20 GB
    batch 1 passk          4×A100-80GB   gmu=0.25  ->  KV pool 20 GB      <- the log says 0.25
    batch 2 passk2         4×A100-40GB   gmu=0.50  ->  KV pool 20 GB      <- this session

`gmu` is a static fraction of the **whole GPU** (deviation `[23]`). Verified one by one from the three files
`p6/passk/*/eval.log.gz` that batch 1 really used `gpu_memory_utilization=0.25`, not taken from what the docs say.

### 0.1 ⚠ New deviation: `max_num_seqs` 256 -> 64

**The first `robospatial` attempt OOMed on 40 GB, even though the arithmetic in ⚠① was correct.**
The problem is that it only aligned the **KV pool**, not the **headroom**:

| | machine | gmu | KV pool | n | policy GPU | result |
|---|---|--:|--:|--:|---|---|
| P4 `robospatial` | 40 GB | 0.5 | 20 GB | **1** | peak 25.6 GB | pass |
| batch 1 `passk` | **80 GB** | 0.25 | 20 GB | **5** | ~60 GB extra headroom on the GPU | pass |
| batch 2 first attempt | 40 GB | 0.5 | 20 GB | **5** | **39.45/39.49 GB** | **OOM** |

    Process 2500371   8.09 GiB   <- FSDP policy (bf16 weights, exactly 8.1 GB)
    Process 2502296  31.36 GiB   <- sglang
    free                31 MiB

**There is no tool actor on the policy GPU** — these two processes filled the GPU by themselves.
On 80 GB a 20 GB pool leaves 60 GB for weights and activations; on 40 GB the same pool leaves only 20 GB.

**The crash site is the vision encoder, not the KV cache** — this turns the choice of knob from a guess into a targeted fix:

    sglang/srt/layers/attention/vision.py:701   apply_rotary_pos_emb
    sglang/srt/layers/rotary_embedding.py:2733  rotate_half -> torch.cat
    torch.OutOfMemoryError: Tried to allocate 70.00 MiB

What blew up is **the number of image patches processed at once in a single forward pass**. `robospatial` averages **2.76 MP**, the largest of the nine
benchmarks (`blinkdepth` is only 0.18 MP); `n=5` multiplies the number of images in the same batch by five.

**Decision: `max_num_seqs` 256 -> 64, `gpu_memory_utilization` unchanged.**
`max_num_seqs` limits the number of concurrent sequences and cuts the activation peak of the vision tower, while **not changing the KV pool size** —
the pool size is the only basis for comparability between the two batches (deviation `[23]`) and must stay at 20 GB.

> **Honest caveat**: changing concurrency also changes batch composition, and batch composition is exactly what this project identified
> as the only mechanism of run-to-run variation. **But this does not add a new class of error** — going cross-hardware (80 -> 40 GB) by itself
> already introduces the same kind of drift, and handoff doc ⚠④ already requires the spread of batch 2 to be written as
> **an upper bound of "sampling noise + cross-hardware drift"**. `max_num_seqs` falls under the same qualification; the wording of the qualification stays the same.

### 0.2 A disk incident that blocked the entire session

That OOM crash wrote a **50 GB core** to the **container disk** (not the volume), filling `/` to 100%;
after that **every command hit ENOSPC**, not even `df -h` would run.
Full record and prescription (`ulimit -c 0`, now added to all five runners) in `records/CHANGES.md` §10.9.

**Why it is worse than an ordinary failure**: the core filled the disk **halfway through the job**,
so the two benchmarks queued after `robospatial` would die from a full disk even if they never came close to OOM —
**one crash contaminates the entire session**, exactly the kind of failure `--strict` is meant to guard against but cannot see.

**seed: this handoff requirement needs correcting.** Handoff doc §3 says "the seed must differ from batch 1",
but **there is no seed knob** on this path — `verl/trainer/config/rollout/rollout.yaml` has no sampling seed,
and `verl/workers/rollout/sglang_rollout/` does not pass a seed through to sglang either,
so sglang picks its own random seed every time the engine starts. **Each rerun is already an independent draw**;
"changing the seed" is neither needed nor possible. The seeds actually used are copied from the eval logs as a record (§3).

---

## 1. Task 0: migration acceptance check — passed

No data needed copying: the `/workspace` volume is fully intact (`envs` 48G · `models` 34G · `hf` 31G ·
`experiments/p4` · `eval-benchmarks` · `SpaceTools-RL @ f0742338`), with paths verbatim identical to P4.
So task 0 shrinks from "copy + acceptance check" to **acceptance check only**.

| item | expected | measured |
|---|---|---|
| `compute_cap` | 8.0 | **8.0** ×4 · 40960 MiB |
| `nvcc` | 12.x | **12.8** |
| `verify_p1.sh` | all five environments usable | **all green** (roborefer torch 2.5.1+cu124 / vlm 2.9.1+cu128 / bbox numpy-only / graspgen 2.3.1+cu121 / rl 2.9.1+cu128), all 8 toolshed modules import successfully |
| `pointnet2_ops` cubin | contains sm_80 | **`sm_80 sm_86 sm_89`** |
| `p2_tool_chain.py` | 12/12 | **12 passed · 0 warned · 0 failed** |
| `--strict` | exit 0 | **PASS**, all six health metrics 0 |
| GPU memory | no GPU above 34.6 GB | GPU0 21.1 · **GPU1 34.6 (87%)** · GPU2 25.7 · GPU3 25.3 |
| wall clock | ~7.5 m | 8m 07s |

**The label is `run12`, not `run9` as the handoff doc says** — `run1..run11` are already taken in `/workspace/experiments/p4/`
(run9/run10 are the fp32 experiments on 80 GB), and the guard in `p4_run.sh` flatly rejects duplicate names.

### 1.1 Reading the numbers: single run inside the band + hard core matches

Per the handoff doc's new definition, we **do not use** the old doc's narrow 107–109 band that came from n=3.

    single run   run12   107/124 = 86.29%          bf16 actual range over five runs 106–109  -> inside the band
    regression   samples every baseline got right but run12 got wrong      0
    hard core    12  [20,22,37,43,55,61,76,84,93,94,114,119]   matches the 12 recorded in the doc one by one

    The four bf16 baselines used as reference
      p4/dumps/run1/blinkdepth       107/124 = 86.29%   (40GB gmu=0.5)
      p4/dumps/run4/blinkdepth       106/124 = 85.48%   (40GB gmu=0.5)
      p6/gmu025/run5/blinkdepth      108/124 = 87.10%   (80GB gmu=0.25)
      p6/gmu025/run6/blinkdepth      109/124 = 87.90%   (80GB gmu=0.25)

**Conclusion: nothing got distorted by the move.**

> **⚠ One thing that must be flagged per `P7_HANDOFF.md` §3.3 item 1, and must not be written up as good news.**
> With the four baselines combined, the upper bound is **111/124** and the hard core is 13; **after adding run12 it becomes 112/124, hard core 12**.
> The upper bound went up by one again — exactly what "the upper bound is a monotonically increasing function of the number of draws; without visible saturation it is not an invariant" means.
> **The fact that 112 = 90.32% equals the paper's number is not used as evidence in this session.**
> The acceptance check rests on **a single run inside the band + the hard core matching + zero regressions**, not on the upper bound hitting the paper value.

### 1.2 The patch was inert during the acceptance check — checked, not argued

The dump patch was being written while the acceptance check was running, and the import time of the worker actors spanned the edits.
With the switch off, the patch should be a no-op, but that is an argument, not a measurement, so we checked directly:

- **0** `[P7]` markers in the acceptance log
- run12's dump field list `acc answer gts image index input num_turns output reward score step uid`
  — **exactly the same** as P4, no new fields at all

---

## 2. Task 1: dump patch — verified

The patch touches only one file: `verl/trainer/ppo/ray_trainer.py`, +141 lines.
Shape copied from `patches/rl/0008`: **everything wrapped in try/except; this is a diagnostic and must never be able to break an eval**.
Controlled by the switch `+trainer.dump_token_diagnostics=True`, **off by default**; normal eval behavior is unchanged.
(The `VERL_DUMP_TOKEN_DIAGNOSTICS=1` environment variable is also kept as a fallback for hydra struct mode; in practice `+` works.)

| field | content |
|---|---|
| `response_mask_rle` | `[[value, run_length], ...]`, covers all 4096 positions, can be rebuilt bit for bit |
| `n_policy_tokens` | `response_mask.sum()`, i.e. `|y_i|` |
| `n_response_tokens` | non-padding response length |
| `response_token_ids` | raw token ids, truncated to `n_response_tokens` (insurance field) |
| `rollout_log_probs` | sglang side, only `mask==1` positions |
| `trainer_log_probs` | FSDP side, same as above |

### 2.1 The open item the handoff doc left: go with (a), not (b)

`P7_GPU_HANDOFF.md` §2.2 says outright that it did not work out for me how to get the trainer-side logprob. **The answer is (a), zero re-tokenization.**

The reasoning is already in `fit()`: after `generate_sequences()` the training loop first calls
`self.checkpoint_manager.sleep_replicas()` (releasing sglang's weights and KV pool), then
`_compute_old_log_prob`. Copying this order into `_validate()` is enough. Measured: GPU3 dropped from 25 GB
to **9.4 GB**, and the FSDP forward pass ran fine in the freed GPU memory.

    [P7] trainer-side log_probs computed: shape (160, 4096)
    [P7] rollout log_probs present:      shape (160, 4096)

**One edge case the original doc did not anticipate; a guard has been added.** When `val_batch_size` is null, verl uses
`len(val_dataset)`, so **the whole benchmark is a single val batch** — after putting the engine to sleep there is no need to wake it.
But if someone sets `val_batch_size`, sleeping in the middle of the loop would ruin generation for the subsequent batches.
So if `len(self.val_dataloader) != 1`, it **skips the trainer side and prints a warning**, instead of silently breaking the eval.

### 2.2 `response_mask` semantics guard: 160/160 pass

    [P7] response_mask semantics check on 160 multi-turn samples: OK

What it guards against is the **silent** fallback `compute_response_mask()` at `ray_trainer.py:157`
(= `attention_mask[:, -L:]`, all tool tokens are 1). The criterion is applied to multi-turn samples
(when `num_turns > 2`, policy tokens must be strictly fewer than non-padding tokens).
**Implemented as print + warning rather than assert** — same contract as §2: a diagnostic should not be able to crash the eval.

Along the way, confirmed from the source something we had worried might be misaligned: `rollout_log_probs` and `response_mask` are **aligned bit for bit**.
`tool_agent_loop.py:432-433, 464-465` pads tool-response blocks with `[0.0] * len(response_ids)`,
and `agent_loop.py:628-629` then pads the tail to `response_length`. So what is taken by `mask==1`
is exactly the policy-generated tokens.

### 2.3 Subset smoke test (boppose 32 rows × 5)

    rows                       160
    RLE rebuilt length == 4096              160/160
    sum(mask) == n_policy_tokens            160/160
    len(rollout_log_probs) == n_policy      160/160
    len(trainer_log_probs) == n_policy      160/160
    len(response_token_ids) == n_response   160/160
    n_policy < n_response (necessarily, multi-turn)   160/160
    parse_dump --strict                     exit=0, all six health metrics 0

**The first substantive result, and it corrects an existing number.**

| | character proxy (`P7_STEP23_RESULTS.md` §1.2) | **real tokens (this run)** |
|---|--:|--:|
| `boppose` raw response / assistant | 2.40× | **3.22×** |

The trap in `P7_DECISION.md` §5.1 item 2 is **more severe than the character proxy showed**: if `|y_i|` mistakenly uses the raw
response length, the effective temperature of `boppose` gets amplified more than threefold by the verbosity of the tool output.
**Qualification**: 32-row subset, n=1. Full numbers in §3.

---

## 3. Task 2: second batch of 5 samples — done

    robospatial  1750 rows  06:59:22 -> 08:23:46   --strict exit=0
    blinkdepth    620 rows  08:23:47 -> 08:47:51   --strict exit=0
    boppose       300 rows  08:47:52 -> 09:05:47   --strict exit=0

Config: `n=5 · T=1.0 · top_p=1.0 · gmu=0.5 (20 GB KV pool) · model_dtype=bf16 ·
calculate_log_probs=True · max_num_seqs=64` (see §0.1). Archived in `p6/passk2/`, **`p6/passk/` not overwritten**.

### 3.1 Health and accuracy: both batches side by side

| | batch 1 (80 GB) | batch 2 (40 GB) | diff |
|---|---|---|---|
| `robospatial` | 1113/1750 = 63.60% | **1123/1750 = 64.17%** | +10 samples / +0.57 pp |
| `blinkdepth` | 527/620 = 85.00% | **543/620 = 87.58%** | +16 samples / +2.58 pp |
| `boppose` (mean IoU) | 63.00% | **62.67%** | −1 sample |

**Neither batch is cleanly all-zero, and they are asymmetric in different ways**: batch 1 has `malformed tool_call 1`,
batch 2 has `cut-off generation 1` + `no <answer> 1` — 1/1750 (0.06%) each;
both batches pass `--strict` (no change contaminated the criterion). **When reporting, mention both batches, not just one.**

### 3.2 The n=1 observation from §3.2 — **reproduced**

"For half to three quarters of prompts, all five rollouts get the same reward" was the item in `P7_DECISION.md` §3.2
most worth betting on, at the time marked "single sampling, to be rechecked".

| fraction of reward-degenerate groups | batch 1 | **batch 2** |
|---|--:|--:|
| `robospatial` | 56.9% (199/350) | **56.3% (197/350)** |
| `blinkdepth` | 75.8% (94/124) | **79.0% (98/124)** |
| `boppose` | 53.3% (32/60) | **58.3% (35/60)** |

**Reproduced; can be promoted from "observation to be rechecked" to a conclusion.**

### 3.3 clip saturation rate — **also reproduced, verbatim**

| saturation rate of non-degenerate rollouts | batch 1 | **batch 2** |
|---|--:|--:|
| `robospatial` | 100% | **100% (765/765)** |
| `blinkdepth` | 100% | **100% (130/130)** |
| `boppose` | 41.4% | **39.2% (49/125)** |

The summary item in `P7_STEP23_RESULTS.md`, "at the starting point the flow gap takes only the three values `{−ε_low, 0, +ε_high}`",
holds unchanged on batch 2. **The inference drawn from it, "β should start near 1, not 8", still stands.**

### 3.4 ⚠ Majority vote@5: **the effect disappeared, and the direction is inconsistent**

This is the **first reason the second sampling batch exists** (`P7_HANDOFF.md` §3.3 item 3: the spread on both sides must be measured).

Batch 1 ran only one voting arm, scoring **176/228** on VQA; batch 2's voting arm scored **166/228**.
**The voting arm's own spread is 10 samples — the same order of magnitude as the spread of the four greedy runs (161/167/168/169, range 8).**

Per-sample paired (two-sided exact McNemar, vs the same four greedy runs):

| | batch 1 net diff / p | **batch 2 net diff / p** |
|---|---|---|
| VQA | +9 / +8 / **+15** / +7 · p=0.163 / 0.230 / **0.014** / 0.281 | **−1 / −2 / +5 / −3 · p=1.000 / 0.868 / 0.500 / 0.720** |
| Vacant | net −2 ~ 0 · p all ≥ 0.75 | **−2 / −3 / −1 / −2 · p all ≥ 0.607** |
| `blinkdepth` | — | **+2 / +3 / +1 / +0 · p all ≥ 0.453** |

**0/12 comparisons are significant, and the VQA direction flipped sign between the two batches.**

> **Conclusion: majority vote@5 has no effect on these three benchmarks.**
> Batch 1's "one in four with p=0.014" is exactly the kind of artifact the §3.3 discipline warns about —
> **the four comparisons share the same voting arm, and that arm had no spread at the time.** Now it does, and the effect is gone.
>
> **Knock-on consequence: the item in `records/P6_REPORT.md` and the project docs "self-consistency fills about 60% of the gap"
> (already downgraded to "consistent direction, not significant") must be downgraded one more step: "no effect".
> The −6.46 pp gap on `robospatial` VQA therefore remains entirely unexplained, and has one fewer candidate explanation.**

**One positive reproduction**: "pure decision flips on Vacant = 0" holds in both batches (batch 1 0/51, batch 2 0/47),
and `blinkdepth` is single-digit in both (3 / 4). Pure decision flips on VQA are 46 / 62 in the two batches —
**the phenomenon is there, the gain is not**: the model really does waver on the same evidence, but the wavering is symmetric, and voting cannot recover it.

---

## 4. The most important output of this session: the difference between the two sides' logprobs

### 4.1 Real token counts of `|y|` — the character proxy **systematically underestimated** the trap

| | character proxy (`P7_STEP23_RESULTS.md` §1.2) | **real tokens** | `|y|` median (tokens) |
|---|--:|--:|--:|
| `robospatial` | 1.14× | **1.22×** | 334 |
| `blinkdepth` | 1.41× | **1.67×** | 423 |
| `boppose` | 2.40× | **3.22×** | 305 |

**All three are larger than the character proxy, and still systematically different per benchmark.**
The trap in `P7_DECISION.md` §5.1 item 2 (`|y_i|` mistakenly using the raw response length = letting the verbosity
of the tool output decide the effective temperature) **is more severe than previously recorded**. The ratios from the character proxy were credible but low; replaced as promised.

### 4.2 T6's dimensional mismatch, with real magnitudes

| | per-token log π (trainer arm) | `|y|` median | whole-sequence log π | vs `β·r ∈ [0,8]` |
|---|--:|--:|--:|--:|
| `robospatial` | −0.1717 | 334 | **−57.3** | **≈ 7×** |
| `blinkdepth` | −0.1255 | 423 | **−53.1** | **≈ 7×** |

The claim in `P7_STEP23_RESULTS.md` §2.5 (`Z_t` grows with the **sum** over `|y|`, while the term it centers is
the **per-token mean**) **is confirmed at real magnitudes**: the term being averaged is about seven times the whole range of `β·r`,
and grows linearly with `|y|`.

### 4.3 Distribution shape — where criterion i-b lands

⚠ What is measured is the **framework difference between rollout (sglang) and trainer (FSDP)**, not training drift
(`π_old = π_ref`; GFlowRL's `log π_ref − log π_old` is identically 0 on this data).
It is the **first real proxy** for drift (the IS weight `w_i` exists precisely for it); **the units and magnitudes are comparable, but it is not the same quantity**.

| per-sequence summed gap | skew | excess kurtosis | Sarle BC | reading |
|---|--:|--:|--:|---|
| `robospatial` | +1.02 | +3.8 | 0.208 | unimodal · right-skewed · mildly heavy-tailed |
| `blinkdepth` | +0.67 | +3.3 | 0.155 | same as above |
| `boppose` (out of scope) | +5.10 | +45.4 | 0.526 | clearly heavier-tailed |

**The prior in `P7_CRITERION_IB.md` §4 is overturned.** It said: "our rewards are close to binary on the two accuracy
benchmarks — if the drift term is also close to bimodal, **the arithmetic mean is actually the best of the three**,
and GFlowRL's choice holds up for us." **Measured, it is not bimodal** (BC 0.155–0.208, threshold 0.5556).

Resampling G=5 on the **measured** distribution and comparing the variance of the three constants (`tools/p7/p7_ib_real.py`):

| | mean (GFlowRL Eq.4) | median | **huber** |
|---|--:|--:|--:|
| `robospatial` | 1.00× | 0.89× | **0.83×** |
| `blinkdepth` | 1.00× | 1.00× | **0.93×** |
| `boppose` (out of scope) | 1.00× | 0.39× | 0.40× |

> **The answer to criterion i-b: median/huber are slightly better in direction, but only by 7–17%.**
> In the synthetic table, t3 gave 0.6× and bimodal gave 3.5–7.4× — **the measurement is neither.**
> **Correct statement: the choice of constant is not where the problem lies.** That is a different conclusion from "GFlowRL chose wrong";
> do not write it as the latter. (`boppose`'s 0.39× looks more like the heavy-tailed case, but per §0.2 it is outside P7's scope.)

### 4.4 Magnitude: **already over the threshold with zero training**

`P7_CRITERION_IB.md` §3 set the threshold at which "the normalization inconsistency starts erasing the reward contribution" at
about **5.0e-4** (`robospatial`) / **7.5e-4** (`boppose`) nat per token.

| mean \|per-token gap\| per rollout | measured | vs threshold |
|---|--:|---|
| `robospatial` | **1.856e-3** | **3.7× over** |
| `blinkdepth` | **1.538e-3** | **3.1× over** |
| `boppose` | 4.481e-4 | not over |

**On both in-scope benchmarks, the framework difference alone is already 3–4 times the threshold, before any training has happened.**
Real training drift stacks on top of it and can only make it larger.

### 4.5 §6 candidate fix: **confirmed effective on real drift**

Normalizing the logprob term of `Eq. 4` by `|y|` as well (`P7_CRITERION_IB.md` §6), `β=8`:

| fraction of groups whose whole `g̃` is clipped to the same value | Eq.4 as is | **after normalization** |
|---|--:|--:|
| `robospatial` | 26.9% | **0.0%** |
| `blinkdepth` | 46.0% | **0.0%** |
| `boppose` | 3.3% | **0.0%** |

At the same time the clip saturation rate drops from 69.6% → 43.7% (`robospatial`), 66.9% → 21.0% (`blinkdepth`).
**The conclusion on synthetic data (36%–93% → 0.0%) holds on real drift.**

One more thing in passing: rollouts where mean and huber give **opposite update directions** make up 10.0%–14.6%
(synthetic prediction 11%–20%); also reproduced.

---

## 5. Net effect on A′

| item | disposition |
|---|---|
| criterion (i) `Var(Z_t)` | "does not matter at the starting point" **reproduced** (100% saturation rate reproduced verbatim) |
| **criterion i-b** | **can be closed**: measured shape unimodal, mildly heavy-tailed; the three constants differ by 7–17%. **The choice of constant is not where the problem lies** |
| criterion (iii) degenerate groups | degeneracy rates 56.3% / 79.0% / 58.3%, **reproduced**, promoted from observation to conclusion |
| **normalization inconsistency** | **promoted to A′'s top finding**: threshold 5e-4, measured 1.5–1.9e-3 (3–4× over), **already crossed with zero training**; and **there is a fix and the fix works on real data** |
| majority vote@5 | **downgraded to "no effect"**; the VQA gap has one fewer candidate explanation |
| `|y|` and T6 | character proxy replaced with real tokens; dimensional mismatch ratio ≈ 7×, confirmed |

**A′'s answer**: `Z_t` is not usable in our setting, **and the reason is neither the variance of the within-batch MC nor a wrong choice of constant,
but the length normalization inconsistency between `Eq. 4` and `Eq. 5/6`**.

---

## 6. Fixed-point cost of the fix — measured (2026-09-02, zero GPU)

The to-do in `P7_CRITERION_IB.md` §6 is closed; full derivation and numbers in **§6.1–6.3** of that doc;
script `tools/p7/p7_fix_cost.py`. Conclusion in one sentence: **the fix is not free.**

    Existence: fixed. Self-consistency of the normalized Eq.4 at the zero-loss point is an **identity**
               (|Z_t − Z*| = 8.9e-16, loss 6.9e-34); the paper as is misses by 4.9–6.4 at the same point, loss 24–40.
    Cost:      in the **location** of the fixed point. The zero-loss condition itself forces log π* = log π_ref + L·(β·r − Z),
               i.e. Remark B.4's inverse temperature **L·β**.

    Homogeneous length sweep (effective support / max p)
        L=1     21.006 / 0.1388     <- bit-for-bit identical to Prop.B.1's β-tilt (mechanical self-check)
        L=10     1.910 / 0.7500
        L=50     1.004 / 0.9996
        L=334    1.000 / 1.000000   <- measured median |y| on robospatial

**At `L = 334` the fixed point is a point mass: distribution matching degenerates into reward maximization** — exactly what GFlowRL is meant to avoid.

| config | whole-group `g̃` same value | self-consistent zero | fixed-point tilt |
|---|---|---|---|
| **A. paper as is** | 26.9%–46.0% (measured) | **does not exist** | minimizer `c ≈ 1` |
| **B. §6 fix** | **0.0%** (measured) | **exists** | **`L·β` → argmax** |
| **C. no normalization on either side** | — | exists | `β`, entropy of the same order as `π_ref` |

**A's illness and B's illness are two sides of the same length normalization.** The only one healthy on both metrics is **C**,
and C's cost is exactly the problem the paper introduced length normalization to solve in the first place (long sequences weigh more in the loss).

> **No contradiction with the item retracted in `P7_STEP23_RESULTS.md` §2.3**: that item said that with the paper as is, the minimizer lands at
> `c ≈ 1` rather than `c = L` — true, because there is no zero there at all. **The fix makes the zero actually exist,
> and under the fix the `L·β` reading comes back.** The two statements apply to different objective functions.

---

## 7. The full answer to A′

> **On SpaceTools' multi-turn tool trajectories, is GFlowRL's within-batch MC estimator `Z_t` still usable?**
>
> **No, and the reason is attributable, specific, and not in the "within-batch MC" idea itself.**

1. **Not the variance.** Criterion (i): at the starting point the flow gap is 100% flattened by the clip (reproduced in both batches),
   so `Var(Z_t)` does not matter there.
2. **Not a wrong constant.** Criterion i-b: the measured drift proxy is unimodal and mildly heavy-tailed (Sarle BC 0.155–0.208);
   the variances of the three constants differ by **7–17%**. **The choice of constant is not where the problem lies.**
3. **It is the length normalization inconsistency between `Eq. 4` and `Eq. 5/6`.** Threshold about 5e-4 per token;
   **the framework difference with zero training is already 1.5–1.9e-3 (3–4× over)**, and real training drift can only be larger.
   The consequence is that the whole group's `g̃` is clipped to the same value and **the reward contribution is completely erased** (measured in 26.9%–46.0% of groups).
4. **And this inconsistency has no free fix.** Normalizing `Eq. 4` removes the symptom (0.0%),
   but pushes the fixed point to a point mass; normalizing neither side goes back to Prop. B.1, at the cost of long sequences dominating the loss.

**Scope**: none of the above includes `bopgrasp` / `boppose` (`P7_DECISION.md` §0.2),
and all of it is based on the **framework difference** as a drift proxy, not training drift (`π_old = π_ref`, that term is identically 0).
