# Route C plan: GFlowRL (C′ arm) vs GRPO from the same starting point

> **This doc answers**: how exactly Route C is done, how many machines and how much time it needs, and whether it can measure anything.
> **It does not answer**: whether to do it — that is a scheduling decision; the basis is in the two decision points of §7.
> Written 2026-09-02. Prerequisites: `P7_DECISION.md` (route) · `P7_GPU_RESULTS.md` (A′'s answer) ·
> `P7_ROUTE_C_GATE.md` (gate for the training arm).

---

## 0. Two prerequisites that determine what this plan looks like

**① A′ has already answered: `Z_t` per the paper as is is not usable in our setting.** The reason is neither the variance of the within-batch MC
nor a wrong choice of constant, but the length normalization inconsistency between `Eq.4` and `Eq.5/6`
(threshold 5e-4 nat/token; the framework difference with zero training is already 1.5–1.9e-3).
**So C cannot directly run "GFlowRL per the paper as is vs GRPO" — that would be testing a known-defective implementation.**

**② The gate has already chosen the arm: C′ (no length normalization on either side: neither `Eq.4` nor `Eq.5/6`).**
`P7_ROUTE_C_GATE.md`: the cost of length dominance on our length distribution is mild
(the longest 10% take 20.7–24.2% of the loss, **only 1.04–1.32× within group**);
on non-degenerate groups reward dominates drift **24×**; on degenerate groups the whole-group same-value rate is **0.0%** (A is 30–49%).
**The cost is that C′ has the highest clip saturation rate (77.9%).**

> **So the one question C asks must be rewritten:**
> ~~"How many points does GFlowRL add over GRPO"~~
> **"If GFlowRL's distribution-matching objective (C′ form) is swapped into SpaceTools' Step 4,
> does accuracy on `robospatial` move?"**
> Swapping the objective cannot touch P6's 273 tool errors and toolset gaps (84.7%),
> so **the question can only be about the movable part**, and that is almost entirely in `robospatial` (§6.1).

---

## 1. The four Steps

The real architecture of `run_rl.sh`: **2 nodes × 8 GPUs** — one whole node runs Toolshed,
the other whole node runs FSDP training + sglang (`trainer.nnodes=1`, `n_gpus_per_node=8`).

### Step 3 · SFT (shared by both arms, run once)

| | |
|---|---|
| data | `siyich/spacetools-sft` — **7463 files / 6.32 GB** (not previously in any record) |
| code | `ChicyChen/SpaceTools-SFT` (LLaMA-Factory fork, Apache 2.0, `b7ebbf32`) |
| config | vision tower frozen, LLM only; 3000 steps |
| paper time | 8×A100-80G, 3–4 h |
| **acceptance check** | the paper's Table 4 has Tool-SFT ablation numbers to use as a target, **but that table's denominator differs from Table 2; check the definition before using it** |

**No tools needed.** So SFT can use all GPUs, and it is the only one of the four Steps that can run on the current machine.

### Step 4 · full-tool RL (k seeds per arm, main experiment)

Hyperparameters copied directly from `run_rl.sh`:

    rollout.n = 5                    ← G=5, this is where criterion i-b's 3.17× variance comes from
    train_batch_size = 64            → 5425 / 64 ≈ 85 steps = 1 epoch
    ppo_mini_batch_size = 64 · ppo_micro_batch_size_per_gpu = 2
    max_prompt_length = 8192 · max_response_length = 8192   ← eval side is 4096
    gpu_memory_utilization = 0.7 · max_num_seqs = 256
    use_kl_loss = True · kl_loss_coef = 0.01 · kl_loss_type = low_var_kl
    entropy_coeff = 0 · lr = 1e-6 · freeze_vision_model = true
    gradient_checkpointing = True · actor param/optimizer offload = False
    ref.param_offload = True · total_epochs = 1 · save_freq = 5 · test_freq = 5

**The training-side tool config is 2.6 times the eval one**, which is easy to overlook:

    RL:   roborefer 6×0.6 + vlm 2×0.6 + sam2 5×0.2 + depth 5×0.2
          + bbox 5×0.1 + vision_ops 8×0 + grasp 5×0.1  =  7.8 GPU   ← fills a whole node
    eval: the same seven tools                          =  3.0 GPU

The reason is batch 64 × n=5 = **320 rollouts calling tools concurrently per step**. With fewer actors,
tool latency dominates step time, and the throughput deficit is worse than extrapolating from GPU count.

### ⚠ Controlled difference between the two arms: the KL term is a confound that must be handled

The GRPO arm has a KL-to-ref with `kl_loss_coef = 0.01`; GFlowRL's distribution matching is **anchored on
`π_ref` by itself** (the target is `π_ref·exp(β·r)`), and the paper's Table 9 KL coefficient is **0.0**.
**If one arm has KL and the other does not, the difference is not just the objective.**

**Handling (pick one, must be fixed in advance):**
1. **Turn KL off in both arms** (`kl_loss_coef=0`) — clean, but the GRPO arm deviates from SpaceTools' released config;
2. **Keep KL in both arms** — the GFlowRL side becomes "distribution matching + extra KL", which differs from the paper's recipe.

**Recommend ①**, and state in the report that the GRPO arm is therefore not SpaceTools' original config.

### Step 5 · eval (once per checkpoint)

Reuse `tools/p4_run.sh` + `parse_dump.py --strict`, nine keys, greedy,
`model_dtype=bf16` (on 40 GB), `gmu` reported together with the numbers. **About 2 h 10 m / checkpoint.**

---

## 2. Implementation: what to change

`P7_PRIOR_ART.md` already established: **in our fork, both the registry and `ref_log_prob` are already in place.**

    SpaceTools-RL @ f0742338 · verl 0.8.0.dev
      core_algos.py    register_policy_loss already exists (11 registered losses)
      dp_actor.py:528  with use_kl_loss, ref_log_prob is already selected into the micro-batch
      dp_actor.py:615  policy_loss_fn call site
      ray_trainer.py:1528  token_level_scores (raw r; advantages are group-normalized)

**Minimal path, about ten lines:**
1. `actor.use_kl_loss=True` + `kl_loss_coef=0` — get `ref_log_prob` for free; KL multiplied by 0 has no effect
2. `select_keys.append("token_level_scores")` — GFlowRL needs the raw `r`
3. Write `@register_policy_loss("gflowrl_cprime")` into `core_algos.py`
4. At the L615 call site, also pass `ref_log_prob` and `token_level_scores`

**Key point of the C′ implementation: both `Eq.4` and `Eq.5/6` use `masked_sum`, not `masked_mean`.**
(FlowRL's `compute_flowrl_objective` has ready-made code for both, but it is built on verl 0.4.0;
**the pattern transfers, the API not necessarily; do not copy it verbatim**.)

**Required guard** (`P7_STEP4_RESULTS.md` §2): `response_mask` has two sources with the same name and different meanings;
the fallback at `ray_trainer.py:157` is `attention_mask[:,-L:]` (all tool tokens are 1) and is **silent**.
On multi-turn samples assert `response_mask.sum() < attention_mask[:,-L:].sum()`.

**β: no lower than 2, start with 8** (`P7_ROUTE_C_GATE.md` §4;
the "β≈1" in `P7_STEP23_RESULTS.md` §1.4 is void under C′).

---

## 3. Machine: the current one cannot run Step 4 — not slow, it does not fit

Tools measured (P6 GPU session, not estimated): Molmo **34.6** + DepthPro **25.7** + RoboRefer **20.5** = **80.8 GB**.
On **40 GB GPUs**: Molmo takes a whole GPU (87%); DepthPro + RoboRefer = 46.2 > 40, needs two.
**Tools take 3 GPUs, leaving only 1 for training.**

Single-GPU training state (4.066 B parameters, read from the safetensors header):

    fp32 master 15.1 + grads 15.1 + Adam m 15.1 + Adam v 15.1 = 60.6 GB  >>  40
    Even with optimizer_offload=True (Adam moved to CPU): 30.2 GB + activations + sglang pool
    (gmu=0.7 → 28 GB)                                      → still OOM

**`model_dtype=bf16` cannot rescue this** — that is the eval-only deviation `[20]`,
and **the fp32 master is exactly what the optimizer step needs**.

**Three ways out:**

| way out | cost | is it still C |
|---|---|---|
| **add GPUs to 2 nodes × 8** (the script's original assumption) | rent machines | **yes**, C as is |
| **fall back to LoRA** | cuts the 45.4 GB of base gradients and Adam state | **no** — full-parameter vs LoRA is a different experiment; conclusions cannot be extrapolated to the paper's config |
| heavy CPU offload + tiny batch | I estimate on the order of days, **not measured** | yes, but throughput could multiply §5's budget several times |

---

## 4. Statistical design — this is what decides whether C is worth it

### 4.1 Ceiling on effect size

`P6_REPORT.md`'s attribution: swapping the objective **cannot touch** the 273 tool errors and toolset gaps (84.7%).
What can move is **reasoning errors 19 + 3b 14 + 2a 11 = 44 samples** (3b's attribution is disputed; counting only reasoning errors gives 19).

    upper bound of movable share   19 – 44 / 2001 = 0.95 – 2.20 pp     and that is "eat all of it, miss none"

### 4.2 Using the total score as primary metric → not enough power, **even at maximum effect**

McNemar's power is determined by the number of **discordant pairs**. Two-sided α=0.05, 80% power requires

    net diff Δ  ≥  2.80 · √D          D = total discordant pairs

Discordance rate of two differently trained models on the same samples: between runs of the same model it is 6.9–10%
(`robospatial` 24/350, `blinkdepth` 13/124); the tool-swap run was 36.5% (145/397).
Taking 15% in between:

    D ≈ 0.15 × 2001 ≈ 300   →   need Δ ≥ 48
    while our **ceiling** is Δ = 44

> **So with the total score over nine benchmarks as the primary metric, this experiment cannot detect anything even at maximum effect.**

### 4.3 Fix: pre-register `robospatial` as the primary metric

P6 already located the movable share cleanly — **about 42 of the 44 are in `robospatial`**:

    3b coordinate frame/semantics   14   all robospatial (§3.1)
    2a should have called, didn't   11   3 manual + 8 front/behind, all robospatial
    reasoning errors                19   of which 15 from robospatial Vacant, 1 robospatial manual

Recomputed on n=350:

    D ≈ 0.15 × 350 ≈ 52   →   need Δ ≥ 20        ceiling Δ = 42

**Enough power.** This is not a statistical trick; it is a direct consequence of P6's attribution: **the effect can only show up there in the first place.**

### 4.4 Pre-registration (fixed before running)

- **Primary metric**: per-sample paired (McNemar) on the 350 `robospatial` samples, same samples for both arms
- **Secondary metrics**: the other six accuracy benchmarks, **as a "did not break anything elsewhere" check, not as gains**
- **Excluded**: `boppose` / `bopgrasp` (`P7_DECISION.md` §0.2 — `r` is insensitive to gripper orientation)
- **seed**: **≥3 per arm**. Seed-to-seed variance in RL training is usually larger than eval noise;
  1 seed per arm cannot attribute anything (§3.3 item 3: the spread on both sides must be measured)
- **Decoding**: greedy, default `val_kwargs` definition; `--strict` as gatekeeper; `gmu` and KV pool reported with the numbers
- **Failure counts as a result too**: "C′ has no effect on `robospatial`" is an independent confirmation of P6's
  "swapping the objective cannot touch 84.7%"

---

## 5. Cost

| | config | time |
|---|---|---|
| implementation + synthetic fixed-point self-check | zero GPU | 1–2 days |
| Step 3 SFT (shared, once) | 8 GPUs | 3–4 h (paper) · 4 GPUs est. 8–12 h |
| **Step 4 × 2 arms × 3 seeds** | **2 nodes × 8 GPUs** | 8–12 h each → **48–72 h** |
| Step 5 eval × 6 checkpoints (+SFT baseline) | 4 GPUs | ~2 h 10 m each → **13–15 h** |
| **Total** | | **about 4–5 days on 16 GPUs** |

Data: SFT 6.32 GB + RL 3.38 GB (5425 files). Storage ~100 GB.

---

## 6. Two decision points

**Decision point 1: rent 2 nodes × 8 GPUs or not.**
The current 4×A100-40GB **cannot run Step 4** (§3). Without renting, only LoRA (a different experiment)
or heavy offload (not measured) remains. **This is not something a technical judgment can settle; it is a spending decision.**

**Decision point 2: accept "the primary metric reports `robospatial` only" or not.**
With the total score as primary metric, detection is impossible (§4.2). Narrowing the primary metric to `robospatial`
is statistically correct and grounded in attribution, but **the report must state this is a pre-registered narrowing, not post-hoc picking**.

---

## 7. When it should not be done

- **If decision point 2 is not accepted** — then there is no primary metric with power; do not run.
- **If only LoRA is possible** — conclusions cannot be extrapolated to the paper's full-parameter config, and the whole point of C is a same-starting-point comparison.
- **If A′'s conclusion is still worth digging into** — C spends 4–5 days answering "does the score move",
  while A′'s item (normalization inconsistency + each of its two fixes has a cost) is an **algorithm-side result**
  that can keep advancing with zero GPU, and **is the only thing on this line nobody else has done**
  (`P7_PRIOR_ART.md`: no implementation found of any GFlowNet-style objective used on multi-turn tool trajectories).

---

## 8. Limitations

- **The D ≈ 15% in §4.2 is an estimate**, taken between same-model run-to-run (6.9–10%) and tool swap (36.5%).
  The real value is only known after running; if D is larger, power on `robospatial` drops too.
- **The 4-GPU SFT time in §5 is an estimate**, not measured.
- **The difference between the two Step 4 arms is not only the objective** (the KL item, §1); it must be handled explicitly via ① or ②.
- **The gate only measured the starting point** (`π_θ = π_old`); C′'s behavior under real training dynamics is unverified.
- **`d` is the framework-difference proxy, not training drift** (`P7_ROUTE_C_GATE.md` §5).
