# SFT training study notes

> Scope: batch-related config, where GPU memory goes, the ZeRO trade-off, where the config comes from, framework layering.
> Setting: SpaceTools Phase-1 SFT, Qwen2.5-VL-3B, 4× A6000.
> Basis: `SpaceTools-SFT` @ `b7ebbf3` (GitHub source), paper Table 4/6, 2× A6000 measured run report.
> Date: 2026-09-08

---

## 1. `per_device_train_batch_size` and `gradient_accumulation_steps`

Both are there to make up **"how many samples one parameter update uses"**.

| | Meaning | Effect |
|---|---|---|
| `per_device_train_batch_size` | How many samples each GPU pushes through one forward/backward | **Sets peak GPU memory** |
| `gradient_accumulation_steps` (short: `ga`) | How many gradient accumulations before one weight update | Uses almost no GPU memory, **trades time for GPU memory** |

```
global batch = per_device × ga × number of GPUs
```

**This product determines the mathematical result of training; how it is split among the three factors does not change the result, only GPU memory and speed.**

What `ga=2` actually does:

```
forward/backward (2 samples) → keep gradients, no update
forward/backward (2 samples) → add gradients → only now optimizer.step()
```

Equivalent to taking 4 samples at once, but GPU memory is only counted for 2.

---

## 2. Why more GPUs is faster

In this project the global batch is locked at 8, split evenly across GPUs:

| GPUs | per_device | ga | per_device × ga<br>(samples each GPU must process) | GPU memory | Measured/estimated time |
|--:|--:|--:|--:|---|--:|
| 2 | 2 | 2 | **4 samples** | Tight (measured headroom 0.6 GiB) | 9.3 h (extrapolated from measurement) |
| **4** | **2** | **1** | **2 samples** | Roomy (~10 GiB) | **5–6 h** |
| 8 | 1 | 1 | **1 sample** | Roomiest | 3–4 h |

The GPUs work **in parallel**, so the time per update depends on "how many samples a single GPU must process", not on the total of 8.
4 → 2 → 1, halving each time.

**Analogy**: one round = moving 8 boxes. 2 people each carrying 2 boxes have to make 2 trips (that is `ga=2`);
4 people each carrying 2 boxes finish in 1 trip. `ga` is "when there aren't enough people, have the same person make more trips" —
the boxes still all get moved (same mathematical result), but the time is serialized.

**Two places where you cannot extrapolate linearly:**

1. **Not exactly 2×.** At the end of every update all GPUs must all-reduce gradients; more GPUs means more sync overhead.
   Measured 2 GPUs 9.3 h, 4 GPUs estimated 5–6 h: close to 2× but not quite.
2. **The 8-GPU row is theoretical.** The dataset has only 4 shards; beyond 4 GPUs some ranks get no data.
   **For this project, 4 GPUs is the practical optimum.**

---

## 3. How per_device sets peak GPU memory

```
GPU memory = fixed part (does not change with it) + per-sample part (grows linearly with it)
```

**Fixed part** — parameters in bf16 + gradients + optimizer state (already sharded across GPUs by ZeRO-2). Same size whether you give it 4 samples or 1.

**Linear part** — activations, and the real culprit: **the logits tensor**.

```
logits GPU memory ≈ per_device × sequence length × vocab size × 2 bytes
```

Qwen2.5-VL has a vocab of **151,936**, so each token of each sample costs 0.3 MB. Plugging in ~6900 tokens:

| per_device | logits spike |
|--:|--:|
| 1 | ~2 GiB |
| 2 | ~4 GiB |
| **4** | **~8 GiB** ← this is what blew up in the measured run |

Measured with `per_device=4` on A6000: **steady state 39.8 GiB + spike 8.4 GiB ≈ 48 GiB**, exactly the GPU's capacity; both GPUs crashed together.
So the script caps `per_device` at 2.

**Two corollaries:**

1. **It is a transient spike, not steady state.** `nvidia-smi` sampling once a second will very likely miss it, but it is what triggers the OOM.
2. **`cutoff_len` multiplies with it.** The current `cutoff_len: 8192` is 19% above the longest observed sequence of 6900;
   on a sample at full length, the `per_device=2` spike grows from 4 GiB to 4.9 GiB.
   That is the arithmetic behind "2 GPUs (headroom 0.6 GiB) are close to certain to OOM".

### ⚠ The 6900 figure is back-solved, not measured

When `per_device=4` OOMed, PyTorch reported the size of the failed allocation (~8.4 GB); back-solving with the formula above:

```
8.4e9 bytes ÷ (4 × 151936 × 2 bytes) ≈ 6900 tokens
```

Properties:

- **It is the longest batch in those 30 steps**, not the average. Fine for computing OOM headroom (headroom is computed against the worst case),
  **but it must not be used as the "typical sequence length"**.
- It relies on one assumption: that the failed allocation really was the logits tensor. Order of magnitude and shape match, but it has not been independently verified.

> The earlier conclusion that training is "overhead-dominated", from estimating MFU with ~3300 tokens, was overturned by this number.
> To confirm it, run the ckpt's tokenizer over `train.json` and get the length distribution (mean / p95 / max); a few minutes of work.

---

## 4. Where the concrete per_device and ga values come from

```bash
PER_DEVICE=$(( 8 / GPU_COUNT ));  [ "$PER_DEVICE" -gt 2 ] && PER_DEVICE=2
GRAD_ACCUM=$(( 8 / (GPU_COUNT * PER_DEVICE) ))
```

| GPUs | 8 ÷ GPUs | per_device | ga |
|--:|--:|--:|--:|
| 8 | 1 | 1 | 1 |
| 4 | 2 | 2 | 1 |
| 2 | 4 → **capped at 2** | 2 | 2 |

The 8-GPU and 4-GPU values are just the division, nothing else considered. **Only the cap at 2 was added by hand** —
2 GPUs should get 4, but `per_device=4` OOMed in the measured run, so it is held at 2 and the difference is handed to `ga=2` to make up.

**Priority: leave `ga` alone if you can** (it is slow); first push `per_device` to the maximum GPU memory allows; only when it can't go higher let `ga` cover the rest.

---

## 5. Why larger ga is slower, and what exactly is slow

**What is slow is not ga itself, but "fewer samples fed per forward/backward".**

Both cases process 2 samples per GPU:

- `per_device=2, ga=1` — compute 2 samples at once
- `per_device=1, ga=2` — split into two passes, 1 sample each

Total compute is the same, but the latter pays these **fixed overheads twice**:

- kernel launches, layernorm/softmax and other overheads that don't scale with batch
- gradient checkpointing recomputation
- data loading and batch assembly

And the matrices are "thinner", so GPU parallelism isn't fully used — this is the main reason.

> ⚠ **For this project the penalty may be small**: a single sample already has ~6900 tokens, the sequence dimension already keeps the GPU fairly well fed,
> and the parallelism gain from stacking one more sample in the batch is limited. Measured `per_device=2` gives MFU 32%;
> `per_device=1 + ga=2` **has not been measured, the difference is unknown**.

**So what makes 2 GPUs twice as slow as 4 is fewer GPUs, not ga** — ga is just the passive consequence of having fewer GPUs and still needing a global batch of 8.

---

## 6. Why the global batch is hard-coded to 8

It was not picked at random; **it is there to match the paper**, and this time two independent sources agree:

1. **Paper Table 6** says Batch Size = 8
2. **Upstream `run_sft.sh`** defaults to `per_device=1, ga=1, NUM_GPUS=8` → 1×1×8 = 8

> Table 6 and the code have already disagreed four times elsewhere (lr 1e-5 vs 2e-5, Epoch 2 vs 3.42, step count exceeded, sample count).
> **batch=8 is one of the few things both sides agree on**, so this number is more trustworthy than the others.

**Why it can't be changed casually**: the global batch, `lr=2e-5` and `max_steps` are a tuned set; changing the batch
is equivalent to switching to a different set of hyperparameters. And this ckpt is the **shared starting point π_ref** for the two-arm RL comparison; change the starting point and none of the later results can be compared with the paper.

If you don't intend to claim a reproduction of SpaceTools, set the batch to whatever you like (and retune lr to match).
**It is the "reproduction" goal that pins it, not a technical limitation.**

---

## 7. ZeRO-2 vs ZeRO-3

**Common misconception: ZeRO-3 communicates faster. It's the opposite — ZeRO-3 communicates more.**

| | What is sharded | Communication volume per step | Communication pattern |
|---|---|--:|---|
| **ZeRO-2** | Gradients + optimizer state (each GPU keeps a full copy of the parameters) | **2Ψ** | One reduce-scatter + one all-gather |
| **ZeRO-3** | Additionally **the parameters themselves** | **3Ψ** (1.5×) | Forward all-gather **per layer**, backward **again**, plus gradient reduce-scatter |

(Ψ = parameter count)

ZeRO-3 is slower for two reasons:

1. **More volume** — 1.5×
2. **Worse pattern** — split into one small communication per layer, **latency-sensitive**. A6000s talk over PCIe (no NVLink),
   high latency and low bandwidth; this kind of fragmented communication suffers the most

**What ZeRO-3 buys is GPU memory**: parameters are sharded too, so you can train large models that otherwise don't fit.
**It is a "use only when it doesn't fit" tool, not an optimization.**

### Which one the paper used

**The paper doesn't say.** Table 6 only lists batch / lr / epoch / warmup / KL / length / GPU count;
**no mention of DeepSpeed or ZeRO anywhere**, and the main text doesn't mention it either. z3 appears only in the code:

```
run_sft.sh:210   deepspeed: examples/deepspeed/ds_z3_config.json
```

Why the code picked z3 **can only be guessed; there is no source to cite**:

1. LLaMA-Factory's default full-finetuning example is z3 — very likely just copied over
2. They ran on 8× A100-80 with NVSwitch 600 GB/s, where z3's fragmented communication costs almost nothing
3. z3 saves the most GPU memory and is least likely to OOM; a safe "always runs" default

> A 3B model on 80 GB GPUs has plenty of room with z2 too, so **on their hardware z2/z3 very likely make no difference**.
> This looks more like no choice was made than like a choice was made.

**Our change: z3 → z2**, because A6000 has no NVLink and the 3B model fits in 48 GB with z2.

---

## 8. `use_reentrant_gc`

**It selects which implementation gradient checkpointing uses.**

Gradient checkpointing itself = don't store intermediate activations in the forward pass, recompute them in backward — **trade compute for GPU memory**.
PyTorch has two implementations:

| | Mechanism | Status |
|---|---|---|
| `use_reentrant=True` | Old, relies on autograd re-entrance | Being deprecated in PyTorch; default will flip to False in the future |
| `use_reentrant=False` | New, relies on saved-tensor hooks | Better compatibility, officially recommended |

**We changed it** (upstream doesn't set it; LLaMA-Factory defaults to `True`):

```yaml
+ use_reentrant_gc: false
```

The reason given in the report was "faster". **This has no measurement behind it**; it was a judgment at the time, not measured.

The more solid reason is a classic silent pitfall of the reentrant version: **if none of a checkpoint segment's inputs require gradients,
the backward is skipped entirely, with no error**. This config has exactly `freeze_vision_tower: true` +
`freeze_multi_modal_projector: true`, the shape most likely to hit it.
(In practice the embedding is trainable, so it very likely won't trigger, but there's no need to keep the risk.)

**It does not change the mathematical result** — both implementations compute the same gradients, only the recomputation path differs.

---

## 9. Where all this config comes from

Three sources, layered on top of each other:

### ① Upstream `run_sft.sh` (the vast majority)

`sft_config.yaml` **is not a standalone file in the repo**; it is a heredoc in the script **generated fresh on every run** (from line 238).
`finetuning_type` / `freeze_*` / `cutoff_len` / `lr` / `streaming` / `save_steps` are all hard-coded there.

### ② Paper Table 6 (covers only seven or eight numbers)

Batch 8, lr, Epoch, Warmup 0.1, cosine, Max Prompt/Response 8192, #GPU 8.
**Nothing else is specified** (including ZeRO stage, cutoff_len, image_max_pixels, eval settings).

### ③ Deviations we added (five)

| Change | From → to | Nature |
|---|---|---|
| `per_device` / `ga` | hard-coded 1/1 → derived from GPU count | **Fix** — the original values would silently become global batch 4 on 4 GPUs |
| `deepspeed` | z3 → z2 | Performance, A6000 has no NVLink |
| `save_only_model` | false → true | Disk; the cost is **no resuming training** |
| `eval_steps` | 5 → 500 | 3000 steps don't need 600 evals |
| `use_reentrant_gc` | unset → false | Compatibility |

**None of the four changes the mathematical result of training**; the first one corrects something that was already wrong.

> **Layer summary: the paper fixed a few key hyperparameters, the code fills in the rest, and we only touched the engineering layer.**

---

## 10. Framework layering

`run_sft.sh` **is not provided by LLaMA-Factory**.

**LLaMA-Factory (the framework) provides:**

- the `llamafactory.cli train` entry point
- the **field definitions** of `sft_config.yaml` — keys such as `finetuning_type` / `freeze_vision_tower` /
  `cutoff_len` / `use_reentrant_gc` / `deepspeed` are the framework's schema
- `examples/deepspeed/ds_z2_config.json` / `ds_z3_config.json`
- the registration mechanism of `data/dataset_info.json`

**Written by the SpaceTools authors themselves:**

- `scripts/spacetools/run_sft.sh` — **this directory does not exist in upstream LLaMA-Factory**
- downloading `siyich/spacetools-sft`, rewriting the system prompt per `toolshed_config.yaml`,
  v1 filtering out 887 robot samples
- the heredoc that generates `sft_config.yaml` (**the values that go in were chosen by them**)
- the Phase 3 ckpt fix-up (delete `text_config`, set `tie_word_embeddings`, copy `preprocessor_config.json`)

**The accurate way to put it: the fields belong to the framework; the values and the pipeline belong to the paper's authors.**

### Overall layering

| Layer | What it is | What it does |
|---|---|---|
| **HF Transformers / Trainer** | Bottom layer | Model, optimizer, parameters such as `per_device` |
| **LLaMA-Factory** | SFT/finetuning framework | Data loading, multimodal template (`template: qwen2_vl`), training loop, DeepSpeed integration |
| **DeepSpeed** | Distributed backend | ZeRO sharding |
| **SpaceTools-SFT** | A **fork** of LLaMA-Factory | Adds `run_sft.sh`: data prep + config + ckpt fix-up |

The RL side has exactly the same structure:

```
SFT:  LLaMA-Factory  ←fork←  SpaceTools-SFT
RL:   verl           ←fork←  SpaceTools-RL
```

---

## 11. Compared with the paper's 8× A100, what distributed optimizations did 4× A6000 use

**Essentially none. We added no optimizations, only one hardware adaptation.**

| Technique | Paper 8×A100 | Ours 4×A6000 | Who decided |
|---|:-:|:-:|---|
| Data parallelism | ✓ | ✓ | Framework |
| ZeRO sharding | **z3** | **z2** | ← the only change |
| Gradient checkpointing | ✓ | ✓ | On by default in LLaMA-Factory |
| bf16 mixed precision | ✓ | ✓ | Config |
| flash-attn 2 | ✓ | ✓ | `flash_attn: auto` default, used if installed |
| Frozen vision tower + projector | ✓ | ✓ | Paper design, trainable params 4.07B → 2.55B |
| Streaming dataset | ✓ | ✓ | Config |
| Gradient accumulation | ✗ (ga=1) | ✗ (ga=1 on 4 GPUs too) | Derived |

**z3 → z2 isn't an "optimization" either; it's a trade-off better suited to PCIe** — spend more GPU memory to get rid of that 1.5× fragmented communication.
On NVSwitch the trade isn't worth it; here it is.

### Things that would actually speed it up but weren't used

(All defaults below were checked against `hparams/model_args.py` / `data_args.py` in `SpaceTools-SFT@b7ebbf3`)

| Option | Default | What it does | Why it's off |
|---|---|---|---|
| `enable_liger_kernel` | `False` | **Fused cross-entropy, the logits tensor is never materialized** — exactly kills that 8 GiB spike | Fused ops change numerics, π_ref would carry a deviation |
| `packing` / `neat_packing` | `None` / `False` | Packs short samples into one, recovering much of the waste from `cutoff_len=8192` | Changes attention-mask semantics, **definitely changes the mathematical result** |
| `disable_gradient_checkpointing` | `False` | 4 GPUs have ~10 GiB headroom; turning it off could be 20–30% faster | The headroom can't absorb it |

Liger is the biggest pity — it targets exactly this project's bottleneck. But this ckpt is the shared starting point of the two-arm RL,
**introducing numerical deviation to run 20% faster isn't worth it**.

> **Honest conclusion: we are running the same job on fewer, weaker GPUs, squeezing it in with ga and z2, not running it faster through optimization.**
> 4 GPUs 5–6 h vs the paper's 8 GPUs 3–4 h; that ratio is essentially the hardware gap itself.

---

## Appendix: three uncertain points in these notes

Written down so they don't later get cited as measured facts.

| Item | Status | How to confirm |
|---|---|---|
| **~6900 tokens** | Back-solved from the OOM error, relies on the assumption "the failed allocation is the logits"; and it's the **longest**, not typical | Run the ckpt's tokenizer over `train.json` for the length distribution |
| **`use_reentrant=False` is faster** | A judgment, not measured. The compatibility reason is stronger | Run 30 steps of each with the same config and compare `s/it` |
| **The speed penalty of `ga` is small on this workload** | Reasoning (6900 tokens already saturate the GPU); `per_device=1+ga=2` has never been run | Run one 30-step `per_device=1, ga=2` on 4 GPUs |

Two more, adjacent to these notes, known but open:

- **`max_steps` 3000 (code) vs Epoch 2 → 1755 (Table 6)** — this run used 3000
- **`lr` 2e-5 (code) vs 1e-5 (Table 6)** — this run used 2e-5
