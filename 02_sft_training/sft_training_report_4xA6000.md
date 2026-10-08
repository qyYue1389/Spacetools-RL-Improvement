# REPORT_B — 4× A6000 full SFT (3000 steps) execution report

Follows on from `REPORT_A.md` (stages A–C on 2× A6000: environment build + 30-step probe).
This document records the full execution of the 4-GPU full training run, the acceptance-check results, and **several measured findings that overturn earlier documents**.

- Execution date: 2026-09-08
- Machine: Vast.ai instance, 4× RTX A6000 (49140 MiB, sm_86), driver 570.181, 377 GiB RAM, 119 G volume
- Conclusion: **training succeeded, all acceptance checks passed, the checkpoint can be used as π_ref**

---

## 0. In one sentence

All 3000 steps completed, exit code 0, took **7 hours 45 minutes**; `eval_loss` decreased monotonically over six points
**0.168 → 0.045**; global batch independently re-checked to be 8; the four Phase 3 fixes are in place.

But **there was one close call during the run**: GPU2 ran right at the GPU memory limit the whole time, with only 1.4 GiB headroom (see §3.1).

---

## 1. Deliverables (per `NEXT_MACHINE.md` §10)

### Hardware

```
index, name, driver_version, compute_cap, memory.total
0..3, NVIDIA RTX A6000, 570.181, 8.6, 49140 MiB
```

### Launch line

```
GPUs 4 · per_device=2 · ga=1 · global batch 8 ✓
```

### B.1 Data definition (all six items pass)

| Item | Expected | Measured |
|---|---|---|
| `<tools>` block length | 8595 | ✓ 8595 |
| sha256 prefix | `72c71c806f64e162` | ✓ match |
| Contains `${obj_name}_detections` | Yes | ✓ |
| Number of tools | 11 | ✓ 11 |
| system identical across the whole table | 1 unique value | ✓ all 7020 identical |
| Leftover robot samples | 0 | ✓ 0 |

Sample pipeline: **7907 raw → 887 robot-tool samples filtered out → 7020 go into training**.

### Speed

| | Value |
|---|---|
| `train_runtime` | 27929.7 s = **7.76 h** |
| Mean | **9.31 s/it** |
| Measured steady state (32 steps / 300 s independent window) | 9.38 s/it |
| `NEXT_MACHINE.md` expectation | 5–6 h |

**30% slower; the cause has been found and is not actionable**, see §3.2.

### GPU memory

| GPU | Peak | Share | Headroom |
|---|---|---|---|
| 0 | 36830 MiB | 75% | 12.0 GiB |
| 1 | 37090 MiB | 76% | 11.8 GiB |
| **2** | **47730 MiB** | **97%** | **1.4 GiB** ⚠️ |
| 3 | 36950 MiB | 75% | 11.9 GiB |

### loss

| | |
|---|---|
| train loss | 0.8621 (step 5) → **0.0625** (step 3000) |
| `train_loss` (mean over the whole run) | 0.19167 |
| Logged points | 600, `global_step=3000/3000` |

`eval_loss` six points, **monotonically decreasing over the whole run, not a single uptick**:

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---|---|---|---|---|---|
| eval_loss | 0.1680 | 0.1064 | 0.0975 | 0.0684 | 0.0559 | **0.0452** |
| Change vs previous | — | −37% | −8% | −30% | −18% | −19% |

⚠️ `val_size` is only 20 samples; single points are noisy, look only at the trend.

### Independent re-check of the global batch

Not relying on what the config says about itself; back-derived from HF Trainer output:

```
train_runtime × train_samples_per_second = 27929.7 × 0.8590 = 23992
expected = max_steps × global batch = 3000 × 8 = 24000
deviation 0.04%  ✓
```

Based on `transformers/trainer.py:5709` — `num_train_samples = args.max_steps *
total_train_batch_size`, and `total_train_batch_size` is computed from the **real world_size at runtime**,
so this check catches "silently running a different global batch on the wrong number of GPUs".

⚠️ It is a **config-level** check; it does not prove the dataloader actually fed 24000 samples.

⚠️ **Do not look at the `epoch` field** (reports 3.122). `DATASET_SPEC` registers the dataset 3 times
(`run_sft.sh:236`); HF computes epochs from 7020×3=21060, so it never matches; it is a false alarm.

### Phase 3 four items (built into upstream, we changed not one line)

| Item | Status |
|---|---|
| Remove `text_config` from `config.json` | ✓ log has `Removed text_config` |
| `tie_word_embeddings = true` | ✓ log has `Set tie_word_embeddings=True` |
| Copy `preprocessor_config.json` | ✓ present |
| `model_type` | ✓ `qwen2_5_vl` (**not** `qwen2_5_vl_text`, which crashes sglang) |

### Format smoke test + base comparison

Sampled 12 (seed=0, greedy):

| Metric | SFT ckpt | base Qwen2.5-VL-3B-Instruct |
|---|---|---|
| `<think>` tag | **100%** | 41.7% |
| `</think>` closed | 100% | 33.3% |
| `<tool_call>` | 100% | 83.3% |
| Valid JSON | **100%** | 83.3% |
| Valid tool name | **100%** | 75.0% |

Pass line 90% (derived from the ground-truth rate of 99.8%). Discriminating item `<think>`: **+58 percentage points**.

⚠️ Samples are taken from the training set; **this only judges "whether SFT took effect", it is not a capability measure**.
Real scores can only come from SpaceTools-RL's `run_eval.sh`.

The distribution of tools called by base also says something: base heavily misuses `vision_ops.index_at` (5/12)
and produced a nonexistent `answer` tool; the ckpt concentrates on `roborefer.detect_one` (8/12)
and `depth_estimator.*`, consistent with the task types.

### Deviations

- **liger not enabled** (explicit user request)
- **None of the B.4 forbidden items touched**: `packing` / `neat_packing` / `finetuning_type` /
  `freeze_vision_tower` / `freeze_multi_modal_projector` / `cutoff_len` /
  `image_max_pixels` / `learning_rate` / `max_steps`

### Environment fingerprint

```
torch 2.9.1+cu128 · transformers 4.57.1 · deepspeed 0.19.6
flash_attn 2.8.3.post1 · llamafactory 0.9.5.dev0
SpaceTools-SFT git b7ebbf320bb130c230856e61107a566bc119e4d8
```

### checkpoint hashes

```
779717e1d8f9611009066b821f338c0af2a80215d8abd630f21d9d3b28540e85  model-00001-of-00002.safetensors
b1ae12021922f489237d50b4f4c25559821b98cc8589cad0aa4c6dce573b0ba6  model-00002-of-00002.safetensors
```

Full list in `EVIDENCE_*/CKPT_SHA256SUMS`.

---

## 2. Artifact locations

```
/workspace/experiments/full/
├── sft_checkpoint/                 final ckpt (7.6 G) — this is π_ref
│   ├── model-0000{1,2}-of-00002.safetensors
│   ├── config.json                 Phase 3 applied
│   ├── chat_template.json          ← added by us, see §4.1
│   ├── toolshed_config.yaml        the 11 v1 schemas
│   └── checkpoint-{500,1000,1500,2000,2500,3000}/   7.6 G each, 46 G total
├── sft_data/
│   ├── data/train.json             7020 samples, schema injected
│   ├── images/{a,b}/               7459 images
│   └── hf_download/                raw download (6.0 G, redundant, can be deleted)
└── EVIDENCE_20260908_08{1901,2425}/  evidence, 1.1 M each (two FINALIZE runs, equivalent content)

/workspace/runpod-handoff/probe3000.log     full training log (incl. launch line)
/workspace/runpod-handoff/gpu_probe3000.csv 28048 GPU samples
/workspace/val/{FINALIZE.sh,smoke_check.py} acceptance-check scripts (fixed, see §5)
```

Disk: 119 G volume, 84 G used, **36 G left**.

---

## 3. Measured findings that overturn earlier documents

> Per `NEXT_MACHINE.md` §9.1 "when unsure, choose the conservative option, and write it down".

### 3.1 ⚠️ Most important: GPU memory headroom across the 4 GPUs is not uniform; the worst rank has only 1.4 GiB left

`NEXT_MACHINE.md` §4 expected the 4 GPUs to have "~10 GiB headroom". **Measured GPU2 peak 47730 / 49140 MiB
(97%), only 1.4 GiB headroom.**

And it is not a momentary spike: **about 90% of samples (25122 / 28000) are above 45 GiB**,
from minute 49 until the end of training. The other three GPUs stayed stable at 36.8–37.1 GiB.

**Suspected mechanism**: the high-water-mark effect of the PyTorch caching allocator — rank 2 hit an extra-long sequence or large-image batch early,
the allocator grew and never returned the GPU memory. Under `streaming: true` the dataset has only 4 shards, one per each of the 4 ranks,
and uneven sequence-length distribution across shards amplifies this effect.

**What this means for the next person**:

- This time we were only 1.4 GiB from OOM, and `save_only_model: true` means **one OOM = start over, 7.75 hours**
- Do not plan on "~10 GiB headroom". The real planning basis should be **worst rank ~1.4 GiB**
- This also explains why `per_device=4` OOMs — it is not "3 GiB short", there was never any headroom to begin with
- To reproduce, suggest adding `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to mitigate fragmentation,
  or monitor per-GPU peaks during training instead of only the overall figure

### 3.2 The 5–6 h expectation was wrong, the real figure is 7.75 h — but not actionable

`NEXT_MACHINE.md` §4 got 5–6 h by linear extrapolation from 2-GPU measurements. The extrapolation missed the change in gradient accumulation:

| | 2 GPUs (REPORT_A measured) | 4 GPUs (this run) |
|---|---|---|
| per_device × ga | 2 × **2** | 2 × **1** |
| Samples per GPU per step | 4 | 2 |
| s/it | 11.22 | **9.31** |
| Throughput | 0.713 samples/s | **0.859 samples/s** |

**Doubling the GPUs only bought 1.20× throughput, 60% scaling efficiency.**

Cause: `ga` dropped from 2 to 1, so the `no_sync` amortization of the first micro-batch disappears;
**every step has to do a full ZeRO-2 gradient reduction**; ZeRO-2 also shards optimizer state 4 ways instead of 2,
so communication volume itself goes up.

**Why not actionable**: global batch is locked to 8, `per_device × ga × GPUs = 8`,
and on 4 GPUs the only solution is `2 × 1`. Going back to 2 GPUs is 9.3 h (slower); 8 GPUs are not available.
**7.75 h is already the fastest of the available options**, unless the global batch is changed — and that would ruin π_ref's definition.

### 3.3 The inference "GPU utilization 100% → compute-bound" does not hold

`NEXT_MACHINE.md` §7 used "median utilization 100%" to overturn `TASK.md`'s "overhead-bound",
and wrote the conclusion as "compute-bound". **The inference itself is not rigorous.**

`nvidia-smi` utilization measures "whether some kernel is running", and
**NCCL all-reduce busy-wait spins, which also counts as 100%**. So 100% only says there is always a kernel on the GPU;
it cannot distinguish compute from communication.

The 60% scaling efficiency in §3.2 is exactly a counterexample: if it were really purely compute-bound, doubling GPUs should give close to 2× throughput.
`analyze_probe.py` still prints "→ compute-bound" for every GPU; **that conclusion line should be deleted or rewritten**.

### 3.4 Checkpoints are 7.6 GiB each, not 15.18

`NEXT_MACHINE.md` §1 says 7.6 GiB each, §7 then says "actually 15.18 GiB each"; the two contradict each other.
**Measured 7.6 GiB each**; §7 does not apply.

6 intermediate ckpts 46 G + final weights in the root dir 7.6 G ≈ **53 G**, consistent with the §1 estimate.
The 15.18 in §7 most likely added one copy in `checkpoint-N/` and one in the root dir,
and the root-dir copy is only written at the end of training.

### 3.5 The root cause of HF rate limiting is the anonymous IP quota, not concurrency

`NEXT_MACHINE.md` §2.1 attributes rate limiting to concurrency ("with 32 concurrent, HF rate-limits after about 2000 files",
"don't go above 8 concurrent"). **Measured: 8 concurrent still gets limited**, 1753 cumulative 429s. HF's verbatim response:

> We had to rate limit your IP (…). To continue using our service, **create a HF account
> or login to your existing account, and make sure you pass a HF_TOKEN** if you're using the API.

This is a **per-IP anonymous quota**. The first round downloaded 7463 files in 22.1 minutes, but 32 failed the size check;
the second round (still anonymous) only fixed half of them. **After logging in with an HF token, the remaining 16 files downloaded in seconds, zero failures.**

**What this means for the next person**: the concurrency back-off in `prefetch.py` is right, but the real fix is
**use a token**. `NEXT_MACHINE.md` §2.1 should add a line "set `HF_TOKEN` first".

### 3.6 "Global batch = 8 is the paper's definition" — the basis is the upstream script default, not the paper text

Both `run_sft.sh` and `NEXT_MACHINE.md` say "paper definition, immutable". Traced back, the basis is:

```
upstream run_sft.sh.orig:
  42:  NUM_GPUS="${NUM_GPUS:-8}"        # docs say "GPUs for training (default: 8)"
  235: per_device_train_batch_size: 1
  236: gradient_accumulation_steps: 1
  → 1 × 1 × 8 = 8
```

**Nothing in the paper text available locally says "global batch = 8".** The only time `TASK.md` cites the paper's
Table 6 is to argue something else (SFT does not need the vision tools deployed).

And this inference has a weak spot: `NUM_GPUS` is declared on line 42 and **never referenced by any other line**; it is a no-op
(what actually decides the GPU count is `FORCE_TORCHRUN=1` + the number of visible GPUs). That "8" lives only in a comment and a dead variable.

**But the practice of "locking it" is still correct**, for reasons unrelated to the paper:

1. The global batch of the upstream code **varies with the machine** — `pd`/`ga` hardcoded to 1, `NUM_GPUS` a no-op,
   so the original code would **silently** train with global batch = 4 on 4 GPUs, 2 on 2 GPUs, without reporting any error
2. π_ref is the shared starting point of both RL arms; what the batch is can be discussed, but it must be fixed before running
3. Change the batch and the model changes — lr fixed at 2e-5, cosine over the full 3000 steps; changing the batch changes the optimization dynamics

**Recommendation**: change "paper definition" in `run_sft.sh` and `NEXT_MACHINE.md` to
"upstream script default definition (8 GPUs × pd1 × ga1)", so the next person does not think it has paper backing.
If the paper text is available, worth a look.

---

## 4. Upstream omissions and our patches

### 4.1 Phase 3 misses copying `chat_template.json`

Upstream Phase 3 copies `preprocessor_config.json` from the base model, but **does not copy
`chat_template.json`** (processor-level template, which sglang reads when loading a VLM).

The insidious part: the checkpoint has `chat_template.jinja` (tokenizer-level, written automatically by transformers 4.57),
so checks like "chat template exists" **still go green**, while the RL side may crash for lack of the
processor-level template.

**Added** (byte-for-byte identical to base):

```bash
BASE=$(ls -d /workspace/hf/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/*/ | head -1)
cp "$BASE/chat_template.json" /workspace/experiments/full/sft_checkpoint/
```

And added it to the required-file list in section 3 of `FINALIZE.sh`.

### 4.2 The PATH landmine in `launch_probe.sh` (we stepped on it ourselves)

The first training launch failed immediately; Phase 1 reported `ModuleNotFoundError: No module named 'huggingface_hub'`.

**The root cause is how it was called**: `launch_probe.sh` was called from a **shell that had already run `conda activate spacetools-sft`**;
its line 4 `export PATH=/opt/conda-st/bin:$PATH` puts base's bin
ahead of the activated env; the immediately following `conda activate spacetools-sft`, because it is "already activated",
**becomes a no-op** and does not move the env's bin back to the front → `python3` resolves to base.

**The most insidious part**: `CONDA_DEFAULT_ENV` still shows `spacetools-sft`,
the env **looks activated**, only `command -v python3` gives it away.

The comment in `conda_hook.sh` predicts exactly this pitfall, but the same pattern also exists in line 4 of `launch_probe.sh`
itself — harmless when called from a clean shell, a landmine when called from an activated shell.

**Fixed**: removed the PATH prepend, install the hook via absolute path instead, and added an assertion.
Verified under three parent-shell conditions; also verified that the rejection path really fires (not dead code).

---

## 5. List of changes to the acceptance-check scripts

Original files backed up at `/workspace/val/*.bak`.

### `FINALIZE.sh`

| # | Problem | Impact | Fix |
|---|---|---|---|
| 1 | Same PATH landmine as §4.2 | **Asymmetric consequences**: sections 1–4 use only stdlib and still go all green under base python3; only section 5 `import torch` and section 8 `huggingface_hub` fail → "all acceptance checks pass + incomplete evidence" | Install hook via absolute path + assertion on `python3` resolution |
| 2 | `REPO_DIR` defaults to `/workspace/SpaceTools-SFT`, glob `sft_v1_*` | Auto-locating the experiment dir **always fails** (`run_sft.sh`'s `REPO_DIR` is `LLAMA_DIR/..` = `/workspace`; and this time the dir name is `full`) | Added `EXP_ROOT=/workspace`, glob covers both `sft_v1_*` and `full` |
| 3 | Logs searched only in `$EXP/*.log` and `scripts/spacetools/launcher.log` | **Does not collect `probe3000.log` or the launch line** — which are exactly the definition evidence the script itself emphasizes | Added `$HANDOFF/probe*.log` and `$HANDOFF/launcher*.log` |
| 4 | Required-file list lacks `chat_template.json` | See §4.1 | Added to list |

**Not fixed (cosmetic, does not affect conclusions)**:

- Section 4 reports "tool schema count ≈ 12 (v1 expects 11)": it counts `"name":` with a loose regex;
  the extra one comes from the template line demonstrating the format in the system prompt
  `{"name": <function-name>, "arguments": <args-json-object>}`. True value 11
- Section 3 prints "dtype None": transformers 4.57 renamed `torch_dtype` to `dtype`,
  and the script reads the old key. config.json actually has `dtype: bfloat16`
- `PRUNE=1` keeps `checkpoint-$KEEP_MID` but section 8's `ignore_patterns` does not upload it —
  once the pod is destroyed, that sanity-comparison ckpt is gone
- `TOT=$(du -sh -c $MIDS ...)`: `$MIDS` is unquoted and relies on word splitting

### `smoke_check.py`

| # | Problem | Impact | Fix |
|---|---|---|---|
| 1 | `census` counts **all** assistant turns, but the script only generates the **first** turn | Completely different distributions (all turns `tool_call` 63.3% / `<answer>` 36.7%; **first turn 99.8% / 0.2%**). The hardcoded 0.7 pass line would mark 75% of bad ckpts as ✓; and the "expected" list includes `<answer>`, while the correct behavior is that it almost never appears | Count only the first turn; thresholds derived from the ground-truth rate (`0.9×` pass / `0.5×` warning) → pass line this time 90% |
| 2 | No guard during training | Would load ckpt + base on `cuda:0`, 6.2 GiB each, while training already uses 35–36/48 GiB per GPU → **may hit OOM and kill training** | Refuse if `train.pid` is alive (exit code 2); `ALLOW_DURING_TRAINING=1` forces it |
| 3 | base goes through the Hub | Anonymous IP already rate-limited; may fail or re-download 7.1 GB | Added `resolve_model()` to resolve the local snapshot |
| 4 | `MARKERS` lacks `<think>` | Using "valid JSON" for the base comparison is brittle | Added `<think>`/`</think>`, discriminating item changed to `<think>` |

**The value of fix 4 was demonstrated by this run**:

```
valid JSON:  ckpt 100% vs base 83.3%  → 17 percentage points apart < 20 threshold → would misjudge "no clear gap" ✗
<think>:   ckpt 100% vs base 41.7%  → 58 percentage points apart              → judged correctly ✓
```

Without the change, this run would have **falsely reported failure**.

### ⚠️ I made a mistake in fix 3 (corrected; recorded as a warning)

The first version used `os.environ.setdefault("HF_HOME", "/workspace/hf")`.
But the Vast instance **already presets `HF_HOME=/workspace/.hf_home`**, and `setdefault` does not overwrite an existing value,
so the glob hit an empty dir and fell back to the Hub, **downloading the base model again (7.1 GB)**.

The fix was ineffective but **looked like success** (download succeeded, acceptance check passed); it was only found by comparing disk usage.
Changed to iterate over several candidate root dirs (`$HF_HOME`, `/workspace/hf`, `~/.cache/huggingface`)
and only accept a snapshot that actually contains `*.safetensors`.

The 7.1 G in `/workspace/.hf_home` is a purely redundant copy (same commit as `/workspace/hf`) and can be deleted.

---

## 6. Unfinished / open items

| Item | Benefit | Status |
|---|---|---|
| **Copy the ckpt off this machine** | Survival — everything is lost when the pod is destroyed | **Not done, highest priority** |
| Delete the redundant copy in `/workspace/.hf_home` | +7.1 G | Awaiting approval |
| Delete `sft_data/hf_download/` | +6.0 G | Awaiting approval (training finished, no longer needed) |
| `PRUNE=1` to clean intermediate ckpts (keep 1500) | +38 G | **Suggest doing this only after the ckpt has been copied off** |
| Merge the two `EVIDENCE_*` dirs | Tidiness | Equivalent content, keep one |
| Run SpaceTools-RL's `run_eval.sh` | Real capability scores | Tool stack not installed on this machine |

Upload command (built into section 8 of `FINALIZE.sh`):

```bash
HF_REPO=<username>/spacetools-sft-v1-4xa6000 HF_TOKEN=hf_xxx \
    bash /workspace/val/FINALIZE.sh /workspace/experiments/full
```

---

## 7. Three things for the next person

1. **Set `HF_TOKEN` first.** The anonymous IP quota will block you while downloading the dataset (7463 files),
   and the symptom is `LocalEntryNotFoundError`, which looks like a concurrency problem (§3.5).
2. **Call `launch_probe.sh` from a clean shell**, do not `conda activate` first.
   The script now has an assertion that will stop it, but understanding the cause is better than relying on the assertion (§4.2).
3. **Plan GPU memory headroom on the worst rank at 1.4 GiB, not ~10 GiB.**
   This run came close to OOM, and `save_only_model: true` means OOM = start over, 7.75 hours (§3.1).

And one methodological point — all four judgment principles in `NEXT_MACHINE.md` §9 came true this time:

- "Exit codes lie": the first launch of `launch_probe.sh` had rc=1 but the launcher printed only two normal log lines
- "A single pass is not evidence": the `setdefault` bug in `resolve_model` looked like complete success
- "It ran, but the numbers are wrong": `census` counting the wrong turns, and the base-comparison threshold in `smoke_check`,
  neither reports an error, they just give the wrong conclusion
- "Verify the actual state after every step": GPU2's 1.4 GiB headroom was only found by going through the csv after training ended;
  during the run every metric was green
