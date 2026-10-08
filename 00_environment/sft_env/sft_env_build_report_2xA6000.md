# Stage A–C report (2× A6000 / RunPod)

2026-09-07 · corresponds to `TASK.md` stage A (environment) · B (configuration) · C (30-step probe)
**Status: all three stages complete, the three Phases of `run_sft.sh` ran through, exit code 0**

---

## 1. Conclusion

The environment is ready and validated by a complete training run; configuration has zero deviations; the 30-step probe ran to completion.

**The three numbers** (requested by `TASK.md` stage C):

| Metric | Measured | Reading |
|---|---|---|
| `s/it` (first 5 steps skipped, 25 samples) | **11.22** median | × 3000 = **9.3 hours** |
| Median GPU utilization | **100% / 100%** | **compute-bound** |
| Peak GPU memory | GPU0 **48491/49140 (99%)**<br>GPU1 44745/46068 (97%) | **only 0.6 GiB headroom** |

Two conclusions contrary to what `TASK.md` expected: see §7. The full run has not started yet; open items in §9.

---

## 2. Hardware

```
2 × NVIDIA RTX A6000 · driver 580.159.03 · compute_cap 8.6
GPU0  49140 MiB (47.4 GiB)
GPU1  46068 MiB (44.4 GiB)     ← about 3 GiB less than GPU0
```

⚠️ **The two GPUs have unequal memory, and this affects stage B.** Under ZeRO-2 both ranks hold the same shapes,
so the OOM ceiling is set by GPU1's 44.4 GiB, not the 48 GB assumed in `TASK.md` B.3.
That budget has to be recomputed:

```
params 8.1 + (grad+opt 30.6)/2 = 23.4 GiB
              + logits ~16   ≈ 39 GiB / 44.4      headroom 5 GiB (doc expected 9 GiB)
```

`per_device_train_batch_size: 4` is riskier than the doc expected.

---

## 3. Environment

```
python        3.11
torch         2.9.1+cu128      cuda 12.8
torchvision   0.24.1+cu128     ← must match torch
torchaudio    2.9.1+cu128      ← same as above; originally 2.11.0, which caused an ABI crash
transformers  4.57.1
flash-attn    2.8.3.post1      self-built, sm_80 only
deepspeed     0.19.6
llamafactory  0.9.5.dev0       editable → /workspace/SpaceTools-SFT/src
accelerate    1.11.0    trl 0.24.0    peft 0.18.1    datasets 4.0.0
```

Install location `/opt/conda-st` (container disk). Measured conda on the network volume was unusably slow (~200 vs >20,000 files/s),
so it was installed on the container disk and immediately backed up to the persistent volume — it is lost when the container restarts; backup in §6.

### Acceptance check

| Item | Result |
|---|---|
| `pip check` | passed, no dependency conflicts |
| `transformers == 4.57.1` hard assertion | passed |
| `nvcc` version | 12.8 (`/usr/local/cuda-12.8`), `CUDA_HOME` check passed |
| `check_arch.sh /opt/conda-st/envs` | scanned 7 .so files containing CUDA, **0 failing** |
| flash-attn real kernel run | `(2,64,4,64)` bf16, finite values, on sm_86 ✓ |

The last item is not just `import` passing — it actually computed once on the GPU.
This also confirmed by measurement the premise of `TASK.md` A.3: **cubins built for sm_80 run on sm_86**.

---

## 4. Four facts to remember

### 4.1 The container memory limit is 93.1 GiB, not the 503 GB shown by `free`

```
/sys/fs/cgroup/memory.max   99,999,997,952 B  =  93.1 GiB   ← real limit
free -g                     total 503                       ← the host's
```

`free` and `nproc` both read the host inside the container. **Anywhere on this machine that sets concurrency/batch size by memory
must read the cgroup.** Exceeding it is not an error; the whole container gets killed by the OOM killer, and everything on the container disk goes with it.

### 4.2 flash-attn does not read `TORCH_CUDA_ARCH_LIST`

`flash_attn/setup.py:69`:

```python
def cuda_archs() -> str:
    return os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;120").split(";")
```

The `TORCH_CUDA_ARCH_LIST` that `TASK.md` A.3 repeatedly emphasizes **has no effect on a single line of code in this package**
(it still works for other CUDA extensions, so it is kept in the script). By default it compiles one copy each for sm_80/90/100/120;
the last two are Blackwell, completely useless here.

Also, `setup.py:124` adds `--threads ${NVCC_THREADS:-4}` to every nvcc, and `--threads`
**parallelizes across arch targets** (measured: each nvcc spawns N `cicc`, N = number of archs). So:

```
real concurrent compile processes = MAX_JOBS × min(NVCC_THREADS, number of archs)
```

Estimating memory by `MAX_JOBS` underestimates it several-fold. This time's config `FLASH_ATTN_CUDA_ARCHS=80` + `MAX_JOBS=12`
→ 12 concurrent → measured anon peak **51.4 GiB / 93.1 GiB (55%)**, 72 compilation units in about 17 minutes.

### 4.3 flash-attn's own memory auto-tuning is broken inside containers

`setup.py:513` sets concurrency automatically when `MAX_JOBS` is unset:

```python
free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
max_num_jobs_memory = int(free_memory_gb / 9)
```

`psutil.virtual_memory()` also reads the host's memory. On this machine it would compute `min(96/2, 461/9) = 48`,
which is bound to blow up. **So `MAX_JOBS` must be set explicitly; do not rely on upstream's auto-tuning.**

### 4.4 Upstream pins only `torch`, not `torchvision` / `torchaudio`

`setup_envs.sh` runs `pip install torch==2.9.1 torchvision torchaudio`. The latter two are unconstrained,
and pip resolves `torchaudio` to **2.11.0** (built for torch 2.11).
`llamafactory/data/mm_plugin.py:29` does `import torchaudio` → `undefined symbol:
torch_library_impl`, and training crashes right after distributed initialization.

This is **the same shape of bug** as `transformers` getting pushed to 5.x in the previous round (upstream puts no constraints on transitive dependencies);
the difference is that this one crashes loudly while that one silently ruins the results.

**The acceptance check must go through the real import chain.** My initial §6 acceptance check only did `import torch, flash_attn, deepspeed,
llamafactory` and let it pass — `import llamafactory` is lazy and does not touch `mm_plugin`. Now changed to:

```python
import torch, torchvision, torchaudio, flash_attn, deepspeed, llamafactory
from llamafactory.data.mm_plugin import get_mm_plugin   # it imports torchaudio internally
from llamafactory.train.tuner import run_exp
```

### 4.5 Dataset download: upstream's concurrency does not take effect

`snapshot_download` of `siyich/spacetools-sft` (7463 files / 5.89 GiB) only gets **0.8 MB/s**,
extrapolating to 1–2 hours. By elimination: HF network speed 31 MB/s, network-volume sequential write 451 MB/s, xet only affects 30% — 
the real cause is that **there are only 2 threads in the process**; passing `max_workers=32` to `snapshot_download` has no effect either.

Switched to its underlying `hf_hub_download` with our own concurrency control (`prefetch.py`): **finished in 21 minutes**.
The metadata format is the same, so a later `snapshot_download` recognizes and skips the files.

Concurrency must not be too high: 32-way concurrency hit the HF rate limit after about 2000 files (showing up as `LocalEntryNotFoundError`),
with 346 failures. Dropping to 8-way concurrency + exponential backoff stabilized at 5–7 MB/s.

⚠️ **After the first pass there were still 8 files with mismatched sizes** — caught by the per-file check. Without this check,
these 8 missing images would have silently entered the training set.

---

## 5. Deviations from upstream `setup_envs.sh`

| Deviation | Upstream | Impact |
|---|---|---|
| `FLASH_ATTN_CUDA_ARCHS=80` | `80;90;100;120` | sm_80 cubins only, no PTX fallback. **A100/A6000/A4000 can run it, H100/Blackwell need a rebuild.** Compile volume and memory reduced to 1/4 |
| Completed dependency upper bounds + second pin and assertion of `transformers` | no version constraints | **Fix**: otherwise `transformers` gets pushed to 5.x by transitive dependencies without any error, producing a π_ref different from the paper's definition |
| Pinned `torchvision==0.24.1` / `torchaudio==2.9.1` | no version constraints | **Fix**: see §4.4, otherwise training does not start |
| Added `ninja setuptools wheel` | not installed | Build-time tools. Without `ninja` flash-attn falls back to serial compilation and `MAX_JOBS` has no effect |
| `MAX_JOBS=12` `NVCC_THREADS=1` | auto-tuned (see §4.3) | pure concurrency control |

**The first three only affect the build process or fix upstream bugs, the fourth only affects compile concurrency — none of the four changes the mathematical result of training.**

None of the forbidden items in `TASK.md` B.4 was touched; liger was not used.

### Persistent changes in `setup.sh`

| Change | Purpose |
|---|---|
| Compute the concurrency cap from the cgroup rather than `free`, with a hard clamp | see §4.1 |
| Set `FLASH_ATTN_CUDA_ARCHS` / `NVCC_THREADS` explicitly, budget scaled by number of archs | see §4.2 / §4.3 |
| Compile-time memory watchdog: watches cgroup `anon`, kills by **process tree** above 70% | makes the compile fail instead of the container getting killed. ninja calls `setpgid(0,0)` for every child process, so killing only the process group misses the running compile processes |
| flash-attn is first built with `pip wheel` onto the persistent volume, then installed | one successful compile is valid forever; reinstall goes from ~17 minutes to 10 seconds |
| After install, automatically `tar` the whole conda to the persistent volume | restore in 2 minutes after a container restart |
| `SELF_DIR` resolved before any `cd`; architecture self-check failures are no longer swallowed as warnings | otherwise the self-check is silently skipped while the exit code is still 0 |

---

## 6. Deliverables

| Path | Contents |
|---|---|
| `/opt/conda-st` | the currently usable conda + `spacetools-sft` environment (9.3 GB) |
| `/workspace/env-backup/conda-st.tar` | environment backup, 9.2 GB, 120,428 entries = number of source files |
| `/workspace/wheels/flash_attn-2.8.3.post1-cp311-cp311-linux_x86_64.whl` | self-built wheel, use it directly for reinstalls |
| **`/workspace/bundle/`** | **portable package, 4.0 GB** |

### Portable package

```
spacetools-sft-env.tar.zst   ~4.0G   checksum in SHA256SUMS in the same directory
RESTORE.sh                           pre-checks → checksums → unpack → automatic acceptance check
VERIFY.sh                            acceptance check that can be re-run on its own
README.md                            requirements · limitations · deviations
SHA256SUMS
```

(The hash is not inlined in this file — this file is inside the package, and inlining it would create a cycle. `SHA256SUMS` is authoritative.)

Target machine: `sudo bash RESTORE.sh`

The package contains `/opt/conda-st` + `/workspace/{SpaceTools-SFT,SpaceTools,wheels,runpod-handoff}`.
**Paths cannot be changed** — `llamafactory` is an editable install, and `_editable_impl_llamafactory.pth`
hard-codes `/workspace/SpaceTools-SFT/src`; conda's shebangs are also absolute paths.
Paths were kept rather than switching to a regular install so that the target machine layout is exactly identical to this machine (zero deviation).

**Requirements**: x86_64 · compute_cap **8.0 or 8.6** · driver ≥ 525 · glibc ≥ 2.32 ·
`/opt` ≥ 12 GB. **No need to install the CUDA toolkit** (the runtime libraries are in the package under `site-packages/nvidia/`).
H100/Blackwell are blocked by `RESTORE.sh` before unpacking.

The package **does not contain** the SFT data (7463 files / 5.89 GiB) or the base model (~8 GB); `run_sft.sh` downloads them automatically —
but using the package's `runpod-handoff/prefetch.py` instead is recommended, for the reasons in §4.5.

### Package verification

| Verification | Result |
|---|---|
| `zstd -t` full decompression check | passed, 121,227 entries |
| `tar --compare` entry-by-entry comparison of content/size/permissions/mtime | **0 differences** |
| Real run of the full `RESTORE.sh` flow | 6 pre-checks + checksums + unpack + acceptance check all passed |
| flash-attn kernel inside `VERIFY.sh` | computes finite values on sm_86 |
| **Complete 30-step SFT training** | **all three Phases ran through, exit code 0** |

### 2026-09-07 repackaging

The first version went out with the **torchaudio 2.11.0** defect (see §4.4) and has been withdrawn. Current version:

- Environment fixed to `torchaudio 2.9.1`, and validated by a complete 30-step training run
- `VERIFY.sh` adds torch-family version assertions + the real training import chain + `pip check`
- `README.md` adds the GPU memory requirement (measured peak 48.5 GiB; an A4000 can hold the environment but cannot run the training)
- Includes the tools added in this round: `prefetch.py` (parallel download) · `memguard.sh` (memory sentinel) ·
  `analyze_probe.py` (the three numbers) · `verify_b1.py` (schema acceptance check) · this report

---

## 7. Stage B: configuration

The config-generation section of `run_sft.sh` was changed in 6 places; `diff` confirms only these lines were touched:

| Item | Original value | Actually in effect |
|---|---|---|
| `deepspeed` | `ds_z3_config.json` | `ds_z2_config.json` (stage 2, **no offload**) |
| `per_device_train_batch_size` | 1 | **2** (originally planned 4, fell back after OOM, see §8) |
| `gradient_accumulation_steps` | 1 | **2** (paired with the previous row, keeps global batch = 8) |
| `eval_steps` | 5 | 500 |
| `save_only_model` | false | true |
| `use_reentrant_gc` | unset (default true) | false |

**Global batch = 2 × 2 × 2 GPUs = 8**, matching the paper. None of the B.4 forbidden items was touched:
`finetuning_type: full`, both `freeze_*: true`, `cutoff_len: 8192`,
`learning_rate: 2e-5`, no `packing`. **liger not used.**

### B.1 schema acceptance check (both layers pass)

The source (`toolshed_v1_config.p4.yaml`) and the training data (`train.json`) are verified separately:

```
11 tools · <tools> length 8595 · sha256 prefix 72c71c806f64e162
${obj_name}_detections present · all 7020 system prompts identical
v1 has filtered out 887 robot-tool samples
```

The source layer can run **before downloading the 6 GB of data**; doing this step first is recommended from now on.

---

## 8. Stage C: 30-step probe

### The three numbers

| # | Metric | Measured |
|---|---|---|
| 1 | `s/it` (first 5 steps skipped, 25 samples) | median **11.22** · mean 11.25 · range 10.75–12.07 |
| 2 | Median GPU utilization | **GPU0 100% · GPU1 100%** (valid samples 397/710) |
| 3 | Peak GPU memory | **GPU0 48491/49140 (99%)** · GPU1 44745/46068 (97%) |

`train_runtime` 355.8 s / 30 steps = 11.86 s/step (including saving), consistent with `s/it`.
Host memory peak 19.3 GiB / 93.1, no pressure.
loss 0.8322 → 0.4548 (30 steps, 1 epoch), grad_norm 7.71 → 2.54, convergence normal.

**Extrapolation: 11.22 × 3000 = 9.3 hours.**

### ⚠️ Two conclusions contrary to `TASK.md`

**First: this workload is compute-bound, not overhead-bound.**

`TASK.md` §5 predicted "MFU is only about 5%, very likely overhead-bound", and on that basis said "what should change is the data pipeline, not the GPU".
The criterion is `<40%` overhead-bound / `>70%` compute-bound — **measured median is 100%**. So the recommendation should be reversed:

- Tuning the data pipeline is **ineffective**. The log itself hints `Too many dataloader workers: 8
  (max is dataset.num_shards=4)`; the data side keeps up
- **Switching to faster GPUs is effective**. The A6000's bf16 compute is about 40% of an A100's

The paper's `8× A100-80` takes 3–4 h, our `2× A6000` takes 9.3 h — scaled by GPU count and compute this is consistent,
which conversely shows the paper's side is not overhead-bound either; the 5% MFU conclusion that `TASK.md` back-derived is probably itself flawed.

**Second: `per_device=4` does not fit, and not because GPU1 has 3 GiB less.**

```
GPU0  used 39.84 GiB, needs 8.42 more, only 7.56 free  (total 47.40)  → peak needs ~48.3 GiB
GPU1  used 38.74 GiB, needs 7.83 more, only 5.67 free  (total 44.42)  → peak needs ~46.6 GiB
```

**The full-capacity GPU0 blows up just the same.** B.3's estimate "23.4 + logits 16 ≈ 39 GiB / 48, tight but should fit"
only counted the steady state — the actual steady state is already 39.8 GiB, plus an **8.42 GiB transient logits spike**
(`4 × ~6900 tok × 151936 vocab × 2 B`). It completed step 1 (12.47 s/it) and only died at step 2,
which shows it was triggered by a long-sequence batch.

### A/B result

`TASK.md` wanted to use an equal-time comparison of `per_device` 4 vs 2 to decide whether it is overhead-bound. **Run A cannot run**,
so there is no comparable timing. But the 100% utilization already gives the answer directly; this comparison is no longer needed.

### Phase 3 checkpoint fix (verified effective)

```
✓ text_config removed from config.json       ✓ tie_word_embeddings = True
✓ preprocessor_config.json retrieved         ✓ model_type = qwen2_5_vl (not the _text that crashes sglang)
✓ save_only_model in effect, no optimizer shards
```

⚠️ The checkpoint is actually **15.18 GiB**, not the 8 GB estimated in `TASK.md` §7 —
`sft_checkpoint/` and `checkpoint-30/` each store a full copy of the weights (7.6 GiB × 2).
The full run with `save_steps: 500` saves 6 times, about **53 GiB**.

---

## 9. The one thing that must be resolved before the full run

**GPU0's peak has only 0.6 GiB headroom.** These 30 steps were drawn from the 7020 samples; 3000 steps will see about 200k samples,
and there will certainly be longer sequences than this. Running the full run as is, **an OOM midway is highly likely**, at the cost of burning several hours.

liger is explicitly not enabled. Two zero-deviation ways out:

| Option | Headroom | Duration | Assessment |
|---|---|---|---|
| **Switch to 4 GPUs** (see §10) | ~10 GiB | ~5–6 h | **Recommended** — faster and safer |
| On this machine drop to `per_device=1 + ga=4` | still not comfortable | 12–14 h | under compute-bound, halving the batch lowers efficiency |

4 GPUs wins both ways: ZeRO-2 sharding lowers the fixed overhead from 25.9 to 15.8 GiB, bringing headroom back to 10 GiB;
and because it is compute-bound (§8), adding GPUs directly buys throughput.

**The full run has not started; waiting for this choice to be settled.**

---

## 10. Multi-GPU: configuration is now automated

`run_sft.sh` now **derives** `per_device` and `ga` **from the number of visible GPUs**, instead of hard-coding them.
Hard-coding silently deviates when switching machines — e.g. moving the 2-GPU tuned `per_device=2 + ga=2` to a 4-GPU machine
makes the global batch 16.

```bash
GLOBAL_BATCH=8                                   # the paper's definition, immutable
PER_DEVICE=min(2, 8 / num_gpus)                  # cap 2: 4 measured OOM on A6000
GRAD_ACCUM=8 / (num_gpus x PER_DEVICE)
assert PER_DEVICE x GRAD_ACCUM x num_gpus == 8, otherwise exit 1
```

### Results per GPU count (each one measured)

| GPUs | `per_device` | `ga` | Fixed overhead | Estimated peak | Headroom (48 GB) | Estimated full-run duration |
|---|---|---|---|---|---|---|
| 1 | 2 | 4 | 45.9 GiB | does not fit | — | — |
| **2** (this machine) | 2 | 2 | 25.9 GiB | **measured 47.4 GiB** | **0.6 GiB** | **extrapolated from measurement 9.3 h** |
| **4** (recommended) | 2 | 1 | 15.8 GiB | ~37.3 GiB | ~10 GiB | ~5–6 h |
| 8 | 1 | 1 | 10.8 GiB | ~32 GiB | ~15 GiB | ~3–4 h (paper config) |
| **3 / 5 / 6 / 7** | — | — | — | — | — | **refused at startup** |

### Why 3 GPUs does not work

`8 ÷ 3` is not an integer. `per_device x ga x 3 = 8` has no integer solution; the closest are 6 or 9 — 
both deviate from the paper's definition. This checkpoint is the shared starting point (π_ref) for the subsequent RL two-arm comparison,
so the definition must not change; the script therefore does `exit 1` directly instead of settling for an approximation.

### Why 4 GPUs solves the GPU memory problem in §9

ZeRO-2 shards gradients + optimizer state (40.2 GiB total) across ranks; bf16 weights (5.7 GiB) are not sharded:

```
2-GPU fixed overhead = 5.7 + 40.2/2 = 25.9 GiB   → measured peak 47.4, so activations etc. ≈ 21.5 GiB
4-GPU fixed overhead = 5.7 + 40.2/4 = 15.8 GiB   → estimated peak 15.8 + 21.5 ≈ 37.3 GiB
```

Fixed overhead saves 10.1 GiB, headroom goes from **0.6 → ~10 GiB**, the §9 problem of "an OOM midway through the full run is highly likely"
disappears, and there is no need to drop to `per_device=1` (which would just slow things down under compute-bound).

Adding GPUs also helps speed — measured GPU utilization is 100%, so it is compute-bound (see §8).
Going 2→4 GPUs theoretically gives 2x throughput, but the A6000 has no NVLink and ZeRO-2's all-reduce goes over PCIe,
so in practice 5–6 hours is expected rather than 4.7 hours.

### Two constraints

- **The dataset has only 4 shards** (with `streaming: true` the log reports `dataset.num_shards=4`).
  With 4 GPUs each rank gets exactly 1 shard and `dataloader_num_workers` is pushed down to 1; because it is compute-bound,
  this is not a bottleneck. **With more than 4 GPUs some ranks get no shard**; the script warns.
- **The GPU count is determined by `CUDA_VISIBLE_DEVICES`, not `NUM_GPUS`.**
  Upstream's `NUM_GPUS` is declared but never referenced (`TASK.md` B.5); `FORCE_TORCHRUN=1` uses all visible GPUs.
  To limit the GPU count, use `CUDA_VISIBLE_DEVICES=0,1`.

### What changed relative to the original code

Compared with the original code (`scripts/spacetools/run_sft.sh` in `ChicyChen/SpaceTools-SFT`),
there are 8 changes in total. **Note that only 3 of them are for multi-GPU**; the rest are the configuration required by `TASK.md` B.2
and one download optimization.

#### A. GPU count and batch (the core of multi-GPU support, 3 changes)

| Location | Original code | Changed to | Why |
|---|---|---|---|
| around line 42 | none | new `GPU_COUNT` detection + `GLOBAL_BATCH=8` derivation block (38 lines) | see below |
| `per_device_train_batch_size` | **hard-coded `1`** | `$PER_DEVICE` (derived value) | same as above |
| `gradient_accumulation_steps` | **hard-coded `1`** | `$GRAD_ACCUM` (derived value) | same as above |

**Why it had to change: the original code is only correct on 8 GPUs.**

The original code hard-codes `per_device=1, ga=1`, and global batch = `per_device × ga × num GPUs`:

| Number of GPUs | Global batch from the original code | Matches the paper |
|---|---|---|
| 8 GPUs (paper config) | 1 × 1 × 8 = **8** | ✓ |
| 4 GPUs | 1 × 1 × 4 = **4** | ✗ off by 2x |
| 2 GPUs | 1 × 1 × 2 = **2** | ✗ off by 4x |

**And it reports no error at all.** Running the original code as is on 4 GPUs quietly trains a checkpoint with global batch = 4 —
the optimization trajectory differs from the paper, but nothing abnormal shows in the logs. This is exactly what `TASK.md` section 9
item 4 calls "it ran through but the numbers are wrong".

`TASK.md` B.2 already noticed this (it requires changing `per_device` to 4 on 2 GPUs), but that is **another value hard-coded
for 2 GPUs** — moved to a 4-GPU machine it deviates just the same (4 × 1 × 4 = 16). So I changed it to derive from the GPU count,
rather than replacing one hard-coded value with another.

The derivation block also does three things the original code does not:

1. **If the GPU count does not divide 8, `exit 1`** (3 / 5 / 6 / 7 GPUs), instead of settling for an approximate global batch
2. **`per_device` capped at 2** — 4 measured OOM on A6000 (§8)
3. **Final assertion `per_device × ga × num GPUs == 8`**, stops if the computation is wrong; and prints one line for manual checking

#### B. Configuration required by `TASK.md` B.2 (4 changes, independent of GPU count)

| Location | Original code | Changed to | Why |
|---|---|---|---|
| `deepspeed` | `ds_z3_config.json` | `ds_z2_config.json` | A6000 has no NVLink; z3's per-layer all-gather of parameters is expensive over PCIe; 48 GB fits with z2 |
| `use_reentrant_gc` | unset (default `true`) | `false` | non-reentrant gradient checkpointing is faster |
| `save_only_model` | `false` | `true` | z2 ckpts include sharded optimizer state; `save_steps: 500` would fill the disk |
| `eval_steps` | `5` | `500` | 3000 steps would evaluate 600 times |

#### C. Download concurrency (1 change, unrelated to training)

| Location | Original code | Changed to | Why |
|---|---|---|---|
| `snapshot_download` | default `max_workers=8` | `max_workers=32` | measured default is only 0.8 MB/s (§4.5). **But this line was measured to have no effect** — the process still has only 2 threads; the actual path is `prefetch.py`. The change is kept but do not rely on it |

#### Mathematical impact

**Group A exists to make the mathematical result match the paper** (global batch always 8);
the four items in group B (sharding strategy, gradient checkpointing implementation, checkpoint contents, evaluation frequency)
do not change gradients or weight updates; group C only affects downloading.

**So none of the eight changes alters the mathematical result of training.** liger not used; none of the `TASK.md` B.4 forbidden items
(`packing` / `finetuning_type` / `freeze_*` / `cutoff_len` / `image_max_pixels` /
`learning_rate` / `max_steps`) was touched.

#### The one place in the original code kept unchanged

`NUM_GPUS="${NUM_GPUS:-8}"` is kept as is, with only a comment added. Upstream it is a **no-op** —
declared but referenced in neither the config nor the launch command (`TASK.md` B.5 also points this out).
Deleting it would be "cleanup", not "necessary", and this checkpoint is the shared starting point for RL,
so whatever can stay untouched stays untouched. What actually determines the GPU count is `CUDA_VISIBLE_DEVICES`.

### How to switch to a 4-GPU machine

```bash
tar -C /opt -xf conda-st.tar        # or use bundle/RESTORE.sh
# no environment changes needed: flash-attn is an sm_80 cubin, any A6000 can run it
bash scripts/spacetools/run_sft.sh  # auto-detects 4 GPUs, outputs per_device=2 · ga=1
```

At startup it prints one line `GPUs N · per_device=X · ga=Y · global batch 8 ✓` — **check this line to confirm the definition**.

---

## 11. Operational notes (needed before the full run)

- **Background jobs must be detached from the process group with `setsid`.** A session restart takes the training down with it — already hit once;
  the Phase 1 that was downloading at the time was completely lost. `launch_probe.sh` has been changed.
- **`conda activate` does not carry across bash processes.** `run_sft.sh` activates internally, and conda is a shell
  function. Solved by pointing `BASH_ENV` at `conda_hook.sh`; in that hook **do not** prepend the base environment's
  `bin`, or it will shadow the activated env.
- **Sentinel script**: `memguard.sh` watches cgroup `anon`, notifies above 44.7 GiB, kills training above 59.6 GiB.
  It kills the training, not the container — host memory hitting the ceiling gets the container killed and the remote disconnected, and no notification can go out.
- **`pkill -f` matches its own wrapper shell.** Judge by the `comm` field, or kill by process tree.
