# SpaceTools SFT environment package (2× A6000 / sm_80·sm_86)

**Validated by a complete 30-step SFT training run** (2026-09-07): all three Phases of `run_sft.sh` ran through,
the checkpoint fix took effect, exit code 0.

## Start here on a new machine

After restoring the environment, **read `/workspace/runpod-handoff/NEXT_MACHINE.md` first** —
it is the task brief prepared for the new machine (full run on 4 GPUs), with the validated numbers, the pitfalls you will hit,
and several judgments in `TASK.md` that measurements have overturned.

Background details are in `REPORT_A.md` (the full record of stages A–C).

## Usage

```bash
sudo bash RESTORE.sh          # pre-checks → checksums → unpack → acceptance check, all in one go
```

`RESTORE.sh` stops if any pre-check fails; it never unpacks halfway. After unpacking it runs `VERIFY.sh` automatically.
To re-verify on its own: `bash VERIFY.sh`.

## Target machine requirements

| | |
|---|---|
| Architecture | x86_64 Linux |
| GPU | compute_cap **8.0 or 8.6** (A100 / A6000 / A4000 / A5000 / 3090) |
| Driver | ≥ 525 (this package was built on 580.159.03) |
| glibc | ≥ 2.32 |
| CUDA toolkit | **not needed** — the runtime libraries are in the package under `site-packages/nvidia/` |
| Disk | disk holding `/opt` ≥ 12 GB |

**H100 (9.0) and Blackwell cannot run it**: the flash-attn in the package only has `sm_80` cubins,
with no PTX fallback (`FLASH_ATTN_CUDA_ARCHS=80`, gencode only gives `code=sm_80`).
Those GPUs require rebuilding flash-attn. `RESTORE.sh` blocks them before unpacking.

Running SFT also has a GPU memory requirement: measured peak with 2 GPUs `per_device=2` is **48.5 GiB**, so 48 GB-class GPUs are needed.
An A4000 (16 GB) can load this environment, but cannot run this training.

## GPU count

`run_sft.sh` **derives** `per_device` / `ga` **automatically from the number of visible GPUs**, guaranteeing
`per_device x ga x num GPUs = 8` (the paper's definition). At startup it prints one line to confirm:

```
GPUs 4 · per_device=2 · ga=1 · global batch 8 ✓
```

| GPUs | `per_device` | `ga` | Headroom (48 GB GPU) | Estimated full run |
|---|---|---|---|---|
| 2 | 2 | 2 | 0.6 GiB (measured, tight) | 9.3 h |
| **4** | 2 | 1 | **~10 GiB** | **~5–6 h** |
| 8 | 1 | 1 | ~15 GiB | ~3–4 h |

**3 / 5 / 6 / 7 GPUs are refused at startup** — 8 is not divisible by them, so the paper's global batch cannot be formed,
and this checkpoint is the shared starting point for the subsequent RL two-arm comparison, so the definition must not change.

The GPU count is determined by `CUDA_VISIBLE_DEVICES` (upstream's `NUM_GPUS` is a no-op).
The dataset has only 4 shards; with more than 4 GPUs some ranks get no data.

## ⚠️ Paths cannot be changed

`llamafactory` is an editable install, and `site-packages/_editable_impl_llamafactory.pth`
hard-codes `/workspace/SpaceTools-SFT/src`. The shebangs in the conda environment are also all absolute paths
under `/opt/conda-st/...`. So both locations must be restored exactly as they were:

```
/opt/conda-st/                    conda + spacetools-sft environment (9.3 GB)
/workspace/SpaceTools-SFT/        LLaMA-Factory fork, contains run_sft.sh (145 MB)
/workspace/SpaceTools/            upstream repo (6 MB)
/workspace/wheels/                self-built flash-attn wheel (57 MB)
/workspace/runpod-handoff/        TASK.md · setup.sh · reports · various acceptance scripts
```

`/workspace` is just an ordinary directory; if the target machine does not have it, `RESTORE.sh` creates it.

## Not in the package

The SFT data (`siyich/spacetools-sft`, 7463 files / 5.89 GiB) and the base model
(`Qwen/Qwen2.5-VL-3B-Instruct`, ~8 GB) are not in the package — `run_sft.sh` downloads them automatically.

⚠️ Upstream's `snapshot_download` only gets ~0.8 MB/s on this dataset (concurrency does not take effect; there are only 2 threads in the process),
so it takes 1–2 hours. The package's `runpod-handoff/prefetch.py` is a parallel replacement that finishes in 21 minutes and checks the size of every file.

## Environment contents

```
python 3.11 · torch 2.9.1+cu128 · cuda 12.8
torchvision 0.24.1+cu128 · torchaudio 2.9.1+cu128     ← must be the same release as torch
transformers 4.57.1 · flash-attn 2.8.3.post1 (self-built, sm_80 only)
deepspeed 0.19.6 · llamafactory 0.9.5.dev0 (editable)
accelerate 1.11.0 · trl 0.24.0 · peft 0.18.1 · datasets 4.0.0
```

## Deviations from upstream setup_envs.sh

| Deviation | Impact |
|---|---|
| `FLASH_ATTN_CUDA_ARCHS=80` (upstream default `80;90;100;120`) | sm_80 cubins only, no PTX; compile volume and memory reduced to 1/4 |
| Added `ninja setuptools wheel` | build-time tools; without `ninja` flash-attn falls back to serial compilation |
| **Pinned `torchvision` / `torchaudio`** | **fix**, see below |
| Completed dependency upper bounds + second pin of `transformers` | **fix**, see below |

Both "fixes" plug upstream gaps; they are not deviations we introduced — upstream `setup_envs.sh` puts no constraints on transitive dependencies:

- **`transformers`** gets pushed up to 5.x. **It takes effect silently** and produces a π_ref different from the paper's definition
- **`torchaudio`** resolves to 2.11.0 (built for torch 2.11). `llamafactory/data/mm_plugin.py`
  does `import torchaudio` → `undefined symbol: torch_library_impl`, and training fails to start

`VERIFY.sh` now checks both kinds: pinned versions + the real import chain used by training.
The latter is necessary — a bare `import llamafactory` does not catch the torchaudio problem, because it is lazy.

None of the four changes the mathematical result of training.
