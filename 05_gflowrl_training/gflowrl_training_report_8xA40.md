# P7 training report: replacing GRPO with GFlowRL C′ in SpaceTools Step 4

> Training time: 2026-09-16 08:35:04 → 2026-09-17 06:32:35 UTC (21 h 57 min)
> Machine: RunPod 8× A40 48 GB (released after this report was generated)
> Result: `FULL_DONE_PASS`, all 85 steps completed, exit code 0
> Artifacts: `qyYue1389/spacetools-p7-gflowrl-cprime-8xa40` (private HF repo)
> This file, together with the raw files in the same directory, forms the complete record of this training run

---

## 0. In one sentence

Replaced SpaceTools' Step 4 RL from GRPO with the **C′ variant** of GFlowRL (arXiv:2607.13394),
ran a full 1 epoch (85 steps) on the self-trained SFT checkpoint, with no crashes, no OOM (training side) and
no interruptions throughout; three checkpoints (85 / 60 / 30) have been merged into HF format and uploaded.
**Effect unknown** — eval has not been run yet.

---

## 1. What was trained and how it was configured

### 1.1 Baseline and target

| | |
|---|---|
| base model | `qyYue1389/spacetools-sft-v1-4xa6000` @ `91fd4bdf6d2ffe3b355cc650e71925c6e91a15c5` |
| Architecture | Qwen2.5-VL-3B-Instruct (4.066 B parameters, vision encoder frozen, 2.55 B trainable) |
| Algorithm | GFlowRL Eq. 4/6/7/8, variant `cprime` |
| Control | the paper's GRPO (**not** run this time; only the single GFlowRL arm was trained) |
| Data | `siyich/spacetools-rlfulltools`, train.parquet **5500 rows** |

### 1.2 Why C′

GFlowRL's Eq. 4 (in-batch log Z) and Eq. 5/6 (flow gap) can each apply or not apply sequence-length
normalization, giving three combinations:

| Variant | Eq.4 | Eq.6 | Conclusion |
|---|---|---|---|
| `paper` | not normalized | normalized | The drift term degenerates into a common within-group shift, wiping out the reward on 30–49% of groups |
| `normalized` | normalized | normalized | Has a fixed point, but that fixed point is a point mass at \|y\|≈334 (pure reward maximization) |
| **`cprime`** | **not normalized** | **not normalized** | **The only one that both has the Prop. B.1 analytic fixed point and a 0 rate of "the whole group clipped to the same value"** |

The fixed point has been verified numerically with `tools/p7/p7_fixedpoint_check.py` to **7.39e-13**.
In the code, `masked_sum` (rather than `masked_mean`) is the **only** difference between C′ and `normalized`, and changing it raises no error,
so this line is load-bearing; if it is touched, the fixedpoint check must be rerun.

### 1.3 Hyperparameters (as-run)

```
Algorithm side
  loss_mode              gflowrl
  variant                cprime
  beta                   8.0
  eps_low / eps_high     0.2 / 0.28        (Eq.7 asymmetric clip)
  eps_is                 0.2               (Eq.8 IS weight lower bound)
  kl_loss_coef           0                 ← overridden by the gflowrl wrapper
  use_kl_loss            True              ← must stay True, see §3.4

Training side (all inherited from upstream run_rl.sh, not a single character changed)
  train_batch_size       64
  rollout.n              5                 (G=5, the paper uses 16; changing it makes the two arms incomparable)
  ppo_mini_batch_size    64                (= train_batch_size → on-policy)
  ppo_micro_batch_size_per_gpu  2
  lr                     1e-6
  grad_clip              1.0
  total_epochs           1
  max_prompt/response_length    8192 / 8192
  max_assistant_turns    8
  max_parallel_calls     8
  max_tool_response_length      2048
```

**85 steps, not 86.** 5500 ÷ 64 = 85.9; verl drops the trailing partial batch, so one epoch
is exactly 85 steps. The 86 in the handoff doc was an estimate; `latest_checkpointed_iteration.txt` is authoritative.

---

## 2. Machine and environment

### 2.1 Hardware

```
GPU        8× NVIDIA A40 (48 GB, GA102, sm_86)
Driver     580.159.04 · CUDA 13.0
CPU/RAM    96 cores / 503 GB
Container disk  120 GB (the five conda environments under /opt live here)
Network volume  250 GB (/workspace: weights, data, checkpoints)
OS         Ubuntu 24.04.3 · glibc 2.39
Topology   GPU0-3 on NUMA0 (PXB interconnect), GPU4-7 on NUMA1, SYS between the two groups
```

The topology line later became important: whether the 4 training GPUs span NUMA nodes affects NCCL, and on this machine
**PCIe P2P is actually broken** (see §5.1).

### 2.2 Software stack

```
torch 2.9.1+cu128 · cuda 12.8 · cudnn 9.16.0
transformers 4.57.1 · numpy 1.26.4 · ray 2.47.1
verl 0.8.0.dev · sglang 0.5.6
```

Five conda environments with mutually conflicting dependencies (torch 2.3.1 / 2.5.1 / 2.9.1 side by side, numpy 1.26 and 2.x side by side),
so the tools must run in separate environments:

```
spacetools-rl              main training environment (verl + sglang)
spacetools-tool-roborefer  RoboRefer-8B
spacetools-tool-vlm        Molmo-7B-D · SAM2 · DepthPro
spacetools-tool-bbox       3D bbox · vision_ops
spacetools-tool-graspgen   GraspGen
```

The environments were not installed fresh; they were restored from `qyYue1389/spacetools-eval-env` (22 GB package) into `/opt`
— the conda environments have absolute paths baked in and must be restored to `/opt/conda-st`; with a different path everything breaks.

---

## 3. How the GPUs are split and how the tools are deployed

### 3.1 Splitting the 8 GPUs

```
GPU 0-3   Toolshed tool actors      TOOL_GPUS=4
GPU 4-7   training (FSDP + sglang)  TRAIN_GPUS=4
```

This split is not what upstream does. Upstream `run_rl.sh` hands the **same** `GPUS_PER_NODE` to both
Toolshed's STRICT_PACK placement group and `trainer.n_gpus_per_node`. The paper uses 2 nodes,
with the PG on node1 and the trainer on node2, so this was never exposed; **on a single node this reserves the same GPUs twice**,
and the result is that the PG eats all the GPUs and the trainer gets none, or the PG is never ready — a deadlock with no error.
Splitting it into `TOOL_GPUS` / `TRAIN_GPUS` is our change (commit `574ae81`, details in §4).

### 3.2 Tool actor deployment

Toolshed is a tool-hosting layer on top of Ray: each tool is a set of Ray actors running in its own conda environment,
and variables (ndarrays, point clouds) are passed across environments via Ray serialization.

Upstream `TOOL_CONFIGS` is written for the paper's 2-node scale, with a logical demand of **7.8 GPUs**:

```
roborefer 6×0.6 + vlm 2×0.6 + sam2 5×0.2 + depth 5×0.2
        + bbox 5×0.1 + vision_ops 8×0 + grasp 5×0.1  = 7.8
```

We only have 4 GPUs for tools, so we added automatic scaling (commit `3e1d02c`); what actually landed is:

```
scaled tool actors by 0.500 to fit tool_gpus=4.0:
  {'roborefer': 3, 'vlm': 1, 'sam2': 2, 'depth_estimator': 2,
   'bounding_box': 2, 'vision_ops': 4, 'grasp_generator': 2}
  -> demand 3.70 logical GPUs
```

**`num_gpus` is a Ray logical reservation, not a GPU memory quota.** Actors on the same physical GPU share all of its GPU memory;
Ray only guarantees "the fractions placed on the same GPU sum to ≤ 1.0". So 3.70 only decides "whether it can be scheduled or not".

The tool side registers **48 methods** in total (`config/rl_toolshed_config.yaml`), by tool:
`roborefer.*`, `vlm.*` (Molmo), `sam2.*`, `depth_estimator.*`, `bounding_box.*`,
`vision_ops.*`, `grasp_generator.*`. The model can call up to 8 concurrently per turn, for up to 8 turns.

### 3.3 The 4 GPUs on the training side

```
FSDP full sharding (param_offload=False, optimizer_offload=False)
ref model parameters offloaded to CPU (ref.fsdp_config.param_offload=True)
sglang rollout: tensor_model_parallel_size=1, gpu_memory_utilization=0.7, max_num_seqs=256
```

Measured peak GPU memory **26.5 → 35.3 GB** (48 GB limit), CPU-side peak 151 GB.

### 3.4 A constraint that must be remembered

`kl_loss_coef=0` but `use_kl_loss=True` looks contradictory, but both are actually necessary:

- `use_kl_loss=False` makes `dp_actor` not compute `ref_log_prob` at all, and GFlowRL's
  `d = Σ(log π_ref − log π_old)` needs exactly that → it will not run
- `kl_loss_coef≠0` **double-counts** the reference term, because d itself is the KL-type quantity in Eq.4/6
  → no error, but it trains a different objective

Both of these errors are silent, so we added a guard in `ray_trainer.py` that raises an exception (commit `0b5098a`).

---

## 4. What was changed relative to the original repo, and why

The baseline is upstream `SpaceTools-RL`. Our branch is `repro-4xa6000`, HEAD = `c6fef78`,
working tree clean (only one untracked `images` symlink). **Nine commits, 6 files in total, +366/−8 lines.**
Full diff in `code/p7_all_changes.diff`, full commit messages in `code/p7_commits.txt`.

### 4.1 Implementing GFlowRL itself (2 commits)

| commit | What changed | Why |
|---|---|---|
| `33036a5` | `ray_trainer.compute_gflowrl_flow_gap()` + `core_algos.compute_policy_loss_gflowrl()` | The paper's Eq.4/6/7 are gradient-free (they depend only on π_old, π_ref, r, all frozen during the actor update), so they are computed once per step on the driver; the gradient-carrying part of Eq.8 is registered as `loss_mode='gflowrl'`. Three variant switches let the three normalization combinations be compared without code changes |
| `0b5098a` | thin `run_rl_gflowrl.sh` wrapper + the guard from §3.4 | A two-arm comparison is only meaningful when "everything except the objective is identical". Copying the 400-line `run_rl.sh` would inevitably drift, so the wrapper is just a 7-line overrides array, relying on the existing `"$@"` pass-through in `run_rl.sh`; hydra takes the last value for a duplicate key, so the `kl_loss_coef=0` here overrides the `=0.01` in `run_rl.sh` |

**`advantages` is reused.** GFlowRL has no notion of advantage, but the signature of the registered loss has no other
per-sequence slot, so g̃ is broadcast into `batch["advantages"]`. Consequence: the
`critic/advantages/*` metrics printed by the trainer report statistics of g̃, not GRPO advantages. Do not misread them in the logs.

### 4.2 Fixing upstream bugs (5 commits)

None of these are GFlowRL-related; they are **things anyone running the upstream code on a single node / small machine will inevitably hit**:

| commit | Symptom | Root cause |
|---|---|---|
| `574ae81` | Ray deadlock on a single node, no error | `GPUS_PER_NODE` is given to both the PG and the trainer, reserving the same GPUs twice (§3.1) |
| `dfbde8e` | `TypeError: PolicyLossConfig.__init__() got an unexpected keyword argument 'gflowrl'`, and it is raised in a Ray worker a few minutes after Toolshed comes up | `PolicyLossConfig` is a structured dataclass; via `omega_conf_to_dataclass → hydra.utils.instantiate` every key under `policy_loss` is passed in as a kwarg, and undeclared keys raise an error rather than being ignored. Also fixed a quieter other half: the driver reads beta/eps from the raw DictConfig, while `core_algos` reads `eps_is` from the dataclass — the dataclass has no such field, so `GF_EPS_IS` was silently dropped, and the two config sources disagreed about the same loss |
| `3e1d02c` | `assert _demand <= tool_gpus` refuses to start outright | Upstream `num_actors` is written for 2 nodes (7.8 logical GPUs); on a small machine it will not start without scaling (§3.2) |
| `5fa00d8` | `FileNotFoundError: 'images/xxx.png'` after 20 minutes of loading | The parquet has relative paths, resolved against cwd; `ray start` starts the raylet in the caller's cwd, and `cd "$VERL_DIR"` comes after it. Moved the `cd` before `ray start` |
| `4474e67` | A successful run reports failure; **and `ray stop --force` was never executed, leaving a live Ray cluster after every run** | `cleanup()` kills its own background jobs and then `wait`s; `wait` returns 128+SIGTERM for a job just killed by a signal, `set -e` is also in effect inside the EXIT trap, so cleanup dies on the `wait` line — the `ray stop` after it never runs. Verified with a minimal reproduction (`scripts/exitprobe.sh`): adding `set +e` as the first line of cleanup fixes it |

### 4.3 Adding observability (2 commits)

| commit | What was added | Why |
|---|---|---|
| `3e93509` | `compute_group_degeneracy_metrics()` (group degeneracy rate / rollout degeneracy rate / within-group reward range) + two IS weight guards + corrected definition of `actor/ppo_kl` | The reward degeneracy rate is **the whole reason** C′ might beat GRPO, and the previously cited 56.9% / 75.8% came from sampling on eval benchmarks, not the Step 4 training set. The call site is after `compute_advantage` and before the gflowrl branch, so both arms log it — numbers from only one arm cannot prove a difference between the two arms. Degeneracy is decided by "within-group range is strictly 0", not variance, because the pointing / IoU rewards are continuous |
| `c6fef78` | Diagnostic instrumentation for the β decomposition (see §5.4) | See below |

---

## 5. The optimization techniques used in training, and what each one did

| Technique | Where configured | What it did |
|---|---|---|
| **Frozen vision encoder** | `model.freeze_vision_model=true` | Only 2.55 B of the 4.066 B parameters are trained. Saves GPU memory and compute, and is also what the paper does |
| **FSDP full sharding** | verl default + `param_offload=False` | Parameters/gradients/optimizer state sharded across 4 GPUs. No offload because 48 GB is enough, and offload would make PCIe the bottleneck (and P2P on this machine is broken anyway) |
| **ref model parameters offloaded to CPU** | `ref.fsdp_config.param_offload=True` | ref only computes log_prob once per step, so keeping it resident in GPU memory is not worth it. Cost: `timing_s/ref ≈ 122 s` per step |
| **Gradient checkpointing** | `enable_gradient_checkpointing=True` | Trades recomputation for activation memory. With 8192+8192 context this is a precondition for running at all |
| **remove padding (sequence packing)** | `use_remove_padding=True` | Variable-length sequences are concatenated into a contiguous token stream, avoiding wasted compute on pad positions |
| **flash-attention-2** | `override_config.attn_implementation` | Attention memory drops from O(L²) to O(L) |
| **micro-batch 2 + mini-batch 64** | `ppo_micro_batch_size_per_gpu=2`, `ppo_mini_batch_size=64` | mini_batch == train_batch makes this step **strictly on-policy**, so the Eq.8 IS weight is identically 1 (`is_weight_min=1.0`, `collapsed_frac=0.0`). This is not just convenience: once mini_batch is reduced to save GPU memory, the IS weight really starts working, and a drift of −3e-3 per token can push w for a 1e3-token sequence down to 0.02 — it is still in the batch, but its contribution is close to 0, and the effective batch quietly shrinks. Those two guard metrics exist to make this visible |
| **sglang async multi-turn rollout** | `rollout.name=sglang`, `max_parallel_calls=8` | Tool calls do not block the whole batch. The per-step `timing_s/gen ≈ 322 s` includes all tool round trips |
| **Toolshed multi-actor + environment isolation** | `TOOL_CONFIGS` | Multiple replicas of the same tool work through the queue in parallel; conda isolation resolves 5 mutually exclusive dependency sets. The paper measured 3.2× faster than naive HTTP at 8-way concurrency |
| **NCCL over shared memory** | `NCCL_P2P_DISABLE=1` | **Not an optimization, a workaround for a hardware fault** (§5.1). Cost: all-reduce goes through host memory, slower |
| **checkpoint janitor** | `scripts/janitor2.py` (our own) | `save_freq=5` × 44 GB per ckpt would fill the 250 GB volume by step 20. The janitor keeps only the latest complete ckpt (for auto-resume), and at steps 30/60 additionally saves a snapshot **containing only the weight shards** (15 GB, without the 7 GB/rank `optim_*.pt`). It archives before deleting, and only acts once the latest one has been written to ≥25 GB |

Not used (recorded to avoid rehashing the discussion later): no LoRA (full-parameter fine-tuning), no optimizer offload,
`expandable_segments` not enabled (it conflicts with sglang's TorchMemorySaver), `rollout.n` not tuned
(changing it makes the two arms incomparable).

### Measured performance (last step)

```
timing_s/step          981 s   (mean over the run 925 s, median ~930 s)
  ├ gen               322 s   rollout + tool round trips
  ├ update_actor      369 s
  ├ ref               122 s
  ├ save_checkpoint    39 s
  └ rest (adv 0.07s etc.)
perf/total_num_tokens        1,360,105 / step
perf/throughput              346 token/s
perf/mfu/actor               0.145
peak GPU memory               35.3 GB / 48 GB
```

85 steps × 925 s ≈ **21.8 hours**, consistent with the wall clock of 21 h 57 min.

---

## 6. How to read `p7_metrics.csv`

File: `metrics/p7_metrics.csv`, 85 rows, one per step. Parsed from the training log by `scripts/p7tab.py`
(the script is run on demand, not a daemon; rerunning it any number of times does not affect training).

### 6.1 Column definitions

| Column | Key in the log | Meaning |
|---|---|---|
| `step` | — | Training step 1–85 |
| `wall_s` | `timing_s/step` | Wall-clock seconds for the step |
| `degen` | `reward/degenerate_group_frac` | **Fraction of reward-degenerate groups**: the fraction of groups whose 5 rollouts have a reward range of strictly 0. GRPO's gradient is identically 0 on these groups |
| `sat` | `gflowrl/clip_saturation` | Eq.7 clip saturation rate: fraction of samples whose g falls outside [−0.2, +0.28] and gets truncated. **Pinned at 1.0 means all magnitude information is lost and only the sign remains** |
| `sat_b2/b4/b16` | `gflowrl/sat_at_beta_*` | Saturation rate recomputed with β replaced by 2/4/16 (reusing the same r and d, no extra forward pass). **`sat_b8` must be bit-for-bit equal to `sat`** — this is the diagnostic block's built-in self-check; the β used in this training run is 8 |
| `reward` | `gflowrl/reward_term_abs_mean` | Mean of the flow gap's **reward half** \|mean_group(β·r) − β·r_i\|, proportional to β |
| `drift` | `gflowrl/drift_term_abs_mean` | Mean of the flow gap's **drift half** \|mean_group(d) − d_i\|, independent of β |
| `rew/drift` | ratio of the two | **Strength of the reward signal relative to the drift signal. < 1 = reward is drowned out by drift** |
| `sign_ok` | `gflowrl/sign_agree_nondegenerate` | On **non-degenerate groups**, the fraction where the post-clip g has the same sign as the reward term. → 0.5 means the drift term has flipped the direction of the reward |
| `d_seq` | `gflowrl/d_seq_abs_mean` | Mean of \|d\| = \|Σ(log π_ref − log π_old)\|, i.e. how far the policy is from the reference policy. **It rising monotonically over training is normal** |
| `grad_norm` | `actor/grad_norm` | Gradient norm **before** clipping |
| `score` | `critic/score/mean` | Mean reward of the step's rollouts (sum of task rewards between 0–1) |
| `mem_gb` | `perf/max_memory_allocated_gb` | Per-GPU torch allocator peak |

### 6.2 β decomposition — what you must know before reading this CSV

`z_t` is the within-group **arithmetic** mean, so Eq.6 splits exactly into two additive halves, and g is linear in β:

```
g_i = [mean_group(β·r) − β·r_i]  +  [mean_group(d) − d_i]
       └── reward term, ∝ β ──┘      └── drift term, independent of β ──┘
```

Two direct consequences, which must accompany any conclusion:

1. **On reward-degenerate groups, the reward term is identically 0 and g comes entirely from the drift term.** There C′ is doing flow matching toward
   `π_ref·exp(βr)/Z`, **not learning the reward**. In this run the measured median of degenerate groups is
   **70.3%** (range 57.8–85.9%) — that is, this is the situation for the majority of groups in every batch.
   You cannot describe it as "GRPO gives 0 gradient while C′ is still learning the reward".
2. **Lowering β does not cure clip saturation.** It only shrinks the reward half; the drift term does not move at all, so the surviving
   gradient is **even more** drift-dominated. The three columns `sat_b2/b4/b16` are the per-step computed counter-evidence:
   in this run `sat_b2` median 0.478, `sat_b4` 0.500, `sat` (β=8) 0.516, `sat_b16` 0.525 —
   β differs by 8×, and the saturation rate moved by only 5 percentage points. **β must be chosen by `rew/drift`, not by `sat`.**

### 6.3 Actual readings over these 85 steps

```
                 min       median    max
degen           0.578    0.703    0.859
sat             0.303    0.516    0.613
sat_b2          0.278    0.478    0.562
sat_b4          0.291    0.500    0.597
sat_b16         0.325    0.525    0.622
reward          0.290    0.672    1.695
drift           0.000    0.339    0.441
rew/drift       0.858    2.014    4.941
sign_ok         0.689    1.000*   1.000     (*median 0.837)
d_seq           0.000    0.374    0.478
grad_norm     103.075  180.312 5229.205
score           0.581    0.806    1.083
mem_gb         26.478   34.618   35.339
wall_s            862      930      981
```

How to interpret:

- **`sat` is stable around 0.5 and does not drift toward 1.0** — magnitude information is still there; clip has not degenerated the gradient into pure sign.
- **`rew/drift` median 2.01; of the 85 steps only step 76 drops below 1 (0.86)** — overall the reward signal
  holds the drift down. In trend, the median goes from 2.125 over the first 10 steps → 1.754 over the last 10, slowly declining but not by much.
- **`d_seq` rises from 0 to ~0.41 and then levels off** — at step 1 the policy equals the reference policy so it is 0; after that it separates normally.
- **`grad_norm` median 180, while `grad_clip=1.0`.** This means **every single update is clipped down to direction only,
  with no magnitude**. The 5229 at step 74 is the only one above 1000 in the whole run. This must go into the conclusion:
  in this configuration, the relationship between learning rate and gradient size is entirely taken over by clip.
- **`score` has no reliable upward trend.** Median 0.766 over the first 10 steps, 0.820 over the last 10, but 0.809 for the first half
  and 0.796 for the second half — back-and-forth of this magnitude is noise. With 85 steps and almost every step a pure-direction update, **training reward
  did not clearly improve in this run**. Most likely eval will not show a large gain either; be prepared for that.
- **`mem_gb` 26.5 → 35.3, steady**, with headroom below 48 GB and no OOM risk.

---

## 7. Problems worth recording (caused by hardware / upstream code)

### 7.1 [Hardware] PCIe P2P inside the container is broken — the most serious one

**Symptom:** after the trainer comes up, the first NCCL collective never completes, and after 10 minutes the watchdog reports

```
Last enqueued NCCL work: 1, last completed NCCL work: -1
```

while `nvidia-smi topo -p2p r` shows everything OK.

**Diagnosis:** instead of repeatedly starting 15-minute training runs to experiment, we wrote a minimal 4-GPU all-reduce probe
(`scripts/ncclprobe.py` + `scripts/nprobe.sh`), and ran 5 variants in 90-second rounds:

```
A  default                       -> 40 s timeout
B  NCCL_IB_DISABLE=1             -> 40 s timeout   (so it is not IB/RoCE)
C  B + NCCL_SOCKET_IFNAME=eth0   -> 40 s timeout
D  B + NCCL_P2P_DISABLE=1        -> 7 s ALLREDUCE_OK   ← this is it
E  B + NCCL_SHM_DISABLE=1        -> 40 s timeout   (so SHM is exactly the path that works)
```

**Root cause:** the typical symptom of ACS / IOMMU not being disabled inside a container — P2P is "available" in the topology, but actual transfers hang.
This is **a platform problem, not a code problem**; a different machine may not have it.

**Cost:** with P2P disabled, NCCL goes over shared memory (relayed through host memory), which is slower than P2P. This run's 925 s per step
was measured under this condition and **cannot be directly compared with other machines' throughput**.

**Lesson:** after renting a new machine, spend 90 seconds running an all-reduce probe before running training. Had we not done that this time,
it would have been 15 minutes of waiting for every single environment variable change, and the day would have been gone.

### 7.2 [Upstream code] Reserving the same GPUs twice on a single node → deadlock with no error

See `574ae81` in §4.2. The paper uses 2 nodes, so this path is never exposed; **any single-node reproduction will hit it**.
The failure mode is "stuck, not moving", with no error message at all — the hardest kind to debug.

### 7.3 [Upstream code] `cleanup()` aborts itself under `set -e`, leaving the Ray cluster behind

See `4474e67` in §4.2. The surface symptom is only "a successful run reports failure" (which looks merely cosmetic);
**the real harm is that `ray stop --force` was never executed** — every run leaves a live Ray cluster behind,
and the next launch collides with it. We only realized this when, 2 h 44 min after a successful smoke test ended, we found the cluster still alive.

Corollary: any signal of the form "wrong exit code but looks harmless" must be traced to its root cause before deciding whether to ignore it.

### 7.4 [Upstream code] The raylet's cwd decides whether the data can be found

See `5fa00d8` in §4.2. The images in the parquet are relative paths `images/xxx.png`, resolved against the process cwd;
and when `ray start` launches the raylet its cwd is the caller's cwd, with `cd "$VERL_DIR"` coming after it.
Consequence: the `FileNotFoundError` is only reported **after 20 minutes of loading**. This kind of "fails very late" bug is especially expensive.

### 7.5 [Upstream code] Structured config raises on unknown keys instead of ignoring them, and only blows up inside a Ray worker

See `dfbde8e` in §4.2. The accompanying "other half" is more insidious: the two config readers (driver reads the raw DictConfig,
`core_algos` reads the dataclass) disagree about the parameter set of the same loss, and `eps_is` is silently dropped.
**When the same config has two read paths, there must be a check asserting that the key sets on both sides match.**

### 7.6 [Upstream code] Environment variables leak from the eval shell into training

`VERL_DUMP_TOKEN_DIAGNOSTICS` was set by hand for eval; it turns on the token diagnostics in `ray_trainer._validate()`;
and `test_freq` makes validation also run during training, so training picks up the extra overhead of
"putting the rollout engine to sleep + a full pass of `compute_log_prob`". Running eval and then training in the same shell
triggers it. Explicitly unset in `574ae81`.

### 7.7 [Tool side] Many runtime errors, but all correctly swallowed

Counts over the whole training log (9.8 MB):

```
No collision-free grasps found          3389 times
  of which raised as RuntimeError       2346 times
Top-down filtering removed all grasps    130 times
selected index k out of range             24 times
CUDA error: invalid configuration argument 12 times  (depth_estimator.get_depth_map)
Mask removed all points                   10 times
torch.cuda.OutOfMemoryError                2 times  (tool side, see below)
ERROR:toolshed.tools.sam2                  3 times
hit the 8-turn limit                       2 times
truncated tool responses                   0 times
training-side OOM / NCCL WARN / crash      0
```

**These are not training failures.** Toolshed turns tool exceptions into text returned to the model, and the model can retry with a different tool or different
parameters — which is exactly what tool-augmented RL is meant to learn. The 3389 from `grasp_generator` are especially normal:
it does collision detection on real point clouds, and "there is no collision-free grasp pose around this object" is a **geometric fact**, not a bug.

**But two are worth recording separately:**

1. Two tool-side OOMs, one trying to allocate **14.36 GiB** and the other trying to allocate **1481.25 GiB** —
   both inside `torch.cdist`. 1481 GiB is obviously not a matter of insufficient GPU memory, but of **degenerate input**
   (an abnormally inflated point count in the point cloud) causing cdist's intermediate matrix to explode. This is a tool robustness problem,
   and may become more frequent in larger-scale training.
2. 12 `CUDA error: invalid configuration argument`, all in `depth_estimator.get_depth_map`,
   typically illegal kernel launch parameters caused by zero-size input.

Both were swallowed and do not affect this run's results; but if a "tool success rate" is ever computed, these must be counted in the denominator.

### 7.8 [Platform] HF's 5000 resolves / 5 minutes quota

Downloading the training data (5425 files) hit 429s, and download speed dropped to 0.5 MB/s. Not a code problem,
it is Hub rate limiting. The fix is batching + landing on the container disk and then copying as a whole, and **once downloaded, do not let scripts re-download it**
(`run_rl.sh` has an `os.path.isfile` check; symlinking the data to where it expects it makes it hit directly).

### 7.9 [Upstream default] `grad_clip=1.0` and `grad_norm` differ by two orders of magnitude

Not a bug, a fact of the config, but it affects how the results are interpreted: `grad_norm` median 180, max 5229, while the clipping threshold is 1.0.
**Every single update keeps only the direction.** This must be stated in the report, otherwise the number "learning rate 1e-6" will be misread.

---

## 8. Other things worth recording

### 8.1 Checkpoint composition and disk

verl's FSDP checkpoint is **44 GB** per step (the 34 GB estimated in the handoff doc was wrong; we only found out by measuring):

```
model_world_size_4_rank_*.pt     4 × 4.07 GB   ← merging into HF format needs only these
optim_world_size_4_rank_*.pt     4 × ~7 GB     ← only needed to resume training, not used by eval
extra_state_* / fsdp_config.json very small
huggingface/                     config + tokenizer
```

`verl/model_merger/fsdp_model_merger.py` only reads `model_world_size_*_rank_*.pt`
(in two places, L91 and L149), so **a 15 GB weight snapshot is enough for merge + eval**; optim does not need to be kept.
This is the key to the janitor surviving on the 250 GB volume.

### 8.2 Merge and upload

```
merge:  python3 -m verl.model_merger merge --backend fsdp \
            --local_dir <ckpt>/actor --target_dir <out>
        about 40 seconds each, producing 7.6 GB in HF format (2 safetensors shards)
upload: three ckpts + provenance, 54 files / 22.8 GiB in total
verify: sha256 recorded by HF compared byte-for-byte with local, all 10 large files match, 0 mismatches
```

### 8.3 What this training run did **not** do

- Did not run the GRPO control arm (the user explicitly asked to run only the single arm first)
- Did not run validation during training (`test_freq=-1`, `val_before_train=False`) — saves time,
  and eval needs to be done separately on the same machine to be comparable
- No multiple seeds (single seed)
- No early stopping / lr schedule tuning

### 8.4 Eval definition (not yet run)

The main result is compared against the SFT starting point, and against the P4-reproduced official checkpoint and the paper's Table 2:

| | SFT starting point | Official checkpoint (after RL) | Paper |
|---|---|---|---|
| RoboSpatial Overall | 61.00 ± 0.77 | 65.43–66.00 | 70.00 |
| RefSpatial three items | 52.58 simple average | 53.35 | 53.07 |

Two points that must be kept in mind:

1. **The SFT baseline was measured on 4× A6000 with `gpu_memory_utilization=0.5` (= 24 GB KV pool).**
   This knob is a **fraction of the whole GPU**, not an absolute value; change GPUs without adjusting it and the pool changes; pool size → batch composition →
   floating-point reduction order → near-tie samples flip. `eval/EVAL_FROM_SCRATCH.sh` back-computes from GPU capacity
   to pin the pool back to 24 GB.
2. On RefSpatial, SFT already ties the post-RL official checkpoint (2 questions apart out of 277); **that item has no
   room to explore in the first place**. RoboSpatial is the one to watch.

### 8.5 Where all the artifacts are

```
private HF repo   qyYue1389/spacetools-p7-gflowrl-cprime-8xa40
  global_step_85/ 60/ 30/        three HF-format ckpts, 7.6 GB each
  provenance/                    p7_metrics.csv · as-run scripts · full logs · commit SHAs
  bundle/p7_bundle.tar.gz        full version of this directory (including the complete 9.8 MB training log)
  EVAL_FROM_SCRATCH.sh           bare machine → nine benchmarks, one command
  parse_dump.py                  eval health gate
local (this directory)           trimmed version, without the full training log — that is in the HF bundle
```

---

## 9. Next steps

1. Rent **4 A6000s** (same GPUs as the SFT baseline), run `EVAL_FROM_SCRATCH.sh`, `P7_STEP=85`,
   nine benchmarks, about 2 h 10 min.
2. Report only step 85 as the main result. If it is clearly poor, then evaluate 60 / 30, and state in the report how many checkpoints were evaluated in total
   and which were picked after the fact.
3. If P7 is within 1–2 points of 61.00, additionally run the SFT arm (same machine, same session) for a paired test;
   otherwise the objection "you changed machines, didn't you" cannot be refuted.
4. Conclusion wording: on the ~70% degenerate groups, C′'s gradient is flow matching toward `π_ref·exp(βr)/Z`,
   **not** reward learning; and every update is clipped to pure direction by `grad_clip=1.0`.
