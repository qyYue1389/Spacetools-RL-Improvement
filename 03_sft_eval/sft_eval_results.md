# SFT checkpoint eval results: can go to RL (report as an interval, not a point value)

Run on 2026-09-11/12 · based on `SFT eval handoff doc (not included)` · RoboSpatial two runs

---

## 0. Criterion conclusion

**Pass, but it must be read as an interval.**

| | Criterion | Measured | Conclusion |
|---|---|---|---|
| RoboSpatial | ≥ 60 | expected **61.00%**, two runs 60.29 / 61.71 | ✅ passes in expectation, about 7% chance a single run reads below the line |
| RefSpatial | ≥ 48 | **52.58%** simple / 53.07 weighted | ✅ margin of 13 samples |

**Do not report a single-run point value for RoboSpatial.** The first run's 61.71 looks like 6 samples of margin, the second's 60.29
leaves only 1. The measurement spread is of the same order as the criterion margin; point values invite the wrong reading.

Recommended reporting:

```
RoboSpatial  61.0%  [always-correct lower bound 56.86 · always-wrong upper bound 65.14]  two observations 60.29 / 61.71
```

---

## 1. Scores

### Two runs (RoboSpatial)

| | n | run 1 | run 2 | always correct | always wrong | flip | expected ± sd |
|---|--:|--:|--:|--:|--:|--:|--:|
| RoboSpatial · Overall | 350 | 61.71% (216) | 60.29% (211) | 199 | 122 | 29 | **61.00% ± 0.77 pp** |
| RoboSpatial · VQA | 228 | 70.61% (161) | 71.93% (164) | 152 | 55 | 21 | 71.27% ± 1.00 pp |
| RoboSpatial · Vacant | 122 | 45.08% (55) | 38.52% (47) | 47 | 67 | 8 | 41.80% ± 1.16 pp |
| RefSpatial · Location | 100 | 54.00% (54) | — | | | | |
| RefSpatial · Placement | 100 | 57.00% (57) | — | | | | |
| RefSpatial · Unseen | 77 | 46.75% (36) | — | | | | |
| RefSpatial · three keys | 277 | 52.58 simple / 53.07 weighted | — | | | | |

The criterion 60% = 210/350 falls **inside the flip band** (199 always correct ~ 228 upper bound); clearing needs 11 of the 29 flips to hit,
observed flip success rate ≈ 0.50 (17/29 in run1, 12/29 in run2), probability of a single run reading below the line **6.8%**.
The probability that the median of 5 runs reads below the line is 0.28%, so **rerunning to 5 runs is pointless**: the criterion will not flip because of it.

### Comparison with the paper / official ckpt reproduction

| Table 2 row | n | Paper | P4/P5 reproduction<br>(official ckpt) | **This SFT ckpt** | Fewer correct by |
|---|--:|--:|--:|--:|--:|
| RoboSpatial · VQA | 228 | 79.38 | 73.25–73.68 | 71.27 (expected) | 4–6 questions |
| RoboSpatial · Vacant | 122 | 52.46 | 50.82–51.64 | 41.80 (expected) | 11–12 questions |
| RoboSpatial · Overall‡ | 350 | 70.00 | 65.43–66.00 | 61.00 (expected) | 15–17 questions |
| RefSpatial · three keys | 277 | 53.07 | 53.35 simple / 53.79 weighted | 52.58 / 53.07 | **2 questions** |

‡ Overall is the sample-weighted VQA/Vacant, not an independent measurement (P5 §1.2).
VQA / Vacant are split by GT form (`Yes`/`No` → VQA, point list → Vacant), giving **228 / 122**, matching P5 one for one.

**The official ckpt is the post-RL model, this ckpt is pure SFT**: this compares a starting point against a finished product. Where the gap lands has an explanation:

| | Reasoning chain | Gap | Reading |
|---|---|--:|---|
| RefSpatial | 277/277 single chain `roborefer×1@2t` | 2 questions | The policy is a thin wrapper around RoboRefer; SFT is enough, RL has nothing to push on |
| RoboSpatial | 1.65 calls/sample, main chain covers only 64% | 15–17 questions | Diverse chains, decisions needed: exactly RL's domain |

**Where it ties is exactly where RL is useless, and where it trails is exactly where RL is useful.** A positive signal for "should we go to RL".

> RefSpatial weighted 53.07 equaling the paper's 53.07 is a coincidence and does not constitute evidence (P5 §6).

---

## 2. Non-determinism: the source is on the policy side, the tool side is deterministic

Per-sample comparison of the 350 samples across the two runs:

```
Fully identical (tool_call + tool_response + answer)   289
C  tool_call already differs        (policy, before the call)      11
B  same call, different response   (roborefer side)        0     ← zero
A  same call same response, different answer (policy)           50
Byte-for-byte identical full generations                            151
```

**Zero non-determinism on the tool side.** The `<think>` text drifts (289 − 151 = 138 samples differ in wording but land in the same place),
while tool interactions and final answers are consistent. The non-determinism source from P5 §4.3 shows up on this chain as **sglang's batched inference**:
continuous batching makes batch composition differ between runs, the reduction order changes, and argmax flips at near-ties.

**We do not recommend making sglang bit-for-bit deterministic** (batch=1 sacrifices throughput, or touching the inference path such as the radix cache).
Accept the spread and report intervals: that is already P5's prescription for small benchmarks.

### Why Vacant is the biggest noise source: binary region criterion + continuous prediction

Scoring function `verl/utils/reward_score/robos_all.py`:

```
"point lies within (or on the boundary of) the convex hull of the ground-truth"
compute_points_score(..., point_evaluation_method="convex_hull")
    1.0 if point is within convex hull, 0.0 otherwise
```

**It judges whether the predicted point falls within the convex hull of the 10 GT points; binary.**
Recomputed with the repo's own `_convex_hull` / `_is_point_in_convex_polygon`,
**fully identical to the on-disk acc on 122/122, zero disagreements** (same method as P5's re-check of the pose metric).

**This is easy to misread, so it is stressed separately: the 10 GT points are samples of the acceptable region, not tolerance-circle centers.**
The criterion is convex hull membership, not "distance to the nearest GT point". The counterexample is right in the data: idx 87 in run1, at distance 0.0605 from the nearest
GT point, judged **correct**; in run2, at distance 0.0191, judged **wrong**: closer to some GT point, yet outside the convex hull.
**So analysis of Vacant wrong answers must compute distance to the convex hull boundary; distance to the nearest point is the wrong measure.**

All 8 flipped Vacant samples go "run1 inside the convex hull → run2 outside the convex hull", with small out-of-bounds distances:

```
idx  87  0.0011    idx 107  0.0188    idx 105  0.0460    idx   5  0.0684
idx  22  0.0169    idx 100  0.0456    idx 106  0.0477    idx  45  0.1004
```

Five of the eight are within 0.05, and idx 87 misses by only 0.0011. **Samples hugging the convex hull boundary get pushed across the line by tiny policy-side
perturbations, and the binary score flips.** The coordinate shift directions vary (mean only +0.009/−0.014); it is not an overall drift.

> All eight going the same direction has probability 1/128 under an independent coin-flip model, but these 8 were **filtered by "differs between the two runs"**;
> the filtering is conditioned on the two runs' results, so the same direction is more likely a selection effect than sample correlation. A third run would give a
> different set of flip samples to test this.

By the correct criterion (distance to the convex hull **boundary**), the 67 wrong Vacant samples:

```
median 0.0761   p25 0.0283   p75 0.1790   min 0.0018   max 0.5157
distance to hull edge ≤0.02   11 questions (16%)      ≤0.05   24 questions (36%)      ≤0.10   41 questions (61%)
```

---

## 3. Run health

Both runs pass the five items of P5 §5.1:

```
                     run 1      run 2
OOM                    0          0
Tool error responses (sample-level)   0/627      0/350
Missing <answer>             0          —
Turns exhausted (8-turn limit)     0          —
Truncated tool responses          0          —
Malformed tool calls          0          —
Every sample has a tool call    627/627      —
```

The tool histogram nearly coincides with the official ckpt behavior recorded in P5:

```
robospatial  this run 1.654 calls/sample · 64.3% exactly 2 calls    P5: 1.63 calls/sample · 62.3% main chain
refplacement 100/100 single call   refunseen 77/77 single call     P5: 100%
reflocation  98 single + 2 double                        P5: 100% single
```

Across the four benchmarks only three tools were actually called:

```
roborefer        627 / 627 samples
depth_estimator    1 call (robospatial index=176)
vision_ops         1 call (same sample)
sam2 / vlm / bounding_box / grasp_generator   0 calls
```

### `Error:` counts in the logs checked one by one: no sample affected

Each run has **8** `RuntimeError: Version mismatch` lines, all of the same type and reproducible,
all coming from the Python version problem in §4; **not one touched any sample** (basis: 627/627 samples
have tool calls, 0 tool responses contain error text, and the tool_response of the only sample that used an affected environment
was checked line by line, `acc=1.0`).

---

## 4. Python version mismatch: fixed, and the fix exposed a tighter constraint

### Problem

```
cluster started with  Python 3.11.16   (spacetools-rl)
these processes started with  Python 3.11.0    (spacetools-tool-vlm / spacetools-tool-bbox)
Ray 2.47.1 identical on both sides
```

Ray's `check_version_info` compares the full Python version string by default; `3.11.0 ≠ 3.11.16` is rejected from joining the cluster outright.
The eight errors correspond exactly to the eight actors in those two environments (vlm env sam2×2+depth×2+vlm×1=5,
bbox env bbox×2+vision_ops×1=3).

`RESTORE.sh`'s environment check **prints this version difference verbatim and still judges it OK** (it only verifies that `bin/python` is executable);
a false green light.

### Fix (deviation #6): Ray's built-in minor level

`check_version_info` in `ray/_private/utils.py:1502` supports
`python_version_match_level="minor"`, in which case it only does `logger.warning` and does not raise;
but `node.py:454` (called by `connect` in `worker.py:2437`) does not pass this argument and defaults to `"patch"`.
In **the two 3.11.0 environments** we added this argument at that one call site, with a backup at `node.py.orig`.

**Why not upgrade Python:**

```
CondaToSNonInteractiveError: Terms of Service have not been accepted for:
    https://repo.anaconda.com/pkgs/main   https://repo.anaconda.com/pkgs/r
```

Accepting Anaconda's commercial terms is an organizational legal decision; and re-solving would touch pinned versions like
numpy 1.26.4 / transformers 4.53.2, which is the silent definition change this project fears most.

**Why the minor level is safe and not a workaround:**

```
spacetools-rl    bytecode magic 3495 | pickle proto 4 | 3.11.16
tool-vlm         bytecode magic 3495 | pickle proto 4 | 3.11.0
tool-bbox        bytecode magic 3495 | pickle proto 4 | 3.11.0
```

The magic and pickle protocol are identical on both sides, so code objects and pickles interoperate: exactly the scenario Ray provides the minor level for.
**The change only relaxes one version check; it cannot change any numbers.**

### Triple verification

```
Unit    minor level accepts 3.11.0 vs 3.11.16 (warning only); default patch level still rejects
VERIFY  three items truly green: env gate RC=0 · 4619 .so missing sm_8x 0 · seven tools 7/7 (first time matching baseline)
        cross-env chain 4 links STEP_OK, sam2/depth in tool-vlm, bbox/vision_ops in tool-bbox,
        both patched environments joined the cluster successfully
End-to-end  RuntimeError: Version mismatch  8 → 0
        Python patch version mismatch (warning)  0 → 8
        all 10 tool actors joined (previously only some)
```

### ⚠️ Constraint exposed by the fix: with all tools alive, 4 GPUs cannot fit the policy

blinkdepth (the benchmark used for end-to-end verification) died on the verl side both times:

```
ValueError: Duplicate device type cpu in backend string: nccl.
    The custom backend string argument is invalid: cpu:gloo,cpu:nccl
    fsdp_workers.py:158  backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}"
ActorDiedError: WorkerDict.__init__()
```

`is_cuda_available = torch.cuda.is_available()` at `verl/utils/device.py:104` is
**evaluated at module import time**, so if no GPU is visible at the moment the WorkerDict process imports it, `get_device_name()`
permanently returns `cpu`, assembling the invalid `cpu:gloo,cpu:nccl`.

The root cause is the GPU budget:

```
tool logical reservation 2.3  +  policy needs a whole 1.0  =  3.3 ≤ 4.0
```

The logical total is enough, but **the policy needs one whole GPU**, which requires Ray to pack the 2.3 into 3 GPUs and leave one whole GPU.

**The previous four evals succeeded because the tools' actual usage was below 2.3.** In nvidia-smi's process attribution at the time, only
**7 ToolActors** were counted, while 10 actors are configured (of which `vision_ops` takes no GPU), i.e. some GPU-holding actors
were not in the cluster at the time: exactly the ones blocked by the version check. After the patch fixed it, all 10 joined, the reservation rose to the full 2.3,
and Ray could no longer free up a whole empty GPU.

**So this is not a bug introduced by the patch; the patch removed a failure that had been freeing a GPU for us all along.**

**No effect** on the existing scores: those four benchmarks only called roborefer / depth_estimator / vision_ops,
and all three returned real results successfully (see the histogram in §3). But this is a fact that must be recorded in the provenance:
**those four evals ran with some tool actors absent.**

---

## 5. Another false green light: a leftover Ray address file

Item 3 of `VERIFY.sh` once reported a self-contradictory result: both `✗ chain did not pass` and
"cross-env chain exit code 0" appeared, and the summary still printed `✓ all three passed`.

Root cause: `/root/tmp/ray/ray_current_cluster` (a 19-byte address file left by the previous eval)
made `28_chain.sh`'s `ray.init(num_cpus=8, num_gpus=1, ...)` think the cluster was still there:

```
Connecting to existing Ray cluster at address: 172.27.124.124:6379
ValueError: When connecting to an existing cluster, num_cpus and num_gpus must not be provided.
```

The python block died in the 1st second, and `28_chain.sh` still did `exit 0`.

**Conclusion: do not run VERIFY.sh while a Ray head node is running, or right after one was killed without clearing the temp dir.**
After clearing `/root/tmp/ray` and rerunning, the chain passed normally (ROUTER_OK 33.2 s, four links STEP_OK).

`ray stop --force` reporting "57/58 stopped" is nothing to worry about: the remaining one is a zombie (already defunct, holds no resources).

---

## 6. GPU usage (measured, 1 Hz × 75 samples)

```
GPUs        4× RTX A6000 (49140 MiB each)
Driver      580.173.02   (packaging machine 580.159.04, same branch)
NUM_GPUS  4     ← must override, default 8
EVAL_GPUS 1     ← must override, default 4
Wall time      run1 four keys 32 min 41 s · run2 single key (robospatial) 20 min 55 s
```

### How Ray actually packed things (layout with some tool actors absent)

```
GPU0   20525 MiB (20.0 GiB)   util peak 100% mean 80%
       roborefer          18628 MiB      ToolActor ×2  634 MiB ×2
       SGLangHttpServer     604 MiB

GPU1   47441 MiB (46.3 GiB)   util 0%   ← nearly full, idle the whole time
       ToolActor          30926 MiB (30.2 GiB)  ← Molmo
       ToolActor          12574 MiB (12.3 GiB)
       ToolActor           3920 MiB ( 3.8 GiB)

GPU2     725 MiB ( 0.7 GiB)   util 0%
GPU3   42616 MiB (41.6 GiB)   util peak 100% mean 44%
       WorkerDict         15310 MiB  ← verl FSDP
       sglang::scheduler  27292 MiB  ← policy rollout
```

**The estimate in §2 of the handoff doc is confirmed by measurement.** It predicted the worst-case packing
`vlm 0.6 + depth 0.2 + depth 0.2` = 32.11 + 7.94×2 = **47.99 GiB**, and on that basis judged 4× A100 40GB
"not enough". GPU1 has exactly this packing, measured at **46.3 GiB**, 1.7 GiB off.
**That bad packing was not avoided by the 4th GPU: it really happened, A6000's 48 GB just barely fits it.
On a 40 GB GPU this one will certainly OOM.**

Molmo measured at 30.2 GiB, corroborating the fp32 hole (`vlm.py:116` swallows `dtype: float16`).

> The `vlm peak 7.66 GiB` reported during the smoke test does not contradict this: that is
> `torch.cuda.max_memory_allocated()`, which only counts the torch allocator, not total usage on the GPU.

**GPU1's 46.3 GiB sits at 0% utilization the whole time**: only 1 of robospatial's 350 samples used
depth/vision_ops, so nearly a whole A6000 stays resident for a single call. This is the measured basis for the "two levers for saving GPUs"
(fix fp32, trim the tool set per benchmark).

### The `EVAL_GPUS` default makes Ray hang forever

```
Seven-tool logical reservation (run_eval.sh:132-142, consistent with the table in §2 of the handoff doc)
roborefer 1×0.6 + vlm 1×0.6 + sam2 2×0.2 + depth 2×0.2
        + bbox 2×0.1 + vision_ops 1×0 + graspgen 1×0.1  = 2.3
default EVAL_GPUS=4  →  2.3 + 4 = 6.3 > 4.0   Ray does not error, actors queue forever
set to 1            →  2.3 + 1 = 3.3 ≤ 4.0   logically enough, but see the whole-GPU constraint in §4
```

---

## 7. Provenance

```
SFT ckpt   qyYue1389/spacetools-sft-v1-4xa6000
           commit 91fd4bdf6d2ffe3b355cc650e71925c6e91a15c5
           Qwen2_5_VLForConditionalGeneration · bfloat16 · 7.6 GB
           num_attention_heads=16 · num_hidden_layers=36

Env package     qyYue1389/spacetools-eval-env
           spacetools-envs-20260910-0959.tar.zst
           22102227452 bytes (after concatenation, byte-for-byte match with the README record) · sha256 12/12 OK

Repos       SpaceTools           17d585539b6cc32f2b3c068ea2591762b4583fd9
           SpaceTools-RL        54270e82443d3d2a4c2a737c2d3b33314a991fcc
           SpaceTools-Toolshed  4f0512d092f53abc1e6c5c934245bf83a211466a
           GraspGen             9b3cfc1e5b664698e047ddd482832f6e7796380c
           RoboRefer            d97a995ad28376720a4c8beb64915c58ed16c844

Data       siyich/spacetools-eval-benchmarks @ 1d539ac935872c7aa712c85a77bf4b0cb469c8e8
           350 / 100 / 100 / 77 = 627 ✓ · blinkdepth 124 ✓
           (this revision and main have byte-for-byte identical sizes for these files)

Machine       4× RTX A6000 · Ubuntu 24.04.2 · glibc 2.39 · driver 580.173.02 · CUDA 13.0
Decoding       greedy (verl val_kwargs default temperature 0 / do_sample False,
           not overridden by run_eval.sh), but the policy side is still not bit-for-bit deterministic, see §2
Tool state   ⚠️ both runs of the four benchmarks completed with some tool actors absent (see §4)
```

---

## 8. Deviations made to get it running (repo code unchanged, six config/library patches)

| # | Deviation | Why it was necessary |
|---|---|---|
| 1 | Install `nvidia-driver-570-server` (actually got 580.173.02) | The machine was delivered with **no driver installed at all**: GPUs visible on PCI, but no `/dev/nvidia*`, no kernel module, 0 nvidia packages in `dpkg -l` |
| 2 | `NUM_GPUS=4 EVAL_GPUS=1` | Defaults 8 / 4; the latter makes Ray hang forever (§6) |
| 3 | `BASH_ENV=/opt/conda-st/etc/profile.d/conda.sh` | `conda activate` at `run_eval.sh:84` does not work in a subprocess; conda is a shell function and does not cross processes |
| 4 | Symlink benchmarks to the nested paths the script expects | The `BENCHMARKS` mapping expects things like `refspatial_bench/location.parquet`, while the dataset (**both** main and the pinned version) is flat `data/<key>.parquet`; without handling it you get `Missing: / exit 1` directly. `run_eval.sh` was not changed because its commit is recorded in MANIFEST |
| 5 | Add `activate.d/zz_cuda_home.sh` to the roborefer environment | `04_smoke.sh:42` calls the env python directly without activating conda, torch cannot find nvcc → deepspeed (a hard dependency of llava inference, README hole #17) raises `MissingCUDAException`. The packaging machine has a system `/usr/local/cuda-12.8` that hits torch's third fallback. The real eval goes through Ray `runtime_env={"conda":...}`, which activates conda and resolves it anyway |
| 6 | **Add `python_version_match_level="minor"` to `ray/_private/node.py:454` in the two 3.11.0 environments** | See §4. Does not change numbers, only relaxes a version check; backup at `node.py.orig` |

**Things not done**: did not fix the fp32 in `vlm.py:116` (the handoff doc requires keeping the definition), did not change any repo code,
did not install a system CUDA toolkit, did not accept the Anaconda ToS.

Deviations #5 and #6 have been pushed to the HF repo with `POSTRESTORE.sh`, take effect automatically on a new machine,
and are guarded by the pre-check item 0 in `VERIFY.sh`; details in `00_environment/eval_rl_env/env_package_revision_20260912.md`.

---

## 9. Next steps

**Can go to RL.** In priority order:

1. **The RL GPU budget must be recomputed for "all tools alive", and must guarantee the policy a dedicated whole GPU.**
   Lesson from §4: at eval scale 4 GPUs already cannot fit (tools 2.3 + a whole policy GPU 1.0,
   requiring Ray to happen to pack it as 1.0/1.0/0.3/empty); RL's actor counts are far larger
   (sam2/depth/bbox/graspgen 5 each, vision_ops 8), so it will certainly be tighter.
   Possible levers: fix the fp32 in `vlm.py:116` (Molmo measured at 30.2 GiB, halving frees 15 GiB),
   trim the tool set per benchmark, or explicitly pin the policy to its own GPU.

2. **Always report RoboSpatial as an interval** (61.0% [56.9 / 65.1], two runs 60.29 / 61.71).
   **Do not rerun to 5 runs**: the probability that the median of 5 runs reads below the line is only 0.28%, the criterion will not flip.
   To quantify "whether the true spread is wider than ±0.8" (two runs can only see a **lower bound** on the flip set),
   a single run gets most of the information: see how many of the 199 "always correct" flip in a third run.
   This number matters beyond this criterion: if the true spread is clearly wider, every single-run number in this project
   (including P4/P5) should come with wider error bars.

3. **Start 1 Hz GPU trace sampling in the RL stage** (P4 did exactly this in `p4/logs/`).

4. **Next time the environment package is rebuilt, unify Python in all five environments to 3.11.16**, so deviation #6 can be removed.

### An input to RL reward design

The Vacant reward is a **binary convex hull membership criterion on a continuous prediction** (§2). Edge samples (of the 67 wrong answers,
16% within 0.02, 36% within 0.05) jump between 0 and 1 under tiny perturbations. This is both the source of eval noise
and means that RL on this kind of task gets a **discontinuous, low-information reward signal**.
If RL is to make progress on RoboSpatial, reward shaping for this cell deserves separate consideration.
