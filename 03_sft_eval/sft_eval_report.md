# SFT checkpoint eval report

Subject: `qyYue1389/spacetools-sft-v1-4xa6000` @ `91fd4bdf6d2ffe3b355cc650e71925c6e91a15c5`
Run: 2026-09-11/12 · 4× RTX A6000 · based on `SFT eval handoff doc (not included)`
Companions: `03_sft_eval/sft_eval_results.md` (raw numbers and provenance), `00_environment/eval_rl_env/env_package_revision_20260912.md` (environment-side changes)

---

## Summary

The self-trained SFT checkpoint completed 627 samples on four spatial reasoning benchmarks. **The criterion passes; it can go to RL**:

```
RoboSpatial   61.0%  [56.9 lower bound / 65.1 upper bound]   two observations 60.29 / 61.71    criterion ≥60  ✅
RefSpatial    52.58% simple mean / 53.07 weighted                              criterion ≥48  ✅
```

More valuable than these two numbers is the **failure mode**: on RefSpatial the SFT is already tied with the official
checkpoint after RL from the paper (2 questions apart out of 277), while on RoboSpatial it trails by 15–17 questions. The reason it trails is not weak reasoning,
but that **it does not call tools when it should**: across 350 samples `depth_estimator` was called only 1 time,
and that single call got the answer right. This is exactly the kind of problem RL can fix and SFT cannot.

---

## 1. How the criterion was set

The go/no-go set in the previous session:

| Result | Conclusion |
|---|---|
| RoboSpatial ≥ 60 **and** RefSpatial ≥ 48 | checkpoint usable, go to RL |
| RoboSpatial < 55 **or** RefSpatial < 40 | stop and investigate, do not go to RL |
| In between | first confirm the run was healthy, then align item by item against the paper table |

The reading order is explicitly specified and **must not be reversed**: first count OOMs → then check for silent tool errors → only then look at scores.
Reasoning in §3.4.

Why these four keys: they cover the spatial reasoning capabilities of the paper's main table, and none of them needs
`grasp_generator` (of the seven tools, it is the only one completely unused on these four keys yet it must still be registered in the schema,
because the interfaces the model has seen cannot be removed). `boppose` / `bopgrasp` are explicitly excluded: the former has a complex metric mapping,
the latter would pull in `grasp_generator` and add variables.

---

## 2. How the eval was done

### 2.1 System structure

What is being evaluated is not one model but a **model + seven tools** system. The policy model
(Qwen2.5-VL-3B, the SFT ckpt in this run) is served by sglang; the tools are seven Ray actors,
**each running in its own conda environment**:

```
Policy model  Qwen2.5-VL-3B (sglang)            ← the subject being evaluated
Tools                                        conda env        num_actors × num_gpus
  roborefer      spatial referring grounding (RoboRefer-8B)  tool-roborefer    1 × 0.6
  vlm            Molmo-7B-D                   tool-vlm          1 × 0.6
  sam2           segmentation                         tool-vlm          2 × 0.2
  depth_estimator DepthPro depth + point cloud          tool-vlm          2 × 0.2
  bounding_box   3D oriented bounding box                tool-bbox         2 × 0.1
  vision_ops     array indexing etc.                   tool-bbox         1 × 0
  grasp_generator GraspGen                     tool-graspgen     1 × 0.1
                                                        total logical reservation  2.3 GPUs
```

The dependencies of the five environments conflict with each other (torch 2.3.1 / 2.5.1 / 2.9.1, numpy 1.26.4 and 2.4.6 side by side),
so they must be separate environments. Toolshed uses Ray's `runtime_env={"conda": <env>}` to start each tool
in its environment (`router.py:158-164`); variables (ndarray) passed across environments are serialized by Ray.

**The three columns in the table above were not chosen by us; they are hard-coded in `run_eval.sh`**: the `TOOL_CONFIGS` dict in the inline
python under the `VERSION=v1` branch (`run_eval.sh:132-142`), which also holds the `conda_env` assignment:

```python
'sam2': {'num_actors': 2, 'resources': {'num_gpus': 0.2},
         'conda_env': 'spacetools-tool-vlm', 'timeout': 600, 'args': {...}}
```

This run **did not change a single character** (otherwise the definition would not match the paper's). The as-run version is stored at
`sft-eval-artifacts/config/run_eval.sh.asrun`.

**`num_gpus` is Ray's logical reservation, not a GPU memory quota; fractions do not partition GPU memory.** Actors on the same physical GPU
share all of that GPU's memory; Ray only guarantees that "the logical shares landing on one GPU add up to no more than 1.0".
So "2.3 GPUs total" only decides **whether things can be scheduled**, and does not guarantee that no GPU will OOM.
GPU memory has to be computed separately; see §6.1 (Molmo fp32 measured at 30.2 GiB) and §6.2 (measured packing layout).

### 2.2 Interaction form

The policy model sees an image and a spatial reasoning question and follows the four steps of the system prompt: first a `<think>` inner monologue,
call tools when it needs information, continue reasoning after getting the tool return, and finally give the answer in `<answer>`. At most 8 turns.

A real complete trajectory (RefSpatial-Location, index=0, answered correctly):

```
<think> I need to find the orange box in the image. Looking at the image, I can see
a box with "Nature Valley" and "GRANOOLA" text on it, which appears to be a Nature
Valley granola bar box. This box is in the upper right area of the image. I will use
the roborefer.detect_one tool to locate this box. </think>
<tool_call>
{"name": "roborefer.detect_one", "arguments": {"obj_name": "orange box", "image_index": 0}}
</tool_call>
          ↓ tool returns
<tool_response>
Detected 1 instance(s) of 'orange box': [(0.95, 0.247)]
</tool_response>
          ↓
<think> I have successfully located the orange box at the coordinates (0.95, 0.247).
Now, I can format this as a list of tuples as requested. </think>
<answer> [(0.95, 0.247)] </answer>
```

### 2.3 Definitions

- **Decoding is greedy.** verl's `val_kwargs` defaults to `temperature 0 / top_p 1.0 / n 1 /
  do_sample False`, and the released repo's `run_eval.sh` does not override it. The `rollout.n=5` in the script
  is the group size for training and does not apply to evaluation.
- **All weights are pulled at pinned revisions** (`WEIGHTS_PINS.txt`), including the benchmark dataset
  `siyich/spacetools-eval-benchmarks @ 1d539ac9...`.
- **The fp32 issue in `vlm.py:116` is not fixed** (see §6.1): the paper's results were produced under that behavior, and
  the definition must stay consistent.

### 2.4 Scoring function (confirmed by reading the source, not by guessing)

For the three RefSpatial keys and the point-type questions of RoboSpatial, the criterion lives in
`verl/utils/reward_score/robos_all.py`:

```
compute_points_score(..., point_evaluation_method="convex_hull")
    1.0 if point is within (or on the boundary of) the convex hull of the GT points
    0.0 otherwise
```

**GT** (ground truth) is given as **10 points**, because a "vacant space" is a region, not a point,
and the dataset describes it with 10 sample points. Scoring takes the **convex hull** of these 10 points (the smallest convex polygon that encloses them all)
and asks only one thing: **is the predicted point inside this polygon**. Inside is 1 point, outside is 0 points, no partial credit.

Take index=1 as an example; the 10 GT points cluster in `x ∈ [0.701, 0.757]`, `y ∈ [0.615, 0.649]`:

```
  y=0.60     ·  ← model's answer (0.715, 0.6)
  y=0.615  ┌──●──────────●──┐   ← convex hull
           │ ●  ●   ●  ●    │      10 GT points inside
  y=0.649  └──────●─────────┘
```

x is within range, but y=0.6 is 0.015 higher than the topmost GT point, **it falls outside the convex hull → 0 points**.

**This is not "distance to the nearest GT point"; the two criteria are not equivalent.** The data has a direct counterexample: idx 87 across two runs,
nearest-GT-point distance 0.0605 judged **correct**, 0.0191 judged **wrong**: closer, yet wrong. A point can sit just outside the convex hull
while grazing one vertex very closely, or sit right in the middle of the convex hull while being far from all 10 sample points.
**So when analyzing "by how much" a wrong answer missed, compute the distance to the convex hull boundary; distance to the nearest point is the wrong measure.**

"Recomputed with the repo's own `_convex_hull` / `_is_point_in_convex_polygon`,
**122/122 identical, zero disagreements**" means: we did not trust our understanding just from reading the code; we took the predicted points and GT points from the rollouts,
imported those two functions from the repo, recomputed "inside or not", and compared against the `acc` that verl wrote to disk,
one by one. This confirms two things at once: the criterion is understood correctly, and `acc` really is convex hull membership and nothing else.

RoboSpatial's Yes/No questions go through `compute_yesno_score`, and split naturally by GT form into
**VQA 228 questions** (`Yes`/`No`) and **Vacant 122 questions** (point lists), matching P5's split one for one.

---

## 3. Process: from bare machine to scores

### 3.1 Machine acceptance check

```
GPU arch   A100 (sm_80) / A6000 (sm_86) / L40S (sm_89) usable
           H100 (sm_90) not usable: compiled without +PTX, no intermediate code that can JIT to Hopper
glibc      must not be older than Ubuntu 24.04's 2.39
Driver     ≥ 550
Container disk     ≥ 80 GB; network volume ≥ 100 GB
Restore path   must be /opt/conda-st + /opt/spacetools; absolute paths are baked into the conda environments
```

**In practice one more item must be added: the delivered machine may have no driver installed at all.** On the machine we got this time, `lspci` could see
4 GA102GL, but `/dev/nvidia*` did not exist, the kernel module was not loaded, and `dpkg -l | grep nvidia`
showed 0 packages. After installing `nvidia-driver-570-server` (actually got 580.173.02), DKMS compilation, and
`modprobe`, all four GPUs worked. **So the first check should be `nvidia-smi -L` rather than `nvidia-smi`**;
the error when the latter does not exist (`command not found`) is easily mistaken for a PATH issue.

Final environment compared with the packaging machine:

| | Packaging machine | This run |
|---|---|---|
| GPU | RTX A6000 | RTX A6000 ✓ |
| Driver | 580.159.04 | 580.173.02 ✓ same branch |
| OS | Ubuntu 24.04.3 | 24.04.2 ✓ |
| glibc | — | 2.39 ✓ |

### 3.2 Environment restore

```
hf download qyYue1389/spacetools-eval-env     21 GB, 6 shards
sha256sum -c                                   12/12 OK
FETCH_WEIGHTS.sh                               79 GB, at pinned revisions
RESTORE.sh                                     concatenated 22102227452 bytes (byte-for-byte match with the README record)
                                               extracted to /opt, five environments in place
POSTRESTORE.sh                                 ← added after this run, see 00_environment/eval_rl_env/env_package_revision_20260912.md
VERIFY.sh                                      four acceptance checks
```

Weights on disk, measured:

```
/workspace/hf           54G   ← MANIFEST records 54G
/workspace/checkpoints  34G   ← 25.7G base weights + 7.6G SFT ckpt
depth_pro.pt sha256     3eb35ca6...85c0ce   actual == expected
```

SFT checkpoint confirmed:

```
architectures   Qwen2_5_VLForConditionalGeneration
model_type      qwen2_5_vl        dtype  bfloat16        7.6 GB
num_attention_heads 16   num_hidden_layers 36
revision        91fd4bdf...a15c5  ← use the SHA rather than main, otherwise results are not reproducible
```

### 3.3 Acceptance checks (four items)

```
Five-env import gate     all pass
Architecture scan               4619 .so files · 108 contain cubin · self-built extensions missing sm_8x: 0 total
                       (1439+591+886+519+1184 / 37+14+21+1+35)
Seven-tool smoke test             7/7, real weights loaded, real outputs
Ray cross-env chain           5 tools · failed links 0 · ndarray passed correctly in both directions
```

The three numbers (4619 / 108 / 0) **match the packaging-time baseline exactly**, showing this environment is reproducible on a new machine.

The cross-env chain item tests the real form of the eval: a variable produced by tool A is fed to tool B, while A and B are in
different conda environments:

```
sam2 (tool-vlm, numpy 1.26.4)  produces $segmentation_mask (bool ndarray)
     → bounding_box (tool-bbox, numpy 2.4.6)  consumes it
depth_estimator (tool-vlm, np 1.26.4)  produces $depth_map (float ndarray)
     → vision_ops (tool-bbox, np 2.4.6)  consumes it
```

### 3.4 Running the eval, and why the reading order must not be reversed

```bash
NUM_GPUS=4 EVAL_GPUS=1 \
ROBOREFER_MODEL=/workspace/checkpoints/RoboRefer-8B-SFT \
DEPTH_CHECKPOINT=/workspace/checkpoints/depth_pro.pt \
bash run_eval.sh /workspace/checkpoints/sft-ckpt \
     robospatial reflocation refplacement refunseen
```

627 samples, 32 min 41 s.

**The first reading step is counting OOMs, not looking at scores**: OOM samples are counted in the denominator, and the score gets diluted into
a number that is "low but plausible-looking".

**The second step is checking whether tools silently returned errors; this is the step most easily skipped and the most expensive to miss.** When a toolshed tool
fails it **does not raise**; it wraps the error message in a normal `ToolResult` and returns it; the router also swallows exceptions from the actor
and wraps them as normal returns. It looks like this:

```
CALL_OK   depth_estimator -> ModuleNotFoundError: No module named 'numpy._core.numeric'
                             return type: ToolResult
```

Note that it says `CALL_OK`. **The caller gets no exception at all.** If a tool keeps returning
error text throughout the eval, the model keeps reasoning on garbage input; the eval finishes, reports no errors, and gives a normal-looking
score, and then you go suspect the checkpoint.

So besides the log-level `grep -cE "Error:|ERROR:toolshed"`, this run also checked tool responses per-sample **on the rollouts written to disk**
(see §4.3). Note that grep uses **substring** matching: `TypeError:`
contains `Error:`, so it is caught; switching to a "stricter" prefix match would actually miss it.

---

## 4. Results

### 4.1 Scores

| | n | run 1 | run 2 | always correct | always wrong | flip | expected ± sd |
|---|--:|--:|--:|--:|--:|--:|--:|
| RoboSpatial · Overall | 350 | 61.71% (216) | 60.29% (211) | 199 | 122 | 29 | **61.00% ± 0.77 pp** |
| RoboSpatial · VQA | 228 | 70.61% (161) | 71.93% (164) | 152 | 55 | 21 | 71.27% |
| RoboSpatial · Vacant | 122 | 45.08% (55) | 38.52% (47) | 47 | 67 | 8 | 41.80% |
| RefSpatial · Location | 100 | 54.00% (54) | — | | | | |
| RefSpatial · Placement | 100 | 57.00% (57) | — | | | | |
| RefSpatial · Unseen | 77 | 46.75% (36) | — | | | | |
| RefSpatial · three keys | 277 | 52.58 simple / 53.07 weighted | — | | | | |

**RoboSpatial must be reported as an interval.** The criterion 60% = 210/350 falls **inside the flip band** (199 always correct ~ 228 upper bound);
whether a single run reads above or below the line depends on how those 29 flip samples land: observed flip success rate ≈ 0.50,
clearing needs 11 hits, **the probability of a single run reading below the line is about 6.8%**.
The probability that the median of 5 runs reads below the line is only 0.28%, so rerunning to 5 runs would not change the conclusion.

### 4.2 Item-by-item comparison with the paper / official ckpt

| Table 2 row | n | Paper | P4/P5 reproduction<br>(official ckpt, after RL) | **This SFT ckpt** | Fewer correct by |
|---|--:|--:|--:|--:|--:|
| RoboSpatial · VQA | 228 | 79.38 | 73.25–73.68 | 71.27 | 4–6 questions |
| RoboSpatial · Vacant | 122 | 52.46 | 50.82–51.64 | 41.80 | 11–12 questions |
| RoboSpatial · Overall‡ | 350 | 70.00 | 65.43–66.00 | 61.00 | 15–17 questions |
| RefSpatial · three keys | 277 | 53.07 | 53.35 simple / 53.79 weighted | 52.58 / 53.07 | **2 questions** |

‡ Overall is the sample-weighted VQA/Vacant, not an independent measurement.

**Converting to question counts is necessary**: n differs by an order of magnitude across benchmarks, so pp are not directly comparable.
RefSpatial's −0.77 pp converts to **2 questions out of 277**, statistically indistinguishable
(at n=277, p≈0.53 the standard error is about 3.0 pp, and the gap is 0.26 standard errors);
RoboSpatial's −4 pp is **14 questions out of 350**, and that is a real gap.

### 4.3 Run health: why the numbers above can be trusted

```
OOM                          0
Sample-level tool error responses            0 / 627
Every sample has a tool call          627 / 627
Missing <answer>                   0
Turns exhausted (8-turn limit)           0
Truncated tool responses                0
Malformed tool calls                0
Row count reconciliation      350 + 100 + 100 + 77 = 627 ✓
```

The tool-call distribution nearly coincides with the official ckpt behavior recorded in P5:

```
robospatial   {1 call: 124, 2 calls: 225, 3 calls: 1}   mean 1.649 calls/sample
                                                  P5 (official ckpt): 1.63 calls/sample
reflocation   {1 call: 98, 2 calls: 2}                mean 1.020
refplacement  {1 call: 100}                        mean 1.000     P5: 1.00
refunseen     {1 call: 77}                         mean 1.000     P5: 1.00
```

Only three tools were actually called:

```
roborefer          627 / 627 samples
depth_estimator      1 call
vision_ops           1 call
sam2 / vlm / bounding_box / grasp_generator    0 calls
```

---

## 5. Trajectory analysis: what happened behind the scores

### 5.1 The three RefSpatial keys: the policy is a thin wrapper around RoboRefer

277/277 samples have only one chain: `roborefer×1@2t`. The trajectory in §2.2 is the entire form:
call `detect_one` once and copy the returned coordinates into `<answer>`.

> **Chain signature notation** (following P5 §5.3): `tool×call count@N t`, where N is the number of assistant turns.
> 1 call always takes 2 turns: one turn issues the call, the tool result is injected back as a **user turn**, and another turn answers.
> The ratio of calls to turns shows whether the model parallelizes: `roborefer×2@2t` is two calls packed into the same turn
> (§5.2), `depth×1+roborefer×2+vision_ops×2@3t` is 5 calls in only 3 turns (§5.4).
> The `num_turns=4` that verl prints in the log counts all turns (including user turns) and describes the same trajectory.

**This explains why SFT can tie the post-RL official ckpt at this tier (2 questions apart)**:
the task degenerates into "call a tool once and relay the result verbatim", which SFT can fully learn, and RL has nothing to push on.
In paper Table 2, RoboRefer-8B-SFT run alone on RefSpatial scores 48.37, and SpaceTools-3B (after RL)
scores 53.07. **The 48.37 → 52.58 stretch can be had without RL**, as confirmed by this eval: our pure SFT
ckpt wrapped around the same roborefer reaches 52.58, and the remaining gap to 53.07 is only 0.49 pp (about 2 questions, within noise).

"Plugging the tool in" refers to how the two systems differ **structurally**:

```
RoboRefer-8B-SFT alone   image + original question text → RoboRefer-8B → point                    48.37
SpaceTools-3B           image + question → 3B policy writes its own obj_name
                                  → same RoboRefer-8B → point
                                  → policy copies into <answer>                        53.07
```

Same perception model, with a 3B policy layer in front: the policy is smaller than the tool, yet adds 4.7 pp.
**What improves is not perception, it is what that wrapper layer does.**

What exactly contributes those 4 pp has **not been decomposed yet**. Candidates are "the policy rewrites `obj_name` itself"
(measured in §5.5: 80% of correctly answered samples used a >30-character query, only 51% of wrong ones did) and "output format normalization";
the validation/retry candidate does not hold up empirically: §5.5 shows the policy does no validation of tool returns at all.

> Note: 48.37 is the paper Table 2 baseline row cited via the P5 report; this run did not reproduce it, nor verify whether it
> used the same parquet and scoring function.

### 5.2 RoboSpatial VQA: two localizations + qualitative reasoning

A correctly answered sample (index=122, GT `Yes`, question "can the tissue box fit behind the chair"):

```
<think> ... 1. Locate the tissue box  2. Locate the chair
             3. Assess the available space behind the chair ... </think>
<tool_call> roborefer.detect_one  obj_name="tissue box" </tool_call>
<tool_call> roborefer.detect_one  obj_name="chair" </tool_call>
   → Detected 1 instance(s) of 'tissue box': [(0.306, 0.531)]
   → Detected 1 instance(s) of 'chair':      [(0.477, 0.517)]
<think> Tissue box at (0.306, 0.531) - left-center, on what appears to be a desk
        Chair at (0.477, 0.517) - center of the image
        "Behind the chair" would mean the area further away from the camera ...
        Looking at the image, behind the chair there appears to be wall space and
        potentially some floor space ... </think>
```

Correct, but note the nature of the reasoning: **what it gets is two 2D centroids, and then it fills in the depth and size judgment
by "eyeballing the image".** Two coordinates carry almost no information for the question "can it fit".

### 5.3 Key counterexample: should call the depth tool but does not (index=128, GT `No`, answered wrong)

Question "can the cup fit in front of the speaker":

```
<tool_call> roborefer.detect_one  obj_name="cup" </tool_call>
<tool_call> roborefer.detect_one  obj_name="speaker" </tool_call>
   → Detected 1 instance(s) of 'cup':     [(0.78, 0.494)]
   → Detected 1 instance(s) of 'speaker': [(0.154, 0.491)]
<think> The question asks if the cup can fit "in front of" the speaker.
        This means: is there space between the speaker and the camera/viewer?
        ... There is desk space in front of the speaker
        ... There appears to be adequate desk space in front of the speaker ... </think>
Answer: Yes        Ground truth: No
```

**It correctly translates the question into a depth question on its own** ("is there space between the speaker and the camera"),
**and then answers it with two 2D coordinates whose y values are nearly identical** (0.494 vs 0.491),
finishing with "there appears to be desk space".

`depth_estimator` and `vision_ops` are right there in its schema, and combined they answer exactly this question.
It did not call them.

### 5.4 The only sample that called the full chain got it right (index=176, acc=1.0)

Out of 350 samples, only this one took the full chain across three conda environments:

```
depth_estimator.estimate_depth_with_pointcloud        (tool-vlm, numpy 1.26.4)
   → Generated 1428x1932 depth map and 2,758,896 3D points.
     Depth range in point cloud: 1.93m to 5.44m (mean: 3.39m).
     Use $depth_map, $point_cloud and $focal_length_px to reference the results.
roborefer.detect_one  "bed"     → [(0.336, 0.551)]
roborefer.detect_one  "table"   → [(0.844, 0.592)]
vision_ops.index_at   $depth_map @ (0.336, 0.551)   (tool-bbox, numpy 2.4.6)
   → Pixel value at (0.336, 0.551) is 2.7431180477142334
vision_ops.index_at   $depth_map @ (0.844, 0.592)
   → Pixel value at (0.844, 0.592) is 2.944430351257324
```

It got the **real distances in meters** for both objects (2.74 m vs 2.94 m), compared them, and answered correctly.

**§5.3 and §5.4 together are the most important finding of this eval: the capability exists, the trigger rate is 1/350.**

### 5.5 Vacant: stuffing the whole spatial condition from the question into the "object name" argument, then not validating

First, what the two terms refer to in this sample:

**Spatial predicate** = the part of the question text that describes a **spatial relation**. Question index=1 asks for "a vacant spot where something can be placed,
close to the vacuum and to its left", where

```
close to and to the left of the vacuum
```

is the spatial predicate. It describes a **relation** between objects (close to, to the left of), not any single object.

**Referring detector** = the `roborefer` tool (RoboRefer-8B underneath). Its job is "given a noun phrase
describing an object, return that object's coordinates in the image". Its interface signature is

```
roborefer.detect_one(obj_name=..., image_index=...)
                     ^^^^^^^^ the parameter is literally named obj_name: object name
```

What the model does is fill the entire spatial predicate, prefix included, into `obj_name`, a slot meant for an object name:

```
<tool_call>
{"name": "roborefer.detect_one", "arguments": {"obj_name":
  "point within the vacant space suitable for placing an object close to and to
   the left of the vacuum", "image_index": 0}}
</tool_call>
   → Detected 1 instance(s) of '...': [(0.715, 0.6)]
<think> The tool detected a point at (0.715, 0.6) which is suitable for placing an
        object close to and to the left of the vacuum. This point is within the
        vacant space and is left of the vacuum. </think>
<answer>[(0.715, 0.6)]</answer>

GT convex hull (10 points): x ∈ [0.701, 0.757],  y ∈ [0.615, 0.649]
Prediction (0.715, 0.6): x within range, y off by 0.015, falls outside the convex hull  → 0 points
```

Two things worth noting:

1. **It hands the entire spatial predicate to `detect_one` as an object name**: this outsources the spatial reasoning of "what counts as vacant, which side counts as
   left" wholesale to roborefer, and the model itself only relays. Interestingly, this approach
   **is statistically effective**: among correctly answered Vacant samples,
   80% used an `obj_name` of >30 characters (mean 54 characters), versus only 51% of wrong ones (mean 38 characters):
   the more complete the query, the more likely it is correct. roborefer is a referring-expression model to begin with and can handle long predicates.
2. **It does no validation of the tool return at all**, and directly asserts "This point is within the vacant space";
   it has no basis for saying so. Yet `vision_ops.index_at` + `$depth_map` could verify exactly which plane that point
   lands on.

Overall the Vacant wrong answers are "near misses": the distance of the 67 wrong answers to the convex hull **boundary** has
median 0.0761; 16% are within 0.02, 36% within 0.05, 61% within 0.10.

### 5.6 VQA confusion matrix: learned the prior, did not learn to discriminate

A **confusion matrix** cross-tabulates "ground truth" against "what the model answered": **rows are the ground truth, columns are the model's answer**;
cells on the diagonal are correct, off-diagonal cells are wrong. VQA answers are only Yes / No, so it is 2×2.

It is more useful than a single accuracy because the one number "70.61%" cannot distinguish two completely different situations: **both classes mediocre**,
or **one class very good and the other a coin flip**. The two situations call for different actions.

228 VQA questions:

```
                 Model says no    Model says yes   |  accuracy of this row
  GT no             31           32      |   49.21%   (63 questions)
                   ┗━ correct ┛      ┗━ wrong ┛
  GT yes            32          133      |   80.61%   (165 questions)
                   ┗━ wrong ┛      ┗━ correct ┛
  ────────────────────────────────────────
  Model answer distribution    no 63       yes 165
  GT distribution        no 63       yes 165     ← identical to the line above
```

**The marginal distribution of predictions is identical to the marginal distribution of the ground truth (165/63), but accuracy on `gt_no` is exactly
a coin flip (49.21%).** The model learned the prior "how often to say yes", not the discriminative signal "when to say no";
SFT's likelihood objective can reach a decent loss just by matching the marginal distribution.

#### One pair of samples makes this crystal clear

index=127 and 128 are adjacent questions on **the same desktop photo**: the evidence is that the speaker coordinates roborefer
detected in both questions are exactly the same, both `(0.154, 0.491)`. Same sentence pattern, the model answers **Yes** to both,
while the ground truth is Yes for one and No for the other.

```
index=127   gt=Yes   acc=1.0
Question: Can the speaker fit in front of the monitor?
  → roborefer.detect_one("speaker")  → [(0.154, 0.491)]
  → roborefer.detect_one("monitor")  → [(0.553, 0.194)]
  <think> ... The space in front of the monitor appears to be very limited -
          there's just a small amount of desk space between the monitor base and
          where the camera is positioned.
          Given the speaker's size and the limited space in front of the monitor,
          it would likely fit, though it would be a very tight fit. </think>
  <answer> Yes </answer>                                     ← right by luck

index=128   gt=No    acc=0.0
Question: Can the cup fit in front of the speaker?
  → roborefer.detect_one("cup")      → [(0.78, 0.494)]
  → roborefer.detect_one("speaker")  → [(0.154, 0.491)]      ← same scene
  <think> ... They are at similar vertical positions (y ≈ 0.49)
          The question asks if the cup can fit "in front of" the speaker. This
          means: is there space between the camera viewpoint and the speaker ...
          There appears to be desk space in front of the speaker ...
          it appears there would be sufficient room to place the cup there. </think>
  <answer> Yes </answer>                                     ← wrong
```

The two trajectories have **exactly the same structure**: two roborefer calls get two 2D centroids → fill in depth
and size by "eyeballing the image" → answer Yes. In 127 it even reasons all the way to "space is very limited, it would be tight", and the conclusion is still
"would likely fit"; in 128 it switches to "there appears to be desk space", and again Yes.

**Neither question ever queried depth.** In 128 it translated the question correctly on its own:
"is there space between the camera viewpoint and the speaker" is a depth question,
yet all it has is x/y, and the two objects' y are nearly identical (0.494 vs 0.491).
`depth_estimator` is right there in its schema, §5.4 proves it can use it, and here it did not call it.

So the model is not discriminating; it is **outputting the prior answer for this sentence pattern**. The shape of the confusion matrix is exactly the
statistical consequence of this behavior: the marginals match, and the `gt_no` row drops to a coin flip.

Compare P5's same table for the official (post-RL) checkpoint: `gt_no` 47.62%, `gt_yes` 83.03%.
**This ckpt is actually slightly better on `gt_no` (+1 question); the 4–6 questions it trails by all come from `gt_yes`.**

This also yields a **negative expectation**: the coin-flip level on `gt_no` **still exists** after the paper's RL,
so do not expect RL to fix it; it is more likely a property at the task/data level.

---

## 6. Problems in the repo itself (hit and confirmed in this run)

These are not environment setup problems; they come with the upstream code/scripts and will be hit on any machine.

### 6.1 `vlm.py:116` swallows the dtype from the config

`run_eval.sh` passes `'dtype': 'float16'` to the vlm tool, but `vlm.py:116` hard-codes
`torch_dtype="auto"`, so Molmo loads in **fp32**. Measured single-GPU usage: **30.2 GiB**.

**Not fixed in this run**: the paper's results were produced under this behavior. But its cost is measurable:
with Molmo and two DepthPro actors on the same GPU, that GPU totals **46.3 GiB / 48 GiB**;
**a 40 GB GPU cannot fit it** (this also confirms empirically the judgment that "4× A100 40GB is not enough").
And vlm was not called even once on these four benchmarks, which means nearly a whole A6000
sits resident at 0% utilization the whole time. This is the first lever for saving GPUs.

### 6.2 Two defaults in `run_eval.sh` make Ray hang silently

```bash
NUM_GPUS="${NUM_GPUS:-8}"      # :50   default 8
EVAL_GPUS="${EVAL_GPUS:-4}"    # :51   default 4 → trainer.n_gpus_per_node
```

The script is written for an 8-GPU node. On a 4-GPU machine without overriding `EVAL_GPUS`, the seven tools' 2.3 plus the 4 verl wants
is 6.3 > 4.0, **Ray does not error, actors queue forever**. You must set `NUM_GPUS=4 EVAL_GPUS=1`.

Even set to 1 it is still tight: **the policy side needs one whole GPU, not a fraction**, which requires Ray to happen to pack the 2.3
into 1.0/1.0/0.3 and leave one whole GPU free. When it cannot get a whole GPU, the symptom is that
`verl/utils/device.py`'s `is_cuda_available = torch.cuda.is_available()`
(**evaluated at module import time**) is False, `get_device_name()` permanently returns `cpu`, and so
`fsdp_workers.py:158` assembles an invalid backend string:

```
ValueError: Duplicate device type cpu in backend string: nccl.
    The custom backend string argument is invalid: cpu:gloo,cpu:nccl.
```

Measured layout (four GPUs, all tools alive):

```
GPU0  20.0 GiB  util 80%    roborefer 18.6 GiB + two small actors + sglang http
GPU1  46.3 GiB  util  0%    Molmo 30.2 + 12.3 + 3.8     ← nearly full, idle the whole time
GPU2   0.7 GiB  util  0%
GPU3  41.6 GiB  util 44%    verl FSDP 15.0 + sglang scheduler 26.7
```

### 6.3 `conda activate` does not work in a subprocess

`run_eval.sh:84` calls `conda activate spacetools-rl`, but `conda` is a shell function
and does not carry across subprocesses:

```
CondaError: Run 'conda init' before 'conda activate'
```

The fix is `BASH_ENV=/opt/conda-st/etc/profile.d/conda.sh`: non-interactive bash sources it automatically,
so the function exists in the subshell. No script change needed.

### 6.4 The benchmark dataset's directory layout does not match the script's mapping

The `BENCHMARKS` mapping in `run_eval.sh` expects a nested layout
(`robospatial_home_multiturn/test.parquet`, `refspatial_bench/location.parquet` …),
while the dataset, **whether `main` or the pinned `1d539ac9...`, is flat** `data/<benchkey>.parquet`.
Upstream changed the layout and the script did not follow; running it as-is always gives:

```
Missing: .../benchmarks/robospatial_home_multiturn/test.parquet
exit 1
```

Also: when `run_eval.sh` does its own `snapshot_download` it **does not pass a revision** (pulls `main`),
which is inconsistent with the revision pinned in `WEIGHTS_PINS.txt`. This run checked that these four files have
byte-for-byte identical sizes on both revisions, so it caused no definition difference, but it is a latent hazard.

### 6.5 The file name `RESTORE.sh` verifies does not match the one in the package

Step 1 of `RESTORE.sh` runs `sha256sum -c SHA256SUMS`, while the package ships it as
`SHA256SUMS.remote`; following the docs it always dies with "package is corrupted".

### 6.6 `28_chain.sh` still `exit 0` on failure

The cross-env chain test writes raw output to `/workspace/logs/chain_raw.log`, but **this directory is not among the required directories listed
in MANIFEST** (only `checkpoints` and `hf` are listed). When the directory does not exist, every subsequent grep reads nothing:

```
28_chain.sh: line 33: /workspace/logs/chain_raw.log: No such file or directory
  Failed links on chain:
  ✗ chain did not pass
```

**Then it `exit 0`s**, so VERIFY's summary shows "exit code 0" for this item and prints "✓ all three passed".
This item **was not tested at all, yet reported as passing**.

The same script has a second trigger: it uses `ray.init(num_cpus=8, num_gpus=1, ...)`,
and when connecting to an **existing** cluster Ray rejects these two arguments outright. The
`/root/tmp/ray/ray_current_cluster` address file left behind by the previous eval is enough to trigger it:

```
Connecting to existing Ray cluster at address: ...:6379
ValueError: When connecting to an existing cluster, num_cpus and num_gpus must not be provided.
```

### 6.7 Python patch versions differ across the five conda environments

```
spacetools-rl               3.11.16   ← Ray cluster head node
spacetools-tool-vlm         3.11.0
spacetools-tool-roborefer   3.11.16
spacetools-tool-bbox        3.11.0
spacetools-tool-graspgen    3.11.16
```

Ray's `check_version_info` compares the full version string by default; `3.11.0 ≠ 3.11.16` raises RuntimeError,
and **all 8 actors** in those two environments fail to join the cluster.

**None of the original three acceptance checks catch this failure**: the cross-env chain test **passes** in the unfixed state;
it only surfaces at eval / RL scale. And `RESTORE.sh`'s environment check **prints this version difference verbatim and still judges it OK**
(it only verifies that `bin/python` is executable).

Impact: the four spatial reasoning keys only use roborefer + one depth/vision_ops call each, so the effect is limited;
keys that depend on the full tool set (`blinkdepth` / `cvb3ddepth` / `boppose` / `bopgrasp`) will be affected;
**in the RL stage these two environments hold 23 actors.**

Fix and verification in `00_environment/eval_rl_env/env_package_revision_20260912.md`; it has been pushed to the HF repo with `POSTRESTORE.sh`,
and `VERIFY.sh` also gained a pre-check item 0 that stops this failure before the eval.

### 6.8 roborefer's deepspeed hard-depends on `CUDA_HOME`

llava's **inference** path unconditionally does `import deepspeed.comm` at module level
(`llava/train/sequence_parallel/globals.py:22`, i.e. environment README hole #17),
and deepspeed reads `CUDA_HOME` at import time. The packaging machine has a system `/usr/local/cuda-12.8`,
which hits torch's third fallback; a machine with only the driver installed does not.

The real eval path goes through Ray's `runtime_env={"conda": ...}`, which activates conda,
so `$CONDA_PREFIX/bin/nvcc` is on PATH and torch can resolve `CUDA_HOME`; **so the eval is not affected**;
but `04_smoke.sh:42` calls the env's python directly without activating conda, and the smoke test fails on roborefer.

---

## 7. Conclusion

**1. The checkpoint is usable and can go to RL.** Both criteria pass, but the two "passes" are not the same thing:

- **RefSpatial has a wide margin.** Criterion 48%, measured 52.58%, 4.58 pp ≈ **13 questions** of margin. Even counting the dozen or so
  closest-to-the-edge questions out of 277 as all wrong it is still above the line, so this item passes on any run.
- **RoboSpatial "passes in expectation", not "passes every time".** Criterion 60% = 210/350, while the possible values of a single run
  can only fall in the interval **199 ~ 228** (199 = the always-correct samples right in both runs, 228 = always-correct + all 29
  flip samples hit), and **210 falls right inside the interval**: this is what "the criterion falls inside the flip band" means.
  The implication: whether a single reading is above or below the line is decided by how those 29 flip samples land this time, not by the checkpoint.
  Reaching 210 needs 11 of the flip samples to land right; the single-run hit rate ≈ 0.5, which works out to **a probability of about 6.8% that one run reads
  below the line**; the expected value of 213.5 questions (61.00%) is above the line, so in expectation it passes.
  The two actual observations are exactly one above and one below: 61.71% (216) and 60.29% (211).

  The practical consequence: **run only once, get unlucky, and you read 59.x% and judge a usable checkpoint
  unusable.** Hence item 2 below requires reporting an interval and deciding by expectation. Rerunning to 5 runs and taking the median would push the misjudgment
  probability down to 0.28%, but that only suppresses reading noise and does not change the 61.00% conclusion (§4.1), so we did not rerun.

**2. Scores must be reported as intervals.** The policy side is not bit-for-bit deterministic: across two runs, 29 of 350 samples flip,
and 199/350 generations differ. The source has been located: **zero non-determinism on the tool side** (zero cases out of 350 samples of "the same query
returning different results"); it all comes from the policy rollout (sglang's continuous batching makes batch composition differ between runs,
the reduction order changes, and argmax flips at near-ties). Vacant is the biggest noise source, because its criterion is
**binary convex hull membership on a continuous prediction**; edge samples flip when pushed across the line by a tiny perturbation.

**3. Nothing suspicious on the run side**: all seven health indicators are zero, and the tool histogram nearly coincides with the official ckpt's behavior.
**So the gap is a real gap, not a run accident.**

**4. The nature of the gap has been located at the sample level**: it is not weak reasoning, it is a **conservative tool-use policy**.
`depth_estimator` fires only 1 time in 350 samples, and that once it was right; on VQA questions that need depth information
(§5.3) it even translates the question into a depth question itself, yet forces an answer from 2D coordinates.

---

## 8. Why RL is still needed

### 8.1 There is a structural gap between the SFT objective and the task objective

SFT learns "imitate the token distribution of demonstration trajectories", while the goal of this task is "answer correctly". The two part ways on
**tool-call decisions**:

- Calling roborefer once and answering, versus calling depth + two vision_ops and then answering, **makes no essential difference in the SFT loss**:
  both are just different token sequences, and whichever appears more in the training set is what gets learned.
- But under the goal of **answering correctly** they are worlds apart: in §5.3 one missing depth call means wrong, in §5.4 calling it means right.
- SFT also has no mechanism to tell the model "your assertion about the tool return is unfounded" (the
  "This point is within the vacant space" line in §5.5). A reward signal does.

The confusion matrix in §5.6 is another cut of the same thing: **the marginal distribution of predictions exactly matches the ground-truth marginal
(165/63), while `gt_no` is exactly a coin flip.** SFT can get a good loss by fitting the prior,
while discriminative ability does not improve. A per-sample reward does not accept this trade.

### 8.2 The headroom has already been quantified; it is not "maybe it can still go up"

| | Current | Official ckpt (after RL) | Paper | Headroom |
|---|--:|--:|--:|---|
| RoboSpatial Overall | 61.00 | 65.43–66.00 | 70.00 | **9 pp** |
| RefSpatial three keys | 52.58 | 53.35 | 53.07 | ~0 |

**All the headroom is in RoboSpatial, and it falls exactly where RL should act.** Compare the chain complexity of the two tiers:

```
RefSpatial    277/277 single chain, 1.00 calls/sample   → SFT already saturated, RL has nothing to push on
RoboSpatial   1.649 calls/sample, main chain covers only 64%      → decision space exists, RL's domain
```

**"Main chain coverage" = the fraction of samples taken by the most frequent chain signature.** RoboSpatial's per-sample call
count distribution is `{1 call: 124, 2 calls: 225, 3 calls: 1}` (§4.3); the most frequent one (`roborefer×2`)
is **225/350 = 64%**, and the remaining 36% take other forms; RefSpatial has 277/277 all taking
`roborefer×1@2t`, 100% coverage.

This number answers "**is there still decision space**":

- 100% coverage = the model does the same thing on every sample; there is no divergence in policy, **all RL can change is how it copies**.
- 64% coverage = the model already makes different choices across samples, so "which chain to take in which situation"
  is a decision that really exists and is currently not made well enough: §5.3 is the wrong side of it (should have called depth, did not),
  §5.4 is the right side (called it, got it right). **This is exactly what a per-sample reward can reweight.**

**Where it ties is exactly where RL is useless, and where it trails is exactly where RL is useful**: this is the strongest piece of evidence for going to RL.

### 8.3 A negative expectation that should be written down in advance

The coin-flip level on `gt_no` **still exists** after the paper's RL (official ckpt 47.62%, this ckpt 49.21%).
**Do not list it among RL's expected gains.** It is more likely a property at the task/data level.

### 8.4 An input to RL reward design

The Vacant reward is a **binary convex hull membership criterion on a continuous prediction**. Of the 67 wrong answers, 16% are within 0.02 of the convex hull boundary
and 36% within 0.05. This means that on this kind of task RL gets a **discontinuous,
low-information** reward: off by 0.001 and off by 0.5 both score 0, and the gradient cannot distinguish "almost right" from "completely wrong".
To make progress on RoboSpatial, reward shaping for this cell deserves separate consideration
(for example, a partial reward decaying with distance for points outside the convex hull).

---

## 9. How far should SFT be trained before RL is worth doing

Several testable gates can be extracted from this run's data. In order of importance:

### Gate 1: the tool-call **format** must already be stable (hard requirement)

```
This run, measured   Every sample has a tool call     627 / 627
           Malformed tool calls         0
           Missing <answer>             0
           Turns exhausted (8-turn limit)     0
```

This is a precondition for RL, not its goal. If SFT has not yet learned the call format solidly, a large fraction of RL rollouts
will get 0 reward straight away due to format errors, the gradient signal is almost all noise, and GRPO-style within-group
advantages will be zeroed out en masse. **If this item is not perfect, keep doing SFT; do not go to RL.**

### Gate 2: at least one **multi-tool chain** must work, even if its trigger rate is very low

index=176 in §5.4 proves this ckpt can run the
`depth_estimator → $depth_map → vision_ops.index_at` chain across three conda environments
and get it right. **The capability exists but the trigger rate is 1/350: this is exactly what RL should amplify.**

Conversely: if no multi-tool chain ever works (not "not called", but "called and wrong"),
that means SFT coverage is insufficient or the tool interface was not learned, and RL will only make a bad policy more confident.

### Gate 3: the target benchmark must **not already be saturated**

RefSpatial is the counterexample: SFT already ties the post-RL official ckpt (2 questions apart out of 277),
and the chain is unchanged on 277/277. **Doing RL on this kind of task wastes compute**: there is no decision space to explore.

The way to judge is what this run used: look at the **distribution of tool calls per sample** and the **main chain coverage**.
A distribution like `{1 call: 100}` means there is no decision; only something like `{1: 124, 2: 225, 3: 1}` has one.

### Gate 4: the failure mode must be **decision-type**, not **capability-type**

This run located the 15–17-question gap at the sample level: the model translated the question correctly, the tool is available, it just does not call it
(§5.3). This is decision-type, and RL's credit assignment targets exactly that.

If the failure mode is "called the tool, got the correct return, still cannot derive the answer", that is insufficient reasoning ability;
keep doing SFT or switch the base model, and RL's gain will be much smaller. **Telling the two apart requires looking at trajectories; if you cannot tell,
do not decide on RL based on scores.**

### A quantitative reference

Where this run stood when it passed the gates:

```
Format health      7 / 7 items zero
Tool-call coverage    100% (627/627)
Multi-tool chain      works, trigger rate 1/350
Target benchmark  61.0% vs post-RL 65.4–66.0, headroom 9 pp, not saturated
Main chain coverage    64% (decision space exists)
Failure mode        decision-type (should call, does not), located at sample level
```

**If all four gates pass and the headroom is ≥ 5 pp, RL is worth doing.** If any one is missing, fill it in with SFT first.

---

## 10. Other things worth recording

**Write criteria as intervals, not point thresholds.** This run's `RoboSpatial ≥ 60` falls right inside the measured
flip band (199 always correct / 228 upper bound), so the criterion formally cannot answer the question: the first reading of 61.71
looks like 6 questions of margin, the second of 60.29 leaves only 1. **Future gates should be written as "expected value ≥ X and
always-correct lower bound ≥ Y", or simply specify "take the median of N runs".**

**The grep for `Error:` must use substring matching.** `TypeError:` contains `Error:`, and substring matching catches it;
switching to prefix matching (`startswith("Error:")`) would miss an entire class of Python built-in exception names.

**Acceptance scripts themselves can give false green lights.** §6.6 is a concrete case: the script prints "✗ chain did not pass" while
doing `exit 0`, and the summary prints "✓ all three passed" based on that. **Whoever consumes the exit code should also check the output for the success
marker**, rather than trusting `$?` alone. After this run `VERIFY.sh` has been changed to do this.

**Before running the eval, confirm the machine has no leftover Ray cluster**, including no leftover
`ray_current_cluster` address file: it is enough to make item 3 of the next acceptance check fail (§6.6).
`ray stop --force` reporting "57/58 stopped" is nothing to worry about; the remaining one is usually a zombie.

**Collect GPU traces during the eval.** This run's per-GPU peak was caught mid-run; checking
`nvidia-smi` after it finishes only shows 0 MiB. Following P4's approach, start 1 Hz sampling to disk; it is also
the only source of the answer to "how Ray packed the actors".

**A100 and A6000 share the same environment package.** The architecture scan confirms 0 self-built extensions missing sm_8x;
A100 (sm_80) / A6000 (sm_86) / L40S (sm_89) are all covered, only H100 (sm_90) is not.
Switching to either of these two GPU types only requires changing `NUM_GPUS` / `EVAL_GPUS` and recomputing the GPU budget; the package does not change.

**The GPU budget for the RL stage must be recomputed for "all tools alive".** At eval scale 4 GPUs are already tight
(tools 2.3 + a whole policy GPU 1.0, and Ray needs to happen to pack it right); the RL actor counts are
roborefer 6, sam2/depth/bbox/graspgen 5 each, vision_ops 8, reserving about 7+ GPUs.
**4 GPUs structurally cannot fit it.**
