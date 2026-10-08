# Replacing GRPO with GFlowRL on SpaceTools' multi-turn tool trajectories

**The full record, from environment setup to per-sample attribution**

> Version v1 · 2026-09-19 · pending review
> 2026-09-25 correction: ① the "C′ gets 3 fewer questions right" on `front/behind` is single-run eval noise (§0, §10.4, §10.6, §11, §12.2);
> ② "C′ makes the policy more proactive, and point override is harmful" has the direction backwards — it is GRPO that suppressed the point override present at the starting point; C′ stays at the starting point (§0, §10.6, §10.7, §11)
> 2026-09-30 addendum: the "129 s accounting gap" in §8.4 has been resolved = old_log_prob 118.8 s + update_weights 10.0 s (§8.3, §8.4)
> Covers: environment build → SFT training → SFT eval gate → derivation and implementation of C′ → GFlowRL training →
> GFlowRL eval → per-sample comparison with P4/P5/P6.
> Sources: `training/` (as-run records for SFT and RL), `eval/GFlowRL/` (dumps, trajectories, gates and analysis scripts for the nine
> benchmarks), `spacetools-repro/` (all P2–P7 reports and the P4 dumps).
> The "Sources" at the end of each section points to the raw files the numbers can be recomputed from.

---

## 0. One-page summary

### What was done

Reproduced the four-step pipeline of SpaceTools (arXiv 2512.04069, CVPR 2026), trained our own SFT checkpoint,
then **replaced its Step 4 reinforcement learning objective, GRPO, with the GFlowRL (arXiv 2607.13394) objective**,
trained a full 1 epoch (85 steps), evaluated on the nine benchmarks in the paper's Table 2, and did a per-sample comparison
against the official checkpoint.

### Why this intersection is worth doing

GFlowRL is a **distribution-matching** objective (samples in proportion to `exp(βr)`); GRPO is a **reward-maximization** objective (pushes mass onto
a single optimal mode). Tool calling naturally has several equally valid chains. P6's offline attribution gave a measured motivation:

> The existing policy's tool orchestration has already collapsed heavily — on the three RefSpatial benchmarks 277/277 use exactly the same chain,
> and in `cvb2drelation` 615 of 650 are identical. **This is exactly what GRPO would do, and exactly what a distribution-matching objective claims to avoid.**

#### What tool-orchestration mode collapse is

When the model solves a question it produces one complete trajectory — **one sample is one rollout** (`rollout.n=5` in training
means 5 rollouts per prompt). Compress the tool calls in one rollout into a string by "**which tools, how many calls each, how many turns**",
and you get its **chain signature** (field name `chain_signature` in the parsed records):

```
roboreferx1@2t                                   one roborefer call, takes 2 turns
roboreferx2@2t                                   two concurrent roborefer calls in one turn
depth_estimatorx1+roboreferx2+vision_opsx2@3t    measure depth first, then locate two objects, then read one depth value for each, 3 turns total
```

> **Chain signature ≠ rollout.** The rollout is the complete trajectory **itself** (including `<think>`, the arguments of each tool call,
> the tools' real returns, and `<answer>`); the chain signature is a **fingerprint** of it that keeps only the "what was called" dimension,
> discarding wording, coordinates and reasoning. **Two rollouts with completely different content can have the same chain signature** —
> precisely because it discards these, it can be used to measure "how uniform the orchestration is".

**Mode collapse = all samples follow the same signature.** This is exactly the term used in the GFlowRL paper:
a reward-maximizing objective "concentrates probability mass on a single high-reward mode, causing mode collapse and loss of solution diversity".
Measured with two numbers:

```
Dominant-chain coverage   share of samples taken by the most frequent signature   100% = full collapse
Signature count           number of distinct signatures across the benchmark      1 = full collapse
```

**Why this relates to the choice of algorithm:** the same question often has **several equally valid** solutions (measure depth then locate /
locate then measure depth; locate with `roborefer` / locate with `vlm`). A reward-maximizing objective (GRPO) pushes probability mass
**onto the best one it has seen**, and the probability of the other equivalent chains is pushed toward 0; a distribution-matching objective claims it
**keeps all of them in proportion to `exp(βr)`**. So "degree of collapse" is exactly where the behavioral difference between these two kinds of objectives should show up —
the GFlowRL paper's own diversity score (GRPO 1.21 / FlowRL 2.64 / GFlowRL 3.93) measures the same thing.

**But only half of this motivation holds up, and that has to be stated up front.** P6's follow-up review (`04_gflowrl_implementation/design_notes/01_route_decision.md` §2) points out:
a RefSpatial question is just "point to where that thing is", **the correct chain is a single `detect_one` call to begin with** —
there is no second equally valid chain to preserve, **that is not collapse, it is a task with only one path**. The same goes for `boppose`'s
60/60 pass-through and the 95.6%/99.8% rule compliance on depth questions.

**There is only one collapse that actually costs something, and P6 pinned it down cleanly:** the 29 `front/behind` questions in `robospatial`
ask about depth order, `depth_estimator` is in that benchmark's tool list, and **it was not called once in 29/29**
(0 times across all 350 samples). This is the only place where "a known valid alternative chain exists, the policy never takes it,
and the cost is quantifiable (+8 samples)" — **this kind of suppressed mode is exactly what distribution matching claims to preserve.**
§10.4 and §10.6 of this report report two things respectively: the change in overall collapse, and whether this one real gap was moved at all.

> **2026-09-25 correction (P0: same machine, same KV pool, each of the three arms evaluated three times): the "gap" here holds; "C′ makes it worse" does not.**
> `front/behind` 29 questions, three-run mean: SFT 17.3 · P4 19.0 · C′ 19.3; no difference among the three arms.
> The 72.4% → 62.1% (3 fewer questions) in §10.4 / §10.6 comes from a single eval and is noise;
> rerunning `analysis/p6_criteria_p4_vs_p7.py` still gives 21 / 18, so the original numbers were computed correctly; a single reading just cannot separate them.
> `depth_estimator` calls over three runs: SFT 0/0/0 · P4 2/2/1 · C′ 0/0/1, not strictly 0.
> In the SFT training data (`siyich/spacetools-sft`) there are 977 RoboSpatial yes/no questions, 0 of which call `depth_estimator` —
> the gap comes from the SFT data distribution. Upper bound by three-run mean is about +10 questions (29 − 19.3).
> Sources: `07_gflowrl_improvement/p0_same_machine_reeval/p0_results.md` §6b · Design Doc P3.

The paper's own limitation section also says: "it remains unclear whether the same estimator-centric design
principle extends to broader **agentic or multimodal RL** settings". A search confirmed there is **no implementation of any
GFlowNet-style objective on multi-turn tool trajectories**.

### Results

```
Training   85 steps · 21 h 57 min · 8× A40 · no crash / no training-side OOM / exit code 0
Eval       nine benchmarks · 2121 samples · zero OOM / zero truncation / zero hitting the turn limit / zero missing <answer>
```

| | SFT starting point | Official ckpt (P4 reproduction) | **C′ step85** | Paper |
|---|--:|--:|--:|--:|
| RoboSpatial Overall | 61.00 ± 0.77 | 65.43–66.00 | **62.14** (mean of two runs) | 70.00 |
| RefSpatial three (weighted) | 53.07 | 53.79 | **53.07** | 53.07 |
| BLINK Relative Depth | not evaluated | 86.29–87.90 | **87.90** | 90.32 |
| CV-Bench 2D / 3D | not evaluated | 94.62 / 96.50 | **94.62 / 96.50** | 94.92 / 96.00 |

**Three main conclusions:**

1. **No benchmark improved.** Per-sample exact McNemar tests on the nine benchmarks give a minimum p of
   0.092, on robospatial. The three that are nominally a bit higher (blinkdepth +2, boppose +1, bopgrasp +2)
   are all non-significant, and for boppose the mean goes the opposite way from the per-sample direction, for bopgrasp the threshold count goes the opposite way from the mean.
   **The opposite direction is equally non-significant** — this measurement does not have the resolution to decide either direction on any single benchmark.

2. **The motivation P6 left for P7 gives a negative reading in practice: distribution matching did not reduce orchestration collapse; it collapsed more.**
   On the four benchmarks with large sample sizes, dominant-chain coverage went up consistently (cvb2d 94.6→98.3%, cvb3d 96.2→99.7%,
   blink 82.3→85.5%, robospatial 62.3→64.9%), and there were fewer distinct chain signatures.

3. **The only measurable behavioral change from C′ is harmful, and it can be accounted for fully.** The same behavior appears once on each of two benchmarks:
   - **RefSpatial three** (pointing questions, 276 parseable samples): verbatim pass-through 276/276 → **268/276**.
   - **RoboSpatial Vacant** (122 questions, also outputs a point, but P6 judged it **not pure pass-through**):
     samples that "modify the point the tool gave" doubled from 21 to **42**, while the point-override accuracy is only 14.3% versus 57.0% for pass-through.

   Both are **the policy modifying the coordinates the tool gave more often**. **The 14-question gap on RoboSpatial can be attributed entirely,
   with nothing left over, to this one behavioral change** (counterfactual check in §10.7).

> **2026-09-25 correction (P0: three re-evals on the same machine, same KV pool, and the first with a dump of the SFT starting point): the direction of this one is reversed —
> it is not that C′ makes the policy more proactive; GRPO suppressed the point override already present at the starting point, and C′ stays at the starting point.**
> - RoboSpatial Vacant point overrides (per run): SFT starting point 49.3 · P4 21.7 · C′ 44.0. RefSpatial pass-through: SFT 272/276 ·
>   P4 276/276 · C′ 272/276 (C′ identical to the starting point sample by sample; this report's 268 is a single reading).
> - By three-run mean the gap is Vacant −8.6 questions (p = 0.0004), VQA +3.7 questions, overall −5.0 questions; the "14 questions" came from single run vs single run.
> - Point overrides mostly happen when the tool's point is already wrong: on point-override samples the tool point hits only 10–19% of the time, and the overridden answer is actually more accurate (19–28%).
>   Forcing pass-through makes Vacant **drop**: SFT 50.7 → 46.3 · P4 62.3 → 60.3 · C′ 53.7 → 49.3.
> - P4's lead comes mainly from tool-call quality: the last point returned by roborefer falls inside the GT for SFT 46.3 · P4 60.3 · C′ 49.3 / 122
>   (three-run mean; P4 vs C′ stable samples 13 : 4, p = 0.049). GRPO learned a better `obj_name`; the extra pass-through mostly follows from that.
> - So "a more proactive policy is worse" does not hold; the accurate statement is "C′ did not learn the tool calling GRPO learned and stayed at the SFT starting point" —
>   a symptom of insufficient training signal (70% degenerate groups, clipped every step), not a harmful property of the C′ objective.
> Sources: `07_gflowrl_improvement/p0_same_machine_reeval/p0_results.md` §6 · `GFlowRL_improve/P1_prep/p1_monitor.py --upper` · Design Doc P1 "Where the primary criterion comes from" (formerly P2).

### Why this negative result is attributable, and not "a bad run"

Three training-side readings narrow the space of explanations a lot:

```
Share of reward-degenerate groups   median 70.3%   (range 57.8–85.9%)
actor/grad_norm                     median 180     while grad_clip = 1.0
Training score                      no reliable upward trend
```

In 70% of groups the within-group rewards are identical; there the reward term of the flow gap is constantly 0 and the gradient comes entirely from the drift term —
**at those times C′ is doing flow matching toward `π_ref·exp(βr)/Z`, not learning the reward**. On top of that every update
is clipped down to just its direction: **an objective that receives almost no reward signal has no reason to make orchestration more diverse.**

**To be precise about "only trained 85 steps":** `total_epochs=1` is hard-coded by the upstream authors, and the paper's 70.0 was trained
exactly that way, **so we cannot say we trained shorter than SpaceTools**. But GFlowRL's own evidence comes from
runs of **30–400+ steps with G=16**, while we are at **85 steps + G=5** (the paper states the variance is O(1/G),
and "especially when the group size is small"). **The accurate statement is: an algorithm validated elsewhere
was put into a regime where it has not been validated, unfavorable on both dimensions, steps and G, at once.**

So what this result falsifies is "C′ can do it in this regime", **not C′'s theoretical properties**.

### The right expectation for P7's success or failure (fixed by P6 in advance)

> **Changing the objective cannot touch the 273/322 (84.7%) tool errors and toolset gaps.**

P6 attributed all 322 wrong answers: tool error 241 (74.8%), toolset gap 32 (9.9%), reasoning error 19 (5.9%).
**Tool error : reasoning error ≈ 12.7 : 1.** P7's scope is part of those 44 (13.7%);
**it should not be expected to move this headroom.**

---

## 1. Background: two papers and the one intersection

### 1.1 SpaceTools / DIRL (what is being reproduced)

VLMs are decent at qualitative visual understanding but lack the **metric-level** spatial reasoning embodied applications need (distance, pose, grasping, occlusion).
SpaceTools' approach is to have the VLM **call vision tools online**, and to use RL twice (DIRL):

```
Teaching phase    single-tool IRL expert -> 2k grounded trajectories
                  + Claude Sonnet 4.5 with the full toolset, keep only correct answers -> 6k trajectories
                  mixed 1:3 into an 8k teaching set -> SFT (learns tool signatures, output format, information flow)
Exploration phase continue IRL from the SFT weights (GRPO + KL), all tools enabled
```

Dialogue format `<think>` / `<tool_call>` / `<answer>`, multi-turn until an answer is produced or the limit is reached.
**Toolshed** is the accompanying systems contribution: a tool-hosting layer on top of Ray, one group of actors per tool, **Python environment
isolation** (resolves dependency conflicts between multiple CV models), and variables (point clouds etc.) passed across environments.

The README splits the paper's two phases into four steps: **Step 1** point-tool RL → **Step 2** teacher data collection →
**Step 3** SFT → **Step 4** full-tool RL. The outputs of Steps 1–2 have been released with the SFT dataset,
so most people **start from Step 3**.

Paper's main results: RoboSpatial Overall **70.0**, Pose 34.37, Grasp-SR 50.0. The most important ablation:
**doing IRL directly on all tasks with all tools → 19.79, nearly unlearnable** — the curriculum is not a nice-to-have.

### 1.2 GFlowRL (the algorithm being transplanted)

Reward maximization of the GRPO/PPO kind pushes probability mass onto a single high-reward mode. GFlowNets offer another route:
**sample in proportion to reward**. But existing GFlowNet-style LLM post-training (FlowRL) has to learn a prompt-conditioned
partition function `Z_φ`, which blows up at real post-training scale.

The paper's root-cause diagnosis is a **learning-timescale mismatch**: the policy is pretrained and only needs a few hundred steps of fine-tuning; `Z_φ` is randomly initialized
and has to learn a complex quantity from scratch, also in only a few hundred steps. Two pieces of evidence: replacing `Z_φ` with random sampling does not hurt performance, it slightly improves it;
in FlowRL's 421 steps, **55 steps have gradient norm ≥ 1e6**.

GFlowRL's contribution is to **delete it** and replace it with an **in-batch Monte Carlo estimate** — GRPO-style training samples G rollouts per
prompt anyway:

```
Eq. 4   Z_t(x) = (1/G) Σ_i [ β·r(x,y_i) + log π_ref(y_i|x) − log π_old(y_i|x) ]
Eq. 6   g_i    = sg[Z_t] + (1/|y_i|)·log( π_old / π_ref ) − β·r
Eq. 7   asymmetric flow-gap clipping, g clipped to [−ε_low, +ε_high], ε_low < ε_high
Eq. 8   L = (1/G) Σ_i w_i · ( g̃_i + (1/|y_i|)·log( π_θ / π_old ) )²
```

`Z_t` enters the residual as a **stop-gradient baseline** and carries no gradient, so the auxiliary network, its optimizer
state and the distributed sync all disappear. Appendix B: with clipping inactive and no length normalization, the self-consistent zero-loss fixed point
satisfies `π_θ ∝ π_ref·exp(βr)`.

Results: 7B math 40.92 (GRPO 32.48, FlowRL 35.63); 14B Codeforces 2048 Elo;
**diversity score GRPO 1.21 / FlowRL 2.64 / GFlowRL 3.93**.

### 1.3 The intersection

| Dimension | SpaceTools / DIRL | GFlowRL |
|---|---|---|
| RL algorithm | GRPO + KL (reward maximization) | GFlowNet TB objective (distribution matching) |
| Core pain point | combinatorial explosion of the multi-tool action space | gradient explosion of the learned `Z_φ` |
| Solution | **curriculum**: narrow first, then wide | **subtraction**: delete the auxiliary network |

**Feasible in engineering terms: both are built on verl** (SpaceTools is a verl fork + Toolshed; GFlowRL is also based on verl),
and SpaceTools has released the full-tool RL dataset (~5500). Swapping the GFlowRL loss into `run_rl.sh`
is an experiment of manageable size.

**Sources:** `docs/paper_notes.md` · `records/P7_PRIOR_ART.md`

---

## 2. The question we want to answer, and the criteria fixed in advance

The existence of this section is itself part of the methodology: **the criteria were written down before seeing any result**, to avoid picking them after the fact.

### 2.1 The question, rewritten twice

The original wording in the handoff doc was "Can GFlowRL train stably on multi-turn tool trajectories?" **That sentence has two problems:**

- **A "yes" comes almost for free.** The meaning of "stable" comes from GFlowRL's comparison against FlowRL, and the root cause the paper itself diagnosed
  is the learning-timescale mismatch of `Z_φ`. **Without `Z_φ`, that failure mode does not exist** — normal gradient norms
  prove nothing.
- **A "no" is not attributable.** `microsoft/gflowrl` is a 404; the loss has to be written ourselves from Eq. 4–8. Then
  is "unstable training" the algorithm not generalizing, or did we write Eq. 8 wrong?

**Rewritten as A′:**

> On SpaceTools' multi-turn tool trajectories, is GFlowRL's in-batch Monte Carlo estimator `Z_t`
> still a usable log-partition estimate?

**The key separation of responsibilities:** "is our loss written correctly" is **guarded separately by a synthetic fixed-point test**
(construct `π_θ = π_ref·exp(βr)/Z` and check loss ≈ 0), not mixed into A′.
Otherwise we are back to a non-attributable negative result.

Later the user stated the goal as "try applying GFlowRL and see whether it improves model accuracy" — **that corresponds to a different route
(C: full comparison from the same starting point)**, costing 3–4 days of machine time. The difference between the two was written down; what was finally run is a simplified version of C
(single arm + comparison against existing baselines), while doing A′'s offline diagnostics first **is exactly the cheapest path to that question**.

### 2.2 Pre-registered criteria

| # | Criterion | What counts as passing |
|---|---|---|
| i | report the **distribution** of `Var(Z_t)/(β·r)²`, not a single number | spread needed on both sides |
| i-b | report several choices of the same within-batch constant side by side | depends on the **shape of the distribution** of the drift |
| ii | run length normalization on/off once each | compare **fixed-point locations**, not scores |
| iii | residuals of degenerate groups | must be compared against "the zero-information control that replaces `π_old` with `π_ref`" |

### 2.3 Explicitly excluded from scope

- **`bopgrasp` / `boppose`.** P6 §4.2: the reward (NCE) is insensitive to gripper orientation — the fallback answer with a median orientation error of **63.9°**
  gets NCE **1.07**, better than the **1.38** of a grasp actually computed by the tool. **Where the reward is mis-specified, any algorithm that targets `r`
  has nothing to say.** And distribution matching is one level worse than reward maximization: GRPO only keeps the single highest bad mode,
  **distribution matching keeps that bad mode alive too, in proportion to `exp(βr)`.**
- **Changing tools.** No second depth model, no stronger pointing model. The reason is scope:
  **P7 tests changing the objective; changing tools is a separate, orthogonal path**, and mixing them makes results non-attributable.

**Sources:** `04_gflowrl_implementation/design_notes/01_route_decision.md` · `records/P7_HANDOFF.md` · `records/P6_REPORT.md`

---

## 3. Environment: why this step took most of the effort

In this project **the environment build took more work than the training itself**. The reason is the system structure: what is evaluated is not one model but
**one model + seven tools**, and the seven tools' dependencies conflict with each other.

### 3.1 The actual shape of the dependency conflicts

```
spacetools-rl              main training env    torch 2.9.1 · verl 0.8.0.dev · sglang 0.5.6 · numpy 1.26.4
spacetools-tool-roborefer  RoboRefer-8B
spacetools-tool-vlm        Molmo-7B-D · SAM2 · DepthPro     torch 2.5.1
spacetools-tool-bbox       3D bbox · vision_ops             numpy 2.4.6
spacetools-tool-graspgen   GraspGen                         torch 2.3.1
```

**torch 2.3.1 / 2.5.1 / 2.9.1 coexist, numpy 1.26 and 2.x coexist.** Toolshed uses Ray's
`runtime_env={"conda": <env>}` to start each tool in its own environment, and variables (ndarray) passed across environments
are serialized by Ray. **This is not complexity that can be removed; it is the paper's systems contribution itself.**

### 3.2 SFT environment (self-built, 4.0 GB portable package)

```
python 3.11 · torch 2.9.1+cu128 · torchvision 0.24.1 · torchaudio 2.9.1
transformers 4.57.1 (pinned a second time and hard-asserted) · flash-attn 2.8.3.post1 (self-compiled, sm_80 only)
deepspeed 0.19.6 · llamafactory 0.9.5.dev0 (editable)
```

Of the five deviations from upstream `setup_envs.sh`, **four fix upstream bugs or only affect the build process**:

| Deviation | Upstream | Nature |
|---|---|---|
| `FLASH_ATTN_CUDA_ARCHS=80` | `80;90;100;120` | cuts compilation to 1/4. **Cost: no PTX fallback, H100/Blackwell need a recompile** |
| `transformers` pinned a second time + assertion | no version constraint | **fix** — otherwise a transitive dependency bumps it to 5.x without any error, producing a π_ref with a different definition from the paper's |
| pin `torchvision` / `torchaudio` | no version constraint | **fix** — otherwise ABI crash, training does not start |
| add `ninja` | not installed | without it flash-attn falls back to serial compilation and `MAX_JOBS` has no effect |
| `MAX_JOBS=12` | adaptive (broken inside containers) | pure concurrency control |

**None of the four changes the mathematical result of training.**

Three container pitfalls to remember:

1. **The container memory limit is the cgroup's 93.1 GiB, not the 503 GB `free` shows.** Setting compile concurrency from `free`
   gets the container killed by the OOM killer. Changed to compute from the cgroup with a hard clamp, plus a
   **compile-time watchdog** watching cgroup `anon` (above 70% it kills by **process tree** — ninja calls `setpgid(0,0)` for every child process,
   so killing only the process group misses the running compile processes).
2. **conda on the network volume is unusably slow** (~200 vs >20,000 files/s). So install on the container disk
   (`/opt/conda-st`) and immediately `tar` it to the persistent volume; restore takes 2 minutes after a container restart.
3. **flash-attn: `pip wheel` onto the persistent volume first, then install** — one successful compile is good forever, and a reinstall goes from ~17 minutes
   to 10 seconds.

The acceptance check is not just "`import` works": `check_arch.sh` scans every `.so` containing CUDA for architecture coverage (0 failures),
and **actually ran** a flash-attn kernel on the GPU once — also confirming in practice that "a cubin compiled for sm_80 runs on
sm_86".

### 3.3 Eval / RL environment (22 GB package, four acceptance checks)

RL and eval share one environment, packaged as `qzpm55555/spacetools-eval-env` (21 GB, 6 shards).
**It must be restored to `/opt/conda-st` + `/opt/spacetools`** — absolute paths are baked into the conda environments,
and changing the path breaks everything.

The four checks in `VERIFY.sh` (all must pass before running):

```
Five-env import gate     all pass
Architecture scan        4619 .so · 108 contain cubin · self-built extensions missing sm_8x: 0
Seven-tool smoke test    7/7, weights really loaded and real results produced
Ray cross-env chain      5 tools · failed links 0 · ndarray passed correctly in both directions
```

The cross-env chain check tests the real shape: `sam2` (numpy 1.26.4) produces `$segmentation_mask` →
`bounding_box` (numpy 2.4.6) consumes it; `depth_estimator` produces `$depth_map` → `vision_ops` consumes it.

**The three numbers (4619 / 108 / 0) match the baseline from packaging time exactly**, showing that this environment is reproducible on a new machine.

### 3.4 Three environment-side lessons

- **A delivered machine may have no driver installed at all.** On one machine `lspci` showed 4 GA102GL, but `/dev/nvidia*`
  did not exist and `dpkg -l | grep nvidia` showed 0 packages. **So the first check should be `nvidia-smi -L`
  rather than `nvidia-smi`** — when the latter is missing it reports `command not found`, which is easy to mistake for a PATH problem.
- **The Python patch version must be the same across all five conda environments.** In practice `spacetools-tool-vlm` and
  `-tool-bbox` were 3.11.0 and the rest 3.11.16; Ray's `check_version_info` compares the full version string by default,
  `3.11.0 ≠ 3.11.16` raises RuntimeError, and **all 8 actors in those two environments fail to join the cluster**.
  None of the original three acceptance checks catches this failure (the cross-env chain **passes** in the unfixed state);
  it only shows up at eval / RL scale — in the RL phase there are 23 actors in these two environments.
  The fix was pushed to HF with `POSTRESTORE.sh`, and `VERIFY.sh` got a check 0 as a precondition.
- **The acceptance script itself can give a false green.** When the chain test failed, `28_chain.sh` printed "✗ chain did not go through" while
  doing `exit 0`, and the summary then printed "✓ all three passed" — **this check was not tested at all, yet reported as passing.**
  **Whoever consumes an exit code must also check the output for a success marker.**

**Sources:** `training/SFT/env/2x_A6000_REPORT.md` · `00_environment/eval_rl_env/BUILD_GUIDE.md` ·
`00_environment/eval_rl_env/env_package_revision_20260912.md`

---

## 4. Step 3: training the SFT checkpoint

### 4.1 Why we had to train it ourselves

Upstream only released the checkpoint **after RL** (`siyich/spacetools-ckpt`; checked: only a `main`
branch, no tags, README labeled `reinforcement-learning`). **There is no SFT checkpoint.**
And the SFT ckpt is the **common starting point π_ref** for the two-arm RL comparison — without it, no RL result can be attributed.

**Good news: the SFT data is public** (`siyich/spacetools-sft`, 7463 files / 6.32 GB),
code under Apache 2.0. **The bottleneck for rerunning Step 3 is compute, not materials.**

### 4.2 Three layers of configuration sources

```
① upstream run_sft.sh (the vast majority)   sft_config.yaml is not a file in the repo; it is generated fresh on every run
                               by a heredoc in the script. finetuning_type / freeze_* /
                               cutoff_len / lr / streaming / save_steps are all hard-coded there
② paper Table 6 (covers only seven or eight numbers) Batch 8 · lr · Epoch · Warmup 0.1 · cosine ·
                               Max Prompt/Response 8192 · #GPU 8. Nothing else is stated
③ deviations we added (five)   see below
```

| Change | From → to | Nature |
|---|---|---|
| `per_device` / `ga` | hard-coded 1/1 → derived from GPU count | **fix** — the original values silently become global batch 4 on 4 GPUs |
| `deepspeed` | z3 → z2 | performance, A6000 has no NVLink |
| `save_only_model` | false → true | disk; the cost is **no resuming training** |
| `eval_steps` | 5 → 500 | 3000 steps do not need 600 evals |
| `use_reentrant_gc` | unset → false | compatibility |

**None of the four changes the mathematical result of training; the first one corrects something that was already wrong.**

> **What `per_device` / `ga` are.** Both are parameters of the HF Trainer (which is what LLaMA-Factory uses),
> full names `per_device_train_batch_size` and `gradient_accumulation_steps`:
>
> - **`per_device`** — **how many samples go into one forward pass on one GPU**. "device" means GPU, not machine.
>   It only determines GPU memory usage and speed, **not how many samples the optimizer sees**.
> - **`ga`** — how many forward passes to accumulate before one weight update. During accumulation there is no `optimizer.step()`, gradients are only summed.
>
> The three are related by an identity:
>
> ```
> global batch = per_device × ga × GPU count
>                ↑            ↑     ↑
>          per GPU per pass  accum  GPUs
> ```
>
> **Only the "global batch" affects the training result**; the first three numbers can be split any way. So the paper's `per_device=1`
> comes from dividing by 8 GPUs (1 × 1 × 8 = 8), not a design choice — carried over unchanged to a 4-GPU machine it becomes
> 1 × 1 × 4 = **4**, silently halving the global batch. This is what the first row of the table above fixes.
>
> What we finally ran is **2 × 1 × 4 = 8** ✓.

**z3 → z2 is not an optimization; it is a trade-off better suited to PCIe.** A common misconception is that ZeRO-3 communicates faster; it is the opposite:
z2 is 2Ψ per step, z3 is 3Ψ (1.5×), and z3 splits it into **one small communication per layer**, which is latency-sensitive — A6000s talk
over PCIe (no NVLink), where this hurts most. **What ZeRO-3 buys is GPU memory; it is a tool for "when it doesn't fit".**
The paper doesn't say which one it used; the code has z3, most likely just copied from the LLaMA-Factory default example (they were on 8× A100-80 +
NVSwitch, where z2/z3 make no difference).

**The global batch is fixed at 8**, consistent across two independent sources (paper Table 6 + upstream `run_sft.sh` default
1×1×8). It is **pinned by the goal of "reproduction", not by a technical limit** — changing the batch amounts to a different set of
hyperparameters, and this ckpt is the common starting point for all later comparisons.

### 4.3 GPU memory accounting: why `per_device` is capped at 2

```
GPU memory = fixed part (bf16 params + grads + optimizer state, already sharded by ZeRO-2) + per-sample part
The real culprit is logits: per_device × sequence length × vocab × 2 bytes
Qwen2.5-VL vocab 151,936 -> 0.3 MB per token per sample
```

| per_device | logits spike |
|--:|--:|
| 1 | ~2 GiB |
| 2 | ~4 GiB |
| **4** | **~8 GiB** ← this is what actually blew up in practice |

`per_device=4` measured on A6000: **steady state 39.8 GiB + spike 8.4 GiB ≈ 48 GiB**, exactly the card's capacity.
**It is a transient spike, not steady state**; `nvidia-smi` sampling once per second will very likely miss it, but it is what triggers the OOM.

> ⚠ The sequence length of ~6900 tokens was **back-solved from the OOM error**, relies on the assumption that "the failed allocation is the logits",
> and it is the **longest** batch in those 30 steps, not a typical value. Fine for computing OOM margin; do not cite it as a typical length.

### 4.4 as-run (4× A6000, 3000 steps)

```
Machine     Vast.ai · 4× RTX A6000 (49140 MiB · sm_86) · driver 570.181 · 377 GiB RAM
Launch      GPU 4 · per_device=2 · ga=1 · global batch 8 ✓ (back-derived independently from HF Trainer output)
Data        7907 raw -> filter out 887 robot tool samples -> 7020 go into training
Time        27929.7 s = 7 h 45 min · mean 9.31 s/it
loss        train 0.8621 (step 5) -> 0.0625 (step 3000) · whole-run mean 0.19167
eval_loss   0.1680 / 0.1064 / 0.0975 / 0.0684 / 0.0559 / 0.0452   six points monotonically decreasing, not one goes back up
```

All six data-definition checks pass: `<tools>` block length 8595 ✓ · sha256 prefix matches ✓ · tool count 11 ✓ ·
system prompt identical across all 7020 ✓ · robot samples remaining 0 ✓.

**One close call: the GPU memory margin is not uniform.**

| GPU | Peak | Share | Margin |
|---|--:|--:|--:|
| 0 | 36830 MiB | 75% | 12.0 GiB |
| 1 | 37090 MiB | 76% | 11.8 GiB |
| **2** | **47730 MiB** | **97%** | **1.4 GiB** ⚠ |
| 3 | 36950 MiB | 75% | 11.9 GiB |

Earlier docs assumed ~10 GiB of uniform margin on 4 GPUs; measured, the worst rank had only 1.4 GiB left.
**The judgment "4 GPUs is very comfortable" was wrong.**

Also, **the 5–6 h estimate was wrong; the real time was 7.75 h** — the cause has been found but is not actionable.
`val_size` is only 20 samples, so single `eval_loss` points are noisy; only look at the trend.

### 4.5 Speedups we did not use (recorded to avoid repeating the discussion)

| Option | What it can do | Why it was not enabled |
|---|---|---|
| `enable_liger_kernel` | **fused cross-entropy, the logits tensor is never materialized** — exactly kills that 8 GiB spike | fused kernels change numerics, so π_ref would carry a deviation |
| `packing` / `neat_packing` | packs short samples into one, recovers the waste of `cutoff_len=8192` | changes attention-mask semantics, **definitely changes the mathematical result** |
| turn off gradient checkpointing | 20–30% faster | the margin can't absorb it |

**Liger is the biggest loss** — it targets exactly this project's bottleneck. But this ckpt is the common starting point for later comparisons;
**introducing a numerical deviation to run 20% faster is not worth it.**

> **The honest conclusion: we are running the same thing on fewer, weaker GPUs, squeezing it in with ga and z2, not making it faster through optimization.**
> 4 GPUs 7.75 h vs the paper's 8 GPUs 3–4 h; that ratio is basically the hardware gap itself.

**Sources:** `training/SFT/env/training_report/REPORT.md` · `docs/sft_training_notes.md` ·
`training/SFT/model_rl_start/` (the ckpt itself)

---

## 5. SFT eval: the gate that decides whether to go on to RL

### 5.1 Why evaluate before training

Step 4 takes 3–4 days of machine time. **If the SFT ckpt itself is not good enough, those days are entirely wasted.**
So first spend 33 minutes evaluating four keys, and decide by a go/no-go rule set in advance:

| Result | Conclusion |
|---|---|
| RoboSpatial ≥ 60 **and** RefSpatial ≥ 48 | checkpoint usable, go to RL |
| RoboSpatial < 55 **or** RefSpatial < 40 | stop and investigate, do not go to RL |
| in between | first confirm the run is healthy, then align item by item with the paper's table |

Why these four keys: they cover the spatial-reasoning abilities in the paper's main table, and none of them needs `grasp_generator`.
**`boppose` / `bopgrasp` are explicitly excluded** — the former has a complicated metric mapping, the latter would pull `grasp_generator`
in and add variables.

### 5.2 The reading order must not be reversed

> **First count OOMs → then check for silent tool errors → only then look at the score.**

- **OOM samples count toward the denominator**, diluting the score into a number that is "on the low side but looks plausible".
- **Tools do not raise exceptions on error**; they wrap the error message in a normal `ToolResult` and return it; the router also swallows exceptions
  inside actors. It looks like this:

```
CALL_OK   depth_estimator -> ModuleNotFoundError: No module named 'numpy._core.numeric'
                             return type: ToolResult
```

  **Note that it is `CALL_OK`. The caller gets no exception at all.** If a tool returns error
  text throughout an entire eval, the model keeps reasoning on garbage input, the eval runs to completion, reports no error, and gives a normal-looking score —
  **and then you go and suspect the checkpoint.**

> An engineering detail along the way: that grep uses **substring** matching (`Error:`), because `TypeError:` contains
> `Error:`; switching to a more "rigorous" prefix match would miss a whole class of Python built-in exceptions.

### 5.3 Results

627 samples, 32 min 41 s.

| | n | run 1 | run 2 | always correct | always wrong | flip | expected ± sd |
|---|--:|--:|--:|--:|--:|--:|--:|
| RoboSpatial · Overall | 350 | 61.71% (216) | 60.29% (211) | 199 | 122 | 29 | **61.00% ± 0.77 pp** |
| RoboSpatial · VQA | 228 | 70.61% | 71.93% | 152 | 55 | 21 | 71.27% |
| RoboSpatial · Vacant | 122 | 45.08% | 38.52% | 47 | 67 | 8 | 41.80% |
| RefSpatial · three | 277 | 52.58 simple / 53.07 weighted | — | | | | |

All seven run-health counts are zero: OOM 0 · sample-level tool error responses 0/627 · every sample has a tool call 627/627 ·
missing `<answer>` 0 · turns exhausted 0 · truncated tool responses 0 · malformed tool call 0.

**RoboSpatial must be reported as an interval.** The 60% criterion = 210/350 **falls right inside the flip band** (199 always correct ~ 228 upper bound):
whether a single reading lands above or below the line is decided by how those 29 flip samples happen to fall this time, **not by the checkpoint**.
Reaching 210 needs 11 of the flips to land correct; with a per-run correct rate ≈ 0.5, this works out to **a probability of about 6.8% that one run reads below the line**.

> **Practical consequence: run it only once with bad luck and you read 59.x%, judging a usable checkpoint unusable.**
> Later gates should be written as "expected value ≥ X and always-correct lower bound ≥ Y", or simply specify "take the median of N runs".

### 5.4 More valuable than the score: the failure modes

| Table 2 row | n | Paper | Official ckpt (P4 reproduction) | **This SFT ckpt** | Questions short |
|---|--:|--:|--:|--:|--:|
| RoboSpatial · VQA | 228 | 79.38 | 73.25–73.68 | 71.27 | 4–6 questions |
| RoboSpatial · Vacant | 122 | 52.46 | 50.82–51.64 | 41.80 | 11–12 questions |
| RoboSpatial · Overall | 350 | 70.00 | 65.43–66.00 | 61.00 | 15–17 questions |
| RefSpatial · three | 277 | 53.07 | 53.35 / 53.79 | 52.58 / 53.07 | **2 questions** |

**Converting to question counts is necessary** — n differs by an order of magnitude across benchmarks, so pp are not directly comparable.
RefSpatial's −0.77 pp is **2 questions out of 277** (standard error about 3.0 pp, the gap is 0.26 standard errors);
RoboSpatial's −4 pp is **14 questions out of 350**, that is a real gap.

**Only three tools were actually called:**

```
roborefer          627 / 627 samples
depth_estimator      1 time
vision_ops           1 time
sam2 / vlm / bounding_box / grasp_generator    0 times
```

**Key counterexample (index=128, GT `No`, answered wrong):** asks "can the cup be placed in front of the speaker"

```
roborefer.detect_one "cup"     -> [(0.78, 0.494)]
roborefer.detect_one "speaker" -> [(0.154, 0.491)]
<think> The question asks if the cup can fit "in front of" the speaker.
        This means: is there space between the speaker and the camera/viewer?
        ... There appears to be adequate desk space in front of the speaker ... </think>
Answer: Yes        Ground truth: No
```

**It correctly translated the question into a depth question by itself** ("is there space between the speaker and the camera"),
**and then answered it with two 2D coordinates whose y values are almost identical** (0.494 vs 0.491).
`depth_estimator` and `vision_ops` are right there in its schema, and together they can answer exactly this question. It did not call them.

**The only sample that called the full chain (index=176) got it right:**

```
depth_estimator.estimate_depth_with_pointcloud   (tool-vlm, numpy 1.26.4)
roborefer.detect_one "bed" / "table"
vision_ops.index_at $depth_map @ the two points      (tool-bbox, numpy 2.4.6)
   -> 2.74 m vs 2.94 m, compares real meters, answered correctly
```

> **Put these two samples together and you have the most important finding of the SFT eval: the ability is there, the trigger rate is 1/350.**

**VQA confusion matrix: learned the prior, did not learn to discriminate.**

```
                 model says no   model says yes   |  share correct in this row
  GT no               31           32      |   49.21%   (63 questions total)
  GT yes              32          133      |   80.61%   (165 questions total)
  ────────────────────────────────────────
  Model answer distribution    no 63       yes 165
  GT distribution              no 63       yes 165     <- exactly the same as the row above
```

**The predicted marginal distribution is exactly the same as the ground truth (165/63), but `gt_no` is exactly a coin flip (49.21%).**
The model learned the prior of "how often to say yes", not the discriminative signal of "when to say no" —
SFT's likelihood objective can get a decent loss just by matching the marginal distribution.

> **A negative expectation written down in advance:** the coin-flip level on `gt_no` **still exists** after the paper's RL
> (official ckpt 47.62%). **Do not list it among the expected gains from RL.**

### 5.5 The four gates and the conclusion

| Gate | Where this run stands |
|---|---|
| ① The **format** of tool calls must already be stable (hard) | §5.3 all seven run-health counts are 0 — malformed tool call 0, missing `<answer>` 0, truncated tool responses 0, turns exhausted 0 |
| ② At least one **multi-tool chain** is usable, even if the trigger rate is very low | usable, trigger rate 1/350 |
| ③ The target benchmark must **not already be saturated** | not saturated — RoboSpatial Overall 61.00, 9 pp of room to the paper's 70.00 |
| ④ The failure mode must be **decision-type**, not **capability-type** | decision-type (should have called, didn't), located down to samples |

**All four gates pass and the room is ≥ 5 pp, so RL is worth doing.** Both criteria pass; go to Step 4.

And **RefSpatial is the counterexample**: SFT already ties the official post-RL ckpt (2 questions apart out of 277),
chain 277/277 unchanged. **Doing RL on this kind of task wastes compute** — there is no decision space to explore.

**Sources:** `03_sft_eval/sft_eval_report.md` · `03_sft_eval/sft_eval_results.md`

---

## 6. C′: which GFlowRL variant, and why

**The paper's configuration cannot be copied as is.** Five mismatches, each measured separately.

### 6.1 Five mismatches

**① `G = 16` → ours is `rollout.n = 5`.** Remark B.2 states the estimator variance is **O(1/G)**;
the first sentence of the limitations in Appendix A is "in-batch MC estimate of log Z can have higher variance
in principle, **especially when the group size is small**".
**16 → 5 is 3.2× the variance; we sit exactly on the least favorable side.** (Changing `rollout.n` would make both arms incomparable,
so it stays at 5.)

**② The length normalization in Eq. 4 and Eq. 5/6 is asymmetric — this is how the paper has it, not a typo.**

```
Eq. 4   Z_t = (1/G) Σ_i ( β·r + log π_ref − log π_old )          no 1/|y|
Eq. 6   g_i = sg[Z_t] + (1/|y_i|)·log( π_old / π_ref ) − β·r      has 1/|y|
```

Eq. 10 in Appendix B likewise has no normalization, which confirms it is not a typesetting issue; **Prop. B.1 is proved under the assumption of "no length
normalization"**. Remark B.4 spells out the consequence of normalization separately:

```
π_θ*(y|x) = π_ref(y|x) · exp( |y| · ( β·r − Z_t ) )
```

**The effective inverse temperature is `|y|·β`, not `β`.** When lengths differ, `exp(−|y|·Z_t)` is no longer a common
normalization constant across sequences, and the fixed point is distorted by length.

**The synthetic experiment quantified the practical consequence of this asymmetry:** within a group the drift becomes a **common shift**; once it exceeds the clip half-width
it clips the whole group's `g̃` to the same value — **the reward contribution is erased completely**, and the threshold is only **≈5e-4 nat/token**.

**③ `|y_i|` must be `response_mask.sum()`, not the raw response length.**
The response of a multi-turn trajectory contains `<tool_response>`, capped at 2048 per call, and one sample can have 4–5 calls.
**If you normalize by raw length, the effective temperature `L·β` is set by how verbose the tool output is.** Measured factors:
`robospatial` **1.14×** · `blinkdepth` **1.41×** · `boppose` **2.40×** —
**and it differs systematically by benchmark**, which amounts to applying a different temperature to each task.

> A subtler layer: in verl the name `response_mask` has **two opposite meanings** — the agent loop gives
> "policy tokens", while the fallback `compute_response_mask()` at `ray_trainer.py:157` is
> `attention_mask[:,-L:]`, where **tool tokens are all 1**, and the fallback is **silent**. A runtime guard is required.

**④ Within-group reward degeneracy.** `p6/passk` happens to be real rollout groups with G=5:

```
fraction of prompts whose reward is identical across all G=5
  robospatial   199/350 = 56.9%
  blinkdepth     94/124 = 75.8%
  boppose        32/60  = 53.3%
```

For **GRPO**: the advantage is 0 after normalization, **the whole prompt produces no gradient**.
For **GFlowRL**: Eq. 8 is a per-trajectory squared residual; even when `r` is all equal the residual is generally nonzero, **so it still produces gradient**.

> ⚠ **But this is not a property of "GFlowRL's published configuration"**: the rollout settings in the paper's Table 9 say
> `Filter groups: Accuracy-based` — **groups with identical rewards are dropped entirely**, so under their recipe
> the problem never shows up. For us it is a dilemma: copying the filter would drop 53%–76% of prompts at G=5;
> not filtering is **a deliberate deviation from the paper's recipe that we must justify ourselves**.
> **This item is downgraded from "one of our advantages" to "a design decision we must make and must justify".**

**⑤ At the starting point the flow gap takes only three values.** On the two binary benchmarks the non-degenerate rollouts are **100% saturated** —
i.e. g only lands in `{−ε_low, 0, +ε_high}`. Avoiding saturation requires `β ≲ 1.4`.

### 6.2 Three variants and the gate

Eq. 4 and Eq. 5/6 can each be length-normalized or not:

| Variant | Eq.4 | Eq.6 | Gate conclusion on real data |
|---|---|---|---|
| `paper` | not normalized | normalized | The drift term degenerates into a within-group common shift and flattens the reward on **30–49%** of groups |
| `normalized` | normalized | normalized | Has a fixed point, but that fixed point is a **point mass** at \|y\|≈334 (pure reward maximization) |
| **`cprime`** | **not normalized** | **not normalized** | **The only one that both has the analytic fixed point of Prop. B.1 and has a 0 rate of "the whole group clipped to one value"** |

**The training arm is set to C′.** Gate readings:

```
Length-dominance cost is mild    the longest 10% take 20.7–24.2% of the loss (even share 10%), only 1.04–1.32× within a group
                                 — because our |y| median is only 305–423 tokens, not the thousands the paper worries about
Reward not swamped by drift      on non-degenerate groups reward dominates drift 24× (drift/reward = 0.041–0.042)
Discriminates on degenerate groups   whole group same value 0.0%, versus 30–49% for the paper variant
Cost                             C′ has the highest clip saturation rate (77.9%)
                                 — trading "more values clipped to ±ε" for "no group has its reward erased"
Knock-on                         under C′, β no lower than 2 (drift/reward ∝ 1/β², at β=1 drift dominates 2.7×); start at 8
```

> **The gate's job is to exclude, not to approve.** It ruled out "all three configurations fail"; it did not answer whether it is worth running.

**The fixed point has been verified numerically with `tools/p7/p7_fixedpoint_check.py` to 7.39e-13.**
This step removed "is our loss written correctly" from P7's list of questions.

### 6.3 β decomposition — the most important analysis tool in the whole report

`Z_t` is the within-group **arithmetic** mean, so Eq. 6 splits exactly into two additive halves, and g is linear in β:

```
g_i = [mean_group(β·r) − β·r_i]  +  [mean_group(d) − d_i]
       └── reward term, ∝ β ──┘     └── drift term, independent of β ──┘
```

Two direct consequences, which every conclusion must carry:

1. **On reward-degenerate groups the reward term is identically 0 and g comes entirely from the drift term.** There C′ is doing flow matching toward
   `π_ref·exp(βr)/Z`, **not learning the reward**.
   **It must not be described as "GRPO gives 0 gradient while C′ is still learning the reward".**
2. **Lowering β does not cure clip saturation.** It only shrinks the reward half; the drift term does not move at all, so the surviving gradient is
   **even more** dominated by drift. The diagnostics block computes `sat_at_beta_{2,4,8,16}` every step as counter-evidence.
   **Choose β by `rew/drift`, not by `sat`.**

**Sources:** `records/P7_ROUTE_C_GATE.md` · `records/P7_STEP23_RESULTS.md` ·
`records/P7_CRITERION_IB.md` · `records/P7_STEP4_RESULTS.md` ·
`training/GFlowRL/docs/`

---

## 7. Implementation: 6 files changed, +366/−8 lines

Branch `repro-4xa6000`, HEAD `c6fef78`, **nine commits**. The full diff is in
`training/GFlowRL/diff/` (three patches).

### 7.1 GFlowRL itself (2 commits)

**Split principle: the paper's Eq. 4/6/7 are gradient-free** (they depend only on `π_old`, `π_ref`, `r`, all frozen during the actor
update), so they are computed once per step on the **driver**; the gradient-carrying part of Eq. 8 is registered as `loss_mode='gflowrl'`.

```python
# ray_trainer.py — the gradient-free half
def compute_gflowrl_flow_gap(data, beta=8.0, eps_low=0.2, eps_high=0.28,
                             variant="cprime"):
    d = masked_sum(ref_lp - old_lp, response_mask, axis=-1).float()
    if   variant == "cprime":     d_in_z, d_in_g = d,     d          # neither side normalized
    elif variant == "paper":      d_in_z, d_in_g = d,     d / L      # as in the paper
    elif variant == "normalized": d_in_z, d_in_g = d / L, d / L
    # Eq. 4: in-batch MC estimate grouped by uid, one value per group
    ...

# core_algos.py — the gradient-carrying half
log_ratio = verl_F.masked_sum(log_prob - old_log_prob, response_mask, axis=-1)
delta = g_tilde + log_ratio
```

> **`masked_sum` (rather than `masked_mean`) is the only difference between C′ and `normalized`, and changing it raises no error.**
> So this line is load-bearing; if it is touched, the fixedpoint check must be rerun. The code comment states this explicitly.

**One constraint to remember: `kl_loss_coef=0` but `use_kl_loss=True`; both are required.**

- `use_kl_loss=False` makes `dp_actor` not compute `ref_log_prob` at all, and GFlowRL's
  `d = Σ(log π_ref − log π_old)` needs exactly that → it cannot run
- `kl_loss_coef≠0` **double-counts** the reference term, because d itself is the KL-type quantity in Eq.4/6
  → no error, but it trains a different objective

**Both of these errors are silent**, so guards that raise exceptions were added.

> **A log-reading trap caused by reuse:** GFlowRL has no such thing as an advantage, but the registered loss's signature
> has no other per-sequence slot, so g̃ is broadcast into `batch["advantages"]`.
> **Consequence: the `critic/advantages/*` printed by the trainer reports statistics of g̃, not GRPO advantages.**

**Discipline for the two-arm comparison:** `run_rl_gflowrl.sh` is a thin wrapper with only a 7-line overrides array,
relying on the existing `"$@"` pass-through in `run_rl.sh`. **Copying 400 lines will inevitably drift** — a two-arm comparison only means something
when "everything except the objective is identical".

### 7.2 Fixing upstream bugs (5 commits)

None of these are GFlowRL-related; they are **what anyone running the upstream code on a small single-node machine will inevitably hit**:

| Symptom | Root cause |
|---|---|
| Ray **deadlocks on a single node, with no error** | `GPUS_PER_NODE` was given to both Toolshed's placement group and the trainer, **reserving the same GPUs twice**. The paper uses 2 nodes, with the PG on node1 and the trainer on node2, **so this path is never exposed** |
| `PolicyLossConfig.__init__() got an unexpected keyword argument 'gflowrl'`, thrown in a Ray worker **several minutes after** Toolshed comes up | The structured dataclass **errors instead of ignoring** undeclared keys. Also fixed the quieter half: the driver reads beta/eps from a bare DictConfig, while `core_algos` reads `eps_is` from the dataclass — the dataclass has no such field, so `GF_EPS_IS` was **silently dropped** |
| `assert _demand <= tool_gpus` refuses to start outright | Upstream `num_actors` is written for 2 nodes (7.8 logical GPUs); without scaling it will not start on a small machine |
| `FileNotFoundError: 'images/xxx.png'` **20 minutes after** loading | The parquet holds relative paths resolved against the cwd; `ray start` launches the raylet with the caller's cwd, and `cd "$VERL_DIR"` comes after it |
| A successful run reports failure; **and `ray stop --force` was never executed** | `cleanup()` kills background jobs and then calls `wait`; `wait` returns 128+SIGTERM, and `set -e` is also in effect inside the EXIT trap → cleanup dies on the `wait` line and the `ray stop` after it never runs. **Every run left a live Ray cluster behind** |

> **Corollary: any "wrong exit code but looks harmless" signal must be traced to its root cause before deciding whether to ignore it.**
> We only realized it when we found the cluster still alive 2 hours 44 minutes after a successful smoke test ended.

### 7.3 Adding observability (2 commits)

**Instrumenting the group degeneracy rate is necessary, not a nice-to-have:** the reward degeneracy rate is the **entire reason** C′ might beat GRPO,
and the 56.9% / 75.8% cited earlier come from **sampling on the eval benchmarks, not from the Step 4 training set**.
The call site sits after `compute_advantage` and before the gflowrl branch, **so both arms record it** —
a number from only one arm cannot prove a difference between both arms. Degeneracy is judged by "within-group range strictly 0" rather than variance,
because pointing / IoU rewards are continuous.

Also added: diagnostics instrumentation for the β decomposition (§6.3) and two IS-weight guards.

**Sources:** `training/GFlowRL/diff/*.patch` · `training/GFlowRL/code/`

---

## 8. Training GFlowRL

### 8.1 Machine and GPU split

```
8× NVIDIA A40 48 GB (GA102 · sm_86) · driver 580.159.04 · 96 cores / 503 GB
container disk 120 GB (five conda environments in /opt) · network volume 250 GB (weights, data, checkpoints)
topology   GPU0-3 on NUMA0 (PXB interconnect), GPU4-7 on NUMA1, SYS between the two groups

GPU 0-3   Toolshed tool actors      TOOL_GPUS=4
GPU 4-7   training (FSDP + sglang)     TRAIN_GPUS=4
```

**This split is not what upstream does** — splitting into `TOOL_GPUS` / `TRAIN_GPUS` is to get around the
"same GPUs reserved twice" deadlock from §7.2.

In upstream `TOOL_CONFIGS`, each tool's `num_actors` and `num_gpus` **were all set for the paper's 2 nodes × 8× A100-80GB**:
of the 16 GPUs, **the tools get an entire 8-GPU node to themselves** (logical demand 7.8 ≤ 8), and training gets the other node.
We have **a single node with 8× A40**; tools and training must share these 8, so only 4 can go to the tools. Hence the auto-scaling:

```
scaled tool actors by 0.500 to fit tool_gpus=4.0:
  {'roborefer': 3, 'vlm': 1, 'sam2': 2, 'depth_estimator': 2,
   'bounding_box': 2, 'vision_ops': 4, 'grasp_generator': 2}
  -> demand 3.70 logical GPUs
```

**How it scales — three steps, triggered only when `_demand > tool_gpus`.** On the paper's 2-node path the tools get an entire
8-GPU node (7.8 ≤ 8), the whole block is skipped, and **every number stays exactly as in the paper**:

```
① First raise vlm's share to 0.7   Molmo actually loads in fp32 (vlm.py:116 hardcodes dtype='auto'),
                            measured single-instance peak 32.11 GiB.
                            Upstream's 0.6 was set for A100-80GB: 0.6 × 80 = 48 GB, fits;
                            on a 46 GB A40, 0.6 is only 27.6 GB — **the same fraction no longer holds on a different GPU**.
                            32.11 / 46 = 0.70, hence raised to 0.7.
                            Left at 0.6, Ray could legally place vlm + 2 depth_estimators
                            on the same GPU (logical 1.0, physical 47.9 GiB) -> the last one OOMs.
                            Demand goes 7.8 -> 8.0 as a result
② Scale actor counts proportionally   _scale = tool_gpus / _demand = 4.0 / 8.0 = 0.500
                            num_actors = max(1, int(num_actors × 0.500))
                            rounded down; `max(1, ...)` guarantees **every tool keeps at least 1 actor**,
                            otherwise that tool would not start a single process and every model call to it would time out
③ Recompute, then assert    new demand = Σ(tool's actor count × each actor's num_gpus)
```

**① does not save GPUs; it asks for 0.2 more — it is a correction; what actually squeezes demand into 4 GPUs is ②.**
(That the two steps happen to give the round ratio 8.0 / 4.0 = 0.500 is a coincidence, not a design.)

**One actor = one process replica Toolshed starts for a given tool** (multiple actors of the same tool serve different requests in parallel);
`num_gpus` is the logical GPU share **each actor** requests from Ray. Item by item:

| Tool | Upstream actor count | ×0.500 rounded down | `num_gpus` per actor | Subtotal |
|---|--:|--:|--:|--:|
| `roborefer` | 6 | 3 | 0.6 | 1.8 |
| `vlm` | 2 | 1 | **0.7** (raised in step ①) | 0.7 |
| `sam2` | 5 | 2 | 0.2 | 0.4 |
| `depth_estimator` | 5 | 2 | 0.2 | 0.4 |
| `bounding_box` | 5 | 2 | 0.1 | 0.2 |
| `vision_ops` | 8 | 4 | 0 (CPU-only tool) | 0 |
| `grasp_generator` | 5 | 2 | 0.1 | 0.2 |
| **Total** | | | | **3.70 ≤ 4.0 ✓** |

**What is scaled is each tool's actor count, not its `num_gpus`** — **all seven tools remain; what drops is concurrency**:
`roborefer` goes from 6 replicas to 3, so the cap on concurrently running roborefer requests is halved.

> Step ① guards against exactly **the thing the eval side failed to guard against** — the same fp32 loading issue turned into a silent contamination in §9.3.

> **`num_gpus` is a Ray logical reservation, not a GPU memory quota.** Actors on the same physical GPU share all of its memory;
> Ray only guarantees "the fractions placed on the same GPU sum to ≤ 1.0". So 3.70 only decides "fits in the schedule or not".
> **This turned into a real incident during the eval phase (§9.3).**

The tool side registers **48 methods** in total; the model makes at most 8 concurrent calls per turn, for at most 8 turns.

### 8.2 Hyperparameters (as-run)

```
algorithm   loss_mode gflowrl · variant cprime · beta 8.0
         eps_low/high 0.2/0.28 · eps_is 0.2
         kl_loss_coef 0 · use_kl_loss True (required, see §7.1)
training   everything inherited from upstream run_rl.sh, not a single character changed
         train_batch_size 64 · rollout.n 5 · ppo_mini_batch_size 64 (= train_batch → strictly on-policy)
         ppo_micro_batch_size_per_gpu 2 · lr 1e-6 · grad_clip 1.0 · total_epochs 1
         max_prompt/response 8192/8192 · max_assistant_turns 8 · max_parallel_calls 8
data     siyich/spacetools-rlfulltools · train.parquet 5500 rows
```

**85 steps, not 86.** 5500 ÷ 64 = 85.9; verl drops the tail that does not fill a batch.

**1 epoch is hardcoded upstream, not our choice.** `trainer.total_epochs=1` is at
`run_rl.sh:435`; git blame points to commit `1e7ff077` (Siyi Chen, 2026-03-24 — the initial commit that added RL
training); our branch has only one commit touching `run_rl.sh` (`574ae81f`, fixing the GPU-reservation deadlock on a single node),
and that line was not changed by a single character. For comparison: **Step 1's point-tool RL defaults to 15 epochs**
(`run_rl_roborefer.sh:36`, `TOTAL_EPOCHS:-15`); **for Step 3's SFT, the paper's Table 6 says Epoch 2
while the code says `max_steps 3000` (≈3.42 epoch); the two conflict, and we ran 3000.**

### 8.3 Optimization techniques used, and what each one did

| Technique | What it did |
|---|---|
| Frozen vision encoder | Only 2.55 B of 4.066 B are trained. The paper's approach; see the notes below |
| FSDP full sharding, **no** offload | 48 GB is enough; offload would make PCIe the bottleneck (and P2P on this machine is broken to begin with); see the notes below |
| ref model parameters offloaded to CPU | ref computes log_prob only once per step, so keeping it resident in GPU memory is not worth it. `timing_s/ref ≈ 122 s`, of which offload itself accounts for only about 5 s (the same single forward pass with parameters on GPU, old_log_prob, has median 112 s vs ref 117 s, §8.4) |
| Gradient checkpointing<br>**gradient checkpointing**<br>(also called activation checkpointing / recomputation; the switch is `model.enable_gradient_checkpointing=True`) | **The forward pass keeps only a few "checkpoint" activations; the rest are recomputed during the backward pass** — trading extra compute for GPU memory. The context here is `max_prompt 8192 + max_response 8192 = 16384` tokens; without it, all layer activations of one forward pass do not fit in 48 GB (**see the calculation in the notes below**); the 35.3 / 48 GB GPU memory peak in §8.4 **was measured with it on**, so there is no headroom to turn it off. For the cost see §4.5 (estimated 20–30% slower on SFT) |
| remove padding | The sequences in a batch differ in length; the usual approach is to **pad the short ones with pad tokens up to the longest one** and compute over the pads too. remove padding **concatenates the real tokens of all sequences end to end into one flat stream**, with boundaries recorded separately in `cu_seqlens`, and **attention does not cross sequence boundaries**. So compute and GPU memory scale only with the **total number of real tokens**, not with `batch × longest sequence length`. **The result is equivalent to computing with padding** (only the floating-point reduction order may differ) — note the difference from `packing`, which was ruled out in §4.5: that one **merges several short samples into one training sample** and changes the semantics of the attention mask |
| flash-attention-2 | Attention memory O(L²) → O(L) |
| **micro 2 + mini 64** | `mini == train_batch` makes the step **strictly on-policy**, so the IS weights in Eq. 8 are identically 1. **Once mini is reduced to save GPU memory, the IS weights actually start working and the effective batch quietly shrinks** — terms and a worked example follow the table |
| sglang async multi-turn rollout | Tool calls do not block the whole batch. `timing_s/gen ≈ 322 s` includes all tool round trips; **see the notes below** |
| Toolshed multi-actor + environment isolation | The paper measured 3.2× faster than naive HTTP at 8-way concurrency |
| `NCCL_P2P_DISABLE=1` | **Not an optimization; a workaround for a hardware fault (§8.5).** The cost is that all-reduce goes through host memory |
| checkpoint janitor (in-house) | `save_freq=5` × **44 GB** per ckpt fills the 250 GB volume by step 20. Keeps only the latest full ckpt (for resume), and additionally saves a **weight-shards-only** snapshot at steps 30/60 (15 GB, without the 7 GB/rank `optim_*.pt`). Archives before deleting, and only acts once the latest one has been written to ≥25 GB |

**remove padding: what it saves, and why it is equivalent.**

**It saves compute and GPU memory, and by a sizable fraction.** Take the real `|y|` distribution of `robospatial` as an example
(measured in §Route C gate: min 45 · median 334 · max 900), and assume a batch happens to pick up exactly these three:

```
padding        pad each one to the longest, 900   ->  3 × 900 = 2700 token slots
remove padding concatenate end to end            ->  45 + 334 + 900 = 1279 real tokens
                                   effective utilization 47.4%; more than half the compute and activations go to pads
```

**Why it is equivalent — because only one operator in a Transformer crosses tokens.**

```
per-token independent:  RMSNorm · q/k/v/o projections · MLP · residual
                 each position only looks at itself; whether there are pads next to it is irrelevant

the only cross-token one:  attention
                 the padding approach masks out the pad columns via the attention mask
                 the remove-padding approach declares boundaries via cu_seqlens; flash-attn does not compute across them
                 both mask out the same set of positions
```

On top of that, the loss is computed only over real tokens anyway (`response_mask`), **so pads contribute nothing to the output of any real token
from start to finish** — removing them therefore cannot change the result mathematically.

> **How far "equivalent" goes: mathematically equivalent, not bit-for-bit identical.** Tensor shapes and reduction order both change,
> so floating-point results differ in very deep decimal places — that is the chain in §9.2.
> In training this difference is swamped by gradient noise, **but it means two runs with remove padding on/off are not bit-for-bit reproducible**.

> ⚠ **Do not confuse this with `packing`, which was ruled out in §4.5.** `packing` **merges several short samples into one training sample**,
> which changes batch semantics and the visible range of attention; remove padding does not merge samples, it only stores the
> sequences of the same batch differently. **One changes semantics, the other does not.**

**How "does not fit in 48 GB" was computed.** Backpropagation needs the intermediate tensors (activations) left by the forward pass.
Without gradient checkpointing, **one copy must be kept for every token in every layer**. Using this model's `config.json`
(`hidden_size 2048` · `intermediate_size 11008` · `num_hidden_layers 36` ·
GQA 2 kv heads × 128), count what must be kept in one layer:

```
attn input (pre/post norm)    2 × 2048 =  4096
q / k / v                 2048 + 256 + 256 = 2560
attn output / o_proj input   2 × 2048 =  4096
mlp gate / up / silu product 3 × 11008 = 33024
                          ---------------------
                          43776 elements/token/layer × 2 B (bf16) = 85.5 KB
× 36 layers                =  3.01 MB / token
```

The configured context limit is `max_prompt 8192 + max_response 8192 = 16384` tokens:

| Sequence length | 1 sequence | at `micro=2` |
|--:|--:|--:|
| **16384 (configured limit)** | **48.1 GB** | **96.2 GB** |
| 8192 | 24.0 GB | 48.1 GB |
| 4096 | 12.0 GB | 24.0 GB |

**An A40 has only 46–48 GB, and it must first hold the sharded weights and optimizer state (~15 GB).**
In other words: at the full limit, activations alone would eat the whole GPU — **it already does not fit at the 8192-token level.**

> **Two easy pitfalls.** ① **FSDP shards parameters/gradients/optimizer state, not activations** —
> each GPU holds its own full activations, and adding GPUs does not make them smaller. ② The table is the **upper bound**; real sequences are much shorter
> (measured in §Route C gate: `|y|` median 305–423 tokens), so normally it is nowhere near this extreme;
> but the configuration must leave headroom for the upper bound, otherwise one long trajectory causes an immediate OOM.
>
> ⚠ This table is **estimated from the config, not measured** — exactly which tensors are kept depends on the implementation
> (flash-attention avoids the O(L²) attention matrix, and PyTorch may fuse some on its own).
> Use it for orders of magnitude; do not quote it as an exact value.

**What "sglang async multi-turn rollout" means — there are two layers of asynchrony here, and both are needed.**

```
verl side    one independent ToolAgentLoop coroutine per sample, each running its own multi-turn state machine
           while sample A waits for roborefer to return, sample B has already spliced its previous tool result back into the context
           and is asking sglang for the next generation segment
           knobs: rollout.name=sglang · max_parallel_calls=8 (cap on concurrent tool calls within one trajectory)

sglang side  continuous batching — whoever finishes generating exits, whoever is ready joins the running batch,
           no waiting for the whole batch to line up
```

**Continuous batching alone is not enough.** If the verl side were lockstep (whole batch generates together →
whole batch calls tools together → generates together again), each turn would take as long as **the slowest tool call in that turn**;
one step is 64 questions × 5 rollouts = 320 trajectories, each with up to 8 turns, and the whole step would repeatedly get stuck on the slowest tail.
Continuous batching can only pack the "generation" segment tightly; **it cannot fill the gaps left by tool round trips**.

So `timing_s/gen ≈ 322 s` contains both **the policy's decoding time** and **thousands of tool round trips**,
and it did not become the sum of the two — that is thanks to the overlap of these two layers.

> ⚠ **The same mechanism: throughput on one side, non-reproducibility on the other.** Continuous batching means a request has different batch companions at every
> decode step — when the batch size changes, the kernel's reduction order changes, and argmax flips at near-ties.
> **The source of the noise chain in §9.2 is right here.**

**What "FSDP full sharding, no offload" means.** Breaking down the four terms:

| | What it is |
|---|---|
| **FSDP** | Fully Sharded Data Parallel, PyTorch's distributed training approach. **Splits parameters, gradients and optimizer state across GPUs**, each GPU stores only 1/N; when a layer needs its full parameters, they are gathered from the other GPUs on the fly (all-gather) and discarded right after use |
| **Full sharding** | All three of the above **are split** (equivalent to DeepSpeed's ZeRO-3 level). The more is split, the more GPU memory is saved, at the cost of more communication |
| **offload** | Moves parameters or optimizer state **to CPU memory** and brings them back to GPU when needed. The last resort when GPU memory is short |
| **PCIe** | The bus between GPU and CPU, and between GPUs. **Bandwidth is one to two orders of magnitude lower than GPU memory**; everything moved goes over it (the A40 has no NVLink) |
| **P2P** | peer-to-peer, two GPUs **transfer directly to each other, bypassing the CPU**. Normally inter-GPU communication uses it |

**The decision this cell describes: FSDP full sharding on, offload off.**

```
FSDP full sharding on   params / grads / Adam state each stored as 1/4 across the 4 training GPUs; fits in 48 GB ✓
offload off             all of this stays in GPU memory the whole time, never moved to the CPU
```

`param_offload=False` / `optimizer_offload=False` (`run_rl.sh.asrun:477–478`).
**The only exception is the ref model**: `ref.fsdp_config.param_offload=True` (:492) —
it does only one log_prob forward pass per step, so keeping it resident in GPU memory is not worth it; `timing_s/ref ≈ 122 s`, of which the extra cost of offload is only about 5 s (§8.4).

**Why offload is off:** every offload transfer goes over PCIe, and P2P on this machine is broken (§8.5);
NCCL is already forced to `NCCL_P2P_DISABLE=1` and relays through host memory — **PCIe is already packed with all-reduce traffic**.
Adding offload on top would load the same already-bottlenecked path again. **Since GPU memory is sufficient, we don't pay that price.**

> In other words: the hardware fault in §8.5 did not just slow down throughput, it also **changed the optimal choice here** —
> on a machine with working P2P, offload would be much better value.

> ⚠ **Do not confuse this with "z3 → z2" in §4.2.** The two are different stages, different frameworks, different trade-offs:
> SFT uses DeepSpeed, and **z2 fits in 48 GB**, so z3 was dropped to z2 to save communication that was paid for nothing;
> RL uses verl's FSDP, and the training state is much larger than SFT's (plus the rollout engine and the ref model),
> **it does not fit without full sharding**, so the communication cost has to be paid. **Same GPU model, opposite conclusions, because whether it fits changed.**

**What "frozen vision encoder" means.** Qwen2.5-VL consists of two halves: the **vision encoder** (cuts the image into
patches and encodes them into a sequence of tokens) + the **language model** (reads those tokens and the text, reasons, and writes `tool_call`).
`freeze_vision_model=true` (`run_rl.sh:392`, upstream default, untouched by us) applies only to the first half:

```
forward    runs as usual — the image is still encoded, the model can still "see"
backward   no gradients computed for it
optimizer  its parameters are not in it at all; Adam's m / v are not allocated for it either
```

So **only 2.55 B of the 4.066 B parameters are updated**; the remaining ~1.5 B are read-only throughout. It saves more than compute:
gradients and Adam state are allocated per **trainable parameter count**, so freezing a block saves three copies of GPU memory proportionally
(gradient + m + v).

**Why it is reasonable:** what this RL step needs to change is policy behavior like "how to orchestrate tools", not "how to look at images";
and the vision encoder was also frozen during SFT (§4.2 `freeze_vision_tower` stays `true`),
**so it was never trained from start to finish**; both stages use the same convention.

**Cost:** the model's visual representation is pinned at the pretraining level and RL cannot change it. So errors of the "misread the image" kind
structurally have no chance of improving in this training run — keep this boundary in mind when citing P6's error attribution.

> 4.066 B was read directly from the safetensors headers; 2.55 B is taken from the training report
> (`05_gflowrl_training/gflowrl_training_report_8xA40.md` §Architecture); this report did not independently re-verify this split.

**What `micro` / `mini` / IS weights each are** — that cell in the table above is the densest in terms, so it gets its own section here:

```
micro-batch   ppo_micro_batch_size_per_gpu = 2
              the sample chunk actually fed to the GPU for one forward+backward. Gradients accumulate over several micros,
              and parameters are updated only once a full mini is collected. A pure GPU-memory knob; does not change the math.
mini-batch    ppo_mini_batch_size = 64
              the number of samples covered by one optimizer.step(). verl splits the collected train_batch
              (64 prompts × rollout.n=5 = 320 trajectories) into several minis,
              and each mini updates the parameters once.
```

**`mini == train_batch` (both 64) ⇒ a rollout batch yields only one mini ⇒ parameters are updated only once per batch.**
Add `ppo_epochs=1`, and **the sampling policy `π_old` and the updated policy `π_θ` are the same set of parameters at that moment** —
this is what "strictly on-policy" means.

**IS weight (importance sampling weight)** is the `w_i` in Eq. 8, used to correct
the mismatch that "trajectories were sampled by the old policy, but the gradient is computed for the new policy":

```
w_i = min( π_θ(y_i|x) / π_old(y_i|x) ,  1+ε_is )     Eq.8 clips only the upper bound, not the lower
    = min( exp( Σ_t [ log π_θ − log π_old ] ) , 1.2 )      (ε_is = 0.2)
```

When strictly on-policy, `π_θ ≡ π_old`, the sum is identically 0, `w = min(exp(0), 1.2) = 1` —
**the weight might as well not exist, and this is the actual state of this run.**

**But it is configuration-dependent.** Once `mini` is reduced to save GPU memory (or `ppo_epochs > 1`), a batch gets
updated several times, and from the second update on `π_θ ≠ π_old`, so `w` really starts to act. And the exponent is **the sum of per-token
log-probability differences over the whole sequence**, so a tiny per-token drift gets amplified by sequence length:

```
per-token drift −3e-3 ·  sequence length 1000 tokens  ->  w = exp(−3.0) ≈ 0.05
                       sequence length 1300 tokens  ->  w = exp(−3.9) ≈ 0.02
```

**The sequence is still in the batch, still takes GPU memory, still runs the forward pass, but its contribution to the gradient is pushed down to a few percent** —
the effective batch quietly shrinks, and nothing shows on the loss curve. The two IS-weight guards (`is_weight_min`,
`is_weight_collapsed_frac`, which in this run should stay at 1.0 and 0.0 respectively throughout) exist to make this visible.

Not used: LoRA (full-parameter fine-tuning), optimizer offload, `expandable_segments` (conflicts with sglang's
TorchMemorySaver), tuning `rollout.n` (changing it makes both arms incomparable).

### 8.4 Performance and readings

```
timing_s/step  981 s (last step; mean over the run 925 s)
  ├ gen              322 s   rollout — sglang generates multi-turn trajectories, including all tool round trips
  ├ old_log_prob     119 s   recompute π_old's log_prob with the current policy, one forward pass (needed for d in Eq. 4/6)
  ├ ref              122 s   one log_prob forward pass of the reference model π_ref (needed for d = Σ(log π_ref − log π_old))
  ├ update_actor     369 s   forward + backward + optimizer.step(), the actual gradient update
  ├ save_checkpoint   39 s   writes the 44 GB checkpoint to disk, only on steps hit by save_freq=5
  ├ update_weights    10 s   syncs the updated weights to sglang
  └ rest            0.3 s
perf/throughput  346 token/s · mfu 0.145 · GPU memory peak 35.3 / 48 GB · CPU peak 151 GB
85 steps × 925 s ≈ 21.8 h, matches wall clock 21 h 57 min
```

Three definitions you must know to read these numbers:

- **The 346 for `throughput` is "per training GPU", not the whole machine.** That step's `perf/total_num_tokens = 1,360,105`;
  `1,360,105 / 981 s = 1386 token/s` is the four-GPU total, divided by the 4 training GPUs gives 346. **Align the GPU count when quoting across runs.**
- **`mfu` = Model FLOPs Utilization**, achieved FLOPs ÷ hardware peak FLOPs; 0.145 means 14.5%.
  This is a **tool-augmented multi-turn** run: within a step, 322 s go to rollout and tool round trips, 119 s and 122 s to the old_log_prob and ref forward passes,
  and only the 369 s of `update_actor` is real dense matmul. **So this number cannot be compared with the mfu of pure LM training**,
  nor with other machines (`NCCL_P2P_DISABLE=1` makes all-reduce go through host memory, §8.5).
- **The breakdown adds up (added 2026-09-30).** Previously only four items were copied — gen / update_actor / ref / save, `322+369+122+39 = 852` —
  and the 129 s gap to 981 was recorded as a bookkeeping gap. After reading the full training log on HF (`provenance/full_train.log.gz`) it was resolved:
  **= `old_log_prob` 118.8 s + `update_weights` 10.0 s + 0.3 s**, both of which verl already times.
  85-step medians: gen 321 · old_log_prob 112 · ref 117 · update_actor 352 · update_weights 10 · 922 s per step
  (`GFlowRL_improve/infra_prep/stage_timing.py`). old_log_prob + ref + update_actor, which scale with the number of rows, together take 63%;
  gen takes 35% and is mostly waiting on tools — these two blocks are the starting point of the Design Doc's "Infra · training throughput".

Key readings over all 85 steps:

```
                 min      median      max
degen           0.578    0.703    0.859      fraction of reward-degenerate groups
sat             0.303    0.516    0.613      Eq.7 clip saturation rate
sat_b2 / b4 / b16  0.478 / 0.500 / 0.525 (median)  recomputed with β set to 2/4/16
reward          0.290    0.672    1.695      reward half of the flow gap
drift           0.000    0.339    0.441      drift half of the flow gap
rew/drift       0.858    2.014    4.941      < 1 = reward swamped by drift
sign_ok         0.689    0.837    1.000      on non-degenerate groups, fraction where post-clip g has the same sign as the reward term
d_seq           0.000    0.374    0.478      |Σ(log π_ref − log π_old)|
grad_norm     103.075  180.312 5229.205      before clipping
score           0.581    0.806    1.083      mean reward of the step
```

How to read them:

- **`sat` stays around 0.5 and does not drift toward 1.0** — magnitude information is still there; clip has not degraded the gradient into pure sign.
- **`rew/drift` median 2.01; of the 85 steps only step 76 drops below 1 (0.86)** — the reward signal keeps drift in check overall.
  Trend: median 2.125 over the first 10 steps → 1.754 over the last 10, a slow decline but not a large one.
- **`sat_b2` 0.478 / `sat_b8` 0.516 / `sat_b16` 0.525 — β differs by 8×, and the saturation rate moves only 5 percentage points.**
  This is the per-step empirical evidence that "lowering β does not cure saturation".
- **`grad_norm` median 180, while `grad_clip=1.0`.** **Every single update is clipped down to direction only, with no magnitude.**
  This must go into the conclusions, otherwise the number "learning rate 1e-6" will be misread.
- **`score` has no reliable upward trend.** Median 0.766 over the first 10 steps, 0.820 over the last 10, but 0.809 for the first half and
  0.796 for the second half — swings of this size are noise. **Training reward did not improve noticeably in this run.**
- **`degen` median 70.3%** — for most groups in every batch, the reward term is identically 0.

### 8.5 A hardware fault: PCIe P2P inside the container is broken

**Symptom:** after the trainer comes up, the first NCCL collective never completes, and after 10 minutes the watchdog reports
`Last enqueued NCCL work: 1, last completed NCCL work: -1`. Yet `nvidia-smi topo -p2p r`
shows all OK.

**Diagnosis:** instead of repeatedly launching 15-minute training runs to test, we wrote a **minimal probe** doing a 4-GPU all-reduce
and ran 5 variants, 90 seconds per round:

```
A  default                          -> 40 s timeout
B  NCCL_IB_DISABLE=1             -> 40 s timeout   (not an IB/RoCE issue)
C  B + NCCL_SOCKET_IFNAME=eth0   -> 40 s timeout
D  B + NCCL_P2P_DISABLE=1        -> 7 s ALLREDUCE_OK   <- this is it
E  B + NCCL_SHM_DISABLE=1        -> 40 s timeout   (SHM is exactly the path that works)
```

**Root cause:** the typical sign of ACS / IOMMU not being disabled inside a container — P2P is "available" in the topology, but actual transfers hang.
**A platform problem, not a code problem.**

**Cost:** with P2P off, NCCL uses shared memory (relayed through host memory), which is slower. **The 925 s per step in this run was measured under
this condition and cannot be compared directly with other machines' throughput.**

> **Lesson: after renting a new machine, spend 90 seconds running an all-reduce probe before running training.**
> Without it this time, it would have been 15 minutes of waiting per environment-variable change, and the day would be gone.

### 8.6 Runtime errors on the tool side (all correctly swallowed)

Counts from the 9.8 MB training log:

```
No collision-free grasps found          3389 times (of which 2346 raised as RuntimeError)
Top-down filtering removed all grasps    130 times
selected index k out of range             24 times
CUDA error: invalid configuration argument 12 times (depth_estimator.get_depth_map)
Mask removed all points                   10 times
torch.cuda.OutOfMemoryError                2 times (tool side)
hit the 8-turn limit                              2 times
truncated tool responses / training-side OOM / NCCL WARN / crashes    0
```

**These are not training failures.** Toolshed turns tool exceptions into text returned to the model, and the model can switch tools or retry with different arguments
— **this is exactly what tool-augmented RL is supposed to learn.** The 3389 from `grasp_generator` are especially normal:
it does collision checking on real point clouds, and "there is no collision-free grasp pose around this object" is a **geometric fact**.

**But two are worth noting separately:** two tool-side OOMs, one trying to allocate **14.36 GiB** and the other
**1481.25 GiB**, both inside `torch.cdist` — 1481 GiB is obviously not a shortage of GPU memory but **degenerate input**
(an abnormally inflated point count in the point cloud) blowing up the intermediate matrix. **If we ever compute a "tool success rate", these must be counted in the denominator.**

**Sources:** `05_gflowrl_training/gflowrl_training_report_8xA40.md` · `training/GFlowRL/A40/` (metrics / logs /
as-run scripts / environment snapshot)

---

## 9. GFlowRL eval

### 9.1 Definition

```
nine benchmarks · 2121 samples · greedy decoding (verl val_kwargs default temperature 0)
4× RTX A6000 (same GPUs as the SFT baseline) · NUM_GPUS=4 EVAL_GPUS=1
```

Three runs:

```
Run A  2026-09-17 23:40 -> 09-18 01:45  2 h 05 m   nine keys
       five (robospatial/reflocation/refplacement/refunseen/cvb2drelation) zero OOM, valid
       the other four were contaminated by a GPU memory problem, results discarded (§9.3); this report does not cite their scores
Run B  09-18 03:06 -> 03:53             46 m       rerun of the four contaminated ones, zero OOM
Run C  09-18 04:16 -> 04:36             19 m 45 s  robospatial second time, zero OOM
```

### 9.2 Three parameter changes

| Change | Why | Effect on scores? |
|---|---|---|
| `BENCHMARKS` nine path lines `xxx/test.parquet` → `data/<key>.parquet` | The upstream script expects nested directories, but the dataset **is flat both on `main` and on the pinned revision**. Upstream changed the layout and the script did not follow. Without the change: `exit 1`, not a single sample runs | **None** — only the path changes; the parquet read is the same one |
| `gpu_memory_utilization` 0.5 → **0.545** | See "Why it must be aligned" below | **Yes, and that is the point** — it brings the condition back to the same one as the baseline |
| `vlm` `num_gpus` 0.6 → **1.0** | See §9.3 | **Does not enter the scores** — only changes Ray placement, not the model, the grading or tool behavior |

**Why `gpu_memory_utilization` must be aligned.** This knob is a **fraction of the whole GPU, not an absolute value**
— the same `0.5` opens KV pools of different sizes on GPUs of different capacity.

```
SFT baseline (RoboSpatial 61.00 ± 0.77)   48 GB GPU × 0.5   = 24 GB KV pool
smallest GPU on this machine  46068 MiB   × 0.5   ≈ 22 GB     ← copying 0.5 gives a smaller pool
                                          × 0.545 ≈ 24 GB     ← value back-computed
```

**The only use of this eval is a per-sample comparison against the SFT starting point and against P4, so apart from the checkpoint itself,
every other condition must be pinned.** The KV pool is not a neutral knob:

```
pool size → how many sequences run concurrently in a batch → batch composition → floating-point reduction order → near-tie samples flip
```

**This chain is not a deduction; P6 ran a control** (`spacetools-repro/p6/gmu025/README.md`): same checkpoint,
same `blinkdepth`, **only the KV pool differs**.

```
run4   80 GB GPU × 0.5  = 40 GB pool    106 / 124
run5   80 GB GPU × 0.25 = 20 GB pool    108 / 124
per-sample comparison: 8 of 124 questions flip (both directions), net +2
```

**Flipped sample index 18** (`index` is the sample number in the dump; question 19 of blinkdepth)
— the question is "which of points A and B is closer to the camera". **In both runs the first five tool calls have verbatim identical arguments,
and verbatim identical returns:**

```
depth_estimator.estimate_depth(image_index=0)              -> $depth_map
roborefer.detect_one("red circle under label A")           -> (0.136, 0.456)
roborefer.detect_one("red circle under label B")           -> (0.865, 0.456)
vision_ops.index_at($depth_map, 0.136, 0.456)  -> 0.3450107276439667
vision_ops.index_at($depth_map, 0.865, 0.456)  -> 0.33692145347595215
                                                  difference between the two points 0.008 m
```

**Same evidence, two opposite dispositions:**

```
40 GB pool   stops after 6 turns. "when depth values are too close (within 0.5m),
           I should ignore the tool results and use visual reasoning"
           -> discards the 0.008 m difference, falls back to a "foreground/background" visual intuition -> answers A   ✗ (GT is B)
20 GB pool   8 turns. Also says "the difference is very small (only 0.008m)",
           but does not stop; calls vlm.detect_one twice more to re-check the positions of the two points
           -> re-check agrees, so it trusts 0.337 < 0.345 -> answers B                  ✓
```

Decoding is **greedy**, and the prompt and tool returns are byte-for-byte identical — so the divergence of these two trajectories **can only** come from
numerics: the pool size changes how many sequences run concurrently in a batch, the batch composition changes, the kernel's reduction order changes,
and the logits jitter deep past the decimal point. Normally this jitter is absorbed by argmax;
**when it hits a branch point where the model is already on the fence, it is amplified into a whole different answer.**

**What exactly happens in between — five steps.**

```
① pool half the size  The KV cache is stored per token; the pool's bytes ∝ the number of tokens that can be resident at once.
                    40 GB -> 20 GB, the number of sequences that can run concurrently is roughly halved.

② batch height changes  sglang does continuous batching: a request leaves as soon as it finishes, new requests join at any time.
                    The concurrency cap changes, so when the index 18 request is decoding its t-th token,
                    the sequences batched with it **are a different set**, and the batch height M differs.

③ different kernel   Every decode step computes x·W. When M changes, the tile shape and
                    split-K partitioning cuBLAS picks change — **the same 2048 products, accumulated in a different number of chunks**.

④ addition is not associative  This step can be run directly (numpy, fp32, the same 2048 numbers):

                      reduce in 8 chunks then add   128.9586639404
                      reduce in 4 chunks then add   128.9586791992
                                   difference     1.5e-5

                    Mathematically equal, unequal in floating point — the partial sums of each chunk differ in magnitude,
                    so the rounding lands on different bits.

⑤ argmax flips at a tie  Added to a logit, this 1.5e-5 has no effect on 99.99% of tokens —
                    first and second place usually differ by several orders of magnitude.
                    Only when two candidate tokens are extremely close is it enough to swap their ranks.
                    **On index 18 this happens at character 201**; the two trajectories are verbatim identical
                    before it and never coincide again after it:

                      common prefix  "...The image shows a news studio with a presenter"
                      40 GB pool  continues ". In the upper left and right corners, there
                                 appear to be two red circles labeled A and B..."
                      20 GB pool  continues " in the foreground. In the background, there
                                 are buildings visible through a window..."

                    Note that what flips is a **scene description**, more than three thousand characters before the answer.
```

> ⚠ **A qualification that must be stated.** This batch of dumps **did not store logprobs**, so "which tokens were close"
> cannot be read off directly — the causal direction of step ⑤ above is **inferred backwards from the outcome**: the two trajectories split at character 201,
> before that all inputs are byte-for-byte identical and decoding is greedy, and the only other variable is batch composition.
> The basis for "the model is on the fence here" is likewise only **what it wrote itself** (both times it wrote "the difference is too small to be reliable"),
> not a measured probability. **To pin this chain down, top-k logprobs must be stored during eval** — already recorded in §12.

**The flip only needs to happen once.** In autoregressive decoding, every generated token is appended to the context and becomes a condition for all later
tokens. So from the flipped step onward, the two runs **are no longer two floating-point versions of the same computation** —
their contexts are already different texts, they compute different conditional distributions, and no force pulls them back together:

```
40 GB pool  first sentence describes A and B as "two red circles in the upper-left/upper-right corners"
          -> the rest keeps circling around the picture's composition -> stops after 6 turns -> visual intuition -> answers A
20 GB pool  first sentence describes A and B as "points on the background buildings outside the window"
          -> the rest keeps chasing the positions of these two points -> calls vlm twice more to re-check -> 8 turns -> answers B
```

The difference over the next three thousand-plus characters is no longer carried by 1.5e-5; it is carried by **the first scene-description sentence** —
1.5e-5 only paid for the first nudge.

**Conversely, this explains why this kind of noise only shows up on particular questions:** where the branch point lands is itself random
(index 18 landed on an irrelevant sentence of scene description), but **only when the question's verdict is already borderline
does the branch go all the way to a different answer**. On questions with sufficient evidence, the two trajectories word things differently and still reach the same conclusion —
the 777 per-sample fully identical samples in §10.3 are exactly this case.
`robospatial`'s Vacant is the worst-hit area (§10), precisely because its criterion is binary convex-hull membership on continuous coordinates,
so borderline samples are ties everywhere.

Reproducing the two numbers in ④:

```python
import numpy as np
rng = np.random.default_rng(0)
p = (rng.standard_normal(2048) * rng.standard_normal(2048)).astype(np.float32)
f = lambda k: np.float32(sum(np.float32(np.add.reduce(c)) for c in p.reshape(k, -1)))
print(f(8), f(4))        # 128.95866  128.95868
```

> **Why we can conclude it is this chain and not the tools.** The SFT eval did a direct check: across 350 samples,
> **zero cases of "same query, different tool result"** — the tool side is bit-for-bit deterministic.
> With the tools excluded, the only remaining variable under greedy decoding is batch composition.

> So the sentence in §10, "a 4-question difference cannot separate the machine term from the model term", is not cautious wording —
> **here the same model differs from itself by 2 questions.**

In other words, **without alignment, the measured difference contains a purely machine-driven term**, and that term cannot be numerically separated from
"how much stronger C′ is than the SFT starting point". Only after alignment is the difference left with just the model term. **The cost: any score cited must carry the
`gmu=0.545` definition**, and switching machines requires back-computing it again.

> ⚠ The alignment is only **approximate**. This machine has mixed ECC (GPU0/1 49140 MiB, GPU2/3 46068 MiB);
> depending on which GPU the **policy model** (policy — the Qwen2.5-VL-3B being evaluated, hosted by sglang; occupies GPUs separately from the seven tool models)
> lands on, the pool differs by `3072 MiB × 0.545 ≈ 1674 MiB ≈ 1.6 GiB`:
>
> ```
> GPU0/1  49140 MiB (ECC off) × 0.545 ≈ 26781 MiB pool
> GPU2/3  46068 MiB (ECC on)  × 0.545 ≈ 25107 MiB pool   <- Run C measured landing on this one (GPU3)
>         ↑ capacity difference 3072 MiB      × 0.545 = pool difference 1674 MiB
> ```
>
> The source of the 3072 MiB is **ECC**: a GPU with ECC on reserves part of its memory for check bits, so the capacity it reports is 3 GiB less.
> Same batch of A6000s, two on and two off.
>
> **Don't mix up two things:** *how much* it differs is set by "capacity difference of the two GPUs × `gmu`"; **how much memory the policy model itself takes does not enter this formula**;
> *whether* it differs is set by which GPU the policy lands on, because `gmu` is a fraction of **the GPU it is on** —
> on GPU0/1 the pool is opened from 49140, on GPU2/3 from 46068.
> **Ray placement decides which GPU's capacity goes into the formula, ECC decides how far apart the two capacities are; the policy model is just the thing being placed.**
> This is one of the sources of the "a 4-question difference cannot separate the machine term from the model term" in §10.

**Two things not changed but that must be given explicitly:** `NUM_GPUS=4 EVAL_GPUS=1` (upstream default 8/4; on a 4-GPU machine,
not overriding them makes Ray **not error out, with actors queued forever**); `BASH_ENV=.../conda.sh` (`conda activate`
is a shell function and does not cross subprocesses).

**Deliberately not changed:** the fp32 issue in `vlm.py:116` itself — the paper's results were produced under that behavior.

### 9.3 A real silent contamination, and how it was caught

Four benchmarks in Run A hit a GPU memory mismatch:

```
run_eval.sh passes 'dtype': 'float16' to vlm, but upstream vlm.py:116 hard-codes torch_dtype="auto"
-> Molmo-7B-D loads in fp32, measured at 30.2 GiB, while it only declares num_gpus=0.6
-> Ray, entirely legally, packs vlm(0.6) + depth_estimator(0.2) + sam2(0.2) = 1.0 onto the same GPU
-> 30.20 + 9.15 + 7.97 = 47.3 GiB / whole GPU 47.4 GiB, only 70 MiB left
-> a tool asks for another 576 MiB -> torch.OutOfMemoryError
OOM counts: blinkdepth 102 · cvb3ddepth 499 · boppose 21 · bopgrasp 19
```

**The symptom is not a crash but a silent score drop.** After a tool OOMs, Toolshed wraps the error into a normal `ToolResult` and returns it;
the model keeps reasoning with the error string, the eval finishes, reports no error, and outputs a plausible-looking number.

**Error attribution quantifies this clearly:**

```
cvb3ddepth   contaminated run 243 wrong -> 230 attributed to OOM, only 13 are genuinely wrong
             rerun        21 wrong ->   0 attributed to OOM, all 21 genuinely wrong
blinkdepth   contaminated run  25 wrong ->  19 attributed to OOM
             rerun        15 wrong ->   0
```

There is also a **second-order effect**: the contaminated run's log has 590 lines of
`TypeError: Expected PIL Image, got <class 'str'>` — that is downstream of the OOM:
the tool returns an error string → the model feeds it as a `$variable` to the next tool → TypeError. After the rerun it drops to 4.

**Why it did not turn into a wrong conclusion: because the order of the gates is fixed.**

```
count OOM first -> then check for silent tool errors -> only then look at the score
```

**Three independent counts corroborate each other:** main log `grep -c OutOfMemoryError` = 0 · each benchmark's own
`eval.log` = 0 · per-sample `OOM samples` from `parse_dump.py --strict` = 0 ·
even the raw text `CUDA out of memory` appears 0 times.

> **This also exposed two methodology defects, both fixed:**
> ① After `run_eval.sh` prints `EVALUATION COMPLETE`, its own cleanup trap kills the background
> toolshed, `wait` carries the SIGTERM out, and **the script's exit code is 15**. All three runs had exit code 15,
> and all three ran to completion — **judging by exit code would discard a batch of fully valid results.** The correct criterion is
> **success marker + sample count + OOM count + router-lost count**, not `$?`.
> ② `run_eval.sh` has no concurrency protection: both instances run `ray stop --force` at startup, and the later one kills the earlier one's
> toolshed together with all its tool actors, while **the earlier eval keeps running and still produces scores**,
> only every tool call gets `Could not find ToolRouterActor`. An atomic lock has been added + this count is now part of the gate.

### 9.4 Health gate (look at scores only if all zero)

```
                      n    OOM  truncated  max-turns  missing<answer>  samples with tool failure
robospatial(A/C)    350     0    0      0        0          0
reflocation         100     0    0      0        0          0
refplacement        100     0    0      0        0          0
refunseen            77     0    0      0        0          0
cvb2drelation       650     0    0      0        0          0
blinkdepth          124     0    0      0        0          1
cvb3ddepth          600     0    0      0        0          0
boppose              60     0    0      0        0          0
bopgrasp             60     0    0      0        0         43   <- domain result, not a fault
```

The 43 for `bopgrasp` are `grasp_generator` returning "no collision-free grasp found" (32) or "top-down filtering removed
all grasps" (11). **P4's corrected number under the same definition is 41/60; the two agree, so this is not a problem of this run.**

> Incidentally reproduced an upstream defect P4 had recorded: `analyze_grasp_result.py` only matches
> `No collision-free grasps`, **missing `Top-down filtering removed all`**,
> so it reports 32 error samples when the real number is 43.

### 9.5 Results

```
RoboSpatial   VQA      228   71.93 / 72.81   (two runs)
              Vacant   122   41.80 / 44.26
              Overall  350   61.43 / 62.86   mean 62.14
RefSpatial    Location 100   52.00      Placement 100  58.00      Unseen 77  48.05
              three items     277   52.68 simple mean / 53.07 weighted
BLINK Relative Depth   124   87.90
CV-Bench 2D / 3D  650/600   94.62 / 96.50
BOP-ASK Pose            60   55.73 mean IoU
BOP-ASK Grasp           60   MACE 44.79 · SR 55.00%
```

### 9.6 Against the SFT starting point: no measurable improvement

| | n | SFT starting point | **C′ step85** | Diff |
|---|--:|--:|--:|--:|
| RoboSpatial VQA | 228 | 71.27 | 72.37 (mean of two runs) | +1.1 |
| RoboSpatial Vacant | 122 | 41.80 | 43.03 (mean of two runs) | +1.2 |
| **RoboSpatial Overall** | 350 | **61.00 ± 0.77** | **62.14** | **+1.14** |
| RefSpatial three items (weighted) | 277 | 53.07 | 53.07 | **0.00** |

**A difference of 4 questions ≈ 1.5 standard errors; not a statistically meaningful improvement:**

```
C′  flips 25 -> single-run sd ≈ 2.5 questions, two-run mean sd ≈ 1.8 questions
SFT flips 29 -> two-run mean sd ≈ 1.9 questions
pooled sd ≈ 2.6 questions, diff 4.0 questions -> about 1.5 σ, two-sided p ≈ 0.12
the two intervals 205–230 and 199–228 overlap almost completely
```

**Where these numbers come from.** The model is one sentence: **always-correct samples are correct every time; a flipping sample is ≈ one coin toss each time.**

```
questions correct = always-correct + Bin(k, 0.5)            k = number of flipping samples
single-run sd            = √k / 2
sd of the mean of two runs = √(k/2) / 2
```

`k` and "always-correct" are both counted, not estimated:

| | Which two runs | n | Always-correct | Flips k | Possible interval | Expected |
|---|---|--:|--:|--:|---|--:|
| **C′** | P7 eval Run A ↔ Run C | 350 | 205 | **25** | 205 – 230 | **217.5** |
| **SFT starting point** | the two SFT eval runs | 350 | 199 | **29** | 199 – 228 | **213.5** |

(C′'s two runs got 215 / 220 correct; the SFT two runs 216 / 211, i.e. 61.71% / 60.29%.)

The rest is arithmetic:

```
two-run mean sd    C′  √(25/2)/2 = 1.77      SFT  √(29/2)/2 = 1.90
pooled sd        √(1.77² + 1.90²) = 2.60
diff             217.5 − 213.5 = 4.0 questions
z              4.0 / 2.60 = 1.54        two-sided p = 0.124
```

> **Two assumptions must be stated together.** ① Each flipping sample comes out correct with probability **0.5** — this is the **maximum-variance** assumption,
> so 1.54 σ is a **conservative** (harder to reach significance) reading; if the true flip probability departs from 0.5, σ only gets smaller and
> z only gets larger, but estimating that requires several more runs per question. ② Samples are assumed independent — samples in the same batch share
> a batch and are strictly not independent, but the chain in §9.2 shows the correlation comes from floating-point noise rather than question content, so its magnitude is negligible.
>
> **The numerator deserves more attention than the denominator:** this 4.0-question difference still contains the machine term (§12.1 items 7 and 10),
> and a statistical test cannot handle that.

**Moreover, the SFT 61.00 was measured on a different machine**; this time SFT was not re-measured on this machine.
Add this machine's **mixed ECC** (GPU0/1 = 49140 MiB / ECC off, GPU2/3 = 46068 MiB / ECC on,
a 3072 MiB difference, and gmu is a fraction of the whole GPU → depending on which GPU the policy lands on, the KV pool differs by about 1.7 GB),
and **the machine-level and model-level explanations of this 4-question difference currently cannot be separated.**

> **There is only one clean way to pin it down: run robospatial on the SFT checkpoint too, on the same machine, same session.**
> This step was not done (the machine has been released); it is the single biggest gap in this report.

**Sources:** `06_gflowrl_eval/gflowrl_eval_report.md` · `eval/GFlowRL/P7_GFLOWRL_EVAL/`
(dumps / logs / gates / scripts / gpu / config / MANIFEST)

---

## 10. Per-sample comparison with P4 / P5 / P6

This section consumes only parsed records, **zero GPU**. P7's parsed records are generated from the dumps by
`spacetools-repro/tools/parse_dump.py --emit`, with fields identical one by one to `p4/parsed/`
(`chain_signature`, `vars_exposed/used/unused/phantom`, `trajectory`,
`num_turns_verl_convention`).

### 10.1 What it can and cannot answer

**Can:** where the official post-RL checkpoint and our C′-trained checkpoint differ in behavior, **per sample**, on the same 2121
samples. Tool chains, variable reuse, error attribution and pass-through rate are all verifiable traces.

**Cannot:** "is C′ better or worse than GRPO". The two checkpoints **have different bases** — the official one starts from the paper's own
SFT starting point, ours is self-trained. **This is not a controlled A/B.**

**The truly controlled comparison is SFT starting point ↔ C′, and that side has only summary numbers, no dumps**
(the SFT eval ran on a machine that has been released). **The next time a checkpoint is evaluated, the dumps must be archived with it.**

### 10.2 Summary table

| benchmark | n | P4 correct | P7 correct | Only P4 correct | Only P7 correct | McNemar p | |
|---|--:|--:|--:|--:|--:|--:|---|
| robospatial | 350 | 229 | 215 | 37 | 23 | 0.092 | not significant |
| reflocation | 100 | 54 | 52 | 3 | 1 | 0.625 | not significant |
| refplacement | 100 | 58 | 58 | **0** | **0** | 1.000 | **identical per sample** |
| refunseen | 77 | 37 | 37 | **0** | **0** | 1.000 | **identical per sample** |
| blinkdepth | 124 | 107 | 109 | 4 | 6 | 0.754 | not significant |
| cvb2drelation | 650 | 615 | 615 | 6 | 6 | 1.000 | not significant |
| cvb3ddepth | 600 | 579 | 579 | **0** | **0** | 1.000 | **identical per sample** |
| boppose | 60 | 38 | 39 | 1 | 2 | 1.000 | not significant |
| bopgrasp | 60 | ~~54~~ | ~~56~~ | ~~0~~ | ~~2~~ | 0.500 | ⚠ definition inverted, see end of §10.2 |

P4 uses run1 (single run); the P4 report itself ran robospatial / blinkdepth / bopgrasp several times and reported intervals.

**The two continuously scored ones use a paired sign test; the mean and the per-sample direction contradict each other:**

```
boppose    P4 mean 0.5336   P7 mean 0.5573   diff +0.0237
           but per sample P7 higher 19 · P4 higher 26 · bit-for-bit identical 15    p = 0.371
bopgrasp   P4 mean 1.9354   P7 mean 1.8576   diff −0.0778
           but per sample P7 higher 25 · P4 higher 31 · bit-for-bit identical  4    p = 0.504
```

**First, what `score` is.** For these two benchmarks `score` is not accuracy; it is two geometric quantities **pointing in opposite directions**,
computed by the official scorer `verl/utils/reward_score/bop_ask_bench.py`:

| benchmark | What `score` is | Range | Direction |
|---|---|---|---|
| `boppose` | The **IoU** of the two polygons obtained by taking the **convex hull** of the 8 predicted cube corners and of the GT 8 points | 0 – 1 | **higher is better** |
| `bopgrasp` | **NCE** = mean error of the 5 gripper keypoints ÷ GT gripper width `d`, then capped at 10 | 0 – 10 | **lower is better** |

Walking through `bopgrasp` sample 0 (numbers taken directly from `parsed/bopgrasp.jsonl`, normalized image coordinates):

```
GT    center (0.358, 0.257)  left finger base (0.326, 0.239)  right finger base (0.387, 0.274)
      left fingertip (0.324, 0.313)  right fingertip (0.385, 0.339)
pred  center (0.381, 0.428)  left finger base (0.341, 0.428)  right finger base (0.421, 0.428)
      left fingertip (0.301, 0.428)  right fingertip (0.461, 0.428)

gripper width d = |left finger base − right finger base| = 0.0701
five-point mean error = 0.1495       (the scorer also tries the left-right mirror and takes the smaller one)
NCE = 0.1495 / 0.0701 = 2.13     ← archived score = 2.1314 ✓
```

The five predicted points **all have y = 0.428** — the model drew a horizontal gripper, while the GT is slanted.
**On average each keypoint is off by 2.1 gripper widths.** So a larger NCE is worse; 0 is perfect.

`boppose` is the opposite: sample 0 has score = 0.0000 (the eight predicted points shrink into a small cluster, the convex hull does not intersect the GT),
sample 1 has score = 0.7939 (the eight corners basically line up, the convex hulls overlap by about 80%).

> **⚠ Following this definition turned up an error.** `tools/parse_dump.py` uses
> `correct = (score >= threshold)` uniformly for all benchmarks, with `threshold` defaulting to 0.5. That is right for `boppose`'s IoU;
> for `bopgrasp`'s NCE **the sign is completely inverted**. An actual run falsifies it: the sample with the worst NCE on P7
> (sid 31, `score = 10.0`, the model did not even give all 5 points) is judged "correct", and the sample with the best NCE
> (sid 20, `score = 0.4035`, the only one with mean error under half a gripper width) is judged "wrong".
>
> **The correct/wrong counts in the `bopgrasp` row of the §10.2 summary table are unusable** and have been struck out. The **direction labels** of the sign test below must likewise be read inverted:
> "P4 higher 31" = P4's NCE is larger = **P7 is better on these 31 questions**; the mean difference `−0.0778` is an **improvement**, not a regression.
> The test itself is unaffected (p = 0.504, it only counts whether the two sides are symmetric, independent of direction); the conclusion is still **not significant**.
>
> `boppose` is unaffected — IoU and `>= 0.5` point the same way.

**How these numbers are computed** (`analysis/significance_p4_p7.py`, both sides paired by `sample_id`, n = 60):

- **Mean** — arithmetic mean of the `score` field over the 60 questions. The two benchmarks' scores are not in the same units:
  `boppose` measured in 0–0.953 (IoU, higher is better), `bopgrasp` in 0.404–10.0 (NCE, lower is better, 10 is the cap).
  **The two columns point in opposite directions; never compare them across.**
- **Three-way counts** — for each question compute `d = P7.score − P4.score`, split into three piles by sign: `d > 0` counts as P7 higher,
  `d < 0` as P4 higher, `|d| < 1e-12` as bit-for-bit identical. "Bit-for-bit identical" means **identical at the floating-point bit level**,
  not equal after rounding — same tool chain, same deterministic tool, the score comes out exactly the same,
  which is why `boppose` has 15 questions in this pile.
- **p-value** — paired **sign test**: ties by definition do not participate; only the non-tied pairs are used
  (`boppose` 19 + 26 = 45, `bopgrasp` 25 + 31 = 56), and the two-sided binomial probability under the null hypothesis "each side gets half" is
  `2 · P(X ≤ min(pos, neg))`, `X ~ Bin(45, 0.5)`.
  **Both are far above 0.05 — the directional difference is entirely within coin-toss range.**

Why the mean and the three-way counts can contradict each other: **the mean is weighted by magnitude, the sign test only counts direction.** On `boppose` P7 wins fewer questions
(19 < 26) but wins those by a large margin, so the mean is pulled positive.

One last definition difference: "P4 correct / P7 correct" in the §10.2 summary table is not the score here but the `correct` field,
i.e. the `score >= 0.5` threshold decision above. **The two tables count different things** — taking `boppose` as an example,
a question whose IoU drops from 0.51 to 0.49 counts only once as "P4 higher" here, but flips a correct/wrong cell in the summary table;
a drop from 0.90 to 0.60 only shows up here, and the summary table does not move at all.

### 10.3 First structural finding: on 777 samples the two checkpoints agree per sample completely

`refplacement` (100) + `refunseen` (77) + `cvb3ddepth` (600) — **777 samples, not a single disagreement.**
Not the same score, but **the same correct/wrong on every single question**.

P6 already gave the mechanism: on these benchmarks the "reasoning" is a rule that can be written as code, and the model executes it verbatim.
The rule's input is the tools' output, the tools are deterministic, so **whichever policy model is swapped in, the result is the same**.

Walking through `cvb3ddepth` sample 0 (the two sides' `parsed/cvb3ddepth.jsonl` compared verbatim).
Question: the table (red box) and the bookcase (blue box), which is closer to the camera? (A) table (B) bookcase

```
turn 1   depth_estimator.estimate_depth(image_index=0)        → $depth_map
         roborefer locates red box / blue box                  → two pixel coordinates
turn 2   vision_ops.index_at($depth_map, u=0.501, v=0.661)    ← red box (table)
         vision_ops.index_at($depth_map, u=0.260, v=0.406)    ← blue box (bookcase)
turn 3   compare, answer the smaller one                       → A

            red box reading         blue box reading    answer
P4      2.7630178928375244     6.707825660705566        A
P7      2.763768196105957      6.710909366607666        A
```

**The "rule" is just the turn 3 line: answer whichever depth is smaller.** The policy model is only responsible for "laying out these three steps";
the coordinates come from RoboRefer, the values from DepthPro, and **neither of these models is within the scope of RL training**.
The two runs' depths only part ways at the 4th decimal place (GPU floating-point nondeterminism),
while 2.76 and 6.71 are 3.9 meters apart — **this jitter is four orders of magnitude away from flipping the outcome.**

Of the 600 questions, **577 have verbatim identical chain signatures**; of the remaining 23, 21 differ only in turn count
(P4 takes 4 turns, P7 takes 3, tools and call counts exactly the same), and in the other 2 P7 dropped `vlm` and called `roborefer` more.
**These 23 also have identical correct/wrong** — the bit of orchestration freedom C′ won does not reach the verdict by a single step.

**Control: `robospatial` sample 54 — same question, P4 judged wrong, P7 judged correct.**

```
Question: there is a sink in the image. In the vacant space right next to and above the sink, point to a spot where something can be placed.

P4  turn 1  roborefer.detect_one(obj_name =
              "point within the vacant space suitable for placing an object
               close to and to the above the sink")     ← copies the whole question stem in
            → returns (0.389, 0.661)
    turn 2  answers [(0.389, 0.661)]                     ✗ judged wrong

P7  turn 1  roborefer.detect_one(obj_name =
              "point close to and above the sink")       ← compresses the question stem into a phrase itself
            → returns (0.07, 0.754)
    turn 2  answers [(0.07, 0.754)]                      ✓ judged correct
```

Both sides' chain signatures **are `roboreferx1@2t`** — same tool, same single call, same two turns,
and RoboRefer is still the same deterministic RoboRefer. **The only difference is the sentence the model wrote into `obj_name`.**

Put together it is clear:

| | `cvb3ddepth` / `refplacement` / `refunseen` | `robospatial` |
|---|---|---|
| What the tools consume | `$depth_map` + coordinates RoboRefer computes itself, **nothing the model can make up** | A natural-language sentence **phrased by the model itself** |
| What the model's last step does | Compare two numbers | **Copy the point returned by the tool verbatim as the answer** |
| Where the model's freedom lies | Only "how many turns to finish in" | Phrasing → detected point → answer, **passed through all the way to the grader** |
| Result | 600 questions, **0 disagreements** | 350 questions, **60 disagreements**, of which **58 have exactly the same chain signature** |

**This is the precise meaning of "policy freedom"**: not whether the model can switch tools, but **whether the model's free choices have a path to the score**.
On `cvb3ddepth` this path is cut off by deterministic tools — however the model goes around, in the end it compares DepthPro's two numbers;
on `robospatial` it is the **only** path — how the model describes that point decides what score it gets.
**RL can only change the model, so RL is only visible on the second kind.**

> **Implication: these three benchmarks cannot measure any difference between RL algorithms.** They measure RoboRefer and DepthPro,
> not the policy. Putting them in an "is RL effective" comparison table only dilutes the signal.
> **Of the nine benchmarks, the only ones with real policy freedom are `robospatial` (60 disagreements) and `blinkdepth` (10).**

### 10.4 A direct answer to the motivation P6 left for P7

P6 §7 item 1 says:

> The existing policy's tool orchestration has already collapsed heavily …… **This is exactly what GRPO does, and exactly what a distribution-matching objective claims to
> avoid.** P7's motivation went from "should in theory" to "already observed".

**C′ did not reduce the collapse; on seven of the nine it is actually more collapsed.**

| benchmark | n | Main-chain coverage P4 → P7 | Chain signature kinds P4 → P7 |
|---|--:|---|---|
| cvb2drelation | 650 | 94.6% → **98.3%** | 10 → **6** |
| cvb3ddepth | 600 | 96.2% → **99.7%** | 4 → **3** |
| blinkdepth | 124 | 82.3% → **85.5%** | 11 → **9** |
| boppose | 60 | 96.7% → **98.3%** | 3 → **2** |
| robospatial | 350 | 62.3% → **64.9%** | 3 → 3 |
| refunseen | 77 | 100.0% → 100.0% | 1 → 1 |
| reflocation | 100 | 100.0% → 98.0% | 1 → **3** |
| refplacement | 100 | 100.0% → 99.0% | 1 → **2** |
| bopgrasp | 60 | 95.0% → **90.0%** | 2 → 2 |

On the three RefSpatial ones C′ did produce a few extra chains (3 of 277 samples took a different path), and `bopgrasp`'s main chain
also loosened by 5 pp. But on **the four with large sample sizes**, the direction is consistently more concentrated, with fewer signature kinds.

> **The qualifications must come with it:**
>
> **(a) Cross-base, not a controlled A/B.**
>
> **(b) The training amount falls within the paper's Step 4 setting, but outside the range where GFlowRL has been validated.**
> `trainer.total_epochs=1` is hard-coded by the upstream author in `run_rl.sh:435` (commit `1e7ff077`,
> Siyi Chen 2026-03-24), **and we did not change a single character** — that is, we trained as long as the paper's Step 4,
> **so "trained shorter than SpaceTools" cannot be used as an excuse.** But GFlowRL's own evidence comes from runs of **30–400+ steps
> with G=16**; ours is **85 steps + G=5**, and the paper itself states the estimator variance is O(1/G),
> "especially when the group size is small".
> **So the accurate statement is: an algorithm validated elsewhere was put into a regime it has not been validated in,
> and both dimensions, step count and G, are unfavorable at the same time.**
>
> **(c) The training side measured 63–80% group reward degeneracy**, and on those groups the gradient comes entirely from the drift term.
>
> **Taken together, this looks more like "this regime is not enough for the objective to express itself" than "C′'s properties are falsified".**

**But this table only answers half the question.** Per the note in §0: most of the "collapse" on RefSpatial / `boppose` / depth questions
is spurious — **the task has only one correct chain to begin with**; there is no second equally valid chain to preserve.
So what the table above can really be read as is "consistently more concentrated on the four with large sample sizes", **not "failed to preserve diversity"**.

**The one place where it really costs something, answered in §10.6:** the 29 `front/behind` questions of `robospatial`,
which P6 judged as clean 2a (should have called, didn't), upper bound +8 samples, the only place where "a known valid alternative chain exists, yet the policy
never takes it". Measured:

```
front/behind accuracy   P4 72.4%  ->  P7 62.1%      (n=29, 3 fewer)
depth_estimator calls  P4 0 times   ->  P7 0 times       (all 350 robospatial samples)
```

> **Correction (2026-09-25):** both lines above are single evaluations. Under the P0 three-run re-eval these 29 questions show no difference across the three arms
> (SFT 17.3 · P4 19.0 · C′ 19.3), and `depth_estimator` is not strictly 0 either (P4 2/2/1, C′ 0/0/1); see the correction box at the end of §0.
> The sample walkthrough below still holds: both sides are eyeballing depth.

**One sample showing how these 29 questions get answered — `robospatial #235`, P4 correct, P7 wrong:**

> *Is the **monitor** behind the **sofa**?*   **GT = Yes**

```
P4   turn 1  THINK  … "Behind" means further away from the camera viewpoint, deeper into
                    the scene. Let me first identify the locations of both objects …
             CALL   roborefer.detect_one({"obj_name": "monitor"})  RESP  [(0.893, 0.156)]
             CALL   roborefer.detect_one({"obj_name": "sofa"})     RESP  [(0.529, 0.526)]
     turn 2  THINK  … the monitor is on a desk on the right side, and the sofa is in the
                    center of the room. From the camera's perspective, the sofa appears to be
                    in the foreground/middle of the room, while the monitor is on a desk
                    that's positioned f…
             ANSWER Yes    ✓

P7   turn 1  THINK  … "Behind" means further away from the camera viewpoint. Let me detect
                    both objects to analyze their spatial relationship.
             CALL   roborefer.detect_one({"obj_name": "monitor"})  RESP  [(0.893, 0.156)]  ← bit-for-bit identical
             CALL   roborefer.detect_one({"obj_name": "sofa"})     RESP  [(0.529, 0.526)]  ← bit-for-bit identical
     turn 2  THINK  … From the camera's perspective, the monitor is to the right and
                    appears to…
             ANSWER No     ✗
```

**Both sides' tool calls, arguments and return values are bit-for-bit identical, both chain signatures are `roboreferx2@2t`,
and both sides correctly wrote "behind = further from the camera" in turn 1.**
The difference is only in that "appears to" in turn 2 — **they are both eyeballing depth.**

**`depth_estimator.estimate_depth` is right there in `robospatial`'s tool list; one call would turn
"which is farther" into two comparable numbers** (that is exactly what the 600 questions on `cvb3ddepth` do, with 96.5% accuracy).
**These 29 questions, two checkpoints, 58 chances, and not a single call to it.**

**C′ did not bring back the suppressed chain, not even once.** This rejects the motivation more directly than the overall collapse table
— because this spot is unaffected by the "the task has only one path" objection: **here a second, better chain clearly exists,
and neither checkpoint takes it.**

> Per sample, of the 29, P4 gets 21 / P7 gets 18 correct; the flips are **P4 correct→P7 wrong 5** (`#235 #238 #324 #333 #338`),
> **P4 wrong→P7 correct 2** (`#240 #313`), net −3. **The sample is too small to support "C′ is worse";
> all it supports is "neither side learned to call depth".**

**One more: P6 §7 item 2 already measured "room to dig on the reasoning side", and the result was zero** (majority vote@5 over 12 cells,
0 significant, and the VQA direction even flipped sign). **So P7's motivation had only "orchestration diversity" left to begin with,
and that one now reads negative too.**

### 10.5 Variable reuse: P6 taxonomy 2d, the two checkpoints are bit-for-bit identical

| benchmark | | Exposed | Used | **Unused** | Phantom |
|---|---|--:|--:|--:|--:|
| blinkdepth | P4 / P7 | 269 / 275 | 125 / 122 | **146 / 154** | 2 / 1 |
| cvb3ddepth | P4 / P7 | 1203 / 1200 | 600 / 600 | **603 / 600** | 0 / 0 |
| boppose | P4 / P7 | 240 / 240 | 180 / 180 | **60 / 60** | 0 / 0 |
| bopgrasp | P4 / P7 | 240 / 240 | 180 / 180 | **60 / 60** | 0 / 0 |

On `cvb3ddepth`, **1200 variables exposed, exactly 600 used, exactly 600 left** — per sample
`estimate_depth` exposes two, `$depth_map` and `$focal_length_px`, and the model always uses only the former.

> P6 warned that `vars_unused` must first have the constant background subtracted. After subtracting it, **genuine "got it but didn't use it" is nearly zero**,
> and the two checkpoints do not differ. **C′ changed nothing on this dimension.**

### 10.6 Rerunning P6's three criteria

**Criterion A — do the depth questions verbatim follow "pick the one measured closer":**

```
blinkdepth    P4 follows 109 violates 5 undecidable 10   95.6%
              P7 follows 109 violates 5 undecidable 10   95.6%     <- bit-for-bit identical
cvb3ddepth    P4 598 / 1 / 1   99.8%
              P7 598 / 2 / 0   99.7%
```

**P6's core insight holds unchanged on P7** — on depth questions the "reasoning" is a rule the model executes verbatim,
and errors can only come from the numbers the tools give. C′ did not move it by a single percentage point.

**Criterion B — are pointing questions (= the three RefSpatial items, excluding RoboSpatial Vacant) a verbatim pass-through of the tool output:**

```
              P4               P7
reflocation   99/99            94/99
refplacement  100/100          98/100
refunseen     77/77            76/77
total         276/276 = 100%   268/276 = 97.1%
```

P6's recorded "276/276 verbatim pass-through" reproduces on P4. **C′ makes the model stop passing through verbatim on 8 samples** —
it looks like "the policy has started to participate", but the next section shows that on this task, participating means getting worse.

> **Correction (2026-09-25):** P0 same-pool re-eval: SFT starting point 272/276 · P4 276/276 · C′ 272/276 —
> C′ is the same as the starting point; it is GRPO that pushed pass-through to 276, not C′ that lowered it (268 is a single-run reading). See the §0 correction box.

**robospatial VQA by question type (P6 §6.4's three-way split):**

```
question type                n     P4 accuracy   P7 accuracy    P4 answers yes   P7 answers yes   GT=yes
fit (needs free space)        105     69.5%      69.5%      83/105     77/105      87
relation (2D-decidable)        94     77.7%      77.7%      69/94      73/94       60
front/behind (needs depth order)     29     72.4%      62.1%      18/29      13/29       18
```

- **On `fit` and `relation`, the two checkpoints' accuracies are bit-for-bit identical.** The biggest gap P6 located
  — "`fit` questions almost always get answered yes" — C′ lowered yes from 83 to 77, **and accuracy did not move at all**.
  **The 6 fewer yes answers did not win a single question.**
- **The only thing that moved is `front/behind`: 72.4% → 62.1%** (3 fewer on n=29). P6 judged this class as
  **clean 2a (should have called, didn't)** — it asks exactly for depth order, the tool gives it directly, and on 29/29 it never called
  `depth_estimator` at all.

**Across all 350 `robospatial` samples, `depth_estimator` was called 0 times — 0 for both P4 and P7.**
(The SFT starting point: 1 time.) **The structural gap P6 §6.4 recorded was not fixed by GRPO, nor by C′.**

> **Correction (2026-09-25):** the `front/behind` row in the table above is a single evaluation; P0 three-run re-eval SFT 17.3 · P4 19.0 · C′ 19.3 / 29,
> **no movement**. "The only thing that moved is `front/behind`" should read "none of the three question types moved measurably".
> `depth_estimator` over three runs: SFT 0/0/0 (not 1 time) · P4 2/2/1 · C′ 0/0/1. The gap itself holds.

### 10.7 The RoboSpatial gap: a causal chain that can be worked out

**The gap is entirely in Vacant; VQA is noise.**

```
           n     P4          P7 first run      P7 second run
VQA       228   167 73.25%   164 71.93%    166 72.81%
Vacant    122    62 50.82%    51 41.80%     54 44.26%

lost (P4 correct / P7 wrong both runs) 29: VQA 16 · Vacant 13
gained (P7 correct both runs / P4 wrong) 17: VQA 14 · Vacant  3
net                              : VQA −2 · Vacant −10
```

The answer directions of the 30 disagreeing VQA samples are **symmetric**: of the 16 lost, 12 are `gt=Yes` flipped from yes to no and
4 are `gt=No` flipped from no to yes; of the 14 gained, 10 are `gt=Yes` flipped from no to yes and
4 are `gt=No` flipped from yes to no. **No systematic direction; it is just flip noise.**

**The −10 questions on Vacant can be worked out to the end.** P6 recorded: `robospatial` Vacant **is not pure pass-through, and whenever the model
changes the point, it gets worse**. Running this criterion on both sides:

```
              pass-through             point override           improved/worsened    original-point median dist -> after
P4       101 (correct 55.4%)   21 (correct 28.6%)   10 / 11     0.0716 -> 0.0483
P7        79 (correct 57.0%)   42 (correct 14.3%)   23 / 19     0.0942 -> 0.0855
```

**C′ doubles the samples that "change the point roborefer gave" from 21 to 42, and the point-override accuracy is only 14.3%
— pass-through is 57.0%.**

**One sample says it all — `robospatial #34`, P4 correct, P7 wrong in both runs:**

> *In the image, there is a **microwave**. Pinpoint one point within the vacant space suitable
> for placing an object **close to and in front of** the microwave.*
>
> **GT is a narrow band of 15 points: x 0.601 – 0.862 · y 0.408 – 0.452** (the countertop in front of the microwave)

```
P4   turn 1  THINK  I need to find a point close to and in front of the microwave …
             CALL   roborefer.detect_one({"obj_name":
                      "point close to and in front of the microwave"})   ← whole question stuffed into the query
             RESP   [(0.722, 0.432)]
     turn 2  THINK  The tool detected a point at (0.722, 0.432) which is close to and in front
                    of the microwave … I can now format this point as … the final answer.
             ANSWER [(0.722, 0.432)]     ← verbatim pass-through; lands in the middle of the GT band   ✓

P7   turn 1  THINK  … To find a suitable vacant space "in front of" the microwave, I need to
                    consider the floor space between the microwave and the camera/viewer …
             CALL   roborefer.detect_one({"obj_name": "microwave"})      ← only asks where the object is
             RESP   [(0.758, 0.348)]                                     ← this is the microwave itself
     turn 2  THINK  … Now I need to find a vacant space that is: 1. Close to the microwave
                    2. In front of the microwave (from the camera's perspective, meaning
                    between the microwave and the camera) Looking at the image, the floor
                    space in front of the microwave …
             ANSWER [(0.77, 0.80)]       ← a point it derived itself             ✗
```

**Two changes, stacked:**

```
① query changed   P4 hands the whole question text to RoboRefer and lets the tool return "that point" directly
             P7 only asks "where is the microwave" and keeps "what in front of means" for itself
② computes itself     P7 interprets in front of as "the floor between the microwave and the camera", so it pushes y from 0.348
             all the way to 0.80 — while the GT's y only goes up to 0.452. The x it gives, 0.77, actually falls within the GT range.
             Both coordinates are also multiples of 0.05 (the "eyeballing" fingerprint P6 quantified)
```

**P7's reasoning itself reads as more thorough** — it worked out that "in front of is relative to the camera", and specifically went looking for the floor.
**But it got the scene wrong** (the microwave is on a countertop, and the GT points at the countertop, not the floor), while P4's lazy approach of "handing the question to the tool as is"
turned out right.

> **This is what the §10.7 causal chain looks like on a single sample: C′ makes the policy more willing to take over the last step itself,
> and the information for that step is not in its hands — only the tool has it.**

Check and counterfactual:

```
P4  101×0.554 + 21×0.286 = 56 + 6 = 62 = 50.82%   ✓
P7   79×0.570 + 42×0.143 = 45 + 6 = 51 = 41.80%   ✓
If P7 kept P4's pass-through/point-override ratio (101/21), at its own per-class accuracy:
     101×0.570 + 21×0.143 = 57.6 + 3.0 ≈ 61 questions = 50.0%   <- right back at P4's level
```

> **So the whole RoboSpatial gap can be attributed, with nothing left over, to one behavior change: C′ makes the policy more often
> "improve" the point the tool gave, and P6 has already quantified this behavior as harmful.**
>
> This and the drop in pass-through rate under criterion B (276/276 → 268/276) are the same thing showing up on two benchmarks:
> **C′ does make the policy "more proactive", but on this tool chain, proactive means worse** — the task's information bottleneck is in the tools,
> and the policy holds no information the tools did not give it.

> **Correction (2026-09-25): the causal direction in this section is reversed; see the §0 correction box.** "21 → 42" compares P4 and C′,
> not the starting point and C′ — the SFT starting point already overrides 49.3 times, C′ 44.0, P4 21.7. The counterfactual check above is an accounting identity
> and says nothing about what is the cause. Broken down with `p1_monitor.py --upper`: point overrides mostly happen when the tool's point is wrong, and forcing pass-through actually loses score;
> P4's advantage is that the tool point itself is more accurate (60.3 vs 49.3 / 122). Sample #34 still holds, but what it shows is
> "whether the query is phrased well", not "a proactive policy gets worse".

### 10.8 Error attribution: P6's "322 wrong answers 100% clean" conclusion holds on P7 as well

```
                    P4 error attribution              P7 error attribution
robospatial         clean 121                 clean 134 · no tool called 1
reflocation         clean 46                  clean 48
refplacement        clean 42                  clean 42
refunseen           clean 40                  clean 40
blinkdepth          clean 17                  clean 15
cvb2drelation       clean 35                  clean 35
cvb3ddepth          clean 21                  clean 21
boppose             clean 22                  clean 21
bopgrasp            tool failure 5 · clean 1      tool failure 4
```

OOM, generation truncation, tool response truncation, max turns, malformed tool_call, answer parse failure — **all 0 on both sides.**

> **This is what makes all the causal explanations above clean**: not the environment, not GPU memory, not truncation.

### 10.9 Relation to the P5 conclusions

All of P5's definition conclusions still hold on P7 and need no re-argument:

- **Of the ten Table 2 numbers only nine are independent** (Overall is the sample-weighted VQA/Vacant).
- **Decoding is greedy**; `rollout.n=5` is the group size used in training and does not apply to evaluation.
- **`gpu_memory_utilization` is a fraction of the whole GPU** — P5 §4.5 lists it as a "new deviation"; P7 back-computes it from GPU capacity as
  0.545 to align the pool back to 24 GB, same definition.
- **blinkdepth should report the stable upper limit with an interval** (P4 three runs 86.29–87.90, upper limit 112/124 = 90.32).
  **P7's single-run 87.90 is exactly the top of P4's interval, not "higher than P4".**
- **The metric mapping for BOP-ASK Pose is still unresolved** (the paper's 34.37 is not a proportion at n=60:
  `60 × 0.3437` is not an integer, while the same test holds for all six other proportion-type numbers).
  **The difference between P7's 55.73 and P4's 53.36 cannot be taken as a capability difference.**

P5 §3.1 recorded "RoboSpatial VQA −6.13 pp is the only gap above noise" — on VQA P7 differs from P4 by only
1–3 questions; **P7 neither widened nor narrowed this gap; it is inherited.**

**Sources:** `06_gflowrl_eval/comparison_p4_p5_p6_vs_p7.md` ·
`eval/GFlowRL/P7_GFLOWRL_EVAL/analysis/` (four scripts + full output) ·
`spacetools-repro/records/` (all P2–P7 reports) · `spacetools-repro/p4/` (dumps / parsed / logs)

---

## 11. Conclusion

**1. No benchmark got better, and the opposite direction is equally non-significant.** The smallest McNemar exact-test
p across the nine benchmarks is 0.092. **Not one piece of evidence supports "P7 is better than P4 on some benchmark"**,
and the same goes for "worse than P4".

**2. Compared with its own SFT starting point, C′ produced no measurable improvement wherever a before/after comparison is possible.**
RoboSpatial differs by 4 questions (1.5 σ), RefSpatial weighted is bit-for-bit identical, Vacant is exactly the same as the starting point.
Not a single one of the roughly 11 questions the official checkpoint leads by was caught up.

**3. This is consistent with the training-side metrics, not a surprise.** Degenerate groups 70.3%, `grad_norm` median 180 while
`grad_clip=1.0`, 1 epoch 85 steps (hard-coded upstream), G=5 (the paper uses 16).
**An objective that receives no reward signal on about 70% of samples, and keeps only the direction on every update, being indistinguishable from the starting point
after 85 steps is the expected result.** Note that the 85 steps here are **not us cutting corners** — the paper's 70.0 was trained exactly
this way; the real problem is that GFlowRL's evidence comes from runs of 30–400+ steps with G=16,
**and we put it into a regime it has not been validated in.**

**4. The motivation P6 left for P7 (distribution matching can avoid orchestration collapse) measured negative this time, and negative under both definitions.**
On overall collapse, the four benchmarks with large sample sizes are consistently more concentrated (this one should be discounted, because for RefSpatial etc.
most of the "collapse" is the task having only one path); **and at the one spot unaffected by that objection — the 29 `front/behind`
questions, `depth_estimator` called 0 times throughout — C′ did not bring it back even once.**
(2026-09-25 correction: under the three-run re-eval C′ called it 1 time and P4 5 times, and accuracy on these 29 questions does not differ across the three arms;
the direction of the conclusion is unchanged — neither side learned to call depth.)
The reading is weakened by three things (cross-base, 85 steps, G=5), so what it falsifies is "C′ can achieve this in this regime",
**not that C′'s theoretical properties fail.**

**5. The only measurable behavior change from C′ is harmful, and it can be worked out to the end.** Pass-through 276/276 → 268/276;
Vacant point overrides 21 → 42 (point-override accuracy 14.3% vs pass-through 57.0%); RoboSpatial's 14-question gap
is attributed to this behavior with nothing left over. **On this tool chain, a "more proactive" policy means worse.**
(2026-09-25 correction: the direction is reversed, see the §0 correction box — the starting point already overrides 49.3 times with pass-through 272/276, C′ is the same as the starting point,
and it is GRPO that suppressed it; P4's advantage comes mainly from better tool calls, and forcing pass-through actually loses score.)

**6. C′ did not move any of the three structural gaps P6 located** — `fit` questions 69.5% → 69.5%, bit-for-bit identical;
`robospatial` 0 `depth_estimator` calls throughout (P4 also 0; three-run re-eval P4 2/2/1, C′ 0/0/1); depth-question rule-following rate bit-for-bit identical.
**P6 judged all three to be toolset gaps or should-have-called-didn't; changing the objective cannot move them, which is in line with P6's expectation.**

**7. Of the nine benchmarks only two can measure a policy difference.** On 777 samples the two checkpoints are literally identical.
**When designing RL comparison experiments in the future, compute should go only to `robospatial` and `blinkdepth`.**

### Methodology worth taking away

- **Fix the gates first.** Criteria, pre-registration, separation of duties ("is the loss written correctly" is guarded separately by a synthetic fixed-point test).
- **The reading order must not be reversed: count OOM first → then check for silent tool errors → only then look at the score.** This time it really caught
  a silent contamination (230 of 243 wrong on cvb3ddepth attributed to OOM), and what it produced was a "plausible-looking" number.
- **Criteria must not rely on exit codes alone.** One script does `exit 0` on failure, another `exit 15` on success; we hit both.
- **Keeping the dumps is the only reason conclusions could be drawn this time.** P4 stored dumps back then, so a per-sample comparison is possible today;
  the SFT eval did not, so the truly controlled comparison cannot be done.
- **Write criteria as intervals, not point thresholds.** SFT's `RoboSpatial ≥ 60` falls right inside the measured flip band,
  so whether a single reading lands above or below the line is decided by noise.

---

## 12. Limitations and next steps

### 12.1 Limitations (by severity)

1. **Cross-base, not a controlled A/B.** P4 is the paper's base + GRPO, P7 is self-trained SFT + C′.
   Strictly, every "C′ caused X" can only be stated as "the P7 checkpoint exhibits X".
2. **The SFT starting point has no dumps, so the controlled SFT ↔ C′ per-sample comparison cannot be done.** Only summary numbers can be compared
   (robospatial main chain SFT 64.3% → C′ 64.9%, calls 1.649 → 1.646, showing the collapse was already like this
   at the starting point) — but those are aggregates, not per-sample evidence.
3. **Single arm, no GRPO control run.** The paper's GRPO arm has not been reproduced on our base.
4. **Single seed, and in P7 only robospatial has two runs.** P4 ran three benchmarks several times and reported intervals.
5. **The algorithm was put into a regime it has not been validated in, with two dimensions unfavorable at the same time.**
   - **Steps: 1 epoch / 85 steps.** `trainer.total_epochs=1` is hard-coded by the upstream author in
     `run_rl.sh:435` (commit `1e7ff077`), and we did not touch it — **the paper's 70.0 was trained exactly this way,
     so the excuse "we trained shorter than SpaceTools" does not hold.** But GFlowRL's evidence comes from runs of
     **30–400+ steps**.
   - **G: `rollout.n = 5` while GFlowRL uses 16.** The paper states the estimator variance is O(1/G), and
     the first limitation in Appendix A is exactly "especially when the group size is small" —
     **5 versus 16 is 3.2× the variance; we stand exactly on the most unfavorable side.** Changing it would make the two arms incomparable,
     so it was not changed; but it means **this result cannot be extrapolated to G=16**.
   - Combined: **this is "an algorithm validated elsewhere, put into a regime it has not been validated in",
     not "not enough training".** A real test of C′ needs G=16 or longer runs,
     and that is no longer reproducing SpaceTools' Step 4.
7. **Mixed ECC leaves the KV pool uncontrolled** (depending on which GPU the policy lands on, the pool differs by about 1.7 GB); this is an
   uncontrolled explanation for the 4-question difference in §9.6.
8. **The reward specs of `bopgrasp` / `boppose` are themselves problematic** (NCE is insensitive to gripper orientation); they have been explicitly excluded from
   the scope, but they still appear in the results tables, so keep that in mind when reading.
9. **The attribution criteria reuse P6's implementation**, including the boundaries P6 itself recorded (3b attribution is disputed; the 31 manual ones have only
   one round of annotation, with no inter-annotator agreement measure).
10. **The two sides of the P4 ↔ P7 per-sample comparison had different KV pools to begin with.** P4 is 40 GB GPU × `gmu=0.5` =
   **20 GB pool**; P7 is 46 GB GPU × `gmu=0.545` ≈ **24 GB pool** — the latter is aligned to the 24 GB of the **SFT baseline**
   run (§9.2), not to P4. The two sides of the §10 summary table differ by 4–5 GB.
   At the magnitude of the §9.2 ablation (pool differs 2× → `blinkdepth` flips 8 questions, net 2), the effect of 4–5 GB
   is most likely smaller than the noise in the table, **but it is a real uncontrolled difference that had not been written down before**,
   and it is the other side of the same thing as item 7.
   > The three pool definitions side by side, so they don't get mixed up again:
   > `P4 20 GB` · `SFT baseline 24 GB` · `P7 ~24 GB (aligned to SFT, not P4)`.
   > **Getting all three on the same pool requires a re-eval** — the same GPU session as §12.2 item 1.

### 12.2 Next steps, ranked by cost-effectiveness

| # | What | Cost | What it can answer |
|---|---|---|---|
| 1 | **Evaluate the SFT starting point on robospatial on the same machine, same session** | ~35 min | Turns the 4-question difference in §9.6 from "machine/model inseparable" into a controlled reading. **The most cost-effective step** |
| 2 | **Run robospatial and blinkdepth 3 times on each side** | ~3 h | Upgrades "none significant" into an interval conclusion with power |
| 3 | **Evaluate step 30 / 60** | ~1.5 h | Confirms that late training did not grind away early gains. **Note this is post-hoc checkpoint picking; the report must state how many were evaluated** |
| 4 | **Run a GRPO control arm** (same base, same seed, same machine) | 3–4 days of machine time | The only way to answer "C′ vs GRPO" |
| 5 | **Raise G from 5 to 16 and train again** | 3–4 days of machine time | Does the paper's estimator-variance claim hold in our setting |
| 6 | **Separate reward shaping for the 29 `front/behind` questions** | Medium | The only place P6 located where "a known valid alternative chain exists, yet the policy never takes it", upper bound +8 questions (about +10 questions under the P0 three-run mean; the SFT data has 0 instances of this chain, so the support set must be filled in first, see Design Doc P3) |

| 7 | **Store top-k logprobs during eval** (dump patch, the shape of `patches/rl/0009`) | Small | The §9.2 chain "batch composition → reduction order → tie flip" can currently only be inferred backwards from outcomes; with logprobs stored, the gap between the two candidate tokens at the branch point can be read directly, pinning down the causality |

> **Not recommended:** switching tools (an orthogonal path; mixed in, it cannot be attributed), tuning prompts on `fit` questions
> (P6 judged it a toolset gap — what the model lacks is information, not distribution).

---

## Appendix A: Artifacts list

```
training/SFT/
  env/2x_A6000_REPORT.md              environment build (stages A–C) + 30-step probe
  env/training_report/REPORT.md        4× A6000 full SFT execution report
  env/bundle/                          4.0 GB portable environment package + RESTORE.sh
  model_rl_start/                      the SFT checkpoint itself (π_ref)
  SFT训练学习笔记.md                    config sources, GPU memory budget, ZeRO trade-offs

training/GFlowRL/
  A40/P7_GFLOWRL_训练报告.md            training report
  A40/metrics/p7_metrics.csv           85 rows, one per step
  A40/logs/ · A40/config/ · A40/scripts/   as-run logs / environment snapshot / scripts
  diff/*.patch                         three patches = all code changes
  docs/                                Route C gate / plan / step 23 results

eval/GFlowRL/
  P7_GFLOWRL_EVAL报告.md               eval report
  P4_P5_P6_vs_P7_对比分析.md            per-sample comparison
  P7_GFLOWRL_EVAL/
    dumps/<run>/<benchmark>/0.jsonl    full trajectories (including the real tool returns), both correct and wrong
    parsed/<benchmark>.jsonl           structured records, same schema as p4/parsed
    parsed_runC/robospatial.jsonl      robospatial second run
    logs/ · gates/ · scripts/ · gpu/ · config/ · MANIFEST.txt
    analysis/compare_p4_p7.py          per-sample pairing
    analysis/robospatial_divergence.py direction and question type of disagreeing samples
    analysis/p6_criteria_p4_vs_p7.py   P6 criteria A/B, question-type three-way split, Vacant pass-through-vs-point-override
    analysis/significance_p4_p7.py     McNemar + sign test

spacetools-repro/                      GitHub: qyYue1389/spacetools-repro
  records/                             29 reports (P2–P7) + CHANGES.md + PROVENANCE.txt
  p4/dumps/run1..4/ · p4/parsed/ · p4/logs/     official checkpoint trajectories and structured records
  p6/                                  ablations and probes (fp32 / gmu025 / passk / swap / probes / manual)
  tools/parse_dump.py · tools/p5/ · tools/p6/ · tools/p7/
```

## Appendix B: Entry points for recomputation

```bash
# regenerate parsed from dumps (no GPU needed)
python3 spacetools-repro/tools/parse_dump.py \
    eval/GFlowRL/P7_GFLOWRL_EVAL/dumps/runB-rerun4/cvb3ddepth/0.jsonl \
    --emit eval/GFlowRL/P7_GFLOWRL_EVAL/parsed

# rerun all comparison analyses (no GPU needed)
cd eval/GFlowRL/P7_GFLOWRL_EVAL/analysis
python3 compare_p4_p7.py            # per-sample pairing + summary table
python3 p6_criteria_p4_vs_p7.py     # P6 criteria A/B + question types + Vacant
python3 significance_p4_p7.py       # McNemar + sign test
python3 robospatial_divergence.py   # disagreeing samples

# bare machine -> scores for the nine benchmarks (needs 4 sm_80/86/89 GPUs with ≥40 GB)
export HF_TOKEN=...
P7_STEP=85 bash eval/GFlowRL/P7_GFLOWRL_EVAL/scripts/EVAL_FROM_SCRATCH.sh
```

**Definition (must accompany any cited score):**

```
gpu_memory_utilization=0.545 (KV pool aligned to 24 GB) · NUM_GPUS=4 EVAL_GPUS=1
4× RTX A6000 · vlm num_gpus=1.0 · greedy decoding
ckpt: qzpm55555/spacetools-p7-gflowrl-cprime-8xa40 @ global_step_85
env:  qzpm55555/spacetools-eval-env
data: siyich/spacetools-eval-benchmarks @ 1d539ac9
```
