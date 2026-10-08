# Project paper notes (2 papers)

> Purpose: read these notes at the start of a new session to pick up the context without rereading the PDFs.
> Covers: `2512.04069v2.pdf` (SpaceTools / DIRL), `2607.13394v1.pdf` (GFlowRL).
> Recorded: 2026-08-24 (includes a survey of the SpaceTools codebase)

---

## 1. SpaceTools: Tool-Augmented Spatial Reasoning via Double Interactive RL
arXiv 2512.04069v2 · NVIDIA + University of Michigan · revised 2026-06 · **accepted at CVPR 2026**

### Problem
VLMs are decent at qualitative visual understanding, but lack the **metric-precision spatial reasoning** that embodied applications need (distance, pose, grasping, occlusion, 3D relations).
Neither existing route is satisfactory:
- SFT on task data → needs large-scale annotation, and almost every new perception capability (depth, pointing, 3D) requires changing the architecture or collecting new data;
- Have the VLM call vision tools → but existing approaches either rely on hand-written prompts or hard-code a fixed tool pipeline, so the model cannot discover the optimal tool combination by itself.

ViGoRL showed that RL can learn to use a **single** vision tool (cropping), but with many tools the action space explodes combinatorially and naive RL exploration simply fails.

### Method: DIRL (Double Interactive RL)
Core insight: **single-tool IRL converges and can teach grounding; multi-tool IRL can refine reasoning but needs a good initialization**. So RL is used twice:

1. **Teaching phase**
   - First use IRL to train a single-tool expert that only uses the pointing tool (RoboRefer) → generate 2k grounded reasoning trajectories;
   - Then have a frontier model (**Claude Sonnet 4.5**) solve problems with the full tool set, keeping only trajectories with correct answers → 6k;
   - Mix the two 1:3 into an 8k teaching set and SFT the base model on it (learns tool signatures, output format, information flow).
2. **Exploration phase**
   - Continue IRL (GRPO + KL) from the SFT weights with all tools open, refining chained tool calls and error-correction strategies.

Dialogue format: `<think>` / `<tool_call>` / `<answer>`, multi-turn until an answer is given or T_max is reached.

**Reward design** (all normalized to [0,1]): multiple choice is binary; 2D bbox uses MIoU; pointing uses NNDC (normalized negative distance to the centroid of the target region, clipped by taking the max with a binary term); pose uses the convex-hull IoU of the 8 projected corner points; grasping uses NNCE. A format reward was tried, gave no gain, and was not used in the end.

### Toolshed (systems contribution, open-sourced)
A Ray-based distributed tool-hosting platform that solves the engineering bottleneck of "calling heavy CV tools online during training":
- Execution decoupled from the policy inference loop (so one blocking call doesn't stall the whole batch);
- Multiple async parallel actors per tool, resource isolation, **Python environment isolation** (solves dependency conflicts between multiple CV models);
- Supports passing text/images/structured variables (point clouds etc.) across nodes; fits naturally with VERL's async multi-turn rollout.
- Tools: SAM2 segmentation, two pointing tools point1 (RoboRefer)/point2 (Molmo), DepthPro depth + point cloud, 3D bbox fitting, GraspGen grasping, image_ops, code_executor; robot tools: capture_image / capture_depth / execute_grasp / place_object (plus mock versions for training).
- Efficiency: 3.2× faster than naive HTTP deployment at 8 concurrent calls; end-to-end pipeline latency 20.2s → 10.6s.

### Experimental results
Base model = **Qwen2.5-VL-3B-Instruct** (only the LLM part is trained, 2.55B trainable parameters; vision encoder frozen).

- Spatial reasoning benchmarks (RoboSpatial-Home / BLINK / RefSpatial / CVBench / BOP-ASK): SOTA on almost all of them:
  RoboSpatial overall **70.0** (beats Gemini-ER 1.5 by +7.5); Pose **34.37** (beats Claude Sonnet 4.5 by +24.4); Grasp-SR **50.0** (beats GPT-5 by +8.3).
- Tool-free controls with the same 8k data and same base model: **+12%** over tool-free SFT, **+16%** over tool-free RL (RoboSpatial).
- Real-robot manipulation (Kinova Jaco + ZED2 + CuRobo, the robot itself as a tool): Pick 86%, Relational Pick 83%, Pick&Place 86%, all better than Claude Sonnet 4.5 + Toolshed and GPT-5 + Toolshed; TTFM 10s (theirs 30–36s). π0.5 all 0.
- Ablations (Table 4): removing the IRL teacher → 52.48→41.68; removing the universal teacher → 42.86 (pose collapses to 8.92); removing Stage-2 IRL → 50.99. Tool SFT / Tool NIRL are 13.4 / 14.4 points lower respectively. **Doing IRL directly on all tasks with all tools → 19.79, practically unlearnable**.
- Interesting generalization: a model trained with IRL using the pointing tool only on RoboSpatial gets 34.3% on unseen RefSpatial, while all other finetuning methods get 0.
- Zero-shot, giving GPT-5 / Claude Toolshed: tasks that need precise geometry (RefSpatial, pose) go up clearly, but high-level tasks like RoboSpatial/BLINK actually drop slightly — the models **over-call tools and misread tool outputs**.

### Limitations
Only short/medium-horizon tasks; tool outputs are mainly text/structured variables, image-type tool outputs are not yet fully used; training uses a mock robot (real robot in the loop is too slow); grasping is still the weakest (23 of the 30/60 failures are tool errors); the bottleneck for grasp/pose is object detection in cluttered scenes.

### Codebase (checked 2026-08-24)
Main repo <https://github.com/spacetools/SpaceTools> · project page <https://spacetools.github.io/> · 34 stars / 1 fork at the time.
It is a **submodule aggregation repo**: three submodules + environment setup scripts:

| Component | Location |
|---|---|
| Toolshed (tool hosting) | `NVlabs/SpaceTools-Toolshed` |
| SFT (LLaMA-Factory fork) | `ChicyChen/SpaceTools-SFT` |
| RL + eval (verl fork) | `ChicyChen/SpaceTools-RL` |
| SFT dataset | HF `siyich/spacetools-sft` (~7900) |
| RL data (point tools) | HF `siyich/spacetools-rlpointtools` (~4000) |
| RL data (full tools) | HF `siyich/spacetools-rlfulltools` (~5500) |
| Eval benchmarks | HF `siyich/spacetools-eval-benchmarks` |
| **Pretrained checkpoint** | HF `siyich/spacetools-ckpt` |

**Four-step pipeline** (the README splits the paper's two phases into 4 steps): Step 1 point-tool RL (GRPO, only `detect_one`) → Step 2 teacher data collection (Claude + Toolshed) → Step 3 SFT → Step 4 full-tool RL (GRPO, 11–17 tools online).
**The outputs of Steps 1–2 are already published in the SFT dataset; most people can start directly from Step 3.**

Reproduction cost: recommended **2 nodes × 8×A100-80G** (one node runs the Toolshed tool actors, one node trains); Step 1 ~10–15h (15 epochs), SFT ~3–4h (single node 8 GPUs, 3000 steps), Step 4 ~8–12h (1 epoch ≈ 86 steps, 5–7 minutes per step). ~100GB storage. All scripts support automatic resume (designed for short SLURM wall clocks).
Two tool weights must be downloaded manually: `Zhoues/RoboRefer-8B-SFT`, Apple `depth_pro.pt`.

**Implementation details beyond the paper**:
- Two tool configs: `toolshed_v1_config.yaml` = 11 tools (pure reasoning), `toolshed_v2_config.yaml` = 17 tools (including robot). SFT also comes in v1/v2 versions.
- The RL side is verl's new agent loop: `AgentLoopManager → ToolAgentLoop`, inference uses **sglang**, tool-call parsing uses the **hermes** format. This corresponds to the passage in the paper's appendix saying "the new verl agent loop alleviated training instability".
- SFT explicitly freezes the vision tower and trains only the language model; the tool schema is injected into the system prompt.
- 9 eval benchmark keys: `robospatial` / `reflocation` / `refplacement` / `refunseen` / `blinkdepth` / `cvb2drelation` / `cvb3ddepth` / `boppose` / `bopgrasp`.
- Six conda environments (sft / rl / tool-roborefer / tool-vlm / tool-bbox / tool-graspgen), all on Python 3.11 + Ray 2.47.1 for cross-environment compatibility.
- The authors state they are **upstreaming** Toolshed's multi-turn tool training into LLaMA-Factory and verl, with the goal of eventually not needing forks.

**Toolshed on its own** (the most reusable piece): subclass `BaseTool` + `@tool_method` decorator + Google-style docstrings (automatically converted into an LLM schema), returning `ToolResult` (value / text / images / variables). Three ways to launch: `toolshed-launch --config`, Python `start_toolkit()`, `web_ui.py`. Requires Linux + Python 3.11 + CUDA 11.8+; vision tools need ≥40GB VRAM. **The README has no explicit license statement** (none seen in either the main repo or Toolshed); confirm the license before using it. The docs also warn that code_executor **has no sandbox**, so executing generated code is a security risk.

---

## 2. GFlowRL: Scaling Distribution-Matching RL to Large Language Models
arXiv 2607.13394v1 · Microsoft Research · 2026-07

### Problem
**Reward-maximizing** objectives such as GRPO/PPO push probability mass onto a single high-reward mode, causing mode collapse and loss of solution diversity. GFlowNets offer another route: **sample in proportion to reward** (distribution matching), which naturally preserves multiple high-reward reasoning paths.
But existing GFlowNet-style LLM post-training (FOR, FlowRL) has to learn a prompt-conditioned partition function Z_φ (FlowRL uses a 3-layer MLP on the prompt's last hidden state), which blows up at real post-training scale.

### Root-cause diagnosis: the learned partition function is the culprit
**Learning-timescale mismatch** — the policy is a pretrained multi-billion-parameter model that only needs a few hundred steps of finetuning; Z_φ is randomly initialized and has to learn a complex quantity from scratch, also in only a few hundred steps. So for most of training log Z_φ is basically a noise function.

Two pieces of evidence:
1. **It is useless**: replacing Z_φ with random samples from `N(0.5, 1)` doesn't hurt performance; it even slightly improves it (36.19 vs 35.61). In a synthetic three-mode Gaussian experiment FlowRL and FlowRL-RandomLogZ behave almost identically, and neither can fit the multimodal structure.
2. **It is harmful**: over 421 steps FlowRL has **55 steps with gradient norm ≥ 1e6**, mean 3.2e14, max 9.6e16; GRPO / GFlowRL have means of only 0.24 / 0.07, max <6.2.

### Method: GFlowRL
Replace the learned log Z_φ(x) with a **within-batch Monte Carlo estimate** — GRPO-style training samples G rollouts per prompt anyway:

```
Z_t(x) = (1/G) Σ_i [ β·r(x,y_i) + log π_ref(y_i|x) − log π_old(y_i|x) ]
```

This uses a property of the TB optimum: every trajectory implies the same target value. Z_t enters the residual as a **stop-gradient baseline** carrying no gradient, so the auxiliary network, its optimizer state and its distributed sync all disappear.

Two stabilizers:
- **Importance sampling correction** for the drift between the rollout policy and the trainer policy, `w = min(π_θ/π_old, 1+ε)`;
- **Asymmetric flow-gap clipping**: clip the flow gap g evaluated under the rollout policy to `[−ε_low, +ε_high]`, ε_low < ε_high (on math tasks correct solutions are undersampled early, so the push upward should be stronger than the push downward).
- In addition, log probabilities are length-normalized by response length, so long sequences don't dominate the loss.

Theory (Appendix B): when the clip is inactive and there is no length normalization, the self-consistent zero-loss fixed point satisfies `π_θ ∝ π_ref · exp(β r)`, i.e. TB's target fixed point is preserved; at the fixed point flow gap = 0, so the clip naturally becomes inactive in the neighborhood of the optimum and does not change the stationary distribution. Length normalization introduces a slight length-dependent bias (negligible when rollout lengths are similar); the authors position it as an engineering trade-off.

### Experimental results
- **7B math** (Qwen2.5-7B, 6 benchmarks, Avg@16): GFlowRL **40.92**, vs GRPO 32.48 (+8.44), FlowRL 35.63 (+5.29). Also best at 32B (50.42 vs FlowRL 48.39).
- **Code** (DeepSeek-R1-Distill-Qwen-7B): LiveCodeBench 38.62, Codeforces 1646 Elo / 88.0 percentile, HumanEval+ 84.93; wins all three.
- **14B Codeforces: 2048 Elo**, beats DeepCoder-14B by +112, FlowRL-14B by +144, OpenAI o1 by +157, only 25 points behind o3-mini (2073). Per-problem comparison with DeepCoder over 408 problems: GFlowRL uniquely solves 108, DeepCoder uniquely solves 2.
- **Red-teaming attacks** (SEMA setting, sparser and noisier rewards): AdvBench ASR@1 average **82.5%**, HarmBench **79.5%**, beating the previous SOTA multi-turn attack SEMA by +2.4 / +4.5; **FlowRL simply does not converge in this setting**.
- **MoE**: Qwen3-30B-A3B math 78.32 (backbone 74.52, GRPO 75.78), Codeforces 1999 Elo (only 3B active parameters, beats o1 by +108); Qwen3-235B-A22B reuses the 30B hyperparameters directly and trains only 30 steps (GRPO trained 100 steps), still getting 83.35 vs GRPO 82.40. **FlowRL does not converge in any MoE setting**. The authors claim this is the first GFlowNet-style RL algorithm that trains stably on both dense and sparse architectures.
- Ablations: β is insensitive within [1,10], peak at β=8; removing flow-gap clipping → drops 3.9 points, mean gradient up 6.3×, max nearly 3×. Diversity score (GPT-o4-mini, 1–5): GRPO 1.21 / PPO 1.15 / FlowRL 2.64 / **GFlowRL 3.93**.
- Another control: ConstantLogZ gets only 25.34 (backbone 23.02), FOR 29.53 — showing that **how log Z is estimated is the key**, not simply removing it.

### Limitations
The within-batch MC estimate has high variance when the group size is small (held down by the two stabilizers); only validated on three language domains (math/code/red-teaming); whether it generalizes to agentic or multimodal RL is unknown.

Code is planned to be open-sourced at <https://github.com/microsoft/gflowrl> — **checked 2026-08-24, still 404, not yet released**.

---

## 3. How the two papers relate

On the surface one is "VLM + vision tools for spatial reasoning" and the other is "an RL algorithm for LLM post-training", but side by side there are several clear connections:

| Dimension | SpaceTools / DIRL | GFlowRL |
|---|---|---|
| RL algorithm | GRPO + KL (reward maximization) | GFlowNet TB objective (distribution matching) |
| Core pain point | Multi-tool action space explodes combinatorially, naive RL exploration fails | Learned partition function Z_φ causes gradient explosion, fails at scale |
| Approach | **Curriculum**: narrow first, then wide; single-tool IRL → SFT initialization → full-tool IRL | **Subtraction**: delete the auxiliary network, replace it with a within-batch MC estimate over the rollout group |
| Stability measures | Good initialization prevents exploration collapse; rewards normalized to [0,1] | IS correction + asymmetric flow-gap clipping; estimator on the same scale as r/β |
| Scale | 3B VLM | 7B–235B, dense + MoE |

**Common theme**: both deal with "**RL is unstable on large action spaces / large models**", and neither answer is to add machinery — SpaceTools does exploration **in stages**, GFlowRL **deletes** what shouldn't be learned. GFlowRL's conclusion says it plainly: "scaling GFlowNet-style RL depends more on identifying which parts of the original objective are unnecessary in this regime than on adding auxiliary machinery." DIRL's ablation (full-tool IRL directly → 19.79) is essentially the same kind of failure: the optimization signal drowns in an oversized search space / variance.

**Possible intersections (nobody has done them yet, worth thinking about)**:
1. SpaceTools' Stage-2 IRL uses GRPO, and tool calling naturally has **multiple equivalent valid tool chains** (point first then segment vs depth first then point). GRPO will collapse onto a single tool pipeline; switching to GFlowRL's distribution-matching objective could in theory preserve diverse tool-orchestration strategies — exactly the kind of effect GFlowRL's diversity score of 3.93 vs GRPO 1.21 is after.
   *It is feasible engineering-wise now*: both are built on verl (SpaceTools is a verl fork + Toolshed; GFlowRL is also based on verl), and SpaceTools has released the SFT checkpoint and the full-tool RL dataset (~5500), so everything before Step 4 comes for free. Swapping GFlowRL's loss into `run_rl.sh` is an experiment of manageable size.
2. The SpaceTools paper itself mentions in its limitations that "a stepwise reward may be more effective for large multi-tool action spaces", while GFlowRL is trajectory-level; their answers to "how to do credit assignment over long tool chains" are complementary.
3. GFlowRL shows its absolute advantage over FlowRL only in **high reward-noise** settings (red-teaming). Tool-calling rewards are similarly noisy (the tools themselves make mistakes — 23 of SpaceTools' 60 grasp failures were tool errors); this is GFlowRL's home turf.
4. Shared citation base: GRPO (Shao et al. 2024), the VERL framework, Qwen-family backbones. SpaceTools uses Claude Sonnet 4.5 as the universal teacher; GFlowRL uses it as one of the red-teaming victims/baselines.

---

## 4. Quick index (which part of the PDF to check for details)

**SpaceTools**: Alg. 1 (multi-turn tool loop) / §4.1 DIRL two phases / §4.3 reward formulas / Table 2 main results / Table 3 real robot / Table 4 ablations / Table 5 giving GPT-5 and Claude tools / Appx A.2 full tool API / Fig. 7 system prompt / Appx E.4 reward and prompt ablations.

**GFlowRL**: §3.1 why Z_φ fails (Table 1 gradient statistics) / Eq. 4 within-batch estimator / Eq. 5–8 full loss / §4.2 random-log Z diagnosis / Table 2–5 main results / Table 6–7 ablations / Appx B fixed-point proof / Appx D all hyperparameters / Appx I per-problem qualitative comparison with DeepCoder.
