# **GFlowRL Optimization · Design Doc**

Reference implementation – Maccchiatooo/spacetools-training-programs

C′ – refers to the setup where we train our own GFlowRL ckpt and run eval with it

Github repo \- [https\://github.com/qyYue1389/Spacetools-RL-Improvement/tree/main](https://github.com/qyYue1389/Spacetools-RL-Improvement/tree/main)

## **Background, Goals and Scope**

**Background**

* What was done: C′ replaces the GRPO in SpaceTools step 4 (the RL stage) with the C′ variant of GFlowRL  
  * C′ is a distribution-matching method; the goal is for the policy to converge to π ∝ π\_ref · exp(βr)  
  * The loss is Eq. 8: the squared residual of each rollout  
  * Training setup: 85 steps, G \= 5, 1 epoch, 8 A40s  
* How we compare: per the project premise, our self-trained SFT starting point is treated as identical to the official one. So the official GRPO ckpt and the C′ step-85 ckpt have the same starting point, data, G and epoch; the only difference is the objective (GRPO vs C′). The difference between the two is the effect of the objective

**Current state (measured in eval, robospatial 350 questions, each ckpt run 3 times and averaged)**

* **Total score**: SFT starting point 213.0, GRPO 226.7, C′ 221.7. The difference between GRPO and C′ is not significant (p \= 0.284)  
* **Split by question type**:  
  * VQA: C′ gets 3.7 more questions right than GRPO, not significant (p \= 0.359)  
  * Vacant: C′ gets 8.6 fewer questions right than GRPO, significant (p \= 0.0004)  
* **Where Vacant falls short: how the obj\_name passed to roborefer is written**  
  * Anchor object only (e.g. "cup"): only 0–3% of the points returned by the tool land in the target region, so the model has to override the point itself  
  * With a location description (e.g. "point close to and in front of the cup"): hit rate 56%  
* **Number of questions written with the anchor object only** (Vacant has 122 questions in total, per run): SFT 42.0, C′ 34.3, GRPO 13.3.  
* **Conclusion**: writing only the anchor object is a habit taught by the SFT data. GRPO got rid of it, C′ barely changed it: 88.5% of C′'s obj\_name are verbatim identical to SFT

**Overall goals (Goals)**

1. Make C′, under the same configuration, learn at least what GRPO learned — the primary criterion is the share of Vacant questions where "`obj_name` contains only the anchor object name (no location description) when calling roborefer"  
2. Optimize without affecting the existing GRPO comparison; if it is affected, add a same-config GRPO run in the same batch  
3. Every conclusion can be reproduced and distinguished by one fixed measurement protocol (§1.2)

**Non-goals.** Overall "collapse degree", fit questions, variable reuse, iteration on refplacement/refunseen/cvb3ddepth, swapping tools, changing SFT data — reasons in §3

## **1\. Overview**

### **1.1 Priorities and one-line design**

| Priority | Problem | Goal | Core hypothesis | Approach | Cost | Status |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| P1(a) | Degenerate groups dilute the reward gradient | C′ starts rewriting queries (object-only share drops) | 70% of the gradient is pure drift term, diluting the reward signal | Set g̃ to 0 for groups whose rewards are all identical, no resampling | **Zero extra cost**; 40 steps ≈ 10 h | Proposal, **next step** |
| P1(b) | Update scale and direction: grad\_clip saturates every step; ε direction may be reversed | grad\_clip only clips outlier steps; confirm ε direction | grad\_norm 180 comes mainly from the sequence sum (×|y|); Eq.7 taken literally is the opposite of the intent in the paper's text | grad\_clip \= L̄ (≡ loss×1/L̄, config only); ε swap; 15-step comparison each | 3×15 steps ≈ 12 h | Proposal |
| P1(c) | Variance of the log Z estimate at G=5 | Reduce variance to \~1/3.2 | After filtering, stagnation caused by Z variance remains | G=8 or 16 | G=16 85 steps ≈ 2.7 days | Conditionally triggered |
| P1(d) | epochs / β | — | — | Leave as is for now, trigger conditions set | — | Frozen |
| P2 | SFT data line (moved out): front/behind do not call depth\_estimator; Vacant asks only for the anchor object | Get this chain into the support set | Not covered by the SFT data distribution | Inspect data → targeted SFT → exploration → shaping | Medium | Moved out, scheduled separately |
| P3 | A new GRPO comparison is needed after config changes | Keep results attributable | — | Retrain the GRPO arm in the same batch only if G / epochs are changed | ≈ one C′ training run | Conditionally triggered |
| Infra P1 | Rows of degenerate groups still go through forward / backward; about 400 s / step is spent on zero-gradient rows | Per step 922 → about 514 s | Dropping degenerate groups entirely gives the same update as P1(a) | Drop whole groups once rewards are in, pad to a multiple of GPU count × micro | Zero; goes on the GPU machine with P1(a) | Implemented, CPU self-check passed |
| Infra P2 | gen 321 s is mostly waiting on tools, and we don't know which | Find the waits that can be eliminated within the 4 tool GPUs | Waiting is concentrated in a few tools or thread pools | Time per tool, reallocate based on the results | Zero; goes on the GPU machine with P1(a) | Implemented, CPU self-check passed |
| Infra P3 | P2P on the training machine may be broken | Detect it before training starts | — | NCCL / P2P acceptance check at GPU session start | About 2 minutes per GPU session | Implemented, pending GPU session |

### 

### **1.2 Measurement protocol (used by all subsequent evaluations)**

* **Environment:** same machine, same session, KV pool fixed at 24.5 GB (`gpu_memory_utilization` back-computed from GPU capacity, 0.511 on a 49140 MiB GPU); `vlm`'s Ray `num_gpus` \= 1.0 (prevents Molmo and the two DepthPro from being packed onto the same GPU); `ulimit -c 0`  
  * The KV pool size determines how many sequences fit into one batch during inference; when the batch composition changes, the accumulation order of floating-point operations changes with it, so some near-tie samples flip their answers. So the KV pool is fixed to avoid machine-induced differences. The 24.5 GB used for eval on 4x A6000 is the standard  
* **Completion check:** the dump exists and the OOM count in the log is 0 (`run_eval.sh` also returns 15 on success, so the return code cannot be used); every dump passes `parse_dump.py --strict`  
* **Repetition:** 3 times per checkpoint, run 3 rounds in the order SFT → GRPO → C′ rather than grouped by checkpoint  
* **Readout:** robospatial **must be reported split into VQA /228 and Vacant /122**; paired tests give both McNemar variants, "stable core" and "three-run paired pooled". Vacant is additionally reported split by query phrasing (anchor object only / with location description)  
  * **Stable core:** use only the questions where each arm's 3 results agree (all 3 right or all 3 wrong), and on those questions count how many are A right B wrong and how many are A wrong B right  
  * **Three-run paired pooled:** run 1 vs run 1, run 2 vs run 2, run 3 vs run 3, compare per question, add up the disagreements of the three runs, then test  
* **Resolution:** the sd of the difference between two arms ≈ 2.7 questions, 2σ ≈ 5.3 questions. Three runs can only resolve differences above 5 questions; smaller gaps need 6 or more runs  
* **Baseline:** the 27 dumps of P0 (local `P0/p0_bundle/`) are the reference for all later re-checks  
  * Sample trajectories saved during re-evaluation  
    * **robospatial:** SFT, GRPO, C′ × 3 runs  
    * **blinkdepth:** SFT, GRPO, C′ × 3 runs  
    * **RefSpatial three tasks:** SFT, GRPO, C′ × 1 run × 3 tasks 

## 

## **2\. P1 · Training-signal starvation — root cause**

**Problem statement.** C′ did not learn the reward in 85 steps. Readings from the report:

* degen median **70.3%** (0.578–0.859) — in most groups the 5 rollouts have exactly the same reward  
* grad\_norm median **180** (max 5229), grad\_clip \= **1.0**, clipped on every step  
* score 0.809 in the first half, 0.796 in the second half, **no upward trend**; rew/drift median 2.01, slowly falling from 2.125 to 1.754  
* G \= 5, the GFlowRL paper uses 16

**Definitions**

| degen | The 5 rollouts sampled for one prompt form a group; a group whose 5 rewards are all identical (all right or all wrong, within-group range \= 0) is called a degenerate group. degen is the share of degenerate groups in one step | Median 70.3%: in a typical step, about 45 of the 64 groups have 5 identical rewards, more than half, hence "most groups". These groups cannot tell good from bad and carry no reward signal |
| :---- | :---- | :---- |
| grad\_norm | L2 norm of the whole model's gradient vector in one step, measured before clipping; it represents how far this step "wants" to push the parameters | Median 180, far above the paper (GFlowRL 0.043, GRPO 0.20, both averaged per token: the paper's GFlowRL divides each rollout by |y| (token count) and then averages over the G rollouts, GRPO uses token-mean; the paper's task and model differ, so only the order of magnitude is comparable); because C′ does not divide by |y| and instead sums over the whole sequence (\~334 tokens), the gradient is amplified several hundred times |
| grad\_clip | Gradient norm cap (1.0 here): if exceeded, the gradient is scaled down proportionally to norm 1.0, direction unchanged | Over the 85 steps even the smallest grad\_norm is 103 ≫ 1.0, so every step is scaled to 1.0: every step's update has the same size, the information "big change or small change" is lost, only the direction remains |
| score | Mean reward of all training rollouts in this step (about 0–1) | 0.809 in the first half, 0.796 in the second half, not going up: over the 85 steps the model did not get better on the training questions, i.e. it did not learn the reward; the small fluctuation is noise from each step having different questions |
| rew/drift | g can be split exactly into two halves: reward term \= the deviation of this rollout's reward from the group mean (what we want to learn); drift term \= the deviation of this rollout's "how far the current policy deviates from π\_ref" from the group mean (unrelated to reward). rew/drift is the ratio of the mean magnitudes of the two; \>1 the reward dominates, \<1 the reward is drowned by drift | Median 2.01: the reward signal is about twice the drift, the update direction is still dominated by the reward overall, drift has not drowned the reward signal, it holds overall. Falling from 2.125 to 1.754: the policy moves further and further from π\_ref and the drift term grows; the trend is worsening but by a small amount; consider raising β when it gets close to 1 (P1(d)) |

| g(flow gap) | Per-rollout residual g\_i \= Z\_t − d\_i − βr\_i (d\_i \= Σ(log π\_ref − log π\_old), Z\_t \= within-group mean of (βr \+ d)), measures how far this rollout's probability deviates from the target π\_ref·exp(βr). Can be split exactly into reward term β(r̄ − r\_i) \+ drift term (d̄ − d\_i) | g \< 0: probability below target, should be pushed up; g \> 0: should be pushed down; g \= 0: already on target. Difference from GRPO: on degenerate groups the reward term is 0, but g is not 0, only drift remains |
| :---- | :---- | :---- |
| ε(ε\_low / ε\_high) | Clipping bounds of the flow gap, ε\_low \= 0.2, ε\_high \= 0.28, g is only allowed to lie in \[−0.2, \+0.28\] | Under on-policy, each rollout's weight in this step's update is −g̃ (the REINFORCE advantage). Without clipping, if one rollout's |g| is very large, e.g. a large reward deviation or large drift, that single rollout can dominate the whole step's gradient and shove the parameters hard in its direction. After clipping to \[−0.2, \+0.28\], each rollout's weight is at most that large, and no single sample in a step can push far. That is what "trust region" means: each step only moves a little around the current policy, and outlier samples cannot skew the update direction of the whole batch. The ε in the Eq. 8 IS weight w \= min(π\_θ/π\_old, 1+ε) is a different quantity; under on-policy w≡1, so it has no effect |
| g̃ | Clipped flow gap: g̃ \= clip(g, −ε\_low, \+ε\_high) | Under on-policy the update is REINFORCE with −g̃ as the advantage: g̃ \< 0 pushes up, g̃ \> 0 pushes down, the larger |g̃| the harder the push. Taken literally, push up by at most 0.2, push down by at most 0.28 (for the direction question see P1(b)) |
| sat | Share of samples in one step whose g falls outside \[−0.2, \+0.28\] and is clipped to the bound | Low sat means most samples' g is already within \[−0.2, \+0.28\] and not clipped; clipping only touches the few outlier samples with very large |g|. That is exactly the design intent of clipping: inactive normally, only guarding against extreme values. Our sat median is 52%, meaning half the samples are clipped at the bound. Clipping is no longer just handling outliers; it is rewriting the weights of most samples: these samples keep only the direction plus a fixed magnitude. The reason is that the reward term median of 0.67 is far larger than the bounds. g equals reward term plus drift term, and the clipping bounds are only −0.2 and \+0.28. The typical magnitude of the reward term, 0.67, is already 2 to 3 times the bounds, so as long as a rollout's reward differs from the group mean, g almost always exceeds the bounds and gets clipped |
| 1/|y| (length normalization) | The paper's Eq. 5/6/8 divides the log ratio by the response length |y|; C′ removed it and sums over the whole sequence instead | Reasons for removing it: ① the fixed-point proof of Prop. B.1 is itself without length normalization; ② Remark B.4 admits that per-sequence 1/|y| makes the effective β become |y|·β and distorts the target distribution with length; ③ gate fixed-point self-check: normalized 5.1e-2 FAIL, C′ 7.4e-13 PASS. The cost is that the gradient is amplified with |y| (grad\_norm 180); P1(b) uses grad\_clip \= L̄ (≡ constant 1/L̄ scaling) to get the scale back without changing the fixed point |
| β | Inverse temperature coefficient in the target distribution π ∝ π\_ref·exp(βr), determines how much the reward is amplified; we use β \= 8, consistent with the paper's Table 9 | Tunes the weight between "chasing reward" and "staying close to π\_ref" (similar to the 1/KL coefficient in GRPO): large β → more concentrated on high reward, closer to pure reward maximization; small β → closer to π\_ref. In g only the reward term ∝ β, the drift term is independent of β, so β determines rew/drift (median 2.01 at β \= 8). Lowering β cannot fix sat saturation (it only shrinks the reward half; an 8× difference in β moves sat by only 5 pp); on degenerate groups the reward term is always 0, which β cannot reach; try β \= 16 when rew/drift keeps falling toward 1 (P1(d)); per-sequence 1/|y| amounts to replacing β with |y|·β ≈ 2700 |

**Key evidence (from P0).** Same SFT starting point, same data, same G=5, same 1 epoch: **GRPO pushed point overrides from 49.3 down to 21.7, C′ only down to 44.0**. So "the signal at G=5 is inherently insufficient" does not hold — otherwise GRPO should not have learned either. The difference is in how degenerate groups are handled:

|  | Degenerate groups (70.3% per step) | Share of each step's gradient that comes from groups with reward differences |
| ----- | ----- | ----- |
| GRPO | advantage \= (r−mean)/std \= 0, **no gradient** | 100% |
| C′ | C′'s g \= reward term \+ drift term. With identical rewards the reward term is 0, but the drift term (d̄ − d\_i) is generally not 0, because the 5 rollouts in a group deviate from π\_ref by different amounts, so **there is still gradient**, **and it is all drift term**. The direction of this part of the gradient is unrelated to reward; it only pulls the rollouts toward π\_ref's proportions. | About 30% |

**Origin of the primary criterion**

On Vacant, C′ gets 8.6 fewer questions right than GRPO (p \= 0.0004). The cause is a chain: query phrasing (obj\_name) determines whether the tool point is accurate, and whether the tool point is accurate determines whether the model overrides the point. This is a symptom of P1, not a defect of the C′ objective, so P1 uses query phrasing as the primary criterion. All data below are computed offline on eval trajectories; numbers are /122 means over three runs

1. Surface symptom: many questions with point overrides. A point override means the model does not use the point given by roborefer and writes its own instead  
   * Questions with a point override per run: SFT 49.3, GRPO 21.7, C′ 44.0  
   * C′ accuracy: 21.2% on point-override questions, 57.6% on pass-through questions  
2. Point overrides are the result of the tool giving a wrong point  
   * Questions where the last point returned by roborefer lands inside the ground truth (GT): SFT 46.3, GRPO 60.3, C′ 49.3. The GRPO vs C′ stable-sample ratio is 13 : 4, p \= 0.049  
   * On point-override samples, the tool point hits only 10–19%, and the model's overridden answer is actually more accurate (19–28%). That is, the model overrides when the tool point is wrong  
   * What GRPO learned is to write a better obj\_name; more pass-through follows from that  
3. The root cause of the bad tool points is the query phrasing obj\_name (p2\_query\_type.py)

| obj\_name | Tool point hit | Point override | Correct |
| ----- | ----- | ----- | ----- |
| Anchor object only ("cup") | 0–3% | 100% | 15–22% |
| With location description ("point close to and in front of the cup") | 56% | 8–11% | 55% |

   * SFT/GRPO/C′ behave almost identically within the same phrasing; the difference is only in how large a share each phrasing takes  
   * Questions asking for the anchor object only: SFT 42.0, C′ 34.3, GRPO 13.3  
   * The 13 questions GRPO gets right but C′ gets wrong are all anchor-object-only  
   * Taking C′'s accuracy under each phrasing and reweighting by GRPO's phrasing proportions gives about 62 questions, matching GRPO's measured 62.3. So the phrasing proportion alone explains this gap  
   * 88.5% of C′'s obj\_name are verbatim identical to the SFT starting point; for GRPO it is 50.8%  
4. This habit is taught by the SFT data. All 410 demonstrations of this RoboSpatial template ask for the object only, while 465 of the 471 in RefSpatial vacant include a location description

**Implementation check: under our configuration, one C′ update step is exactly REINFORCE.** `ppo_mini_batch_size = train_batch_size = 64`, only one gradient update per step, strictly on-policy (during training `is_weight_min=1.0`, `collapsed_frac=0.0`). So at the moment of differentiation log(π\_θ/π\_old)=0, w≡1, and the gradient of Eq. 8 is exactly `2·g̃_i·∇Σ_t log π_θ` — REINFORCE with **−g̃\_i as the advantage**, summed over the sequence. The items below are all read from this:

* On degenerate groups −g̃ \= −clip(d̄ − d\_i), pure drift (a)  
* The per-step mean of the reward term |β(r̄ − r\_i)| has a median of 0.67 (including the 0s of degenerate groups), while the clipping bounds are only 0.2 / 0.28, so the sat median is 52%: half the samples' advantage is reduced to a sign plus a fixed magnitude (b)  
* The gradient magnitude scales linearly with sequence length (mean measured in the gate stage \~334 tokens), which is the main cause of grad\_norm 180 (b)

For an item-by-item comparison with the external implementation `Maccchiatooo/spacetools-training-programs`, see Appendix B

### 

### **P1(a) · Filter degenerate groups**

**Problem statement**

* Degenerate groups, where all 5 rewards are identical (all right or all wrong), are the majority: about 70% of C′'s groups per step are degenerate groups; rewards within the group are identical, the reward term is 0, and these groups only contribute drift-term gradient  
* After clip the reward signal is diluted: grad\_clip clips each step's gradient to unit norm, keeping only the direction. 70% of the groups are pulling toward π\_ref, so the direction left after clipping is determined mainly by the drift term; the signal that actually comes from reward differences is only a small part  
* This is caused by our own deviation from the paper: the paper's Table 9 lists `Filter groups: Accuracy-based`, i.e. filtering out all-right or all-wrong groups by accuracy. When implementing C′ we deliberately skipped this step, and this is the price paid. The reasoning at the time:  
* Degenerate groups are many: from real G=5 rollout group statistics, the share of prompts with 5 identical rewards is robospatial 199/350 \= 56.9%, blinkdepth 94/124 \= 75.8%, boppose 32/60 \= 53.3%  
* GRPO's advantage on degenerate groups is 0, so no gradient; C′'s per-trajectory squared residual is generally not 0, so it still produces gradient. At the time this was seen as C′ being able to use the part of the rollouts that GRPO throws away  
* Copying the filter as is would drop 53–76% of the prompts at G=5; not filtering is an active deviation from the paper's recipe, which needs its own justification

**Objective**

1. Filter degenerate groups – test whether "the drift gradient of degenerate groups dilutes the reward signal" is why C′ fails to learn  
2. If so, fix it: after filtering, C′ should start learning to write a location description into the query to roborefer, and the share of anchor-object-only rollouts should drop accordingly

**Hypothesis(H1).**

* C′ did not drop the "anchor object only" query habit (point overrides are its consequence) because the reward gradient signal is diluted by the drift term of degenerate groups. That is, as long as the signal is not diluted, the C′ objective itself can learn to switch phrasing:  
  * At the SFT starting point, about 34% of Vacant questions write only the anchor object name and 66% include a location description, which shows the SFT starting point can already write location queries; no new behavior has to be created from scratch.  
  * Training samples at T=1.0 with 5 rollouts per prompt, so a group very likely contains both object-only and location-bearing ones (to be confirmed from the rollout text of the first 10 steps once P1(a) starts)  
  * robospatial eval results show a tool-point hit rate of 56% with a location description vs 0–3% for object only, and final accuracy of 55% vs 15–22%. So when both phrasings coexist in one group, the rewards diverge; these groups are not degenerate and produce a reward signal.  
  * The objective amplifies this signal: the target distribution is π\_ref·exp(βr); at β=8, two trajectories with a reward difference of 1 differ in weight by e^8 ≈ 3000, so once C′ has learned it should strongly prefer the location-bearing phrasing

**Falsifiable predictions:**

* Mask only degenerate groups and change nothing else; within 30–40 steps, the share of Vacant-type "anchor object only" rollouts among training rollouts drops clearly  
* **If it does not move at all**, H1 is rejected, the bottleneck is loss scale / clip (go to P1b), and there is no longer a reason to try G=16: the only effect of a larger G is to lower the degenerate-group share, and if P1a removes degenerate groups directly and the share still does not drop, degenerate groups are not the bottleneck and G=16 will not solve the problem either  
* Direction: in eval, SFT starting point 34% → GRPO 11%; point-override rate \~40% → \~18% drops along with it; eval generation picks the highest-probability token at every step without random sampling, T=0. Training rollouts are sampled at T=1.0, and the wording of training questions differs from the eval template, so the results will also differ from eval; only look at the relative change between windows


**Proposed solution.** "Filter only, no resampling":

1. In the driver's `compute_gflowrl_flow_gap()`, set g̃ to 0 for groups whose within-group reward range is 0 (same criterion as the existing `reward/degenerate_group_frac`).  
   1. Setting g̃ to 0 for all 5 rollouts in a degenerate group makes their advantage 0 and their gradient contribution 0. The pure drift-term gradient these groups used to contribute is removed, the same effect as GRPO producing no gradient on degenerate groups. Each step only the roughly 30% of groups with reward differences take part in the update.  
   2. By the equivalence above, under on-policy this is **bit-for-bit equivalent** to "setting the loss weight of these groups to 0"; no actor-side change is needed, and no extra mask has to be passed to the loss. The reference implementation does it the same way (Appendix B).  
2. The denominator still uses all sequences, **consistent with GRPO**: GRPO's degenerate groups have advantage 0 but still count in the token-mean denominator.  
3. Guard: assert `ppo_mini_batch_size == train_batch_size`. If mini-batches are later split to save GPU memory, the equivalence no longer holds (degenerate groups become a proximal term pulling π\_θ back to π\_old), and it then has to become a real loss mask.  
   1. **Form of Eq. 8:** each trajectory's loss is roughly the squared residual (Σ log π\_θ/π\_old \+ g̃)², and the gradient is 2·(Σ log π\_θ/π\_old \+ g̃)·∇log π\_θ  
   2. **Now (on-policy):** only one update per step; at differentiation π\_θ \= π\_old, the log ratio is identically 0, and the gradient is 2·g̃·∇log π\_θ. After setting g̃ to 0, the gradient is exactly 0, which is the same as filtering these groups out  
   3. **After splitting mini-batches:** one step does several consecutive updates. From the second mini-batch on, π\_θ is no longer equal to π\_old and the log ratio is not 0. Then even with g̃ \= 0, the loss is still (Σ log π\_θ/π\_old)², the gradient is not 0, and its effect is to pull π\_θ back to π\_old, equivalent to a proximal (trust-region) constraint. Degenerate groups start affecting the update again and are not really filtered out  
   4. **So switch to a real loss mask:** multiply these groups' loss by 0 directly, which guarantees no gradient regardless of how far π\_θ is from π\_old. The current patch uses the assertion `ppo_mini_batch_size == train_batch_size` to prevent misuse  
4. New log: number of groups kept per step (expected 64 → \~19: 64 prompts per step means 64 groups, degenerate groups at a median of 70.3% are set to 0, leaving about 64×0.297≈19 groups in training; used to confirm the filter takes effect, and to watch for larger gradient noise/variance when too few groups are kept)  
   1. After filtering, only about 19 groups per step take part in the update. With fewer samples, the step's gradient direction is more easily skewed by the few questions drawn this time, and the direction jitters back and forth between steps instead of moving steadily in one direction  
   2. So log the number of kept groups. If some steps keep only a few groups, the gradient is unreliable. The risks section also gives the response: if the point-override rate moves in the right direction but jitters a lot, add resampling (DAPO-style, filling to a full batch)  
      1. Resampling: after filtering out degenerate groups, draw new questions from the dataset to generate rollouts, fill the non-degenerate groups up to 64 before updating; the group count is increased by drawing more questions

**Alternatives considered.**

| Option | Per step | 85 steps | Conclusion |
| ----- | ----- | ----- | ----- |
| **Filter only, no resampling** | 925 s (unchanged; about 514 s with Infra P1 on) | Unchanged | **Adopted**: zero cost, decisive, does not invalidate the comparison |
| Filter \+ DAPO-style resampling to a full batch | \~1744 s (gen ×3.37) | \~41 h | Only if H1 holds but \~19 groups per step are too few and the variance is too large |
| G=16 directly | \~2770 s | \~65 h | Demoted to P1c |

**Execution plan.**

1. **Implementation**  
   1. The patches are in GFlowRL\_improve/P1\_prep/: p1a\_ray\_trainer.diff (ray\_trainer.py \+41 lines) and p1a\_run\_rl\_gflowrl.diff (new switch GF\_FILTER\_DEGEN).  
      1. The verl main trainer indeed has no filter\_groups (only the FilterGroupsConfig used for DAPO).  
   2. The guard sits in the driver: when filter is on, assert ppo\_mini\_batch\_size \== train\_batch\_size and ppo\_epochs \== 1.  
   3. When on, degenerate groups get g̃ \= 0, everything else unchanged  
   4. Under on-policy, the gradient is bit-for-bit equal to "a real loss mask with the denominator keeping all sequences" (max|diff| \= 0, because dp\_actor's on-policy branch sets old\_log\_prob \= log\_prob.detach(), so log\_ratio is identically 0); an off-policy counterexample confirms that degenerate groups receive a proximal-term gradient, so the guard is necessary  
2. **Monitoring script**: P1\_prep/p1\_monitor.py, reads \<step\>.jsonl from rollout\_data\_dir and outputs per window: object-only share, point-override rate (± per-prompt bootstrap SE), count on the 0.05 grid, pass-through / point-override accuracy, tool-point hit rate; --upper additionally computes forced pass-through.  
3. **Training:**  
   1. Same SFT starting point, same seed, same data order, G=5, other hyperparameters unchanged, only turn on the mask, and set trainer.rollout\_data\_dir to save the rollout text; run 40 steps, save a ckpt every 10 steps.  
   2. Also turn on in-training validation  
      1. The reference implementation runs the whole robospatial validation set with TEST\_FREQ=20, see Appendix B: replace the val set with P1\_prep/robospatial\_vacant.parquet containing only the 122 Vacant questions (already generated), add data.val\_files=… trainer.test\_freq=10 trainer.val\_before\_train=True;  
      2. Greedy decoding: validation uses T=0, so the same model gives the same output on the same question every time; the readings carry no sampling noise and can be compared directly across steps  
      3. With `val_before_train=True`, one validation runs before the first training step. The weights are still SFT at that point, so that result is the SFT starting point's value on these 122 Vacant questions; step 0 is the starting-point reading on the same machine, same KV pool; each run takes about 7 minutes (scaled from about 20 minutes for 350 questions on P0), 5 runs over 40 steps ≈ 35 minutes. The validation\_data\_dir dump is read directly with p1\_monitor.py for the number of object-only questions.  
4. **Readout:** every 10 steps is one window; read the object-only share and the point-override rate; also look at degen, kept group count, score, rew/drift, grad\_norm.  
5. **(Optional) confirmation:** run robospatial ×3 on the step-40 ckpt per the §1.2 protocol, read VQA / Vacant separately.

**Success criteria / decisions.**

| Readout | Verdict | Next step |
| ----- | ----- | ----- |
| Object-only rollout share: the last 20 steps are lower than the first 20 steps by ≥ 12 percentage points (pp), about half of the 23 pp gap between the SFT starting point and GRPO; or in in-training validation (Vacant 122 questions, greedy) the number of object-only questions is ≥ 10 lower than at step 0 (robospatial eval differs by only ±1 question across three runs: SFT 42 / 43 / 41, GRPO 14 / 13 / 13, C′ 33 / 35 / 35; the gap from the SFT starting point to GRPO is 29 questions) | H1 holds | Keep the filter, go to P1b |
| Drops but by less than 12 pp | Partially holds | Go to P1b; if still not enough after P1b, then consider resampling or P1c |
| Basically no change | H1 rejected | Go to P1b |

* 12 pp is a provisional threshold: simulated from P0 eval data, the SD of the difference between two windows in object-only share is ≈ 8.6 pp for a 10-step window (58 questions × 5) and about 6 pp for a 20-step window (116 questions), 2× ≈ 12 pp, so read with 20-step windows  
* RefSpatial pointing demonstrations almost all include a location description (465 / 471), so the object-only share on those questions is already close to 0. Mixing them in enlarges the denominator and thins out the signal. So look only at RoboSpatial Vacant, and suppress noise by widening the window to 20 steps  
* Below is the estimate for the point-override rate, as a secondary reference  
  * Calibrate with the per-prompt bootstrap standard deviation within P1a's 10-step window (the threshold is at least 2× the fluctuation)  
  * Simulated from eval data  
    * Reading only RoboSpatial vacant (58 questions × 5 per window), the SD of the difference between two windows ≈ 9 pp, 2× is about 18 pp; if the point-override rate changed by 10 pp, it could not be distinguished from noise  
    * Combined with RefSpatial pointing (174 questions × 5 per window) SD ≈ 5 pp, 2× ≈ 10 pp, so when reading the point-override rate, combine the two question types RoboSpatial Vacant and RefSpatial pointing into one point-override rate and compare that with the threshold. eval is greedy (T=0), training is T=1.0; the actual threshold is set by the bootstrap of the first training window

**Risks**

* **The drift term also acts as a regularizer that "pulls back to π\_ref".** Removing 70% of the groups may make the policy drift away from π\_ref faster. Monitor rew/drift and entropy; if entropy drops clearly or the format error rate rises, record it and stop  
* **Only \~19 groups left per step, so gradient variance grows.** If the point-override rate moves in the right direction but jitters a lot, add resampling  
  * Resampling: after filtering out degenerate groups, draw new questions from the dataset to generate rollouts, fill the non-degenerate groups up to 64 before updating; the group count is increased by drawing more questions  
  * robospatial's pointing reward has only two values, 0 and 1  
    * The judging method is `convex_hull`: only the first point the model outputs is considered; if it lands inside the convex hull of the target region it gets 1, otherwise 0; no partial credit by distance  
    * So pointing groups can also be entirely degenerate and get filtered; what remains are exactly the groups with some right and some wrong among the 5, and GRPO's signal also comes only from these groups  
* **score will most likely not move within 40 steps**  
  * The reference implementation (Maccchiatooo/spacetools-training-programs) already has degenerate-group filtering + G=16 + wider clip bounds ε. After 39 steps, a linear regression of training reward on step gives a slope t value of −0.25. |t| is far below 2, so the slope is indistinguishable from 0 and the reward basically did not rise. So the criterion uses the object-only share, not score  
  * Another risk: the "object-only" habit watched by the primary criterion may not appear at all on the training questions  
* **Where the habit comes from**: in the SFT data, all object-only demonstrations appear on the question template used by RoboSpatial eval (410 / 410). What the model learned is "when you see this phrasing, ask only for the object"  
* **Training questions are phrased differently**: the corresponding questions in the RL training set (unary\_spatial\_context) are worded like "several points … situated", unlike the eval template. So whether the model makes the same mistake on training questions is unknown in advance; we can only see it once the first window's rollouts are out  
* **If object-only is rare on training questions to begin with**: then the object-only share on training rollouts has no room to drop, and using it as the criterion is meaningless. In that case use the in-training validation readings instead, i.e. one run on the 122 Vacant questions every 10 steps. Those 122 questions use the eval template, so the habit will definitely show up  
* **Sample size of the point-override rate on rollouts**  
  * Data source: the RL training set is `siyich/spacetools-rlfulltools` on HF  
  * Questions of the same type as Vacant: the training set has 500 RoboSpatial unary\_spatial\_context items, the same question type as eval's Vacant, 9.1% of the training set. On average about 5.8 of the 64 questions per step are drawn from it, times 5 rollouts is 29; a 10-step window is about 58 questions, 290 rollouts  
  * Other pointing questions: there are also two pointing types, RefSpatial vacant and RefSpatial object, 500 each, which can also be used for the point-override rate  
  * Conclusion: under the standard "a difference must exceed 2× SD to count as a real change", using only those 500 same-type items, the two-window STD is about 9 pp and the threshold is 18 pp. The expected change in point-override rate is about 10 pp, which does not reach the threshold and cannot be seen. With the three types combined SD ≈ 5 pp and the threshold is about 10 pp, which the expected 10 pp change can reach. If the actual noise during training is still too large, rely on the in-training validation every 10 steps (Vacant 122 questions)

**Cost.** 40 × 925 s ≈ 10.3 h (8 GPUs; about 40 × 514 s ≈ 5.7 h with Infra P1 on) + in-training validation about 35 minutes; optional confirmation eval about 1.2 h (4 GPUs)

### 

### **P1(b) · Update scale and direction: loss scale, grad\_clip, flow-gap ε**

**Problem statement.** Three issues, all about how large each step's update is and which way it leans:

1. **grad\_norm median 180, grad\_clip=1.0, clipped on every step**  
   1. The main cause is C′'s sequence sum: the gradient scales linearly with |y|. The paper's Eq. 5/8 normalizes by 1/|y|; in Table 1, GFlowRL's grad\_norm median is 0.043 and GRPO's is 0.20; 180 / 334 ≈ 0.54, the same order of magnitude as GRPO. grad\_clip=1.0 was set for the normalized scale; carried over to C′ unchanged, it clips every step  
2. **The direction of ε may be reversed**  
   1. **Formula**: Eq. 7 clips g to \[−ε\_low, \+ε\_high\], where ε\_low \= 0.2, ε\_high \= 0.28.  
   2. **Sign of g**: g \= Z\_t − d − βr. For samples with reward above the group mean, the reward term is negative and g is usually \< 0; for samples below the group mean, the reward term is positive and g is usually \> 0  
   3. **Direction each rollout is pushed**: after differentiating Eq. 8 with respect to θ, each rollout's update direction is −g̃·∇log π. This is our own derivation; the paper does not state it. So:  
      1. Good samples: g is clipped at the lowest to −0.2, so the push up is at most 0.2  
      2. Bad samples: g is clipped at the highest to \+0.28, so the push down is at most 0.28  
   4. The conclusion is that there is more room to push down than to push up  
   5. **The intent in the paper's text is exactly the opposite**: the original says "give more room for positive corrections than negative ones", the reason being that correct solutions are under-sampled early on and their probability should be pushed up more aggressively. By this intent, the cap for pushing up should be 0.28 and for pushing down 0.2  
   6. **Impact**: both we and the reference implementation wrote Eq. 7 literally. Currently about 52% of g are clipped to the bounds, so this asymmetry is in effect on every step  
3. **The paper's 0.2 / 0.28 may not be bounds for g at all, but parameters of a different quantity**  
   1. The two have different units: DAPO's ratio clip clips the probability ratio πθ/πold to the range \[1 − 0.2, 1 \+ 0.28\]; it measures how much the new and old policies differ and is itself a small relative quantity. GFlowRL's g has the units of log probability plus βr; at β=8 a single reward difference can reach several units, not on the same scale as 0.2 at all  
   2. Why we suspect Table 9 means the former: Table 9 is a hyperparameter table shared by three methods, GFlowRL, FlowRL and GRPO. Its "Clip ratio εlow 0.2 / εhigh 0.28" uses the name and default values of DAPO clip-higher, and the same table also says "Advantage estimator: GRPO". So these two numbers are very likely the setting of GRPO's ratio clip, and the paper does not give a separate ε for g  
   3. What the reference implementation does: following this reading, it enlarges the bounds on g to 2.7 / 3.8, keeping the 0.2 : 0.28 ratio. That way g is almost never clipped, sat ≈ 0. We used 0.2 / 0.28 literally, so sat ≈ 52%, more than half the samples are clipped

**Objective**

1. Make grad\_clip clip only abnormal steps. Currently grad\_norm is about 180, every step exceeds the 1.0 threshold and gets scaled, so grad\_clip has turned into a fixed scaler. Setting the threshold to L̄ (mean response length) means normal steps are no longer clipped and only abnormally large steps get scaled down  
2. Decide the direction of ε by experiment. The paper's literal formula contradicts the intent in its text, and the official code is not released. So add a comparison arm that only swaps ε (push up 0.28, push down 0.2) and compare it with the current-state arm that follows the formula literally; use whichever works better

**Hypothesis(H2a)**

* What clip saturation loses: every step's raw gradient norm is far above 1.0 (about 180), so every step is scaled to norm 1. For example, if one step was originally 360 and another 90, both become 1 after clipping, and the original "one big, one small" difference between the two steps is gone.  
* Why the effect may be small: the optimizer is AdamW, which divides the update by a moving average of squared gradients (the second moment). If all gradients are multiplied by the same constant, numerator and denominator grow together and the update is unchanged. So "the overall scale being shrunk" by itself has no effect on AdamW. The only thing clip really changes is the relative size between steps: steps with large gradients should have pushed more, but now all steps are the same. This loss exists, but is not large.  
* So decide with a short run: change to grad\_clip \= L̄ (about 334) and run only a 15-step comparison to see whether the change actually helps, rather than putting it directly into the main training run

**Hypothesis(H2b)**

* The ε direction changes the preference of the update: taken literally (push up 0.2, push down 0.28), the update leans toward "pushing down bad samples". After the swap (push up 0.28, push down 0.2), it leans more toward "lifting good samples"

* Relation to point overrides: a point override means the model does not use the point returned by the tool as is and changes it to a different point. This behavior has an error rate of 79%, so it mostly appears in bad samples  
  * When leaning toward "pushing down bad samples", these point-override rollouts are pushed down directly, and point overrides may drop faster  
  * When leaning toward "lifting good samples", the pass-through correct rollouts are pushed up and point overrides are only crowded out indirectly, but correct solutions are learned faster  
  * Both mechanisms are plausible; which one is better for the point-override rate and score cannot be derived in advance, so it can only be measured

**Proposed solution.**

1. **Constant loss scaling ×1/L̄**  
   1. Tests H2a  
   2. **What**: multiply the whole loss by a fixed constant 1/L̄. L̄ is the mean response length, measured earlier at about 334; after the GPU session starts, fix a value from `y_len_mean` in the logs of the first few steps, and do not change it afterwards  
   3. **Why it does not affect the objective**: multiplying all losses by the same constant leaves the minimum unchanged, so C′'s fixed point (loss, gradient 0) is also unchanged. Only dividing each sample by its own length |y\_i| would change the fixed point (that is the paper's original normalized approach, which C′ does not use)  
   4. **Why only one line of config is needed**: under AdamW, multiplying the loss by 1/L̄ has exactly the same effect as raising the grad\_clip threshold from 1.0 to L̄. Because Adam is insensitive to constant multiples of the gradient, eps is small enough to ignore, and KL is off, the relative weights of the terms do not change. So arm B only needs `grad_clip = L̄`, no code change  
   5. **What L̄ to choose, and how many steps then escape clipping**: back-computed from the grad\_norm of C′'s 85 steps, with the threshold set to L̄ the fraction of clipped steps is:  
      1. 334 → 10.6%  
      2. 250 → 21%  
      3. 200 → 40%  
      4. 150 → 71%  
   6. The criterion requires the clipped fraction \< 20%, so L̄ below about 260 does not work  
   7. **Must be decided after the GPU session starts**: 334 was measured on robospatial eval; the training set also has bop and refspatial, whose lengths may differ, so look at the actual lengths in the first few training steps  
2. **ε swap**  
   1. Tests H2b  
   2. Change the clipping range of g from \[−0.2, \+0.28\] to \[−0.28, \+0.2\], swapping the two bounds  
      1. Good samples (negative g) are pushed up by at most 0.28, previously 0.2  
      2. Bad samples (positive g) are pushed down by at most 0.2, previously 0.28  
      3. This gives more room to push up than to push down, matching the intent of the paper's text. Only the two config values ε\_low and ε\_high need to change, no code change  
3. **Logging**  
   1. Add the p50 / p90 / p99 of |g| and clip\_saturation\_kept computed only over kept groups, so that future choices of ε have a basis  
      1. **p50 / p90 / p99 of |g|**: the median, 90th percentile and 99th percentile of |g| over all samples in each step, showing how wide the distribution of g is. For example, a p90 of 1.5 means the 0.2 bound clips over 90% of the large g

      2. **clip\_saturation\_kept**: the clipped fraction computed only within the groups kept after filtering. The original sat also counts degenerate groups, whose g̃ has already been set to 0, and mixing them in distorts the number

**Execution plan**

* Starting point: reuse P1a's configuration. Filter degenerate groups if H1 holds, otherwise do not filter. G \= 5  
* Three comparison arms: 15 steps each, run one after another on the same 8-GPU machine  
  * A: P1a's configuration as is, as the baseline  
  * B: A with grad\_clip set to L̄ (equivalent to loss × 1/L̄), tests H2a  
  * C: A with ε swapped, tests H2b  
* How the conclusion is used: compare B and C each against A. Adopt whichever beats A; if both do, put both changes into the main training run

**Success criteria**

* **Conditions for adopting B** (all three must hold):  
  1. Fewer than 20% of steps are clipped by grad\_clip (provisional). This requires L̄ of at least about 260.  
  2. Training does not diverge  
  3. The object-only rollout share is no worse than A  
* **Condition for adopting C**: neither the object-only rollout share nor score is worse than A. Whether or not it is adopted, record both differences relative to A, including direction and size, as evidence on the ε direction question

**Risks**

* 15 steps is too short to see a trend in score. So for B, look mainly at two things: whether the fraction of clipped steps went down, and whether training is stable  
* C's effect may be drowned in noise. The difference in anchor-object-only share over 15 steps may be smaller than the window's own fluctuation. If they cannot be told apart, keep implementing the paper's formula literally and state "ε direction undetermined" in the report

**Cost**

* 3 × 15 × 925 s ≈ 11.6 h  
* B and C are both config only (grad\_clip, GF\_EPS\_LOW / GF\_EPS\_HIGH), zero code

### 

### 

### **P1(c) · Increase G (conditionally triggered)**

**Problem statement**

* log Z (i.e. Z\_t) is estimated as the mean over each group's G rollouts; the variance of the estimation error is proportional to 1/G. The smaller the group, the less accurate the estimate  
* We use G \= 5 and the paper uses G \= 16, so our variance is 16/5 ≈ 3.2 times the paper's  
* All of the paper's evidence that GFlowRL works comes from experiments with G \= 16 and 30 to 400+ training steps. Our G is smaller and our step count is also lower (85 steps); both conditions are weaker than the paper's, and it is uncertain whether its conclusions carry over to our setup

**Objective**

* Increase G to lower the estimation variance of Z.  
  * G \= 8: variance becomes 5/8 of the original, down 37.5%  
  * G \= 16: variance becomes 5/16 of the original (about 1/3.2), down about 69%, matching the paper's setting

**Hypothesis(H3)**

* After degenerate-group filtering (P1a) and clip (P1b) are both fixed, if training still stagnates, the remaining cause is that the variance of the Z estimate is too large  
  * *A larger G originally had a second reason: fewer degenerate groups. With a per-rollout accuracy of p \= 0.8, the probability that a group is all right or all wrong is 32.8% at G \= 5 and only 2.8% at G \= 16. But P1a's filtering already removes degenerate groups directly at much lower cost, so this reason no longer counts*

**Trigger**

* After both P1a and P1b are done, start P1c only if either of the following holds:  
  * The anchor-object-only share drops but stalls clearly above about 11% (GRPO's level)  
  * score still has no reproducible upward trend

**Proposed solution.** Use G=8 if machines are tight, otherwise G=16

**Cost & trade-offs.** G=16: \~2770 s per step, 85 steps ≈ 65 h ≈ 2.7 days. **Changing G invalidates the existing GRPO comparison**; GRPO has to be retrained in the same batch, roughly doubling the total cost; it also can no longer be compared directly with the paper's 70.0

### **P1(d) · epochs, β (unchanged for now, trigger conditions set)**

|  | Why not change it now | Trigger condition | What to do once triggered |
| ----- | ----- | ----- | ----- |
| epochs | 1 epoch is hard-coded upstream (`run_rl.sh:435`), and the paper's RoboSpatial Overall score of 70.0 was also from training for only 1 epoch. Under the current config, score did not rise over 85 steps, and most of the gradient comes from the drift term of degenerate groups rather than from learning the reward. Running over the data once more at this point would only add more steps that likewise do not learn the reward, and would not help. | score shows a reproducible upward trend over the first 85 steps | Increase to 2–3 epochs, evaluate robospatial \+ blinkdepth every \~30 steps, state in the report how many ckpts were evaluated. The GRPO comparison run must be extended too |
| β | At β \= 8, the median magnitude ratio of the reward term to the drift term is 2.01; the reward term is about 2 times the drift term, **the reward already dominates**, and the drift term does not swamp the reward signal. Tuning β has very little effect on the clipped fraction. Recomputing the same batch of rollouts with β from 2 to 16, sat only varies between 48%–53%. On degenerate groups the reward term is always 0, so however large β is, it multiplies 0. The degenerate-group problem can only be solved by P1(a)'s filtering. | After P1a-c, rew/drift still keeps falling and gets close to 1 | Try β=16 |

### Trigger condition for the β row: when to touch β

* ### rew/drift: the magnitude ratio of the reward term to the drift term; its median is currently 2.01, the reward roughly dominates

* ### Keeps falling: in C’ it fell from 2.125 in the first 10 steps to 1.754 in the last 10 steps. The reason is that the policy moves further and further from π\_ref and the drift term keeps growing

* ### Gets close to 1: means the drift term is already as large as the reward term, the reward signal is about to be drowned, and the update is increasingly pulling the policy back to π\_ref rather than learning the reward

* ### So the condition is: after P1 (filtering, clip fix, etc.) is done, if rew/drift still falls all the way to near 1, raise β from 8 to 16, doubling the reward term so that it outweighs the drift term again

### **P1 · Main training run and overall exit conditions**

**Main training run**

* Configuration: on top of the original C′, stack the changes that the earlier experiments validated as effective:  
  * P1a's degenerate-group filtering (if H1 holds)  
  * The changes adopted in P1b: grad\_clip \= L̄ (B), and/or ε swap (C)  
  * If P1c is triggered, increase G from 5 to 8 or 16  
* Training amount: 1 epoch, i.e. 85 steps  
* Evaluation: evaluate per the §1.2 protocol about every 30 steps. After each evaluation, run two scripts offline on the eval dump (no GPU needed) to compute the quantities in the exit-condition table:  
  * p1\_monitor.py: decides per sample whether the answer used the tool's point as is (pass-through) or changed it (point override), and outputs the point-override rate (± per-prompt bootstrap SE), pass-through / point-override accuracy, tool-point hit (whether the first point of roborefer's last return lands inside the GT convex hull), anchor-object-only share, and the number of answers on the 0.05 grid. It can read both training rollouts and eval dumps; training rollouts are output in windows of 10 steps  
    * Add \--upper: forced pass-through, replaces the answers of point-override samples with the tool point and re-scores them, to see what score "no point overrides at all" would get. On P0 all three arms go down (SFT 50.7 → 46.3, GRPO 62.3 → 60.3, C′ 53.7 → 49.3), which shows point overrides help on average; suppressing point overrides alone will not raise the score, a higher score has to come from better tool points  
  * p2\_query\_type.py: splits Vacant questions into two classes by obj\_name phrasing (anchor object only / with location description), outputs per class the question count, tool-point hit, point-override rate and accuracy, and also gives the share of queries verbatim identical to the SFT starting point. The phrasing table in "Origin of the primary criterion" was computed with it. It currently hard-codes reading the dump file names of P0's three arms; an input argument has to be added before evaluating new ckpts  
* Time: about 22 h at G \= 5, about 65 h at G \= 16

**Exit conditions:**

1. **Primary criterion:** the share of Vacant anchor-object-only queries drops from 34% at the SFT starting point to near GRPO's 11%  
   1. **Where to read it**: first look at the trend on training rollouts, then confirm with eval results at the end  
   2. **Criteria on eval** (Vacant 122 questions, mean of three runs; for why these quantities, see "Origin of the primary criterion"):

| Metric | Target | GRPO reference | C′ current |
| :---- | :---- | :---- | :---- |
| Share of queries with obj\_name=anchor object only | ≤ 15% (provisional) | 11% (13.3 questions) | 28% (34.3 questions) |
| Tool point hit | Close to GRPO | 60.3 questions | 49.3 questions |
| Point overrides (per run) | ≤ 25 (provisional) | 21.7 | 44.0 |
| Vacant score | No more than 5.3 questions below GRPO | — | −8.6 questions |

2. Scores are **reported split into VQA / Vacant**; the overall score is not used as a criterion. VQA must still lead GRPO (currently \+3.7 questions): C′ is not behind everywhere (VQA \+3.7 questions, blinkdepth \+2.6 questions; C′ has the fewest questions that flip across the three runs; 0 cases of tool misuse), and fixing Vacant must not lose these advantages  
3. Per the §1.2 resolution, with each arm run 3 times and averaged, the pairwise fluctuation sd is about 2.7 questions. So only a gap ≥ 5.3 questions (2× sd) counts as a real difference; anything smaller is treated as noise  
4. **Condition for exiting to P3**: if after the P1 main training run the anchor-object-only share is still around 30%, the root cause is not the training signal but the objective or the SFT data; go to P3 (rewrite those 410 demonstrations with location descriptions)  
5. degen drops clearly; score has a reproducible upward trend  
   1. A linear regression of training score on step gives a slope clearly above 0 (t \> 2). The reference implementation's slope t \= −0.25 counts as not significant  
   2. **Not driven by a few steps**: by window, later windows stay consistently above earlier windows, rather than one or two spikes pulling up the mean  
   3. **Also visible on eval**: evaluating about every 30 steps, ckpt scores also rise with training

## 

## **P2. SFT data line: front/behind never calls depth estimation, Vacant only asks about the anchor object**

**Problem statement**

* The RoboSpatial templates in the SFT data have two gaps:  
  * Vacant: all 410 demonstrations only ask about the anchor object  
  * yes/no: 977 demonstrations, not a single one calls depth\_estimator  
  * The most direct fix is to change the SFT data, but that replaces the common starting point of all the comparisons, invalidates the existing comparisons, and is no longer a "C′ vs GRPO" comparison, so it is moved out of this doc's priorities and scheduled separately. Below, only the investigation and approach for front/behind are kept  
* Current state of front/behind (robospatial, 29 questions, asking which of two objects is farther from the camera):  
  * The tool that should be used is almost never called. depth\_estimator can answer this kind of question directly, but the model almost never calls it on robospatial. After three eval runs of each of the three compared models, the call counts are SFT 0/0/0, GRPO 2/2/1, C′ 0/0/1; each run has about 577 tool calls, almost all roborefer  
  * It is not that the model can't use it; it just doesn't think of using it with this question wording. The same models call depth\_estimator on almost every question on blinkdepth (n=124): SFT 121/122/120, GRPO 124/123/123, C′ 122/122/122  
  * No accuracy difference between SFT/GRPO/C’. Mean of three eval runs (same machine, same KV pool): SFT 17.3, GRPO 19.0, C′ 19.3 (/29)

**Objective**  
Make the model call depth\_estimator on robospatial-style front/behind questions, i.e. make this chain able to appear in training-time sampling

* First it has to "be able to appear": RL can only reinforce behaviors that appear in samples. Right now the starting point almost never calls this tool, so RL has nothing to amplify  
* How much can be recovered: C′ currently gets 19.3 of these 29 questions right on average, so at most about 10 more can be gained (29 − 19.3), about 2.8 pp of robospatial's 350 questions. That is the all-correct upper bound; the goal is to recover part of it

**Hypothesis(H4)**

The model does not call depth\_estimator mainly because the SFT data has no such demonstrations, not because the model can't use this tool

* **Not a capability problem**: the same models call it on almost every blinkdepth question; they just don't think of calling it with the robospatial question wording  
* **Not suppressed by RL**: the SFT starting point itself has 0 calls; it was absent before RL  
* **The RL objective only determines how much this tiny probability can be amplified**:  
  * In theory, GRPO only chases reward and can amplify a low-probability good behavior more aggressively; C′ pulls it back toward its proportion under π\_ref  
  * Measured, both are close to 0: GRPO 2/2/1 calls, C′ 0/0/1 calls, out of about 577 tool calls per run, about 0.3%. Too few to tell the two apart  
* **So RL alone can't fix it**: C′'s target distribution is π ∝ π\_ref·exp(βr); for a behavior with π\_ref close to 0, the target probability is also close to 0. And RL can only reinforce sampled trajectories. GFlowRL's **distribution matching can preserve existing behaviors but cannot create missing ones**; this chain must first appear in the starting point

**Proposed solution (in order; if a step doesn't hold, don't do the next).**

1. **Check the data (done)**  
   1. Conclusion: the SFT data never demonstrates depth\_estimator on the robospatial templates; H4 holds at the data level  
   2. The 29 eval questions don't specify a viewpoint and are answered from the camera viewpoint by default, so a depth comparison can be used directly. Most front/behind in SFT / RL is "seen from the second object's viewpoint", which camera depth can't answer. There are only 40 camera-viewpoint positional questions in SFT and 22 in RL  
2. **Make this chain appear in the starting point**:  
   1. Approach: following the teaching phase, have the Claude teacher answer with the full tool set mounted and keep only correct answers, generating a small batch of front/behind trajectories with depth\_estimator. Then pick one of two: mix them into the SFT data and retrain the starting point, or do a small targeted SFT before RL  
   2. Pick mainly camera-viewpoint (or viewpoint-unspecified) positional questions, aligned with those 29 eval questions.  
3. **Encourage exploration in the RL stage**:  
   1. Raise the sampling temperature for these prompts; or  
   2. Within each group of G rollouts, add a hint "you can use depth\_estimator" to a few of them. These rollouts are off-policy; either add importance weights or use them only early in training  
4. **Reward shaping comes last**: only once the starting point already has this chain does shaping have something to reinforce

**Execution plan**  
Retraining the SFT starting point changes the common starting point used by GRPO and C′, so all existing comparisons have to be redone. Therefore:

* Timing: do it after P1 reaches a conclusion  
* Attribution: it must not be mixed into the same training run as P1's changes, otherwise it's impossible to tell whether the effect comes from the added data or from P1

**Success criteria**

1. It calls the tool: on the 29 front/behind questions, the depth\_estimator call rate is clearly above 0. Currently 0–2 calls per run  
2. More correct answers: accuracy on the 29 front/behind questions goes up. Currently SFT 17.3, GRPO 19.0, C′ 19.3 (/29)  
3. No drop elsewhere: scores on the other robospatial question types (the rest of VQA, and Vacant) and on other benchmarks don't drop

**Why P2**

* Limited gain: front/behind can gain at most about 10 more correct questions.  
* The problem is in the SFT data, i.e. upstream in the DIRL pipeline, not in the C′ objective itself. And this doc mainly compares "C′ vs GRPO".  
* It would replace the common starting point: the problem of Vacant only asking about the anchor object can be fixed the same way. Rewriting those 410 demonstrations to include location descriptions is expected to let C′ directly catch up to close to GRPO on Vacant (currently 8.6 questions behind). But that likewise requires retraining the SFT starting point, and all existing comparisons would have to be redone.

## **P3 · GRPO comparison (conditionally triggered)**

**Problem statement.** For the conclusion to be attributable to "C′ vs GRPO", a GRPO comparison with the same configuration is needed

**Current state.** For the current configuration a comparison already exists: the official checkpoint \= GRPO with the same SFT starting point, same RL data, same G=5, same 1 epoch, and the same eval. The remaining uncontrolled factors are the seed and the training hardware (paper 2×8 A100 vs our 8× A40)

**Design: which changes invalidate the comparison.**

| P1 change | Comparison still valid | Reason |
| ----- | ----- | ----- |
| G | **Invalid** | Changes the amount of data seen per step |
| epochs | **Invalid** | Changes the number of updates |
| Infra P1 drop degenerate groups | Valid | Only changes compute allocation; the C′ arm's updates are identical to P1(a), the GRPO arm doesn't enable it |
| Infra P2 tool replica reallocation | Valid | Tool outputs unchanged, only queueing changes |

**Execution plan.** Triggered only if G or epochs are changed: retrain a GRPO with the same base, same seed, same G, same number of steps, on the same machine, scheduled in the same machine booking as the P1 formal training; evaluate per §1.2

**Cost.** About one C′ training run with the same configuration (about 2.7 days at G=16)

## **Infra · Training throughput (in parallel with algorithm optimization, does not change the objective)**

**Why a separate section.** Every P1 experiment round is priced at "8 GPUs × about 10 h", so throughput directly determines how many hypotheses can be tested per day. The three items in this section only change how compute is spent, not C′'s objective or updates. They are numbered Infra P1–P3, a separate series from the algorithm-side P1–P3 above. Code, self-checks and notes are in `GFlowRL_improve/infra_prep/` (README).

**Where one 922 s step goes.** `infra_prep/stage_timing.py` reads the full training log on HF (`provenance/full_train.log.gz`), median over 85 steps, in seconds:

| Stage | What it does | Time | Share | Scales with |
| ----- | ----- | ----- | ----- | ----- |
| gen | Uses sglang to sample G multi-turn trajectories per question under the current policy (rollouts, one per row): generate → call tool → wait for the result → generate again, until an answer is given | 321 | 35% | Mostly waiting on tools (see Infra P2) |
| old\_log\_prob | old\_log\_prob uses the training-side policy model (the actor in FSDP) to run one forward pass over every trajectory produced by gen. The parameters haven't been updated yet within the step, so the model at this point is π\_old, and this forward pass gives the log probability of every token in the trajectory under π\_old. It is needed for two reasons: **As the baseline for the loss.** update\_actor compares the new policy with π\_old, and C′'s d \= Σ(log π\_ref − log π\_old) also uses it (Eq. 4/6). **The logprobs sglang returns at sampling time can't be used directly.** The numerical precision of the two engines isn't exactly the same, so it is recomputed in the training engine to keep the definition consistent before and after. | 112 | 12% | No. of rollout trajectories |
| ref | The ref stage uses the reference model π\_ref to run one more forward pass over every trajectory, giving the log probability of each token under π\_ref. π\_ref is the frozen SFT starting point. C′'s d \= Σ(log π\_ref − log π\_old) uses this result; it measures how far the policy has already moved from the starting point. In GRPO the counterpart is the KL penalty term. It does almost the same thing as old\_log\_prob, just with a different model, and likewise forward only. The difference is that the ref model's parameters normally sit on the CPU and are moved to the GPU only when used, so it is a bit slower than old\_log\_prob At the start of RL training, two identical copies of the SFT checkpoint weights are loaded: **actor(π\_θ)**: updated every step, drifting further from SFT as training goes on. **ref(π\_ref)**: never updated, keeps the SFT weights from the moment training started throughout. "Frozen" means ref takes no part in backward and its parameters never change from start to end. "SFT starting point" means it is exactly the model RL started from. So ref acts as a fixed anchor: comparing the actor with it tells how far the policy has drifted from the starting point. | 117 | 13% | No. of rollout |
| update\_actor | forward \+ backward \+ optimizer.step(), updates the actor weights with the C′ loss; the only stage in each step that changes parameters It needs both forward and backward, and backward costs about twice as much as forward. Add FSDP's communication of parameters and gradients between GPUs, and it is about 3× slower than old\_log\_prob and ref, which only run forward. | 352 | 38% | No. of rollout |
| save\_checkpoint | Model and optimizer state saved to disk; save\_freq \= 5, i.e. once every 5 steps | 40 | about 1% amortized per step | save\_freq=5 steps out of 85 steps in total \~17 times \~40s/time |
| update\_weights | Syncs the updated actor weights to sglang for the next step's gen | 10 | 1% | — |
| Rest | Computing the advantage and other small bits | 0.3 | 0% | — |

* These stages are the names of the timing sections in the verl training loop.  
* In the main loop of `ray_trainer.py`, verl wraps each stage with `marked_timer()` for timing, e.g. `with marked_timer("gen", ...)`. These names serve as timing labels and, once written to the log, become metrics such as `timing_s/gen` and `timing_s/update_actor`; our timing table is computed from these metrics.  
* The "remaining 129 s" (step 85) in report §8.4 = old\_log\_prob 118.8 \+ update\_weights 10.0: verl already timed both sections; the report only copied four items at the time  
* The three sections that scale with No. of rollout total 580 s, 63%; gen is 35%. These two blocks are handled by Infra P1 and Infra P2 respectively  
* ref on the CPU (117 s) differs by only about 5 s from old\_log\_prob (112 s), which is also one forward pass but with parameters on the GPU; ref offload is not the bottleneck (§3)

**actor and weights.** The actor is the policy model being trained, π\_θ (Qwen2.5-VL-3B); weights are the model's parameters, which both the actor and ref (SFT ckpt) have, except that ref's weights are frozen and never updated. verl keeps the weights of the same actor in two places, handed to two engines:

|  | Training engine FSDP (PyTorch built-in tool) | Generation engine sglang |
| :---- | :---- | :---- |
| What it stores | actor weights (sharded across the 4 training GPUs) | A copy of the same actor weights |
| Purpose | Forward, backward, parameter updates | Fast rollout generation (gen) |

* update\_actor: does backward and optimizer.step() in FSDP, changing the FSDP copy of the policy-model weights; this is the real update  
* update\_weights: computes no gradients, only copies the freshly updated weights from FSDP to sglang so the next step's gen samples with the new policy  
* Both steps update weights, but only update\_actor "learns"; update\_weights is just a sync that keeps the weights used for generation consistent with the trained ones. Skip it and sglang keeps sampling with the old policy, and training is no longer on-policy  
* The vision encoder is frozen throughout (`freeze_vision_model=true`, the upstream default); it still runs forward as usual (the model can still see the image), but computes no gradients, and its parameters aren't in the optimizer.  
* Only all the Transformer layers of the language-model part (2.55B) are trained, **all updated**, full-parameter fine-tuning, no LoRA

Note: the actor here and the actor in tool deployment are not the same thing; the names just collide:

|  | actor in RL | actor in tool deployment |
| :---- | :---- | :---- |
| Concept it comes from | The actor-critic terminology of reinforcement learning | The Ray framework |
| What it refers to | The policy model being trained, π\_θ (Qwen2.5-VL-3B) | A resident worker process holding one tool model (roborefer, sam2, depth\_estimator, etc.) |
| Do the weights change | Updated every step | Unchanged, inference only |
| Related numbers | — | num\_actors × num\_gpus: how many processes this tool starts and how many GPUs each process takes |

* One easily confused point: verl itself also starts processes with Ray, and the processes that train the actor model (ActorRolloutRefWorker) are themselves Ray actors. But in this doc, actor in the training part means the policy model and in the tool part means a Ray process; they are not the same layer

### **Infra P1 · Drop degenerate groups: old\_log\_prob / ref / update\_actor only compute kept rows**

**Problem statement**

* old\_log\_prob \+ ref \+ update\_actor take a median 580 s per step (63%), and the time scales linearly with No. of rollout  
* With P1(a) on, groups whose 5 rewards are all identical (degenerate groups) have g̃ set to 0; on-policy, the gradient of these trajectories is exactly 0 and doesn't affect the parameter update. These trajectories are a median 70.3% of each step (about 225 out of 320). But old\_log\_prob, ref and update\_actor still run forward / backward over all 320, so about 400 s (580 × 70.3%) of those three sections' 580 s is wasted computation.  
  * `siyich/spacetools-rlfulltools` RL training set `train.parquet`, 5500 questions in total  
  * train\_batch\_size=64 comes from upstream run\_rl.sh, **64 questions per step**: 5500 ÷ 64 \= 85.9; verl drops the last incomplete batch, so 1 epoch is 85 steps. 64 questions × G=5 rollouts per question  
* When gen finishes, each trajectory's reward is already computed and returned together with the trajectory (stored in the `rm_scores` field). So there's no need to wait for the later three sections to run; which groups have all 5 rewards identical (degenerate groups) can be determined before old\_log\_prob starts  
  * `rm_scores(`reward model scores`)` is the field in verl's batch that stores rewards  
    * It is each trajectory's reward score, i.e. whether the question was answered correctly (scored by question type).  
    * Trajectories are scored as soon as the gen stage finishes sampling them, and the scores are written into the batch together with the trajectories, so they come back with the rollouts  
    * If the 5 trajectories of one prompt have exactly the same `rm_scores` (within-group range \= 0), the group is a degenerate group  
    * A per-token tensor. **Shape**: `[num trajectories, response length]`  
      * `rm_scores` is a table, one row per trajectory and one column per token. It is almost all 0s; only the cell at each trajectory's last valid token holds that trajectory's reward.  
      * E.g. 3 trajectories of lengths 4, 3, 5 tokens, with rewards 1, 0, 1:  
        * Trajectory 1: \[0, 0, 0, 1, 0\]     ← the 4th token is the last one, holds reward 1  
        * Trajectory 2: \[0, 0, 0, 0, 0\]     ← reward is 0, the whole row is 0  
        * Trajectory 3: \[0, 0, 0, 0, 1\]  
        * Positions past a trajectory's length are padded with 0, so the table is as wide as the longest one  
      * Sum along the token dimension: each row has only one nonzero cell, so summing a row gives the trajectory's reward: `rm_scores.sum(dim=-1)` gives `[1, 0, 1]`. This way there's no need to find "where the last valid token is"; one sum gets every trajectory's reward.  
      * Infra P1 \-   
        * Sum over each trajectory to get each one's reward.  
        * Group by `uid`. The 5 trajectories sampled from the same prompt have the same `uid` and form one group  
        * Look at the max and min reward within each group. The 5 rewards of a degenerate group are exactly identical, so the max equals the min. It can be all correct (all 1) or all wrong (all 0). If the two are equal, all 5 rewards are the same, the group is degenerate, and the whole group is dropped. The 5 rewards of a non-degenerate group include at least two different ones, so the max does not equal the min. E.g. 3 correct and 2 wrong: max 1, min 0  
      * Per-token is verl's generic format: some tasks have a reward at every step, so there's a slot per token. Our task is scored only once, when the answer is complete, so only the last cell is used  
    * Source: the agent loop computes reward\_score for each trajectory in the gen stage and writes it into rm\_scores when packing the batch  
  * Order within one step:  
    * **gen**: sample rollouts, return rewards.  
    * **Drop degenerate groups** (the Infra P1 patch).  
    * **old\_log\_prob, ref**: one forward pass each over the remaining rollouts.  
    * **update\_actor**: split the remaining rollouts into micro-batches, multiply each loss by 1/40, backward, accumulate, and finally `_optimizer_step()` updates the parameters once.

**Objective**

* With updates exactly identical to P1(a), go from 922 s to about 510 s per step; P1(a)'s 40 steps go from about 10 h to about 6 h  
  * P1(a) sets g̃ of degenerate groups to 0; these trajectories are still computed as usual, just with zero gradient  
  * Infra P1 drops degenerate groups as a whole and doesn't compute them  
  * Under both approaches, the gradient and parameter update computed per step are the same. The dropped trajectories had zero gradient anyway, and the loss denominator is still based on all 320, so the result doesn't change

**Hypothesis(Infra H1)**

* Drop degenerate groups as a whole, then pad the remaining row count to a multiple of 8 (4 training GPUs × micro-batch of 2 rollouts per GPU). After this, the gradient of every kept trajectory is exactly the same as in P1(a). Three reasons:  
  * **Each trajectory's weight is unchanged.** Each trajectory's loss weight is fixed at 1/320, computed from the config, independent of how many rows are actually fed in:  
    * The goal is to compute the average gradient over all 320 rollouts and update the model parameters (the actor's 2.55B weights) with the average gradient  
      * (The summed gradient would also work; the two have the same direction and differ only by a factor (320×). The average is used so that the update size doesn't change with batch size  
      * **Learning rate is easy to set**: with the sum, a larger batch gives a larger gradient, which amounts to scaling the learning rate up with it. Every change of batch size would require retuning the learning rate. With the average, `lr = 1e-6` means the same thing across batch sizes  
      * **Gradient clipping is easy to set**: our `grad_clip = 1.0` is set for the magnitude of the average gradient. With the sum, the gradient would be 320× larger and would be clipped every step)  
    * `ppo_mini_batch_size = 64` comes from upstream `run_rl.sh`   
      * Its unit is prompts, not rollouts. At startup verl automatically converts it to rollouts per GPU: 64 × 5 (`rollout.n`) = 320 rollouts. **320 ÷ 4 (number of training GPUs) = 80**. Each GPU processes 80 rollouts  
      * It equals `train_batch_size` (also 64), which means the 320 rollouts sampled in one step are split into just one mini-batch, so parameters are updated only once per step  
      * SpaceTools-RL/verl/workers/fsdp\_workers.py  
    * `ppo_micro_batch_size_per_gpu = 2` also comes from upstream `run_rl.sh`  
      * Determines how many are fed into the GPU at a time  
      * One GPU's memory can't hold forward / backward for 80 trajectories at the same time, so only 2 rollouts are computed at a time; those 2 form one micro-batch. Parameters aren't updated right after each; instead gradients are accumulated, and the update happens once after all 80 are done.  
      * A trajectory's context limit is 16384 tokens (prompt 8192 \+ response 8192); forward \+ backward has to store the activations of every layer, which takes a lot of GPU memory. The report records that without gradient checkpointing, the activations of one forward pass don't fit on a 48 GB A40  
      * **Measured headroom**: with micro \= 2 and gradient checkpointing on, peak training GPU memory is 35.3 / 48 GB, leaving about 13 GB  
      * SpaceTools-RL/verl/workers/actor/dp\_actor.py  
    * gradient\_accumulation \= **80 / 2 \= 40**: each GPU computes 2 rollouts at a time, so it has to compute 40 micro-batches  
    * On each GPU the loss of each micro-batch x 1/40, micro-batch=2, so on each GPU the loss of each rollout x 1/80. So each rollout accounts for 1/80 of each GPU's average gradient  
    * **Average over 4 GPUs**: FSDP averages the gradients of the 4 GPUs, so each rollout's gradient accounts for 1/320 of the average gradient, and summed up this is the average gradient over all 320.

| \# SpaceTools-RL/verl/workers/actor/dp\_actor.pyself.gradient\_accumulation \= ppo\_mini\_batch\_size // ppo\_micro\_batch\_size\_per\_gpu   \# 80 // 2 \= 40self.actor\_optimizer.zero\_grad()                  \# zero the gradientsfor micro\_batch in micro\_batches:                 \# 40 micro-batches    loss\_scale\_factor \= 1 / self.gradient\_accumulation    \# 1/40    ...                                           \# forward pass, compute policy\_loss for these 2 rollouts    loss \= policy\_loss \* loss\_scale\_factor        \# multiply loss by 1/40    loss.backward()                               \# backward pass, gradients accumulate on the modelgrad\_norm \= self.\_optimizer\_step()                \# after all 40, update the parameters once |
| :---- |

    * After dropping degenerate groups, the dropped rollouts no longer contribute gradient, and the kept rollouts still account for 1/320 each of the average gradient. This matches the P1(a) result: in P1(a) the rollouts of degenerate groups have zero gradient but still count in the denominator.  
  * **Must pad to a multiple of the micro-batch.** Otherwise some GPU is left with a micro-batch of only 1 row; its loss is still multiplied by 1/40, so that row's weight becomes 1/40, 2× the normal 1/80.  
  * **The within-group baseline is unchanged.** Z\_t (Eq. 4) is the within-group mean. Whole groups are dropped and the kept groups lose nothing, so Z\_t is unaffected.  
    * Z\_t is the mean of (β·r \+ d) over the 5 trajectories in the group, where r is the trajectory's reward and d \= Σ(log π\_ref − log π\_old). It is the group's baseline, and each trajectory's g \= Z\_t − d − β·r is its deviation from this baseline.  
* The time of these three sections is roughly proportional to No. of rollout (estimated per row, not per token; length differences between degenerate and kept groups introduce bias)

**Falsifiable prediction:**

* Prediction: after dropping degenerate groups, the time of the three sections old\_log\_prob, ref and update\_actor drops in proportion to the "fraction of rollouts kept", and the other stages stay the same. Estimating step by step from the actual degenerate-group fraction of each of the 85 completed steps, the median step time is about 514 s (originally 922 s); 80% of steps fall within 461–571 s, and the slowest step is about 602 s.  
* What would falsify the prediction: on the GPU machine, look at `gflowrl_drop/rows_after` in the log (the number of rollouts left after dropping). If fewer rollouts remain but the time of these three sections doesn't go down with it, the time is not mainly determined by the number of rollouts (e.g. it is spent on moving weights or on cross-GPU communication). In that case the second point of Infra H1 (the three sections' time is roughly proportional to the row count) does not hold.

**Proposed solution.**

1. **Drop degenerate groups**: located in `ray_trainer.fit`: after `response_mask` is computed, before `balance_batch` (which spreads the rollouts evenly over the 4 GPUs). Test: if, for the 5 rollouts of the same prompt, the max of the rewards (`rm_scores`) equals the min, the group is degenerate and is dropped as a whole.  
2. **Pad the count**: pad the number of remaining rollouts to a multiple of 8 (4 training GPUs × micro-batch of 2 per GPU), so every GPU gets the same count and no micro-batch has only 1 rollout. The padding uses the dropped degenerate rollouts with the shortest responses: with P1(a) on they have g̃ \= 0 and contribute no gradient; the shortest are picked to take less compute  
   1. Most rows are dropped; only a few are padded back to reach the multiple, at most 7  
3. **Switch.** `GF_DROP_DEGEN`, off by default. Must be turned on together with P1(a)'s switch `GF_FILTER_DEGEN=true`; turning it on alone raises an error. Reason: only with P1(a) on is the gradient of degenerate groups 0, so dropping them afterwards doesn't change the update. Both switches are our own  
4. **Logging.**  
   1. The existing degenerate-group metrics `reward/degenerate_*` are still computed over all 320 before dropping, comparable with previous runs.  
   2. 5 new metrics, prefix `gflowrl_drop/`:  
      1. `rows_before`: count before dropping (320)  
      2. `rows_after`: count after dropping and padding  
      3. `pad_rows`: number of padded rows  
      4. `kept_rollout_frac`: fraction of non-degenerate rollouts among all (padding excluded)  
      5. `score_mean_full`: mean reward over all 320  
   3. The rollout dump still writes all 320, so the readings of `p1_monitor.py` are unaffected.

| reward/degenerate\_\* | Training metrics: per-step degenerate-group fraction, etc. | Added to `ray_trainer.py` by us when implementing GFlowRL; already in use in GFlowRL training |
| :---- | :---- | :---- |
| `gflowrl_drop/*` (5 of them) | Training metrics: counts before and after dropping, etc. | Our new Infra P1 patch |
| rollout dump | Writes all rollouts of each step to one JSONL file | A built-in feature of upstream verl (`trainer.rollout_data_dir`); we only turn it on and make sure all 320 are still written after dropping degenerate groups |
| p1\_monitor.py | Standalone script that reads the rollout dump and computes the point-override rate, etc. | Written by us |



**Alternatives considered.**

| Option | Per step | Conclusion |
| ----- | ----- | ----- |
| Only set g̃ to 0 (P1(a) original plan) | 922 s | Same update, but about 70% of compute goes to zero-gradient rows |
| **Set to 0 \+ drop rollouts (this plan)** | about 514 s | **Adopted** |
| After dropping, change the loss to average only over the kept rows | about 514 s | Not adopted: after dropping there are fewer rollouts in total and each rollout's share grows, which amounts to multiplying the learning signal, and it doesn't match the grad\_clip convention of P1(a), P1(b) |
| After dropping, pad only to a multiple of the GPU count | about 514 s | Not adopted: single-rollout micro-batches appear, and that rollout's share of its GPU's average gradient goes from 1/80 to 1/40, 2× the other rollouts (the counterexample in the self-check) |

**Execution plan.**

1. **Implementation (done)**  
   1. `infra_prep/patched/ray_trainer.py` (= the SpaceTools-RL `c6fef78a` used for training \+ P1(a) \+ this item)  
   2. `run_rl_gflowrl.sh` (new switch `GF_DROP_DEGEN`)  
   3. Full diff `infra_all_vs_c6fef78a.diff`, verified to apply with `git apply` on a clean c6fef78a  
2. **Self-check (done, CPU)**  
   1. `infra_drop_check.py` builds 16 groups of data, with 0, 5, 11 and 16 of them degenerate respectively. In all four cases, the gradient computed after dropping degenerate groups is exactly identical to P1(a) (no dropping, only setting g̃ to 0).  
   2. Counterexample: padding only to a multiple of the GPU count, ignoring the micro-batch, gives a gradient that differs from P1(a). This shows that padding to a multiple of "GPUs × micro-batch" is required.  
   3. The existing 6 GFlowRL self-check scripts were changed to run against the full patched code, and all pass, showing the patch doesn't break existing logic.  
3. **On the GPU machine:** P1(a)'s 40-step training directly turns on both `GF_FILTER_DEGEN=true` and `GF_DROP_DEGEN=true`. Afterwards, read each stage's time with `stage_timing.py` and compare it with `gflowrl_drop/rows_after` in the log (the number of rollouts left after dropping) to see whether the three sections' time goes down with it.

**Success criteria / Decision.**

| Reading | Verdict | Next step |
| ----- | ----- | ----- |
| The three sections' time roughly proportional to rows\_after, median step time ≤ \~600s; rows\_after always a multiple of 8 | Effective | On by default for subsequent C′ training |
| Time doesn't drop, OOM, or rows\_after not a multiple of 8 | Not effective | Turn the switch off and run with the original P1(a) plan; doesn't affect the experiment itself |

**Risks**

* The scope of 3 log metrics changes. With this item on, `critic/score/mean` (mean reward), `perf/throughput` and `perf/total_num_tokens` are computed only over the kept rollouts, no longer over all 320. The kept ones are all from non-degenerate groups, so their mean reward differs from the full batch's; when comparing mean reward across runs, use `gflowrl_drop/score_mean_full` (mean reward over all 320). Take particular care when comparing with the old P7 logs: in the old logs `critic/score/mean` is computed over all 320  
  * `critic/score/mean, gflowrl_drop/score_mean_full, etc. are all metric names`  
* Usable only for C′, not for GRPO. GRPO's loss is averaged over tokens, with the number of tokens in the micro-batch as the denominator. Dropping rollouts changes the composition of the micro-batch, the denominator changes with it, and the update differs from the no-drop case. So the P3 GRPO comparison does not enable this item and takes the original path.  
* The actual time saved may be less than estimated. The estimate assumes time is proportional to the number of rollouts. In practice only about 45–135 rollouts remain per step, i.e. only a dozen to thirty-odd per GPU; per-GPU utilization may drop and fixed overhead takes a larger share, so the time doesn't drop as low as the proportional estimate.  
  * Part of the overhead doesn't scale with No. of rollout. Each stage's time can be viewed as "fixed part \+ per-rollout part". The fixed part includes: distributing the data to the 4 GPUs, the one `optimizer.step()` that updates the 2.55B parameters, gradient clipping, and cross-GPU gradient sync. When rollouts go from 320 down to e.g. 96, only the latter part shrinks proportionally while the fixed part stays the same, so the total time doesn't drop to 30%.  
  * **Harder to balance across GPUs.** `balance_batch` tries to give the 4 GPUs similar token counts. With 80 per GPU, long and short ones easily offset each other; with only a dozen to thirty-odd per GPU, one or two especially long ones make balancing impossible. The 4 GPUs have to wait for the slowest one to finish before they can sync gradients, and the other GPUs sit idle while waiting; this is the utilization drop.  
  * The kept rollouts may be longer. The estimate is based on No. of rollout, not on token count. If trajectories in non-degenerate groups are longer than in degenerate groups (e.g. harder questions, a few more rounds of tool calls), each kept one takes longer to compute, and the time saved will also be less than the per-count estimate. The doc's Hypothesis mentions this.

**Cost.** Zero GPU cost for development; on the GPU machine it runs together with P1(a), no extra time

### 

### **Infra P2 · Per-tool timing: find out what gen is waiting on**

**Problem statement**

* gen takes 321 s per step (35%), and Infra P1 can't help with it. Infra P1 saves time by dropping degenerate groups, but it needs rewards to decide whether a group is degenerate, and rewards only exist after gen finishes. So gen has to run fully over all 320 rollouts.  
* **gen's time is mostly spent waiting for tools to return, not on model generation.** The `agent_loop/*` metrics (median over 85 steps) show:  
  * Per rollout on average: waiting on tools 135 s, model generation 20 s  
  * The slowest rollout of each step: waiting on tools 281 s, model generation 26 s  
* The 320 rollouts run concurrently, and gen only ends when the slowest one finishes, so gen's time is determined by the slowest one.  
* The existing logs don't show where the time goes. verl only records each rollout's total tool time and can't distinguish three things:  
  * Which tool is slow  
  * Whether the tool itself computes slowly, or requests are queueing inside the tool service  
  * Or whether it gets stuck even earlier: in the agent loop's thread pool. Toolshed calls tools synchronously via `run_in_executor(None, ...)`, each call occupying one thread; the agent loop has 8 processes (rollout.agent.num\_workers, default 8), about 40 rollouts per process, each process with a thread pool of at most 32 threads; each rollout issues at most 8 calls at once per turn, and when the outstanding calls in one process exceed 32, the extra ones can only queue for a thread.  
    * rollout.multi\_turn.max\_parallel\_calls=8 —- upstream run\_rl.sh  
* Tool replica counts aren't allocated by actual usage. To fit the tools into 4 GPUs, the actors of every tool were scaled down uniformly by ×0.5 (roborefer 3 replicas, vlm 1, sam2 2, depth 2, bbox 2, vision\_ops 4, grasp 2). This is a uniform scale-down, so heavily called tools may not have enough replicas.

**Objective**

* Time each tool call individually to find out which tool and which part (queueing or compute) gen's waiting time goes to. The goal is to find waiting time that can be eliminated without adding hardware: still using the existing 4 tool GPUs, fixed only by reallocating the replica counts of the tools or enlarging the thread pool.

**Hypothesis(Infra H2)**

* The waiting is concentrated in a few tools or in the agent loop's thread pool, rather than in all tools being generally slow to compute. Most likely roborefer: each robospatial eval run has about 577 tool calls, almost all roborefer

**Falsifiable prediction:**

* **If the hypothesis holds, one of the following two will be seen on the GPU machine:**  
  * Some tool is queueing: the median of its remote time (from the request being sent to Toolshed until the result returns) is far above the minimum. The minimum approximates "how long one computation takes without queueing", and the part by which the median exceeds it is queueing time  
    * The cause is the replicas. The replica count determines how many requests the tool can handle at once, and queueing means the requests exceed the replica count. One replica is one process holding that tool's model, handling one request at a time (max\_concurrency=1). roborefer has 3 replicas; when a 4th request arrives while all 3 replicas are busy, it can only queue until one frees up. The queueing time counts toward the "remote time", so remote time \= queueing time \+ compute time.  
  * The thread pool is queueing: the time a tool call waits for a thread (exec\_wait) is clearly above 0, and within the same agent loop process, the number of tool calls issued across all tools but not yet returned often exceeds that process's thread count (at most 32).  
* What result would falsify the hypothesis: the median remote time of every tool is close to its minimum, and the thread-pool queueing time is also close to 0. That means there is no queueing and the time is spent on the tool computation itself, so reallocating replica counts won't help, and Infra H2 does not hold.

**Proposed solution.**

1. Set our own environment variable `TOOL_TIMING_DIR(tool_agent_loop.py), which controls whether tool calls are timed: `log one line for each tool call. Each line records three kinds of information:  
   1. **Basic info**: tool name, submit time, total time, whether it errored.  
   2. **Total time split into three parts**:  
      1. Thread-pool queueing (exec\_wait): the time from the tool call being submitted until it gets a thread  
      2. Toolshed remote: the time after the tool call gets a thread and the thread's request reaches Toolshed, including routing (router), queueing in front of the tool actor, and tool compute  
      3. Post-return processing: the wrap-up after the result comes back to the event loop, e.g. ray.put  
         1. **Event loop**: the agent loop's main thread, which schedules all rollouts. Tool calls are handed to threads in the thread pool to run, and when done the results are handed back to the main thread.  
         2. **Wrap-up**: after the main thread gets the result, it has to tidy it up before handing it to the model, e.g. storing the images and variables returned by the tool in Ray's shared object store (that's `ray.put`), then assembling them into a tool reply message.  
         3. **How it's computed**: total time minus the first two parts (thread-pool queueing, Toolshed remote); what remains is this part.  
   3. **Load at the moment of submission (within this process)**: the number of tool calls for this tool issued but not yet returned; the number of tool calls across all tools issued but not yet returned; the thread pool size.  
2. **Summary.** `tool_timing_summary.py` reads these logs and gives per-tool time quantiles (min, median, etc.) and each tool's share of total time, broken down by training step.  
   1. A self-built script that reads the timing logs and summarizes them per tool  
3. Decide the next step from the summary. The decision rules are in the Success criteria below

**Execution plan.**

1. **Implementation (done):** `infra_prep/patched/tool_agent_loop.py` (SpaceTools-RL side) and `patched_toolshed/verl.py` (Toolshed side, based on `712e557`), one diff each, verified to apply with `git apply`  
2. **Self-check (done, CPU):** `tool_timing_check.py`: with the variable unset, return values are exactly identical to the original function; with it set, there is one record per call, the three-part split is correct, queueing time grows when there aren't enough threads, and Toolshed errors and unknown tools are both recorded as failures  
3. **On the GPU machine:** P1(a)'s 40 steps also set `TOOL_TIMING_DIR=$OUTPUT_DIR/tool_timing`; afterwards summarize with `tool_timing_summary.py`  
4. If tool replicas need to be reallocated, do it together with the tool packing check in §4 "before the GPU session"

**Success criteria / Decision.**

| Reading | Verdict | Next step |
| ----- | ----- | ----- |
| Some tool's remote time p50 is far above min, and its share of time is the largest | That tool is queueing | Within the 4 tool GPUs, cut replicas of tools that are rarely called and use the freed GPU share to add replicas to frequently called tools (pass the tool packing check first); in the next training run, compare whether gen time drops |
| Thread-pool queueing is significant, and the number of tool calls issued but not yet returned often exceeds the thread count | The thread pool is the bottleneck | Raise the agent loop's default thread count, no extra GPUs needed |
| Remote p50 ≈ min, thread-pool queueing ≈ 0 | Compute bottleneck | Batching or different GPUs; don't touch the tool allocation |
| After the adjustment the gen median drops, and the drop is clearly larger than gen's step-to-step time variation over the 85 steps of the previous training | Effective | Use the new configuration by default afterwards |

**Risks**

* The patches on both sides have to be applied together to split the time. The timing patch has two parts: SpaceTools-RL and Toolshed. With only the former, each tool call only records the total time; separating the thread-pool queueing and Toolshed remote parts relies on the timestamps added by the Toolshed-side patch.  
  * Thread-pool queueing \= time the thread starts executing − submit time  
  * Toolshed remote \= time Toolshed returns − time the thread starts executing  
  * Post-return processing \= total time − the first two parts  
  * Changes  
    * SpaceTools-RL \- verl/experimental/agent\_loop/tool\_agent\_loop.py  
    * Toolshed \- toolshed/integration/verl.py  
* Only make adjustments that don't change the task. Reallocating tool replicas only affects how fast the queues move; tool outputs are unchanged, so rewards are unchanged, and runs remain comparable with earlier ones. Setting tool timeouts or limiting call counts would change the tool results the model gets, which amounts to changing the task; not done (see §3).  
* The overhead of the timing itself is negligible. Each tool call writes one line to a file, only a few hundred lines per step.

**Cost.** Timing has zero GPU cost and runs together with P1(a); subsequent adjustments depend on the results and need no extra machine

### **Infra P3 · GPU session acceptance check: GPU P2P / NCCL**

**Problem statement**

* **On the machine used for GFlowRL training, P2P between GPUs was broken, but the check command didn't show it.**  
  * Symptom: `nvidia-smi topo -p2p r` shows P2P as all OK, but once training starts, the first NCCL communication that needs multiple GPUs together hangs and never returns  
  * Workaround: set `NCCL_P2P_DISABLE=1` to turn off direct GPU-to-GPU transfer. Cross-GPU gradient sync (all-reduce) then has to relay the data through host memory, which is slower than direct transfer.  
  * Impact: update\_actor (352 s, 38%) has to sync across GPUs on every micro-batch. The 925 s per step figure was measured with P2P turned off and can't be compared directly with other machines.  
* **Found with a minimal probe we built** (at the time it was ncclprobe.py; nccl\_bw\_probe.py is the newer version with bandwidth added later)  
* The probe loads no model; it just puts one tensor on each of the 4 training GPUs, does one all-reduce, and checks whether it completes within 40 seconds and whether the result is correct. One round takes about 90 seconds.  
* It was used to try NCCL environment variables one by one, each turning off one communication path: `NCCL_IB_DISABLE` (InfiniBand), `NCCL_SOCKET_IFNAME` (specify the NIC), `NCCL_P2P_DISABLE` (direct GPU transfer), `NCCL_SHM_DISABLE` (shared memory). Of the 5 configurations, only the one with direct GPU transfer turned off worked, which established that P2P was broken.  
* Without the probe, each configuration would require restarting training once and waiting 15 minutes for the result.

**Objective**

* Before training starts, determine within about 2 minutes whether P2P works and what the all-reduce bandwidth is

**Hypothesis(Infra H3)**

* On a machine with working P2P, update\_actor will be faster; a machine with broken P2P should be identified before training starts, not after training is already running

**Proposed solution**

* Self-built `infra_prep/preflight_nccl.sh`. It runs the all-reduce probe twice on the 4 training GPUs, once with P2P on and once with P2P off, each run reporting whether it works and the bandwidth, with the verdict: P2P\_OK / P2P\_BROKEN / NCCL\_BROKEN

**Execution plan.**

1. **Implementation (done).**  
   1. `nccl_bw_probe.py`: the probe itself  
      1. Do one all-reduce and check whether it completes within 40 seconds  
      2. Put an all-ones tensor on each GPU; after summing across the 4 GPUs every element should equal 4  
      3. Measure bandwidth: do 10 more all-reduces in a row, take the mean time, and convert it to GB/s  
   2. `preflight_nccl.sh`: the GPU session acceptance-check script. Calls the probe twice (P2P on/off) and gives a verdict  
   3. Verification status: the script logic has been run successfully on CPU (with PyTorch's CPU communication backend gloo in place of NCCL); it has not yet been run on real GPUs, which has to wait until a machine is rented.  
2. Procedure for each time a training machine is rented. Before training, run two checks first: the tool packing check and this item (NCCL acceptance check). Start training only if both pass, and record the results in the deviation list.  
   1. Tool packing check (check\_tool\_packing.py): whether the sum of the GPU shares taken by the tools fits into the tool GPUs

**Success criteria / Decision.**

| Verdict | Next step |
| ----- | ----- |
| P2P\_OK | Don't set `NCCL_P2P_DISABLE`, start training normally |
| P2P\_BROKEN | Setting `NCCL_P2P_DISABLE=1` also works, but update\_actor will be slower; record it in the deviation list; switch machines if time allows |
| NCCL\_BROKEN | Don't start training, switch machines |

**Cost.** About 2 minutes per GPU session

## 

## 

## **3\. What we won't do (Non-goals)**

* **Do not treat the "overall degree of collapse" as an optimization target.** Cases such as RefSpatial, boppose and depth questions that "only take one chain" mostly don't count as collapse, because these tasks only have one correct chain to begin with. Diversity is measured only on questions that really have multiple equivalent valid chains  
* **Do not tune prompts or rewards for fit questions.** On fit questions GRPO and C′ both have 69.5% accuracy, identical per question. What's missing is free-space and 3D size information in the tool set, which adjusting the output distribution can't make up for  
* **Do not repeatedly evaluate on refplacement, refunseen, cvb3ddepth.** On these 777 samples GRPO and C′ have exactly identical per-question results; the tool outputs determine the answers, so no difference between RL algorithms can be measured  
* **Do not change tools, and do not change the SFT data.** Changing tools is a separate, independent improvement route; mixing it in would make it impossible to tell where the effect comes from. Changing the SFT data would replace the common starting point (see P2)  
* **Do not make infra changes whose gain is too small or that would change the experiment itself.** Putting the ref model back on the GPU: measured to save only about 5 s / step while taking 2 GB more per GPU (patch kept, off by default); slimming checkpoints: saving amortizes to about 8 s per step, and dropping the optimizer state would break automatic resume; overlapping rollout with training (async / off-policy): would break the on-policy equivalence that P1(a) and Infra P1 rely on; setting tool timeouts or limiting call counts: would change the rewards and the task

## 

## **4\. Execution plan and milestones**

| Phase | Content | GPU | Output / decision |
| ----- | ----- | ----- | ----- |
| Before the GPU session (no GPU) Implemented | Switch for the rollout dump to store only text and no images is implemented (+trainer.dump\_images=false), self-check passed Tool packing check for the training machine is implemented (infra\_prep/check\_tool\_packing.py, Appendix B), self-check passed Code for Infra P1–P3 is implemented, CPU self-checks passed `P1 - ray_trainer.py` patch and switch `GF_DROP_DEGEN P2 - tool_agent_loop.py and the Toolshed-side verl.py, two patches, plus the summary script tool_timing_summary.py P3 - nccl_bw_probe.py and preflight_nccl.sh` After the GPU session starts, first run the tool packing check and the Infra P3 acceptance check | None | Can start running with the current machine configuration |
| Day 1 | **P1(a)**: only filter degenerate groups, run 40 steps; at the same time turn on Infra P1 (drop degenerate groups) and Infra P2 (per-tool timing) | 8 GPUs \~10 h (about 6 h with Infra P1 on) | H1 holds / partly holds / rejected |
| Day 2 | **P1(b)**: A (baseline) / B (grad\_clip \= L̄) / C (ε swap), 15 steps each | 8 GPUs \~12 h | Whether to adopt B and C, each separately |
| Day 3 onward | **P1 formal training**: stack the changes adopted earlier, 85 steps; when the trigger condition is met, add P1(c) (G=8/16), in which case the GRPO comparison must also be added | 8 GPUs: G=5 about 22 h; G=16 about 65 h, plus the same duration again for the GRPO comparison | Interpret according to the P1 exit conditions, and recompute the query phrasing / pass-through / point-override tables |
| Later | P1(d): add epochs once score shows an upward trend P2: SFT data line | — | — |

## 

## 

## **5\. Open questions**

1. The literal form of the paper's formula is the opposite of the intent stated in the text; see P1(b).  
   1. There is no other implementation to compare against: the official code is still unreleased. The reference implementation also follows the literal form, and it widens the bounds to 2.7 / 3.8, so g is almost never clipped and whether the direction is right makes no difference, so it can't be used to decide.  
   2. It can only be settled by measuring C (ε swap) of P1(b).  
2. Rental time and model of the 8-GPU training machine; if it isn't A40, record it in the deviation list, and rerun the parts needed for the comparison on the same machine. At the start of the GPU session, first pass the Infra P3 acceptance check (P2P).

## 

## 

## **Appendix · Key numbers**

| Quantity | Value | Source |
| ----- | ----- | ----- |
| Time per step | mean 925 s / median 922 s (gen 321 / old\_log\_prob 112 / ref 117 / update\_actor 352 / update\_weights 10; save 40 only on 17/85 steps) | Report §8.4; infra\_prep/stage\_timing.py |
| degen median | 70.3% (0.578–0.859) | Report §8.4 |
| grad\_norm median / max / clip | 180 / 5229 / 1.0 | Report §8.4 |
| robospatial mean of three runs | SFT 213.0 · GRPO 226.7 · C′ 221.7 | Eval |
| VQA /228 | 162.3 · 164.3 · 168.0 | Eval |
| Vacant /122 | 50.7 · 62.3 · 53.7 | Eval |
| Point overrides / run (questions that only ask about the anchor object / run: 42.0 · 13.3 · 34.3) | 49.3 · 21.7 · 44.0 | Eval |
| Difference between the two arms sd / 2σ | 2.7 / 5.3 questions | Eval |

## 

## **Appendix B · Comparison with an external implementation: Maccchiatooo/spacetools-training-programs**

It likewise replaces SpaceTools' GRPO with GFlowRL (Qwen2.5-VL-3B, 8×B200). The algorithm itself is at lines 2375–2606 of `algo_gflowrl_core_algos.py`, verbatim identical to `algo_gflowrl_excerpt.py`.

| Item | Reference implementation | Ours | Conclusion |
| ----- | ----- | ----- | ----- |
| Hook point | `register_adv_est` computes g̃ \+ `register_policy_loss` | `compute_gflowrl_flow_gap()` on the driver \+ `register_policy_loss` | Equivalent, no change |
| Length normalization | Each rollout's log-probability ratio is divided by its length |y|, in all three places, Eq. 4, 6 and 8 (default `gflowrl_normalize_z_logp=True`), equivalent to our `normalized` variant |  Reference implementation (normalized) C′ Eq. 4 computing Z\_t within-group mean of (βr \+ d/|y|) within-group mean of (βr \+ d) Eq. 6 computing g drift term d/|y| drift term d, not divided Eq. 8 computing the loss (g̃ \+ log(π\_θ/π\_old)/|y|)² (g̃ \+ Σ\_t log(π\_θ/π\_old))² where d \= Σ\_t(log π\_ref − log π\_old), summed per token over the whole rollout. In code the only difference between the two is whether `masked_sum` is used or the result is divided by the length; changing this line raises no error, so if it is touched, the fixed-point self-check (at the fixed point the loss is 0 and the gradient is 0, so further training doesn't move the parameters) must be rerun. Why it's done this way: **The fixed point is correct**: the paper's Prop. B.1 proves π ∝ π\_ref·exp(βr) precisely under the premise of no length normalization. Our numerical self-check: C′ error 7.39e-13, pass; normalized 5.1e-2, fail. **Normalization distorts the target**: per the paper's Remark B.4, dividing by |y| is equivalent to replacing β with |y|·β (about 334×8 ≈ 2700). The target distribution becomes close to pure reward maximization, distribution matching is gone, and answers of different lengths effectively use different temperatures. The cost is that the gradient grows linearly with length, which is the main cause of the grad\_norm median of 180; P1(b) uses constant scaling (grad\_clip \= L̄) to bring the scale back.  | **Not adopted. It would change the training objective**: the paper's Remark B.4 points out that dividing each one by |y| is equivalent to replacing β with |y|·β. Our |y| is about 334, so the effective β is about 2700, and the target distribution puts almost all its mass on the highest reward; it becomes pure reward maximization and distribution matching is gone. **The self-check also fails**: constructing the target distribution and computing the loss, normalized error 5.1e-2, fail; C′ error 7.4e-13, pass. **We get its benefit another way**: the only benefit of dividing by |y| is a smaller gradient scale. We use constant scaling instead (grad\_clip \= L̄, see P1(b)), which brings the scale back just as well without changing the objective.  |
| Degenerate-group filtering | g̃ set to zero, no resampling | Implemented (P1(a)) | **Approach adopted**: g̃ is set to 0 for whole degenerate groups, no resampling (see P1(a)). On-policy this is bit-for-bit identical to actually masking these groups out with a loss mask. **Criterion for degenerate groups: not adopted**: Reference implementation: a group counts as degenerate if its within-group variance ≤ 1e-8. The variance is computed as E\[r²\] − E\[r\]², and subtracting two close numbers has floating-point error; with continuous rewards, a group whose 5 rewards are exactly identical may come out with a variance slightly above 1e-8 and be misjudged as non-degenerate. Ours: a group counts as degenerate only if its within-group max minus min is exactly 0, without this error.  |
| flow-gap ε | 2.7 / 3.8, sat≈0 | 0.2 / 0.28, sat median 52% | **Not adopting the** 2.7 / 3.8 **values**: with 2.7 / 3.8, g is almost never clipped, which is equivalent to no clipping. In the paper's Table 7, the no-clipping version loses 3.9 in average score, with grad\_norm 6.3× the original. **We agree with its judgment**: the 0.2 / 0.28 in Table 9 is very likely meant for GRPO's ratio clip, not the bounds on g. **We additionally found a direction problem**: both it and we follow the literal form, under which the upper limit for pushing good samples up is actually smaller. This is verified with C (ε swap) of P1(b).  |
| β | | β | **2.5**. The paper's β=8 was set for 0/1 rewards; its reward is a 0–10 composite score, so it was converted to 2.5 by "keeping the standard deviation of β·r consistent with the paper"  | **8**, consistent with the paper, because our rewards are also in 0–1. The current rew/drift median is 2.01, so the reward term dominates the drift term; hence it doesn't apply: the reward ranges on the two sides differ, so 2.5 can't be carried over directly  | Not applicable |
| IS weight | **Per-token average**: w \= exp(mean of the per-token log-probability ratios), capped at 2.0. Corresponds to "Geometric" and "IS threshold 2.0" in the paper's Table 9 | **Sum over the whole sequence**: w \= exp(sum of the log-probability ratios of all tokens), capped at 1.2 (= 1 \+ ε\_is) | **No effect for now**: we are strictly on-policy, so w is always 1. **If we split mini-batches later, switch to its approach**, with the cap changed to 2.0. The reason is that in the sum version the exponent is the total of the per-token log-probability differences; the longer the sequence, the larger the total and the further w moves from 1. A tiny deviation per token, multiplied over a few hundred tokens, can push w down to the order of 0.02, and that sample then contributes almost no gradient  |
| G / batch / epochs | 16 / 63 / 4 (348 steps) | 5 / 64 / 1 (85 steps) | See P1(c) / P1(d) |
| Logging | **Recorded**: quantiles of |g|, sat (fraction clipped), degenerate-group fraction. **Validation during training**: a full robospatial 350-question run every 20 steps (TEST\_FREQ=20) | **Recorded**: sat, β decomposition (reward term / drift term), sign\_ok, degenerate-group fraction, IS-weight guard. **Validation during training**: none (test\_freq=-1) | Adopted, already added, see P1(a)  |
| Tool packing | Before changing tool actor counts or fractions, run `check_tool_packing.py` to check the sum of `num_gpus` (its config sums to exactly 2.000, zero headroom) | During eval, vlm was manually raised to 1.0 to avoid OOM | **Idea adopted, implemented:** infra\_prep/check\_tool\_packing.py, run before training. Besides the share total, it also checks shares per GPU and the GPU memory corresponding to the shares |
| Run results | 39 steps, reward slope t=−0.25, degenerate groups 53.3% (G=16) | 85 steps, no trend, 70.3% (G=5) | **Neither side has learned yet**: it ran 39 steps with no upward trend in training reward (slope t \= −0.25); we ran 85 steps, also with no upward trend. Split the questions into two kinds: **Moderate-difficulty questions**: the model is sometimes right and sometimes wrong. At G=5, cases where all 5 happen to be right or all wrong are common; at 16, there is almost always a mix, so these questions are basically no longer degenerate. **Extreme questions**: the model is always right or always wrong. No matter how many are sampled, the whole group is all right or all wrong. Increasing G only rescues the first kind, not the second. Measured at G=16, 53% of groups are still degenerate, showing that most of the degeneracy comes from the second kind. So increasing G has limited effect and isn't worth prioritizing. |

