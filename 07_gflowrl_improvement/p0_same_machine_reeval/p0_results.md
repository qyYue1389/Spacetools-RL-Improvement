# P7 P0 results: three-arm same-machine, same-KV-pool re-eval and point-override attribution

Run on 2026-09-23 · 4× RTX 6000 Ada · per the P0 plan in `P7 problems and optimization directions (superseded by docs/design_doc_gflowrl_optimization.md)`
Raw artifacts: `/workspace/exp/p0/` (dumps of the 12 runs), `/workspace/parsed/` (enriched records)

---

## 0. In one sentence

**After running each of the three checkpoints 3 times on the same machine, in the same session, with the same KV pool: overall P4 (GRPO) and C′ can't be told apart (p=0.28), but once robospatial is split into VQA and Vacant, two significant effects in opposite directions surface — C′ is slightly better on VQA and significantly worse on Vacant (p=0.0004); the entire Vacant gap is explained by one behavior, "changing the point the tool returned", and that behavior is inherited from the SFT starting point: GRPO suppressed it, C′ did not.**

---

## 1. How it was run

| | |
|---|---|
| Machine | 4× RTX 6000 Ada (sm_89, 49140 MiB, all four GPUs same capacity, no mixed ECC) · Ubuntu 24.04 · driver 610.43.02 |
| Environment | `qyYue1389/spacetools-eval-env`, all four `VERIFY.sh` checks pass |
| KV pool | Whole GPU 47 GB × `gpu_memory_utilization=0.511` = **24.5 GB**, identical across all three arms |
| Scale | 12 independent invocations · 5097 samples · about 24 minutes each · about 4 h 10 m in total |
| Order | run1 (sft→p4→cp) → run2 → run3 → ref1, **interleaved by run**, not grouped by ckpt |
| dump | All kept |

Provenance of the three arms (pinned by revision):

| Arm | repo | revision |
|---|---|---|
| SFT starting point | `qyYue1389/spacetools-sft-v1-4xa6000` | `91fd4bdf` |
| P4 official GRPO | `siyich/spacetools-ckpt` | `f953b1a1` |
| C′ step85 | `qyYue1389/spacetools-p7-gflowrl-cprime-8xa40` | `global_step_85` |

### 1.1 Two deviations (must be recorded in PROVENANCE)

**① `vlm`'s `num_gpus` 0.6 → 1.0.** On the first blinkdepth run Ray packed `vlm 0.6 + depth 0.2 + depth 0.2 = 1.0` onto the same GPU: Molmo fp32 32.11 + DepthPro 7.94×2 = 48 GiB, while torch-visible capacity is 47.40 GiB. **88 OOMs were wrapped by toolshed into normal `ToolResult`s and swallowed** (README hole #22), and blinkdepth silently dropped to 66.94% (normal band 85–88%). Raising `vlm` to 1.0 forces Molmo to have a GPU to itself, and the GPU memory distribution went back to what it looked like in the healthy P7 run (18.7 / 31.3 / 9.1 / 43.1 GB). **This only changes Ray's placement hint, not any computation.**

`ulimit -c 0` was added as well: P7_GPU_RESULTS recorded one OOM that wrote a 50 GB core and filled the container disk.

**② Config patch for the official checkpoint.** `siyich/spacetools-ckpt` was published **without** the fixes that upstream `docs/SETUP.md` documents and that `run_sft.sh` applies automatically to self-trained ckpts; loading it directly dies with `RuntimeError: Unimplemented model type: qwen2_5_vl_text`. The same script P4 used back then, `spacetools-repro/tools/fix_checkpoint.py`, was applied as-is, making three changes:

- delete `text_config` from `config.json` (with it present, transformers 4.57.1 resolves `AutoConfig` to the text sub-model)
- set `tie_word_embeddings=True` (the script first compares `lm_head.weight` and `embed_tokens.weight` bit-for-bit and sets it only after confirming they are identical)
- replace `preprocessor_config.json` with the base model's non-Fast version (the Fast processor's image token count doesn't match the vision encoder)

**Not a single byte of the weights was touched**; the originals are backed up as `*.orig`. The environment versions are exactly the same as P4's back then (sglang 0.5.6 / transformers 4.57.1, see `records/env-freeze/spacetools-rl.txt`), so this is not version drift; the published ckpt itself is missing the fix.

---

## 2. Health gate

**Infrastructure layer all zero:** all 27 dumps pass `parse_dump.py --strict`; OOM 0, truncation 0, malformed tool_call 0.

**Model-behavior layer: 6 cases** (0.12% of about 5000 samples, distribution **SFT 3 / P4 3 / C′ 0**):

```
grasp_generator.compute_grasp  ×3   calls the grasp tool on blinkdepth (depth questions)
laser_radar.detect_one         ×1   tool does not exist (not in the v1 tool set)
depth_estimator.index_at       ×1   method attributed to the wrong tool (index_at belongs to vision_ops)
vision_ops.index_at            ×1   passes the string "$point_cloud" as a variable (an old acquaintance recorded in P4/P5)
```

Checked one by one: **none of them falls in a reported disagreement pair**; no conclusion is affected. The only borderline one is blinkdepth #22 (SFT run1 errored, pushed toward wrong, in P4's favor), and the conclusion for that pair is "no difference" anyway (p=0.85).

**This is not pipeline contamination; it is a tool-discipline problem.** When P4 reported "health metrics all zero without exception", that used the infrastructure-layer definition; recommend reporting the two layers separately from now on.

---

## 3. Main table

| | robospatial /350 | Always right | Always wrong | Flip | blinkdepth /124 | Always right | Flip | RefSpatial /277 |
|---|---|--:|--:|--:|---|--:|--:|--:|
| SFT starting point | 215 · 211 · 213 → **213.0** | 185 | 108 | 57 | 107 · 109 · 108 → **108.0** | 104 | 7 | 147 |
| P4 official GRPO | 223 · 229 · 228 → **226.7** | 205 | 102 | 43 | 105 · 110 · 111 → **108.7** | 103 | 9 | 149 |
| C′ step85 | 223 · 221 · 221 → **221.7** | 203 | 105 | 42 | 111 · 113 · 110 → **111.3** | 109 | 5 | 148 |

**Cross-machine comparability confirmed:** the SFT starting point's 215 / 211 / 213 nearly coincide with 216 / 211 on the old machine (A6000, sm_86). Changing GPU, architecture and machine did not distort anything.

**RefSpatial is a three-arm tie** (147 / 149 / 148), confirming §10.3 "this one has no policy freedom".

**Resolution (now quantified):** sd of the mean of three runs = √(flip band/3)/2 → on robospatial SFT 2.18 / P4 1.89 / C′ 1.87 questions; **sd of the difference between two arms ≈ 2.7 questions, 2σ ≈ 5.3 questions**. P4 alone varies by 6 questions across single runs (223 vs 229); **a single-run reading is still unusable**.

---

## 4. Per-sample paired tests

Both methods are listed; the truth is in between:

- **Stable core**: only samples that are consistent across all 3 runs on both sides. Conservative — drops samples that "wobble under A and become always-right under B".
- **Three-run paired pooling**: compare run i with run i per sample, pool the disagreements of the three runs. Uses all the data, but the same samples are counted three times, **optimistic**.

| Comparison | Stable-core McNemar | Three-run paired pooling |
|---|---|---|
| robospatial SFT→P4 | net +10, p=0.052 | 71 vs 112, net +41, **p=0.003** |
| robospatial SFT→C′ | net +3, p=0.250 | 47 vs 73, net +26, **p=0.022** |
| robospatial P4→C′ | net −5, p=0.405 | 93 vs 78, net −15, p=0.284 |
| blinkdepth SFT→P4 | net −2, p=0.500 | 13 vs 15, net +2, p=0.851 |
| blinkdepth SFT→C′ | net +1, p=1.000 | 3 vs 13, net +10, **p=0.021** |
| blinkdepth P4→C′ | net +3, p=0.250 | 6 vs 14, net +8, p=0.115 |

---

## 5. The key step: split into VQA and Vacant

Overall P4 and C′ can't be told apart (p=0.284). **Once split, it is two effects in opposite directions cancelling each other out.**

| | VQA /228 | Vacant /122 |
|---|---|---|
| SFT starting point | 163 · 160 · 164 → **162.3** | 52 · 51 · 49 → **50.7** |
| P4 official GRPO | 160 · 166 · 167 → **164.3** | 63 · 63 · 61 → **62.3** |
| C′ step85 | 167 · 167 · 170 → **168.0** | 56 · 54 · 51 → **53.7** |

| Comparison | VQA (three-run pooled) | Vacant (three-run pooled) |
|---|---|---|
| SFT→P4 | 56 vs 62, net +6, p=0.65 | 15 vs 50, net +35, **p=0.00002** |
| SFT→C′ | 42 vs 59, net +17, p=0.111 | 5 vs 14, net +9, p=0.064 |
| P4→C′ | 54 vs 65, net +11, p=0.359 | 39 vs 13, net **−26**, **p=0.0004** |

**How to read it: C′ is 3.7 questions better than GRPO on VQA, which needs reasoning (not significant), and 8.6 questions worse than GRPO on Vacant, which is pointing (significant).** The latter is larger, so overall P4 leads by 5 questions; the two effects point in opposite directions, so taken together no significance shows up. Part of §10's original "none of them significant" came from this.

> **Methodology: from now on results must be read separately for VQA / Vacant.** This is the most transferable takeaway of this run — pooling subsets hides real effects in opposite directions.

---

## 6. Point-override attribution (the P2 question)

Criterion: whether **all** coordinates in the answer appear in the set of coordinates returned by the tool (exact match at 3 decimal places). Averaged over three runs.

**The classification has been calibrated:** on P4 it reproduces the **pass-through 101 / point override 21 / point-override accuracy 28.6%** recorded in §10.6, matching the report.

| | Pass-through (per run) | Pass-through accuracy | Point override | Point-override accuracy | Override on the 0.05 grid |
|---|--:|--:|--:|--:|--:|
| **SFT starting point** | **72.7** | 56.9% | **49.3** | 18.9% | 18.7 |
| P4 official GRPO | **100.3** | 56.1% | **21.7** | 27.7% | 7.0 |
| C′ step85 | 77.0 | 57.6% | 44.0 | 21.2% | 18.3 |

### Conclusion: it's "GRPO suppressed it, C′ didn't", not "C′ learned to override points"

**The SFT starting point already overrides 49.3 points, even more than C′'s 44.0.** GRPO pushed it down to 21.7; C′ barely moved it. The 0.05-grid fingerprint (the "eyeballing" signature quantified in P6) gives the same ordering: starting point 18.7 → GRPO 7.0 → C′ 18.3.

**Counterfactual accounting:** if C′ kept GRPO's pass-through ratio, Vacant = 100.3×57.6% + 21.7×21.2% ≈ **62.4 questions**, and P4 measured **62.3 questions**. The 8.6-question gap between P4 and C′ on Vacant is explained, with nothing left over, by the pass-through/override ratio.

### This changes three things

1. **C′'s objective does not "encourage point override".** Listing it earlier as "the only measurable harmful change of C′ relative to GRPO" was reading it backwards.
2. **It becomes direct evidence for P1 (signal starvation).** The point-override error rate is about 79%, which the reward should strongly suppress; GRPO suppressed it with the same data, same G=5, same 1 epoch, C′ did not — two sides of the same thing as the readings in §8.4 (70.3% of groups degenerate, gradient reduced to just the drift term, grad_clip=1.0).
3. **Don't write reward shaping.** Since GRPO can suppress it under the same config, this is a signal problem, not a missing reward term; writing a heuristic into the reward would mask P1's effect. Downgraded to "consider only if it still persists after P1".

### C′ is not behind across the board

On the same set of dumps: C′ is +3.7 questions on VQA, +2.6 questions on blinkdepth, has the narrowest flip band (robospatial 42, blinkdepth 5, lowest of the three arms), and 0 model-side tool misuses (SFT and P4 have 3 each). **C′'s problem is concentrated in one specific pointing behavior.**

---

## 6b. Tool calls: actual depth_estimator call counts (recomputed 2026-09-23 from this batch of dumps)

This is the first time there are SFT-starting-point dumps to check this with.

| robospatial (n=350) | depth_estimator calls | Total tool calls |
|---|---|---|
| SFT starting point | **0 · 0 · 0** | 577 · 577 · 577 |
| P4 official GRPO | 2 · 2 · 1 | 576 · 575 · 573 |
| C′ step85 | 0 · 0 · 1 | 578 · 576 · 579 |

**The starting point itself doesn't call it.** It's not that the two RL arms suppressed it; this chain was never in the support set — §10.4's judgment holds, P3 stays last.
(One correction: §10.4's "SFT starting point is 1 call" came from a single run on the old machine; three measured runs give 0/0/0.)

**Two pieces of side evidence:**

1. **On robospatial only one of the seven tools, roborefer, is used** (all 578 calls are to it, 1.65 calls/sample); the other six are never used.
   This makes §6's point-override problem clearer: the model only has the one point roborefer gives it; changing it is pure eyeballing, with no second information source to cross-check against.
2. **Compare blinkdepth (n=124)**: depth_estimator calls SFT 121/122/120, P4 124/123/123, C′ 122/122/122 — almost once per sample.
   **The model does call this tool; it just doesn't think of it given robospatial's question wording.** This confirms it as a "data distribution problem", not a capability problem.

How it was recomputed: after `parse_dump.py --emit-slim`, aggregate the `tool_stats` of each record; zero GPU.
Note that the output file name of `--emit` is taken from the parent directory name of the input path; for this batch it is `dumps.jsonl`; its `correct` field
cannot be used directly because the benchmark name is inferred differently; `tool_stats` is unaffected.

---

## 7. Two monitoring quantities to add during training (no GPU needed)

- **Point-override rate** on Vacant / pointing-type questions: target moving from the starting point's ~40% toward GRPO's ~18%.
- Share of answer coordinates that fall on the **0.05 grid**: if it rises, the model is learning to "eyeball it itself".

---

## 8. Not done

- **bopgrasp scoring sign: decided not to change it, marked "unusable"** (2026-09-23). `parse_dump.py` uniformly uses `correct = score ≥ 0.5`, while bopgrasp's `score` is RL's NCE (lower is better, mean about 2.0, see `records/CHANGES.md` §7) — that row is not accuracy, and its sign is inverted (the worse the grasp, the more likely it is judged correct). **But changing it to `NCE ≤ threshold` only swaps one wrong number for another questionable one**: P5 showed NCE is dominated by position and SR by orientation, and the two rank in opposite orders (fallback answers with a median orientation error of 63.9° actually have lower NCE). The right approach is to report continuous NCE + a sign test, or split into MACE / SR, and only do it if bopgrasp is really needed for a GRPO vs C′ comparison. For now bopgrasp is only side evidence that "the environment reproduces correctly" and does not feed into any conclusion.
- RefSpatial was run only 1 time (disagreements are very rare anyway: sft↔p4 4 in total, p4↔cp 3 in total).
- P2's "forced pass-through upper-bound check" was not done.

---

## 9. Recomputing

```bash
# health gate
python parse_dump.py --strict /workspace/exp/p0/<arm>/<run>/<bench>/0.jsonl

# main table + flip band + per-sample pairing
python /root/flips.py        # stable-core McNemar + three-run interval
python /root/paired.py       # three-run paired pooling + sd
python /root/split2.py       # VQA / Vacant split
python /root/p2.sh           # pass-through / point-override table
```

All scripts only consume dumps; zero GPU.
