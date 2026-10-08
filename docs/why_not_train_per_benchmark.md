# Why we can't train per benchmark

Written 2026-09-15 · Trigger: to save GPUs, the idea "when training a given benchmark, deploy only the tools that benchmark
needs" came up · Related: `GFlowRL/README.md`, `RL training-prep handoff doc (not included)`

---

## Conclusion

**Mechanically it can be done, but it shouldn't be.** Four reasons, any one of which is enough on its own; reasons 2 and 3 are hard ones,
and reason 4 shows it doesn't even achieve its original purpose of saving money.

Section 6 at the end records **the only variant of this idea worth evaluating on its own**, and section 7 an alternative that gets the same GPU savings
with zero deviation.

---

## 1. Mechanically it can be done — there is a precedent

The tool set is not constrained through the prompt; **only the wanted tools are registered when the tool service starts**:

```
examples/toolshed/run_rl_roborefer.sh      ← this is exactly what Step 1 does, registers only roborefer
examples/toolshed/generate_toolshed_config.py
       queries the tool schemas back from the running service, then generates the YAML handed to verl
```

So unregistered tools **really do disappear from the `<tools>` section of the system prompt**, and the model can't produce that tool name;
even if it invents a call out of thin air, there is no server to receive it and it errors out directly.

This section only means: the three objections below are **not "it can't be done"**, they are "doing it breaks things".

---

## 2. There is no such thing as "the training set of a benchmark"

This is a mismatch at the level of the premise, not a trade-off.

```
training data    spacetools-rlfulltools
run_rl.sh:335   RL_PARQUET="$EXPERIMENT_DIR/rl_data/data/train.parquet"
run_rl.sh:439   data.train_files="[$RL_PARQUET]"        ← one pooled file
```

**A benchmark is an eval-side concept** — nine keys, each with its own `data/<key>.parquet`, read by Step 5's
`run_eval.sh`. The training side has only one mixed pool, `train.parquet`.

So the first step of "training per benchmark" would be **splitting the training pool yourself**, and:

- It's unclear what to split on. Whether `train.parquet` has a column that maps to a benchmark (`data_source` or similar)
  has not been verified.
- It's unclear how many samples each piece would have left. Full set 5425 samples / `train_batch_size=64` ≈ 85–86 steps = 1 epoch.
  After splitting into N pieces each might have only a dozen or so steps left, **not enough to train**.
- Even if it could be split, what comes out is "a subset of the training distribution", not "that benchmark's training set".
  The two are not the same thing.

---

## 3. It turns Step 4 back into Step 1

**The whole point of Step 4 is learning tool orchestration** — choosing among 11–17 tools, chaining them, recovering when something fails.
Deploy only the tools a subset needs and the model has nothing to choose from; the search space is artificially collapsed.

The paper's own ablation points the direction here (Table 4):

```
full four-step pipeline                        52.48
full-tool GRPO directly from the base model    19.79     ← search space too large, can't learn
```

Step 1 **opens only one tool** precisely because the search space must be small enough that "random tries still hit the right answer", so that degenerate groups don't
stall training. Step 4's premise is exactly the opposite: the model already knows how to use the tools, and now all tools are opened
so it can explore orchestrations better than the demonstrations on its own.

**Shrinking the tool set back to a subset swaps Step 4's goal for Step 1's goal.** What you get is a set of narrow experts,
while Step 5's eval wants **one model**'s performance across nine benchmarks.

---

## 4. Prompt distribution mismatched with eval, plus catastrophic forgetting

The tool list is injected into the system prompt, and `generate_toolshed_config.py` **queries it back** from the running service
— whatever is deployed is what the prompt says.

So staged training has two knock-on consequences:

**① The final ckpt is trained under a prompt distribution that doesn't exist at eval time.**
In each stage the model sees a different `<tools>` section, i.e. a **different task format**. Step 5's eval always
opens all tools, and the prompt has the complete list. A model that never saw the complete list in training faces
an unfamiliar format at eval.

**② Sequential training forgets.** Tool usage learned in an earlier stage has neither the corresponding tool to call
nor a reward to maintain it in a later stage. The last stage's tools get over-reinforced; the earlier ones degrade.

Taken together: **the last checkpoint from staged training is not "a model that has learned all the tools",
but "a model that has just finished learning the last subset".**

---

## 5. The cost is actually higher

The original intent was to save GPUs. But the full bill has to be added up:

```
saved     fewer tools deployed per training run → tool side uses 1–2 fewer GPUs
paid      N× as many training runs, N = number of subsets
```

What the tool side saves is **GPUs per run**, while splitting costs **number of runs**. The latter is multiplicative.

And the savings themselves are limited: the bulk of tool GPU memory is three models (see section 6), and most
benchmarks need at least pointing, so `roborefer`, one of the biggest, can't be cut.

---

## 6. The only variant worth evaluating on its own: can `vlm` be dropped

The genuinely valuable part of this idea is not "per benchmark" but **"is the tool that takes the most GPU memory actually required"**.

Per-instance GPU memory (measured in SFT eval, GiB):

```
vlm (Molmo fp32)     30.2      ← single largest
roborefer            17.2
depth_estimator       7.9
sam2 / bbox / grasp   1.3 / 0.5 / 1.0 each
vision_ops            0
```

The layout compressed to 13 actors totals **86.0 GiB**, of which `vlm` is 35%. Dropping it:

```
86.0 → 55.8 GiB     one 80 GB GPU fits roborefer×2
                    or two 48 GB GPUs can give roborefer more replicas
```

This resolves exactly the pain point "with few GPUs `roborefer` can only keep 1 replica" — and `roborefer` is the most
frequently called tool; its concurrency directly determines the time per step.

**But two things stand in the way:**

**① You can't assume it's unused.** In the system prompt of `p6/passk`, `vlm` and `roborefer` are **parallel
pointing options**:

```
{"name": "vlm.detect_one",       "description": "Detect *one* instance of *obj_name*..."}
{"name": "roborefer.detect_one", "description": "Detect *one* instance of *obj_name*..."}
```

The model may well have taken `vlm` on some of the samples. This is **checkable**: count the share of `vlm.*` among the
`<tool_call>`s in the SFT starting-point trajectories. The data is in `eval/SFT/sft-eval-artifacts/rollouts/`
— note that `p6/passk/*/0.jsonl` **is not enough**: it contains only the prompt and the final turn,
the tool calls of the intermediate turns are not in it.

**② Even if the share is 0, this is still "changing tools".** The user decided on 2026-09-02 not to do that, the reason being
"changing the objective and changing the tools are orthogonal; mixing them makes results unattributable". And the 84.7% in the P6 attribution is exactly
tool errors + tool-set gaps — touching the tool set touches the foundation of that attribution.

**So either don't do it, or run it as a separate experiment.** ("Step 4 with only pointing tools open" is a
clean question in its own right, but that belongs to a different paper.)

---

## 7. Saving GPUs with zero deviation

Don't touch the tool set; only reduce `num_actors`:

```python
# roborefer 6→2 · vlm 2→1 · sam2 5→2 · depth 5→2
# bbox 5→2 · grasp 5→2 · vision_ops 8→2
# 13 actors total, concurrency drops only 2.8×, logical GPU demand 7.8 → 3.0
```

**This has zero deviation** — the tool list is unchanged, the prompt is unchanged, not a single number changes; rollouts just queue.
Its GPU savings overlap heavily with "deploy fewer tools", and the only cost is time, and the time cost **can be measured with a three-step
smoke test** (look at the actor busy/idle ratio in the Toolshed log).

⚠ When reducing `num_actors` you must raise `TOOL_GPUS` accordingly: Ray's `num_gpus` is a **logical reservation**, not a GPU memory
quota. When the PG reserves less than the total the actors request, the PG becomes ready but the actors can never be scheduled
— there is an assert in the `run_rl.sh` heredoc that catches this.

---

## Sources cited

| Claim | Source |
|---|---|
| Tool set is decided by registration; single-tool precedent exists | `examples/toolshed/run_rl_roborefer.sh`, `generate_toolshed_config.py` |
| Training data is one pooled parquet | `run_rl.sh:335`, `run_rl.sh:439` |
| Eval has one parquet per benchmark | `交接 RL 训练准备.md` §6 data |
| 5425 samples / 85–86 steps = 1 epoch | Same as above |
| Skipping the first three steps gets only 19.79 | Paper Table 4 |
| Per-instance GPU memory | Measured in SFT eval, `sft-eval-artifacts/gpu/gputrace_1hz.log` |
| `vlm` and `roborefer` are parallel pointing options | System prompt of `p6/passk/robospatial/0.jsonl` |
| `p6/passk` does not contain intermediate-turn tool calls | File structure checked on 2026-09-15 |
| Changing tools has been ruled out | User decision on 2026-09-02, recorded in `交接 RL 训练准备.md` §3 |
| Reducing `num_actors` has zero deviation / `TOOL_GPUS` assert | `交接 RL 训练准备.md` §4.1, §5.8 |
