# P7 step 4 results: definitions of `response_mask` and `|y_i|`

> `P7_DECISION.md` §6 step 4. **Zero GPU.** Written 2026-09-01.
> This step is a **correctness prerequisite** for the GPU forward pass: the `log π_ref` / `log π_old`
> that the forward pass computes must cover assistant tokens only; if the mask is wrong, the whole `Z_t` decomposition is worthless.

---

## Summary

**`|y_i| = response_mask.sum()`, and it really does count only policy-generated tokens** — this time confirmed
line by line from the verl source, not inferred backwards from how FlowRL uses `masked_mean`.

**But this step dug up two things we did not know before, and both change what we do next:**

1. **The name `response_mask` has two opposite meanings in the same codebase.**
   The one the agent loop provides = "policy-generated tokens"; the fallback in `ray_trainer.py`,
   `compute_response_mask()` = `attention_mask[:, -L:]`, **has all tool-response tokens set to 1**.
   Using the wrong one inflates `|y|` by 1.14–2.40× (step 2 already measured this factor).
2. **`p4/dumps/` stores only text, not token ids or masks.**
   So running a forward pass over recorded trajectories requires **re-deriving the mask from text** — exactly what P3 explicitly forbids.
   **The way out is to add a dump patch before the GPU session for the second sampling batch**, see §4.

---

## 1. Source evidence: which tokens does the mask actually count

`SpaceTools-RL @ f0742338` · verl 0.8.0.dev

    verl/experimental/agent_loop/tool_agent_loop.py
      L287-289   agent_data.response_ids = output.token_ids            <- what sglang actually generated
                 agent_data.response_mask += [1] * len(response_ids)

      L430-431   response_ids = apply_chat_template(add_messages, ...)  <- tool response, templated as one block
                 agent_data.response_mask += [0] * len(response_ids)

      L462-463   same as above, interaction (user) branch

    verl/experimental/agent_loop/agent_loop.py
      L291-317   apply_chat_template(..., add_generation_prompt=True)
      L631       response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
      L463 comment   "1 for LLM generated tokens, 0 for observation/padding tokens"

**Three inferences, all from the lines above:**

- **`mask = 1` ⟺ tokens emitted by sglang itself.** No templated content at all.
- **The next turn's `<|im_start|>assistant\n` falls inside a `mask = 0` block** —
  because of `add_generation_prompt=True`, it is the tail of the tool-response template block, not generated.
- **Padding is also 0** (L631 multiplies by `attention_mask` once more).

> This also explains, at the token level, the root cause of that P3 bug: **the first turn's
> `<|im_start|>assistant` is in the prompt** (the initial `apply_chat_template` also uses
> `add_generation_prompt=True`), so `output` starts **in the middle of the body** of the first assistant turn.
> The `leading_is_assistant` handling in `parse_dump.py` is consistent with the token facts here.

---

## 2. ⚠ Same name, different meaning: two `response_mask`s

    verl/trainer/ppo/ray_trainer.py
      L111-126  def compute_response_mask(data):
                    return attention_mask[:, -response_length:]        <- tool tokens are also 1
      L157-158  if "response_mask" not in data.batch.keys():
                    data.batch["response_mask"] = compute_response_mask(data)   <- silent fallback

    verl/experimental/agent_loop/agent_loop.py
      L800      "response_mask": response_mask,   # [bsz, response_length]  <- the one we actually want

SpaceTools goes through the agent loop, so the batch **already contains** the correct one and the fallback does not trigger.
**But it is silent**: if one day the batch lacks this key (switching rollout path, changing dataproto,
assembling a batch ourselves for an offline forward pass), `compute_response_mask()` will **without any error** return a mask with the opposite meaning.

> **This is yet another instance of the CHANGES.md §10 rule "do not infer runtime behavior from static declarations",
> and this time the two things even share the same name.**

**Prescription — a runtime guard written into the loss** (must hold on multi-turn samples):

    # In a multi-turn trajectory, policy tokens are always strictly fewer than all non-padding tokens
    L_resp = response_mask.size(1)
    n_policy = response_mask.sum(-1)
    n_nonpad = attention_mask[:, -L_resp:].sum(-1)
    multi_turn = (num_turns > 2)
    assert (n_policy[multi_turn] < n_nonpad[multi_turn]).all(), \
        "response_mask looks like the attention-mask fallback: |y| would include tool tokens"

On single-turn samples the two being equal is normal, so the criterion has to be applied to multi-turn samples.
Per step 2's measurements, if the guard fails, `|y|` is inflated by **1.14× (robospatial) / 1.41× (blinkdepth) /
2.40× (boppose)**, and **systematically differently per benchmark**.

---

## 3. Structural check: all of `p4/parsed/`

| benchmark | n | turn-count cross-check | truncated | turns exhausted | assistant turns mean/min/max | tool turns mean |
|---|--:|--:|--:|--:|--:|--:|
| `blinkdepth` | 124 | 124/124 | 0 | 0 | 3.13 / 1 / 5 | 2.13 |
| `bopgrasp` | 60 | 60/60 | 0 | 0 | 4.95 / 4 / 5 | 3.95 |
| `boppose` | 60 | 60/60 | 0 | 0 | 5.03 / 5 / 6 | 4.03 |
| `cvb2drelation` | 650 | 650/650 | 0 | 0 | 2.06 / 2 / 6 | 1.06 |
| `cvb3ddepth` | 600 | 600/600 | 0 | 0 | 3.04 / 3 / 5 | 2.04 |
| `reflocation` | 100 | 100/100 | 0 | 0 | 2.00 / 2 / 2 | 1.00 |
| `refplacement` | 100 | 100/100 | 0 | 0 | 2.00 / 2 / 2 | 1.00 |
| `refunseen` | 77 | 77/77 | 0 | 0 | 2.00 / 2 / 2 | 1.00 |
| `robospatial` | 350 | 350/350 | 0 | 0 | 2.00 / 2 / 2 | 1.00 |
| **Total** | **2121** | **2121/2121** | **0** | **0** | | |

**`assistant turns = tool turns + 1` holds on all nine benchmarks.**
So the mask shape is strictly alternating `1…1 / 0…0 / 1…1 / … / 1…1`, ending with a 1-block —
no consecutive assistant turns and no trailing tool turn. This matches the agent loop structure.

Recompute: `python3 -c` over the `num_turns_agree` /
`num_turns_derived` / `num_user_turns_derived` fields of `p4/parsed/*.jsonl` (fields produced by `parse_dump.py`).

---

## 4. ⚠ This step changes the spec of the GPU forward pass

**Problem: `p4/dumps/` and `p6/passk/` store `input` / `output` text, with no token ids and no mask.**

So "run one forward pass over recorded trajectories" actually requires:

1. Re-tokenizing the text — but `output` was decoded with `skip_special_tokens=False`,
   and re-encoding is **not guaranteed** to be bit-for-bit identical to the original token sequence (whitespace, special-token boundaries);
2. **Re-deriving the mask from text** — exactly what `P7_HANDOFF.md` §2① explicitly forbids:
   "do not rewrite the assistant/tool split yourself".

**The way out (recommended): before the GPU session for the second sampling batch, first add a P3-style dump patch.**
`patches/rl/0008` already shows how: defensively write two extra fields in `_dump_generations()`.
This time the fields to write are:

    base_data["response_mask_rle"]  # [[value, run_length], ...], run-length encoding
    base_data["n_policy_tokens"]    # = response_mask.sum(), i.e. |y_i|

Run-length encoding because a 4-turn trajectory has only about 8 runs, so the size is negligible, and the mask can be **rebuilt bit for bit**.
The data is in `test_batch.batch["response_mask"]` (the one the agent loop puts there),
in the same scope as `sample_turns` / `sample_uids`, so the patch has the same shape.

> **This way the forward pass consumes verl's own mask, not one we re-derived.**
> Cost: the second sampling batch must **be patched first, then run**, otherwise it has to be rerun after it finishes. **This is a scheduling dependency.**

---

## 5. Limitations

- **This step did not actually run the tokenizer**; the token count of `|y|` is still unmeasured (no tokenizer on this machine,
  `models/` is on the GPU machine). Step 2's character proxy is still the only order-of-magnitude basis we have.
- The guard in §2 **has not been run on a real batch**; it is written from reading the source and must be verified at the first training/forward run.
- The patch in §4 **is only specified, not implemented** (by agreement, no training-side code is written before the route is settled).
- The turn-count cross-check comes from `parse_dump.py`'s own fields, reconciled against verl's `num_turns`;
  it verifies **the number of turns**, not **token boundaries**. Token boundaries can only be verified once we have the real mask.
