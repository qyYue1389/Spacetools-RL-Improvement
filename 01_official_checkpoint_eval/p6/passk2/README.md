# Second group of 5 samples (passk2) · 2026-09-02

The first group is in `p6/passk/` (4×A100-**80GB**, `gmu=0.25`); this group is on 4×A100-**40GB**, `gmu=0.50`.
**Both have a 20 GB KV pool** — this is the only basis for comparability between the two groups (deviation `[23]`: `gmu` is a fraction of the whole GPU).

    n=5 · T=1.0 · top_p=1.0 · model_dtype=bf16 · calculate_log_probs=True
    max_num_seqs=64   <- differs from the first group, see below

**⚠ How to read this without misreading it:**

1. **`max_num_seqs` 256 -> 64 is a new deviation added in this group.** With 256, `robospatial` OOMs on 40 GB
   (crashing in the **vision encoder**, not the KV cache). Lowering it cuts the activation peak **without touching the KV pool**.
   The cost is that batch composition changes — but going across hardware 80->40 GB already introduces the same kind of drift, so it falls under the same caveat.
2. **The spread measured in this group = sampling noise + cross-hardware drift**; it is an **upper bound** on the sampling spread, not the pure sampling spread.
3. **Neither group is cleanly all-zero**: group 1 has `malformed tool_call 1`, group 2 has
   `cut-off generation 1` + `no <answer> 1`, each 1/1750. Both groups pass `--strict`.
   **When reporting, mention both groups.**
4. The dump has six extra fields (`response_mask_rle` / `n_policy_tokens` / `n_response_tokens` /
   `response_token_ids` / `rollout_log_probs` / `trainer_log_probs`),
   from `patches/rl/0009`. `rollout_log_probs` and `response_mask` are **aligned bit-for-bit**.

Results are in `records/P7_GPU_RESULTS.md` §3–§7.
