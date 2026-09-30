# 第二组 5 次采样(passk2)· 2026-09-02

第一组在 `p6/passk/`(4×A100-**80GB**,`gmu=0.25`);本组在 4×A100-**40GB**,`gmu=0.50`。
**两者的 KV 池都是 20 GB** —— 这是两组唯一的可比性依据(偏离 `[23]`:`gmu` 是占整卡的比例)。

    n=5 · T=1.0 · top_p=1.0 · model_dtype=bf16 · calculate_log_probs=True
    max_num_seqs=64   <- 与第一组不同,见下

**⚠ 怎么读才不会读错:**

1. **`max_num_seqs` 256 -> 64 是本组新增的偏离。** 用 256 时 `robospatial` 在 40 GB 上
   OOM(崩在**视觉编码器**,不是 KV cache)。降它砍激活峰值而**不动 KV 池**。
   代价是 batch 组成会变 —— 但跨硬件 80->40 GB 本来就引入同类漂移,归入同一条限定。
2. **本组测到的散布 = 采样噪声 + 跨硬件漂移**,是采样散布的**上界**,不是纯采样散布。
3. **两组都不是干净全零**:组1 有 `malformed tool_call 1`,组2 有
   `cut-off generation 1` + `no <answer> 1`,各 1/1750。两组都过 `--strict`。
   **报的时候两组都要提。**
4. dump 里多了六个字段(`response_mask_rle` / `n_policy_tokens` / `n_response_tokens` /
   `response_token_ids` / `rollout_log_probs` / `trainer_log_probs`),
   来自 `patches/rl/0009`。`rollout_log_probs` 与 `response_mask` **逐位对齐**。

结果见 `records/P7_GPU_RESULTS.md` §3–§7。
