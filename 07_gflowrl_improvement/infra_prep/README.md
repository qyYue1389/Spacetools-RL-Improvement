# infra_prep: training-side infra optimizations (before the GPU session, no GPU)

2026-09-30. The objective is not changed; with default values the behavior is bit-for-bit identical to SpaceTools-RL `c6fef78a` as used in training.

## Files

| File | Purpose |
|---|---|
| `patched/` | The 5 patched files: `ray_trainer.py` (= c6fef78a + P1-a + drop degenerate groups + switch to not save images in dumps), `fsdp_workers.py`, `tool_agent_loop.py`, `run_rl.sh`, `run_rl_gflowrl.sh` |
| `infra_all_vs_c6fef78a.diff` | Full diff of the 5 files in `patched/` against c6fef78a, verified to work with `git apply` |
| `toolshed_tool_timing_vs_712e557.diff` | Toolshed-side timing patch (changes `toolshed/integration/verl.py`). Only the diff is included, not the full patched file (Toolshed uses its own license); `git apply` on SpaceTools-Toolshed `712e557` |
| `infra_drop_only.diff` | Only the "drop degenerate groups" item (relative to P1-a) |
| `infra_drop_check.py` | CPU self-check for dropping degenerate groups (needs torch) |
| `stage_timing.py` | Extracts all `timing_s/*` from the training log to find those 129 s |
| `nccl_bw_probe.py` + `preflight_nccl.sh` | GPU-session acceptance check: does P2P work, what is the all-reduce bandwidth |
| `dump_images_check.py` | CPU self-check for the switch to not save images in dumps |
| `check_tool_packing.py` | GPU-session acceptance check: do the tool actors fit per GPU, are the shares enough GPU memory (reads `TOOL_CONFIGS` from the run script, no GPU needed) |
| `tool_timing_check.py` / `tool_timing_summary.py` | Self-check / summary for per-tool timing |
| `stage_timing_p7_85steps.csv` | Per-stage timing of the 85 steps of P7 training |

## 1. Drop degenerate groups (`GF_DROP_DEGEN=true`, also requires `GF_FILTER_DEGEN=true`)

After rewards are computed and before old_log_prob, delete entire groups whose rewards are all identical; old_log_prob / ref / update_actor only compute the remaining rows.
Degenerate groups are 70% at the median; these three stages (about 369 + 122 + part of the unaccounted s) shrink with the row count.

Why the update is unchanged: dp_actor multiplies each micro-batch's loss by `1/gradient_accumulation`, and that = the configured
`ppo_mini_batch_size / micro` (80/2 per GPU), independent of the actual row count → per-GPU gradient = Σ loss of this GPU's rows / 80,
FSDP averages across GPUs → Σ all / 320, the same as P1-a (only zero out g~, keep all rows in the denominator).
Two constraints: pad the row count to a multiple of dp×micro (otherwise a 1-row micro-batch appears and that row's weight doubles; section 5 of the self-check has a counterexample);
pad with the shortest degenerate rows (under filter g~=0, they contribute no gradient).

Logs: `reward/degenerate_*` are still full-batch values; new `gflowrl_drop/{rows_before,rows_after,pad_rows,kept_rollout_frac,score_mean_full}`.
**Note** `critic/score/mean`, `perf/total_num_tokens`, `perf/throughput` etc. now count only the kept rows; for cross-run comparison use `gflowrl_drop/score_mean_full`.
The rollout dump (`trainer.rollout_data_dir`) still writes all 320 samples; `p1_monitor.py`'s readings are unaffected.

## 2. Keep the ref model on GPU (`REF_PARAM_OFFLOAD=False`)

Upstream `fsdp_workers.py` **forces** `CPUOffload(offload_params=True)` for ref, regardless of `ref.fsdp_config.param_offload` —
previously, setting that switch to False would not put ref on GPU either. The patch makes False actually take effect; default True behavior is unchanged.
Cost about 2 GB/GPU (4.07 B parameters in bf16 split over 4 GPUs). Measured (§3) it saves only about 5 s/step (<1%), **not recommended to turn on**; the patch is kept, off by default.

## 3. Those 129 s — resolved (2026-09-30, read from HF `provenance/full_train.log.gz`)

`python3 stage_timing.py <full_train.log>`, median over 85 steps (seconds):

```
gen            321.2   34.8%
old_log_prob   111.9   12.1%   ← missed in the report
ref            116.8   12.7%
update_actor   351.8   38.1%
save_ckpt       39.6    4.3%   (only on steps where save_freq hits)
update_weights   9.7    1.0%   ← missed in the report
other            0.3
step           922
```

The 129 s in step 85's 981 s = old_log_prob 118.8 + update_weights 10.0 + 0.3. Per-step results are in `stage_timing_p7_85steps.csv`.

Three conclusions from this:

1. **The part that scales with row count is old_log_prob + ref + update_actor ≈ 580 s (63%)**, larger than estimated earlier — this is exactly the block that dropping degenerate groups (§1) can cut.
2. **Putting ref on CPU costs only about 5 s/step extra** (ref 117 vs old_log_prob 112, which is also one forward pass with parameters on GPU). The §2 switch gains < 1% and costs 2 GB/GPU more, **not recommended to turn on**; the patch is kept, off by default.
3. **gen's 321 s is mostly waiting on tools, not generating.** The slowest trajectory spent 281 s in tools and only 26 s generating; on average 135 s tools and 20 s generation per trajectory.
   The rollout is stalled by the tool service (the tool actors were scaled down to ×0.5 to fit into 4 GPUs, roborefer has only 3 replicas).
   Dropping degenerate groups can't help this stage (the reward is only known after generation finishes). The next thing to check is which tool is queueing.

## 4. Per-tool timing (`TOOL_TIMING_DIR=<directory>`)

§3 found that gen's 321 s is mostly waiting on tools, but verl only records the total tool time per turn. With `TOOL_TIMING_DIR` set,
every tool call writes a line to `$TOOL_TIMING_DIR/tool_calls_<pid>.jsonl`: tool name, submit time, total duration, and three split-out stages —

- `exec_wait`: queueing in the agent loop process's default thread pool (Toolshed calls synchronously via `run_in_executor(None, ...)`, the thread count is capped)
- `remote`: the Toolshed call itself (router + tool actor queueing + compute)
- the rest is post-processing after returning to the event loop (`ray.put` of images/variables etc.)

Plus, at submit time, the in-flight call count for that tool and for all tools, the thread pool size, and whether it errored. Without this variable the behavior is exactly as before.
`remote` and `exec_wait` need the Toolshed-side patch (`git -C <SpaceTools-Toolshed> apply toolshed_tool_timing_vs_712e557.diff`); without it there is only the total duration.

```bash
export TOOL_TIMING_DIR=$OUTPUT_DIR/tool_timing        # before run_rl_gflowrl.sh, passed to workers with ray start
bash examples/toolshed/run_rl_gflowrl.sh ...
python3 tool_timing_summary.py $OUTPUT_DIR/tool_timing
```

How to read it: `remote` p50 much larger than min → tool actors are queueing, add replicas / adjust GPU allocation; p50 ≈ min → compute is slow,
needs batching or a different GPU; `exec_wait` large and in-flight count often above the thread count → the bottleneck is the agent loop's thread pool, not the tools.

## 5. GPU-session acceptance checks

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash preflight_nccl.sh     # the GPUs used for training
```
Only on P2P_OK leave `NCCL_P2P_DISABLE` unset; on P2P_BROKEN consider switching machines.

```bash
python3 check_tool_packing.py examples/toolshed/run_rl.sh 4 --card-gib 46   # TOOL_GPUS=4, 46 GiB per GPU
```
Run after changing the tool actor count or `num_gpus`, before starting training. `run_rl.sh` itself only asserts that the shares sum to ≤ TOOL_GPUS, and only triggers once training starts;
this checks two more things: the shares on each GPU add up to ≤ 1.0 (an actor that doesn't fit on any GPU waits forever), and share × per-GPU memory ≥ measured peak
(currently only vlm has a measured value, 32.11 GiB; supply the rest with `--peak-gib tool=GiB`). The tool table is not copied separately into the script; instead the run script's
`TOOL_CONFIGS` section (including auto-scaling) is cut out and executed.
On PACK_FAIL don't start training. PACK_ORDER_DEPENDENT is not blocked: the paper's 2-node config (7.8 / 8) and P7's config (3.7 / 4) both give this result,
meaning there is no headroom and a different creation order would leave some actor unable to fit; if a tool never responds after training starts, check here first.

## 6. Don't save images in the rollout dump (`+trainer.dump_images=false`)

P1's monitoring needs `trainer.rollout_data_dir` on, and upstream `_dump_generations`, besides writing JSONL, also saves each sample's images as PNG
(one `images_<step>/` per step, as many images as the 320 samples have). `p1_monitor.py` only reads the JSONL; these images are unused.
Added a config item: `trainer.dump_images`, default True (behavior exactly as before); when set to false image saving is skipped and the JSONL is still written.
It is not in the upstream config file, so the command line needs `+`:

```bash
GF_FILTER_DEGEN=true bash examples/toolshed/run_rl_gflowrl.sh \
    trainer.rollout_data_dir=$OUT/rollouts +trainer.dump_images=false
```

The validation dump during training (`validation_data_dir`) goes through the same function, so the same switch applies to it too. The eval script doesn't pass this switch; behavior unchanged.

## Self-check results

- `infra_drop_check.py`: in the four cases of 0/5/11/16 degenerate groups, the gradients of the kept rows match full P1-a exactly; the counterexample without padding to micro indeed doesn't match. PASS
- `../p1_prep/p1a_check.py` pointed at `patched/ray_trainer.py`: PASS
- `04_gflowrl_implementation/checks/run_checks.sh` pointed at the full patched tree: config / guard / gpusplit / onpolicy / fixedpoint / degenerate all PASS
- `infra_all_vs_c6fef78a.diff` after `git apply` on a clean c6fef78a is byte-for-byte identical to `patched/`

- `check_tool_packing.py --selftest`: sum over the limit, sum within the limit but not fitting on a single GPU, the P7 config (order-dependent), vlm share 0.6 / 0.7 on a 46 GiB GPU — all verdicts as expected; on `c6fef78a`'s `run_rl.sh` with TOOL_GPUS=4 it reproduces 3.70 and P7's actor counts. PASS
- `dump_images_check.py`: unset / set to true, image saving is the same as before; set to false, no `images_<step>/` is created and the JSONL is identical line by line; the same script pointed at `../p1_prep/patched/ray_trainer.py`, which has no switch, fails. PASS
- `tool_timing_check.py`: with the variable unset, the return value is exactly the same as the original function; with it set, one record per call, the `remote` / `exec_wait` split is correct, `exec_wait` grows when there aren't enough threads, and Toolshed errors and unknown tools are both recorded as failures. PASS

Not done: checkpoint slimming. Saved every 5 steps, 39 s each, amortized to about 8 s per step (<1%), and dropping the optimizer state would break automatic resume.
