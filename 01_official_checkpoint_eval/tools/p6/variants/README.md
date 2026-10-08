# run_eval.sh variants and resident scripts used in the P6 GPU session

Without these, neither deviation `[23]` nor experiment F can be reproduced. Both variants are **verbatim copies
of `run_eval.sh` + the specified minimal change**, and are run from the same directory, `examples/toolshed/`,
because `run_eval.sh:64` uses `BASH_SOURCE` to locate `SCRIPT_DIR`; a different directory breaks it.

| File | Difference from `run_eval.sh` | Purpose |
|---|---|---|
| `run_eval_gmu025.sh` | **One line**: `rollout.gpu_memory_utilization` `0.5` → `0.25` | run5 / run6. Reproduces the 20 GB KV pool of 40 GB on 80 GB, see deviation `[23]` |
| `run_eval_fp32_gmu025.sh` | **Two lines**: remove `actor.fsdp_config.model_dtype=bf16`; `gpu_memory_utilization` `0.5` → `0.25` | run7, experiment F. With the flag removed, `model_dtype` falls back to the default `fp32` in `verl/trainer/config/engine/fsdp.yaml:33` |

Neither `0.25` is optional: the criterion of experiment F is that fp32 and bf16 fall in the same band,
so the two arms may differ in only one variable, dtype, and the bf16 reference arm (run5/run6) is at 0.25.
Running fp32 at 0.5 would change two variables at once and void the conclusion.

    # Check the differences (should be 1 line and 2 lines respectively)
    diff examples/toolshed/run_eval.sh examples/toolshed/run_eval_gmu025.sh
    diff examples/toolshed/run_eval.sh examples/toolshed/run_eval_fp32_gmu025.sh

## Run scripts

`run_gmu025.sh` / `run_fp32.sh` copy the logic and both guards of `tools/p4_run.sh`
(refuse to overwrite an existing dump, refuse labels that are not `run<N>`), and only swap the eval script they call.

> **The label `run_fp32` given in doc §1.5 does not run** — the guard in `p4_run.sh` is
> `^run[0-9]+$` and rejects it outright. This time `run7` was used.

## Toolshed resident process

`toolshed_v1_holder.py` is a **verbatim copy** of lines 193–211 of `run_eval.sh` (the v1 branch),
used for experiments A / B — those two experiments do not need a policy, only the tools up and running.

The return value of `start_toolkit(detached=False)` must be held by a long-lived process
(the original file does this with `while True: time.sleep(60)`); otherwise Ray reclaims the router,
and after a few calls you get `Could not find ToolRouterActor`, which looks like an intermittent failure.

    source /workspace/env.sh && conda activate spacetools-rl
    ray stop --force; ray start --head --num-gpus=4 --port=6379
    export RAY_ADDRESS=127.0.0.1:6379
    nohup python toolshed_v1_holder.py "$ROBOREFER_MODEL" "$DEPTH_CHECKPOINT" &
    # Liveness check: ray status should show 3.0/4.0 GPU — but that is not enough, see below

> **`ray status` showing 3.0/4.0 GPU does not mean the tools can be called yet.** It only means the actors have
> **claimed** the GPUs, while the models are still at `Loading checkpoint shards`. Calling too early on that basis,
> all 397 probes got a `ToolRouterActor.call_tool()` exception — **and the failure looks exactly like
> a real result**: a complete summary table, every number 0.00%.
> `run_eval.sh` says `sleep 120`; do that.
>
> Three signals that give it away: the `tool failures` count equals the number of samples, each benchmark takes only 1–4 seconds,
> and it differs from the baseline by 210 samples (far beyond the threshold of "a difference of ten already means the path is wrong").
> **Before the real run, do a smoke test with `--limit 5` first**; five seconds is enough to confirm the tools are really alive.

## Experiment C (pass@k)

| File | Difference from `run_eval.sh` |
|---|---|
| `run_eval_passk.sh` | **Five lines**: `gpu_memory_utilization` 0.5 → 0.25, plus four `val_kwargs` (`do_sample=True` / `n=5` / `temperature=1.0` / `top_p=1.0`) |

verl's validation is greedy by default (`val_kwargs`: `temperature 0 / n 1 / do_sample False`);
without an explicit override it will not sample. How to verify it took effect: the resolved config should contain `do_sample: True`,
the dump row count should be number of samples × 5, and the 5 outputs per sample should not all be identical.
