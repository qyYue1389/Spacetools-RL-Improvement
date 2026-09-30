# 07 · GFlowRL optimization / GFlowRL 优化

Work driven by [`docs/design_doc_gflowrl_optimization.md`](../docs/design_doc_gflowrl_optimization.md). Phase names here (P0–P4) are the design doc's, not the reproduction phases.
按 design doc 推进的优化工作。这里的 P0–P4 是 design doc 的编号,和复现阶段的 P0–P7 不是一套。

## p0_same_machine_reeval/ — done / 已完成

SFT start, official GRPO checkpoint and C′ step 85, evaluated on the same machine (4× RTX 6000 Ada) with the same 24.5 GB KV pool: RoboSpatial and BLINK depth 3 runs each, RefSpatial once.

| Path | What |
|---|---|
| `p0_results.md` | Results and attribution of the Vacant gap / 结果与改点归因 |
| `scripts/p0_go.sh`, `p0_eval.sh`, `run_eval.as-run.sh`, `p4-official.config.*.json` | How the 12 runs were launched (the official ckpt config had to be patched) |
| `scripts/paired.py`, `flips.py`, `tfcheck.py`, `gate.sh`, `p2.sh` | Paired tests, run-to-run flips, health gate |
| `dumps/` | `{sft,p4,cp}_{run1..3,ref1}_<benchmark>.jsonl.gz` (`cp` = C′) |
| `logs/` | Per-run eval logs and the driver log (gz) |

## p1_prep/ — prepared, next GPU run / 已准备,下一次 GPU 实验

| Path | What |
|---|---|
| `p1a_ray_trainer.diff`, `p1a_run_rl_gflowrl.diff` | P1(a): zero g̃ for reward-degenerate groups (switch `GF_FILTER_DEGEN`, default off), with a driver guard that the update is on-policy |
| `patched/` | The two files with the patch applied (apply on top of `04_gflowrl_implementation`) |
| `p1a_check.py` | CPU self-check: filter off = identical to original; filter on = only degenerate rows change; on-policy gradient equals a true loss mask (needs torch) |
| `p1_monitor.py` | Live monitor over rollout / validation dumps: anchor-only query share (primary metric), edit rate ± bootstrap SE, pass-through / edit accuracy, tool-point hit rate; `--upper` = forced pass-through |
| `p2_query_type.py` | Splits Vacant questions by `obj_name` query type; reads the P0 dumps above by default |
| `robospatial_vacant.parquet` | The 122 Vacant questions, used as the in-training validation set |
| `README.md` | Launch commands and findings (Chinese) |
