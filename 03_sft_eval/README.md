# 03 · SFT eval (gate before RL) / SFT eval 闸门

The SFT checkpoint evaluated on RoboSpatial (twice) and the three RefSpatial splits on 4× A6000, to decide whether it is a sound start point for RL.
在 RoboSpatial(两次)与 RefSpatial 三项上评 SFT checkpoint,决定能否作为 RL 起点。

| Path | What |
|---|---|
| `sft_eval_report.md` | Report: results, failure modes, gate decision / 报告 |
| `sft_eval_results.md` | Raw results and deviations |
| `ARTIFACTS.md` | Field-by-field description of the rollouts and logs / 产物说明 |
| `rollouts/*.jsonl.gz` | Full trajectories; the two RoboSpatial runs are aligned by `index` (non-determinism study) |
| `config/` | `run_eval.sh` as run, tool config, pinned weight revisions (`WEIGHTS_PINS.txt`) |
| `scripts/` + `analysis/` | Audit / health / non-determinism / Vacant scripts and their outputs |
| `verify/` · `logs/` · `gpu/` | Env acceptance logs, eval logs (gz), GPU trace |
