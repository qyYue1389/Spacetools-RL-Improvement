# 01 · Official checkpoint eval (reproduction phases P2–P6) / 官方 checkpoint 评测

Evaluation of the released SpaceTools checkpoint (`siyich/spacetools-ckpt`, trained with GRPO) on the nine benchmarks of the paper's Table 2, then accuracy and per-sample error attribution. The P4 run here is the GRPO side of every later GRPO-vs-GFlowRL comparison.
官方 checkpoint 在九个 benchmark 上的全量评测、accuracy 与逐样本错题归因。这里的 P4 就是后面所有 GRPO vs GFlowRL 对比中的 GRPO 一侧。

| Path | What |
|---|---|
| `reports/changes_to_upstream.md` (+ `.zh.md`) | Every change made to every upstream repo, and why / 对上游的每处改动与原因 |
| `reports/provenance.txt` | Weight commits, env matrix, deviations |
| `reports/p2_smoke_test_results.md` | P2 smoke test on 4× A6000 |
| `reports/p3_trajectory_capture.md` (+ `.zh.md`) | P3: how trajectories are captured from verl dumps |
| `reports/p4_full_eval_results.md` (+ `.zh.md`) | P4: full eval, 9 benchmarks / 2121 samples on 4× A100-40GB |
| `reports/p5_accuracy_report.md` | P5: accuracy vs paper, gap attribution |
| `reports/p6_error_attribution_report.md`, `p6_error_attribution_full.md` | P6: error attribution and headroom (summary / full) |
| `reports/p6_gpu_experiments.md` | P6 GPU experiments: fp32, KV pool size, pass@k, tool swaps |
| `tools/p4_run.sh`, `p4_check.sh`, `make_subset.py` | Run / gate the eval, subset a benchmark |
| `tools/p5/`, `tools/p6/` | P5 accuracy and P6 attribution scripts (CPU only) |
| `p4/dumps/` · `p4/parsed/` · `p4/logs/` | Raw trajectories (gz), parsed records, eval logs |
| `p6/` | Ablation dumps (fp32, gmu025, passk, passk2), probes, tool swaps, manual verdicts |

Run the scripts from this folder after `bash ../tools/unpack_dumps.sh`, e.g. `python3 tools/p6/p6_relations.py`. `tools/p5/metrics.py` and `perm.py` also read the reward code of a SpaceTools-RL checkout (`SPACETOOLS_RL=/path`).
