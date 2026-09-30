# 06 · GFlowRL eval / GFlowRL 评测

C′ step 85 on all nine benchmarks (2121 samples), with zero OOM / truncation / max-turn / missing-`<answer>`, and a per-sample comparison against the official GRPO checkpoint (`01_official_checkpoint_eval/p4`).
C′ step 85 在九个 benchmark 上的评测,以及与官方 GRPO checkpoint 的逐样本对比。

| Path | What |
|---|---|
| `gflowrl_eval_report.md` | Eval report: protocol, parameter changes, the silent-OOM contamination and how it was caught, health gates, results / 评测报告 |
| `comparison_p4_p5_p6_vs_p7.md` | Per-sample comparison with the P4/P5/P6 analyses / 逐样本对比分析 |
| `scripts/EVAL_FROM_SCRATCH.sh` | Bare GPU machine → nine scores, idempotent |
| `scripts/run_eval.sh` (+ `.asrun`, `.diff`), `run_eval_fix2.sh`, `run_rs2.sh` | Eval launchers as run (fix2 = rerun of 4 benchmarks with `vlm num_gpus=1.0`; rs2 = second RoboSpatial run) |
| `dumps/runA-9bench/` | Valid benchmarks of the first run: robospatial, reflocation, refplacement, refunseen, cvb2drelation |
| `dumps/runB-rerun4/` | blinkdepth, bopgrasp, boppose, cvb3ddepth (replace runA's, which had silent tool OOMs) |
| `dumps/runC-robospatial2/` | Second RoboSpatial run |
| `parsed/`, `parsed_runC/` | Structured trajectory records (same schema as `01_official_checkpoint_eval/p4/parsed`) |
| `gates/` | Health gates per benchmark (all must be zero before a score is read) |
| `analysis/compare_p4_p7.py` → `compare_p4_p7.txt` | Per-sample pairing and summary table |
| `analysis/significance_p4_p7.py` | McNemar exact test + sign tests |
| `analysis/p6_criteria_p4_vs_p7.py` | P6 criteria, question-type split, Vacant pass-through vs edit |
| `analysis/robospatial_divergence.py` | Direction and type of the diverging RoboSpatial samples |
| `logs/`, `gpu/`, `config/` | Eval logs (gz), GPU trace, env and weight pins |

The analysis scripts need only Python 3 and read `parsed/`; they run as-is from any directory.
