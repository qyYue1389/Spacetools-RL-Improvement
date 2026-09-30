# Path map / 路径对照

The reports were written before this repo existed and quote paths from the original working folders
(`spacetools-repro/`, `training/`, `eval/`, `GFlowRL_improve/`, project docs).
This table maps each of them to its location here. Paths not listed were intentionally left out
(session transcripts, hand-off notes for past GPU sessions, superseded drafts, model weights and benchmark data that live on Hugging Face).

报告写于本 repo 建立之前,引用的是原工作目录的路径。下表给出每个旧路径在本 repo 中的位置;未列出的是有意不收录的内容(会话记录、历次开机交接、被取代的草稿,以及放在 Hugging Face 上的权重和 benchmark 数据)。

| Original path / 原路径 | In this repo / 本 repo |
|---|---|
| `eval/GFlowRL/P7_完整报告_v1.md` | `docs/full_report_v1.md` |
| `GFlowRL_improve/P7 GFlowRL(C′)优化 · Design Doc.md` | `docs/design_doc_gflowrl_optimization.md` |
| `两篇论文笔记.md` | `docs/paper_notes.md` |
| `training/SFT/SFT训练学习笔记.md` | `docs/sft_training_notes.md` |
| `training/为什么不能按 benchmark 分开训练.md` | `docs/why_not_train_per_benchmark.md` |
| `training/SFT/runpod-handoff/setup.sh` | `00_environment/sft_env/setup_sft_env.sh` |
| `training/SFT/runpod-handoff/check_arch.sh` | `00_environment/sft_env/check_gpu_arch.sh` |
| `training/SFT/env/bundle/README.md` | `00_environment/sft_env/env_package/README.md` |
| `training/SFT/env/bundle/RESTORE.sh` | `00_environment/sft_env/env_package/RESTORE.sh` |
| `training/SFT/env/bundle/VERIFY.sh` | `00_environment/sft_env/env_package/VERIFY.sh` |
| `training/SFT/env/bundle/SHA256SUMS` | `00_environment/sft_env/env_package/SHA256SUMS` |
| `training/SFT/env/2x_A6000_REPORT.md` | `00_environment/sft_env/sft_env_build_report_2xA6000.md` |
| `eval/eval-env/README.md` | `00_environment/eval_rl_env/BUILD_GUIDE.md` |
| `eval/eval-env/build_eval_envs.sh` | `00_environment/eval_rl_env/build_eval_envs.sh` |
| `eval/eval-env/verify_envs.sh` | `00_environment/eval_rl_env/verify_envs.sh` |
| `eval/eval-env/smoke_tools.sh` | `00_environment/eval_rl_env/smoke_tools.sh` |
| `eval/eval-env/fix_vlm_transformers.sh` | `00_environment/eval_rl_env/fix_vlm_transformers.sh` |
| `eval/eval-env/fix_weights_roborefer.sh` | `00_environment/eval_rl_env/fix_weights_roborefer.sh` |
| `eval/eval-env/repair_graspgen.sh` | `00_environment/eval_rl_env/graspgen_fixes/repair_graspgen_step1.sh` |
| `eval/eval-env/repair_graspgen2.sh` | `00_environment/eval_rl_env/graspgen_fixes/repair_graspgen_step2.sh` |
| `eval/eval-env/repair_graspgen3.sh` | `00_environment/eval_rl_env/graspgen_fixes/repair_graspgen_step3.sh` |
| `eval/eval-env/repair_graspgen4.sh` | `00_environment/eval_rl_env/graspgen_fixes/repair_graspgen_step4.sh` |
| `eval/eval-env/repair_graspgen5.sh` | `00_environment/eval_rl_env/graspgen_fixes/repair_graspgen_step5.sh` |
| `eval/eval-env/repair_graspgen6.sh` | `00_environment/eval_rl_env/graspgen_fixes/repair_graspgen_step6.sh` |
| `eval/SFT/sft-eval-artifacts/config/POSTRESTORE.sh` | `00_environment/eval_rl_env/POSTRESTORE.sh` |
| `eval/SFT/sft-eval-artifacts/config/VERIFY.sh.new` | `00_environment/eval_rl_env/VERIFY.sh` |
| `eval/SFT/环境包修订 20260912.md` | `00_environment/eval_rl_env/env_package_revision_20260912.md` |
| `spacetools-repro/records/env-freeze` | `00_environment/env_freeze` |
| `spacetools-repro/patches/toolshed` | `00_environment/upstream_patches/SpaceTools-Toolshed` |
| `spacetools-repro/patches/graspgen` | `00_environment/upstream_patches/GraspGen` |
| `spacetools-repro/patches/roborefer` | `00_environment/upstream_patches/RoboRefer` |
| `spacetools-repro/tools/fix_checkpoint.py` | `00_environment/official_ckpt_fix/fix_checkpoint.py` |
| `spacetools-repro/ckpt-orig` | `00_environment/official_ckpt_fix/original_config` |
| `spacetools-repro/tools/env.sh` | `00_environment/env.sh` |
| `spacetools-repro/tools/tool-constraints.txt` | `00_environment/tool-constraints.txt` |
| `spacetools-repro/tools/verify_p1.sh` | `00_environment/verify_env_imports.sh` |
| `spacetools-repro/tools/rebuild_pointnet2_sm80.sh` | `00_environment/rebuild_pointnet2_sm80.sh` |
| `spacetools-repro/tools/p2_tool_chain.py` | `00_environment/tool_chain_smoke_test.py` |
| `spacetools-repro/records/CHANGES.md` | `01_official_checkpoint_eval/reports/changes_to_upstream.md` |
| `spacetools-repro/records/CHANGES.zh.md` | `01_official_checkpoint_eval/reports/changes_to_upstream.zh.md` |
| `spacetools-repro/records/PROVENANCE.txt` | `01_official_checkpoint_eval/reports/provenance.txt` |
| `spacetools-repro/records/P2_RESULTS.md` | `01_official_checkpoint_eval/reports/p2_smoke_test_results.md` |
| `spacetools-repro/records/P3_NOTES.md` | `01_official_checkpoint_eval/reports/p3_trajectory_capture.md` |
| `spacetools-repro/records/P3_NOTES.zh.md` | `01_official_checkpoint_eval/reports/p3_trajectory_capture.zh.md` |
| `spacetools-repro/records/P4_RESULTS.md` | `01_official_checkpoint_eval/reports/p4_full_eval_results.md` |
| `spacetools-repro/records/P4_RESULTS.zh.md` | `01_official_checkpoint_eval/reports/p4_full_eval_results.zh.md` |
| `spacetools-repro/records/P5_REPORT.md` | `01_official_checkpoint_eval/reports/p5_accuracy_report.md` |
| `spacetools-repro/records/P6_REPORT.md` | `01_official_checkpoint_eval/reports/p6_error_attribution_report.md` |
| `eval/GFlowRL/P6_错题归因_完整版.md` | `01_official_checkpoint_eval/reports/p6_error_attribution_full.md` |
| `spacetools-repro/records/P6_GPU_RESULTS.md` | `01_official_checkpoint_eval/reports/p6_gpu_experiments.md` |
| `spacetools-repro/tools/p4_run.sh` | `01_official_checkpoint_eval/tools/p4_run.sh` |
| `spacetools-repro/tools/p4_check.sh` | `01_official_checkpoint_eval/tools/p4_check.sh` |
| `spacetools-repro/tools/make_subset.py` | `01_official_checkpoint_eval/tools/make_subset.py` |
| `spacetools-repro/tools/p5` | `01_official_checkpoint_eval/tools/p5` |
| `spacetools-repro/tools/p6` | `01_official_checkpoint_eval/tools/p6` |
| `spacetools-repro/p4/parsed` | `01_official_checkpoint_eval/p4/parsed` |
| `spacetools-repro/p4/dumps` | `01_official_checkpoint_eval/p4/dumps  (*.jsonl / *.log gzipped)` |
| `spacetools-repro/p4/logs` | `01_official_checkpoint_eval/p4/logs` |
| `spacetools-repro/p6` | `01_official_checkpoint_eval/p6  (*.jsonl / *.log gzipped)` |
| `training/SFT/env/training_report/REPORT.md` | `02_sft_training/sft_training_report_4xA6000.md` |
| `training/SFT/env/training_report/val/FINALIZE.sh` | `02_sft_training/FINALIZE.sh` |
| `training/SFT/env/training_report/val/smoke_check.py` | `02_sft_training/smoke_check.py` |
| `training/SFT/sft-schema-check` | `02_sft_training/tool_schema_check` |
| `eval/SFT/SFT eval 报告.md` | `03_sft_eval/sft_eval_report.md` |
| `eval/SFT/SFT eval 结果.md` | `03_sft_eval/sft_eval_results.md` |
| `eval/SFT/sft-eval-artifacts/README.md` | `03_sft_eval/ARTIFACTS.md` |
| `eval/SFT/sft-eval-artifacts/analysis` | `03_sft_eval/analysis` |
| `eval/SFT/sft-eval-artifacts/scripts` | `03_sft_eval/scripts` |
| `eval/SFT/sft-eval-artifacts/verify` | `03_sft_eval/verify` |
| `eval/SFT/sft-eval-artifacts/gpu` | `03_sft_eval/gpu` |
| `eval/SFT/sft-eval-artifacts/rollouts` | `03_sft_eval/rollouts  (*.jsonl / *.log gzipped)` |
| `eval/SFT/sft-eval-artifacts/logs` | `03_sft_eval/logs  (*.jsonl / *.log gzipped)` |
| `eval/SFT/sft-eval-artifacts/config/run_eval.sh.asrun` | `03_sft_eval/config/run_eval.sh.asrun` |
| `eval/SFT/sft-eval-artifacts/config/toolshed_config.yaml` | `03_sft_eval/config/toolshed_config.yaml` |
| `eval/SFT/sft-eval-artifacts/config/WEIGHTS_PINS.txt` | `03_sft_eval/config/WEIGHTS_PINS.txt` |
| `eval/SFT/sft-eval-artifacts/config/pkg_MANIFEST.txt` | `03_sft_eval/config/pkg_MANIFEST.txt` |
| `eval/SFT/sft-eval-artifacts/config/SHA256SUMS.remote` | `03_sft_eval/config/SHA256SUMS.remote` |
| `training/GFlowRL/README.md` | `04_gflowrl_implementation/IMPLEMENTATION_NOTES.md` |
| `spacetools-repro/records/P7_DECISION.md` | `04_gflowrl_implementation/design_notes/01_route_decision.md` |
| `spacetools-repro/records/P7_PRIOR_ART.md` | `04_gflowrl_implementation/design_notes/02_prior_art_survey.md` |
| `spacetools-repro/records/P7_CRITERION_IB.md` | `04_gflowrl_implementation/design_notes/03_criterion_ib_batch_constant.md` |
| `spacetools-repro/records/P7_STEP23_RESULTS.md` | `04_gflowrl_implementation/design_notes/04_zt_offline_and_fixed_point.md` |
| `spacetools-repro/records/P7_STEP4_RESULTS.md` | `04_gflowrl_implementation/design_notes/05_response_mask_and_length.md` |
| `spacetools-repro/records/P7_GPU_RESULTS.md` | `04_gflowrl_implementation/design_notes/06_gpu_resampling_and_logprob.md` |
| `spacetools-repro/records/P7_ROUTE_C_GATE.md` | `04_gflowrl_implementation/design_notes/07_cprime_gate_length_normalization.md` |
| `spacetools-repro/records/P7_ROUTE_C_PLAN.md` | `04_gflowrl_implementation/design_notes/08_route_c_plan.md` |
| `spacetools-repro/records/P7_STEP_C12_RESULTS.md` | `04_gflowrl_implementation/design_notes/09_implementation_and_fixed_point_check.md` |
| `spacetools-repro/tools/p7` | `04_gflowrl_implementation/checks` |
| `training/GFlowRL/run_checks.sh` | `04_gflowrl_implementation/checks/run_checks.sh` |
| `spacetools-repro/p7/accept` | `04_gflowrl_implementation/migration_acceptance  (*.jsonl / *.log gzipped)` |
| `training/GFlowRL/A40/P7_GFLOWRL_训练报告.md` | `05_gflowrl_training/gflowrl_training_report_8xA40.md` |
| `training/GFlowRL/A40/config` | `05_gflowrl_training/config` |
| `training/GFlowRL/A40/metrics/p7_metrics.csv` | `05_gflowrl_training/training_metrics_85steps.csv` |
| `training/GFlowRL/A40/scripts` | `05_gflowrl_training/scripts` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/P7_GFLOWRL_EVAL报告.md` | `06_gflowrl_eval/gflowrl_eval_report.md` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/config` | `06_gflowrl_eval/config` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/gates` | `06_gflowrl_eval/gates` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/gpu` | `06_gflowrl_eval/gpu` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/parsed` | `06_gflowrl_eval/parsed` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/parsed_runC` | `06_gflowrl_eval/parsed_runC` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/logs/eval.log.gz` | `06_gflowrl_eval/logs/eval.log.gz` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/logs/eval_fix2.log.gz` | `06_gflowrl_eval/logs/eval_fix2.log.gz` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/logs/eval_rs2.log.gz` | `06_gflowrl_eval/logs/eval_rs2.log.gz` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/logs/runB-rerun4` | `06_gflowrl_eval/logs/runB-rerun4  (*.jsonl / *.log gzipped)` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/logs/runC-robospatial2` | `06_gflowrl_eval/logs/runC-robospatial2  (*.jsonl / *.log gzipped)` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/dumps/runB-rerun4` | `06_gflowrl_eval/dumps/runB-rerun4  (*.jsonl / *.log gzipped)` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/dumps/runC-robospatial2` | `06_gflowrl_eval/dumps/runC-robospatial2  (*.jsonl / *.log gzipped)` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/dumps/runA-9bench/robospatial` | `06_gflowrl_eval/dumps/runA-9bench/robospatial  (*.jsonl / *.log gzipped)` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/logs/runA-9bench/robospatial` | `06_gflowrl_eval/logs/runA-9bench/robospatial  (*.jsonl / *.log gzipped)` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/dumps/runA-9bench/reflocation` | `06_gflowrl_eval/dumps/runA-9bench/reflocation  (*.jsonl / *.log gzipped)` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/logs/runA-9bench/reflocation` | `06_gflowrl_eval/logs/runA-9bench/reflocation  (*.jsonl / *.log gzipped)` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/dumps/runA-9bench/refplacement` | `06_gflowrl_eval/dumps/runA-9bench/refplacement  (*.jsonl / *.log gzipped)` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/logs/runA-9bench/refplacement` | `06_gflowrl_eval/logs/runA-9bench/refplacement  (*.jsonl / *.log gzipped)` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/dumps/runA-9bench/refunseen` | `06_gflowrl_eval/dumps/runA-9bench/refunseen  (*.jsonl / *.log gzipped)` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/logs/runA-9bench/refunseen` | `06_gflowrl_eval/logs/runA-9bench/refunseen  (*.jsonl / *.log gzipped)` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/dumps/runA-9bench/cvb2drelation` | `06_gflowrl_eval/dumps/runA-9bench/cvb2drelation  (*.jsonl / *.log gzipped)` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/logs/runA-9bench/cvb2drelation` | `06_gflowrl_eval/logs/runA-9bench/cvb2drelation  (*.jsonl / *.log gzipped)` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/scripts/EVAL_FROM_SCRATCH.sh` | `06_gflowrl_eval/scripts/EVAL_FROM_SCRATCH.sh` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/scripts/run_eval.sh` | `06_gflowrl_eval/scripts/run_eval.sh` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/scripts/run_eval.sh.asrun` | `06_gflowrl_eval/scripts/run_eval.sh.asrun` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/scripts/run_eval.sh.diff` | `06_gflowrl_eval/scripts/run_eval.sh.diff` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/scripts/run_eval_fix2.sh` | `06_gflowrl_eval/scripts/run_eval_fix2.sh` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/scripts/run_rs2.sh` | `06_gflowrl_eval/scripts/run_rs2.sh` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/analysis/compare_p4_p7.py` | `06_gflowrl_eval/analysis/compare_p4_p7.py` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/analysis/compare_p4_p7.txt` | `06_gflowrl_eval/analysis/compare_p4_p7.txt` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/analysis/p6_criteria_p4_vs_p7.py` | `06_gflowrl_eval/analysis/p6_criteria_p4_vs_p7.py` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/analysis/robospatial_divergence.py` | `06_gflowrl_eval/analysis/robospatial_divergence.py` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/analysis/significance_p4_p7.py` | `06_gflowrl_eval/analysis/significance_p4_p7.py` |
| `eval/GFlowRL/P4_P5_P6_vs_P7_对比分析.md` | `06_gflowrl_eval/comparison_p4_p5_p6_vs_p7.md` |
| `GFlowRL_improve/P0/P7 P0 结果.md` | `07_gflowrl_improvement/p0_same_machine_reeval/p0_results.md` |
| `GFlowRL_improve/P0/p0_bundle/scripts` | `07_gflowrl_improvement/p0_same_machine_reeval/scripts` |
| `GFlowRL_improve/P0/p0_bundle/dumps` | `07_gflowrl_improvement/p0_same_machine_reeval/dumps  (*.jsonl / *.log gzipped)` |
| `GFlowRL_improve/P0/p0_bundle/logs` | `07_gflowrl_improvement/p0_same_machine_reeval/logs  (*.jsonl / *.log gzipped)` |
| `GFlowRL_improve/P1_prep` | `07_gflowrl_improvement/p1_prep` |
| `spacetools-repro/tools/parse_dump.py` | `tools/parse_dump.py` |
| `spacetools-repro/tools/gputrace.sh` | `tools/gputrace.sh` |
| `SpaceTools-RL @ c6fef78a (7 changed files)` | `04_gflowrl_implementation/spacetools_rl_modified/` |
| `SpaceTools-RL 54270e82..c6fef78a (19 commits)` | `04_gflowrl_implementation/patches_spacetools_rl/` |
| `SpaceTools-SFT/scripts/spacetools/run_sft.sh (as run)` | `02_sft_training/run_sft.sh` |
| `training/GFlowRL/diff/*.patch, training/GFlowRL/code/` | `superseded by 04_gflowrl_implementation/ (the trained commit c6fef78a)` |
| `training/GFlowRL/A40/scripts/janitor2.py` | `05_gflowrl_training/scripts/ckpt_janitor.py` |
| `training/GFlowRL/A40/scripts/p7tab.py` | `05_gflowrl_training/scripts/extract_metrics_table.py` |
| `training/GFlowRL/A40/scripts/save_ckpts.sh` | `05_gflowrl_training/scripts/merge_and_upload_ckpts.sh` |
| `training/GFlowRL/A40/scripts/mkfull.py` | `05_gflowrl_training/scripts/make_full_train_script.py` |
| `training/GFlowRL/A40/scripts/{exitprobe,exitprobe2,ncclprobe,nprobe}` | `05_gflowrl_training/scripts/debug_probes/` |
| `GFlowRL_improve/P0/p0_bundle/scripts/parse_dump.py` | `tools/parse_dump.py (identical)` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/scripts/parse_dump.py` | `tools/parse_dump.py (identical)` |
| `eval/GFlowRL/P7_GFLOWRL_EVAL/dumps/runA-9bench/{blinkdepth,cvb3ddepth,boppose,bopgrasp}` | `not included: invalid (silent tool OOM); use runB-rerun4` |
