# 00 · Environment / 环境

SFT needs one conda env; eval and RL need five (driver `spacetools-rl` + tool envs `vlm`, `roborefer`, `bbox`, `graspgen`) that must agree on Python / Ray versions.
SFT 只需一个 conda 环境;eval 与 RL 需要五个(driver + 四个工具环境),且 Python / Ray 版本必须一致。

| Path | What |
|---|---|
| `sft_env/setup_sft_env.sh` | Build the SFT env from scratch (flash-attn compiled for sm_80/86/89) / 从零搭 SFT 环境 |
| `sft_env/env_package/` | Restore + verify the packaged SFT env (`RESTORE.sh`, `VERIFY.sh`) / 还原打包好的 SFT 环境 |
| `sft_env/sft_env_build_report_2xA6000.md` | Build log and 30-step probe on 2× A6000 / 搭建记录与 30 步探测 |
| `eval_rl_env/BUILD_GUIDE.md` | **Main guide**: architecture, per-env versions, build steps, the 20 upstream issues, acceptance criteria / 主文档 |
| `eval_rl_env/build_eval_envs.sh` | Build the five envs / 搭五个环境 |
| `eval_rl_env/graspgen_fixes/repair_graspgen_step1..6.sh` | GraspGen fixes, apply in order / GraspGen 的六轮修复,按顺序跑 |
| `eval_rl_env/fix_vlm_transformers.sh`, `fix_weights_roborefer.sh` | Molmo transformers pin, RoboRefer weights layout |
| `eval_rl_env/verify_envs.sh`, `smoke_tools.sh`, `VERIFY.sh` | Import checks, per-tool smoke test, 4-item acceptance / 验收 |
| `eval_rl_env/POSTRESTORE.sh` | Three fixes to apply after restoring the HF env package / 从 HF 还原后要补的三件事 |
| `eval_rl_env/env_package_revision_20260912.md` | What was missing from the first env package and how it was fixed |
| `upstream_patches/` | `git am`-able patches for SpaceTools-Toolshed, GraspGen, RoboRefer (SpaceTools-RL patches are in `04_gflowrl_implementation/`) |
| `official_ckpt_fix/` | `fix_checkpoint.py` repairs the released `siyich/spacetools-ckpt` config so sglang can load it; `original_config/` keeps the originals |
| `env_freeze/` | `pip freeze` of all five envs |
| `env.sh`, `tool-constraints.txt`, `verify_env_imports.sh`, `rebuild_pointnet2_sm80.sh`, `tool_chain_smoke_test.py` | Cache/path setup, pip constraints, import check, pointnet2 rebuild for A100, all-tools chain test without a language model |

Packaged env on Hugging Face / HF 上的环境包: [`qzpm55555/spacetools-eval-env`](https://huggingface.co/qzpm55555/spacetools-eval-env) — see `BUILD_GUIDE.md` §4.1.
