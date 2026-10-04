# Spacetools-RL-Improvement

**Replacing GRPO with GFlowRL in the tool-augmented RL stage of SpaceTools — full reproduction, implementation, evaluation, and ongoing optimization.**

**在 SpaceTools 的多轮工具调用 RL 阶段用 GFlowRL 替换 GRPO —— 完整复现、实现、评测与持续优化。**

[English](#english) · [中文](#中文)

---

<a id="english"></a>

## 1. Background

[SpaceTools](https://github.com/spacetools/SpaceTools) ([arXiv:2512.04069](https://arxiv.org/abs/2512.04069), CVPR 2026) trains Qwen2.5-VL-3B to solve metric spatial-reasoning tasks by calling seven vision tools (RoboRefer pointing, Molmo, DepthPro depth, SAM2, GraspGen, bounding-box and vision ops) over multiple turns. Its pipeline (DIRL) has four steps: (1) point-tool RL, (2) teacher-trajectory collection, (3) SFT, (4) full-tool RL with **GRPO**.

[GFlowRL](https://arxiv.org/abs/2607.13394) (arXiv:2607.13394) is a **distribution-matching** RL objective: instead of pushing all probability mass onto the single best mode (reward maximization), it trains the policy toward π ∝ π_ref · exp(βr). Tool use naturally has several equally valid tool chains, and our error analysis of the official SpaceTools checkpoint found strong **tool-orchestration mode collapse** (e.g. 277/277 RefSpatial samples use the exact same tool chain). That is the gap this project targets.

## 2. Goal

1. Reproduce the SpaceTools pipeline end to end on rented, smaller GPUs (A6000 / A100 / A40 instead of 2×8 A100-80GB).
2. Train our own SFT checkpoint (Step 3), then replace the **Step 4 GRPO objective with GFlowRL** while keeping data, group size, epochs and start point fixed.
3. Evaluate on the nine benchmarks of the paper's Table 2 and compare **per sample** against the official GRPO checkpoint.
4. Diagnose why GFlowRL does or does not help, and iterate on the algorithm (current work — see the design doc).

## 3. What has been done

| Stage | What | Result | Where |
|---|---|---|---|
| Environment | 5 conda envs (driver + 4 tool envs), 20 upstream bugs fixed, packaged to HF | Restore + verify scripts; sm_80 / 86 / 89 GPUs | [`00_environment/`](00_environment) |
| Official checkpoint eval (repro P2–P6) | Full eval of `siyich/spacetools-ckpt` on 9 benchmarks (2121 samples), accuracy report, per-sample error attribution | Every gap vs the paper attributed; all 322 wrong answers classified | [`01_official_checkpoint_eval/`](01_official_checkpoint_eval) |
| SFT (Step 3) | Qwen2.5-VL-3B, 3000 steps, 4× A6000, global batch 8 | eval_loss 0.168 → 0.045, used as π_ref | [`02_sft_training/`](02_sft_training) |
| SFT eval gate | RoboSpatial + RefSpatial on the SFT ckpt | RoboSpatial 61.00 ± 0.77, RefSpatial 53.07 → go for RL | [`03_sft_eval/`](03_sft_eval) |
| GFlowRL implementation | **C′ variant** (no length normalization, fixed point exact), 2 verl plug-in points; 19 commits on SpaceTools-RL (10 eval/infra fixes + 9 GFlowRL) | Fixed-point self-check 7.4e-13 PASS | [`04_gflowrl_implementation/`](04_gflowrl_implementation) |
| GFlowRL training | 85 steps (1 epoch), G = 5, β = 8, 8× A40, 21 h 57 min | No crash / no OOM; ckpts at step 30 / 60 / 85 | [`05_gflowrl_training/`](05_gflowrl_training) |
| GFlowRL eval | 9 benchmarks, zero OOM / truncation; per-sample comparison with GRPO | See table below | [`06_gflowrl_eval/`](06_gflowrl_eval) |
| Optimization P0 | Same machine, same KV pool, 3 runs each for SFT / GRPO / C′ | Gap located to one behaviour: the `obj_name` query written to RoboRefer | [`07_gflowrl_improvement/`](07_gflowrl_improvement) |

**Main results** (single-run numbers from the full report, §0):

| | SFT start | Official ckpt (GRPO, our re-eval) | **GFlowRL C′ step 85** | Paper |
|---|--:|--:|--:|--:|
| RoboSpatial overall | 61.00 ± 0.77 | 65.43–66.00 | **62.14** | 70.00 |
| RefSpatial (3 splits, weighted) | 53.07 | 53.79 | **53.07** | 53.07 |
| BLINK relative depth | – | 86.29–87.90 | **87.90** | 90.32 |
| CV-Bench 2D / 3D | – | 94.62 / 96.50 | **94.62 / 96.50** | 94.92 / 96.00 |

Same-machine 3-run re-evaluation on RoboSpatial (350 questions, mean correct): SFT 213.0 · GRPO 226.7 · C′ 221.7 (GRPO vs C′ p = 0.28). Split by type, C′ is +3.7 on VQA (n.s.) and **−8.6 on Vacant (p = 0.0004)**. The Vacant gap comes from how the policy queries RoboRefer: asking only for the anchor object ("cup") makes the tool point land in the target region 0–3 % of the time, while a location-aware query ("point in front of the cup") hits 56 %. GRPO learned to write better queries; C′ stayed at the SFT start (88.5 % of its queries are identical to SFT). Diagnosis: the training signal was starved (≈ 70 % reward-degenerate groups, gradient clipped every step), not that the objective is harmful.

## 4. What is next

The plan lives in [`docs/design_doc_gflowrl_optimization.md`](docs/design_doc_gflowrl_optimization.md) (Chinese) and is updated as work progresses:

| Priority | Change | Status |
|---|---|---|
| P1(a) | Filter reward-degenerate groups (set g̃ = 0), no resampling | Patch written and CPU-verified in [`07_gflowrl_improvement/p1_prep/`](07_gflowrl_improvement/p1_prep); **next GPU run** |
| P1(b) | Loss scale / `grad_clip = L̄`, and an ε-swap arm to settle the direction of Eq. 7 | Config-only, planned |
| P1(c) | Larger group size G = 8/16 | Conditional |
| P1(d) | epochs / β | Frozen, with trigger conditions |
| P2 | SFT data line (front/behind never calls the depth tool; Vacant queries only the anchor object) | Moved out, separate schedule |
| P3 | Re-train a GRPO control arm whenever G / epochs change | Conditional |

Primary metric for the next runs: the share of Vacant questions whose RoboRefer query names only the anchor object (monitored live by `p1_monitor.py`).

## 5. Repository layout

The folders follow the order of the pipeline. Reports and the design doc are written in Chinese; code, file names and this README are in English. Old paths quoted inside the reports are mapped to this repo in [`docs/PATH_MAP.md`](docs/PATH_MAP.md).

```
docs/                            Full report, design doc, paper notes, path map, file index
00_environment/                  Building / restoring the SFT env and the 5 eval+RL envs
  sft_env/                         SFT env build script + packaged-env restore/verify
  eval_rl_env/                     Eval/RL env build guide, build + verify scripts, POSTRESTORE
  upstream_patches/                Patches to SpaceTools-Toolshed, GraspGen, RoboRefer
  official_ckpt_fix/               Repair the config of the released checkpoint
  env_freeze/                      pip freeze of every env
01_official_checkpoint_eval/     Reproduction phases P2–P6 on the official GRPO checkpoint
  reports/                         Smoke test, trajectory capture, full eval, accuracy, error attribution
  tools/                           Eval runner, P5 accuracy + P6 attribution scripts
  p4/  p6/                         Trajectory dumps, parsed records, logs, ablations
02_sft_training/                 Step 3 SFT: our run_sft.sh (+ diff vs upstream), FINALIZE.sh, report
03_sft_eval/                     SFT eval gate: report, rollouts, logs, audit scripts
04_gflowrl_implementation/       GFlowRL (C′) code
  spacetools_rl_modified/          The 7 files changed in SpaceTools-RL, at the trained commit
  patches_spacetools_rl/           Same changes as 19 git patches on upstream 54270e82
  checks/                          Self-checks (fixed point, config guards, on-policy, …)
  design_notes/                    Why C′: route decision, gates, fixed-point derivation
05_gflowrl_training/             As-run configs, 85-step metrics CSV, training report, ops scripts
06_gflowrl_eval/                 Eval report, scripts, dumps, parsed records, gates, analysis
07_gflowrl_improvement/          Optimization work driven by the design doc
  p0_same_machine_reeval/          3 ckpts × 3 runs on one machine: results, scripts, dumps
  p1_prep/                         P1(a) degenerate-group filter patch, monitor scripts
  infra_prep/                      Training speed-ups (drop degenerate groups, ref on GPU), timing, NCCL pre-flight
tools/                           Shared: parse_dump.py (dump → trajectory records), unpack_dumps.sh
```

Where to find each piece:

| You want | Path |
|---|---|
| The whole story in one document | [`docs/full_report_v1.md`](docs/full_report_v1.md) |
| The current optimization plan | [`docs/design_doc_gflowrl_optimization.md`](docs/design_doc_gflowrl_optimization.md) |
| What we changed upstream, and every file we created | [§6](#6-what-we-changed-upstream) · [`docs/FILE_INDEX.md`](docs/FILE_INDEX.md) |
| GFlowRL loss (Eq. 8, gradient half) | [`04_gflowrl_implementation/spacetools_rl_modified/verl/trainer/ppo/core_algos.py`](04_gflowrl_implementation/spacetools_rl_modified/verl/trainer/ppo/core_algos.py) → `compute_policy_loss_gflowrl` |
| Flow gap / log Z (Eq. 4–7, no-grad half) | [`04_gflowrl_implementation/spacetools_rl_modified/verl/trainer/ppo/ray_trainer.py`](04_gflowrl_implementation/spacetools_rl_modified/verl/trainer/ppo/ray_trainer.py) → `compute_gflowrl_flow_gap` |
| GFlowRL launcher | [`04_gflowrl_implementation/spacetools_rl_modified/examples/toolshed/run_rl_gflowrl.sh`](04_gflowrl_implementation/spacetools_rl_modified/examples/toolshed/run_rl_gflowrl.sh) |
| SFT training script | [`02_sft_training/run_sft.sh`](02_sft_training/run_sft.sh) |
| GFlowRL training as run | [`05_gflowrl_training/config/`](05_gflowrl_training/config) |
| Eval from a bare GPU machine | [`06_gflowrl_eval/scripts/EVAL_FROM_SCRATCH.sh`](06_gflowrl_eval/scripts/EVAL_FROM_SCRATCH.sh) |
| Per-sample GRPO vs GFlowRL analysis | [`06_gflowrl_eval/analysis/`](06_gflowrl_eval/analysis) |
| Trajectories (full multi-turn transcripts incl. tool outputs) | `*/dumps/**.jsonl.gz`, parsed records in `*/parsed/` |

## 6. What we changed upstream

Everything this project runs is upstream code plus the changes below; all other files in this repo were written for the project. [`docs/FILE_INDEX.md`](docs/FILE_INDEX.md) has the long form: each change explained, the 19 patches one by one, and every file we created with what it does.

Upstream projects and the versions we started from:

| Upstream | Started from |
|---|---|
| [SpaceTools-RL](https://github.com/ChicyChen/SpaceTools-RL), a fork of [verl](https://github.com/volcengine/verl) | `54270e82` |
| [SpaceTools-Toolshed](https://github.com/NVlabs/SpaceTools-Toolshed) | `4f0512d` |
| [GraspGen](https://github.com/NVlabs/GraspGen) | `2dd8852` |
| [RoboRefer](https://github.com/Zhoues/RoboRefer) | `d97a995` |
| [SpaceTools-SFT](https://github.com/ChicyChen/SpaceTools-SFT) | `b7ebbf32` |
| [Ray](https://github.com/ray-project/ray) | 2.47.1 (pip) |
| Official checkpoint [`siyich/spacetools-ckpt`](https://huggingface.co/siyich/spacetools-ckpt) | `f953b1a1` |

Every upstream file we modified. Changes marked **P1(a)** or **Infra** are prepared and CPU-checked for the next GPU run; the 85-step training used the others.

| Upstream file | From | What we changed, and why | In this repo |
|---|---|---|---|
| `examples/toolshed/run_eval.sh` | SpaceTools-RL | Benchmark paths corrected to the current HF dataset layout; GPU budget and tool GPU shares set for a 4-GPU node, so the tools stop running out of memory; `DATA_DIR` override; conda shell hook; policy held in bf16 to fit a 40 GB card | [modified file](04_gflowrl_implementation/spacetools_rl_modified/examples/toolshed/run_eval.sh) · [patches](04_gflowrl_implementation/patches_spacetools_rl) 0001–0007, 0009 |
| `examples/toolshed/run_rl.sh` | SpaceTools-RL | Separate tool and training GPUs, so RL runs on a single node; tool actors scaled to the tool GPUs; Ray head started from the repo folder; correct exit code. **Infra:** `REF_PARAM_OFFLOAD` switch | [modified file](04_gflowrl_implementation/spacetools_rl_modified/examples/toolshed/run_rl.sh) · patches 0014, 0016–0018 · [Infra](07_gflowrl_improvement/infra_prep/patched/run_rl.sh) |
| `examples/toolshed/run_rl_gflowrl.sh` | new file, added to the SpaceTools-RL tree | GFlowRL launcher: a thin wrapper over `run_rl.sh` that selects the loss, sets β and ε, and keeps the two arms' outputs apart. **P1(a):** `GF_FILTER_DEGEN`. **Infra:** `GF_DROP_DEGEN` | [file](04_gflowrl_implementation/spacetools_rl_modified/examples/toolshed/run_rl_gflowrl.sh) · patch 0012 · [P1(a)](07_gflowrl_improvement/p1_prep/patched/run_rl_gflowrl.sh) · [Infra](07_gflowrl_improvement/infra_prep/patched/run_rl_gflowrl.sh) |
| `verl/trainer/ppo/core_algos.py` | SpaceTools-RL (verl) | The GFlowRL policy loss (Eq. 8), registered as `"gflowrl"` | [modified file](04_gflowrl_implementation/spacetools_rl_modified/verl/trainer/ppo/core_algos.py) · patches 0011, 0015 |
| `verl/trainer/ppo/ray_trainer.py` | SpaceTools-RL (verl) | The flow gap (Eq. 4–7), a guard against two silent misconfigurations, reward-degeneracy and β metrics; extra fields in eval dumps for offline analysis. **P1(a):** zero the flow gap of reward-degenerate groups. **Infra:** drop those groups before the three forward / backward passes; `dump_images` switch | [modified file](04_gflowrl_implementation/spacetools_rl_modified/verl/trainer/ppo/ray_trainer.py) · patches 0008, 0010–0012, 0015, 0019 · [P1(a)](07_gflowrl_improvement/p1_prep/patched/ray_trainer.py) · [Infra](07_gflowrl_improvement/infra_prep/patched/ray_trainer.py) |
| `verl/workers/config/actor.py`, `verl/trainer/config/actor/actor.yaml` | SpaceTools-RL (verl) | Declare the GFlowRL config keys so Hydra accepts them | [`actor.py`](04_gflowrl_implementation/spacetools_rl_modified/verl/workers/config/actor.py) · [`actor.yaml`](04_gflowrl_implementation/spacetools_rl_modified/verl/trainer/config/actor/actor.yaml) · patch 0013 |
| `verl/workers/fsdp_workers.py` | SpaceTools-RL (verl) | **Infra:** the reference model can stay on GPU (upstream ignores the flag) | [Infra](07_gflowrl_improvement/infra_prep/patched/fsdp_workers.py) |
| `verl/experimental/agent_loop/tool_agent_loop.py` | SpaceTools-RL (verl) | **Infra:** per-call tool timing log, to find what rollout generation waits for | [Infra](07_gflowrl_improvement/infra_prep/patched/tool_agent_loop.py) |
| `toolshed/tools/vlm.py` | SpaceTools-Toolshed | Honour the `dtype` argument and cast inputs to the model dtype | [patch](00_environment/upstream_patches/SpaceTools-Toolshed) |
| `toolshed/tools/graspgen_franka_panda.yml` | SpaceTools-Toolshed | Checkpoint paths made absolute, so they resolve inside a Ray actor | same patch |
| `toolshed/integration/verl.py` | SpaceTools-Toolshed | **Infra:** timestamps that split a tool call's latency into thread-pool wait and remote time | [Infra](07_gflowrl_improvement/infra_prep/patched_toolshed/verl.py) |
| `pyproject.toml`, `requirements.txt` | GraspGen | Drop `pickle5`, which cannot compile on Python 3.11 | [patch](00_environment/upstream_patches/GraspGen) |
| `pointnet2_ops/pointnet2_ops/pointnet2_utils.py` | GraspGen | The hardcoded GPU architecture list no longer overrides the build setting | [`repair_graspgen_step6.sh`](00_environment/eval_rl_env/graspgen_fixes/repair_graspgen_step6.sh) |
| `env_setup.sh` | RoboRefer | Skip a hardcoded Python 3.10 wheel that aborts the install on Python 3.11 | [patch](00_environment/upstream_patches/RoboRefer) |
| `scripts/spacetools/run_sft.sh` | SpaceTools-SFT | Batch size derived from the GPU count so the global batch stays 8; ZeRO-2; `save_only_model`; `eval_steps` 500 | [`run_sft.sh`](02_sft_training/run_sft.sh) · [diff](02_sft_training/run_sft.sh.diff_vs_upstream) |
| `ray/_private/node.py` | Ray | Python version check relaxed to the minor level, so tool environments on a different patch version can join the cluster | [`POSTRESTORE.sh`](00_environment/eval_rl_env/POSTRESTORE.sh) |
| `config.json`, `preprocessor_config.json` | official checkpoint | Config repaired so sglang can load the checkpoint | [`fix_checkpoint.py`](00_environment/official_ckpt_fix/fix_checkpoint.py) |

## 7. Hugging Face artifacts

| Repo | Contents |
|---|---|
| [`qzpm55555/spacetools-sft-v1-4xa6000`](https://huggingface.co/qzpm55555/spacetools-sft-v1-4xa6000) | Our Step 3 SFT checkpoint (π_ref for RL) + training evidence (eval logs, data provenance). Eval pinned at revision `91fd4bdf6d2ffe3b355cc650e71925c6e91a15c5` |
| [`qzpm55555/spacetools-p7-gflowrl-cprime-8xa40`](https://huggingface.co/qzpm55555/spacetools-p7-gflowrl-cprime-8xa40) | GFlowRL C′ checkpoints `global_step_30/60/85` (HF format, 7.6 GB each) + provenance bundle with full training logs |
| [`qzpm55555/spacetools-eval-env`](https://huggingface.co/qzpm55555/spacetools-eval-env) | Packaged eval/RL environment (5 conda envs, ~22 GB split archive) with `FETCH_WEIGHTS.sh`, `RESTORE.sh`, `VERIFY.sh`, `POSTRESTORE.sh` |

Upstream artifacts used: base model [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), official checkpoint [`siyich/spacetools-ckpt`](https://huggingface.co/siyich/spacetools-ckpt), SFT data [`siyich/spacetools-sft`](https://huggingface.co/datasets/siyich/spacetools-sft), RL data [`siyich/spacetools-rlfulltools`](https://huggingface.co/datasets/siyich/spacetools-rlfulltools), eval benchmarks [`siyich/spacetools-eval-benchmarks`](https://huggingface.co/datasets/siyich/spacetools-eval-benchmarks) @ `1d539ac9`, tool weights [`Zhoues/RoboRefer-8B-SFT`](https://huggingface.co/Zhoues/RoboRefer-8B-SFT), [`allenai/Molmo-7B-D-0924`](https://huggingface.co/allenai/Molmo-7B-D-0924), [`facebook/sam2.1-hiera-small`](https://huggingface.co/facebook/sam2.1-hiera-small), [`adithyamurali/GraspGenModels`](https://huggingface.co/adithyamurali/GraspGenModels) (all revisions pinned in [`03_sft_eval/config/WEIGHTS_PINS.txt`](03_sft_eval/config/WEIGHTS_PINS.txt)).

## 8. Reproducing

All GPU steps need Linux x86_64, ≥ 4 GPUs of compute capability 8.0 / 8.6 / 8.9 with ≥ 40 GB each (H100 is not covered by the packaged env), driver ≥ 550, glibc ≥ 2.39 for the eval env. Set `HF_TOKEN` before downloading (anonymous downloads get rate-limited).

**Offline analysis (no GPU).** Everything under `*/analysis`, `*/tools`, `04_gflowrl_implementation/checks` and `07_gflowrl_improvement/p1_prep` runs on a laptop with Python 3 + numpy / pyyaml:

```bash
git clone https://github.com/qyYue1389/Spacetools-RL-Improvement && cd Spacetools-RL-Improvement
bash tools/unpack_dumps.sh                       # dumps are stored as .jsonl.gz
cd 06_gflowrl_eval/analysis
python3 compare_p4_p7.py                         # per-sample GRPO vs C′ (rewrites compare_p4_p7.txt)
python3 significance_p4_p7.py                    # McNemar + sign tests
python3 p6_criteria_p4_vs_p7.py                  # P6 criteria, question types, Vacant pass-through vs edit
python3 ../../07_gflowrl_improvement/p1_prep/p2_query_type.py   # obj_name query type split (P0 dumps)
```

**Step 0 — environment.** Fastest: restore the packaged env from HF (`00_environment/eval_rl_env/BUILD_GUIDE.md` §4.1), then run `POSTRESTORE.sh` and `VERIFY.sh`. From scratch: follow `BUILD_GUIDE.md` §3 with `build_eval_envs.sh`, the six `graspgen_fixes/` in order, and the patches in `00_environment/upstream_patches/`. The SFT env is separate and much smaller: `00_environment/sft_env/setup_sft_env.sh`.

**Step 1 — official checkpoint eval.** Download `siyich/spacetools-ckpt`, repair its config with `00_environment/official_ckpt_fix/fix_checkpoint.py`, then run the patched `run_eval.sh` (see `01_official_checkpoint_eval/tools/p4_run.sh`).

**Step 2 — SFT.** Clone [SpaceTools-SFT](https://github.com/ChicyChen/SpaceTools-SFT) (@ `b7ebbf32`), replace `scripts/spacetools/run_sft.sh` with [`02_sft_training/run_sft.sh`](02_sft_training/run_sft.sh) (it derives per-device batch / grad-accum so the global batch stays 8 on any GPU count), run it, then `bash 02_sft_training/FINALIZE.sh` to verify and upload.

**Step 3 — GFlowRL code.**

```bash
git clone https://github.com/ChicyChen/SpaceTools-RL && cd SpaceTools-RL
git checkout 54270e82443d3d2a4c2a737c2d3b33314a991fcc
git am ../Spacetools-RL-Improvement/04_gflowrl_implementation/patches_spacetools_rl/*.patch
SPACETOOLS_RL=$PWD bash ../Spacetools-RL-Improvement/04_gflowrl_implementation/checks/run_checks.sh
```

**Step 4 — GFlowRL training** (8 GPUs: 4 tools + 4 training), exactly as run in [`05_gflowrl_training/config/full_train.sh.asrun`](05_gflowrl_training/config/full_train.sh.asrun):

```bash
export SFT_CHECKPOINT=/path/to/sft-ckpt ROBOREFER_MODEL=/path/to/RoboRefer-8B-SFT DEPTH_CHECKPOINT=/path/to/depth_pro.pt
export OUTPUT_DIR=/path/to/exp GPUS_PER_NODE=8 TOOL_GPUS=4 TRAIN_GPUS=4 SAVE_FREQ=5 GF_VARIANT=cprime
export NCCL_P2P_DISABLE=1          # PCIe P2P was broken in our container, see the training report
bash examples/toolshed/run_rl_gflowrl.sh trainer.test_freq=-1 trainer.val_before_train=False
# GRPO control arm, same start: bash examples/toolshed/run_rl.sh actor_rollout_ref.actor.kl_loss_coef=0
```

**Step 5 — eval.** `P7_STEP=85 bash 06_gflowrl_eval/scripts/EVAL_FROM_SCRATCH.sh` (idempotent; checks OOM / truncation gates before any score is read). Always check the gates: an out-of-memory tool call does not crash the eval, it silently lowers the score.

Scores are only comparable under the same protocol: `gpu_memory_utilization` set so the KV pool is 24 GB, greedy decoding, `vlm num_gpus=1.0`.

## 9. References

- SpaceTools: *SpaceTools: Tool-Augmented Spatial Reasoning via Double Interactive RL*, [arXiv:2512.04069](https://arxiv.org/abs/2512.04069) · code [spacetools/SpaceTools](https://github.com/spacetools/SpaceTools) · [SpaceTools-RL](https://github.com/ChicyChen/SpaceTools-RL) · [SpaceTools-SFT](https://github.com/ChicyChen/SpaceTools-SFT) · [SpaceTools-Toolshed](https://github.com/NVlabs/SpaceTools-Toolshed)
- GFlowRL: *GFlowRL: Scaling Distribution-Matching RL to Large Language Models*, [arXiv:2607.13394](https://arxiv.org/abs/2607.13394) (official code not released at the time of writing)
- Frameworks: [verl](https://github.com/volcengine/verl) · [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) · [sglang](https://github.com/sgl-project/sglang) · [Ray](https://github.com/ray-project/ray)
- Tools: [RoboRefer](https://github.com/Zhoues/RoboRefer) · [Molmo](https://huggingface.co/allenai/Molmo-7B-D-0924) · [Depth Pro](https://github.com/apple/ml-depth-pro) · [SAM 2](https://github.com/facebookresearch/sam2) · [GraspGen](https://github.com/NVlabs/GraspGen)
- Related objectives discussed in the design notes: [FlowRL](https://github.com/Xuekai-Zhu/FlowRL) · [FoR](https://github.com/Yu-Fangxu/FoR) · [TBA](https://github.com/bbartoldson/TBA) · GRPO ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300)) · PPO ([arXiv:1707.06347](https://arxiv.org/abs/1707.06347))

## License

Apache-2.0 (see [LICENSE](LICENSE)). Files under `04_gflowrl_implementation/spacetools_rl_modified/`, `07_gflowrl_improvement/p1_prep/patched/` and `07_gflowrl_improvement/infra_prep/patched/` are modified versions of [SpaceTools-RL](https://github.com/ChicyChen/SpaceTools-RL) / [verl](https://github.com/volcengine/verl) files, and `02_sft_training/run_sft.sh` of a [SpaceTools-SFT](https://github.com/ChicyChen/SpaceTools-SFT) file; all three projects are Apache-2.0. `07_gflowrl_improvement/infra_prep/patched_toolshed/verl.py` and the patches under `00_environment/upstream_patches/` derive from [SpaceTools-Toolshed](https://github.com/NVlabs/SpaceTools-Toolshed), [GraspGen](https://github.com/NVlabs/GraspGen) and [RoboRefer](https://github.com/Zhoues/RoboRefer) and remain under those projects' own licenses. The full list of modified upstream files is in §6.

---

<a id="中文"></a>

## 1. 背景

[SpaceTools](https://github.com/spacetools/SpaceTools)([arXiv:2512.04069](https://arxiv.org/abs/2512.04069),CVPR 2026)让 Qwen2.5-VL-3B 通过多轮调用七个视觉工具(RoboRefer 定位、Molmo、DepthPro 深度、SAM2、GraspGen、bbox、vision_ops)解决度量级空间推理题。它的流水线(DIRL)分四步:① 单工具 RL ② 采教师轨迹 ③ SFT ④ 全工具 RL(**GRPO**)。

[GFlowRL](https://arxiv.org/abs/2607.13394)(arXiv:2607.13394)是**分布匹配**目标:不把概率质量压到单一最优模式上,而是让策略收敛到 π ∝ π_ref · exp(βr)。工具调用天然有多条等价有效的链路,而我们对官方 checkpoint 的错题归因发现了明显的**工具编排坍塌**(例如 RefSpatial 277/277 条样本走完全相同的工具链)。这就是本项目切入的交叉点。

## 2. 目标

1. 在租来的小规模 GPU(A6000 / A100 / A40,而不是论文的 2×8 A100-80GB)上端到端复现 SpaceTools。
2. 自己训 SFT checkpoint(第 3 步),再把**第 4 步的 GRPO 目标换成 GFlowRL**,数据、G、epoch、起点保持不变。
3. 在论文 Table 2 的九个 benchmark 上评测,并与官方 GRPO checkpoint **逐样本**对比。
4. 查清 GFlowRL 为什么有效或无效,并迭代算法(进行中,见 design doc)。

## 3. 已完成

| 阶段 | 内容 | 结果 | 位置 |
|---|---|---|---|
| 环境 | 5 个 conda 环境(driver + 4 个工具环境),修了 20 个上游问题,打包上传 HF | 还原与验收脚本;支持 sm_80 / 86 / 89 | [`00_environment/`](00_environment) |
| 官方 ckpt 评测(复现 P2–P6) | `siyich/spacetools-ckpt` 九个 benchmark 全量评测(2121 样本)、accuracy 报告、逐样本错题归因 | 与论文的每处差距都做了归因;322 个错题全部分类 | [`01_official_checkpoint_eval/`](01_official_checkpoint_eval) |
| SFT(第 3 步) | Qwen2.5-VL-3B,3000 步,4× A6000,全局 batch 8 | eval_loss 0.168 → 0.045,作为 π_ref | [`02_sft_training/`](02_sft_training) |
| SFT eval 闸门 | SFT ckpt 上评 RoboSpatial + RefSpatial | RoboSpatial 61.00 ± 0.77、RefSpatial 53.07 → 进入 RL | [`03_sft_eval/`](03_sft_eval) |
| GFlowRL 实现 | **C′ 变体**(不做长度归一化,不动点精确),verl 两个插入点;SpaceTools-RL 上 19 个 commit(10 个 eval/基础设施修复 + 9 个 GFlowRL) | 不动点自检 7.4e-13 PASS | [`04_gflowrl_implementation/`](04_gflowrl_implementation) |
| GFlowRL 训练 | 85 步(1 epoch),G = 5,β = 8,8× A40,21 h 57 min | 无崩溃 / 无 OOM;存了 step 30 / 60 / 85 | [`05_gflowrl_training/`](05_gflowrl_training) |
| GFlowRL eval | 九个 benchmark,零 OOM / 零截断;与 GRPO 逐样本对比 | 见上文英文部分的结果表 | [`06_gflowrl_eval/`](06_gflowrl_eval) |
| 优化 P0 | 同机同池,SFT / GRPO / C′ 各跑 3 次 | 差距定位到一个行为:给 RoboRefer 的 `obj_name` 怎么写 | [`07_gflowrl_improvement/`](07_gflowrl_improvement) |

**结论摘要。** 同机三次复评 RoboSpatial(350 题,答对均值):SFT 213.0 · GRPO 226.7 · C′ 221.7(GRPO vs C′ p = 0.28)。按题型拆开,C′ 在 VQA 上多 3.7 题(不显著),在 **Vacant 上少 8.6 题(p = 0.0004)**。Vacant 的差距来自查询写法:`obj_name` 只写锚物体("cup")时工具点落在目标区域的比例只有 0–3%,带位置描述时是 56%。GRPO 学会了改写查询,C′ 停在 SFT 起点(88.5% 的查询与 SFT 一字不差)。判断:是训练信号不足(约 70% 退化组、每步都被 clip),不是 C′ 目标函数有害。

## 4. 下一步

计划写在 [`docs/design_doc_gflowrl_optimization.md`](docs/design_doc_gflowrl_optimization.md),随进展持续更新:

| 优先级 | 改动 | 状态 |
|---|---|---|
| P1(a) | 过滤退化组(g̃ 置 0,不补采) | 补丁已写好并在 CPU 上自检通过([`07_gflowrl_improvement/p1_prep/`](07_gflowrl_improvement/p1_prep)),**下一次 GPU 实验** |
| P1(b) | loss 尺度 / `grad_clip = L̄`,以及交换 ε 的对照臂(确定 Eq. 7 的方向) | 只改配置,计划中 |
| P1(c) | 增大 G 到 8/16 | 条件触发 |
| P1(d) | epochs / β | 冻结,设了触发条件 |
| P2 | SFT 数据线(front/behind 不调深度工具;Vacant 只问锚物体) | 移出,另立排期 |
| P3 | 动了 G / epochs 时同批重训 GRPO 对照臂 | 条件触发 |

主判据:Vacant 里调 RoboRefer 时 `obj_name` 只写锚物体的题目占比(`p1_monitor.py` 训练中实时监控)。

## 5. 目录结构

目录按流水线顺序编号。报告和 design doc 是中文;代码、文件名和本 README 是英文。报告里引用的旧路径与本 repo 的对应关系见 [`docs/PATH_MAP.md`](docs/PATH_MAP.md)。

| 想找 | 路径 |
|---|---|
| 一份文档看完全部 | [`docs/full_report_v1.md`](docs/full_report_v1.md) |
| 当前优化计划 | [`docs/design_doc_gflowrl_optimization.md`](docs/design_doc_gflowrl_optimization.md) |
| 对上游的改动、我们创建的每个文件 | [§6](#6-对上游的改动) · [`docs/FILE_INDEX.md`](docs/FILE_INDEX.md) |
| 环境搭建 | [`00_environment/`](00_environment) |
| 官方 ckpt 评测与错题归因 | [`01_official_checkpoint_eval/`](01_official_checkpoint_eval) |
| SFT 训练 / SFT eval | [`02_sft_training/`](02_sft_training) · [`03_sft_eval/`](03_sft_eval) |
| GFlowRL 代码实现 | [`04_gflowrl_implementation/spacetools_rl_modified/`](04_gflowrl_implementation/spacetools_rl_modified)(改动的 7 个文件)· [`patches_spacetools_rl/`](04_gflowrl_implementation/patches_spacetools_rl)(19 个 patch) |
| GFlowRL 训练 | [`05_gflowrl_training/`](05_gflowrl_training) |
| GFlowRL eval 与逐样本分析 | [`06_gflowrl_eval/`](06_gflowrl_eval) |
| 优化工作(P0 复评、P1 准备) | [`07_gflowrl_improvement/`](07_gflowrl_improvement) |
| 轨迹 / dump | 各目录下 `dumps/**.jsonl.gz`(完整多轮轨迹,含工具真实返回),结构化记录在 `parsed/` |
| 公用工具 | [`tools/`](tools):`parse_dump.py`(dump → 轨迹记录)、`unpack_dumps.sh` |

**阶段编号说明:** 复现阶段沿用 P0–P7(P4 = 官方 ckpt 全量评测,P5 = accuracy,P6 = 错题归因,P7 = 换成 GFlowRL);design doc 里的优化阶段另有一套 P0–P3,放在 `07_gflowrl_improvement/`。

## 6. 对上游的改动

本项目跑的代码 = 上游代码 + 下面这些改动;repo 里其余文件都是为本项目写的。详细版在 [`docs/FILE_INDEX.md`](docs/FILE_INDEX.md):每处改动的说明、19 个 patch 逐个的作用,以及我们创建的每个文件是做什么的。

上游项目与起始版本:

| 上游 | 起始版本 |
|---|---|
| [SpaceTools-RL](https://github.com/ChicyChen/SpaceTools-RL)([verl](https://github.com/volcengine/verl) 的 fork) | `54270e82` |
| [SpaceTools-Toolshed](https://github.com/NVlabs/SpaceTools-Toolshed) | `4f0512d` |
| [GraspGen](https://github.com/NVlabs/GraspGen) | `2dd8852` |
| [RoboRefer](https://github.com/Zhoues/RoboRefer) | `d97a995` |
| [SpaceTools-SFT](https://github.com/ChicyChen/SpaceTools-SFT) | `b7ebbf32` |
| [Ray](https://github.com/ray-project/ray) | 2.47.1(pip) |
| 官方 checkpoint [`siyich/spacetools-ckpt`](https://huggingface.co/siyich/spacetools-ckpt) | `f953b1a1` |

改动过的全部上游文件。标 **P1(a)** 或 **Infra** 的改动已写好并在 CPU 上自检通过,留给下一次 GPU 实验;85 步训练用的是其余改动。

| 上游文件 | 来源 | 改了什么、为什么 | 本 repo 位置 |
|---|---|---|---|
| `examples/toolshed/run_eval.sh` | SpaceTools-RL | benchmark 路径改成 HF 数据集现在的布局;GPU 预算和工具的 GPU 份额按 4 卡设置,工具不再 OOM;新增 `DATA_DIR` 覆盖;加载 conda shell hook;策略模型用 bf16,40 GB 的卡才放得下 | [改后的文件](04_gflowrl_implementation/spacetools_rl_modified/examples/toolshed/run_eval.sh) · [patch](04_gflowrl_implementation/patches_spacetools_rl) 0001–0007、0009 |
| `examples/toolshed/run_rl.sh` | SpaceTools-RL | 工具和训练的 GPU 分开,RL 才能在单机上跑;工具 actor 数按工具卡数缩放;Ray head 从仓库目录启动;退出码修正。**Infra:**`REF_PARAM_OFFLOAD` 开关 | [改后的文件](04_gflowrl_implementation/spacetools_rl_modified/examples/toolshed/run_rl.sh) · patch 0014、0016–0018 · [Infra](07_gflowrl_improvement/infra_prep/patched/run_rl.sh) |
| `examples/toolshed/run_rl_gflowrl.sh` | 新文件,加在 SpaceTools-RL 目录里 | GFlowRL 启动脚本:`run_rl.sh` 的薄封装,选 loss、设 β 和 ε,两臂输出分开。**P1(a):**`GF_FILTER_DEGEN`。**Infra:**`GF_DROP_DEGEN` | [文件](04_gflowrl_implementation/spacetools_rl_modified/examples/toolshed/run_rl_gflowrl.sh) · patch 0012 · [P1(a)](07_gflowrl_improvement/p1_prep/patched/run_rl_gflowrl.sh) · [Infra](07_gflowrl_improvement/infra_prep/patched/run_rl_gflowrl.sh) |
| `verl/trainer/ppo/core_algos.py` | SpaceTools-RL(verl) | GFlowRL 的 policy loss(Eq. 8),注册为 `"gflowrl"` | [改后的文件](04_gflowrl_implementation/spacetools_rl_modified/verl/trainer/ppo/core_algos.py) · patch 0011、0015 |
| `verl/trainer/ppo/ray_trainer.py` | SpaceTools-RL(verl) | flow gap(Eq. 4–7)、两种静默错误配置的守卫、奖励退化与 β 分解指标;eval dump 多写几个字段供离线分析。**P1(a):**把奖励退化组的 flow gap 置 0。**Infra:**在三次前向 / 反向之前把这些组删掉;`dump_images` 开关 | [改后的文件](04_gflowrl_implementation/spacetools_rl_modified/verl/trainer/ppo/ray_trainer.py) · patch 0008、0010–0012、0015、0019 · [P1(a)](07_gflowrl_improvement/p1_prep/patched/ray_trainer.py) · [Infra](07_gflowrl_improvement/infra_prep/patched/ray_trainer.py) |
| `verl/workers/config/actor.py`、`verl/trainer/config/actor/actor.yaml` | SpaceTools-RL(verl) | 声明 GFlowRL 的配置键,Hydra 才接受 | [`actor.py`](04_gflowrl_implementation/spacetools_rl_modified/verl/workers/config/actor.py) · [`actor.yaml`](04_gflowrl_implementation/spacetools_rl_modified/verl/trainer/config/actor/actor.yaml) · patch 0013 |
| `verl/workers/fsdp_workers.py` | SpaceTools-RL(verl) | **Infra:**参考模型可以留在 GPU 上(上游不看这个开关) | [Infra](07_gflowrl_improvement/infra_prep/patched/fsdp_workers.py) |
| `verl/experimental/agent_loop/tool_agent_loop.py` | SpaceTools-RL(verl) | **Infra:**逐次记录工具调用的耗时,用来查 rollout 生成在等什么 | [Infra](07_gflowrl_improvement/infra_prep/patched/tool_agent_loop.py) |
| `toolshed/tools/vlm.py` | SpaceTools-Toolshed | 让 `dtype` 参数生效,并把输入转成模型的 dtype | [patch](00_environment/upstream_patches/SpaceTools-Toolshed) |
| `toolshed/tools/graspgen_franka_panda.yml` | SpaceTools-Toolshed | checkpoint 路径改成绝对路径,在 Ray actor 里才解析得到 | 同一个 patch |
| `toolshed/integration/verl.py` | SpaceTools-Toolshed | **Infra:**加时间戳,把一次工具调用的耗时拆成等线程和远端两段 | [Infra](07_gflowrl_improvement/infra_prep/patched_toolshed/verl.py) |
| `pyproject.toml`、`requirements.txt` | GraspGen | 去掉 `pickle5`,它在 Python 3.11 上编不过 | [patch](00_environment/upstream_patches/GraspGen) |
| `pointnet2_ops/pointnet2_ops/pointnet2_utils.py` | GraspGen | 写死的 GPU 架构列表不再覆盖构建时的设置 | [`repair_graspgen_step6.sh`](00_environment/eval_rl_env/graspgen_fixes/repair_graspgen_step6.sh) |
| `env_setup.sh` | RoboRefer | 跳过写死的 Python 3.10 wheel,它会让 Python 3.11 下的安装中止 | [patch](00_environment/upstream_patches/RoboRefer) |
| `scripts/spacetools/run_sft.sh` | SpaceTools-SFT | batch 按 GPU 数推导,全局 batch 保持 8;ZeRO-2;`save_only_model`;`eval_steps` 500 | [`run_sft.sh`](02_sft_training/run_sft.sh) · [diff](02_sft_training/run_sft.sh.diff_vs_upstream) |
| `ray/_private/node.py` | Ray | Python 版本检查放宽到 minor 档,补丁版本不同的工具环境才能加入集群 | [`POSTRESTORE.sh`](00_environment/eval_rl_env/POSTRESTORE.sh) |
| `config.json`、`preprocessor_config.json` | 官方 checkpoint | 修复配置,sglang 才能加载这个 checkpoint | [`fix_checkpoint.py`](00_environment/official_ckpt_fix/fix_checkpoint.py) |

## 7. Hugging Face

| Repo | 内容 |
|---|---|
| [`qzpm55555/spacetools-sft-v1-4xa6000`](https://huggingface.co/qzpm55555/spacetools-sft-v1-4xa6000) | 我们训的 SFT checkpoint(RL 的 π_ref)及训练证据 |
| [`qzpm55555/spacetools-p7-gflowrl-cprime-8xa40`](https://huggingface.co/qzpm55555/spacetools-p7-gflowrl-cprime-8xa40) | GFlowRL C′ 的 `global_step_30/60/85`(HF 格式)与完整训练日志 |
| [`qzpm55555/spacetools-eval-env`](https://huggingface.co/qzpm55555/spacetools-eval-env) | eval/RL 环境包(5 个 conda 环境,约 22 GB)及还原、验收脚本 |

## 8. 复现

步骤见上文英文部分第 8 节,命令相同。无 GPU 的离线分析:`bash tools/unpack_dumps.sh` 之后直接跑 `06_gflowrl_eval/analysis/` 下的脚本。

## 9. 参考

见上文英文部分第 9 节。
