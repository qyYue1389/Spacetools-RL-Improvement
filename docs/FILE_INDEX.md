# File index / 文件清单

Which files in this repository come from an upstream project and were modified by us, and which files we wrote ourselves — with what each change or file does.

本 repo 里哪些文件来自上游并被我们改过,哪些是我们自己写的;每处改动、每个文件是做什么的。

| Section | Content | 内容 |
|---|---|---|
| [§1](#1-upstream-sources--上游来源) | Upstream projects and the versions we started from | 上游项目与起始版本 |
| [§2](#2-upstream-files-we-modified--改动过的上游文件) | Every upstream file we modified: what changed, why, where it is here | 每个改动过的上游文件:改了什么、为什么、在本 repo 的位置 |
| [§3](#3-run-specific-variants-and-unmodified-upstream-copies--按次运行的变体与未改动的上游副本) | Copies of upstream files kept as run, or kept unmodified for reference | 按实际运行保存的上游文件副本,以及未改动的参考副本 |
| [§4](#4-files-we-created--自己创建的文件) | Every file we created, by folder | 自己创建的全部文件,按目录 |

Conventions / 约定:

- Paths are relative to the repository root. `*` stands for any text. 路径相对 repo 根目录,`*` 表示任意字符。
- Code, scripts, configs and documents are listed one by one. Run outputs (dumps, logs, parsed records, images) are grouped by pattern. 代码、脚本、配置、文档逐个列出;运行产物(dump、日志、解析记录、图片)按模式分组。
- Keep this file in step with the repo: a new or renamed file gets a row here. 新增或改名的文件要在这里补一行。

---

## 1. Upstream sources / 上游来源

| Upstream / 上游 | URL | Version we started from / 起始版本 | Role / 用途 |
|---|---|---|---|
| SpaceTools | https://github.com/spacetools/SpaceTools | `17d58553` | Entry repo: `setup_envs.sh`, docs, and the three repos below as submodules / 入口仓库,下面三个是它的子模块 |
| SpaceTools-RL | https://github.com/ChicyChen/SpaceTools-RL | `54270e82443d3d2a4c2a737c2d3b33314a991fcc` | RL and eval code; a fork of [verl](https://github.com/volcengine/verl), so files under `verl/` originate there / RL 与 eval 代码,是 verl 的 fork,`verl/` 下的文件源自 verl |
| SpaceTools-SFT | https://github.com/ChicyChen/SpaceTools-SFT | `b7ebbf320bb130c230856e61107a566bc119e4d8` | SFT code, based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) / SFT 代码 |
| SpaceTools-Toolshed | https://github.com/NVlabs/SpaceTools-Toolshed | `4f0512d` | Tool server (Ray actors for the vision tools) / 工具服务 |
| GraspGen | https://github.com/NVlabs/GraspGen | `2dd8852` | Grasp tool, installed by Toolshed / 抓取工具 |
| RoboRefer | https://github.com/Zhoues/RoboRefer | `d97a995` | Pointing tool, installed by Toolshed / 指点工具 |
| Ray | https://github.com/ray-project/ray | 2.47.1 (pip) | Cluster runtime / 集群运行时 |
| Official checkpoint | https://huggingface.co/siyich/spacetools-ckpt | `f953b1a1` | The released GRPO checkpoint / 官方 GRPO checkpoint |

Our commits on top of upstream / 我们在上游之上的 commit:

- SpaceTools-RL `c6fef78a` = `54270e82` + the 19 patches in `04_gflowrl_implementation/patches_spacetools_rl/`. This is the commit the GFlowRL run was trained with. 这是 GFlowRL 训练实际使用的 commit。
- SpaceTools-Toolshed `712e557` = `4f0512d` + the patch in `00_environment/upstream_patches/SpaceTools-Toolshed/`.

---

## 2. Upstream files we modified / 改动过的上游文件

Each row is one upstream file. "In this repo" gives the modified copy and the patch that produces it.

每行一个上游文件。「本 repo 位置」给出改好的副本和生成它的 patch。

### 2.1 SpaceTools-RL (base `54270e82`)

| Upstream file / 上游文件 | Change and purpose | 改动与作用 | In this repo / 本 repo 位置 |
|---|---|---|---|
| `examples/toolshed/run_eval.sh` | Benchmark parquet paths corrected to the current layout of the HF dataset (otherwise the script stops at its "Missing:" precheck). GPU budget retargeted to a 4-GPU node: `NUM_GPUS` 8→4, `EVAL_GPUS`→1, tool GPU fractions re-balanced to measured memory so the large tools no longer share a card and run out of memory. `DATA_DIR` override, to reuse or subset the benchmarks. Molmo `dtype` set to `'auto'` (keeps the fp32 behaviour upstream actually had). Loads conda's shell hook so the script runs under plain `bash`. Policy held in bf16 so it fits a 40 GB card (eval only). | benchmark 的 parquet 路径改成 HF 数据集现在的布局(否则脚本在 "Missing:" 预检处退出)。GPU 预算改为 4 卡:`NUM_GPUS` 8→4、`EVAL_GPUS`→1,工具的 GPU 份额按实测显存重排,大工具不再挤在同一张卡上 OOM。新增 `DATA_DIR` 覆盖,可复用或截取 benchmark。Molmo 的 `dtype` 设为 `'auto'`(保持上游实际的 fp32 行为)。加载 conda 的 shell hook,脚本可直接用 `bash` 跑。策略模型用 bf16,40 GB 的卡才放得下(仅 eval)。 | `04_gflowrl_implementation/spacetools_rl_modified/examples/toolshed/run_eval.sh`; patches 0001–0007, 0009 |
| `examples/toolshed/run_rl.sh` | `TOOL_GPUS` / `TRAIN_GPUS` split, so tools and training no longer book the same cards on a single node; an assert that the tool actors' GPU demand fits; tool actor counts scaled to `TOOL_GPUS` (and `vlm` raised to 0.7 on the scaled path); Ray head started from `VERL_DIR` so workers resolve the relative image paths; `set +e` in `cleanup` so a successful run exits 0 and `ray stop` runs; `VERL_DUMP_TOKEN_DIAGNOSTICS` unset. **Infra:** `ref.fsdp_config.param_offload` now reads `REF_PARAM_OFFLOAD` (default `True`, unchanged). | 拆出 `TOOL_GPUS` / `TRAIN_GPUS`,单机上工具和训练不再预订同一批卡;断言工具 actor 的 GPU 需求放得下;工具 actor 数按 `TOOL_GPUS` 缩放(缩放时 `vlm` 提到 0.7);Ray head 从 `VERL_DIR` 启动,worker 才能解析相对的图片路径;`cleanup` 开头加 `set +e`,成功的训练才返回 0 并执行 `ray stop`;取消 `VERL_DUMP_TOKEN_DIAGNOSTICS`。**Infra:**`ref.fsdp_config.param_offload` 改读 `REF_PARAM_OFFLOAD`(默认 `True`,行为不变)。 | `04_gflowrl_implementation/spacetools_rl_modified/examples/toolshed/run_rl.sh`; patches 0014, 0016, 0017, 0018. Infra: `07_gflowrl_improvement/infra_prep/patched/run_rl.sh` |
| `examples/toolshed/run_rl_gflowrl.sh` (new file in the upstream tree / 加在上游目录里的新文件) | Thin wrapper over `run_rl.sh`: selects the GFlowRL loss, passes β, ε and the variant, sets `kl_loss_coef=0`, and writes to its own output folder so the two arms cannot mix. **P1(a):** switch `GF_FILTER_DEGEN`. **Infra:** switch `GF_DROP_DEGEN`. | `run_rl.sh` 的薄封装:选 GFlowRL loss,传 β、ε 和变体,设 `kl_loss_coef=0`,输出到独立目录,两臂不会混。**P1(a):**开关 `GF_FILTER_DEGEN`。**Infra:**开关 `GF_DROP_DEGEN`。 | `04_gflowrl_implementation/spacetools_rl_modified/examples/toolshed/run_rl_gflowrl.sh`; patch 0012. P1(a): `07_gflowrl_improvement/p1_prep/patched/run_rl_gflowrl.sh`, `07_gflowrl_improvement/p1_prep/p1a_run_rl_gflowrl.diff`. Infra: `07_gflowrl_improvement/infra_prep/patched/run_rl_gflowrl.sh` |
| `verl/trainer/ppo/core_algos.py` | New policy loss `compute_policy_loss_gflowrl()`, registered as `"gflowrl"`: the gradient half of GFlowRL (Eq. 8) with a stop-gradient importance weight. Adds IS-weight metrics and reports `ppo_kl` as a per-token mean like the other losses. | 新增 policy loss `compute_policy_loss_gflowrl()`,注册为 `"gflowrl"`:GFlowRL 带梯度的一半(Eq. 8),重要性权重不传梯度。新增 IS 权重指标,`ppo_kl` 改成和其他 loss 一样的逐 token 均值。 | `04_gflowrl_implementation/spacetools_rl_modified/verl/trainer/ppo/core_algos.py`; patches 0011, 0015 |
| `verl/trainer/ppo/ray_trainer.py` | **Eval dumps:** writes `uid`, `num_turns`, and (optionally) the response mask, token ids and both log-prob arms, for offline trajectory analysis. **GFlowRL:** `compute_gflowrl_flow_gap()`, the no-gradient half (Eq. 4 group log Z, Eq. 6 flow gap, Eq. 7 clip), written into `advantages`; a guard that raises on the two silent misconfigurations (`kl_loss_coef != 0`, `use_kl_loss` off); reward-degeneracy metrics on both arms; the β decomposition of the flow gap. **P1(a):** `filter_degenerate` sets g̃ = 0 for groups whose rewards are all equal, with an on-policy guard and new logs. **Infra:** `drop_degenerate` removes those groups before `old_log_prob` / `ref` / `update_actor` (update unchanged, about 30 % of the rows remain) and logs `gflowrl_drop/*`; `trainer.dump_images=false` skips the per-rollout PNGs. | **eval dump:**写出 `uid`、`num_turns`,以及(可选)response mask、token id 和两侧的 log 概率,供离线轨迹分析。**GFlowRL:**`compute_gflowrl_flow_gap()`,不带梯度的一半(Eq. 4 组内 log Z、Eq. 6 flow gap、Eq. 7 截断),结果写进 `advantages`;对两种不报错的错误配置(`kl_loss_coef != 0`、`use_kl_loss` 关闭)直接抛错的守卫;两臂都记奖励退化指标;flow gap 的 β 分解。**P1(a):**`filter_degenerate` 把组内奖励全同的组的 g̃ 置 0,带 on-policy 守卫和新日志。**Infra:**`drop_degenerate` 在 `old_log_prob` / `ref` / `update_actor` 之前把这些组删掉(更新不变,只剩约 30% 的行),并记 `gflowrl_drop/*`;`trainer.dump_images=false` 时不存每条 rollout 的 PNG。 | `04_gflowrl_implementation/spacetools_rl_modified/verl/trainer/ppo/ray_trainer.py`; patches 0008, 0010, 0011, 0012, 0015, 0019. P1(a): `07_gflowrl_improvement/p1_prep/patched/ray_trainer.py`, `07_gflowrl_improvement/p1_prep/p1a_ray_trainer.diff`. Infra: `07_gflowrl_improvement/infra_prep/patched/ray_trainer.py`, `07_gflowrl_improvement/infra_prep/infra_drop_only.diff` |
| `verl/workers/config/actor.py`, `verl/trainer/config/actor/actor.yaml` | Declare `policy_loss.gflowrl` so Hydra and the config dataclass accept the GFlowRL keys (without it the run fails inside a Ray worker). | 声明 `policy_loss.gflowrl`,Hydra 和配置 dataclass 才接受 GFlowRL 的键(不加会在 Ray worker 里报错)。 | `04_gflowrl_implementation/spacetools_rl_modified/verl/workers/config/actor.py`, `04_gflowrl_implementation/spacetools_rl_modified/verl/trainer/config/actor/actor.yaml`; patch 0013 |
| `verl/workers/fsdp_workers.py` | **Infra:** honours `ref.fsdp_config.param_offload=False`, so the reference model can stay on GPU (upstream forces CPU offload regardless of the flag). Default unchanged. | **Infra:**让 `ref.fsdp_config.param_offload=False` 真正生效,参考模型可以留在 GPU 上(上游不管这个开关,强制放 CPU)。默认行为不变。 | `07_gflowrl_improvement/infra_prep/patched/fsdp_workers.py` |
| `verl/experimental/agent_loop/tool_agent_loop.py` | **Infra:** with `TOOL_TIMING_DIR` set, logs one line per tool call (latency, thread-pool wait, remote time, calls in flight), to find what rollout generation is waiting for. Off by default. | **Infra:**设了 `TOOL_TIMING_DIR` 后,每次工具调用记一行(总耗时、等线程的时间、远端耗时、未返回的调用数),用来查 rollout 生成在等什么。默认关闭。 | `07_gflowrl_improvement/infra_prep/patched/tool_agent_loop.py` |

All five Infra files as one diff against `c6fef78a`: `07_gflowrl_improvement/infra_prep/infra_all_vs_c6fef78a.diff`.
Infra 的五个文件相对 `c6fef78a` 的完整 diff:同上。

The 19 patches, in order (`04_gflowrl_implementation/patches_spacetools_rl/*.patch`, apply with `git am` on `54270e82`) / 19 个 patch,按顺序:

| No. | What it does | 作用 | File / 文件 |
|---|---|---|---|
| 0001 | Correct benchmark parquet paths; GPU budget for a 4-GPU node | 修正 benchmark 路径;GPU 预算改为 4 卡 | `run_eval.sh` |
| 0002 | `DATA_DIR` override | 新增 `DATA_DIR` 覆盖 | `run_eval.sh` |
| 0003 | Keep Molmo in fp32; stop Ray co-locating the two large tools | Molmo 保持 fp32;不让 Ray 把两个大工具放在同一张卡 | `run_eval.sh` |
| 0004 | Load conda's shell hook | 加载 conda 的 shell hook | `run_eval.sh` |
| 0005 | Give the tools three GPUs (DepthPro needs 12.5 GB per actor) | 工具用三张卡(DepthPro 每个 actor 要 12.5 GB) | `run_eval.sh` |
| 0006 | Set `PYTORCH_ALLOC_CONF` (removed again by 0007) | 设 `PYTORCH_ALLOC_CONF`(被 0007 撤掉) | `run_eval.sh` |
| 0007 | Drop `expandable_segments`: it breaks sglang's memory saver | 去掉 `expandable_segments`:它会让 sglang 的 memory saver 报错 | `run_eval.sh` |
| 0008 | Dump `uid` and `num_turns` | dump 里写出 `uid` 和 `num_turns` | `ray_trainer.py` |
| 0009 | Hold the policy in bf16 (eval only) | 策略模型用 bf16(仅 eval) | `run_eval.sh` |
| 0010 | Dump response mask, token ids and both log-prob arms | dump 里写出 response mask、token id 和两侧 log 概率 | `ray_trainer.py` |
| 0011 | GFlowRL policy loss (Eq. 4–8), C′ variant | GFlowRL policy loss(Eq. 4–8),C′ 变体 | `core_algos.py`, `ray_trainer.py` |
| 0012 | `run_rl_gflowrl.sh` and the guard for the two silent misconfigurations | `run_rl_gflowrl.sh` 与两种静默错误配置的守卫 | `run_rl_gflowrl.sh`, `ray_trainer.py` |
| 0013 | Declare `policy_loss.gflowrl` in the config | 在配置里声明 `policy_loss.gflowrl` | `actor.yaml`, `actor.py` |
| 0014 | Stop booking the same GPUs twice on a single node | 单机上不再重复预订同一批 GPU | `run_rl.sh` |
| 0015 | Reward-degeneracy metrics on both arms; two IS-weight metrics | 两臂都记奖励退化指标;两个 IS 权重指标 | `core_algos.py`, `ray_trainer.py` |
| 0016 | Scale tool actors to `TOOL_GPUS` | 工具 actor 数按 `TOOL_GPUS` 缩放 | `run_rl.sh` |
| 0017 | Start the Ray head from `VERL_DIR` | Ray head 从 `VERL_DIR` 启动 | `run_rl.sh` |
| 0018 | Stop `cleanup` from aborting itself under `set -e` | `cleanup` 不再被 `set -e` 中断 | `run_rl.sh` |
| 0019 | Log the β decomposition of the flow gap | 记录 flow gap 的 β 分解 | `ray_trainer.py` |

Long-form reasoning for patches 0001–0010: `01_official_checkpoint_eval/reports/changes_to_upstream.md`. 0001–0010 的详细理由见该文件。

### 2.2 SpaceTools-Toolshed (base `4f0512d`)

| Upstream file / 上游文件 | Change and purpose | 改动与作用 | In this repo / 本 repo 位置 |
|---|---|---|---|
| `toolshed/tools/vlm.py` | Honours the `dtype` argument (upstream hardcodes `torch_dtype="auto"` and ignores it) and casts float inputs to the model dtype. Behaviour under the config we run is unchanged; it makes a precision ablation a one-string change. | 让 `dtype` 参数生效(上游写死 `torch_dtype="auto"`,不看这个参数),并把浮点输入转成模型的 dtype。在我们跑的配置下行为不变;做精度对照时只需改一个字符串。 | `00_environment/upstream_patches/SpaceTools-Toolshed/0001-fix-tools-honour-the-vlm-dtype-argument-cast-inputs-.patch` |
| `toolshed/tools/graspgen_franka_panda.yml` | The three checkpoint paths made absolute; relative paths do not resolve from a Ray actor. | 三个 checkpoint 路径改成绝对路径;相对路径在 Ray actor 里解析不到。 | same patch / 同一个 patch |
| `toolshed/integration/verl.py` | **Infra:** stamps when a worker thread picks a tool call up and when it returns, so the agent loop can split its latency into thread-pool wait and remote time. | **Infra:**记录线程开始处理工具调用和调用返回的时刻,agent loop 才能把耗时拆成「等线程」和「远端」两段。 | `07_gflowrl_improvement/infra_prep/toolshed_tool_timing_vs_712e557.diff` (diff only; apply it to Toolshed `712e557` / 只放 diff,打在 Toolshed `712e557` 上) |

### 2.3 GraspGen (base `2dd8852`)

| Upstream file / 上游文件 | Change and purpose | 改动与作用 | In this repo / 本 repo 位置 |
|---|---|---|---|
| `pyproject.toml`, `requirements.txt` | Drop `pickle5`, which cannot compile on Python 3.11; the only import site already falls back to `pickle`. (The torch / numpy / spconv pins in the same patch are applied by upstream's own install script, not by us.) | 去掉 `pickle5`,它在 Python 3.11 上编不过;唯一的 import 处本来就会回退到 `pickle`。(同一 patch 里的 torch / numpy / spconv 版本是上游安装脚本自己改的,不是我们改的。) | `00_environment/upstream_patches/GraspGen/0001-build-drop-pickle5-which-cannot-compile-on-Python-3..patch` |
| `pointnet2_ops/pointnet2_ops/pointnet2_utils.py` | The hardcoded `TORCH_CUDA_ARCH_LIST` assignment becomes `setdefault`, so the build honours `8.0;8.6;8.9` (CUDA 12 no longer supports the hardcoded `sm_37`). Applied in place by a script. | 写死的 `TORCH_CUDA_ARCH_LIST` 赋值改成 `setdefault`,构建时才会用 `8.0;8.6;8.9`(CUDA 12 已不支持写死的 `sm_37`)。由脚本原地修改。 | `00_environment/eval_rl_env/graspgen_fixes/repair_graspgen_step6.sh` |

### 2.4 RoboRefer (base `d97a995`)

| Upstream file / 上游文件 | Change and purpose | 改动与作用 | In this repo / 本 repo 位置 |
|---|---|---|---|
| `env_setup.sh` | Skip the hardcoded Python 3.10 flash-attn wheel: pip refuses it on Python 3.11 and `set -e` then aborts before llava is installed. (The `pyproject.toml` change in the same patch, disabling `lighteval`, is applied by upstream's install script, not by us.) | 跳过写死的 Python 3.10 flash-attn wheel:Python 3.11 下 pip 拒装,`set -e` 随即在安装 llava 之前中止。(同一 patch 里 `pyproject.toml` 关掉 `lighteval` 是上游安装脚本自己改的,不是我们改的。) | `00_environment/upstream_patches/RoboRefer/0001-build-skip-the-hardcoded-py3.10-flash-attn-wheel.patch` |

### 2.5 SpaceTools-SFT (base `b7ebbf32`)

| Upstream file / 上游文件 | Change and purpose | 改动与作用 | In this repo / 本 repo 位置 |
|---|---|---|---|
| `scripts/spacetools/run_sft.sh` | Per-device batch and gradient accumulation derived from the visible GPU count, so the global batch stays 8 (the paper's value) on any machine; DeepSpeed ZeRO-3→ZeRO-2; `save_only_model: true`; `eval_steps` 5→500; `use_reentrant_gc: false`; parallel dataset download. | 每卡 batch 和梯度累积按可见 GPU 数推导,任何机器上全局 batch 都是 8(论文值);DeepSpeed ZeRO-3→ZeRO-2;`save_only_model: true`;`eval_steps` 5→500;`use_reentrant_gc: false`;数据集并行下载。 | `02_sft_training/run_sft.sh`, `02_sft_training/run_sft.sh.diff_vs_upstream` |

### 2.6 Ray 2.47.1 (installed package / 已安装的包)

| Upstream file / 上游文件 | Change and purpose | 改动与作用 | In this repo / 本 repo 位置 |
|---|---|---|---|
| `ray/_private/node.py` | The version check is called with `python_version_match_level="minor"`, so tool environments on Python 3.11.0 can join a 3.11.16 head. Without it their actors silently never join. Only applied in environments whose Python patch version differs from the head; the original is kept as `node.py.orig`. | 版本检查调用时加 `python_version_match_level="minor"`,Python 3.11.0 的工具环境才能加入 3.11.16 的 head;不加的话这些环境的 actor 会静默缺席。只对 Python 补丁版本与 head 不同的环境打,原文件备份为 `node.py.orig`。 | `00_environment/eval_rl_env/POSTRESTORE.sh` (step 1) |

### 2.7 Official checkpoint `siyich/spacetools-ckpt` (`f953b1a1`)

| Upstream file / 上游文件 | Change and purpose | 改动与作用 | In this repo / 本 repo 位置 |
|---|---|---|---|
| `config.json` | Remove `text_config` (it makes the model type resolve to the text sub-model, and sglang refuses to load it); set `tie_word_embeddings = True`. | 删掉 `text_config`(它让模型类型被解析成文本子模型,sglang 拒绝加载);设 `tie_word_embeddings = True`。 | `00_environment/official_ckpt_fix/fix_checkpoint.py`; originals in `00_environment/official_ckpt_fix/original_config/config.json.orig` |
| `preprocessor_config.json` | Replaced with the base model's (`Qwen/Qwen2.5-VL-3B-Instruct`). | 换成基础模型(`Qwen/Qwen2.5-VL-3B-Instruct`)的版本。 | `00_environment/official_ckpt_fix/fix_checkpoint.py`; original in `00_environment/official_ckpt_fix/original_config/preprocessor_config.json.orig` |

---

## 3. Run-specific variants and unmodified upstream copies / 按次运行的变体与未改动的上游副本

These are copies of upstream files saved exactly as they were run, so that each result can be traced to the script that produced it.

这些是上游文件按实际运行时的样子保存的副本,用来把每个结果对应到产生它的脚本。

| File | What it is | 说明 |
|---|---|---|
| `03_sft_eval/config/run_eval.sh.asrun` | Upstream `run_eval.sh`, **unmodified**, as run for the SFT eval | 上游 `run_eval.sh`,**未改动**,SFT eval 时实际运行的版本 |
| `06_gflowrl_eval/scripts/run_eval.sh.asrun` | Upstream `run_eval.sh` + three changes, as run for the GFlowRL eval: corrected benchmark paths, `vlm` `num_gpus` 0.6→1.0 (Molmo alone on a card, no silent tool OOM), `gpu_memory_utilization` 0.5→0.545 (fixes the KV pool size) | 上游 `run_eval.sh` 加三处改动,GFlowRL eval 实际运行的版本:benchmark 路径修正;`vlm` 的 `num_gpus` 0.6→1.0(Molmo 独占一张卡,不再出现静默的工具 OOM);`gpu_memory_utilization` 0.5→0.545(固定 KV 池大小) |
| `06_gflowrl_eval/scripts/run_eval.sh.diff` | The diff between the two files above | 上面两个文件之间的 diff |
| `07_gflowrl_improvement/p0_same_machine_reeval/scripts/run_eval.as-run.sh` | Same as the GFlowRL-eval version, with `gpu_memory_utilization` 0.511 (same KV pool on a 49140 MiB card) | 与 GFlowRL eval 的版本相同,`gpu_memory_utilization` 为 0.511(49140 MiB 的卡上对应同样的 KV 池) |
| `01_official_checkpoint_eval/tools/p6/variants/run_eval_gmu025.sh` | Our `run_eval.sh` + one line: `gpu_memory_utilization` 0.5→0.25 (reproduces a 40 GB card's KV pool on an 80 GB card) | 我们的 `run_eval.sh` 改一行:`gpu_memory_utilization` 0.5→0.25(在 80 GB 的卡上复现 40 GB 卡的 KV 池) |
| `01_official_checkpoint_eval/tools/p6/variants/run_eval_fp32_gmu025.sh` | Our `run_eval.sh` + two lines: no `model_dtype=bf16`, `gpu_memory_utilization` 0.25 (fp32 control) | 我们的 `run_eval.sh` 改两行:去掉 `model_dtype=bf16`,`gpu_memory_utilization` 0.25(fp32 对照) |
| `01_official_checkpoint_eval/tools/p6/variants/run_eval_passk.sh` | Our `run_eval.sh` + `gpu_memory_utilization` 0.25 and sampling (`do_sample`, `n=5`, `temperature=1.0`, `top_p=1.0`): 5 samples per question | 我们的 `run_eval.sh` 加 `gpu_memory_utilization` 0.25 和采样参数:每题采 5 次 |
| `01_official_checkpoint_eval/tools/p6/variants/run_eval_passk2.sh` | Our `run_eval.sh` + sampling as above, `calculate_log_probs`, token diagnostics in the dump, `max_num_seqs` 256→64: the second 5-sample set, with log-probs | 我们的 `run_eval.sh` 加同样的采样参数、`calculate_log_probs`、dump 里的 token 诊断,`max_num_seqs` 256→64:带 log 概率的第二组 5 次采样 |
| `01_official_checkpoint_eval/tools/p6/variants/toolshed_v1_holder.py` | The v1 tool-start block of `run_eval.sh`, copied into a long-lived process, for experiments that need the tools but no policy | `run_eval.sh` 里启动 v1 工具的那一段,放进一个常驻进程;给只需要工具、不需要策略模型的实验用 |
| `05_gflowrl_training/config/run_rl.sh.asrun`, `05_gflowrl_training/config/run_rl_gflowrl.sh.asrun` | The launchers as run for the 85-step training; byte-identical to the files under `04_gflowrl_implementation/spacetools_rl_modified/` | 85 步训练实际运行的启动脚本;与 `04_gflowrl_implementation/spacetools_rl_modified/` 下的文件逐字节相同 |
| `02_sft_training/tool_schema_check/toolshed_v1_config.repo.yaml` | Upstream `SpaceTools-RL/examples/toolshed/toolshed_v1_config.yaml`, **unmodified**, kept for the schema comparison | 上游的 `toolshed_v1_config.yaml`,**未改动**,留作 schema 比对 |
| `00_environment/official_ckpt_fix/original_config/config.json.orig`, `00_environment/official_ckpt_fix/original_config/preprocessor_config.json.orig` | The two config files of the official checkpoint, **unmodified** | 官方 checkpoint 的两个配置文件,**未改动** |
| `07_gflowrl_improvement/p0_same_machine_reeval/scripts/p4-official.config.orig.json`, `07_gflowrl_improvement/p0_same_machine_reeval/scripts/p4-official.config.patched.json` | `config.json` of the official checkpoint before and after `fix_checkpoint.py`, as used in P0 | 官方 checkpoint 的 `config.json` 在 `fix_checkpoint.py` 前后的样子,P0 实际使用 |

---

## 4. Files we created / 自己创建的文件

Everything below was written for this project. Where a script follows an upstream script, the row says so.

以下都是为本项目写的。脚本若参照了上游脚本,该行会注明。

### 4.1 Top level, `docs/`, `tools/`

| File | What it is | 说明 |
|---|---|---|
| `README.md` | Overview, results, layout, reproduction steps (English and Chinese) | 总览、结果、目录、复现步骤(中英) |
| `LICENSE` | Apache-2.0 | Apache-2.0 许可证 |
| `.gitignore` | Ignores unpacked `*.jsonl` dumps and caches | 忽略解压出的 `*.jsonl` 和缓存 |
| `docs/README.md` | Index of `docs/` | `docs/` 的索引 |
| `docs/FILE_INDEX.md` | This file | 本文件 |
| `docs/full_report_v1.md` | The full report (Chinese): environment, SFT, eval gate, C′ derivation and implementation, training, eval, per-sample comparison | 完整报告:环境、SFT、eval 闸门、C′ 推导与实现、训练、评测、逐样本对比 |
| `docs/design_doc_gflowrl_optimization.md` | The current optimization plan (Chinese): P1–P3 and Infra | 当前优化计划:P1–P3 与 Infra |
| `docs/paper_notes.md` | Notes on the SpaceTools and GFlowRL papers and the SpaceTools code | 两篇论文与 SpaceTools 代码的笔记 |
| `docs/sft_training_notes.md` | Where each SFT config value comes from; batch, memory and ZeRO trade-offs | SFT 每个配置值的来由;batch、显存、ZeRO 的取舍 |
| `docs/why_not_train_per_benchmark.md` | Why RL is not split per benchmark | 为什么 RL 不按 benchmark 分开训 |
| `docs/PATH_MAP.md` | Original working paths quoted in the reports → paths in this repo | 报告里引用的旧路径 → 本 repo 的路径 |
| `tools/README.md` | Index of `tools/` | `tools/` 的索引 |
| `tools/parse_dump.py` | verl validation dump → one structured record per sample (tool chain, calls per turn, correctness); `--strict` is the health gate | verl 的 validation dump → 每个样本一条结构化记录(工具链、每轮调用、对错);`--strict` 用作健康闸门 |
| `tools/unpack_dumps.sh` | Unpacks every `*.jsonl.gz` next to itself; run once after cloning | 把所有 `*.jsonl.gz` 原地解压;clone 后跑一次 |
| `tools/gputrace.sh` | 1 Hz GPU memory trace | 每秒一次的 GPU 显存记录 |

### 4.2 `00_environment/`

| File | What it is | 说明 |
|---|---|---|
| `00_environment/README.md` | Index of the folder | 目录索引 |
| `00_environment/env.sh` | Shared paths, caches and conda root for every shell | 每个 shell 共用的路径、缓存和 conda 根目录 |
| `00_environment/tool-constraints.txt` | pip constraints for the tool environments (no CUDA 13 torch) | 工具环境的 pip 约束(不装 CUDA 13 的 torch) |
| `00_environment/verify_env_imports.sh` | Imports each tool's module in its own environment; replaces upstream `test_envs.sh`, which does not cover the tool environments | 在各自环境里 import 每个工具的模块;替代上游的 `test_envs.sh`(它不覆盖工具环境) |
| `00_environment/rebuild_pointnet2_sm80.sh` | Rebuilds `pointnet2_ops` for sm_80 / 86 / 89 after moving from A6000 to A100 | 从 A6000 换到 A100 后,为 sm_80 / 86 / 89 重编 `pointnet2_ops` |
| `00_environment/tool_chain_smoke_test.py` | Calls every v1 tool once in dependency order, with no language model, so a failure is a tool problem | 不用语言模型,按依赖顺序把每个 v1 工具调一遍;失败就一定是工具的问题 |
| `00_environment/env_freeze/*.txt` | `pip freeze` of the five environments | 五个环境的 `pip freeze` |
| `00_environment/eval_rl_env/BUILD_GUIDE.md` | Main guide: architecture, versions per environment, build steps, the 20 upstream issues, acceptance criteria | 主文档:架构、各环境版本、搭建步骤、20 个上游问题、验收标准 |
| `00_environment/eval_rl_env/build_eval_envs.sh` | Builds the five environments; follows upstream `setup_envs.sh` (SpaceTools) and `install_tools/setup_tool_env.sh` (Toolshed), each deviation marked | 搭五个环境;主干照上游的 `setup_envs.sh` 和 `install_tools/setup_tool_env.sh`,每处偏离都有标注 |
| `00_environment/eval_rl_env/graspgen_fixes/repair_graspgen_step*.sh` | Six fixes for the GraspGen environment, applied in order: non-interactive install, `pickle5`, CUDA toolkit version, `pointnet2_ops` toolchain, compiler version, GPU architecture list | GraspGen 环境的六轮修复,按顺序跑:非交互安装、`pickle5`、CUDA toolkit 版本、`pointnet2_ops` 工具链、编译器版本、GPU 架构列表 |
| `00_environment/eval_rl_env/fix_vlm_transformers.sh` | Installs `transformers==4.53.2` in the vlm environment, as upstream pins it for Molmo | 在 vlm 环境装 `transformers==4.53.2`(上游为 Molmo 钉的版本) |
| `00_environment/eval_rl_env/fix_weights_roborefer.sh` | Fixes three "built but empty" items: GraspGen weights that were git-lfs pointers, the RoboRefer weights folder, and the default weight paths of the depth and grasp tools | 修三处「看起来建好了其实是空的」:GraspGen 权重是 git-lfs 指针、RoboRefer 权重目录是空的、depth 与 grasp 工具的默认权重路径 |
| `00_environment/eval_rl_env/verify_envs.sh` | Acceptance of the five environments (imports, `.so` GPU architecture scan) and the version freeze | 五个环境的验收(import、`.so` 的 GPU 架构扫描)与版本冻结 |
| `00_environment/eval_rl_env/smoke_tools.sh` | Smoke test of the seven tools, one at a time on one GPU | 七个工具逐个冒烟,单卡 |
| `00_environment/eval_rl_env/VERIFY.sh` | Acceptance after restoring the packaged environment; judged on real state, not exit codes | 还原环境包之后的验收;看实际状态,不看退出码 |
| `00_environment/eval_rl_env/POSTRESTORE.sh` | Three fixes to apply after restoring the environment package: the Ray version check (§2.6), `CUDA_HOME` for the roborefer environment, runtime folders | 还原环境包后要补的三件事:Ray 版本检查(§2.6)、roborefer 环境的 `CUDA_HOME`、运行时目录 |
| `00_environment/eval_rl_env/env_package_revision_20260912.md` | What the first environment package missed and how it was fixed | 第一版环境包漏了什么、怎么补的 |
| `00_environment/official_ckpt_fix/fix_checkpoint.py` | Applies the config fixes of §2.7 to the official checkpoint | 对官方 checkpoint 做 §2.7 的配置修复 |
| `00_environment/sft_env/setup_sft_env.sh` | Builds the SFT environment from scratch; follows upstream `setup_envs.sh` and `docs/SETUP.md` | 从零搭 SFT 环境;参照上游的 `setup_envs.sh` 与 `docs/SETUP.md` |
| `00_environment/sft_env/check_gpu_arch.sh` | Build-time gate: every compiled `.so` must contain code for the target GPU architecture | 构建期闸门:每个编出来的 `.so` 都必须带目标 GPU 架构的代码 |
| `00_environment/sft_env/sft_env_build_report_2xA6000.md` | Build log and a 30-step probe on 2× A6000 | 2× A6000 上的搭建记录与 30 步探测 |
| `00_environment/sft_env/env_package/README.md`, `00_environment/sft_env/env_package/RESTORE.sh`, `00_environment/sft_env/env_package/VERIFY.sh`, `00_environment/sft_env/env_package/SHA256SUMS` | Restore and verify the packaged SFT environment; checksums of the package | 还原并验收打包好的 SFT 环境;包的校验和 |

### 4.3 `01_official_checkpoint_eval/`

| File | What it is | 说明 |
|---|---|---|
| `01_official_checkpoint_eval/README.md` | Index of the folder | 目录索引 |
| `01_official_checkpoint_eval/reports/changes_to_upstream.md`, `01_official_checkpoint_eval/reports/changes_to_upstream.zh.md` | Every change made to an upstream repo during the reproduction, with the reasoning | 复现阶段对上游的每处改动与理由 |
| `01_official_checkpoint_eval/reports/provenance.txt` | Weight commits, environment matrix, list of deviations | 权重版本、环境矩阵、偏离清单 |
| `01_official_checkpoint_eval/reports/p2_smoke_test_results.md` | P2: smoke test on 4× A6000 | P2:4× A6000 上的冒烟测试 |
| `01_official_checkpoint_eval/reports/p3_trajectory_capture.md`, `01_official_checkpoint_eval/reports/p3_trajectory_capture.zh.md` | P3: how trajectories are captured from verl dumps | P3:怎样从 verl 的 dump 采集轨迹 |
| `01_official_checkpoint_eval/reports/p4_full_eval_results.md`, `01_official_checkpoint_eval/reports/p4_full_eval_results.zh.md` | P4: full eval, 9 benchmarks, 2121 samples | P4:全量评测,9 个 benchmark、2121 个样本 |
| `01_official_checkpoint_eval/reports/p5_accuracy_report.md` | P5: accuracy against the paper and where each gap comes from | P5:与论文的 accuracy 对比及差距来源 |
| `01_official_checkpoint_eval/reports/p6_error_attribution_report.md`, `01_official_checkpoint_eval/reports/p6_error_attribution_full.md` | P6: error attribution and headroom (summary and full version) | P6:错题归因与 headroom(摘要版与完整版) |
| `01_official_checkpoint_eval/reports/p6_gpu_experiments.md` | P6 GPU experiments: fp32, KV pool size, pass@k, tool swaps | P6 的 GPU 实验:fp32、KV 池大小、pass@k、换工具 |
| `01_official_checkpoint_eval/tools/p4_run.sh` | Runs one benchmark with a GPU trace alongside; refuses to overwrite an existing dump | 跑一个 benchmark 并同时记 GPU;拒绝覆盖已有 dump |
| `01_official_checkpoint_eval/tools/p4_check.sh` | Gate after a run: a non-zero exit means the benchmark has no valid result | 跑完后的闸门:非零退出表示该 benchmark 没有有效结果 |
| `01_official_checkpoint_eval/tools/make_subset.py` | Builds a truncated copy of the benchmarks for smoke runs (`run_eval.sh` has no sample-count option) | 生成截短的 benchmark 副本用于冒烟(`run_eval.sh` 没有样本数选项) |
| `01_official_checkpoint_eval/tools/p5/metrics.py` | BOP-ASK pose: recomputes the score with the repo's own reward code and compares candidate metrics | BOP-ASK pose:用仓库自己的打分代码重算分数,并比较几种候选指标 |
| `01_official_checkpoint_eval/tools/p5/perm.py` | BOP-ASK pose: corner distance in the given order against the best one-to-one matching, i.e. how much error comes from corner order | BOP-ASK pose:按给出顺序的角点距离对比最优一一匹配,即顺序造成了多少误差 |
| `01_official_checkpoint_eval/tools/p5/stats.py` | Trajectory statistics per benchmark: turns, tool calls per sample, tool errors, unused variables | 每个 benchmark 的轨迹统计:轮数、每样本工具调用数、工具报错、未使用的变量 |
| `01_official_checkpoint_eval/tools/p6/p6_inventory.py` | Error inventory per benchmark: wrong answers split into pipeline damage (tool error, OOM, truncation, turn cap, no answer) and clean errors | 每个 benchmark 的错误清单:错题分成流程损伤(工具报错、OOM、截断、顶轮、无答案)和干净的错 |
| `01_official_checkpoint_eval/tools/p6/p6_split.py` | Criteria A and B: does the model always answer the point with the smaller measured depth; does it pass the tool's point through | 判据 A、B:模型是否总是选测得深度更小的点;是否原样透传工具给的点 |
| `01_official_checkpoint_eval/tools/p6/p6_split2.py` | Pass-through test on RoboSpatial Vacant | RoboSpatial Vacant 上的透传检验 |
| `01_official_checkpoint_eval/tools/p6/p6_split3.py` | RoboSpatial VQA question types and the Vacant analysis | RoboSpatial VQA 的题型与 Vacant 分析 |
| `01_official_checkpoint_eval/tools/p6/p6_relations.py` | Automatic criterion for relation questions, replacing manual review | 关系题的自动判据,替代人工审阅 |
| `01_official_checkpoint_eval/tools/p6/p6_continuous.py` | Attribution for the two continuous-metric benchmarks (`boppose`, `bopgrasp`) | 两个连续指标 benchmark(`boppose`、`bopgrasp`)的归因 |
| `01_official_checkpoint_eval/tools/p6/p6_consistency.py` | From the 5-sample dumps: all right, all wrong or split, and whether split samples made identical tool calls | 读 5 次采样的 dump:全对、全错还是分裂,以及分裂样本的工具调用是否相同 |
| `01_official_checkpoint_eval/tools/p6/make_cards.py` | Renders review cards (image, tool points, question, ground truth, answer) for the cases that need a human | 给需要人工看的错题渲染审阅卡(原图、工具给的点、题面、GT、答案) |
| `01_official_checkpoint_eval/tools/p6/verdicts.py` | The manual verdicts, each with its category and evidence | 人工归类的结论,每条带类别和依据 |
| `01_official_checkpoint_eval/tools/p6/gpu_pointing_swap.py` | Experiment A: swap the pointing tool and re-score, without the policy | 实验 A:换指点工具后重新打分,不需要策略模型 |
| `01_official_checkpoint_eval/tools/p6/gpu_depth_swap.py` | Experiment B: swap the depth tool and re-apply the model's rule, without the policy | 实验 B:换深度工具后重新套用模型的规则,不需要策略模型 |
| `01_official_checkpoint_eval/tools/p6/variants/README.md` | What each `run_eval.sh` variant changes and how to run it | 每个 `run_eval.sh` 变体改了什么、怎么跑 |
| `01_official_checkpoint_eval/tools/p6/variants/run_gmu025.sh`, `01_official_checkpoint_eval/tools/p6/variants/run_fp32.sh`, `01_official_checkpoint_eval/tools/p6/variants/run_passk.sh` | `p4_run.sh` with the eval script swapped for the matching variant in §3 | `p4_run.sh` 把 eval 脚本换成 §3 里对应的变体 |
| `01_official_checkpoint_eval/p4/dumps/run*/*/0.jsonl.gz` | Raw verl validation dumps of the official checkpoint: run1 covers all nine benchmarks, run2–run4 repeat some | 官方 checkpoint 的 verl 原始 dump:run1 是全部九个 benchmark,run2–run4 是部分重复 |
| `01_official_checkpoint_eval/p4/dumps/run2/robospatial/eval.log.gz`, `01_official_checkpoint_eval/p4/logs/*.gz` | Eval logs and 1 Hz GPU traces of those runs | 这些运行的 eval 日志与每秒 GPU 记录 |
| `01_official_checkpoint_eval/p4/parsed/*.jsonl` | Parsed records of run1, one file per benchmark (output of `tools/parse_dump.py`) | run1 的解析记录,每个 benchmark 一个文件(`tools/parse_dump.py` 的输出) |
| `01_official_checkpoint_eval/p6/fp32/*`, `01_official_checkpoint_eval/p6/gmu025/*` | fp32 control runs and `gpu_memory_utilization=0.25` runs: a README each, dumps, logs, GPU traces | fp32 对照与 `gpu_memory_utilization=0.25` 的运行:各有 README、dump、日志、GPU 记录 |
| `01_official_checkpoint_eval/p6/passk/*`, `01_official_checkpoint_eval/p6/passk2/*` | The two 5-sample sets (dumps, logs, GPU traces; README for the second) | 两组 5 次采样(dump、日志、GPU 记录;第二组有 README) |
| `01_official_checkpoint_eval/p6/probes/*.jsonl.gz` | Probe points taken from the P4 dumps, input to the tool-swap experiments | 从 P4 dump 取出的探测点,换工具实验的输入 |
| `01_official_checkpoint_eval/p6/swap/*.jsonl.gz` | Results of the tool-swap experiments | 换工具实验的结果 |
| `01_official_checkpoint_eval/p6/manual/*` | Manual review: README, pending cases, verdicts, and the review cards (`cards/*.jpg`) | 人工归类:README、待审清单、结论、审阅卡(`cards/*.jpg`) |

### 4.4 `02_sft_training/`

| File | What it is | 说明 |
|---|---|---|
| `02_sft_training/README.md` | Index of the folder and how to run | 目录索引与运行方法 |
| `02_sft_training/FINALIZE.sh` | After training: verifies the run, collects the evidence, prunes and uploads to Hugging Face | 训练后的收尾:验收、收集证据、清理并上传 HF |
| `02_sft_training/smoke_check.py` | Format smoke test of the checkpoint against the base model: did SFT take effect (not a capability score) | checkpoint 的格式冒烟并对照基础模型:SFT 是否生效(不是能力分数) |
| `02_sft_training/sft_training_report_4xA6000.md` | Execution report: memory headroom, speed, findings | 执行报告:显存余量、速度、发现 |
| `02_sft_training/tool_schema_check/DIFF_RESULT.md` | Result: the tool schema in the repo's static YAML differs from what the model saw at eval in two places | 结论:仓库里静态 YAML 的工具 schema 与 eval 时模型实际看到的有两处不一致 |
| `02_sft_training/tool_schema_check/diff_tool_text.py` | Compares the schema text two YAMLs put into the SFT system prompt | 比较两份 YAML 最终拼进 SFT system prompt 的 schema 文本 |
| `02_sft_training/tool_schema_check/p4_tools_block.txt` | The tool schema block the model actually saw in the P4 eval, taken from a dump | P4 eval 时模型实际看到的工具 schema,取自 dump |
| `02_sft_training/tool_schema_check/toolshed_v1_config.p4.yaml` | The tool config exported from the live tool server; the one to use for SFT | 从运行中的工具服务导出的配置;SFT 应该用这一份 |

### 4.5 `03_sft_eval/`

| File | What it is | 说明 |
|---|---|---|
| `03_sft_eval/README.md` | Index of the folder | 目录索引 |
| `03_sft_eval/sft_eval_report.md` | Report: results, failure modes, gate decision | 报告:结果、失败模式、闸门结论 |
| `03_sft_eval/sft_eval_results.md` | Raw results and deviations | 原始结果与偏离 |
| `03_sft_eval/ARTIFACTS.md` | Field-by-field description of the rollouts and logs | rollout 与日志的逐字段说明 |
| `03_sft_eval/scripts/audit2.sh`, `03_sft_eval/scripts/audit3.sh` | Audit which tools were really called and what they returned | 核对实际调了哪些工具、返回了什么 |
| `03_sft_eval/scripts/health.sh` | Health indicators: missing answer, turn cap, truncated tool responses, malformed calls | 健康指标:无答案、顶轮、工具返回被截断、调用格式错误 |
| `03_sft_eval/scripts/compare.sh`, `03_sft_eval/scripts/nondet.sh`, `03_sft_eval/scripts/stable.sh` | The two RoboSpatial runs compared; divergence split into policy side and tool side; always-right, always-wrong and flipping samples | 两次 RoboSpatial 运行的对比;分歧拆成策略侧和工具侧;恒对、恒错、翻转样本 |
| `03_sft_eval/scripts/split.sh`, `03_sft_eval/scripts/hull.sh`, `03_sft_eval/scripts/vacant.sh` | RoboSpatial split into VQA and Vacant; Vacant re-scored with the real convex-hull criterion; how the query to RoboRefer is written | RoboSpatial 拆成 VQA 与 Vacant;用真实的凸包判据重算 Vacant;给 RoboRefer 的查询怎么写 |
| `03_sft_eval/scripts/material.sh` | Material for the report: confusion matrix, turn and call distributions, sample trajectories | 报告素材:混淆矩阵、轮数与调用数分布、样例轨迹 |
| `03_sft_eval/scripts/gputrace.sh` | 1 Hz per-card and per-process GPU trace | 每秒一次、按卡和按进程的 GPU 记录 |
| `03_sft_eval/analysis/*.log` | Outputs of the scripts above | 上面这些脚本的输出 |
| `03_sft_eval/config/toolshed_config.yaml` | The tool config generated at run time | 运行时生成的工具配置 |
| `03_sft_eval/config/WEIGHTS_PINS.txt` | Pinned revisions of every weight and dataset | 所有权重与数据集的固定版本 |
| `03_sft_eval/config/pkg_MANIFEST.txt`, `03_sft_eval/config/SHA256SUMS.remote` | Manifest and checksums of the environment package used | 所用环境包的清单与校验和 |
| `03_sft_eval/gpu/*.log` | GPU trace and its summary | GPU 记录及汇总 |
| `03_sft_eval/verify/*.log` | Environment acceptance logs: `VERIFY.sh` before and after the patch, one smoke log per tool, the tool-chain test, the Ray patch | 环境验收日志:补丁前后的 `VERIFY.sh`、每个工具的冒烟、工具链测试、Ray 补丁 |
| `03_sft_eval/logs/*.log.gz` | Eval logs | eval 日志 |
| `03_sft_eval/rollouts/*.jsonl.gz` | Full trajectories: RoboSpatial twice and the three RefSpatial splits | 完整轨迹:RoboSpatial 两次、RefSpatial 三项 |

### 4.6 `04_gflowrl_implementation/`

The modified upstream files and the patches are in §2.1. 改动过的上游文件和 patch 见 §2.1。

| File | What it is | 说明 |
|---|---|---|
| `04_gflowrl_implementation/README.md` | The objective as implemented, where the code is, the self-checks | 实现的目标函数、代码位置、自检 |
| `04_gflowrl_implementation/IMPLEMENTATION_NOTES.md` | Walkthrough of everything the Step 4 training uses | Step 4 训练用到的全部东西的讲解 |
| `04_gflowrl_implementation/checks/run_checks.sh` | Runs the six self-checks below against a patched SpaceTools-RL checkout | 对打好补丁的 SpaceTools-RL 跑下面六个自检 |
| `04_gflowrl_implementation/checks/p7_fixedpoint_check.py` | Loss and gradient are exactly 0 at π ∝ π_ref·exp(βr) for C′; runs the real source, not a copy | C′ 在 π ∝ π_ref·exp(βr) 处 loss 和梯度恰为 0;执行的是真实源码,不是抄写的副本 |
| `04_gflowrl_implementation/checks/p7_degenerate_check.py` | C′ still produces gradient on reward-degenerate groups, where GRPO gives exactly zero | C′ 在奖励退化组上仍有梯度,而 GRPO 恰为零 |
| `04_gflowrl_implementation/checks/p7_config_check.py` | Every config key the code reads is passed by the wrapper, and nothing else differs between the arms | 代码读的每个配置键都由封装脚本传入,两臂之间没有别的差异 |
| `04_gflowrl_implementation/checks/p7_guard_check.py` | The guard raises on both silent misconfigurations | 两种静默错误配置都会触发守卫 |
| `04_gflowrl_implementation/checks/p7_gpusplit_check.py` | The tool / training GPU split on one and two nodes | 单机与双机下工具 / 训练的 GPU 划分 |
| `04_gflowrl_implementation/checks/p7_onpolicy_check.py` | The update is on-policy (importance weight ≡ 1) for the planned GPU counts | 计划的 GPU 数下更新是 on-policy(重要性权重恒为 1) |
| `04_gflowrl_implementation/checks/p7_fixedpoint.py` | Synthetic check of Eq. 4–8 on an enumerable space, independent of the real code | 在可穷举的空间上对 Eq. 4–8 做合成检验,与真实代码无关 |
| `04_gflowrl_implementation/checks/p7_estimators.py`, `04_gflowrl_implementation/checks/p7_ib_real.py` | Three ways to take the batch constant, on synthetic drift and on measured log-prob gaps | 批内常数的三种取法,分别在合成漂移和实测 log 概率差上比较 |
| `04_gflowrl_implementation/checks/p7_zt_offline.py` | Offline decomposition of Z_t on the 5-sample groups | 在 5 次采样的组上离线拆解 Z_t |
| `04_gflowrl_implementation/checks/p7_logprob_gap.py` | Distribution of the log-prob gap between the rollout engine and the trainer | rollout 引擎与训练侧之间 log 概率差的分布 |
| `04_gflowrl_implementation/checks/p7_cprime_gate.py`, `04_gflowrl_implementation/checks/p7_fix_cost.py` | The three length-normalization configurations compared on real data; the fixed-point cost of the candidate fix | 长度归一化的三种配置在真实数据上的比较;候选修法的不动点代价 |
| `04_gflowrl_implementation/checks/p7_accept_blinkdepth.py` | Acceptance criterion for moving to a new GPU machine | 换 GPU 机器时的迁移验收判据 |
| `04_gflowrl_implementation/checks/p7_passk2_run.sh` | Runs the second 5-sample set, one benchmark per call | 跑第二组 5 次采样,一次一个 benchmark |
| `04_gflowrl_implementation/design_notes/*.md` | Nine notes in the order the decisions were made: route, prior art, batch constant, offline Z_t and fixed point, response mask and length, GPU resampling, the C′ gate, the training plan, the implementation and its check | 九篇设计记录,按决策顺序:路线、现成实现、批内常数、Z_t 离线拆解与不动点、response mask 与长度、GPU 重采样、C′ 闸门、训练计划、实现与自检 |
| `04_gflowrl_implementation/migration_acceptance/README.md`, `04_gflowrl_implementation/migration_acceptance/run12/blinkdepth/*.gz` | The blinkdepth run used to accept the new machine before training: note, dump, log, GPU trace | 训练前验收新机器用的 blinkdepth 运行:说明、dump、日志、GPU 记录 |

### 4.7 `05_gflowrl_training/`

| File | What it is | 说明 |
|---|---|---|
| `05_gflowrl_training/README.md` | Index of the folder | 目录索引 |
| `05_gflowrl_training/gflowrl_training_report_8xA40.md` | Training report: machine, hyper-parameters, metrics, incidents | 训练报告:机器、超参、指标、事故 |
| `05_gflowrl_training/training_metrics_85steps.csv` | One row per step: time, degenerate-group share, clip saturation, reward and drift terms, gradient norm, score, memory | 每步一行:耗时、退化组占比、截断饱和率、奖励项与漂移项、梯度范数、分数、显存 |
| `05_gflowrl_training/config/full_train.sh.asrun` | The exact launch script of the run | 这次训练实际的启动脚本 |
| `05_gflowrl_training/config/rl_toolshed_config.yaml` | The tool config generated at run time | 运行时生成的工具配置 |
| `05_gflowrl_training/config/ENVIRONMENT.txt` | Snapshot of the machine, driver and environments | 机器、驱动、环境的快照 |
| `05_gflowrl_training/scripts/extract_metrics_table.py` | Parses the training log into the metrics table and CSV | 把训练日志解析成指标表和 CSV |
| `05_gflowrl_training/scripts/ckpt_janitor.py` | Keeps disk use bounded during training while snapshotting milestone checkpoints | 训练中控制磁盘占用,同时保留里程碑 checkpoint |
| `05_gflowrl_training/scripts/merge_and_upload_ckpts.sh` | Merges the checkpoint shards to Hugging Face format and uploads them | 把 checkpoint 分片合并成 HF 格式并上传 |
| `05_gflowrl_training/scripts/make_full_train_script.py` | Derives the full-run script from the 3-step smoke script | 从 3 步冒烟脚本生成全量训练脚本 |
| `05_gflowrl_training/scripts/debug_probes/nccl_allreduce_probe.py`, `05_gflowrl_training/scripts/debug_probes/nccl_env_sweep.sh` | Minimal 4-GPU all-reduce probe and the sweep over NCCL environment variables that located the hang | 最小的 4 卡 all-reduce 探针,以及定位挂死原因的 NCCL 环境变量逐项测试 |
| `05_gflowrl_training/scripts/debug_probes/exit_code_probe.sh`, `05_gflowrl_training/scripts/debug_probes/exit_code_probe_fixed.sh` | Minimal reproduction of the wrong exit code of `run_rl.sh`, before and after the fix (patch 0018) | `run_rl.sh` 退出码错误的最小复现,修复前后(patch 0018) |

### 4.8 `06_gflowrl_eval/`

| File | What it is | 说明 |
|---|---|---|
| `06_gflowrl_eval/README.md` | Index of the folder | 目录索引 |
| `06_gflowrl_eval/gflowrl_eval_report.md` | Eval report: protocol, the silent tool OOM and how it was caught, health gates, results | 评测报告:协议、静默的工具 OOM 及其发现过程、健康闸门、结果 |
| `06_gflowrl_eval/comparison_p4_p5_p6_vs_p7.md` | Per-sample comparison with the official checkpoint | 与官方 checkpoint 的逐样本对比 |
| `06_gflowrl_eval/scripts/EVAL_FROM_SCRATCH.sh` | From a bare GPU machine to the nine scores; every step idempotent | 从一台裸 GPU 机器到九个分数;每一步幂等 |
| `06_gflowrl_eval/scripts/run_eval.sh` | Driver of the first run (all nine benchmarks) | 第一次运行(九个 benchmark)的驱动脚本 |
| `06_gflowrl_eval/scripts/run_eval_fix2.sh` | Re-runs the four benchmarks hit by tool OOM, with `vlm` alone on a card | 重跑被工具 OOM 污染的四个 benchmark,`vlm` 独占一张卡 |
| `06_gflowrl_eval/scripts/run_rs2.sh` | The second, independent RoboSpatial run | 第二次独立的 RoboSpatial 运行 |
| `06_gflowrl_eval/analysis/compare_p4_p7.py`, `06_gflowrl_eval/analysis/compare_p4_p7.txt` | Per-sample pairing of GRPO and C′, and its output table | GRPO 与 C′ 的逐样本配对,以及输出的表 |
| `06_gflowrl_eval/analysis/significance_p4_p7.py` | McNemar exact test and sign tests per benchmark | 每个 benchmark 的 McNemar 精确检验与符号检验 |
| `06_gflowrl_eval/analysis/p6_criteria_p4_vs_p7.py` | The P6 criteria, question types and Vacant analysis on both checkpoints | 在两个 checkpoint 上跑 P6 判据、题型与 Vacant 分析 |
| `06_gflowrl_eval/analysis/robospatial_divergence.py` | What the diverging RoboSpatial samples look like | RoboSpatial 分歧样本的方向与类型 |
| `06_gflowrl_eval/config/ENVIRONMENT.txt`, `06_gflowrl_eval/config/WEIGHTS_PINS.txt` | Machine snapshot and pinned weight revisions | 机器快照与权重固定版本 |
| `06_gflowrl_eval/gates/*.txt` | Health gate per benchmark and run; all counts must be zero before a score is read | 每个 benchmark、每次运行的健康闸门;全部为零才读分数 |
| `06_gflowrl_eval/gpu/gpu_rs2.log` | GPU trace of the second RoboSpatial run | 第二次 RoboSpatial 运行的 GPU 记录 |
| `06_gflowrl_eval/dumps/runA-9bench/*/0.jsonl.gz` | Valid benchmarks of the first run | 第一次运行里有效的 benchmark |
| `06_gflowrl_eval/dumps/runB-rerun4/*/0.jsonl.gz` | The four re-run benchmarks, replacing the contaminated ones | 重跑的四个 benchmark,替换被污染的结果 |
| `06_gflowrl_eval/dumps/runC-robospatial2/*/0.jsonl.gz` | The second RoboSpatial run | 第二次 RoboSpatial 运行 |
| `06_gflowrl_eval/logs/*.gz`, `06_gflowrl_eval/logs/run*/*/eval.log.gz` | Driver logs and per-benchmark eval logs | 驱动日志与每个 benchmark 的 eval 日志 |
| `06_gflowrl_eval/parsed/*.jsonl`, `06_gflowrl_eval/parsed_runC/*.jsonl` | Parsed records, same schema as `01_official_checkpoint_eval/p4/parsed/` | 解析记录,格式与 `01_official_checkpoint_eval/p4/parsed/` 相同 |

### 4.9 `07_gflowrl_improvement/`

The patched upstream files (`p1_prep/patched/`, `infra_prep/patched/`) and the diffs are in §2. 打过补丁的上游文件及 diff 见 §2。

| File | What it is | 说明 |
|---|---|---|
| `07_gflowrl_improvement/README.md` | Index of the folder | 目录索引 |
| `07_gflowrl_improvement/p0_same_machine_reeval/p0_results.md` | P0 results: three checkpoints on one machine, and where the Vacant gap comes from | P0 结果:三个 checkpoint 同机重评,以及 Vacant 差距的来源 |
| `07_gflowrl_improvement/p0_same_machine_reeval/scripts/p0_go.sh`, `07_gflowrl_improvement/p0_same_machine_reeval/scripts/p0_eval.sh` | Set the machine up, then launch the runs (3 checkpoints × 3 runs, plus RefSpatial once) | 准备机器,然后启动全部运行(3 个 checkpoint × 3 次,加 RefSpatial 一次) |
| `07_gflowrl_improvement/p0_same_machine_reeval/scripts/gate.sh` | Health gate: every dump must pass `parse_dump.py --strict` | 健康闸门:每个 dump 都要过 `parse_dump.py --strict` |
| `07_gflowrl_improvement/p0_same_machine_reeval/scripts/paired.py` | Paired per-sample tests between checkpoints over the three runs | 三次运行上 checkpoint 之间的逐样本配对检验 |
| `07_gflowrl_improvement/p0_same_machine_reeval/scripts/flips.py` | Always-right, always-wrong and flipping samples per checkpoint | 每个 checkpoint 的恒对、恒错、翻转样本 |
| `07_gflowrl_improvement/p0_same_machine_reeval/scripts/tfcheck.py` | Lists samples with tool failures or the turn cap, and how they did in the other runs | 列出工具失败或顶轮的样本,以及它们在其他运行里的表现 |
| `07_gflowrl_improvement/p0_same_machine_reeval/scripts/p2.sh` | Vacant analysis on the three checkpoints: pass-through, edits, tool-point hits | 三个 checkpoint 的 Vacant 分析:透传、改点、工具点命中 |
| `07_gflowrl_improvement/p0_same_machine_reeval/dumps/*.jsonl.gz` | The 27 dumps: `{sft,p4,cp}_{run1..3,ref1}_<benchmark>` (`cp` = C′) | 27 个 dump:`{sft,p4,cp}_{run1..3,ref1}_<benchmark>`(`cp` 即 C′) |
| `07_gflowrl_improvement/p0_same_machine_reeval/logs/*.gz` | Eval log per run and the driver log | 每次运行的 eval 日志与驱动日志 |
| `07_gflowrl_improvement/p1_prep/README.md` | Launch commands and findings (Chinese) | 启动命令与发现 |
| `07_gflowrl_improvement/p1_prep/p1a_check.py` | CPU self-check of P1(a): filter off is identical to the original, filter on changes only degenerate rows, and the on-policy gradient equals a true loss mask | P1(a) 的 CPU 自检:关闭时与原函数相同,打开时只改退化行,on-policy 下梯度等于真正的 loss mask |
| `07_gflowrl_improvement/p1_prep/p1_monitor.py` | Monitor over rollout and validation dumps: share of anchor-only queries (the primary metric), edit rate, accuracy, tool-point hit rate | 读 rollout 与 validation dump 的监控:只问锚物体的查询占比(主判据)、改点率、正确率、工具点命中率 |
| `07_gflowrl_improvement/p1_prep/p2_query_type.py` | Splits Vacant questions by how the query to RoboRefer is written | 按给 RoboRefer 的查询写法拆分 Vacant 题目 |
| `07_gflowrl_improvement/p1_prep/robospatial_vacant.parquet` | The 122 Vacant questions, used as the in-training validation set (taken from the upstream eval benchmark) | 122 道 Vacant 题,作训练中的验证集(取自上游的 eval benchmark) |
| `07_gflowrl_improvement/infra_prep/README.md` | What each Infra change does, how to switch it on, self-check results (Chinese) | 每项 Infra 改动做什么、怎么开、自检结果 |
| `07_gflowrl_improvement/infra_prep/infra_drop_check.py` | CPU self-check of dropping degenerate groups: gradients on the kept rows equal P1(a) | 丢退化组的 CPU 自检:保留行的梯度与 P1(a) 相同 |
| `07_gflowrl_improvement/infra_prep/dump_images_check.py` | CPU self-check of the `dump_images` switch | `dump_images` 开关的 CPU 自检 |
| `07_gflowrl_improvement/infra_prep/tool_timing_check.py` | CPU self-check of the per-tool timing: return values unchanged, latency split correct | 按工具计时的 CPU 自检:返回值不变,耗时拆分正确 |
| `07_gflowrl_improvement/infra_prep/tool_timing_summary.py` | Summarizes the per-tool timing logs | 汇总按工具计时的日志 |
| `07_gflowrl_improvement/infra_prep/stage_timing.py`, `07_gflowrl_improvement/infra_prep/stage_timing_p7_85steps.csv` | Per-stage time from a training log, and the result for the 85-step run | 从训练日志抽每个阶段的耗时,以及 85 步训练的结果 |
| `07_gflowrl_improvement/infra_prep/preflight_nccl.sh`, `07_gflowrl_improvement/infra_prep/nccl_bw_probe.py` | Machine acceptance before training: is GPU peer-to-peer usable, and the all-reduce bandwidth | 开训前的机器验收:GPU P2P 能不能用、all-reduce 带宽多少 |
| `07_gflowrl_improvement/infra_prep/check_tool_packing.py` | Machine acceptance before training: do the tool actors fit on the tool GPUs card by card, and does each share cover the tool's memory | 开训前的机器验收:工具 actor 按卡排不排得下、份额够不够显存 |
