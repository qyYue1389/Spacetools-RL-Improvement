# REPORT_B —— 4× A6000 全量 SFT（3000 步）执行报告

承接 `REPORT_A.md`（2× A6000 上的阶段 A–C：环境构建 + 30 步探测）。
本文记录 4 卡全量训练的完整执行、验收结果，以及**若干条推翻前序文档的实测发现**。

- 执行日期：2026-09-08
- 机器：Vast.ai 实例，4× RTX A6000（49140 MiB，sm_86），驱动 570.181，377 GiB 内存，119 G 卷
- 结论：**训练成功，验收全过，checkpoint 可作为 π_ref 使用**

---

## 0. 一句话

3000 步跑满，退出码 0，耗时 **7 小时 45 分**；`eval_loss` 六点单调下降
**0.168 → 0.045**；全局 batch 经独立复核确为 8；Phase 3 四项修复到位。

但**过程中有一处险情**：GPU2 全程贴着显存上限跑，余量仅 1.4 GiB（详见 §3.1）。

---

## 1. 交付清单（对应 `NEXT_MACHINE.md` §10）

### 硬件

```
index, name, driver_version, compute_cap, memory.total
0..3, NVIDIA RTX A6000, 570.181, 8.6, 49140 MiB
```

### 启动行

```
GPU 数 4 · per_device=2 · ga=1 · 全局 batch 8 ✓
```

### B.1 数据口径（六项全过）

| 项 | 期望 | 实测 |
|---|---|---|
| `<tools>` 块长度 | 8595 | ✓ 8595 |
| sha256 前缀 | `72c71c806f64e162` | ✓ 一致 |
| 含 `${obj_name}_detections` | 是 | ✓ |
| 工具数 | 11 | ✓ 11 |
| system 全表一致 | 唯一值 1 个 | ✓ 7020 条全一致 |
| robot 样本残留 | 0 | ✓ 0 |

样本流水：**7907 条原始 → 过滤 887 条 robot 工具样本 → 7020 条进训练**。

### 速度

| | 值 |
|---|---|
| `train_runtime` | 27929.7 s = **7.76 h** |
| 均值 | **9.31 s/it** |
| 稳态实测（32 步 / 300 s 独立窗口） | 9.38 s/it |
| `NEXT_MACHINE.md` 预期 | 5–6 h |

**慢 30%，原因已查清且不可操作**，见 §3.2。

### 显存

| GPU | 峰值 | 占比 | 余量 |
|---|---|---|---|
| 0 | 36830 MiB | 75% | 12.0 GiB |
| 1 | 37090 MiB | 76% | 11.8 GiB |
| **2** | **47730 MiB** | **97%** | **1.4 GiB** ⚠️ |
| 3 | 36950 MiB | 75% | 11.9 GiB |

### loss

| | |
|---|---|
| train loss | 0.8621（step 5）→ **0.0625**（step 3000） |
| `train_loss`（全程均值） | 0.19167 |
| 记录点 | 600 个，`global_step=3000/3000` |

`eval_loss` 六点，**全程单调下降、无一回升**：

| step | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---|---|---|---|---|---|
| eval_loss | 0.1680 | 0.1064 | 0.0975 | 0.0684 | 0.0559 | **0.0452** |
| 环比 | — | −37% | −8% | −30% | −18% | −19% |

⚠️ `val_size` 仅 20 条，单点噪声大，只看趋势。

### 全局 batch 独立复核

不依赖配置自述，从 HF Trainer 的产出反推：

```
train_runtime × train_samples_per_second = 27929.7 × 0.8590 = 23992
期望 = max_steps × 全局 batch = 3000 × 8 = 24000
偏差 0.04%  ✓
```

依据 `transformers/trainer.py:5709` —— `num_train_samples = args.max_steps *
total_train_batch_size`，而 `total_train_batch_size` 由**运行时真实 world_size** 算出，
所以这条能抓住「在错误卡数上静默跑出不同全局 batch」。

⚠️ 它是**配置层**校验，不证明 dataloader 真喂了 24000 条样本。

⚠️ **不要看 `epoch` 字段**（报 3.122）。`DATASET_SPEC` 把数据集注册了 3 遍
（`run_sft.sh:236`），HF 按 7020×3=21060 算 epoch，永远对不上，是假警报。

### Phase 3 四项（上游自带，我们一行未改）

| 项 | 状态 |
|---|---|
| 移除 `config.json` 的 `text_config` | ✓ 日志有 `Removed text_config` |
| `tie_word_embeddings = true` | ✓ 日志有 `Set tie_word_embeddings=True` |
| 拷 `preprocessor_config.json` | ✓ 存在 |
| `model_type` | ✓ `qwen2_5_vl`（**不是**会让 sglang 崩的 `qwen2_5_vl_text`） |

### 格式冒烟 + base 对照

抽 12 条（seed=0，greedy）：

| 指标 | SFT ckpt | base Qwen2.5-VL-3B-Instruct |
|---|---|---|
| `<think>` 标签 | **100%** | 41.7% |
| `</think>` 闭合 | 100% | 33.3% |
| `<tool_call>` | 100% | 83.3% |
| JSON 合法 | **100%** | 83.3% |
| 工具名合法 | **100%** | 75.0% |

合格线 90%（由真值率 99.8% 推导）。判别项 `<think>`：**+58 个百分点**。

⚠️ 样本取自训练集，**这只判断「SFT 有没有生效」，不是能力度量**。
真实分数只有 SpaceTools-RL 的 `run_eval.sh` 能给。

base 调用到的工具分布也说明问题：base 大量误用 `vision_ops.index_at`（5/12）
并产出了不存在的 `answer` 工具；ckpt 则集中在 `roborefer.detect_one`（8/12）
和 `depth_estimator.*`，与任务类型吻合。

### 偏离

- **liger 未开**（用户明确要求）
- **B.4 禁止项一个未碰**：`packing` / `neat_packing` / `finetuning_type` /
  `freeze_vision_tower` / `freeze_multi_modal_projector` / `cutoff_len` /
  `image_max_pixels` / `learning_rate` / `max_steps`

### 环境指纹

```
torch 2.9.1+cu128 · transformers 4.57.1 · deepspeed 0.19.6
flash_attn 2.8.3.post1 · llamafactory 0.9.5.dev0
SpaceTools-SFT git b7ebbf320bb130c230856e61107a566bc119e4d8
```

### checkpoint 哈希

```
779717e1d8f9611009066b821f338c0af2a80215d8abd630f21d9d3b28540e85  model-00001-of-00002.safetensors
b1ae12021922f489237d50b4f4c25559821b98cc8589cad0aa4c6dce573b0ba6  model-00002-of-00002.safetensors
```

完整清单见 `EVIDENCE_*/CKPT_SHA256SUMS`。

---

## 2. 产物位置

```
/workspace/experiments/full/
├── sft_checkpoint/                 最终 ckpt(7.6 G)—— 这是 π_ref
│   ├── model-0000{1,2}-of-00002.safetensors
│   ├── config.json                 已过 Phase 3
│   ├── chat_template.json          ← 我们补的,见 §4.1
│   ├── toolshed_config.yaml        v1 的 11 个 schema
│   └── checkpoint-{500,1000,1500,2000,2500,3000}/   各 7.6 G,合计 46 G
├── sft_data/
│   ├── data/train.json             7020 条,已注入 schema
│   ├── images/{a,b}/               7459 张
│   └── hf_download/                原始下载(6.0 G,冗余,可删)
└── EVIDENCE_20260908_08{1901,2425}/  证据各 1.1 M(两次 FINALIZE,内容等价)

/workspace/runpod-handoff/probe3000.log     完整训练日志(含启动行)
/workspace/runpod-handoff/gpu_probe3000.csv 28048 次 GPU 采样
/workspace/val/{FINALIZE.sh,smoke_check.py} 验收脚本(已修,见 §5)
```

磁盘：119 G 卷，已用 84 G，**剩 36 G**。

---

## 3. 推翻前序文档的实测发现

> 按 `NEXT_MACHINE.md` §9.1「拿不准选保守的，并记下来」。

### 3.1 ⚠️ 最重要：4 卡的显存余量不是均匀的，最坏 rank 只剩 1.4 GiB

`NEXT_MACHINE.md` §4 预期 4 卡「余量 ~10 GiB」。**实测 GPU2 峰值 47730 / 49140 MiB
（97%），余量仅 1.4 GiB。**

而且不是瞬时尖峰：**约 90% 的采样点（25122 / 28000）都在 45 GiB 以上**，
从第 49 分钟起持续到训练结束。其余三张卡稳定在 36.8–37.1 GiB。

**机制推测**：PyTorch 缓存分配器的高水位效应 —— rank 2 早期撞上一个超长序列或大图批次，
分配器扩容后不归还显存。`streaming: true` 下数据集只有 4 个分片、4 个 rank 各吃一片，
分片间的序列长度分布不均会放大这种效应。

**给下一个人的含义**：

- 这次离 OOM 只差 1.4 GiB，而 `save_only_model: true` 意味着**一旦 OOM 就要从头再来 7.75 小时**
- 不要照着「余量 ~10 GiB」规划。真实的规划依据应该是**最坏 rank ~1.4 GiB**
- 这也解释了为什么 `per_device=4` 会 OOM —— 不是「差 3 GiB」，是本来就没有余量
- 若要复现，建议加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 缓解碎片，
  或在训练中监控每卡峰值而不是只看总体

### 3.2 5–6 h 的预期错了，真实是 7.75 h —— 但不可操作

`NEXT_MACHINE.md` §4 按 2 卡实测线性外推得到 5–6 h。外推漏了梯度累积的变化：

| | 2 卡（REPORT_A 实测） | 4 卡（本次） |
|---|---|---|
| per_device × ga | 2 × **2** | 2 × **1** |
| 每卡每步样本数 | 4 | 2 |
| s/it | 11.22 | **9.31** |
| 吞吐 | 0.713 样本/s | **0.859 样本/s** |

**加倍卡数只换来 1.20× 吞吐，扩展效率 60%。**

原因：`ga` 从 2 掉到 1，第一个 micro-batch 的 `no_sync` 摊销消失，
**每一步都要做一次完整的 ZeRO-2 梯度规约**；ZeRO-2 又把优化器状态从 2 份切成 4 份，
通信量本身也涨了。

**为什么不可操作**：全局 batch 锁死 8，`per_device × ga × 卡数 = 8`，
4 卡上唯一解就是 `2 × 1`。退回 2 卡是 9.3 h（更慢），8 卡没有。
**7.75 h 已经是可选方案里最快的**，除非改全局 batch —— 而那会毁掉 π_ref 的口径。

### 3.3 「GPU 利用率 100% → 算力主导」这个推论不成立

`NEXT_MACHINE.md` §7 用「利用率中位数 100%」推翻了 `TASK.md` 的「开销主导」，
结论写成「是算力主导」。**这个推论本身不严谨。**

`nvidia-smi` 的 utilization 统计的是「是否有 kernel 在跑」，而
**NCCL 的 all-reduce 是忙等自旋，同样计为 100%**。所以 100% 只能说明 GPU 上一直有 kernel，
不能区分算力还是通信。

§3.2 的 60% 扩展效率正是反例：若真是纯算力主导，加倍卡数该接近 2× 吞吐。
`analyze_probe.py` 至今仍在为每张卡打印「→ 算力主导」，**这行结论应该删掉或改写**。

### 3.4 checkpoint 是 7.6 GiB/次，不是 15.18

`NEXT_MACHINE.md` §1 说 7.6 GiB/次，§7 又说「实际 15.18 GiB / 次」，两处打架。
**实测 7.6 GiB/次**，§7 不适用。

6 次中间 ckpt 46 G + 根目录最终权重 7.6 G ≈ **53 G**，与 §1 的估算一致。
§7 那个 15.18 大概率是把 `checkpoint-N/` 和根目录各一份加起来算的，
而根目录那份要到训练结束才写。

### 3.5 HF 限速的根因是匿名 IP 配额，不是并发数

`NEXT_MACHINE.md` §2.1 把限速归因于并发（「32 并发跑到约 2000 个文件后 HF 会限速」
「并发别超过 8」）。**实测用 8 并发照样被限**，累计 1753 次 429。HF 返回的原文是：

> We had to rate limit your IP (…). To continue using our service, **create a HF account
> or login to your existing account, and make sure you pass a HF_TOKEN** if you're using the API.

这是**按 IP 的匿名配额**。第一轮 22.1 分钟下完 7463 个文件，但 32 个大小核对不符；
第二轮（仍匿名）只修好一半。**用 HF token 登录后，剩余 16 个文件几秒下完，零失败。**

**给下一个人的含义**：`prefetch.py` 的并发退避机制是对的，但真正的解法是
**带 token**。`NEXT_MACHINE.md` §2.1 应该加一句「先设 `HF_TOKEN`」。

### 3.6 「全局 batch = 8 是论文口径」—— 依据是上游脚本默认，不是论文正文

`run_sft.sh` 和 `NEXT_MACHINE.md` 都写「论文口径，不可变」。追溯下来，依据是：

```
上游 run_sft.sh.orig:
  42:  NUM_GPUS="${NUM_GPUS:-8}"        # 文档写 "GPUs for training (default: 8)"
  235: per_device_train_batch_size: 1
  236: gradient_accumulation_steps: 1
  → 1 × 1 × 8 = 8
```

**本地找不到任何论文原文说「global batch = 8」。** `TASK.md` 唯一一次引用论文
Table 6 是为了论证另一件事（SFT 不需要部署视觉工具）。

而且这条推断有软肋：`NUM_GPUS` 在第 42 行声明后**从未被任何一行引用**，是空操作
（真正决定卡数的是 `FORCE_TORCHRUN=1` + 可见 GPU 数）。那个「8」只活在一个注释和一个死变量里。

**但「锁死」这个做法本身仍然正确**，理由与论文无关：

1. 上游代码的全局 batch 是**随机器变的** —— `pd`/`ga` 写死为 1、`NUM_GPUS` 空操作，
   所以原版代码在 4 卡上会**静默**训出全局 batch = 4，2 卡上是 2，且不报任何错
2. π_ref 是 RL 两臂的共同起点，batch 是多少可以商量，但必须先定死再开跑
3. batch 变了模型就变了 —— lr 固定 2e-5、cosine 跑满 3000 步，改 batch 等于改优化动力学

**建议**：把 `run_sft.sh` 和 `NEXT_MACHINE.md` 里的「论文口径」改成
「上游脚本默认口径（8 卡 × pd1 × ga1）」，避免下一个人以为它有论文背书。
若有论文原文，值得对一眼。

---

## 4. 上游的遗漏与我们的补丁

### 4.1 Phase 3 漏拷 `chat_template.json`

上游 Phase 3 从 base model 拷了 `preprocessor_config.json`，但**没拷
`chat_template.json`**（processor 级模板，sglang 加载 VLM 时读它）。

阴险之处：checkpoint 里有 `chat_template.jinja`（tokenizer 级，transformers 4.57
自动写出），所以「chat template 存在」这类检查**照样变绿**，而 RL 侧可能因缺
processor 级模板而崩。

**已补**（与 base 逐字节相同）：

```bash
BASE=$(ls -d /workspace/hf/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/*/ | head -1)
cp "$BASE/chat_template.json" /workspace/experiments/full/sft_checkpoint/
```

并已加入 `FINALIZE.sh` 第 3 节的必需文件列表。

### 4.2 `launch_probe.sh` 的 PATH 地雷（我们自己踩的）

首次启动训练即失败，Phase 1 报 `ModuleNotFoundError: No module named 'huggingface_hub'`。

**根因是调用方式**：从一个**已经 `conda activate spacetools-sft` 的 shell** 调用
`launch_probe.sh`，其第 4 行 `export PATH=/opt/conda-st/bin:$PATH` 把 base 的 bin
前置到已激活 env 之前；紧接着的 `conda activate spacetools-sft` 因为「已激活」
**变成空操作**，不会重新把 env 的 bin 提前 → `python3` 落到 base。

**最阴险的地方**：`CONDA_DEFAULT_ENV` 仍然显示 `spacetools-sft`，
环境**看起来是激活的**，只有 `command -v python3` 才露馅。

`conda_hook.sh` 的注释精确预告了这个坑，但同样的写法在 `launch_probe.sh`
第 4 行本身也有一份 —— 从干净 shell 调用时无害，从已激活的 shell 调用就踩雷。

**已修**：删掉 PATH 前置，改用绝对路径装 hook，并加断言。
三种父 shell 条件下均验证通过，拒绝路径也验证会真的触发（非死代码）。

---

## 5. 验收脚本的修改清单

原文件备份在 `/workspace/val/*.bak`。

### `FINALIZE.sh`

| # | 问题 | 影响 | 修法 |
|---|---|---|---|
| 1 | 同 §4.2 的 PATH 地雷 | **后果不对称**：第 1–4 节只用 stdlib，在 base python3 下照样全绿；只有第 5 节 `import torch` 和第 8 节 `huggingface_hub` 会失败 → 「验收全过 + 证据不全」 | 绝对路径装 hook + `python3` 解析断言 |
| 2 | `REPO_DIR` 默认 `/workspace/SpaceTools-SFT`，glob `sft_v1_*` | 自动定位实验目录**必然失败**（`run_sft.sh` 的 `REPO_DIR` 是 `LLAMA_DIR/..` = `/workspace`；且本次目录名是 `full`） | 新增 `EXP_ROOT=/workspace`，glob 兼顾 `sft_v1_*` 和 `full` |
| 3 | 日志只找 `$EXP/*.log` 和 `scripts/spacetools/launcher.log` | **收不到 `probe3000.log` 和启动行** —— 而那正是脚本自己强调的口径证据 | 加 `$HANDOFF/probe*.log` 和 `$HANDOFF/launcher*.log` |
| 4 | 必需文件列表缺 `chat_template.json` | 见 §4.1 | 加入列表 |

**未修（cosmetic，不影响结论）**：

- 第 4 节报「工具 schema 数 ≈ 12（v1 期望 11）」：用松正则数 `"name":`，
  多算的一处来自 system prompt 里演示格式的模板行
  `{"name": <function-name>, "arguments": <args-json-object>}`。真实值 11
- 第 3 节打印「dtype None」：transformers 4.57 把 `torch_dtype` 改名为 `dtype`，
  脚本读的是旧键。config.json 里实际是 `dtype: bfloat16`
- `PRUNE=1` 保留 `checkpoint-$KEEP_MID` 但第 8 节 `ignore_patterns` 不上传它 ——
  pod 一销毁那份 sanity 对比 ckpt 就没了
- `TOT=$(du -sh -c $MIDS ...)`：`$MIDS` 未加引号靠分词

### `smoke_check.py`

| # | 问题 | 影响 | 修法 |
|---|---|---|---|
| 1 | `census` 统计**所有** assistant 轮，但脚本只生成**第一个**轮 | 分布完全不同（所有轮 `tool_call` 63.3% / `<answer>` 36.7%；**第一轮 99.8% / 0.2%**）。写死的 0.7 合格线会把 75% 的坏 ckpt 判成 ✓；且「期望」清单里含 `<answer>`，而正确行为是几乎不出现 | 只统计第一轮；阈值由真值率推导（`0.9×` 合格 / `0.5×` 警戒）→ 本次合格线 90% |
| 2 | 无训练期保护 | 会在 `cuda:0` 加载 ckpt + base 各 6.2 GiB，而训练每卡已占 35–36/48 GiB → **可能撞 OOM 杀掉训练** | 检测 `train.pid` 存活即拒绝（退出码 2），`ALLOW_DURING_TRAINING=1` 可强制 |
| 3 | base 走 Hub | 匿名 IP 已被限速，可能失败或重下 7.1 GB | 新增 `resolve_model()` 解析本地 snapshot |
| 4 | `MARKERS` 缺 `<think>` | base 对照用「JSON 合法」很脆 | 加 `<think>`/`</think>`，判别项改用 `<think>` |

**第 4 条的修复被本次运行实证了价值**：

```
JSON 合法:  ckpt 100% vs base 83.3%  → 差 17 个百分点 < 20 阈值 → 会误判「拉不开差距」✗
<think>:   ckpt 100% vs base 41.7%  → 差 58 个百分点              → 正确判定 ✓
```

如果不改，这次会**误报失败**。

### ⚠️ 我在第 3 条的修复里犯了一个错（已修正，记录以儆效尤）

第一版写的是 `os.environ.setdefault("HF_HOME", "/workspace/hf")`。
但 Vast 实例**已经预设了 `HF_HOME=/workspace/.hf_home`**，`setdefault` 不覆盖已有值，
于是 glob 到空目录、回落到 Hub，**把 base model 重下了一遍（7.1 GB）**。

修复无效但**表现为成功**（下载成功、验收通过），只有对比磁盘用量才发现。
已改为遍历多个候选根目录（`$HF_HOME`、`/workspace/hf`、`~/.cache/huggingface`）
并要求 snapshot 内确有 `*.safetensors` 才认。

`/workspace/.hf_home` 里那 7.1 G 是纯冗余副本（与 `/workspace/hf` 同一 commit），可删。

---

## 6. 未完成 / 待决事项

| 事项 | 收益 | 状态 |
|---|---|---|
| **把 ckpt 传出这台机器** | 保命 —— pod 销毁即全失 | **未做，最高优先级** |
| 删 `/workspace/.hf_home` 冗余副本 | +7.1 G | 待批准 |
| 删 `sft_data/hf_download/` | +6.0 G | 待批准（训练已结束，不再需要） |
| `PRUNE=1` 清中间 ckpt（保留 1500） | +38 G | **建议等 ckpt 传出后再做** |
| 合并两个 `EVIDENCE_*` 目录 | 整洁 | 内容等价，留一份即可 |
| 跑 SpaceTools-RL 的 `run_eval.sh` | 真实能力分数 | 本机未装工具栈 |

上传命令（`FINALIZE.sh` 第 8 节已内置）：

```bash
HF_REPO=<用户名>/spacetools-sft-v1-4xa6000 HF_TOKEN=hf_xxx \
    bash /workspace/val/FINALIZE.sh /workspace/experiments/full
```

---

## 7. 给下一个人的三条

1. **先设 `HF_TOKEN`。** 匿名 IP 配额会在下数据集（7463 个文件）时把你卡住，
   而且症状是 `LocalEntryNotFoundError`，看起来像并发问题（§3.5）。
2. **从干净 shell 调用 `launch_probe.sh`**，不要先 `conda activate`。
   脚本现在有断言会拦下，但理解原因比依赖断言好（§4.2）。
3. **显存余量按最坏 rank 1.4 GiB 规划，不是 ~10 GiB。**
   这次离 OOM 很近，而 `save_only_model: true` 意味着 OOM = 从头再来 7.75 小时（§3.1）。

以及一条方法论 —— `NEXT_MACHINE.md` §9 的四条判断原则这次全部应验：

- 「退出码会骗人」：`launch_probe.sh` 首次启动 rc=1 但 launcher 只打印了两行正常日志
- 「单次通过不是证据」：`resolve_model` 的 `setdefault` bug 表现为完全成功
- 「跑通了但数字是错的」：`census` 统计错轮次、`smoke_check` 的 base 对照阈值，
  两者都不会报错，只会给出错误结论
- 「每一步之后验证实际状态」：GPU2 的 1.4 GiB 余量是训练结束后翻 csv 才发现的，
  过程中所有指标都是绿的
