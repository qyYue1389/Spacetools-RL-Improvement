# 阶段 A–C 报告(2× A6000 / RunPod)

2026-09-07 · 对应 `TASK.md` 阶段 A(环境)· B(配置)· C(30 步探测)
**状态:三个阶段全部完成,`run_sft.sh` 三个 Phase 走通,退出码 0**

---

## 1. 结论

环境就绪并经完整训练验证;配置零偏离;30 步探测跑完。

**三个数**(`TASK.md` 阶段 C 要的):

| 指标 | 实测 | 判读 |
|---|---|---|
| `s/it`(跳过前 5 步,25 样本) | **11.22** 中位数 | × 3000 = **9.3 小时** |
| GPU 利用率中位数 | **100% / 100%** | **算力主导** |
| 显存峰值 | GPU0 **48491/49140 (99%)**<br>GPU1 44745/46068 (97%) | **余量仅 0.6 GiB** |

两个和 `TASK.md` 预期相反的结论,见 §7。全量尚未启动,待定项见 §9。

---

## 2. 硬件

```
2 × NVIDIA RTX A6000 · driver 580.159.03 · compute_cap 8.6
GPU0  49140 MiB (47.4 GiB)
GPU1  46068 MiB (44.4 GiB)     ← 比 GPU0 少约 3 GiB
```

⚠️ **两卡显存不等,这会影响阶段 B。** ZeRO-2 下两个 rank 持有相同形状,
OOM 天花板由 GPU1 的 44.4 GiB 决定,不是 `TASK.md` B.3 假设的 48 GB。
那笔账要重算:

```
参数 8.1 + (grad+opt 30.6)/2 = 23.4 GiB
              + logits ~16   ≈ 39 GiB / 44.4      余量 5 GiB(文档预期 9 GiB)
```

`per_device_train_batch_size: 4` 比文档预期的更悬。

---

## 3. 环境

```
python        3.11
torch         2.9.1+cu128      cuda 12.8
torchvision   0.24.1+cu128     ← 必须与 torch 配套
torchaudio    2.9.1+cu128      ← 同上,原为 2.11.0 导致 ABI 崩溃
transformers  4.57.1
flash-attn    2.8.3.post1      自编,仅 sm_80
deepspeed     0.19.6
llamafactory  0.9.5.dev0       editable → /workspace/SpaceTools-SFT/src
accelerate    1.11.0    trl 0.24.0    peft 0.18.1    datasets 4.0.0
```

安装位置 `/opt/conda-st`(容器盘)。网络卷上实测 conda 慢到不可用(~200 vs >20,000 files/s),
所以装在容器盘并立刻备份到持久卷 —— 容器重启即失,备份见 §6。

### 验收

| 项 | 结果 |
|---|---|
| `pip check` | 通过,无依赖冲突 |
| `transformers == 4.57.1` 硬断言 | 通过 |
| `nvcc` 版本 | 12.8(`/usr/local/cuda-12.8`),`CUDA_HOME` 校验通过 |
| `check_arch.sh /opt/conda-st/envs` | 扫描 7 个含 CUDA 的 .so,**0 个不合格** |
| flash-attn 实跑 kernel | `(2,64,4,64)` bf16,有限值,在 sm_86 上 ✓ |

最后一项不只是 `import` 通过 —— 在 GPU 上真算了一次。
这同时实测确认了 `TASK.md` A.3 的前提:**为 sm_80 编的 cubin 在 sm_86 上能跑**。

---

## 4. 需要记住的四个事实

### 4.1 容器内存上限是 93.1 GiB,不是 `free` 显示的 503 GB

```
/sys/fs/cgroup/memory.max   99,999,997,952 B  =  93.1 GiB   ← 真实上限
free -g                     total 503                       ← 宿主机的
```

`free` 和 `nproc` 在容器里读的都是宿主机。**这台机器上任何按内存定并发/批量的地方,
都必须读 cgroup。** 超了不是报错,是整个容器被 OOM killer 干掉、容器盘上的东西一起没。

### 4.2 flash-attn 不读 `TORCH_CUDA_ARCH_LIST`

`flash_attn/setup.py:69`:

```python
def cuda_archs() -> str:
    return os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;120").split(";")
```

`TASK.md` A.3 反复强调的 `TORCH_CUDA_ARCH_LIST` **对这个包一行代码都不起作用**
(对其他 CUDA 扩展仍有效,所以脚本里保留了)。默认会为 sm_80/90/100/120 各编一份,
后两个是 Blackwell,这里毫无用处。

另外 `setup.py:124` 给每个 nvcc 加 `--threads ${NVCC_THREADS:-4}`,而 `--threads`
是**按 arch 目标并行**的(实测:每个 nvcc 挂 N 个 `cicc`,N = arch 数)。所以:

```
真实并发编译进程 = MAX_JOBS × min(NVCC_THREADS, arch 数)
```

按 `MAX_JOBS` 估内存会低估数倍。本次配置 `FLASH_ATTN_CUDA_ARCHS=80` + `MAX_JOBS=12`
→ 12 个并发 → 实测 anon 峰值 **51.4 GiB / 93.1 GiB(55%)**,72 个编译单元约 17 分钟。

### 4.3 flash-attn 自己的内存自适应在容器里是坏的

`setup.py:513` 在 `MAX_JOBS` 未设时会自动定并发:

```python
free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
max_num_jobs_memory = int(free_memory_gb / 9)
```

`psutil.virtual_memory()` 同样读宿主机内存。在这台机器上它会算出 `min(96/2, 461/9) = 48`,
必然炸。**所以 `MAX_JOBS` 必须显式设置,不能依赖上游的自适应。**

### 4.4 上游只钉 `torch`,不钉 `torchvision` / `torchaudio`

`setup_envs.sh` 写的是 `pip install torch==2.9.1 torchvision torchaudio`。后两个无约束,
pip 把 `torchaudio` 解析到 **2.11.0**(给 torch 2.11 编的)。
`llamafactory/data/mm_plugin.py:29` 会 `import torchaudio` → `undefined symbol:
torch_library_impl`,训练在分布式初始化后立刻崩。

这和上一轮 `transformers` 被顶成 5.x 是**同一形状的 bug**(上游对传递依赖不设约束),
区别是这个会响亮地崩、那个会静默地毁结果。

**验收必须走真实导入链。** 我最初的 §6 验收只 `import torch, flash_attn, deepspeed,
llamafactory` 就放行了 —— `import llamafactory` 是惰性的,不触碰 `mm_plugin`。现在改成:

```python
import torch, torchvision, torchaudio, flash_attn, deepspeed, llamafactory
from llamafactory.data.mm_plugin import get_mm_plugin   # 它内部 import torchaudio
from llamafactory.train.tuner import run_exp
```

### 4.5 数据集下载:上游的并发不生效

`snapshot_download` 下 `siyich/spacetools-sft`(7463 文件 / 5.89 GiB)只有 **0.8 MB/s**,
外推 1–2 小时。排除法:HF 网速 31 MB/s、网络卷顺序写 451 MB/s、xet 只影响 30% —— 
真因是**进程里只有 2 个线程**,给 `snapshot_download` 传 `max_workers=32` 也不生效。

改用它底层的 `hf_hub_download` 自己控并发(`prefetch.py`),**21 分钟下完**。
元数据格式一致,`snapshot_download` 之后会认出并跳过。

并发不能太高:32 并发跑到约 2000 个文件后触发 HF 限速(表现为 `LocalEntryNotFoundError`),
失败 346 个。降到 8 并发 + 指数退避后稳定在 5–7 MB/s。

⚠️ **第一轮跑完仍有 8 个文件大小不符** —— 逐文件核对抓到了。没有这步核对,
这 8 张缺图会静默进入训练集。

---

## 5. 相对上游 `setup_envs.sh` 的偏离

| 偏离 | 上游 | 影响 |
|---|---|---|
| `FLASH_ATTN_CUDA_ARCHS=80` | `80;90;100;120` | 仅 sm_80 cubin、无 PTX 回退。**A100/A6000/A4000 可跑,H100/Blackwell 需重编。**编译量与内存降到 1/4 |
| 依赖上界补全 + `transformers` 二次钉死并断言 | 无版本约束 | **修正**:否则 `transformers` 会被传递依赖顶成 5.x 且不报错,产出与论文口径不同的 π_ref |
| 钉死 `torchvision==0.24.1` / `torchaudio==2.9.1` | 无版本约束 | **修正**:见 §4.4,否则训练起不来 |
| 增补 `ninja setuptools wheel` | 未装 | 构建期工具。缺 `ninja` 时 flash-attn 退回串行编译,`MAX_JOBS` 失效 |
| `MAX_JOBS=12` `NVCC_THREADS=1` | 自适应(见 §4.3) | 纯并发控制 |

**前三项只影响构建过程或修正上游 bug,第四项只影响编译并发 —— 四项都不改变训练的数学结果。**

`TASK.md` B.4 的禁止项一个都没碰;没有使用 liger。

### `setup.sh` 的持久改动

| 改动 | 作用 |
|---|---|
| 按 cgroup 而非 `free` 计算并发上限,并硬性钳位 | 见 §4.1 |
| `FLASH_ATTN_CUDA_ARCHS` / `NVCC_THREADS` 显式设置,预算按 arch 数折算 | 见 §4.2 / §4.3 |
| 编译期内存看门狗:盯 cgroup `anon`,超 70% 按**进程树**杀 | 让编译失败而非容器被杀。ninja 给每个子进程 `setpgid(0,0)`,只杀进程组会漏掉在跑的编译进程 |
| flash-attn 先 `pip wheel` 落持久卷再安装 | 编译成功一次永久有效,重装从 ~17 分钟变 10 秒 |
| 装完自动 `tar` 整个 conda 到持久卷 | 容器重启后 2 分钟恢复 |
| `SELF_DIR` 在任何 `cd` 之前解析;架构自检失败不再吞成警告 | 否则自检会被静默跳过而退出码仍是 0 |

---

## 6. 交付物

| 路径 | 内容 |
|---|---|
| `/opt/conda-st` | 当前可用的 conda + `spacetools-sft` 环境(9.3 GB) |
| `/workspace/env-backup/conda-st.tar` | 环境备份,9.2 GB,120,428 条目 = 源文件数 |
| `/workspace/wheels/flash_attn-2.8.3.post1-cp311-cp311-linux_x86_64.whl` | 自编 wheel,重装直接用 |
| **`/workspace/bundle/`** | **可迁移包,4.0 GB** |

### 可迁移包

```
spacetools-sft-env.tar.zst   ~4.0G   校验和见同目录 SHA256SUMS
RESTORE.sh                           前置检查 → 校验和 → 解包 → 自动验收
VERIFY.sh                            可单独复跑的验收
README.md                            要求 · 限制 · 偏离
SHA256SUMS
```

(哈希不内联在本文件里 —— 本文件在包内,内联会形成循环。以 `SHA256SUMS` 为准。)

目标机:`sudo bash RESTORE.sh`

包内含 `/opt/conda-st` + `/workspace/{SpaceTools-SFT,SpaceTools,wheels,runpod-handoff}`。
**路径不能改** —— `llamafactory` 是 editable 安装,`_editable_impl_llamafactory.pth`
里写死了 `/workspace/SpaceTools-SFT/src`;conda 的 shebang 也是绝对路径。
保留路径而非改成普通安装,是为了让目标机布局与本机完全一致(零偏离)。

**要求**:x86_64 · compute_cap **8.0 或 8.6** · 驱动 ≥ 525 · glibc ≥ 2.32 ·
`/opt` ≥ 12 GB。**不需要装 CUDA toolkit**(运行时库在包内 `site-packages/nvidia/`)。
H100/Blackwell 会被 `RESTORE.sh` 在解包前拦下。

包内**不含** SFT 数据(7463 文件 / 5.89 GiB)和 base model(~8 GB),`run_sft.sh` 会自动下 ——
但建议改用包内的 `runpod-handoff/prefetch.py`,理由见 §4.5。

### 包的验证

| 验证 | 结果 |
|---|---|
| `zstd -t` 全量解压校验 | 通过,121,227 条目 |
| `tar --compare` 逐条比对内容/大小/权限/mtime | **0 处差异** |
| 实跑 `RESTORE.sh` 全流程 | 6 项前置检查 + 校验和 + 解包 + 验收全过 |
| `VERIFY.sh` 内的 flash-attn kernel | 在 sm_86 上算出有限值 |
| **完整 30 步 SFT 训练** | **三个 Phase 走通,退出码 0** |

### 2026-09-07 重打包

首个版本带着 **torchaudio 2.11.0** 的缺陷(见 §4.4)流出去过,已作废。当前版本:

- 环境修正为 `torchaudio 2.9.1`,并经完整 30 步训练验证
- `VERIFY.sh` 增加 torch 家族版本断言 + 训练真实导入链 + `pip check`
- `README.md` 补上显存要求(实测峰值 48.5 GiB,A4000 装得下环境但跑不了训练)
- 包含本轮新增的工具:`prefetch.py`(并行下载)· `memguard.sh`(内存哨兵)·
  `analyze_probe.py`(三个数)· `verify_b1.py`(schema 验收)· 本报告

---

## 7. 阶段 B:配置

`run_sft.sh` 的配置生成段共改 6 处,`diff` 确认只动了这些行:

| 项 | 原值 | 实际生效 |
|---|---|---|
| `deepspeed` | `ds_z3_config.json` | `ds_z2_config.json`(stage 2,**无 offload**) |
| `per_device_train_batch_size` | 1 | **2**(原定 4,OOM 后退档,见 §8) |
| `gradient_accumulation_steps` | 1 | **2**(配合上一行,保持全局 batch = 8) |
| `eval_steps` | 5 | 500 |
| `save_only_model` | false | true |
| `use_reentrant_gc` | 未设(默认 true) | false |

**全局 batch = 2 × 2 × 2 卡 = 8**,与论文一致。B.4 禁止项一个未碰:
`finetuning_type: full`、两个 `freeze_*: true`、`cutoff_len: 8192`、
`learning_rate: 2e-5`、无 `packing`。**未使用 liger。**

### B.1 schema 验收(两层都过)

源头(`toolshed_v1_config.p4.yaml`)和训练数据(`train.json`)分别验:

```
工具数 11 · <tools> 长度 8595 · sha256 前缀 72c71c806f64e162
${obj_name}_detections 存在 · 7020 条 system 全部一致
v1 已过滤 887 条 robot 工具样本
```

源头这一层能在**下载 6 GB 数据之前**跑,建议以后先做这步。

---

## 8. 阶段 C:30 步探测

### 三个数

| # | 指标 | 实测 |
|---|---|---|
| 1 | `s/it`(跳过前 5 步,25 样本) | 中位数 **11.22** · 均值 11.25 · 范围 10.75–12.07 |
| 2 | GPU 利用率中位数 | **GPU0 100% · GPU1 100%**(有效采样 397/710) |
| 3 | 显存峰值 | **GPU0 48491/49140 (99%)** · GPU1 44745/46068 (97%) |

`train_runtime` 355.8 s / 30 步 = 11.86 s/步(含保存),与 `s/it` 自洽。
宿主内存峰值 19.3 GiB / 93.1,不构成压力。
loss 0.8322 → 0.4548(30 步,1 epoch),grad_norm 7.71 → 2.54,收敛正常。

**外推:11.22 × 3000 = 9.3 小时。**

### ⚠️ 两个和 `TASK.md` 相反的结论

**其一:这个负载是算力主导,不是开销主导。**

`TASK.md` §5 预判"MFU 只有约 5%,很可能是开销主导",据此说"该动的是数据管线,不是换卡"。
判据是 `<40%` 开销主导 / `>70%` 算力主导 —— **实测中位数 100%**。所以建议要反过来:

- 调数据管线**无效**。日志本身就在提示 `Too many dataloader workers: 8
  (max is dataset.num_shards=4)`,数据侧供得过来
- **换更快的卡有效**。A6000 的 bf16 算力约 A100 的 40%

论文 `8× A100-80` 跑 3–4 h,我们 `2× A6000` 跑 9.3 h —— 按卡数和算力折算是自洽的,
反过来说明论文那边也不是开销主导,`TASK.md` 反推的 5% MFU 结论恐怕本身有问题。

**其二:`per_device=4` 装不下,而且不是因为 GPU1 少 3 GiB。**

```
GPU0  已用 39.84 GiB,还需 8.42,仅剩 7.56  (总 47.40)  → 峰值需 ~48.3 GiB
GPU1  已用 38.74 GiB,还需 7.83,仅剩 5.67  (总 44.42)  → 峰值需 ~46.6 GiB
```

**满容量的 GPU0 一样炸。** B.3 估的"23.4 + logits 16 ≈ 39 GiB / 48,紧但应该能装"
只算了稳态 —— 实际稳态就有 39.8 GiB,再加 **8.42 GiB 的 logits 瞬时尖峰**
(`4 × ~6900 tok × 151936 词表 × 2 B`)。它跑完第 1 步(12.47 s/it)才死在第 2 步,
说明是长序列 batch 触发的。

### A/B 的结果

`TASK.md` 想用 `per_device` 4 vs 2 的等时对比判断是否开销主导。**跑法 A 跑不起来**,
没有可比耗时。但利用率 100% 已经直接给出了答案,这个对比不再需要。

### Phase 3 checkpoint 修复(已验证生效)

```
✓ config.json 已删 text_config       ✓ tie_word_embeddings = True
✓ preprocessor_config.json 已取回     ✓ model_type = qwen2_5_vl(不是会让 sglang 崩的 _text)
✓ save_only_model 生效,无优化器分片
```

⚠️ checkpoint 实际 **15.18 GiB**,不是 `TASK.md` §7 估的 8 GB ——
`sft_checkpoint/` 和 `checkpoint-30/` 各存一份完整权重(7.6 GiB × 2)。
全量 `save_steps: 500` 存 6 次约 **53 GiB**。

---

## 9. 全量之前必须解决的一件事

**GPU0 峰值只剩 0.6 GiB 余量。** 这 30 步是从 7020 条里抽的;3000 步会见到约 20 万条样本,
一定有比这更长的序列。按现状直接跑全量,**中途 OOM 是大概率事件**,代价是烧掉数小时。

liger 已明确不开。两条零偏离的出路:

| 方案 | 余量 | 时长 | 评价 |
|---|---|---|---|
| **换 4 卡**(见 §10) | ~10 GiB | ~5–6 h | **推荐** —— 更快且更安全 |
| 本机降到 `per_device=1 + ga=4` | 仍不宽裕 | 12–14 h | 算力主导下 batch 减半会降效率 |

4 卡两头都赢:ZeRO-2 的分片让固定开销从 25.9 降到 15.8 GiB,余量回到 10 GiB;
同时因为是算力主导(§8),加卡直接换来吞吐。

**全量尚未启动,等这一档定下来。**

---

## 10. 多卡:配置已自动化

`run_sft.sh` 现在**按可见 GPU 数推导** `per_device` 和 `ga`,不再写死。
写死会在换机器时静默偏离 —— 比如把 2 卡调好的 `per_device=2 + ga=2` 搬到 4 卡机,
全局 batch 会变成 16。

```bash
GLOBAL_BATCH=8                                   # 论文口径,不可变
PER_DEVICE=min(2, 8 / 卡数)                      # 上限 2:实测 4 在 A6000 上 OOM
GRAD_ACCUM=8 / (卡数 x PER_DEVICE)
断言 PER_DEVICE x GRAD_ACCUM x 卡数 == 8,否则 exit 1
```

### 各卡数的结果(已逐一实测)

| 卡数 | `per_device` | `ga` | 固定开销 | 预估峰值 | 余量(48 GB) | 预估全量时长 |
|---|---|---|---|---|---|---|
| 1 | 2 | 4 | 45.9 GiB | 装不下 | — | — |
| **2**(本机) | 2 | 2 | 25.9 GiB | **实测 47.4 GiB** | **0.6 GiB** | **实测外推 9.3 h** |
| **4**(推荐) | 2 | 1 | 15.8 GiB | ~37.3 GiB | ~10 GiB | ~5–6 h |
| 8 | 1 | 1 | 10.8 GiB | ~32 GiB | ~15 GiB | ~3–4 h(论文配置) |
| **3 / 5 / 6 / 7** | — | — | — | — | — | **拒绝启动** |

### 为什么 3 卡不行

`8 ÷ 3` 不是整数。`per_device x ga x 3 = 8` 无整数解,最接近的是 6 或 9 —— 
两者都偏离论文口径。本 checkpoint 是后续 RL 两臂对比的共同起点(π_ref),
口径不能动,所以脚本直接 `exit 1` 而不是凑一个近似值。

### 4 卡为什么能解决 §9 的显存问题

ZeRO-2 把梯度+优化器态(共 40.2 GiB)分片到各 rank,bf16 权重(5.7 GiB)不分片:

```
2 卡固定开销 = 5.7 + 40.2/2 = 25.9 GiB   → 实测峰值 47.4,反推激活等 ≈ 21.5 GiB
4 卡固定开销 = 5.7 + 40.2/4 = 15.8 GiB   → 预估峰值 15.8 + 21.5 ≈ 37.3 GiB
```

固定开销省 10.1 GiB,余量从 **0.6 → ~10 GiB**,§9 那个"全量中途 OOM 是大概率事件"
的问题消失,而且不必降到 `per_device=1`(那会在算力主导下白白变慢)。

加卡对速度也有效 —— 实测 GPU 利用率 100%,是算力主导(见 §8)。
2→4 卡理论 2x 吞吐,但 A6000 无 NVLink、ZeRO-2 的 all-reduce 走 PCIe,
实际预计 5–6 小时而不是 4.7 小时。

### 两个约束

- **数据集只有 4 个分片**(`streaming: true` 下日志报 `dataset.num_shards=4`)。
  4 卡时每 rank 正好 1 个分片,`dataloader_num_workers` 会被压到 1;因为是算力主导,
  这不构成瓶颈。**超过 4 卡会有 rank 分不到分片**,脚本会警告。
- **卡数由 `CUDA_VISIBLE_DEVICES` 决定,不是 `NUM_GPUS`。**
  上游的 `NUM_GPUS` 声明了但从未被引用(`TASK.md` B.5),`FORCE_TORCHRUN=1` 会用全部可见 GPU。
  想限制卡数就 `CUDA_VISIBLE_DEVICES=0,1`。

### 相对原代码改了什么

原代码(`ChicyChen/SpaceTools-SFT` 的 `scripts/spacetools/run_sft.sh`)对比,
一共 8 处改动。**注意其中只有 3 处是为多卡改的**,其余是 `TASK.md` B.2 要求的配置
和一处下载优化。

#### A. 卡数与 batch(多卡支持的核心,3 处)

| 位置 | 原代码 | 改成 | 为什么 |
|---|---|---|---|
| 第 42 行附近 | 无 | 新增 `GPU_COUNT` 探测 + `GLOBAL_BATCH=8` 推导块(38 行) | 见下 |
| `per_device_train_batch_size` | **写死 `1`** | `$PER_DEVICE`(推导值) | 同上 |
| `gradient_accumulation_steps` | **写死 `1`** | `$GRAD_ACCUM`(推导值) | 同上 |

**为什么必须改:原代码只在 8 卡上是对的。**

原代码写死 `per_device=1, ga=1`,而全局 batch = `per_device × ga × 卡数`:

| 跑在几卡 | 原代码给出的全局 batch | 是否符合论文 |
|---|---|---|
| 8 卡(论文配置) | 1 × 1 × 8 = **8** | ✓ |
| 4 卡 | 1 × 1 × 4 = **4** | ✗ 差 2 倍 |
| 2 卡 | 1 × 1 × 2 = **2** | ✗ 差 4 倍 |

**而且它不会报任何错。** 直接拿原代码在 4 卡上跑,会安静地训出一个全局 batch = 4 的
checkpoint —— 优化轨迹与论文不同,但日志里看不出任何异常。这正是 `TASK.md` 第 9 节
第 4 条说的"跑通了但数字是错的"。

`TASK.md` B.2 已经注意到这一点(要求 2 卡上把 `per_device` 改成 4),但那是**针对 2 卡
写死的另一个值** —— 搬到 4 卡机同样会偏(4 × 1 × 4 = 16)。所以我改成按卡数推导,
而不是把一个写死的值换成另一个写死的值。

推导块还做了三件原代码没有的事:

1. **卡数不能整除 8 就 `exit 1`**(3 / 5 / 6 / 7 卡),而不是凑一个近似的全局 batch
2. **`per_device` 上限锁在 2** —— 实测 4 在 A6000 上 OOM(§8)
3. **收尾断言 `per_device × ga × 卡数 == 8`**,算错就停;并打印一行供人工核对

#### B. `TASK.md` B.2 要求的配置(4 处,与卡数无关)

| 位置 | 原代码 | 改成 | 为什么 |
|---|---|---|---|
| `deepspeed` | `ds_z3_config.json` | `ds_z2_config.json` | A6000 无 NVLink,z3 每层 all-gather 参数在 PCIe 上很贵;48 GB 用 z2 装得下 |
| `use_reentrant_gc` | 未设(默认 `true`) | `false` | 非 reentrant 的 gradient checkpointing 更快 |
| `save_only_model` | `false` | `true` | z2 的 ckpt 含分片优化器态,`save_steps: 500` 会存爆磁盘 |
| `eval_steps` | `5` | `500` | 3000 步会评 600 次 |

#### C. 下载并发(1 处,与训练无关)

| 位置 | 原代码 | 改成 | 为什么 |
|---|---|---|---|
| `snapshot_download` | 默认 `max_workers=8` | `max_workers=32` | 实测默认只有 0.8 MB/s(§4.5)。**但这一行实测不生效** —— 进程里仍只有 2 个线程,实际走的是 `prefetch.py`。改动保留但不要依赖它 |

#### 数学影响

**A 组是为了让数学结果与论文一致**(全局 batch 恒为 8);
B 组的四项(分片策略、gradient checkpointing 实现、checkpoint 内容、评估频率)
都不改变梯度和权重更新;C 组只影响下载。

**所以八处改动没有一处改变训练的数学结果。** 未使用 liger,`TASK.md` B.4 的禁止项
(`packing` / `finetuning_type` / `freeze_*` / `cutoff_len` / `image_max_pixels` /
`learning_rate` / `max_steps`)一个未碰。

#### 原代码保留未动的一处

`NUM_GPUS="${NUM_GPUS:-8}"` 保留原样,只加了注释。它在上游是**空操作** ——
声明了但配置和启动命令里都没引用(`TASK.md` B.5 也指出了这点)。
删掉它是"清理"而非"必要",而这个 checkpoint 是 RL 的共同起点,
能不动的就不动。真正决定卡数的是 `CUDA_VISIBLE_DEVICES`。

### 换到 4 卡机器怎么做

```bash
tar -C /opt -xf conda-st.tar        # 或用 bundle/RESTORE.sh
# 环境不用改:flash-attn 是 sm_80 cubin,任何 A6000 都能跑
bash scripts/spacetools/run_sft.sh  # 自动检测 4 卡,输出 per_device=2 · ga=1
```

启动时会打印一行 `GPU 数 N · per_device=X · ga=Y · 全局 batch 8 ✓` —— **看这一行确认口径**。

---

## 11. 运维要点(全量跑之前会用到)

- **后台任务必须 `setsid` 脱离进程组。** 会话重启会连带杀掉训练 —— 已经踩过一次,
  当时正在下载的 Phase 1 全没了。`launch_probe.sh` 已改。
- **`conda activate` 不跨 bash 进程。** `run_sft.sh` 内部要 activate,而 conda 是 shell
  函数。用 `BASH_ENV` 指向 `conda_hook.sh` 解决;那个 hook 里**不要**前置 base 环境的
  `bin`,否则会盖住已激活的 env。
- **哨兵脚本**:`memguard.sh` 盯 cgroup `anon`,越 44.7 GiB 通知、越 59.6 GiB 杀训练。
  杀的是训练不是容器 —— 宿主内存触顶会让容器被杀、remote 断开,通知都发不出去。
- **`pkill -f` 会匹配到自己的 wrapper shell。** 用 `comm` 字段判断,或按进程树杀。
