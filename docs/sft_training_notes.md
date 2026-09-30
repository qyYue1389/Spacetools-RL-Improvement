# SFT 训练学习笔记

> 范围:batch 相关的配置、显存的来源、ZeRO 的取舍、配置的出处、框架分层。
> 场景:SpaceTools Phase-1 SFT,Qwen2.5-VL-3B,4× A6000。
> 依据:`SpaceTools-SFT` @ `b7ebbf3`(GitHub 源码)、论文 Table 4/6、2× A6000 实测报告。
> 日期:2026-09-08

---

## 1. `per_device_train_batch_size` 与 `gradient_accumulation_steps`

两个都是在凑**"一次参数更新用多少条样本"**。

| | 含义 | 影响 |
|---|---|---|
| `per_device_train_batch_size` | 每张卡一次前向/反向塞几条 | **决定显存峰值** |
| `gradient_accumulation_steps`(简称 `ga`) | 累积几次梯度才更新一次权重 | 几乎不占显存,**用时间换显存** |

```
全局 batch = per_device × ga × 卡数
```

**这个乘积决定训练的数学结果;三个因子怎么分配不影响结果,只影响显存和速度。**

`ga=2` 的实际过程:

```
前向反向(2条) → 梯度存着,不更新
前向反向(2条) → 梯度加上去 → 这时才 optimizer.step()
```

等价于一次吃 4 条,但显存只按 2 条算。

---

## 2. 为什么卡越多越快

本项目全局 batch 锁死为 8,按卡数均分:

| 卡数 | per_device | ga | per_device × ga<br>(每张卡要过几条) | 显存 | 实测/估计时长 |
|--:|--:|--:|--:|---|--:|
| 2 | 2 | 2 | **4 条** | 紧(实测余量 0.6 GiB) | 9.3 h(实测外推) |
| **4** | **2** | **1** | **2 条** | 宽(~10 GiB) | **5–6 h** |
| 8 | 1 | 1 | **1 条** | 最宽 | 3–4 h |

卡是**同时**干活的,所以一次更新的耗时取决于"单张卡要过几条",不是总共 8 条。
4 → 2 → 1,每次减半。

**比喻**:8 箱货搬完算一轮。2 个人每人扛 2 箱要跑 2 趟(这就是 `ga=2`);
4 个人每人扛 2 箱,1 趟搞定。`ga` 就是"人不够时让同一个人多跑几趟"——
货照样搬完(数学结果一样),但时间是串起来的。

**两个不能线性外推的地方:**

1. **不是严格翻倍。** 每次更新结束所有卡要 all-reduce 梯度,卡越多同步开销越大。
   实测 2 卡 9.3 h,4 卡估 5–6 h,接近 2 倍但不到。
2. **8 卡那行是理论值。** 数据集只有 4 个分片,超过 4 卡会有 rank 分不到数据。
   **对本项目 4 卡就是实际最优点。**

---

## 3. per_device 怎么决定显存峰值

```
显存 = 固定部分(不随它变) + 每条样本的部分(随它线性增长)
```

**固定部分** —— 参数 bf16 + 梯度 + 优化器态(ZeRO-2 已按卡分片)。给 4 条还是 1 条都一样大。

**线性部分** —— 激活值,以及真正的元凶:**logits 张量**。

```
logits 显存 ≈ per_device × 序列长度 × 词表大小 × 2 字节
```

Qwen2.5-VL 词表 **151,936**,每条样本每个 token 就要 0.3 MB。代入 ~6900 token:

| per_device | logits 尖峰 |
|--:|--:|
| 1 | ~2 GiB |
| 2 | ~4 GiB |
| **4** | **~8 GiB** ← 实测撑爆的就是它 |

A6000 上 `per_device=4` 的实测:**稳态 39.8 GiB + 尖峰 8.4 GiB ≈ 48 GiB**,正好等于卡容量,两张卡一起炸。
所以脚本把 `per_device` 上限锁在 2。

**两个推论:**

1. **它是瞬时尖峰,不是稳态。** `nvidia-smi` 隔秒采样很可能看不到,但 OOM 是它触发的。
2. **`cutoff_len` 和它相乘。** 现在 `cutoff_len: 8192` 比观测到的最长序列 6900 还高 19%,
   遇到顶格长样本,`per_device=2` 的尖峰会从 4 GiB 涨到 4.9 GiB。
   这就是"2 卡(余量 0.6 GiB)接近必然 OOM"的算法。

### ⚠ 6900 这个数是反解的,不是量的

`per_device=4` OOM 时 PyTorch 报出失败分配的大小(~8.4 GB),套上面的公式反解:

```
8.4e9 字节 ÷ (4 × 151936 × 2 字节) ≈ 6900 token
```

性质:

- **是那 30 步里最长的那个 batch**,不是平均值。用来算 OOM 余量正合适(余量按最坏情况算),
  **不能当"典型序列长度"**。
- 依赖一个假设:那次失败分配确实是 logits 张量。数量级和形状对得上,但没有独立验证。

> 早先用 ~3300 token 估 MFU 得出"开销主导"的结论,就是被这个数推翻的。
> 要坐实它,用 ckpt 的 tokenizer 对 `train.json` 跑一遍出长度分布(mean / p95 / max),几分钟的事。

---

## 4. per_device 和 ga 的具体取值怎么来的

```bash
PER_DEVICE=$(( 8 / GPU_COUNT ));  [ "$PER_DEVICE" -gt 2 ] && PER_DEVICE=2
GRAD_ACCUM=$(( 8 / (GPU_COUNT * PER_DEVICE) ))
```

| 卡数 | 8 ÷ 卡数 | per_device | ga |
|--:|--:|--:|--:|
| 8 | 1 | 1 | 1 |
| 4 | 2 | 2 | 1 |
| 2 | 4 → **封顶 2** | 2 | 2 |

8 卡和 4 卡就是除出来的,没有额外考虑。**上限 2 那一档才是人为加的** ——
2 卡本该分到 4,但 `per_device=4` 实测 OOM,所以卡在 2,差额甩给 `ga=2` 去补。

**优先级:能不动 `ga` 就不动**(它慢),先把 `per_device` 顶到显存允许的最大;顶不动了再让 `ga` 兜底。

---

## 5. 为什么 ga 越大越慢,慢的是什么

**慢的不是 ga 本身,是"每次前向反向喂的样本变少了"。**

同样是每卡过 2 条:

- `per_device=2, ga=1` —— 一次算 2 条
- `per_device=1, ga=2` —— 分两次,每次算 1 条

总计算量一样,但后者把这些**固定开销付了两遍**:

- kernel 启动、layernorm/softmax 这类不随 batch 变的开销
- gradient checkpointing 的重算
- 数据加载与拼 batch

而且矩阵更"瘦",GPU 并行度用不满 —— 这是主要原因。

> ⚠ **对本项目这个惩罚可能很小**:单条样本就有 ~6900 token,序列维度已经把 GPU 喂得挺饱,
> 再叠一条 batch 的并行度增益有限。实测 `per_device=2` 是 MFU 32%,
> `per_device=1 + ga=2` **没测过,不知道差多少**。

**所以让 2 卡比 4 卡慢一倍的是卡少,不是 ga** —— ga 只是卡少之后为了凑够全局 batch 8 的被动结果。

---

## 6. 全局 batch 为什么写死是 8

不是随便定的,**为了对齐论文**,而且这次有两个独立来源一致:

1. **论文 Table 6** 写 Batch Size = 8
2. **上游 `run_sft.sh`** 默认 `per_device=1, ga=1, NUM_GPUS=8` → 1×1×8 = 8

> Table 6 和代码在别处已经打过四次架(lr 1e-5 vs 2e-5、Epoch 2 vs 3.42、步数超出、样本数)。
> **batch=8 是少数两边对得上的**,所以这个数比其他几个可信。

**不能随便改的原因**:全局 batch 和 `lr=2e-5`、`max_steps` 是一套调好的组合,动了 batch
等于换了一组超参。而这个 ckpt 是两臂 RL 对比的**共同起点 π_ref**,起点换了后面所有结果都没法和论文对话。

如果不打算声称复现 SpaceTools,batch 想设多少设多少(配套重调 lr)。
**是"复现"这个目标把它钉死的,不是技术限制。**

---

## 7. ZeRO-2 vs ZeRO-3

**常见误解:ZeRO-3 通信更快。实际相反 —— ZeRO-3 通信更多。**

| | 分片什么 | 每步通信量 | 通信形态 |
|---|---|--:|---|
| **ZeRO-2** | 梯度 + 优化器态(参数每卡一份完整的) | **2Ψ** | 一次 reduce-scatter + 一次 all-gather |
| **ZeRO-3** | 再加上**参数本身** | **3Ψ**(1.5×) | 前向**每层** all-gather、反向**再来一遍**、加梯度 reduce-scatter |

(Ψ = 参数量)

ZeRO-3 更慢有两层原因:

1. **量更大** —— 1.5 倍
2. **形态更差** —— 拆成每层一次小通信,**延迟敏感**。A6000 之间走 PCIe(无 NVLink),
   延迟高带宽低,这种碎通信最吃亏

**ZeRO-3 换来的是显存**:参数也分片,能训放不下的大模型。
**它是"装不下时才用"的手段,不是优化手段。**

### 论文用的是哪个

**论文没写。** Table 6 只列 batch / lr / epoch / warmup / KL / 长度 / GPU 数,
**没有任何 DeepSpeed 或 ZeRO 字样**,正文也没提。z3 只出现在代码里:

```
run_sft.sh:210   deepspeed: examples/deepspeed/ds_z3_config.json
```

代码为什么选 z3,**只能推测,没有依据可引**:

1. LLaMA-Factory 全量微调的默认示例就是 z3 —— 很可能只是照搬
2. 他们在 8× A100-80 上跑,有 NVSwitch 600 GB/s,z3 的碎通信几乎不花钱
3. z3 最省显存最不会 OOM,是"总能跑通"的稳妥默认

> 3B 模型在 80 GB 卡上用 z2 也绰绰有余,所以**在他们的硬件上 z2/z3 大概率没区别**。
> 这更像是没做选择,而不是做了选择。

**我们的改动:z3 → z2**,因为 A6000 无 NVLink,3B 模型 48 GB 用 z2 装得下。

---

## 8. `use_reentrant_gc`

**是 gradient checkpointing 用哪个实现。**

Gradient checkpointing 本身 = 前向不存中间激活值,反向时重算 —— **用算力换显存**。
PyTorch 有两套实现:

| | 机制 | 状态 |
|---|---|---|
| `use_reentrant=True` | 老的,靠 autograd 重入 | PyTorch 已在弃用,未来默认翻成 False |
| `use_reentrant=False` | 新的,靠 saved-tensor hooks | 兼容性更好,官方推荐 |

**我们改了**(上游没设,LLaMA-Factory 默认 `True`):

```yaml
+ use_reentrant_gc: false
```

报告里给的理由是"更快"。**这条没有实测支撑**,是当时的判断,不是量出来的。

更实在的理由是 reentrant 版本的一个经典静默坑:**如果一个 checkpoint 段的输入全都不需要梯度,
反向会被直接跳过,不报错**。本配置正好 `freeze_vision_tower: true` +
`freeze_multi_modal_projector: true`,是最容易撞上的形状。
(实际 embedding 可训练,大概率不触发,但没必要留这个风险。)

**不改变数学结果** —— 两种实现算出的梯度相同,只是重算路径不同。

---

## 9. 这些配置都是哪里来的

三个来源,层层叠上去:

### ① 上游 `run_sft.sh`(绝大部分)

`sft_config.yaml` **不是仓库里的独立文件**,是脚本里一段 heredoc **每次运行现生成的**(第 238 行起)。
`finetuning_type` / `freeze_*` / `cutoff_len` / `lr` / `streaming` / `save_steps` 都写死在那儿。

### ② 论文 Table 6(只覆盖七八个数)

Batch 8、lr、Epoch、Warmup 0.1、cosine、Max Prompt/Response 8192、#GPU 8。
**其余全没写**(含 ZeRO stage、cutoff_len、image_max_pixels、eval 设置)。

### ③ 我们加的偏离(五条)

| 改动 | 从 → 到 | 性质 |
|---|---|---|
| `per_device` / `ga` | 写死 1/1 → 按卡数推导 | **修正** —— 原值在 4 卡上会静默变成全局 batch 4 |
| `deepspeed` | z3 → z2 | 性能,A6000 无 NVLink |
| `save_only_model` | false → true | 磁盘,代价是**不能续训** |
| `eval_steps` | 5 → 500 | 3000 步不必评 600 次 |
| `use_reentrant_gc` | 未设 → false | 兼容性 |

**四条都不改变训练的数学结果**,第一条是把已经错了的改对。

> **层次总结:论文定了少数几个关键超参,代码填满其余,我们只动了工程层。**

---

## 10. 框架分层

`run_sft.sh` **不是 LLaMA-Factory 提供的**。

**LLaMA-Factory(框架)提供:**

- `llamafactory.cli train` 入口
- `sft_config.yaml` 的**字段定义** —— `finetuning_type` / `freeze_vision_tower` /
  `cutoff_len` / `use_reentrant_gc` / `deepspeed` 这些 key 是框架 schema
- `examples/deepspeed/ds_z2_config.json` / `ds_z3_config.json`
- `data/dataset_info.json` 的注册机制

**SpaceTools 作者自己写的:**

- `scripts/spacetools/run_sft.sh` —— **这个目录在上游 LLaMA-Factory 里不存在**
- 下载 `siyich/spacetools-sft`、按 `toolshed_config.yaml` 重写 system prompt、
  v1 过滤 887 条 robot 样本
- 生成 `sft_config.yaml` 的那段 heredoc(**填什么值是他们定的**)
- Phase 3 的 ckpt 修补(删 `text_config`、置 `tie_word_embeddings`、拷 `preprocessor_config.json`)

**准确的说法:字段是框架的,值和流程是论文作者的。**

### 整体分层

| 层 | 是什么 | 干什么 |
|---|---|---|
| **HF Transformers / Trainer** | 底层 | 模型、优化器、`per_device` 这些参数 |
| **LLaMA-Factory** | SFT/微调框架 | 数据加载、多模态模板(`template: qwen2_vl`)、训练循环、接 DeepSpeed |
| **DeepSpeed** | 分布式后端 | ZeRO 分片 |
| **SpaceTools-SFT** | LLaMA-Factory 的 **fork** | 加 `run_sft.sh`:数据准备 + 配置 + ckpt 修补 |

RL 侧结构完全平行:

```
SFT:  LLaMA-Factory  ←fork←  SpaceTools-SFT
RL:   verl           ←fork←  SpaceTools-RL
```

---

## 11. 4× A6000 相比论文 8× A100,用到了什么分布式优化

**基本没有。我们没加优化,只做了一处硬件适配。**

| 技巧 | 论文 8×A100 | 我们 4×A6000 | 谁定的 |
|---|:-:|:-:|---|
| 数据并行 | ✓ | ✓ | 框架 |
| ZeRO 分片 | **z3** | **z2** | ← 唯一的改动 |
| Gradient checkpointing | ✓ | ✓ | LLaMA-Factory 默认开 |
| bf16 混合精度 | ✓ | ✓ | 配置 |
| flash-attn 2 | ✓ | ✓ | `flash_attn: auto` 默认,装了就用 |
| 冻结 vision tower + projector | ✓ | ✓ | 论文设计,可训参数 4.07B → 2.55B |
| streaming 数据集 | ✓ | ✓ | 配置 |
| 梯度累积 | ✗(ga=1) | ✗(4 卡也是 ga=1) | 推导出来的 |

**z3 → z2 也不是"优化",是换了个更适合 PCIe 的权衡** —— 用多占显存换掉那 1.5 倍碎通信。
在 NVSwitch 上这个换划不来,在我们这儿划得来。

### 真正能提速但没用的

(以下默认值均核自 `SpaceTools-SFT@b7ebbf3` 的 `hparams/model_args.py` / `data_args.py`)

| 选项 | 默认 | 能干什么 | 为什么没开 |
|---|---|---|---|
| `enable_liger_kernel` | `False` | **融合 cross-entropy,logits 张量根本不落地** —— 正好干掉那 8 GiB 尖峰 | 融合算子改变数值,π_ref 带偏离 |
| `packing` / `neat_packing` | `None` / `False` | 短样本拼进一条,`cutoff_len=8192` 的浪费能收回一大块 | 改变 attention mask 语义,**确定改变数学结果** |
| `disable_gradient_checkpointing` | `False` | 4 卡有 ~10 GiB 余量,关掉能快 20–30% | 余量吃不下 |

Liger 是最可惜的一个 —— 它针对的正是本项目的瓶颈。但这个 ckpt 是两臂 RL 的共同起点,
**为跑快 20% 引入数值偏离不划算**。

> **诚实的结论:我们是在用更少更弱的卡跑同一件事,靠 ga 和 z2 把它塞进去,不是靠优化跑得更快。**
> 4 卡 5–6 h vs 论文 8 卡 3–4 h,这个比例基本就是硬件差距本身。

---

## 附:本笔记里不确定的三处

写下来是为了避免它们日后被当成实测事实引用。

| 项 | 状态 | 怎么坐实 |
|---|---|---|
| **~6900 token** | 从 OOM 报错反解,依赖"失败分配即 logits"的假设;且是**最长**不是典型 | 用 ckpt 的 tokenizer 对 `train.json` 跑长度分布 |
| **`use_reentrant=False` 更快** | 判断,非实测。兼容性理由更硬 | 同配置各跑 30 步比 `s/it` |
| **`ga` 的速度惩罚在本负载上很小** | 推理(6900 token 已喂饱 GPU),`per_device=1+ga=2` 没跑过 | 4 卡上跑一次 `per_device=1, ga=2` 的 30 步 |

另外两个和本笔记相邻、已知但未决的:

- **`max_steps` 3000(代码)vs Epoch 2 → 1755(Table 6)** —— 本次跑的是 3000
- **`lr` 2e-5(代码)vs 1e-5(Table 6)** —— 本次跑的是 2e-5
