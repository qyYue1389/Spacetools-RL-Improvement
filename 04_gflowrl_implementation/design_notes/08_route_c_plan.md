# C 路计划:GFlowRL(C′ 臂)对 GRPO 的同起点对比

> **这份文档回答**:C 路具体怎么做、要多少机器和时间、能不能测出东西。
> **不回答**:要不要做——那是排期决定,依据见 §7 的两个决策点。
> 写于 2026-09-02。前置:`P7_DECISION.md`(路线)· `P7_GPU_RESULTS.md`(A′ 的答案)·
> `P7_ROUTE_C_GATE.md`(训练臂的闸门)。

---

## 0. 两条前置,决定了这份计划长什么样

**① A′ 已经答了:`Z_t` 按论文原样在我们这里不可用。** 原因不是批内 MC 的方差、
也不是常数选得不对,而是 `Eq.4` 与 `Eq.5/6` 的长度归一化不一致
(阈值 5e-4 nat/token,零训练下的框架差就已 1.5–1.9e-3)。
**所以 C 不能直接跑「论文原样的 GFlowRL vs GRPO」——那是在测一个已知有缺陷的实现。**

**② 闸门已经选出了臂:C′(`Eq.4` 与 `Eq.5/6` 两边都不做长度归一化)。**
`P7_ROUTE_C_GATE.md`:长度主导的代价在我们的长度分布上是温和的
(最长 10% 占 loss 的 20.7–24.2%,**组内仅 1.04–1.32×**);
非简并组上奖励主导漂移 **24×**;简并组的整组同值率 **0.0%**(A 是 30–49%)。
**代价是 C′ 的 clip 饱和率最高(77.9%)。**

> **所以 C 要问的那一句话必须改写:**
> ~~「GFlowRL 对 GRPO 能提多少分」~~
> **「把 GFlowRL 的分布匹配目标(C′ 形式)换进 SpaceTools 的 Step 4,
> `robospatial` 上的 accuracy 会不会动?」**
> 换目标函数动不了 P6 那 273 条工具错与工具集缺口(84.7%),
> 所以**问的只能是可动的那一块**,而它几乎全在 `robospatial`(§6.1)。

---

## 1. 四个 Step

`run_rl.sh` 的真实架构:**2 节点 × 8 卡** —— 一整个节点跑 Toolshed,
另一整个节点跑 FSDP 训练 + sglang(`trainer.nnodes=1`,`n_gpus_per_node=8`)。

### Step 3 · SFT(两臂共用,只跑一次)

| | |
|---|---|
| 数据 | `siyich/spacetools-sft` — **7463 文件 / 6.32 GB**(此前未出现在任何记录里) |
| 代码 | `ChicyChen/SpaceTools-SFT`(LLaMA-Factory fork,Apache 2.0,`b7ebbf32`) |
| 配置 | vision tower 冻结,只训 LLM;3000 步 |
| 论文耗时 | 8×A100-80G,3–4 h |
| **验收** | 论文 Table 4 有 Tool-SFT 的消融数字可作靶子,**但那张表的基数与 Table 2 不同,用之前必须核对口径** |

**不需要工具。** 所以 SFT 可以用满全部卡,是四个 Step 里唯一在当前机器上跑得动的。

### Step 4 · full-tool RL(两臂各跑 k 个 seed,主实验)

超参直接抄自 `run_rl.sh`:

    rollout.n = 5                    ← G=5,判据 i-b 的 3.17× 方差就是它
    train_batch_size = 64            → 5425 / 64 ≈ 85 步 = 1 epoch
    ppo_mini_batch_size = 64 · ppo_micro_batch_size_per_gpu = 2
    max_prompt_length = 8192 · max_response_length = 8192   ← eval 侧是 4096
    gpu_memory_utilization = 0.7 · max_num_seqs = 256
    use_kl_loss = True · kl_loss_coef = 0.01 · kl_loss_type = low_var_kl
    entropy_coeff = 0 · lr = 1e-6 · freeze_vision_model = true
    gradient_checkpointing = True · actor param/optimizer offload = False
    ref.param_offload = True · total_epochs = 1 · save_freq = 5 · test_freq = 5

**训练侧的工具配置是 eval 的 2.6 倍**,容易被忽略:

    RL:   roborefer 6×0.6 + vlm 2×0.6 + sam2 5×0.2 + depth 5×0.2
          + bbox 5×0.1 + vision_ops 8×0 + grasp 5×0.1  =  7.8 GPU   ← 塞满一个节点
    eval: 同样七个工具                                  =  3.0 GPU

原因是 batch 64 × n=5 = **每步 320 条 rollout 并发调工具**。actor 减少,
工具延迟就主导整步时间,吞吐赤字会比按卡数外推的更糟。

### ⚠ 两臂的受控差异:KL 项是一个必须处理的混淆

GRPO 臂带 `kl_loss_coef = 0.01` 的 KL-to-ref;而 GFlowRL 的分布匹配**本身就锚在
`π_ref` 上**(目标是 `π_ref·exp(β·r)`),论文 Table 9 的 KL 系数是 **0.0**。
**两臂若一个带 KL 一个不带,差的就不只是目标函数。**

**处置(二选一,必须预先写死):**
1. **两臂都关 KL**(`kl_loss_coef=0`)—— 干净,但 GRPO 臂偏离了 SpaceTools 的发布配置;
2. **两臂都保留 KL** —— GFlowRL 侧变成「分布匹配 + 额外 KL」,与论文配方不同。

**推荐 ①**,并在报告里写明 GRPO 臂因此不是 SpaceTools 的原配置。

### Step 5 · eval(每个 ckpt 一次)

复用 `tools/p4_run.sh` + `parse_dump.py --strict`,九个 key、greedy、
`model_dtype=bf16`(40 GB 上)、`gmu` 随数字一并报。**约 2 h 10 m / ckpt。**

---

## 2. 实现:改哪里

`P7_PRIOR_ART.md` 已经查清:**我们的 fork 里 registry 与 `ref_log_prob` 都已就位。**

    SpaceTools-RL @ f0742338 · verl 0.8.0.dev
      core_algos.py    register_policy_loss 已存在(11 个已注册 loss)
      dp_actor.py:528  use_kl_loss 时 ref_log_prob 已选进 micro-batch
      dp_actor.py:615  policy_loss_fn 调用点
      ray_trainer.py:1528  token_level_scores(原始 r,advantages 是组内归一化过的)

**最小路径,约十行:**
1. `actor.use_kl_loss=True` + `kl_loss_coef=0` —— 白拿 `ref_log_prob`,KL 乘 0 不生效
2. `select_keys.append("token_level_scores")` —— GFlowRL 要原始 `r`
3. `@register_policy_loss("gflowrl_cprime")` 写进 `core_algos.py`
4. L615 调用点多传 `ref_log_prob` 与 `token_level_scores`

**C′ 的实现要点:`Eq.4` 与 `Eq.5/6` 都用 `masked_sum`,不用 `masked_mean`。**
(FlowRL 的 `compute_flowrl_objective` 两种都有现成写法,但它建在 verl 0.4.0 上,
**模式可迁、API 不一定,不要照抄**。)

**必须带的守卫**(`P7_STEP4_RESULTS.md` §2):`response_mask` 有两个同名不同义的来源,
`ray_trainer.py:157` 的 fallback 是 `attention_mask[:,-L:]`(工具 token 全是 1)且**静默**。
多轮样本上断言 `response_mask.sum() < attention_mask[:,-L:].sum()`。

**β:不低于 2,起步用 8**(`P7_ROUTE_C_GATE.md` §4;
`P7_STEP23_RESULTS.md` §1.4 的「β≈1」在 C′ 下已作废)。

---

## 3. 机器:当前这台跑不了 Step 4 —— 不是慢,是装不下

工具实测(P6 开机,非估算):Molmo **34.6** + DepthPro **25.7** + RoboRefer **20.5** = **80.8 GB**。
在 **40 GB 卡**上:Molmo 独占一张(87%);DepthPro + RoboRefer = 46.2 > 40,要两张。
**工具吃掉 3 张,训练只剩 1 张。**

单卡训练态(4.066 B 参数,从 safetensors 头实读):

    fp32 master 15.1 + 梯度 15.1 + Adam m 15.1 + Adam v 15.1 = 60.6 GB  >>  40
    即使 optimizer_offload=True(Adam 下放 CPU):30.2 GB + 激活 + sglang 池
    (gmu=0.7 → 28 GB)                                      → 仍然爆

**`model_dtype=bf16` 不能用来救场** —— 那是 eval 专用的偏离 `[20]`,
**fp32 master 正是优化器步骤需要的东西**。

**三条出路:**

| 出路 | 代价 | 它还是不是 C |
|---|---|---|
| **加卡到 2 节点 × 8**(脚本的原假设) | 租机器 | **是**,原样的 C |
| **退到 LoRA** | 砍掉 base 的梯度与 Adam 状态那 45.4 GB | **不是** —— 全参 vs LoRA 是另一个实验,结论不能外推到论文配置 |
| 重度 CPU offload + 极小 batch | 我估计按天算,**未实测** | 是,但吞吐可能让 §5 的预算翻几倍 |

---

## 4. 统计设计——这才是决定 C 值不值的地方

### 4.1 效应量的天花板

`P6_REPORT.md` 的归因:换目标函数**动不了** 273 条工具错与工具集缺口(84.7%)。
可动的是 **推理错 19 + 3b 14 + 2a 11 = 44 条**(3b 归属有争议,只算推理错则是 19)。

    可动份额上界   19 – 44 / 2001 = 0.95 – 2.20 pp     且是「全部吃掉、一个不漏」

### 4.2 用总分做主指标 → 功效不够,**即使效应拉满**

McNemar 的功效由**不一致对**的数量决定。两侧检验 α=0.05、功效 80% 需要

    净差 Δ  ≥  2.80 · √D          D = 不一致对总数

两个不同训练的模型在同一批样本上的不一致率:同模型运行间是 6.9–10%
(`robospatial` 24/350、`blinkdepth` 13/124),换工具那次是 36.5%(145/397)。
取中间的 15%:

    D ≈ 0.15 × 2001 ≈ 300   →   需要 Δ ≥ 48
    而我们的**天花板**是 Δ = 44

> **所以以九个 benchmark 的总分为主指标,这个实验在效应拉满时仍然测不出来。**

### 4.3 修法:把 `robospatial` 预登记为主指标

P6 已经把可动份额定位得很干净 —— **44 条里约 42 条在 `robospatial`**:

    3b 坐标系/语义   14 条   全部 robospatial(§3.1)
    2a 该调没调      11 条   手工 3 条 + front/behind 8 条,全部 robospatial
    推理错           19 条   其中 15 条来自 robospatial Vacant、1 条 robospatial 手工

在 n=350 上重算:

    D ≈ 0.15 × 350 ≈ 52   →   需要 Δ ≥ 20        天花板 Δ = 42

**功效够了。** 这条不是统计技巧,是 P6 归因的直接后果:**效应本来就只可能出现在那里。**

### 4.4 预登记(跑之前写死)

- **主指标**:`robospatial` 350 条上的逐样本配对(McNemar),两臂同一批样本
- **次指标**:其余六个准确率 benchmark,**作为「没有把别处弄坏」的检查,不作为收益**
- **排除**:`boppose` / `bopgrasp`(`P7_DECISION.md` §0.2 —— `r` 对夹爪朝向不敏感)
- **seed**:**两臂各 ≥3**。RL 训练的种子间方差通常大于 eval 噪声,
  各 1 个 seed 什么都归因不了(§3.3 第 3 条:两侧的散布都要测)
- **解码**:greedy,`val_kwargs` 默认口径;`--strict` 守门;`gmu` 与 KV 池随数字一并报
- **失败也算结果**:「C′ 在 `robospatial` 上无效应」是对 P6 那条
  「换目标函数动不了 84.7%」的独立确认

---

## 5. 成本

| | 配置 | 时间 |
|---|---|---|
| 实现 + 合成不动点自检 | 零 GPU | 1–2 天 |
| Step 3 SFT(共用一次) | 8 卡 | 3–4 h(论文)· 4 卡估 8–12 h |
| **Step 4 × 2 臂 × 3 seed** | **2 节点 × 8 卡** | 每次 8–12 h → **48–72 h** |
| Step 5 eval × 6 ckpt(+SFT 基线) | 4 卡 | 每次 ~2 h 10 m → **13–15 h** |
| **合计** | | **约 4–5 天 16 卡时** |

数据:SFT 6.32 GB + RL 3.38 GB(5425 文件)。存储 ~100 GB。

---

## 6. 两个决策点

**决策点 1:租不租 2 节点 × 8 卡。**
当前 4×A100-40GB **跑不了 Step 4**(§3)。不租就只剩 LoRA(那是另一个实验)
或重度 offload(未实测)。**这一条不做技术判断能解决,是花钱决定。**

**决策点 2:接受不接受「主指标只报 `robospatial`」。**
以总分为主指标必然测不出来(§4.2)。把主指标收窄到 `robospatial`
在统计上是对的、在归因上有依据,但**报告里必须写明这是预登记的收窄,不是事后挑**。

---

## 7. 什么情况下不该做

- **若决策点 2 不接受** —— 那就没有一个有功效的主指标,不要跑。
- **若只能用 LoRA** —— 结论无法外推到论文的全参配置,而 C 的全部意义是同起点对比。
- **若 A′ 的结论还想再挖** —— C 花 4–5 天回答「分数动不动」,
  而 A′ 那条(归一化不一致 + 它的两种修法各有代价)是**算法侧的结果**,
  零 GPU 就能继续推进,而且**是这条线上唯一别人没做过的东西**
  (`P7_PRIOR_ART.md`:没有找到任何 GFlowNet 式目标用在多轮工具轨迹上的实现)。

---

## 8. 局限

- **§4.2 的 D ≈ 15% 是估计**,取自同模型运行间(6.9–10%)与换工具(36.5%)之间。
  真实值只有跑完才知道;若 D 更大,`robospatial` 的功效也会下降。
- **§5 的 4 卡 SFT 时间是估算**,没有实测。
- **Step 4 的两臂差异不止目标函数**(KL 那条,§1),必须按 ① 或 ② 显式处置。
- **闸门只测了起点**(`π_θ = π_old`);C′ 在真实训练动力学下的表现未验证。
- **`d` 是框架差代理,不是训练漂移**(`P7_ROUTE_C_GATE.md` §5)。
