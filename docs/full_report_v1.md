# 在 SpaceTools 的多轮工具轨迹上把 GRPO 换成 GFlowRL

**从环境搭建到逐样本归因的完整记录**

> 版本 v1 · 2026-09-19 · 待 review
> 2026-09-25 更正:① `front/behind` 的「C′ 少 3 题」是单次评测噪声(§0、§10.4、§10.6、§11、§12.2);
> ② 「C′ 让策略更主动、改点有害」方向反了 —— 是 GRPO 把起点的改点压住了,C′ 停在起点(§0、§10.6、§10.7、§11)
> 2026-09-30 补:§8.4 的「129 s 记账缺口」已查清 = old_log_prob 118.8 s + update_weights 10.0 s(§8.3、§8.4)
> 覆盖:环境构建 → SFT 训练 → SFT eval 闸门 → C′ 的推导与实现 → GFlowRL 训练 →
> GFlowRL eval → 与 P4/P5/P6 的逐样本对比。
> 材料:`training/`(SFT 与 RL 的 as-run 记录)、`eval/GFlowRL/`(九个 benchmark 的
> dump、轨迹、门禁、分析脚本)、`spacetools-repro/`(P2–P7 的全部报告与 P4 的 dump)。
> 每一节末尾的「材料」指向可复算的原始文件。

---

## 0. 一页看懂

### 做了什么

复现 SpaceTools(arXiv 2512.04069,CVPR 2026)的四步流水线,自己训出 SFT checkpoint,
然后把它的 **Step 4 强化学习从 GRPO 换成 GFlowRL(arXiv 2607.13394)的目标函数**,
训满 1 个 epoch(85 步),在论文 Table 2 的九个 benchmark 上评测,并与官方 checkpoint
做逐样本对比。

### 为什么这个交叉点值得做

GFlowRL 是**分布匹配**目标(按 `exp(βr)` 比例采样),GRPO 是**奖励最大化**目标(把质量压到
单一最优模式)。而工具调用天然存在多条等价有效的链路。P6 的离线归因给出了实测动机:

> 现有策略的工具编排已经高度坍塌——三个 RefSpatial 上 277/277 用完全相同的链路,
> `cvb2drelation` 650 里 615 相同。**这正是 GRPO 会做的事,也正是分布匹配目标声称能避免的。**

#### 什么是 tool-orchestration mode collapse(工具编排坍塌)

模型解一道题会产出一条完整轨迹 —— **一次采样就是一条 rollout**(训练时 `rollout.n=5`
就是每个 prompt 采 5 条)。把一条 rollout 里的工具调用按「**哪些工具、各调几次、用了几轮**」
压成一个字符串,就是它的**链路签名**(parsed 记录里的字段名 `chain_signature`):

```
roboreferx1@2t                                   调一次 roborefer,占 2 轮
roboreferx2@2t                                   一轮里并发调两次 roborefer
depth_estimatorx1+roboreferx2+vision_opsx2@3t    先测深度,再定位两个物体,再各取一次深度值,共 3 轮
```

> **链路签名 ≠ rollout。** rollout 是那条完整轨迹**本身**(含 `<think>`、每次工具调用的
> 参数、工具的真实返回、`<answer>`);链路签名是它的一个**指纹**,只保留「调了什么」这一维,
> 把措辞、坐标、推理过程全部丢掉。**两条内容完全不同的 rollout 可以有同一个链路签名** ——
> 正因为它丢掉了这些,才能用来量「编排有多单一」。

**Mode collapse(坍塌)= 所有样本都走同一条签名。** 这正是 GFlowRL 论文里那个词:
奖励最大化的目标「把概率质量压到单一高奖励模式上,导致 mode collapse、解法多样性丧失」。
用两个数衡量:

```
主链路覆盖率   出现最多的那条签名占了多少样本      100% = 完全 collapse
签名种类数     整个 benchmark 一共出现过几种签名   1 = 完全 collapse
```

**为什么这件事和算法选择有关:** 同一道题常常有**多条等价有效**的解法(先测深度再定位 /
先定位再测深度;用 `roborefer` 定位 / 用 `vlm` 定位)。奖励最大化的目标(GRPO)会把概率质量
**压到它见过的最好的那一条上**,其余等价链路的概率被压向 0;分布匹配的目标声称会
**按 `exp(βr)` 的比例把它们都保留下来**。所以「坍塌程度」正是这两类目标函数行为差异应该出现的地方——
GFlowRL 论文自己的多样性打分(GRPO 1.21 / FlowRL 2.64 / GFlowRL 3.93)量的就是同一件事。

**但这条动机只有一半站得住,这一点必须写在前面。** P6 的后续复核(`P7 路线决定` §2)指出:
RefSpatial 的题就是「指出那个东西在哪」,**正确链路本来就只有 `detect_one` 一次** ——
没有第二条等价有效的链可供保留,**那不是坍塌,是任务只有一条路**。同理 `boppose` 的
60/60 透传、深度题 95.6%/99.8% 的规则遵守。

**真正有代价的坍塌只有一处,而且 P6 把它定位干净了:** `robospatial` 的 `front/behind`
29 道题问的是深度序,`depth_estimator` 就在该 benchmark 的工具表里,**29/29 一次没调**
(全部 350 个样本 0 次)。这是唯一一处「存在已知有效的替代链、策略却从不走它、
且代价可量化(+8 个样本)」的地方 —— **分布匹配声称能保住的正是这种被压掉的模式。**
本报告 §10.4 与 §10.6 分别报两件事:整体坍塌度的变化,以及这一处真正的缺口有没有被动过。

> **2026-09-25 更正(P0:同机同池、三臂各评三次):这一处的「缺口」成立,「C′ 让它变差」不成立。**
> `front/behind` 29 题三次均值 SFT 17.3 · P4 19.0 · C′ 19.3,三臂没有差别。
> §10.4 / §10.6 的 72.4% → 62.1%(少 3 题)来自单次评测,是噪声;
> `analysis/p6_criteria_p4_vs_p7.py` 重跑仍得 21 / 18,原数没算错,只是单次读数分不开。
> `depth_estimator` 三次调用:SFT 0/0/0 · P4 2/2/1 · C′ 0/0/1,不是严格的 0。
> SFT 训练数据(`siyich/spacetools-sft`)里 RoboSpatial 的 yes/no 题 977 条,调 `depth_estimator` 的 0 条 ——
> 缺口来自 SFT 数据分布。上界按三次均值约 +10 题(29 − 19.3)。
> 材料:`07_gflowrl_improvement/p0_same_machine_reeval/p0_results.md` §6b · Design Doc P3。

论文自己的 limitation 也写着:「it remains unclear whether the same estimator-centric design
principle extends to broader **agentic or multimodal RL** settings」。检索确认**没有任何
GFlowNet 式目标用在多轮工具轨迹上的实现**。

### 结果

```
训练     85 步 · 21 h 57 min · 8× A40 · 无崩溃 / 无训练侧 OOM / 退出码 0
Eval     九个 benchmark · 2121 样本 · 零 OOM / 零截断 / 零顶轮数 / 零缺 <answer>
```

| | SFT 起点 | 官方 ckpt(P4 复现) | **C′ step85** | 论文 |
|---|--:|--:|--:|--:|
| RoboSpatial Overall | 61.00 ± 0.77 | 65.43–66.00 | **62.14**(两次均值) | 70.00 |
| RefSpatial 三项(加权) | 53.07 | 53.79 | **53.07** | 53.07 |
| BLINK Relative Depth | 未评 | 86.29–87.90 | **87.90** | 90.32 |
| CV-Bench 2D / 3D | 未评 | 94.62 / 96.50 | **94.62 / 96.50** | 94.92 / 96.00 |

**三条主要结论:**

1. **没有任何 benchmark 变好。** 对九个 benchmark 逐样本做 McNemar 精确检验,最小 p 是
   robospatial 的 0.092。名义上高一点的三个(blinkdepth +2、boppose +1、bopgrasp +2)
   都不显著,而且 boppose 的均值与逐样本方向相反、bopgrasp 的阈值计数与均值方向相反。
   **反方向同样不显著** —— 这次测量的分辨率不足以在任何单个 benchmark 上判定任一方向。

2. **P6 留给 P7 的那个动机,实测是负的读数:分布匹配没有减轻编排坍塌,反而更坍塌。**
   样本量大的四个 benchmark 上主链路覆盖率一致上升(cvb2d 94.6→98.3%、cvb3d 96.2→99.7%、
   blink 82.3→85.5%、robospatial 62.3→64.9%),链路签名种类还少了。

3. **C′ 唯一可测到的行为改变是有害的,而且能算到底。** 同一个行为在两个 benchmark 上各出现一次:
   - **RefSpatial 三项**(pointing 题,276 个可解析样本):原样透传 276/276 → **268/276**。
   - **RoboSpatial Vacant**(122 题,也是输出一个点,但 P6 判定它**不是纯透传**):
     「改动工具给的点」的样本从 21 个翻倍到 **42 个**,而改点正确率只有 14.3%、透传是 57.0%。

   两者都是**策略更频繁地去改工具给的坐标**。**RoboSpatial 的 14 题缺口可以一步不剩地
   归给这一个行为变化**(反事实验算见 §10.7)。

> **2026-09-25 更正(P0:同机同池三次复评,且第一次有 SFT 起点的 dump):这条的方向是反的 ——
> 不是 C′ 让策略更主动,是 GRPO 把起点就有的改点压住了,C′ 停在起点。**
> - RoboSpatial Vacant 改点(每次):SFT 起点 49.3 · P4 21.7 · C′ 44.0。RefSpatial 透传:SFT 272/276 ·
>   P4 276/276 · C′ 272/276(C′ 与起点逐个相同;本报告的 268 是单次读数)。
> - 缺口按三次均值是 Vacant −8.6 题(p = 0.0004)、VQA +3.7 题,整体 −5.0 题;「14 题」来自单次对单次。
> - 改点多发生在工具点本来就错的时候:改点样本上工具点命中只有 10–19%,改后的答案反而更准(19–28%)。
>   强制透传会让 Vacant **下降**:SFT 50.7 → 46.3 · P4 62.3 → 60.3 · C′ 53.7 → 49.3。
> - P4 领先主要来自工具调用质量:roborefer 最后一次返回的点落在 GT 内,SFT 46.3 · P4 60.3 · C′ 49.3 / 122
>   (三次均值;P4 vs C′ 稳定样本 13 : 4,p = 0.049)。GRPO 学到了更好的 `obj_name`,透传多是随之而来。
> - 所以「策略更主动等于变差」不成立;准确的说法是「C′ 没学到 GRPO 学到的工具调用,停在 SFT 起点」——
>   这是训练信号不足(退化组 70%、每步都被 clip)的症状,不是 C′ 目标函数的有害性质。
> 材料:`07_gflowrl_improvement/p0_same_machine_reeval/p0_results.md` §6 · `GFlowRL_improve/P1_prep/p1_monitor.py --upper` · Design Doc P1「主判据的来由」(原 P2)。

### 为什么这个负结果是可归因的,而不是「没跑好」

训练侧的三个读数把解释空间收窄到很小:

```
奖励退化组占比        中位 70.3%   (范围 57.8–85.9%)
actor/grad_norm      中位 180     而 grad_clip = 1.0
训练 score           无可靠上升趋势
```

在 70% 的组上组内奖励完全相同,此时 flow gap 的奖励项恒为 0、梯度全部来自漂移项——
**那时候 C′ 做的是向 `π_ref·exp(βr)/Z` 的 flow matching,不是在学奖励**。叠加每次更新
都被裁到只剩方向:**一个几乎没收到奖励信号的目标函数,没有理由让编排变多样。**

**关于「只训了 85 步」要说准确:** `total_epochs=1` 是上游作者写死的,论文的 70.0 就是
这么训出来的,**所以不能说我们训得比 SpaceTools 短**。但 GFlowRL 自己的证据来自
**30–400+ 步且 G=16** 的运行,我们是 **85 步 + G=5**(论文写明方差是 O(1/G),
且「especially when the group size is small」)。**准确的说法是:把一个在别处验证过的
算法放进了它没被验证过的 regime,步数与 G 两个维度同时不利。**

所以这个结果证伪的是「C′ 在这个 regime 下能做到」,**不是 C′ 的理论性质**。

### 对 P7 成败的正确期待(P6 事先写死的)

> **换目标函数动不了 273/322(84.7%)的工具错与工具集缺口。**

P6 把 322 个错题全部归因完毕:工具错 241(74.8%)、工具集缺口 32(9.9%)、推理错 19(5.9%)。
**工具错 : 推理错 ≈ 12.7 : 1。** P7 的作用域是那 44 条(13.7%)里的一部分,
**不该被期待去动这块 headroom。**

---

## 1. 背景:两篇论文与那一个交叉点

### 1.1 SpaceTools / DIRL(被复现的对象)

VLM 的定性视觉理解不错,但缺乏具身应用要的**度量级**空间推理(距离、位姿、抓取、遮挡)。
SpaceTools 的做法是让 VLM **在线调用视觉工具**,并把 RL 用两次(DIRL):

```
Teaching phase    单工具 IRL 专家 -> 2k 条 grounded 轨迹
                  + Claude Sonnet 4.5 挂全套工具、只留答对的 -> 6k 条
                  1:3 混成 8k 教学集 -> SFT(学会工具签名、输出格式、信息流)
Exploration phase 从 SFT 权重继续 IRL(GRPO + KL),开放全部工具
```

对话格式 `<think>` / `<tool_call>` / `<answer>`,多轮直到出答案或到上限。
**Toolshed** 是配套的系统贡献:Ray 之上的工具托管层,每个工具一组 actor、**Python 环境
隔离**(解决多 CV 模型依赖冲突)、变量(点云等)跨环境传递。

README 把论文两阶段拆成四步:**Step 1** point-tool RL → **Step 2** teacher 数据采集 →
**Step 3** SFT → **Step 4** full-tool RL。Step 1–2 的产出已随 SFT 数据集发布,
所以多数人**从 Step 3 开始**。

论文主结果:RoboSpatial Overall **70.0**、Pose 34.37、Grasp-SR 50.0。消融里最关键的一条:
**直接对所有任务所有工具做 IRL → 19.79,几乎不可学** —— 课程化不是锦上添花。

### 1.2 GFlowRL(被移植的算法)

GRPO/PPO 这类奖励最大化会把概率质量压到单一高奖励模式上。GFlowNet 提供另一条路:
**按奖励比例采样**。但已有的 GFlowNet 式 LLM 后训练(FlowRL)要学一个 prompt 条件的
配分函数 `Z_φ`,在真实后训练规模下会炸。

论文的根因诊断是**学习时程失配**:policy 是预训练好的、只需微调几百步;`Z_φ` 是随机初始化、
要从零学一个复杂量,也只有几百步。两条实证:把 `Z_φ` 换成随机采样性能不降反微升;
FlowRL 421 步里 **55 步梯度范数 ≥ 1e6**。

GFlowRL 的贡献就是**把它删掉**,换成**批内 Monte Carlo 估计**——反正 GRPO 式训练每个
prompt 本来就要采 G 条 rollout:

```
Eq. 4   Z_t(x) = (1/G) Σ_i [ β·r(x,y_i) + log π_ref(y_i|x) − log π_old(y_i|x) ]
Eq. 6   g_i    = sg[Z_t] + (1/|y_i|)·log( π_old / π_ref ) − β·r
Eq. 7   非对称 flow-gap clipping,g 裁到 [−ε_low, +ε_high],ε_low < ε_high
Eq. 8   L = (1/G) Σ_i w_i · ( g̃_i + (1/|y_i|)·log( π_θ / π_old ) )²
```

`Z_t` 以 **stop-gradient baseline** 的形式进入残差,不带梯度,于是辅助网络、它的优化器
状态、分布式同步全部消失。Appendix B:clip 未激活且不做长度归一化时,零损失的自洽不动点
满足 `π_θ ∝ π_ref·exp(βr)`。

结果:7B 数学 40.92(GRPO 32.48、FlowRL 35.63);14B Codeforces 2048 Elo;
**多样性打分 GRPO 1.21 / FlowRL 2.64 / GFlowRL 3.93**。

### 1.3 交叉点

| 维度 | SpaceTools / DIRL | GFlowRL |
|---|---|---|
| RL 算法 | GRPO + KL(奖励最大化) | GFlowNet TB 目标(分布匹配) |
| 核心痛点 | 多工具动作空间组合爆炸 | 学习式 `Z_φ` 梯度爆炸 |
| 解法 | **课程化**:先窄后宽 | **做减法**:删掉辅助网络 |

**工程上可行:两边都建在 verl 上**(SpaceTools 是 verl fork + Toolshed;GFlowRL 也基于 verl),
且 SpaceTools 已放出 full-tool RL 数据集(~5500)。把 GFlowRL 的 loss 换进 `run_rl.sh`
是一个规模可控的实验。

**材料:** `docs/paper_notes.md` · `records/P7_PRIOR_ART.md`

---

## 2. 我们要回答的问题,以及事先写死的判据

这一节的存在本身是方法学的一部分:**判据在看到任何结果之前就写下来了**,避免事后挑。

### 2.1 问题的两次改写

交接文档的原句是「GFlowRL 能不能在多轮工具轨迹上稳定训练?」**这句话有两个毛病:**

- **为「是」几乎白送。**「稳定」的含义来自 GFlowRL 对 FlowRL 的对比,而论文自己诊断的
  根因是 `Z_φ` 的学习时程失配。**没有 `Z_φ`,就没有那个失败模式** —— 梯度范数正常
  说明不了什么。
- **为「否」不可归因。** `microsoft/gflowrl` 是 404,loss 要照 Eq. 4–8 自己写。那么
  「训练不稳定」是算法不推广,还是我们 Eq. 8 写错了?

**改写为 A′:**

> 在 SpaceTools 的多轮工具轨迹上,GFlowRL 的批内 Monte Carlo 估计量 `Z_t`
> 还是不是一个可用的 log-partition 估计?

**关键的职责分离:** 「我们的 loss 写得对不对」由**合成不动点检验单独守住**
(构造 `π_θ = π_ref·exp(βr)/Z` 查 loss ≈ 0),不和 A′ 混在一起。
否则又回到不可归因的否定结果。

后来用户把目标表述为「尝试应用 GFlowRL 看能不能提高模型 accuracy」——**这对应的是另一条路
(C:全量同起点对比)**,成本 3–4 天机时。两者的区别写明了,最终执行的是 C 的简化版
(单臂 + 与已有基线对照),而 A′ 的离线诊断先行,**正是通向那个问题的最省路径**。

### 2.2 预登记判据

| # | 判据 | 怎么算通过 |
|---|---|---|
| i | 报 `Var(Z_t)/(β·r)²` 的**分布**,不报单个数 | 两侧散布都要有 |
| i-b | 同一批内常数的多种取法并排报 | 取决于漂移的**分布形状** |
| ii | 长度归一化 on/off 各跑一遍 | 比较**不动点位置**,不比较分数 |
| iii | 简并组的残差 | 要与「把 `π_old` 换成 `π_ref` 的零信息对照」比 |

### 2.3 明确排除在作用域之外

- **`bopgrasp` / `boppose`。** P6 §4.2:奖励(NCE)对夹爪朝向不敏感——朝向中位错 **63.9°**
  的回退答案 NCE **1.07**,优于工具真算出抓取的 **1.38**。**奖励规格错的地方,任何以 `r`
  为目标的算法都无话可说。** 而且分布匹配比奖励最大化更糟一层:GRPO 只保住最高的那个坏模式,
  **分布匹配会按 `exp(βr)` 比例把那个坏模式一起养着。**
- **换工具。** 不装第二个深度模型、不装更强的 pointing 模型。理由是范围:
  **P7 检验的是换目标函数,换工具是另一条正交路径**,混在一起不可归因。

**材料:** `04_gflowrl_implementation/design_notes/01_route_decision.md` · `records/P7_HANDOFF.md` · `records/P6_REPORT.md`

---

## 3. 环境:为什么这一步占了大头

这个项目里**环境构建的工作量超过训练本身**。原因是系统结构:被评的不是一个模型,是
**一个模型 + 七个工具**,而七个工具的依赖互相冲突。

### 3.1 依赖冲突的实际形状

```
spacetools-rl              训练主环境    torch 2.9.1 · verl 0.8.0.dev · sglang 0.5.6 · numpy 1.26.4
spacetools-tool-roborefer  RoboRefer-8B
spacetools-tool-vlm        Molmo-7B-D · SAM2 · DepthPro     torch 2.5.1
spacetools-tool-bbox       3D bbox · vision_ops             numpy 2.4.6
spacetools-tool-graspgen   GraspGen                         torch 2.3.1
```

**torch 2.3.1 / 2.5.1 / 2.9.1 并存,numpy 1.26 与 2.x 并存。** Toolshed 通过 Ray 的
`runtime_env={"conda": <env>}` 把每个工具起在对应环境里,变量(ndarray)跨环境传递
由 Ray 序列化。**这不是可以省掉的复杂度,是论文的系统贡献本身。**

### 3.2 SFT 环境(自建,4.0 GB 可迁移包)

```
python 3.11 · torch 2.9.1+cu128 · torchvision 0.24.1 · torchaudio 2.9.1
transformers 4.57.1(二次钉死并硬断言)· flash-attn 2.8.3.post1(自编,仅 sm_80)
deepspeed 0.19.6 · llamafactory 0.9.5.dev0(editable)
```

相对上游 `setup_envs.sh` 的五条偏离,**四条是修正上游 bug 或只影响构建过程**:

| 偏离 | 上游 | 性质 |
|---|---|---|
| `FLASH_ATTN_CUDA_ARCHS=80` | `80;90;100;120` | 编译量降到 1/4。**代价:无 PTX 回退,H100/Blackwell 需重编** |
| `transformers` 二次钉死 + 断言 | 无版本约束 | **修正** —— 否则会被传递依赖顶成 5.x 且不报错,产出与论文口径不同的 π_ref |
| 钉死 `torchvision` / `torchaudio` | 无版本约束 | **修正** —— 否则 ABI 崩溃,训练起不来 |
| 增补 `ninja` | 未装 | 缺它 flash-attn 退回串行编译,`MAX_JOBS` 失效 |
| `MAX_JOBS=12` | 自适应(在容器里是坏的) | 纯并发控制 |

**四项都不改变训练的数学结果。**

三个必须记住的容器坑:

1. **容器内存上限是 cgroup 的 93.1 GiB,不是 `free` 显示的 503 GB。** 编译并发按 `free`
   算会被 OOM killer 杀掉容器。改成按 cgroup 算并硬性钳位,再加一个盯 cgroup `anon` 的
   **编译期看门狗**(超 70% 按**进程树**杀 —— ninja 给每个子进程 `setpgid(0,0)`,
   只杀进程组会漏掉在跑的编译进程)。
2. **网络卷上 conda 慢到不可用**(~200 vs >20,000 files/s)。所以装在容器盘
   (`/opt/conda-st`)并立刻 `tar` 到持久卷;容器重启后 2 分钟恢复。
3. **flash-attn 先 `pip wheel` 落持久卷再装** —— 编译成功一次永久有效,重装从 ~17 分钟
   变 10 秒。

验收不是 `import` 通过就算:`check_arch.sh` 扫描所有含 CUDA 的 `.so` 查架构覆盖(0 个不合格),
并在 GPU 上**真算了一次** flash-attn kernel —— 同时实测确认「为 sm_80 编的 cubin 在
sm_86 上能跑」。

### 3.3 Eval / RL 环境(22 GB 包,四项验收)

RL 与 eval 共用一套环境,打包成 `qzpm55555/spacetools-eval-env`(21 GB,6 个切片)。
**必须还原到 `/opt/conda-st` + `/opt/spacetools`** —— conda 环境里烧死了绝对路径,
换路径全线崩。

`VERIFY.sh` 的四项(全过才能跑):

```
五环境 import 闸门     全通
架构实扫               4619 个 .so · 108 个含 cubin · 自编扩展缺 sm_8x 共 0 个
七工具冒烟             7/7,真加载权重真出结果
Ray 跨环境链           5 工具 · 失败环数 0 · ndarray 双向传递正确
```

跨环境链这一项测的是真实形态:`sam2`(numpy 1.26.4)产出 `$segmentation_mask` →
`bounding_box`(numpy 2.4.6)吃它;`depth_estimator` 产出 `$depth_map` → `vision_ops` 吃它。

**三个数字(4619 / 108 / 0)与打包时的基线逐字对上**,说明这套环境在新机器上可复现。

### 3.4 三条环境侧的教训

- **交付的机器可能根本没装驱动。** 一台机器 `lspci` 能看到 4 张 GA102GL,但 `/dev/nvidia*`
  不存在、`dpkg -l | grep nvidia` 是 0 个包。**所以第一条检查应该是 `nvidia-smi -L`
  而不是 `nvidia-smi`** —— 后者不存在时报 `command not found`,容易被当成 PATH 问题。
- **五个 conda 环境的 Python 补丁版本必须一致。** 实测 `spacetools-tool-vlm` 与
  `-tool-bbox` 是 3.11.0,其余是 3.11.16;Ray 的 `check_version_info` 默认比完整版本串,
  `3.11.0 ≠ 3.11.16` 抛 RuntimeError,**那两个环境里的 8 个 actor 全部入不了集群**。
  这个故障原来的三项验收全都抓不到(跨环境链在未修状态下是**通过**的),
  只在 eval / RL 规模下暴露 —— RL 阶段这两个环境里有 23 个 actor。
  修法随 `POSTRESTORE.sh` 推到 HF,`VERIFY.sh` 加了第 0 项前置检查。
- **验收脚本本身会给假绿灯。** `28_chain.sh` 在链测试失败时打印「✗ 链没通」的同时
  `exit 0`,汇总据此打「✓ 三项全过」——**这一项根本没测,却报了通过。**
  **消费退出码的一方必须同时检查输出里的成功标记。**

**材料:** `training/SFT/env/2x_A6000_REPORT.md` · `00_environment/eval_rl_env/BUILD_GUIDE.md` ·
`00_environment/eval_rl_env/env_package_revision_20260912.md`

---

## 4. Step 3:训练 SFT checkpoint

### 4.1 为什么必须自己训

官方只发布了 **RL 之后**的 checkpoint(`siyich/spacetools-ckpt`,实查:只有 `main` 一个
分支、无 tag、README 标 `reinforcement-learning`)。**没有 SFT checkpoint。**
而 SFT ckpt 是两臂 RL 对比的**共同起点 π_ref** —— 没有它,任何 RL 结果都无法归因。

**好消息:SFT 的数据是公开的**(`siyich/spacetools-sft`,7463 文件 / 6.32 GB),
代码 Apache 2.0。**重跑 Step 3 的卡点是算力,不是资料。**

### 4.2 配置的三层来源

```
① 上游 run_sft.sh(绝大部分)   sft_config.yaml 不是仓库里的文件,是脚本里一段
                               heredoc 每次运行现生成的。finetuning_type / freeze_* /
                               cutoff_len / lr / streaming / save_steps 都写死在那儿
② 论文 Table 6(只覆盖七八个数) Batch 8 · lr · Epoch · Warmup 0.1 · cosine ·
                               Max Prompt/Response 8192 · #GPU 8。其余全没写
③ 我们加的偏离(五条)          见下
```

| 改动 | 从 → 到 | 性质 |
|---|---|---|
| `per_device` / `ga` | 写死 1/1 → 按卡数推导 | **修正** —— 原值在 4 卡上会静默变成全局 batch 4 |
| `deepspeed` | z3 → z2 | 性能,A6000 无 NVLink |
| `save_only_model` | false → true | 磁盘,代价是**不能续训** |
| `eval_steps` | 5 → 500 | 3000 步不必评 600 次 |
| `use_reentrant_gc` | 未设 → false | 兼容性 |

**四条都不改变训练的数学结果,第一条是把已经错了的改对。**

> **`per_device` / `ga` 是什么。** 都是 HF Trainer(LLaMA-Factory 用的就是它)的参数,
> 全名 `per_device_train_batch_size` 和 `gradient_accumulation_steps`:
>
> - **`per_device`** —— **一张卡一次前向里塞几条样本**。「device」就是 GPU,不是机器。
>   它只决定显存占用和速度,**不决定优化器看到多少样本**。
> - **`ga`** —— 攒几次前向再更新一次权重。攒的过程不 `optimizer.step()`,只累加梯度。
>
> 三者的关系是一个恒等式:
>
> ```
> 全局 batch = per_device × ga × 卡数
>              ↑            ↑     ↑
>            每卡每次     攒几次  几张卡
> ```
>
> **只有「全局 batch」影响训练结果**,前三个数怎么拆都行。所以论文的 `per_device=1`
> 是 8 卡除出来的(1 × 1 × 8 = 8),不是一个设计选择 —— 原样搬到 4 卡机上就变成
> 1 × 1 × 4 = **4**,全局 batch 悄悄减半。这就是上表第一条要修的东西。
>
> 我们最后跑的是 **2 × 1 × 4 = 8** ✓。

**z3 → z2 不是优化,是换了个更适合 PCIe 的权衡。** 常见误解是 ZeRO-3 通信更快,实际相反:
z2 每步 2Ψ、z3 是 3Ψ(1.5×),而且 z3 拆成**每层一次小通信**、延迟敏感 —— A6000 之间
走 PCIe(无 NVLink)最吃亏。**ZeRO-3 换来的是显存,是「装不下时才用」的手段。**
论文没写用哪个,代码里是 z3,大概率只是照搬 LLaMA-Factory 默认示例(他们在 8× A100-80 +
NVSwitch 上,z2/z3 没区别)。

**全局 batch 写死是 8**,两个独立来源一致(论文 Table 6 + 上游 `run_sft.sh` 默认
1×1×8)。这是**「复现」这个目标把它钉死的,不是技术限制** —— 动了 batch 等于换了一组
超参,而这个 ckpt 是后续所有对比的共同起点。

### 4.3 显存账:为什么 `per_device` 上限锁在 2

```
显存 = 固定部分(参数 bf16 + 梯度 + 优化器态,ZeRO-2 已分片)+ 每条样本的部分
真正的元凶是 logits:per_device × 序列长度 × 词表 × 2 字节
Qwen2.5-VL 词表 151,936 -> 每条样本每 token 0.3 MB
```

| per_device | logits 尖峰 |
|--:|--:|
| 1 | ~2 GiB |
| 2 | ~4 GiB |
| **4** | **~8 GiB** ← 实测撑爆的就是它 |

A6000 上 `per_device=4` 实测:**稳态 39.8 GiB + 尖峰 8.4 GiB ≈ 48 GiB**,正好等于卡容量。
**它是瞬时尖峰不是稳态**,`nvidia-smi` 隔秒采样很可能看不到,但 OOM 是它触发的。

> ⚠ 序列长度 ~6900 token 这个数是**从 OOM 报错反解的**,依赖「失败分配即 logits」的假设,
> 而且是那 30 步里**最长**的那个 batch,不是典型值。用来算 OOM 余量合适,不能当典型长度引用。

### 4.4 as-run(4× A6000,3000 步)

```
机器      Vast.ai · 4× RTX A6000(49140 MiB · sm_86)· 驱动 570.181 · 377 GiB 内存
启动      GPU 4 · per_device=2 · ga=1 · 全局 batch 8 ✓(从 HF Trainer 产出独立反推)
数据      7907 条原始 -> 过滤 887 条 robot 工具样本 -> 7020 条进训练
耗时      27929.7 s = 7 h 45 min · 均值 9.31 s/it
loss      train 0.8621(step 5)-> 0.0625(step 3000)· 全程均值 0.19167
eval_loss 0.1680 / 0.1064 / 0.0975 / 0.0684 / 0.0559 / 0.0452   六点单调下降,无一回升
```

数据口径六项全过:`<tools>` 块长度 8595 ✓ · sha256 前缀一致 ✓ · 工具数 11 ✓ ·
system prompt 7020 条全一致 ✓ · robot 样本残留 0 ✓。

**一处险情:显存余量不是均匀的。**

| GPU | 峰值 | 占比 | 余量 |
|---|--:|--:|--:|
| 0 | 36830 MiB | 75% | 12.0 GiB |
| 1 | 37090 MiB | 76% | 11.8 GiB |
| **2** | **47730 MiB** | **97%** | **1.4 GiB** ⚠ |
| 3 | 36950 MiB | 75% | 11.9 GiB |

前序文档假设 4 卡有 ~10 GiB 均匀余量,实测最坏 rank 只剩 1.4 GiB。
**「4 卡很宽松」这个判断是错的。**

另外 **5–6 h 的预期错了,真实 7.75 h** —— 原因已查清但不可操作。
`val_size` 只有 20 条,`eval_loss` 单点噪声大,只看趋势。

### 4.5 没用的加速手段(记录一下,避免重复讨论)

| 选项 | 能干什么 | 为什么没开 |
|---|---|---|
| `enable_liger_kernel` | **融合 cross-entropy,logits 张量根本不落地** —— 正好干掉那 8 GiB 尖峰 | 融合算子改变数值,π_ref 带偏离 |
| `packing` / `neat_packing` | 短样本拼进一条,`cutoff_len=8192` 的浪费能收回 | 改变 attention mask 语义,**确定改变数学结果** |
| 关掉 gradient checkpointing | 能快 20–30% | 余量吃不下 |

**Liger 是最可惜的一个** —— 它针对的正是本项目的瓶颈。但这个 ckpt 是后续对比的共同起点,
**为跑快 20% 引入数值偏离不划算。**

> **诚实的结论:我们是在用更少更弱的卡跑同一件事,靠 ga 和 z2 把它塞进去,不是靠优化跑得更快。**
> 4 卡 7.75 h vs 论文 8 卡 3–4 h,这个比例基本就是硬件差距本身。

**材料:** `training/SFT/env/training_report/REPORT.md` · `docs/sft_training_notes.md` ·
`training/SFT/model_rl_start/`(ckpt 本体)

---

## 5. SFT eval:决定要不要继续训 RL 的闸门

### 5.1 为什么先评再训

Step 4 要 3–4 天机时。**如果 SFT ckpt 本身不合格,那几天全是浪费。**
所以先花 33 分钟评四个 key,按事先定好的 go/no-go 判断:

| 结果 | 结论 |
|---|---|
| RoboSpatial ≥ 60 **且** RefSpatial ≥ 48 | checkpoint 可用,进 RL |
| RoboSpatial < 55 **或** RefSpatial < 40 | 停下来查,不进 RL |
| 中间带 | 先确认运行健康,再和论文表逐项对齐 |

选这四个 key 的原因:覆盖论文主表的空间推理能力,且都用不到 `grasp_generator`。
**明确排除 `boppose` / `bopgrasp`** —— 前者 metric 映射复杂、后者会把 `grasp_generator`
拉进来增加变量。

### 5.2 判读顺序不能颠倒

> **先数 OOM → 再查工具静默错误 → 最后才看分数。**

- **OOM 的样本会算进分母**,分数被稀释成一个「偏低但看着合理」的数字。
- **工具出错时不抛异常**,而是把错误信息包成正常的 `ToolResult` 返回;router 也会把 actor
  里的异常吞掉。表现是这样的:

```
CALL_OK   depth_estimator -> ModuleNotFoundError: No module named 'numpy._core.numeric'
                             返回类型: ToolResult
```

  **注意那是 `CALL_OK`。调用方拿不到任何异常。** 如果某个工具在整个 eval 里一直返回报错
  文本,模型就一直拿着垃圾输入推理,eval 会跑完、不报错、给出一个看着正常的分数——
  **然后你会去怀疑 checkpoint。**

> 顺带一条工程细节:那个 grep 用的是**子串**匹配(`Error:`),因为 `TypeError:` 里含
> `Error:`;换成更「严谨」的前缀匹配反而会漏掉整类 Python 内建异常。

### 5.3 结果

627 个样本,32 分 41 秒。

| | n | run 1 | run 2 | 恒对 | 恒错 | 翻转 | 期望 ± sd |
|---|--:|--:|--:|--:|--:|--:|--:|
| RoboSpatial · Overall | 350 | 61.71% (216) | 60.29% (211) | 199 | 122 | 29 | **61.00% ± 0.77 pp** |
| RoboSpatial · VQA | 228 | 70.61% | 71.93% | 152 | 55 | 21 | 71.27% |
| RoboSpatial · Vacant | 122 | 45.08% | 38.52% | 47 | 67 | 8 | 41.80% |
| RefSpatial · 三项 | 277 | 52.58 简单 / 53.07 加权 | — | | | | |

运行健康七项全为零:OOM 0 · 样本级工具错误响应 0/627 · 每个样本都有工具调用 627/627 ·
缺 `<answer>` 0 · 轮数耗尽 0 · 截断的工具响应 0 · 畸形 tool call 0。

**RoboSpatial 必须按区间报。** 判据 60% = 210/350 **正好落在翻转带内**(199 恒对 ~ 228 上界):
单次读数在线上还是线下,由那 29 个翻转样本这一次怎么落决定,**不由 checkpoint 决定**。
要够 210 需要翻转中 11 个落对,单次落对率 ≈ 0.5,算下来**一次运行读到线下的概率约 6.8%**。

> **这一条的实践后果:只跑一次、运气差,就会读到 59.x%,把一个可用的 checkpoint 判成不可用。**
> 后续闸门应写成「期望值 ≥ X 且恒对下界 ≥ Y」,或直接规定「取 N 次中位数」。

### 5.4 比分数更有价值的是失败模式

| Table 2 行 | n | 论文 | 官方 ckpt(P4 复现) | **本 SFT ckpt** | 少答对几题 |
|---|--:|--:|--:|--:|--:|
| RoboSpatial · VQA | 228 | 79.38 | 73.25–73.68 | 71.27 | 4–6 题 |
| RoboSpatial · Vacant | 122 | 52.46 | 50.82–51.64 | 41.80 | 11–12 题 |
| RoboSpatial · Overall | 350 | 70.00 | 65.43–66.00 | 61.00 | 15–17 题 |
| RefSpatial · 三项 | 277 | 53.07 | 53.35 / 53.79 | 52.58 / 53.07 | **2 题** |

**换算成题数是必要的** —— 不同 benchmark 的 n 差一个量级,pp 不可直接比较。
RefSpatial 的 −0.77 pp 是 **277 题里差 2 题**(标准误约 3.0 pp,差距是 0.26 个标准误);
RoboSpatial 的 −4 pp 是 **350 题里差 14 题**,那才是真差距。

**实际被调用的工具只有三个:**

```
roborefer          627 / 627 样本
depth_estimator      1 次
vision_ops           1 次
sam2 / vlm / bounding_box / grasp_generator    0 次
```

**关键反例(index=128,GT `No`,答错):** 问「杯子能放到音箱前面吗」

```
roborefer.detect_one "cup"     -> [(0.78, 0.494)]
roborefer.detect_one "speaker" -> [(0.154, 0.491)]
<think> The question asks if the cup can fit "in front of" the speaker.
        This means: is there space between the speaker and the camera/viewer?
        ... There appears to be adequate desk space in front of the speaker ... </think>
答:Yes        真值:No
```

**它自己把问题正确地翻译成了深度问题**(「音箱和相机之间有没有空间」),
**然后用两个 y 值几乎相同的 2D 坐标去回答它**(0.494 vs 0.491)。
`depth_estimator` 和 `vision_ops` 就在它的 schema 里,组合起来正好能回答这个问题。它没有调。

**唯一调了完整链路的样本(index=176)算对了:**

```
depth_estimator.estimate_depth_with_pointcloud   (tool-vlm, numpy 1.26.4)
roborefer.detect_one "bed" / "table"
vision_ops.index_at $depth_map @ 两个点              (tool-bbox, numpy 2.4.6)
   -> 2.74 m vs 2.94 m,拿到真实米数再比较,答对
```

> **这两个样本放在一起就是 SFT eval 最重要的一条发现:能力是有的,触发率是 1/350。**

**VQA 混淆矩阵:学会了先验,没学会判别。**

```
                 模型答 no    模型答 yes   |  这一行答对的比例
  真值 no             31           32      |   49.21%   (共 63 题)
  真值 yes            32          133      |   80.61%   (共 165 题)
  ────────────────────────────────────────
  模型答案的分布    no 63       yes 165
  真值的分布        no 63       yes 165     <- 与上一行完全相同
```

**预测的边缘分布和真值完全相同(165/63),但 `gt_no` 恰好是掷硬币(49.21%)。**
模型学到了「多久该说 yes」这个先验,没学到「什么时候该说 no」这个判别信号——
SFT 的似然目标可以只靠匹配边缘分布就拿到不错的 loss。

> **一条事先写下的负面预期:** `gt_no` 的掷硬币水平在论文的 RL 之后**依然存在**
> (官方 ckpt 47.62%)。**不要把它列进 RL 的预期收益。**

### 5.5 四条闸门与结论

| 闸门 | 本次位置 |
|---|---|
| ① 工具调用的**格式**必须已经稳定(硬性) | §5.3 七项运行健康计数全为 0 —— 畸形 tool call 0、缺 `<answer>` 0、截断的工具响应 0、轮数耗尽 0 |
| ② 至少一条**多工具链路**可用,哪怕触发率很低 | 可用,触发率 1/350 |
| ③ 目标 benchmark 上**不能已经饱和** | 未饱和 —— RoboSpatial Overall 61.00,离论文的 70.00 还有 9 pp 可涨 |
| ④ 失败模式要是**决策型**的,不是**能力型**的 | 决策型(该调没调),已定位到样本 |

**四条闸门全过、且空间 ≥ 5 pp,就值得上 RL。** 判据两项都过,进 Step 4。

而 **RefSpatial 是反例**:SFT 已经和 RL 后的官方 ckpt 平手(277 题差 2 题),
链路 277/277 无变化。**在这种任务上做 RL 是浪费算力** —— 没有可探索的决策空间。

**材料:** `03_sft_eval/sft_eval_report.md` · `03_sft_eval/sft_eval_results.md`

---

## 6. C′:GFlowRL 的哪个变体,以及为什么

**不能照搬论文配置。** 五处失配,每一处都被单独量过。

### 6.1 五处失配

**① `G = 16` → 我们 `rollout.n = 5`。** Remark B.2 写明估计量方差是 **O(1/G)**;
Appendix A 的局限第一句就是「in-batch MC estimate of log Z can have higher variance
in principle, **especially when the group size is small**」。
**16 → 5 是 3.2× 的方差,我们正好站在最不利的一侧。**(改 `rollout.n` 会让两臂不可比,
所以保持 5。)

**② Eq. 4 与 Eq. 5/6 的长度归一化是不对称的 —— 这是论文原样,不是笔误。**

```
Eq. 4   Z_t = (1/G) Σ_i ( β·r + log π_ref − log π_old )          无 1/|y|
Eq. 6   g_i = sg[Z_t] + (1/|y_i|)·log( π_old / π_ref ) − β·r      有 1/|y|
```

Appendix B 的 Eq. 10 同样没有归一化,可确认不是排版问题;**Prop. B.1 是在「不做长度归一化」
的前提下证的**。Remark B.4 把归一化的后果单独写出来:

```
π_θ*(y|x) = π_ref(y|x) · exp( |y| · ( β·r − Z_t ) )
```

**有效逆温度是 `|y|·β`,不是 `β`。** 长度不一致时 `exp(−|y|·Z_t)` 不再是跨序列的公共
归一化常数,不动点被长度扭曲。

**这个不对称的实际后果,合成实验量化了:** 漂移在组内成为**公共平移**,超过 clip 半宽
就把整组 `g̃` 削成同值 —— **奖励贡献被完全抹掉**,而阈值只有 **≈5e-4 nat/token**。

**③ `|y_i|` 必须是 `response_mask.sum()`,不能是原始 response 长度。**
多轮轨迹的 response 里含 `<tool_response>`,单次上限 2048,一个样本可有 4–5 次调用。
**若按原始长度归一化,有效温度 `L·β` 就由工具输出的啰嗦程度决定。** 实测倍数:
`robospatial` **1.14×** · `blinkdepth` **1.41×** · `boppose` **2.40×** ——
**而且按 benchmark 系统性不同**,等于给不同任务施加了不同温度。

> 更隐蔽的一层:verl 里 `response_mask` 这个名字有**两种相反语义** —— agent loop 给的是
> 「策略 token」,而 `ray_trainer.py:157` 的 fallback `compute_response_mask()` 是
> `attention_mask[:,-L:]`,**工具 token 全是 1**,且 fallback 是**静默**的。必须加运行时守卫。

**④ 组内奖励简并。** `p6/passk` 恰好是 G=5 的真实 rollout 组:

```
奖励在 G=5 内完全相同的 prompt 比例
  robospatial   199/350 = 56.9%
  blinkdepth     94/124 = 75.8%
  boppose        32/60  = 53.3%
```

对 **GRPO**:advantage 归一化后为 0,**整个 prompt 不产生梯度**。
对 **GFlowRL**:Eq. 8 是逐轨迹的残差平方,即使 `r` 全同,残差一般不为零,**它仍然出梯度**。

> ⚠ **但这不是「GFlowRL 已发表配置」的性质**:论文 Table 9 的 rollout 设置写着
> `Filter groups: Accuracy-based` —— **奖励一致的组被整组丢掉**,所以在他们的配方下
> 这个问题根本不出现。对我们是两难:照搬 filter,会在 G=5 上丢掉 53%–76% 的 prompt;
> 不 filter,是**对论文配方的一处主动偏离,必须自己论证**。
> **这条从「我们的一个优势」降级为「一个必须做且必须论证的设计决定」。**

**⑤ 起点上 flow gap 只取三个值。** 两个二值 benchmark 上非简并 rollout **100% 饱和** ——
即 g 只落在 `{−ε_low, 0, +ε_high}`。要不饱和需要 `β ≲ 1.4`。

### 6.2 三个变体与闸门

Eq. 4 和 Eq. 5/6 各自可以做 / 不做长度归一化:

| 变体 | Eq.4 | Eq.6 | 闸门在真实数据上的结论 |
|---|---|---|---|
| `paper` | 不归一 | 归一 | 漂移项退化成组内公共平移,在 **30–49%** 的组上把奖励抹平 |
| `normalized` | 归一 | 归一 | 有不动点,但那个不动点是 \|y\|≈334 处的**点质量**(纯奖励最大化) |
| **`cprime`** | **不归一** | **不归一** | **唯一同时有 Prop. B.1 解析不动点、且「整组被 clip 成同一个值」率为 0 的** |

**训练臂定为 C′。** 闸门的读数:

```
长度主导的代价温和   最长 10% 占 loss 的 20.7–24.2%(均分 10%),组内仅 1.04–1.32×
                     —— 因为我们的 |y| 中位只有 305–423 token,不是论文担心的 thousands
奖励不被漂移淹掉     非简并组上奖励主导漂移 24 倍(漂移/奖励 = 0.041–0.042)
简并组上有区分       整组同值 0.0%,而 paper 变体是 30–49%
代价                 C′ 的 clip 饱和率最高(77.9%)
                     —— 用「更多被削到 ±ε」换「没有一组的奖励被抹掉」
连带                 C′ 下 β 不低于 2(漂移/奖励 ∝ 1/β²,β=1 时漂移主导 2.7×),起步用 8
```

> **闸门的作用是排除,不是批准。** 它排掉了「三个配置全军覆没」,没有回答值不值得跑。

**不动点已用 `tools/p7/p7_fixedpoint_check.py` 数值验到 7.39e-13。**
这一步把「我们的 loss 写得对不对」从 P7 的问题里摘了出去。

### 6.3 β 分解 —— 全报告最重要的分析工具

`Z_t` 是组内的**算术**均值,所以 Eq. 6 可以精确拆成两个可加的半边,且 g 关于 β 线性:

```
g_i = [mean_group(β·r) − β·r_i]  +  [mean_group(d) − d_i]
       └── 奖励项,正比于 β ──┘     └── 漂移项,与 β 无关 ──┘
```

两个直接后果,写任何结论都必须带上:

1. **在奖励退化的组上,奖励项恒为 0,g 全部来自漂移项。** 那时候 C′ 做的是向
   `π_ref·exp(βr)/Z` 的 flow matching,**不是在学奖励**。
   **不能把它说成「GRPO 给 0 梯度而 C′ 还在学奖励」。**
2. **调小 β 治不了 clip 饱和。** 它只缩小奖励那一半,漂移项纹丝不动,于是幸存的梯度反而
   **更**被漂移主导。诊断块每步实算 `sat_at_beta_{2,4,8,16}` 作为反证。
   **β 要按 `rew/drift` 选,不能按 `sat` 选。**

**材料:** `records/P7_ROUTE_C_GATE.md` · `records/P7_STEP23_RESULTS.md` ·
`records/P7_CRITERION_IB.md` · `records/P7_STEP4_RESULTS.md` ·
`training/GFlowRL/docs/`

---

## 7. 实现:改了 6 个文件、+366/−8 行

分支 `repro-4xa6000`,HEAD `c6fef78`,**九个 commit**。完整 diff 在
`training/GFlowRL/diff/`(三个 patch)。

### 7.1 GFlowRL 本身(2 个 commit)

**切分原则:论文 Eq. 4/6/7 是无梯度的**(只依赖 `π_old`、`π_ref`、`r`,actor 更新期间
全冻结),所以放在 **driver** 上每步算一次;Eq. 8 的带梯度部分注册成 `loss_mode='gflowrl'`。

```python
# ray_trainer.py — 无梯度的那一半
def compute_gflowrl_flow_gap(data, beta=8.0, eps_low=0.2, eps_high=0.28,
                             variant="cprime"):
    d = masked_sum(ref_lp - old_lp, response_mask, axis=-1).float()
    if   variant == "cprime":     d_in_z, d_in_g = d,     d          # 两边都不归一
    elif variant == "paper":      d_in_z, d_in_g = d,     d / L      # 论文原样
    elif variant == "normalized": d_in_z, d_in_g = d / L, d / L
    # Eq. 4: 按 uid 分组做批内 MC 估计,每组一个值
    ...

# core_algos.py — 带梯度的那一半
log_ratio = verl_F.masked_sum(log_prob - old_log_prob, response_mask, axis=-1)
delta = g_tilde + log_ratio
```

> **`masked_sum`(而不是 `masked_mean`)是 C′ 和 `normalized` 的唯一区别,改了不报错。**
> 所以这一行是 load-bearing 的,动过就必须重跑 fixedpoint check。代码注释里写死了这条。

**一个必须记住的约束:`kl_loss_coef=0` 但 `use_kl_loss=True`,两者都必要。**

- `use_kl_loss=False` 会让 `dp_actor` 根本不去算 `ref_log_prob`,而 GFlowRL 的
  `d = Σ(log π_ref − log π_old)` 正需要它 → 跑不起来
- `kl_loss_coef≠0` 会**重复计入**参考项,因为 d 本身就是 Eq.4/6 里的那个 KL 型量
  → 不报错,但训的是另一个目标

**这两种错都是静默的**,所以加了会抛异常的 guard。

> **一处复用带来的读日志陷阱:** GFlowRL 没有 advantage 这个东西,但注册损失的签名里
> 没有别的 per-sequence 槽位,所以 g̃ 被广播进 `batch["advantages"]`。
> **后果:trainer 打印的 `critic/advantages/*` 报的是 g̃ 的统计量,不是 GRPO 的优势。**

**两臂对比的纪律:** `run_rl_gflowrl.sh` 是个薄封装,只有一个 7 行的 overrides 数组,
靠 `run_rl.sh` 现成的 `"$@"` 透传。**复制 400 行一定会漂移** —— 两臂对比只有在
「除了目标函数以外全都一样」时才有意义。

### 7.2 修上游的 bug(5 个 commit)

这些都不是 GFlowRL 相关的,是**在单节点小机器上跑上游代码必然会撞的**:

| 症状 | 根因 |
|---|---|
| 单节点上 Ray **死锁,不报错** | `GPUS_PER_NODE` 同时给了 Toolshed 的 placement group 和 trainer,**同一批卡预订两次**。论文用 2 节点,PG 落 node1、trainer 占 node2,**这条路径永远不会暴露** |
| `PolicyLossConfig.__init__() got an unexpected keyword argument 'gflowrl'`,且在 Toolshed 起来**几分钟后**在 Ray worker 里抛 | 结构化 dataclass 对未声明的 key **报错而不是忽略**。顺带修了更安静的半边:driver 从裸 DictConfig 读 beta/eps,`core_algos` 从 dataclass 读 `eps_is` —— dataclass 里没这个字段,于是 `GF_EPS_IS` 被**静默丢弃** |
| `assert _demand <= tool_gpus` 直接拒绝启动 | 上游 `num_actors` 按 2 节点写的(7.8 逻辑卡),小机器不缩放起不来 |
| 加载 **20 分钟后** `FileNotFoundError: 'images/xxx.png'` | parquet 里是相对路径,靠 cwd 解析;`ray start` 起 raylet 用的是调用者的 cwd,而 `cd "$VERL_DIR"` 在它之后 |
| 成功的 run 报失败;**而且 `ray stop --force` 从没执行过** | `cleanup()` 杀掉后台任务然后 `wait`,`wait` 返回 128+SIGTERM,`set -e` 在 EXIT trap 里同样生效 → cleanup 死在 `wait` 那一行,后面的 `ray stop` 再也跑不到。**每次跑完都留一个活着的 Ray 集群** |

> **推论:凡是「退出码不对但看起来无害」的信号,都要查到根因再决定要不要忽略。**
> 我们是在一次成功的冒烟结束 2 小时 44 分之后,发现集群还活着才意识到的。

### 7.3 加观测(2 个 commit)

**组退化率的埋点是必需的,不是锦上添花:** 奖励退化率是 C′ 可能赢 GRPO 的**全部理由**,
而此前引用的 56.9% / 75.8% 来自 **eval benchmark 的采样,不是 Step 4 的训练集**。
调用点放在 `compute_advantage` 之后、gflowrl 分支之前,**这样两臂都会记录** ——
只有一臂的数字证明不了两臂的差异。退化按「组内极差严格为 0」判定而不是方差,
因为 pointing / IoU 奖励是连续值。

另加 β 分解的诊断埋点(§6.3)与两个 IS 权重守卫。

**材料:** `training/GFlowRL/diff/*.patch` · `training/GFlowRL/code/`

---

## 8. 训练 GFlowRL

### 8.1 机器与分卡

```
8× NVIDIA A40 48 GB(GA102 · sm_86)· 驱动 580.159.04 · 96 核 / 503 GB
容器盘 120 GB(/opt 的五个 conda 环境)· 网络卷 250 GB(权重、数据、checkpoint)
拓扑   GPU0-3 同 NUMA0(PXB 互联),GPU4-7 同 NUMA1,两组之间 SYS

GPU 0-3   Toolshed 工具 actor      TOOL_GPUS=4
GPU 4-7   训练(FSDP + sglang)     TRAIN_GPUS=4
```

**这个划分不是上游的做法** —— 拆成 `TOOL_GPUS` / `TRAIN_GPUS` 是为了绕开 §7.2 那个
「同一批卡预订两次」的死锁。

上游 `TOOL_CONFIGS` 里每个工具的 `num_actors` 和 `num_gpus` **都是论文那套 2 节点 × 8× A100-80GB
定的**:16 张卡里,**工具独占一整个 8 卡节点**(逻辑需求 7.8 ≤ 8),训练独占另一个节点。
我们是**单节点 8× A40**,工具和训练得共用这 8 张,只能分 4 张给工具。所以加了自动缩放:

```
scaled tool actors by 0.500 to fit tool_gpus=4.0:
  {'roborefer': 3, 'vlm': 1, 'sam2': 2, 'depth_estimator': 2,
   'bounding_box': 2, 'vision_ops': 4, 'grasp_generator': 2}
  -> demand 3.70 logical GPUs
```

**怎么缩的 —— 三步,只在 `_demand > tool_gpus` 时触发。** 论文的 2 节点路径上工具独占一整个
8 卡节点(7.8 ≤ 8),整段不进入,**每个数都保持论文原样**:

```
① 先把 vlm 的配额顶到 0.7   Molmo 实际按 fp32 加载(vlm.py:116 写死 dtype='auto'),
                            单实例实测峰值 32.11 GiB。
                            上游的 0.6 是按 A100-80GB 定的:0.6 × 80 = 48 GB,装得下;
                            换到 46 GB 的 A40,0.6 只有 27.6 GB —— **同一个分数换了卡就不成立了**。
                            32.11 / 46 = 0.70,所以顶到 0.7。
                            留在 0.6,Ray 就可以合法地把 vlm + 2 个 depth_estimator
                            摆进同一张卡(逻辑 1.0,物理 47.9 GiB)-> 最后一个 OOM。
                            需求随之 7.8 -> 8.0
② 按比例缩 actor 数         _scale = tool_gpus / _demand = 4.0 / 8.0 = 0.500
                            num_actors = max(1, int(num_actors × 0.500))
                            向下取整;`max(1, ...)` 保证**每种工具至少留 1 个 actor**,
                            否则那个工具会一个进程都不起,模型调它就永远超时
③ 重算后再断言              新需求 = Σ(该工具的 actor 数 × 每个 actor 的 num_gpus)
```

**① 不省卡,反而多要 0.2 张 —— 它是纠错;真正把需求压进 4 张的是 ②。**
(两步恰好凑成 8.0 / 4.0 = 0.500 这个整数比,是巧合,不是设计。)

**一个 actor = Toolshed 为某个工具起的一个进程副本**(同一工具的多个 actor 并行服务不同请求),
`num_gpus` 是**每个 actor** 向 Ray 申请的逻辑卡份额。逐项算给你看:

| 工具 | 上游 actor 数 | ×0.500 向下取整 | 每 actor 的 `num_gpus` | 小计 |
|---|--:|--:|--:|--:|
| `roborefer` | 6 | 3 | 0.6 | 1.8 |
| `vlm` | 2 | 1 | **0.7**(第 ① 步顶上去的) | 0.7 |
| `sam2` | 5 | 2 | 0.2 | 0.4 |
| `depth_estimator` | 5 | 2 | 0.2 | 0.4 |
| `bounding_box` | 5 | 2 | 0.1 | 0.2 |
| `vision_ops` | 8 | 4 | 0(纯 CPU 工具) | 0 |
| `grasp_generator` | 5 | 2 | 0.1 | 0.2 |
| **合计** | | | | **3.70 ≤ 4.0 ✓** |

**缩的是每种工具的 actor 数,不是它们的 `num_gpus`** —— **七种工具一种不少,降的是并发度**:
`roborefer` 从 6 个副本变 3 个,同时在跑的 roborefer 请求上限就减半。

> 第 ① 步防的正是 **eval 侧没防住的那件事** —— 同一个 fp32 加载问题在 §9.3 变成了一次静默污染。

> **`num_gpus` 是 Ray 的逻辑预留,不是显存配额。** 同一张物理卡上的 actor 共享全部显存,
> Ray 只保证「落在同一张卡上的分数加起来 ≤ 1.0」。所以 3.70 只决定「排得下排不下」。
> **这一条在 eval 阶段变成了一次真实的事故(§9.3)。**

工具侧共注册 **48 个方法**;模型每轮最多并发 8 个调用,最多 8 轮。

### 8.2 超参(as-run)

```
算法侧   loss_mode gflowrl · variant cprime · beta 8.0
         eps_low/high 0.2/0.28 · eps_is 0.2
         kl_loss_coef 0 · use_kl_loss True(必须,见 §7.1)
训练侧   全部继承上游 run_rl.sh,一个字没改
         train_batch_size 64 · rollout.n 5 · ppo_mini_batch_size 64(= train_batch → 严格 on-policy)
         ppo_micro_batch_size_per_gpu 2 · lr 1e-6 · grad_clip 1.0 · total_epochs 1
         max_prompt/response 8192/8192 · max_assistant_turns 8 · max_parallel_calls 8
数据     siyich/spacetools-rlfulltools · train.parquet 5500 行
```

**85 步不是 86。** 5500 ÷ 64 = 85.9,verl 丢掉不满一个 batch 的尾巴。

**1 epoch 是上游写死的,不是我们的选择。** `trainer.total_epochs=1` 在
`run_rl.sh:435`,git blame 指向 commit `1e7ff077`(Siyi Chen,2026-03-24 —— 加 RL 训练的
那个初始 commit);我们这个分支对 `run_rl.sh` 只有一个 commit(`574ae81f`,修单节点上
的卡预订死锁),那一行一个字没动。对比:**Step 1 的 point-tool RL 默认是 15 epoch**
(`run_rl_roborefer.sh:36`,`TOTAL_EPOCHS:-15`);**Step 3 的 SFT 论文 Table 6 写 Epoch 2
而代码是 `max_steps 3000`(≈3.42 epoch),两边打架,我们跑的是 3000。**

### 8.3 用到的优化手段,以及各自起了什么作用

| 手段 | 起了什么作用 |
|---|---|
| 冻结视觉编码器 | 4.066 B 里只训 2.55 B。论文的做法,见下方说明 |
| FSDP 全分片,**不** offload | 48 GB 够;offload 会把 PCIe 变成瓶颈(而这台机器 P2P 本来就坏),见下方说明 |
| ref model 参数 offload 到 CPU | ref 每步只算一次 log_prob,常驻显存不划算。`timing_s/ref ≈ 122 s`,其中 offload 本身只占约 5 s(同样一次前向、参数在 GPU 的 old_log_prob 中位 112 s vs ref 117 s,§8.4) |
| 梯度检查点<br>**gradient checkpointing**<br>(也叫 activation checkpointing / recomputation;开关是 `model.enable_gradient_checkpointing=True`) | **前向时只留少量「检查点」激活,其余的在反向时重新算一遍** —— 拿额外计算换显存。这里上下文是 `max_prompt 8192 + max_response 8192 = 16384` token,不开的话一次前向的全部层激活装不下 48 GB(**算式见下方说明**);§8.4 那个 35.3 / 48 GB 的显存峰值**就是开着它测出来的**,所以没有关掉它的余量。代价见 §4.5(SFT 上估的是慢 20–30%) |
| remove padding(去填充) | 一个 batch 里的序列长短不一,常规做法是**给短的补 pad token 到最长那条**,然后连 pad 一起算。remove padding 把各条序列的真实 token **首尾相接拼成一条扁平流**,边界用 `cu_seqlens` 单独记着,**注意力不跨序列边界**。于是计算量和显存只随**真实 token 总数**走,不随 `batch × 最长序列长度`。**结果与补零算等价**(只有浮点归约顺序可能不同)—— 注意区别于 §4.5 里被排除的 `packing`:那个是把**多条短样本合并成一条训练样本**,会改 attention mask 语义 |
| flash-attention-2 | 注意力显存 O(L²) → O(L) |
| **micro 2 + mini 64** | `mini == train_batch` 让这一步**严格 on-policy**,于是 Eq. 8 的 IS 权重恒等于 1。**一旦为省显存把 mini 调小,IS 权重就会真的开始工作,有效 batch 会悄悄缩水** —— 术语与算例见表后 |
| sglang 异步多轮 rollout | 工具调用不阻塞整个 batch。`timing_s/gen ≈ 322 s` 含全部工具往返,**说明见下方** |
| Toolshed 多 actor + 环境隔离 | 论文实测 8 并发下比朴素 HTTP 快 3.2× |
| `NCCL_P2P_DISABLE=1` | **不是优化,是绕过硬件故障(§8.5)。** 代价是 all-reduce 过主机内存 |
| checkpoint janitor(自研) | `save_freq=5` × 每个 ckpt **44 GB**,250 GB 卷第 20 步就满。只留最新的完整 ckpt(给 resume),并在第 30/60 步另存**只含权重分片**的快照(15 GB,不含 7 GB/rank 的 `optim_*.pt`)。先存档再删除,只在最新那个已写到 ≥25 GB 时才动手 |

**remove padding:省的是什么,凭什么等价。**

**省的是算力和显存,而且比例不小。** 拿 `robospatial` 的真实 `|y|` 分布举例
(§C 路闸门实测:min 45 · 中位 334 · max 900),假设一批正好抓到这三条:

```
补 pad   每条都补到最长的 900   ->  3 × 900 = 2700 个 token 位
去填充   首尾相接              ->  45 + 334 + 900 = 1279 个真 token
                                   有效利用率 47.4%,一半多的算力和激活花在 pad 上
```

**凭什么等价 —— 因为 Transformer 里只有一个算子会跨 token。**

```
逐 token 独立:  RMSNorm · q/k/v/o 投影 · MLP · 残差
                 每个位置只看自己,旁边有没有 pad 与它无关

唯一的跨 token:  注意力
                 补 pad 的做法靠 attention mask 屏蔽掉 pad 列
                 去填充的做法靠 cu_seqlens 声明边界,flash-attn 不跨界算
                 两者屏蔽掉的是同一批位置
```

再加上 loss 本来就只在真 token 上求(`response_mask`),**pad 从头到尾对任何一个真 token
的输出都没有贡献** —— 所以把它们删掉,数学上不可能改变结果。

> **「等价」到什么程度:数学等价,不是逐位相同。** 张量形状和归约顺序都变了,
> 浮点结果会在很深的小数位上不同 —— 就是 §9.2 那条链。
> 训练里这点差异被梯度噪声淹没,**但它意味着开/关 remove padding 的两次运行不会逐位可复现**。

> ⚠ **别和 §4.5 里被排除的 `packing` 搞混。** `packing` 是把**多条短样本合并成一条训练样本**,
> 会改变 batch 语义和 attention 的可见范围;remove padding 不合并样本,只是把同一批里的
> 序列换个存法。**一个改语义,一个不改。**

**「装不下 48 GB」是怎么算的。** 反向传播要用到前向留下的中间张量(激活)。
不开梯度检查点时,**每层每个 token 都得留一份**。按本模型的 `config.json`
(`hidden_size 2048` · `intermediate_size 11008` · `num_hidden_layers 36` ·
GQA 2 个 kv head × 128)数一遍一层里必须留的东西:

```
attn 输入(norm 前/后)    2 × 2048 =  4096
q / k / v                 2048 + 256 + 256 = 2560
attn 输出 / o_proj 输入   2 × 2048 =  4096
mlp gate / up / silu 乘积 3 × 11008 = 33024
                          ---------------------
                          43776 元素/token/层 × 2 B(bf16) = 85.5 KB
× 36 层                =  3.01 MB / token
```

配置的上下文上限是 `max_prompt 8192 + max_response 8192 = 16384` token:

| 序列长度 | 1 条 | `micro=2` 时 |
|--:|--:|--:|
| **16384(配置上限)** | **48.1 GB** | **96.2 GB** |
| 8192 | 24.0 GB | 48.1 GB |
| 4096 | 12.0 GB | 24.0 GB |

**一张 A40 只有 46–48 GB,而且还要先装下分片后的权重和优化器状态(~15 GB)。**
也就是说:跑满上限时,光激活就要吃掉整张卡 —— **在 8192 token 这一档就已经装不下了。**

> **两个容易踩的点。** ① **FSDP 分的是参数/梯度/优化器状态,不分激活** ——
> 激活是每张卡各自的全量,加卡不会让它变小。② 上表是**上限**,真实序列短得多
> (§C 路闸门实测 `|y|` 中位 305–423 token),所以平时远没这么夸张;
> 但配置必须按上限留余量,否则遇到一条长轨迹就当场 OOM。
>
> ⚠ 这张表是**按 config 估的,不是测的** —— 具体留哪些张量取决于实现
> (flash-attention 就省掉了 O(L²) 的注意力矩阵,PyTorch 也可能自行融合一些)。
> 量级可用,不要当精确值引。

**「sglang 异步多轮 rollout」是什么意思 —— 这里有两层异步,缺一不可。**

```
verl 侧    每条样本一个独立的 ToolAgentLoop 协程,各自跑自己的多轮状态机
           样本 A 在等 roborefer 返回时,样本 B 已经把上一次的工具结果拼回上下文、
           继续向 sglang 要下一段生成
           旋钮:rollout.name=sglang · max_parallel_calls=8(单条轨迹内的并发工具调用上限)

sglang 侧  continuous batching —— 谁生成完就退出、谁准备好就补进正在跑的 batch,
           不等整批对齐
```

**只有 continuous batching 是不够的。** 如果 verl 那边是 lockstep(全 batch 一起生成 →
全 batch 一起调工具 → 再一起生成),每一轮的耗时就等于**那一轮里最慢的那次工具调用**;
一步 64 题 × 5 rollout = 320 条轨迹、每条最多 8 轮,整步会被反复地卡在最慢的那个尾巴上。
continuous batching 只能让「生成」这一段挤紧,**填不上工具往返留下的空洞**。

所以 `timing_s/gen ≈ 322 s` 里同时装着**策略的解码时间**和**上千次工具往返**,
而它没有变成两者相加 —— 靠的就是这两层重叠。

> ⚠ **同一个机制,一面是吞吐,一面是不可复现。** continuous batching 让一条请求在每个
> decode step 的同批伙伴都不一样 —— 批高变了,kernel 的归约顺序就变,近平局处 argmax 翻转。
> **§9.2 那条噪声链的源头就在这里。**

**「FSDP 全分片、不 offload」是什么意思。** 四个词拆开:

| | 是什么 |
|---|---|
| **FSDP** | Fully Sharded Data Parallel,PyTorch 的分布式训练方式。**把参数、梯度、优化器状态按卡切开**,每张卡只存 1/N;某一层要用完整参数时才临时向其他卡收齐(all-gather),用完立刻丢掉 |
| **全分片** | 上面这三样**都切**(相当于 DeepSpeed 的 ZeRO-3 一档)。切得越多越省显存,代价是通信越多 |
| **offload** | 把参数或优化器状态**挪到 CPU 内存**,要用时再搬回 GPU。显存不够时的最后手段 |
| **PCIe** | GPU 和 CPU、GPU 和 GPU 之间的那条总线。**带宽比显存低一到两个数量级**,搬东西都得走它(A40 没有 NVLink) |
| **P2P** | peer-to-peer,两张 GPU **绕过 CPU 直接互传**。正常情况下 GPU 间通信走它 |

**这一格在说的决定是:开 FSDP 全分片,但不开 offload。**

```
开 FSDP 全分片   参数 / 梯度 / Adam 状态在 4 张训练卡间各存 1/4,48 GB 装得下 ✓
不开 offload     这些东西全程留在显存里,不往 CPU 搬
```

`param_offload=False` / `optimizer_offload=False`(`run_rl.sh.asrun:477–478`)。
**唯一的例外是 ref model**:`ref.fsdp_config.param_offload=True`(:492)——
它每步只做一次 log_prob 前向,常驻显存不划算;`timing_s/ref ≈ 122 s`,其中 offload 的额外代价只有约 5 s(§8.4)。

**为什么不开 offload:** offload 的每一次搬运都走 PCIe,而这台机器的 P2P 是坏的(§8.5),
NCCL 已经被迫 `NCCL_P2P_DISABLE=1` 走主机内存中转 —— **PCIe 上本来就挤满了 all-reduce 的流量**。
再叠一层 offload,等于把同一条已经是瓶颈的路再压一遍。**显存既然够,就不付这个钱。**

> 换句话说:§8.5 那个硬件故障不只是拖慢了吞吐,它还**改变了这里的最优选择** ——
> 在一台 P2P 正常的机器上,offload 的性价比会高得多。

> ⚠ **别和 §4.2 那条「z3 → z2」搞混。** 两处是不同阶段、不同框架、不同取舍:
> SFT 用 DeepSpeed,**48 GB 用 z2 就装得下**,所以把 z3 降到 z2,省掉白付的通信;
> RL 用 verl 的 FSDP,训练态比 SFT 大得多(多了 rollout 引擎和 ref model),
> **不全分片装不下**,只能付这个通信钱。**同一台卡型,结论相反,因为装不装得下变了。**

**「冻结视觉编码器」是什么意思。** Qwen2.5-VL 由两半组成:**视觉编码器**(把图像切成
patch、编码成一串 token)+ **语言模型**(读这串 token 和文字,做推理、写 `tool_call`)。
`freeze_vision_model=true`(`run_rl.sh:392`,上游默认,我们没动)只作用于前一半:

```
前向    照常跑 —— 图像照样被编码,模型照样"看得见"
反向    不给它算梯度
优化器  里面根本没有它的参数,Adam 的 m / v 也不为它开
```

于是 **4.066 B 参数里只有 2.55 B 在更新**,另外约 1.5 B 全程只读。省的不只是算力:
梯度和 Adam 状态是按**可训参数量**开的,冻住一块就等比例省掉三份显存
(梯度 + m + v)。

**为什么合理:** 这一步 RL 要改的是「怎么编排工具」这种策略行为,不是「怎么看图」;
而且视觉编码器在 SFT 阶段也是冻的(§4.2 `freeze_vision_tower` 保持 `true`),
**从头到尾没被训过**,两个阶段口径一致。

**代价:** 模型的视觉表征被钉死在预训练水平,RL 改不动它。所以「看错了图」这一类错误
本次训练结构上就没有改善的可能 —— 引用 P6 的错题归因时要记得这条边界。

> 4.066 B 是从 safetensors 头实读的;2.55 B 取自训练报告
> (`training/GFlowRL/A40/P7_GFLOWRL_训练报告.md` §架构),本报告未独立复核这一项拆分。

**`micro` / `mini` / IS 权重各是什么** —— 上表那一格术语最密,单列在这里:

```
micro-batch   ppo_micro_batch_size_per_gpu = 2
              一次真正喂进 GPU 做前向+反向的样本块。梯度在多个 micro 上累加,
              攒够一个 mini 才更新一次参数。纯显存旋钮,不改数学结果。
mini-batch    ppo_mini_batch_size = 64
              一次 optimizer.step() 覆盖的样本量。verl 把采集来的 train_batch
              (64 个 prompt × rollout.n=5 = 320 条轨迹)切成若干个 mini,
              每个 mini 更新一次参数。
```

**`mini == train_batch`(都是 64)⇒ 一个 rollout batch 只切得出一个 mini ⇒ 每批只更新一次参数。**
再加上 `ppo_epochs=1`,**采样用的策略 `π_old` 与被更新的策略 `π_θ` 在那一刻是同一组参数** ——
这就是「严格 on-policy」。

**IS 权重(importance sampling weight,重要性采样权重)** 是 Eq. 8 里的 `w_i`,用来纠正
「轨迹是旧策略采的、梯度却要算在新策略上」这个失配:

```
w_i = min( π_θ(y_i|x) / π_old(y_i|x) ,  1+ε_is )     Eq.8 只 clip 上界,下界不管
    = min( exp( Σ_t [ log π_θ − log π_old ] ) , 1.2 )      (ε_is = 0.2)
```

严格 on-policy 时 `π_θ ≡ π_old`,那个求和恒为 0,`w = min(exp(0), 1.2) = 1` ——
**权重形同不存在,这也是本次运行的实际状态。**

**但它是配置依赖的。** 一旦为省显存把 `mini` 调小(或让 `ppo_epochs > 1`),一个 batch 里就要
更新多次,从第二次更新起 `π_θ ≠ π_old`,`w` 开始真的起作用。而指数里是**整条序列逐 token 的
log 概率差之和**,所以极小的逐 token 漂移会被序列长度放大:

```
每 token 漂移 −3e-3 ·  序列长 1000 token  ->  w = exp(−3.0) ≈ 0.05
                       序列长 1300 token  ->  w = exp(−3.9) ≈ 0.02
```

**这条序列还在 batch 里、照样占显存、照样算前向,但它对梯度的贡献被压到百分之几** ——
有效 batch 悄悄缩水,而 loss 曲线上看不出来。两个 IS 权重守卫(`is_weight_min`、
`is_weight_collapsed_frac`,本次全程应分别恒为 1.0 与 0.0)就是为了让这件事被看见。

没用到的:LoRA(全参数微调)、offload 优化器、`expandable_segments`(与 sglang 的
TorchMemorySaver 冲突)、调 `rollout.n`(改了两臂不可比)。

### 8.4 性能与读数

```
timing_s/step  981 s(最后一步;全程均值 925 s)
  ├ gen              322 s   rollout —— sglang 生成多轮轨迹,含全部工具往返
  ├ old_log_prob     119 s   用当前策略重算 π_old 的 log_prob,一次前向(Eq. 4/6 的 d 要用它)
  ├ ref              122 s   参考模型 π_ref 的一次 log_prob 前向(d = Σ(log π_ref − log π_old) 要用它)
  ├ update_actor     369 s   前向 + 反向 + optimizer.step(),真正的梯度更新
  ├ save_checkpoint   39 s   落盘 44 GB 的 checkpoint,只在 save_freq=5 命中的步上发生
  ├ update_weights    10 s   把更新后的权重同步给 sglang
  └ 其余            0.3 s
perf/throughput  346 token/s · mfu 0.145 · 显存峰值 35.3 / 48 GB · CPU 峰值 151 GB
85 步 × 925 s ≈ 21.8 h,与墙钟 21 h 57 min 吻合
```

三个读这组数必须知道的口径:

- **`throughput` 的 346 是「每张训练卡」,不是全机。** 该步 `perf/total_num_tokens = 1,360,105`,
  `1,360,105 / 981 s = 1386 token/s` 是四卡合计,除以 4 张训练卡才是 346。**跨运行引用时要对齐卡数。**
- **`mfu` = Model FLOPs Utilization**,实际达到的 FLOPs ÷ 硬件峰值 FLOPs,0.145 就是 14.5%。
  这是个 **tool-augmented 多轮**的运行:一步里 322 s 在 rollout 与工具往返、119 s 与 122 s 在 old_log_prob 与 ref 两次前向,
  真正做稠密矩阵乘的只有 `update_actor` 那 369 s。**所以这个数不能和纯 LM 训练的 mfu 比**,
  也不能和别的机器比(`NCCL_P2P_DISABLE=1` 让 all-reduce 过主机内存,§8.5)。
- **分项对得上(2026-09-30 补)。** 此前只抄了 gen / update_actor / ref / save 四项,`322+369+122+39 = 852`,
  离 981 差的 129 s 曾记成记账缺口。读 HF 上的完整训练日志(`provenance/full_train.log.gz`)后查清:
  **= `old_log_prob` 118.8 s + `update_weights` 10.0 s + 0.3 s**,两项 verl 本来就计时。
  85 步中位:gen 321 · old_log_prob 112 · ref 117 · update_actor 352 · update_weights 10 · 每步 922 s
  (`GFlowRL_improve/infra_prep/stage_timing.py`)。随行数变的 old_log_prob + ref + update_actor 合计占 63%,
  gen 占 35% 且主要在等工具 —— 这两块是 Design Doc「Infra · 训练吞吐」的出发点。

85 步全程的关键读数:

```
                 最小      中位      最大
degen           0.578    0.703    0.859      奖励退化组占比
sat             0.303    0.516    0.613      Eq.7 clip 饱和率
sat_b2 / b4 / b16  0.478 / 0.500 / 0.525(中位)  β 换成 2/4/16 重算
reward          0.290    0.672    1.695      flow gap 的奖励半边
drift           0.000    0.339    0.441      flow gap 的漂移半边
rew/drift       0.858    2.014    4.941      < 1 = 奖励被漂移淹没
sign_ok         0.689    0.837    1.000      非退化组上 clip 后 g 与奖励项同号的比例
d_seq           0.000    0.374    0.478      |Σ(log π_ref − log π_old)|
grad_norm     103.075  180.312 5229.205      裁剪前
score           0.581    0.806    1.083      该步平均奖励
```

怎么判读:

- **`sat` 稳定在 0.5 附近,没有向 1.0 漂** —— 幅度信息还在,clip 没把梯度退化成纯符号。
- **`rew/drift` 中位 2.01,85 步里只有 step 76 跌破 1(0.86)** —— 奖励信号整体压得住漂移。
  趋势上前 10 步中位 2.125 → 后 10 步 1.754,缓慢下降但幅度不大。
- **`sat_b2` 0.478 / `sat_b8` 0.516 / `sat_b16` 0.525 —— β 差 8 倍,饱和率只动 5 个百分点。**
  这是「调小 β 治不了饱和」的每步实证。
- **`grad_norm` 中位 180,而 `grad_clip=1.0`。** **每一次更新都被裁到只剩方向、没有幅度。**
  这条必须写进结论,否则「学习率 1e-6」这个数字会被误读。
- **`score` 没有可靠的上升趋势。** 前 10 步中位 0.766、后 10 步 0.820,但前半 0.809、
  后半 0.796 —— 这个量级的来回就是噪声。**训练奖励在这次运行里没有明显改善。**
- **`degen` 中位 70.3%** —— 每个 batch 的大多数组,奖励项恒为 0。

### 8.5 一个硬件故障:容器里的 PCIe P2P 是坏的

**症状:** trainer 起来后第一个 NCCL collective 永远不完成,10 分钟后 watchdog 报
`Last enqueued NCCL work: 1, last completed NCCL work: -1`。而 `nvidia-smi topo -p2p r`
显示全是 OK。

**定位:** 没有靠反复起 15 分钟的训练去试,而是写了一个 4 卡 all-reduce 的**最小探针**,
90 秒一轮跑了 5 个变体:

```
A  默认                          -> 40 s 超时
B  NCCL_IB_DISABLE=1             -> 40 s 超时   (不是 IB/RoCE 的事)
C  B + NCCL_SOCKET_IFNAME=eth0   -> 40 s 超时
D  B + NCCL_P2P_DISABLE=1        -> 7 秒 ALLREDUCE_OK   <- 就是它
E  B + NCCL_SHM_DISABLE=1        -> 40 s 超时   (SHM 正是能用的那条路)
```

**根因:** 容器里 ACS / IOMMU 没关的典型表现 —— P2P 在拓扑上「可用」,实际传输挂死。
**平台问题,不是代码问题。**

**代价:** 关掉 P2P 后 NCCL 走共享内存(过主机内存中转),更慢。**本次每步 925 s 是在
这个前提下测的,不能直接拿去和别的机器比吞吐。**

> **教训:租到新机器后,跑训练之前先花 90 秒跑一次 all-reduce 探针。**
> 这次如果不做,就是每次改一个环境变量等 15 分钟,一天就没了。

### 8.6 工具侧的运行时错误(都被正确吞掉了)

9.8 MB 训练日志里的统计:

```
No collision-free grasps found          3389 次(其中 2346 次作为 RuntimeError 抛出)
Top-down filtering removed all grasps    130 次
selected index k out of range             24 次
CUDA error: invalid configuration argument 12 次(depth_estimator.get_depth_map)
Mask removed all points                   10 次
torch.cuda.OutOfMemoryError                2 次(工具侧)
达到 8 轮上限                              2 次
截断的工具响应 / 训练侧 OOM / NCCL WARN / 崩溃    0
```

**这些不是训练故障。** Toolshed 把工具异常转成文本返回给模型,模型可以换工具或换参数重试
—— **这正是 tool-augmented RL 要学的东西。** `grasp_generator` 那 3389 次尤其正常:
它在真实点云上做碰撞检测,「这个物体周围没有无碰撞抓取位姿」是**几何事实**。

**但两条值得单独记:** 两次工具侧 OOM,一次试图分配 **14.36 GiB**、另一次
**1481.25 GiB**,都在 `torch.cdist` 里 —— 1481 GiB 显然不是显存不够,是**输入退化**
(点云点数异常膨胀)导致中间矩阵爆炸。**如果以后要统计「工具成功率」,这些必须计入分母。**

**材料:** `05_gflowrl_training/gflowrl_training_report_8xA40.md` · `training/GFlowRL/A40/`(metrics / logs /
as-run 脚本 / 环境快照)

---

## 9. GFlowRL eval

### 9.1 口径

```
九个 benchmark · 2121 样本 · greedy 解码(verl val_kwargs 默认 temperature 0)
4× RTX A6000(与 SFT 基线同卡)· NUM_GPUS=4 EVAL_GPUS=1
```

三次运行:

```
Run A  2026-09-17 23:40 -> 09-18 01:45  2 h 05 m   九个 key
       五个(robospatial/reflocation/refplacement/refunseen/cvb2drelation)零 OOM,有效
       另外四个被显存问题污染,结果作废(§9.3),本报告不引用它们的分数
Run B  09-18 03:06 -> 03:53             46 m       被污染那四个重跑,零 OOM
Run C  09-18 04:16 -> 04:36             19 m 45 s  robospatial 第二次,零 OOM
```

### 9.2 三处参数改动

| 改动 | 为什么 | 对分数有没有影响 |
|---|---|---|
| `BENCHMARKS` 九行路径 `xxx/test.parquet` → `data/<key>.parquet` | 上游脚本期望嵌套目录,而数据集**无论 `main` 还是钉死的 revision 都是扁平的**。上游改过结构、脚本没跟上。不改则 `exit 1`,一个样本都跑不了 | **无** —— 改的是路径,读到的 parquet 是同一个 |
| `gpu_memory_utilization` 0.5 → **0.545** | 见下方「为什么非对齐不可」 | **有,而且这正是目的** —— 把它压回与基线同一条件 |
| `vlm` 的 `num_gpus` 0.6 → **1.0** | 见 §9.3 | **不进入分数** —— 只改 Ray 的摆放,不改模型、判分或工具行为 |

**`gpu_memory_utilization` 为什么非对齐不可。** 这个旋钮是 **整卡比例,不是绝对值**
—— 同一个 `0.5` 在不同容量的卡上开出不同大小的 KV 池。

```
SFT 基线(RoboSpatial 61.00 ± 0.77)   48 GB 卡 × 0.5   = 24 GB KV 池
这台机器最小的卡  46068 MiB           × 0.5   ≈ 22 GB     ← 照抄 0.5 就小了
                                      × 0.545 ≈ 24 GB     ← 反算出来的值
```

**这个 eval 的唯一用途是和 SFT 起点、和 P4 做逐样本对比,所以除了 checkpoint 本身,
其余条件必须全部钉死。** KV 池不是一个中性旋钮:

```
池子大小 → 一批能并发几条序列 → batch 的组成 → 浮点归约顺序 → 接近平局的样本翻转
```

**这条链不是推演,P6 做过对照组**(`spacetools-repro/p6/gmu025/README.md`):同一个 ckpt、
同一份 `blinkdepth`、**只有 KV 池不同**。

```
run4   80 GB 卡 × 0.5  = 40 GB 池    106 / 124
run5   80 GB 卡 × 0.25 = 20 GB 池    108 / 124
逐样本比:124 题里 8 题翻转(两个方向都有),净 +2
```

**翻转的样本 index 18**(`index` 是 dump 里的样本序号,blinkdepth 的第 19 题)
—— 题目是「A、B 两点哪个离相机更近」。**两次运行的前五次工具调用,参数逐字相同、
返回也逐字相同:**

```
depth_estimator.estimate_depth(image_index=0)              -> $depth_map
roborefer.detect_one("red circle under label A")           -> (0.136, 0.456)
roborefer.detect_one("red circle under label B")           -> (0.865, 0.456)
vision_ops.index_at($depth_map, 0.136, 0.456)  -> 0.3450107276439667
vision_ops.index_at($depth_map, 0.865, 0.456)  -> 0.33692145347595215
                                                  两点差 0.008 m
```

**同一份证据,两个相反的处置:**

```
40 GB 池   6 轮收手。「when depth values are too close (within 0.5m),
           I should ignore the tool results and use visual reasoning」
           -> 丢掉 0.008 m 这个差,改用「前景/背景」的视觉直觉 -> 答 A   ✗(GT 是 B)
20 GB 池   8 轮。同样说了「the difference is very small (only 0.008m)」,
           但没有收手,又叫了两次 vlm.detect_one 复核两个点的位置
           -> 复核一致,于是采信 0.337 < 0.345 -> 答 B                  ✓
```

解码是 **greedy**,提示词和工具返回又逐字节相同 —— 所以这两条轨迹的分岔**只能**来自
数值:池子大小改变了一批里并发几条序列,batch 组成变了,kernel 的归约顺序就变,
logits 在小数点后很深的位上抖动。平时这点抖动被 argmax 吸收掉;
**碰上模型本来就在两可之间的分叉点,它就被放大成一个完整的答案差。**

**这中间到底发生了什么 —— 五步。**

```
① 池子小一半        KV cache 按 token 存,池子的字节数 ∝ 能同时驻留的 token 数。
                    40 GB -> 20 GB,同一时刻能并发的序列数大致减半。

② 批的高度变了      sglang 是连续批处理:谁生成完就退出、新请求随时补进来。
                    并发上限变了,index 18 这条请求在解码到第 t 个 token 时,
                    和它拼在同一批里的**是另外一组序列**,批高 M 不同。

③ 换了 kernel       每个 decode step 要算 x·W。M 变了,cuBLAS 挑的 tile 形状和
                    split-K 切法就变 —— **同样 2048 个乘积,分几段累加变了**。

④ 加法不满足结合律  这一步可以直接跑(numpy,fp32,同一组 2048 个数):

                      分 8 段归约再相加   128.9586639404
                      分 4 段归约再相加   128.9586791992
                                   差     1.5e-5

                    数学上两者相等,浮点下不等 —— 每段的中间和大小不同,
                    舍入落在不同的位上。

⑤ argmax 在平局处翻 这 1.5e-5 加到 logit 上,99.99% 的 token 毫无影响 ——
                    第一名和第二名通常差好几个数量级。
                    只有在两个候选 token 咬得极近时,它才够把名次换过来。
                    **index 18 上这一下发生在第 201 个字符**,两条轨迹在此之前
                    逐字相同,之后再没重合过:

                      共同前缀  "...The image shows a news studio with a presenter"
                      40 GB 池  接 ". In the upper left and right corners, there
                                 appear to be two red circles labeled A and B..."
                      20 GB 池  接 " in the foreground. In the background, there
                                 are buildings visible through a window..."

                    注意翻的是一句**场景描述**,当时离答案还有三千多个字符。
```

> ⚠ **一个必须说清的限定。** 这批 dump 里**没有存 logprob**,所以「哪个 token 咬得近」
> 无法直接读出来 —— 上面第 ⑤ 步的因果方向是**从结果反推的**:两条轨迹在第 201 字符
> 分开,而在此之前所有输入逐字节相同、解码是 greedy,除了批组成没有别的变量。
> 「模型在这里两可」的依据也只是**它自己写的话**(两次都写了「差值太小不可靠」),
> 不是测出来的概率。**要把这条链钉死,得在 eval 时存 top-k logprob** —— 已记进 §12。

**翻转只需要发生一次。** 自回归解码里,每生成一个 token 都会追加进上下文、成为后面所有
token 的条件。所以从翻转的那一步之后,两次运行**不再是同一个计算的两个浮点版本** ——
它们的上下文已经是不同的文本,算的是不同的条件分布,没有任何力把它们拉回来:

```
40 GB 池  第一句话把 A、B 描述成「左上/右上角的两个红圈」
          -> 后文一直在画面构图里打转 -> 6 轮收手 -> 视觉直觉 -> 答 A
20 GB 池  第一句话把 A、B 描述成「窗外背景建筑上的点」
          -> 后文一直在追这两个点的位置 -> 又叫两次 vlm 复核 -> 8 轮 -> 答 B
```

后面三千多个字符的差异已经不是 1.5e-5 撑起来的,是**第一句场景描述**撑起来的 ——
1.5e-5 只付了第一下的钱。

**反过来也解释了这类噪声为什么只在特定题上显形:** 分叉点本身是随机落的
(index 18 落在一句无关紧要的场景描述上),但**只有当题目的判决本来就贴边时,
分叉才会一路走到不同答案**。证据充分的题上,两条轨迹措辞不同、结论照样一样 ——
§10.3 那 777 个逐样本完全一致的样本就是这种情况。
`robospatial` 的 Vacant 是重灾区(§10),正因为它的判据是连续坐标上的二值凸包归属,
贴边样本本来就处处是平局。

复现 ④ 那两个数:

```python
import numpy as np
rng = np.random.default_rng(0)
p = (rng.standard_normal(2048) * rng.standard_normal(2048)).astype(np.float32)
f = lambda k: np.float32(sum(np.float32(np.add.reduce(c)) for c in p.reshape(k, -1)))
print(f(8), f(4))        # 128.95866  128.95868
```

> **为什么能断定是这条链,而不是工具。** SFT eval 那次做过直接检查:350 个样本里
> **零例「同一个查询、工具返回不同结果」** —— 工具侧逐位确定。
> 排除工具之后,greedy 解码下剩下的唯一变量就是批组成。

> 所以 §10 那句「4 题之差分不开机器项和模型项」不是谨慎措辞 ——
> **这里同一个模型自己跟自己就差了 2 题。**

也就是说,**不对齐,测到的差值里就混进了一项纯机器效应**,而这一项与「C′ 比 SFT 起点强多少」
在数值上无法分离。对齐之后差值才只剩模型项。**代价是:引用任何分数都必须带上 `gmu=0.545`
这个口径**,换机器要重新反算。

> ⚠ 对齐只是**近似**。这台机器混合 ECC(GPU0/1 49140 MiB,GPU2/3 46068 MiB),
> **策略模型**(policy —— 被评的那个 Qwen2.5-VL-3B,由 sglang 托管;与七个工具模型分开占卡)
> 落在哪张卡上,池子就差 `3072 MiB × 0.545 ≈ 1674 MiB ≈ 1.6 GiB`:
>
> ```
> GPU0/1  49140 MiB(ECC off)× 0.545 ≈ 26781 MiB 池
> GPU2/3  46068 MiB(ECC on) × 0.545 ≈ 25107 MiB 池   <- Run C 实测落在这张(GPU3)
>         ↑ 容量差 3072 MiB      × 0.545 = 池子差 1674 MiB
> ```
>
> 3072 MiB 的源头是 **ECC**:开了 ECC 的卡要留一部分显存存校验位,对外报出的容量就少 3 GiB。
> 同一批 A6000,两张开两张关。
>
> **两件事别混:** *差多少* 由「两张卡的容量差 × `gmu`」决定,**策略模型自己占多大不进这个算式**;
> *差不差* 由策略落哪张卡决定,因为 `gmu` 算的是**它所在那张卡**的比例 ——
> 落 GPU0/1 就按 49140 开池,落 GPU2/3 就按 46068 开。
> **Ray 的摆放决定用哪张卡的容量代进公式,ECC 决定两个容量差多少;策略模型只是被摆的那个东西。**
> 这是 §10 里那个「4 题之差分不开机器项和模型项」的来源之一。

**没有改、但必须显式给的两件事:** `NUM_GPUS=4 EVAL_GPUS=1`(上游默认 8/4,4 卡机上
不覆盖会让 Ray **不报错、actor 无限排队**);`BASH_ENV=.../conda.sh`(`conda activate`
是 shell 函数,不跨子进程)。

**特意没改的:** `vlm.py:116` 的 fp32 问题本身 —— 论文结果是在那个行为下跑出来的。

### 9.3 一次真实的静默污染,以及它是怎么被抓住的

Run A 的四个 benchmark 命中了一个显存错配:

```
run_eval.sh 给 vlm 传的是 'dtype': 'float16',但上游 vlm.py:116 写死 torch_dtype="auto"
-> Molmo-7B-D 以 fp32 加载,实测占 30.2 GiB,而它只声明 num_gpus=0.6
-> Ray 完全合法地把 vlm(0.6) + depth_estimator(0.2) + sam2(0.2) = 1.0 摆进同一张卡
-> 30.20 + 9.15 + 7.97 = 47.3 GiB / 整卡 47.4 GiB,只剩 70 MiB
-> 工具再要 576 MiB -> torch.OutOfMemoryError
OOM 计数:blinkdepth 102 · cvb3ddepth 499 · boppose 21 · bopgrasp 19
```

**表现不是崩溃,是静默掉分。** 工具 OOM 之后 Toolshed 把错误包成正常的 `ToolResult` 返回,
模型拿着错误字符串继续推理,eval 跑完、不报错、给出一个看着合理的数字。

**错题归因把这件事量化得很清楚:**

```
cvb3ddepth   污染那次 243 错 -> 230 归因 OOM,只有 13 个是真错
             重跑        21 错 ->   0 归因 OOM,21 个全是真错
blinkdepth   污染那次  25 错 ->  19 归因 OOM
             重跑        15 错 ->   0
```

还有一条**二阶效应**:污染那次日志里有 590 条
`TypeError: Expected PIL Image, got <class 'str'>` —— 那是 OOM 的下游,
工具返回错误字符串 → 模型把它当 `$变量` 喂给下一个工具 → TypeError。重跑后降到 4 条。

**为什么它没有变成一个错误的结论:因为门禁的顺序是写死的。**

```
先数 OOM -> 再查工具静默错误 -> 最后才看分数
```

**三个独立口径互相印证:** 主日志 `grep -c OutOfMemoryError` = 0 · 每个 benchmark 自己的
`eval.log` = 0 · `parse_dump.py --strict` 的逐样本 `OOM samples` = 0 ·
连 `CUDA out of memory` 的原文都是 0 次。

> **同时暴露了两条方法学缺陷,都已修:**
> ① `run_eval.sh` 打完 `EVALUATION COMPLETE` 之后,它自己的 cleanup 陷阱 kill 后台
> toolshed,`wait` 把 SIGTERM 带出来,**脚本退出码 15**。三次运行退出码都是 15,
> 三次都跑完了 —— **按退出码判死会把一批完全有效的结果判废。** 正确判据是
> **成功标记 + 样本数 + OOM 数 + router 失联数**,不是 `$?`。
> ② `run_eval.sh` 没有并发保护:两个实例在开头都执行 `ray stop --force`,后起的会把先起的
> toolshed 连同全部工具 actor 一起杀掉,而**先起的那个 eval 继续跑、照样产出分数**,
> 只是每次调工具都拿到 `Could not find ToolRouterActor`。已加原子锁 + 把这个计数纳入门禁。

### 9.4 健康门禁(全零才看分数)

```
                      n    OOM  截断  顶轮数  缺<answer>  工具失败样本
robospatial(A/C)    350     0    0      0        0          0
reflocation         100     0    0      0        0          0
refplacement        100     0    0      0        0          0
refunseen            77     0    0      0        0          0
cvb2drelation       650     0    0      0        0          0
blinkdepth          124     0    0      0        0          1
cvb3ddepth          600     0    0      0        0          0
boppose              60     0    0      0        0          0
bopgrasp             60     0    0      0        0         43   <- 领域结果,非故障
```

`bopgrasp` 的 43 是 `grasp_generator` 返回「找不到无碰撞抓取」(32)或「top-down 过滤掉
全部抓取」(11)。**P4 修正后的同口径数字是 41/60,两边一致,所以不是本次的问题。**

> 顺带复现了 P4 记录过的一个上游缺陷:`analyze_grasp_result.py` 只匹配
> `No collision-free grasps`,**漏掉 `Top-down filtering removed all`**,
> 因此它报 32 个 error sample,真实是 43。

### 9.5 结果

```
RoboSpatial   VQA      228   71.93 / 72.81   (两次)
              Vacant   122   41.80 / 44.26
              Overall  350   61.43 / 62.86   均值 62.14
RefSpatial    Location 100   52.00      Placement 100  58.00      Unseen 77  48.05
              三项     277   52.68 简单平均 / 53.07 加权
BLINK Relative Depth   124   87.90
CV-Bench 2D / 3D  650/600   94.62 / 96.50
BOP-ASK Pose            60   55.73 平均 IoU
BOP-ASK Grasp           60   MACE 44.79 · SR 55.00%
```

### 9.6 与 SFT 起点的对照:没有可测到的提升

| | n | SFT 起点 | **C′ step85** | 差 |
|---|--:|--:|--:|--:|
| RoboSpatial VQA | 228 | 71.27 | 72.37(两次均值) | +1.1 |
| RoboSpatial Vacant | 122 | 41.80 | 43.03(两次均值) | +1.2 |
| **RoboSpatial Overall** | 350 | **61.00 ± 0.77** | **62.14** | **+1.14** |
| RefSpatial 三项(加权) | 277 | 53.07 | 53.07 | **0.00** |

**差 4 题 ≈ 1.5 个标准误,不构成有统计意义的提升:**

```
C′  翻转 25 -> 单次 sd ≈ 2.5 题,两次均值 sd ≈ 1.8 题
SFT 翻转 29 -> 两次均值 sd ≈ 1.9 题
合并 sd ≈ 2.6 题,差 4.0 题 -> 约 1.5 σ,双侧 p ≈ 0.12
两边区间 205–230 与 199–228 几乎完全重叠
```

**这些数是怎么来的。** 模型只有一句:**恒对的样本每次都对,翻转的样本每次≈抛一次硬币。**

```
对题数 = 恒对 + Bin(k, 0.5)            k = 翻转样本数
单次 sd            = √k / 2
两次运行取均值的 sd = √(k/2) / 2
```

`k` 和「恒对」都是数出来的,不是估的:

| | 哪两次 | n | 恒对 | 翻转 k | 可能区间 | 期望 |
|---|---|--:|--:|--:|---|--:|
| **C′** | P7 eval Run A ↔ Run C | 350 | 205 | **25** | 205 – 230 | **217.5** |
| **SFT 起点** | SFT eval 的两次 | 350 | 199 | **29** | 199 – 228 | **213.5** |

(C′ 两次分别对 215 / 220;SFT 那两次是 216 / 211,即 61.71% / 60.29%。)

剩下的就是算:

```
两次均值 sd    C′  √(25/2)/2 = 1.77      SFT  √(29/2)/2 = 1.90
合并 sd        √(1.77² + 1.90²) = 2.60
差             217.5 − 213.5 = 4.0 题
z              4.0 / 2.60 = 1.54        双侧 p = 0.124
```

> **两个前提要一起说。** ① 每个翻转样本按 **0.5** 落对 —— 这是**方差最大**的假设,
> 所以 1.54 σ 是个**保守**(偏难显著)的读数;真实翻转概率若偏离 0.5,σ 只会更小、
> z 只会更大,但那需要每题多跑几次才能估。② 假设样本间独立 —— 同一批里的样本共享
> batch,严格讲不独立,不过 §9.2 那条链说明相关性来自浮点噪声而非题目内容,量级可忽略。
>
> **更该注意的是分子而不是分母:** 这 4.0 题的差里还混着机器项(§12.1 第 7、10 条),
> 而统计检验管不了那个。

**而且 SFT 的 61.00 是在另一台机器上测的**,本次没有在这台机器上重测 SFT。
叠加这台机器的**混合 ECC**(GPU0/1 = 49140 MiB / ECC off,GPU2/3 = 46068 MiB / ECC on,
差 3072 MiB,而 gmu 是整卡比例 → 策略落哪张卡 KV 池就差约 1.7 GB),
**这 4 题的差机器层面的解释和模型层面的解释目前分不开。**

> **要钉死只有一个干净做法:在同一台机器、同一 session 里把 SFT ckpt 也跑一遍
> robospatial。** 这一步没做(机器已释放),是本报告最大的一个空缺。

**材料:** `06_gflowrl_eval/gflowrl_eval_report.md` · `eval/GFlowRL/P7_GFLOWRL_EVAL/`
(dumps / logs / gates / scripts / gpu / config / MANIFEST)

---

## 10. 与 P4 / P5 / P6 的逐样本对比

这一节只吃 parsed 记录,**零 GPU**。P7 的 parsed 由
`spacetools-repro/tools/parse_dump.py --emit` 从 dumps 生成,字段与 `p4/parsed/` 逐个相同
(`chain_signature`、`vars_exposed/used/unused/phantom`、`trajectory`、
`num_turns_verl_convention`)。

### 10.1 能回答什么、不能回答什么

**能:** 官方那个 RL 之后的 checkpoint 与我们用 C′ 训出来的 checkpoint,在同一批 2121 个
样本上**逐样本**的行为差在哪。工具链路、变量复用、错题归因、透传率都是可验证的痕迹。

**不能:** 「C′ 比 GRPO 好还是差」。两个 checkpoint **base 不同** —— 官方那个是论文自己的
SFT 起点,我们这个是自训的。**这不是一次受控的 A/B。**

**真正受控的对比是 SFT 起点 ↔ C′,而那一边只有汇总数字、没有 dump**
(SFT eval 是在一台已释放的机器上跑的)。**下次评 checkpoint 必须把 dump 一起存档。**

### 10.2 总表

| benchmark | n | P4 对 | P7 对 | 只有 P4 对 | 只有 P7 对 | McNemar p | |
|---|--:|--:|--:|--:|--:|--:|---|
| robospatial | 350 | 229 | 215 | 37 | 23 | 0.092 | 不显著 |
| reflocation | 100 | 54 | 52 | 3 | 1 | 0.625 | 不显著 |
| refplacement | 100 | 58 | 58 | **0** | **0** | 1.000 | **逐样本相同** |
| refunseen | 77 | 37 | 37 | **0** | **0** | 1.000 | **逐样本相同** |
| blinkdepth | 124 | 107 | 109 | 4 | 6 | 0.754 | 不显著 |
| cvb2drelation | 650 | 615 | 615 | 6 | 6 | 1.000 | 不显著 |
| cvb3ddepth | 600 | 579 | 579 | **0** | **0** | 1.000 | **逐样本相同** |
| boppose | 60 | 38 | 39 | 1 | 2 | 1.000 | 不显著 |
| bopgrasp | 60 | ~~54~~ | ~~56~~ | ~~0~~ | ~~2~~ | 0.500 | ⚠ 口径反了,见 §10.2 末 |

P4 用 run1(单次);P4 报告本身对 robospatial / blinkdepth / bopgrasp 是跑多次报区间的。

**连续判分的两个用配对符号检验,均值和逐样本方向互相矛盾:**

```
boppose    P4 均值 0.5336   P7 均值 0.5573   差 +0.0237
           但逐样本 P7 更高 19 · P4 更高 26 · 逐位相同 15    p = 0.371
bopgrasp   P4 均值 1.9354   P7 均值 1.8576   差 −0.0778
           但逐样本 P7 更高 25 · P4 更高 31 · 逐位相同  4    p = 0.504
```

**先说 `score` 是什么。** 这两个 benchmark 的 `score` 不是准确率,是两个**方向相反**的几何量,
由官方 scorer `verl/utils/reward_score/bop_ask_bench.py` 算出:

| benchmark | `score` 是什么 | 范围 | 方向 |
|---|---|---|---|
| `boppose` | 预测的 8 个立方体角点与 GT 8 点,各取**凸包**后两个多边形的 **IoU** | 0 – 1 | **越大越好** |
| `bopgrasp` | **NCE** = 5 个夹爪关键点的平均误差 ÷ GT 的夹爪宽度 `d`,再截到上限 10 | 0 – 10 | **越小越好** |

拿 `bopgrasp` 的样本 0 走一遍(数字直接取自 `parsed/bopgrasp.jsonl`,归一化图像坐标):

```
GT    中心 (0.358, 0.257)  左指根 (0.326, 0.239)  右指根 (0.387, 0.274)
      左指尖 (0.324, 0.313)  右指尖 (0.385, 0.339)
预测  中心 (0.381, 0.428)  左指根 (0.341, 0.428)  右指根 (0.421, 0.428)
      左指尖 (0.301, 0.428)  右指尖 (0.461, 0.428)

夹爪宽 d = |左指根 − 右指根| = 0.0701
五点平均误差 = 0.1495       (scorer 会同时试左右镜像,取较小的那个)
NCE = 0.1495 / 0.0701 = 2.13     ← 存档 score = 2.1314 ✓
```

预测的五个点 **y 坐标全是 0.428** —— 模型画了一把水平的夹爪,而 GT 是斜的。
**平均每个关键点偏掉了 2.1 个夹爪宽度。** 所以 NCE 越大越糟,0 才是完美。

`boppose` 反过来:样本 0 的 score = 0.0000(预测的八个点缩成一小团,凸包与 GT 不相交),
样本 1 的 score = 0.7939(八个角基本压住,凸包重叠了约八成)。

> **⚠ 顺着这个口径查出一处错误。** `tools/parse_dump.py` 对所有 benchmark 统一用
> `correct = (score >= threshold)`,`threshold` 默认 0.5。这对 `boppose` 的 IoU 是对的,
> 对 `bopgrasp` 的 NCE **符号整个反了**。实跑即可证伪:P7 上 NCE 最差的样本
> (sid 31,`score = 10.0`,模型连 5 个点都没给全)被判成「对」,NCE 最好的样本
> (sid 20,`score = 0.4035`,唯一一个平均误差小于半个夹爪宽的)被判成「错」。
>
> **§10.2 总表 `bopgrasp` 那一行的对错计数不可用**,已划掉。下面符号检验的**方向标签**同样要反过来读:
> 「P4 更高 31」= P4 的 NCE 更大 = **P7 在这 31 道题上更好**;均值差 `−0.0778` 是**改善**不是退步。
> 检验本身不受影响(p = 0.504,只数两侧对不对称,与方向无关),结论仍是**不显著**。
>
> `boppose` 不受此影响 —— IoU 与 `>= 0.5` 同向。

**这几个数怎么算的**(`analysis/significance_p4_p7.py`,两边按 `sample_id` 配对,n = 60):

- **均值** —— 60 道题 `score` 字段的算术平均。两个 benchmark 的 score 不是一个量纲:
  `boppose` 实测落在 0–0.953(IoU,越大越好),`bopgrasp` 落在 0.404–10.0(NCE,越小越好,10 是截断上限)。
  **两列方向相反,绝不能横比。**
- **三分计数** —— 对每道题算 `d = P7.score − P4.score`,按符号分三堆:`d > 0` 记 P7 更高、
  `d < 0` 记 P4 更高、`|d| < 1e-12` 记逐位相同。「逐位相同」是**浮点位级相同**,
  不是四舍五入后相等 —— 同一条工具链、同一个确定性工具,score 会一模一样,
  所以 `boppose` 才有 15 道题落在这一堆。
- **p 值** —— 配对**符号检验**:平局按定义不参与,只拿非平局的对
  (`boppose` 19 + 26 = 45,`bopgrasp` 25 + 31 = 56),在「两边各占一半」的原假设下算双侧二项概率
  `2 · P(X ≤ min(pos, neg))`,`X ~ Bin(45, 0.5)`。
  **两个都远大于 0.05 —— 方向上的差异完全在抛硬币的范围内。**

均值和三分之所以能互相矛盾:**均值按幅度加权,符号检验只数方向。**`boppose` 上 P7 赢的题更少
(19 < 26)但赢的那几题幅度大,均值就被拉成了正的。

最后一个口径差:§10.2 总表里的「P4 对 / P7 对」不是这里的 score,是 `correct` 字段,
即上面那条 `score >= 0.5` 的阈值判定。**两张表数的不是同一件事** —— 以 `boppose` 为例,
一道题的 IoU 从 0.51 掉到 0.49,在这里只记一次「P4 更高」,在总表里却会翻一格对错;
从 0.90 掉到 0.60 则只进这里、总表纹丝不动。

### 10.3 第一个结构性发现:777 个样本上两个 checkpoint 逐样本完全一致

`refplacement`(100)+ `refunseen`(77)+ `cvb3ddepth`(600)—— **777 个样本,没有一个分歧。**
不是分数相同,是**每一道题的对错都相同**。

机制 P6 早就给出了:在这些 benchmark 上「推理」是一条能写成代码的规则,模型逐字执行它。
规则的输入是工具的输出,工具是确定性的,于是**策略模型换成谁,结果都一样**。

拿 `cvb3ddepth` 的样本 0 走一遍(两边 `parsed/cvb3ddepth.jsonl` 逐字对照)。
题目:桌子(红框)和书柜(蓝框),哪个离相机更近?(A) table (B) bookcase

```
turn 1   depth_estimator.estimate_depth(image_index=0)        → $depth_map
         roborefer 定位红框 / 蓝框                             → 两个像素坐标
turn 2   vision_ops.index_at($depth_map, u=0.501, v=0.661)    ← 红框(桌子)
         vision_ops.index_at($depth_map, u=0.260, v=0.406)    ← 蓝框(书柜)
turn 3   比大小,答小的那个                                     → A

            红框读数                蓝框读数            答案
P4      2.7630178928375244     6.707825660705566        A
P7      2.763768196105957      6.710909366607666        A
```

**「规则」就是 turn 3 那一句:两个深度谁小答谁。** 策略模型负责的只有「摆出这三步」;
坐标是 RoboRefer 给的,数值是 DepthPro 给的,**这两个模型都不在 RL 的训练范围内**。
两次运行的深度在小数点后第 4 位才分家(GPU 浮点非确定性),
而 2.76 与 6.71 之间隔着 3.9 米 —— **这点抖动离翻盘差了四个数量级。**

600 道题里 **577 道的链路签名逐字相同**;剩下 23 道中 21 道只差轮数
(P4 走 4 轮、P7 走 3 轮,工具与调用次数完全一致),另 2 道 P7 换掉了 `vlm`、多调了 `roborefer`。
**这 23 道的对错也全部相同** —— C′ 争取到的那点编排自由度,一步都落不到判决上。

**对照组:`robospatial` 的样本 54 —— 同一道题,P4 判错、P7 判对。**

```
题目:图中有一个水槽。在紧邻水槽、位于其上方的空位里,指出一个可以放东西的点。

P4  turn 1  roborefer.detect_one(obj_name =
              "point within the vacant space suitable for placing an object
               close to and to the above the sink")     ← 把题干整句抄进去
            → 返回 (0.389, 0.661)
    turn 2  答 [(0.389, 0.661)]                          ✗ 判错

P7  turn 1  roborefer.detect_one(obj_name =
              "point close to and above the sink")       ← 自己把题干压成短语
            → 返回 (0.07, 0.754)
    turn 2  答 [(0.07, 0.754)]                           ✓ 判对
```

两边的链路签名**都是 `roboreferx1@2t`** —— 同一个工具、同一次调用、同样两轮,
RoboRefer 也还是那个确定性的 RoboRefer。**唯一的差别是模型写进 `obj_name` 的那句话。**

放在一起就清楚了:

| | `cvb3ddepth` / `refplacement` / `refunseen` | `robospatial` |
|---|---|---|
| 工具吃什么 | `$depth_map` + RoboRefer 自己算出的坐标,**模型没得编** | 一句**模型自己措辞**的自然语言 |
| 模型最后一步干什么 | 比两个数的大小 | **把工具返回的点原样抄成答案** |
| 模型的自由度落在哪 | 只剩「分几轮做完」 | 措辞 → 检测点 → 答案,**一路透传到判分** |
| 结果 | 600 道 **0 分歧** | 350 道 **60 分歧**,其中 **58 道链路签名完全相同** |

**「策略自由度」的准确含义是这个**:不是模型能不能换工具,而是**模型的自由选择有没有一条通往分数的路**。
`cvb3ddepth` 上这条路被确定性工具掐断了 —— 模型怎么绕,最后都是拿 DepthPro 的两个数比大小;
`robospatial` 上它是**唯一**的一条路 —— 模型怎么描述那个点,就决定了它得几分。
**RL 只能改模型,所以 RL 只在第二类上可见。**

> **含义:这三个 benchmark 测不出任何 RL 算法的差别。** 它们测的是 RoboRefer 和 DepthPro,
> 不是策略。把它们放进「RL 有没有效」的对照表里,只会稀释信号。
> **九个 benchmark 里真正有策略自由度的只有 `robospatial`(分歧 60)和 `blinkdepth`(10)。**

### 10.4 直接回答 P6 留给 P7 的那个动机

P6 §7 第 1 条写的是:

> 现有策略的工具编排已经高度坍塌 …… **这正是 GRPO 会做的事,也正是分布匹配目标声称能
> 避免的。** P7 的动机从「理论上应该」变成了「已经观察到」。

**C′ 没有减轻坍塌,在九个里有七个反而更坍塌了。**

| benchmark | n | 主链路覆盖率 P4 → P7 | 链路签名种类 P4 → P7 |
|---|--:|---|---|
| cvb2drelation | 650 | 94.6% → **98.3%** | 10 → **6** |
| cvb3ddepth | 600 | 96.2% → **99.7%** | 4 → **3** |
| blinkdepth | 124 | 82.3% → **85.5%** | 11 → **9** |
| boppose | 60 | 96.7% → **98.3%** | 3 → **2** |
| robospatial | 350 | 62.3% → **64.9%** | 3 → 3 |
| refunseen | 77 | 100.0% → 100.0% | 1 → 1 |
| reflocation | 100 | 100.0% → 98.0% | 1 → **3** |
| refplacement | 100 | 100.0% → 99.0% | 1 → **2** |
| bopgrasp | 60 | 95.0% → **90.0%** | 2 → 2 |

三个 RefSpatial 上 C′ 确实多出了几条链路(277 里 3 个样本走了别的路),`bopgrasp` 的主链路
也松了 5 pp。但在**样本量大的四个**上,方向一致地更集中,签名种类还少了。

> **限定必须带上:**
>
> **(a) 跨 base,不是受控 A/B。**
>
> **(b) 训练量落在论文 Step 4 的设定之内,却落在 GFlowRL 已验证范围之外。**
> `trainer.total_epochs=1` 是上游作者写死在 `run_rl.sh:435` 的(commit `1e7ff077`,
> Siyi Chen 2026-03-24),**我们一个字没改** —— 也就是说我们训得和论文的 Step 4 一样长,
> **不能拿「训得比 SpaceTools 短」当借口。** 但 GFlowRL 自己的证据来自 **30–400+ 步
> 且 G=16** 的运行;我们是 **85 步 + G=5**,而论文自己写明估计量方差是 O(1/G)、
> 且「especially when the group size is small」。
> **所以准确的说法是:把一个在别处验证过的算法放进了一个它没被验证过的 regime,
> 而且步数与 G 两个维度同时不利。**
>
> **(c) 训练侧测到 63–80% 的组奖励退化**,那些组上梯度全部来自漂移项。
>
> **三条放在一起,这更像「这个 regime 不足以让目标函数表达自己」,而不是「C′ 的性质被证伪」。**

**但这张表只回答了一半的问题。** 按 §0 那个说明:RefSpatial / `boppose` / 深度题的「坍塌」
大部分是伪的——**任务本来就只有一条正确链路**,没有第二条等价有效的链可供保留。
所以上表真正能读的是「样本量大的四个上方向一致地更集中」,**不能读成「保住多样性失败」**。

**真正有代价的那一处,答案在 §10.6:** `robospatial` 的 `front/behind` 29 道题,
P6 判定为干净的 2a(该调没调)、上界 +8 个样本、是唯一一处「存在已知有效替代链、策略却
从不走它」的地方。实测:

```
front/behind 正确率   P4 72.4%  ->  P7 62.1%      (n=29,少 3 题)
depth_estimator 调用  P4 0 次   ->  P7 0 次       (robospatial 全部 350 个样本)
```

> **更正(2026-09-25):** 上面两行都是单次评测。P0 三次复评下这 29 题三臂无差别
> (SFT 17.3 · P4 19.0 · C′ 19.3),`depth_estimator` 也不是严格 0 次(P4 2/2/1、C′ 0/0/1),见 §0 末的更正框。
> 下面的样本走读仍然成立:两边都在目测深度。

**拿一个样本看这 29 道题是怎么答的 —— `robospatial #235`,P4 对、P7 错:**

> *Is the **monitor** behind the **sofa**?*   **GT = Yes**

```
P4   turn 1  THINK  … "Behind" means further away from the camera viewpoint, deeper into
                    the scene. Let me first identify the locations of both objects …
             CALL   roborefer.detect_one({"obj_name": "monitor"})  RESP  [(0.893, 0.156)]
             CALL   roborefer.detect_one({"obj_name": "sofa"})     RESP  [(0.529, 0.526)]
     turn 2  THINK  … the monitor is on a desk on the right side, and the sofa is in the
                    center of the room. From the camera's perspective, the sofa appears to be
                    in the foreground/middle of the room, while the monitor is on a desk
                    that's positioned f…
             ANSWER Yes    ✓

P7   turn 1  THINK  … "Behind" means further away from the camera viewpoint. Let me detect
                    both objects to analyze their spatial relationship.
             CALL   roborefer.detect_one({"obj_name": "monitor"})  RESP  [(0.893, 0.156)]  ← 逐位相同
             CALL   roborefer.detect_one({"obj_name": "sofa"})     RESP  [(0.529, 0.526)]  ← 逐位相同
     turn 2  THINK  … From the camera's perspective, the monitor is to the right and
                    appears to…
             ANSWER No     ✗
```

**两边的工具调用、参数、返回值逐位相同,链路签名都是 `roboreferx2@2t`,
而且两边都在 turn 1 正确写出了「behind = 离相机更远」。**
差别只在 turn 2 那句「appears to」—— **它们都在目测深度。**

**`depth_estimator.estimate_depth` 就在 `robospatial` 的工具表里,一次调用就能把
「谁更远」变成两个可比的数**(`cvb3ddepth` 上 600 道题正是这么做的,正确率 96.5%)。
**这 29 道题、两个 checkpoint、58 次机会,没有一次调用它。**

**C′ 没有把那条被压掉的链路唤回来,一次都没有。** 这比整体坍塌度那张表更直接地否掉了动机
——因为这一处不受「任务只有一条路」那个质疑的影响:**这里明确存在第二条更好的链路,
两个 checkpoint 都不走。**

> 逐样本看,29 道里 P4 对 21 / P7 对 18,翻转是 **P4对→P7错 5 条**(`#235 #238 #324 #333 #338`)、
> **P4错→P7对 2 条**(`#240 #313`),净 −3。**样本量太小,不足以支撑「C′ 更差」;
> 能支撑的只有「两边都没学会调深度」。**

**还有一条:P6 §7 第 2 条已经测过「推理侧可挖空间」,结果为零**(多数表决@5 在 12 个格子
上 0 个显著,VQA 方向还翻号)。**所以 P7 的动机本来就只剩「编排多样性」这一条,
而这一条现在也是负的读数。**

### 10.5 变量复用:P6 taxonomy 2d,两个 checkpoint 逐位相同

| benchmark | | 暴露 | 使用 | **未用** | 幻觉 |
|---|---|--:|--:|--:|--:|
| blinkdepth | P4 / P7 | 269 / 275 | 125 / 122 | **146 / 154** | 2 / 1 |
| cvb3ddepth | P4 / P7 | 1203 / 1200 | 600 / 600 | **603 / 600** | 0 / 0 |
| boppose | P4 / P7 | 240 / 240 | 180 / 180 | **60 / 60** | 0 / 0 |
| bopgrasp | P4 / P7 | 240 / 240 | 180 / 180 | **60 / 60** | 0 / 0 |

`cvb3ddepth` 上 **1200 个变量暴露、正好用掉 600、正好剩 600** —— 每个样本
`estimate_depth` 暴露 `$depth_map` 与 `$focal_length_px` 两个,模型每次都只用前者。

> P6 提醒过 `vars_unused` 要先扣掉常量背景。扣掉之后**真正的「拿到了却不用」几乎为零**,
> 两个 checkpoint 也没有区别。**C′ 在这一维上什么都没改变。**

### 10.6 P6 的三条判据重新跑一遍

**判据 A —— 深度题是否逐字遵守「选测得更近的那个」:**

```
blinkdepth    P4 遵守 109 违反 5 无法判定 10   95.6%
              P7 遵守 109 违反 5 无法判定 10   95.6%     <- 逐位相同
cvb3ddepth    P4 598 / 1 / 1   99.8%
              P7 598 / 2 / 0   99.7%
```

**P6 的核心洞察在 P7 上原封不动地成立** —— 在深度题上「推理」是一条规则,模型逐字执行它,
错只能错在工具给的数上。C′ 一个百分点都没动。

**判据 B —— pointing 题(= RefSpatial 三项,不含 RoboSpatial Vacant)是不是工具输出的原样透传:**

```
              P4               P7
reflocation   99/99            94/99
refplacement  100/100          98/100
refunseen     77/77            76/77
合计          276/276 = 100%   268/276 = 97.1%
```

P6 记的「276/276 原样透传」在 P4 上复现了。**C′ 让模型在 8 个样本上不再原样透传** ——
看上去像「策略开始参与了」,但下一节说明,在这个任务上参与就是变差。

> **更正(2026-09-25):** P0 同池复评 SFT 起点 272/276 · P4 276/276 · C′ 272/276 ——
> C′ 与起点相同,是 GRPO 把透传推到 276,不是 C′ 让它下降(268 是单次读数)。见 §0 更正框。

**robospatial VQA 按题型(P6 §6.4 的三分法):**

```
题型                       n     P4 正确率   P7 正确率    P4 答 yes   P7 答 yes   GT=yes
fit(要自由空间)           105     69.5%      69.5%      83/105     77/105      87
relation(2D 可判定)        94     77.7%      77.7%      69/94      73/94       60
front/behind(要深度序)     29     72.4%      62.1%      18/29      13/29       18
```

- **`fit` 和 `relation` 上两个 checkpoint 的正确率逐位相同。** P6 定位的最大那块缺口
  ——「`fit` 题几乎总是答 yes」—— C′ 把 yes 从 83 降到 77,**正确率纹丝不动**。
  **少答的 6 次 yes 没有换来任何一题。**
- **唯一移动的是 `front/behind`:72.4% → 62.1%**(n=29 上少 3 题)。这一类 P6 判定为
  **干净的 2a(该调没调)** —— 问的就是深度序、工具直接给,而 29/29 全程没调
  `depth_estimator`。

**`robospatial` 全部 350 个样本,`depth_estimator` 调用 0 次 —— P4 和 P7 都是 0。**
(SFT 起点是 1 次。)**P6 §6.4 记的这个结构性缺口,GRPO 没修掉,C′ 也没修掉。**

> **更正(2026-09-25):** 上表 `front/behind` 一行是单次评测;P0 三次复评 SFT 17.3 · P4 19.0 · C′ 19.3 / 29,
> **没有移动**。「唯一移动的是 `front/behind`」应读作「三类题型都没有可测移动」。
> `depth_estimator` 三次:SFT 0/0/0(不是 1 次)· P4 2/2/1 · C′ 0/0/1。缺口本身成立。

### 10.7 RoboSpatial 的缺口:一条能算清的因果链

**差距全部在 Vacant,VQA 是噪声。**

```
           n     P4          P7 第一次      P7 第二次
VQA       228   167 73.25%   164 71.93%    166 72.81%
Vacant    122    62 50.82%    51 41.80%     54 44.26%

丢掉(P4 对 / P7 两次都错)29 个:VQA 16 · Vacant 13
捡到(P7 两次都对 / P4 错)17 个:VQA 14 · Vacant  3
净                              :VQA −2 · Vacant −10
```

VQA 那 30 个分歧样本的答案方向是**对称的**:丢掉的 16 个里 12 个是 `gt=Yes` 从 yes 翻成 no、
4 个是 `gt=No` 从 no 翻成 yes;捡到的 14 个里 10 个是 `gt=Yes` 从 no 翻成 yes、
4 个是 `gt=No` 从 yes 翻成 no。**没有系统性方向,就是翻转噪声。**

**Vacant 的 −10 题,可以算到底。** P6 记过:`robospatial` Vacant **不是纯透传,而且模型
一改就变差**。把这条判据同时跑在两边:

```
              透传               改点                改好/改坏    原点中位距 -> 改后
P4       101 个(正确 55.4%)   21 个(正确 28.6%)   10 / 11     0.0716 -> 0.0483
P7        79 个(正确 57.0%)   42 个(正确 14.3%)   23 / 19     0.0942 -> 0.0855
```

**C′ 把「改动 roborefer 给的点」的样本从 21 个翻倍到 42 个,而改点的正确率只有 14.3%
—— 透传是 57.0%。**

**一个样本把这件事说完 —— `robospatial #34`,P4 对、P7 两次都错:**

> *In the image, there is a **microwave**. Pinpoint one point within the vacant space suitable
> for placing an object **close to and in front of** the microwave.*
>
> **GT 是 15 个点组成的一条窄带:x 0.601 – 0.862 · y 0.408 – 0.452**(微波炉前方的台面)

```
P4   turn 1  THINK  I need to find a point close to and in front of the microwave …
             CALL   roborefer.detect_one({"obj_name":
                      "point close to and in front of the microwave"})   ← 整句问题塞进查询
             RESP   [(0.722, 0.432)]
     turn 2  THINK  The tool detected a point at (0.722, 0.432) which is close to and in front
                    of the microwave … I can now format this point as … the final answer.
             ANSWER [(0.722, 0.432)]     ← 原样透传;落在 GT 窄带正中   ✓

P7   turn 1  THINK  … To find a suitable vacant space "in front of" the microwave, I need to
                    consider the floor space between the microwave and the camera/viewer …
             CALL   roborefer.detect_one({"obj_name": "microwave"})      ← 只问物体在哪
             RESP   [(0.758, 0.348)]                                     ← 这是微波炉本身
     turn 2  THINK  … Now I need to find a vacant space that is: 1. Close to the microwave
                    2. In front of the microwave (from the camera's perspective, meaning
                    between the microwave and the camera) Looking at the image, the floor
                    space in front of the microwave …
             ANSWER [(0.77, 0.80)]       ← 自己推出来的点             ✗
```

**两处变化,叠在一起:**

```
① 查询变了   P4 把整句题面交给 RoboRefer,让工具直接返回「那个点」
             P7 只问「微波炉在哪」,把「什么叫 in front of」留给自己
② 自己算     P7 把 in front of 解释成「微波炉与相机之间的地面」,于是把 y 从 0.348
             一路推到 0.80 —— 而 GT 的 y 最大只到 0.452。x 给的 0.77 其实落在 GT 范围内。
             两个坐标又都是 0.05 的整数倍(P6 量化过的「目测」指纹)
```

**P7 的推理本身读起来更周全** —— 它想清楚了「in front of 是相对相机的」,还专门去找地面。
**但它想错了场景**(微波炉在台面上,GT 指的是台面而不是地面),而 P4 那种「把问题原样交给工具」
的懒办法反而对了。

> **这就是 §10.7 那条因果链在单个样本上的样子:C′ 让策略更愿意自己接管最后一步,
> 而这一步的信息它手上没有 —— 只有工具有。**

验算与反事实:

```
P4  101×0.554 + 21×0.286 = 56 + 6 = 62 = 50.82%   ✓
P7   79×0.570 + 42×0.143 = 45 + 6 = 51 = 41.80%   ✓
若 P7 保持 P4 的透传/改点比例(101/21),按它自己的分类正确率:
     101×0.570 + 21×0.143 = 57.6 + 3.0 ≈ 61 题 = 50.0%   <- 正好回到 P4 的水平
```

> **所以 RoboSpatial 的全部缺口,可以一步不剩地归给一个行为变化:C′ 让策略更频繁地去
> 「改进」工具给的点,而这个行为 P6 已经量化过是有害的。**
>
> 这条和判据 B 的透传率下降(276/276 → 268/276)是同一件事在两个 benchmark 上的表现:
> **C′ 确实让策略「更主动」了,但在这套工具链上,主动等于变差** —— 任务的信息瓶颈在工具,
> 策略手上没有任何工具没给的信息。

> **更正(2026-09-25):本节的因果方向反了,见 §0 更正框。** 「21 → 42」比的是 P4 与 C′,
> 不是起点与 C′ —— SFT 起点就改 49.3 次,C′ 44.0,P4 21.7。上面的反事实验算是记账恒等式,
> 不说明谁是原因。按 `p1_monitor.py --upper` 拆开:改点主要发生在工具点错的时候,强制透传反而掉分;
> P4 的优势在工具点本身更准(60.3 vs 49.3 / 122)。#34 这个样本仍然成立,但它说明的是
> 「查询写得好不好」,不是「策略主动就变差」。

### 10.8 错题归因:P6「322 个错题 100% clean」的结论在 P7 上同样成立

```
                    P4 错题归因              P7 错题归因
robospatial         干净 121                 干净 134 · 未调用工具 1
reflocation         干净 46                  干净 48
refplacement        干净 42                  干净 42
refunseen           干净 40                  干净 40
blinkdepth          干净 17                  干净 15
cvb2drelation       干净 35                  干净 35
cvb3ddepth          干净 21                  干净 21
boppose             干净 22                  干净 21
bopgrasp            工具失败 5 · 干净 1      工具失败 4
```

OOM、生成截断、工具响应截断、顶轮数、畸形 tool_call、答案解析失败 —— **两边全部为 0。**

> **这条让上面所有因果解释是干净的**:不是环境、不是显存、不是截断。

### 10.9 与 P5 结论的关系

P5 的几条口径结论在 P7 上全部照旧成立,不需要重新论证:

- **十个 Table 2 数字里只有九个独立**(Overall 是 VQA/Vacant 的样本加权)。
- **解码是 greedy**,`rollout.n=5` 是训练用的 group size,不作用于评测。
- **`gpu_memory_utilization` 是整卡比例** —— P5 §4.5 列为「新增偏离」,P7 按卡容量反算成
  0.545 把池子对齐回 24 GB,口径一致。
- **blinkdepth 要报稳定上限并附区间**(P4 三跑 86.29–87.90,上限 112/124 = 90.32)。
  **P7 单次 87.90 正好是 P4 区间的上端,不是「比 P4 高」。**
- **BOP-ASK Pose 的指标映射仍未解决**(论文 34.37 在 n=60 下不是比例:
  `60 × 0.3437` 非整数,而同一检验对其余六个比例型数字全部成立)。
  **P7 的 55.73 与 P4 的 53.36 之间的差不能当成能力差。**

P5 §3.1 记的「RoboSpatial VQA −6.13 pp 是唯一超噪声的缺口」—— P7 在 VQA 上与 P4 只差
1–3 题,**这个缺口 P7 既没扩大也没缩小,它是继承下来的。**

**材料:** `06_gflowrl_eval/comparison_p4_p5_p6_vs_p7.md` ·
`eval/GFlowRL/P7_GFLOWRL_EVAL/analysis/`(四个脚本 + 完整输出)·
`spacetools-repro/records/`(P2–P7 全部报告)· `spacetools-repro/p4/`(dumps / parsed / logs)

---

## 11. 结论

**1. 没有任何 benchmark 变好,反方向同样不显著。** 九个 benchmark 的 McNemar 精确检验最小
p 是 0.092。**「P7 在某个 benchmark 上比 P4 好」这句话没有一条证据支持**,
「比 P4 差」也一样。

**2. 与自己的 SFT 起点比,C′ 在能做前后对比的地方没有产生可测的提升。**
RoboSpatial 差 4 题(1.5 σ)、RefSpatial 加权逐位相同、Vacant 与起点完全一致。
官方 ckpt 领先的那约 11 题没有被追上任何一题。

**3. 这和训练侧的指标自洽,不是意外。** 退化组 70.3%、`grad_norm` 中位 180 而
`grad_clip=1.0`、1 epoch 85 步(上游写死)、G=5(论文用 16)。
**一个在约 70% 的样本上收不到奖励信号、每次更新又只剩方向的目标函数,跑 85 步之后与起点
分不开,是预期内的结果。** 注意这里的 85 步**不是我们偷工减料** —— 论文的 70.0 就是这么
训出来的;真正的问题是 GFlowRL 的证据来自 30–400+ 步且 G=16 的运行,
**我们把它放进了一个它没被验证过的 regime。**

**4. P6 留给 P7 的动机(分布匹配能避免编排坍塌)本次实测是负的读数,而且两个口径都是负的。**
整体坍塌度上,样本量大的四个 benchmark 方向一致地更集中(这一条要打折,因为 RefSpatial 等
的「坍塌」大部分是任务本身只有一条路);**而不受这个质疑影响的那一处 —— `front/behind`
29 道题、`depth_estimator` 全程 0 次调用 —— C′ 一次都没有把它唤回来。**
(2026-09-25 更正:三次复评下 C′ 调过 1 次、P4 调过 5 次,这 29 题正确率三臂无差别;
结论方向不变 —— 两边都没学会调深度。)
读数被三件事削弱(跨 base、85 步、G=5),所以它证伪的是「C′ 在这个 regime 下能做到」,
**不是 C′ 的理论性质不成立。**

**5. C′ 唯一可测到的行为改变是有害的,而且能算到底。** 透传 276/276 → 268/276;
Vacant 改点 21 → 42(改点正确率 14.3% vs 透传 57.0%);RoboSpatial 的 14 题缺口
一步不剩地归给这个行为。**在这套工具链上,策略「更主动」等于变差。**
(2026-09-25 更正:方向反了,见 §0 更正框 —— 起点就改 49.3 次、透传 272/276,C′ 与起点相同,
是 GRPO 压住了;P4 的优势主要来自更好的工具调用,强制透传反而掉分。)

**6. P6 定位的三块结构性缺口,C′ 一块也没动** —— `fit` 题 69.5% → 69.5% 逐位相同;
`robospatial` 全程 0 次 `depth_estimator`(P4 也是 0;三次复评为 P4 2/2/1、C′ 0/0/1);深度题规则遵守率逐位相同。
**这三块 P6 都判定为工具集缺口或该调没调,换目标函数动不了它们,是符合 P6 预期的。**

**7. 九个 benchmark 里只有两个能测出策略差别。** 777 个样本上两个 checkpoint 字面相同。
**以后设计 RL 对照实验,算力应该只投 `robospatial` 和 `blinkdepth`。**

### 方法学上值得带走的

- **闸门要先写死。** 判据、预登记、职责分离(「loss 写得对不对」由合成不动点检验单独守住)。
- **判读顺序不能颠倒:先数 OOM → 再查工具静默错误 → 最后才看分数。** 本次真的抓到了
  一次静默污染(cvb3ddepth 243 错里 230 归因 OOM),而它产出的是一个「看着合理」的数字。
- **判据不能只信退出码。** 一个脚本失败时 `exit 0`,另一个成功时 `exit 15`,两种都遇到了。
- **保留 dump 是这次能得出结论的唯一原因。** P4 当年存了 dump,所以今天能做逐样本对比;
  SFT eval 没存,所以真正受控的那一组对比做不了。
- **判据要写成区间,不要写成点阈值。** SFT 的 `RoboSpatial ≥ 60` 正好落在测量的翻转带内,
  单次读数在线上还是线下由噪声决定。

---

## 12. 局限与下一步

### 12.1 局限(按严重程度)

1. **跨 base,不是受控 A/B。** P4 是论文 base + GRPO,P7 是自训 SFT + C′。
   所有「C′ 造成了 X」严格讲只能说成「P7 这个 checkpoint 表现为 X」。
2. **SFT 起点没有 dump,受控的 SFT ↔ C′ 逐样本对比做不了。** 只能比汇总数
   (robospatial 主链路 SFT 64.3% → C′ 64.9%、调用 1.649 → 1.646,说明坍塌在起点就已经
   如此)—— 但那是汇总量,不是逐样本证据。
3. **单臂,没有跑 GRPO 对照。** 论文的 GRPO 臂没有在我们的 base 上复现过。
4. **单 seed,且 P7 只有 robospatial 有两次运行。** P4 对三个 benchmark 是多跑报区间的。
5. **算法被放在一个它没被验证过的 regime 里,而且是两个维度同时不利。**
   - **步数:1 epoch / 85 步。** `trainer.total_epochs=1` 是上游作者写死在
     `run_rl.sh:435` 的(commit `1e7ff077`),我们没动 —— **论文的 70.0 就是这么训出来的,
     所以「我们训得比 SpaceTools 短」这个借口不成立。** 但 GFlowRL 的证据来自
     **30–400+ 步**的运行。
   - **G:`rollout.n = 5` 而 GFlowRL 用 16。** 论文写明估计量方差是 O(1/G)、
     Appendix A 的第一条局限就是「especially when the group size is small」——
     **5 相对 16 是 3.2× 的方差,我们正好站在最不利的一侧。** 改它会让两臂不可比,
     所以没改;但这意味着**本次结果不能外推到 G=16**。
   - 两者叠加:**这是「一个在别处验证过的算法,被放进了它没被验证过的 regime」,
     不是「训练量不够」。** 要真正检验 C′,需要的是 G=16 或更长的运行,
     而那已经不是复现 SpaceTools 的 Step 4 了。
7. **混合 ECC 让 KV 池不受控**(策略落哪张卡池子差约 1.7 GB),这是 §9.6 那 4 题差的一个
   未受控解释。
8. **`bopgrasp` / `boppose` 的奖励规格本身有问题**(NCE 对夹爪朝向不敏感),已明确排除在
   作用域外,但它们仍出现在结果表里,读的时候要记得。
9. **归因判据沿用 P6 的实现**,包括 P6 自己记下的边界(3b 归属有争议;人工那 31 条只有
   一轮标注、无标注一致性度量)。
10. **P4 ↔ P7 逐样本对比的两边,KV 池本来就不一样。** P4 是 40 GB 卡 × `gmu=0.5` =
   **20 GB 池**;P7 是 46 GB 卡 × `gmu=0.545` ≈ **24 GB 池** —— 后者对齐的是**SFT 基线**
   那次的 24 GB(§9.2),不是 P4。§10 那张总表两边差 4–5 GB。
   按 §9.2 那组消融的量级(池子差一倍 → `blinkdepth` 翻 8 题、净 2 题),4–5 GB 的影响
   大概率小于表里的噪声,**但这是一项确实存在、此前没写出来的未受控差异**,
   和第 7 条是同一件事的两个侧面。
   > 三个池子口径放在一起,免得再混:
   > `P4 20 GB` · `SFT 基线 24 GB` · `P7 ~24 GB(对齐 SFT,非 P4)`。
   > **要做到三边同池,只能重评** —— 与 §12.2 第 1 项是同一次机时。

### 12.2 下一步,按性价比排

| # | 做什么 | 成本 | 能回答什么 |
|---|---|---|---|
| 1 | **在同一台机器、同一 session 里评 SFT 起点的 robospatial** | ~35 min | 把 §9.6 那 4 题的差从「机器/模型分不开」变成受控读数。**性价比最高的一步** |
| 2 | **robospatial 与 blinkdepth 两边各跑 3 次** | ~3 h | 把「无一显著」升级成有功效的区间结论 |
| 3 | **评 step 30 / 60** | ~1.5 h | 确认不是训练晚期把早期收益磨掉了。**注意这是事后挑 checkpoint,报告必须写明评了几个** |
| 4 | **跑 GRPO 对照臂**(同 base、同 seed、同机) | 3–4 天机时 | 唯一能回答「C′ vs GRPO」的做法 |
| 5 | **G 从 5 提到 16 再训一次** | 3–4 天机时 | 论文的估计量方差论断在我们的 setting 上成立吗 |
| 6 | **把 `front/behind` 那 29 道单独做奖励塑形** | 中 | P6 定位的唯一一处「存在已知有效替代链、策略却从不走」的地方,上界 +8 题(P0 三次均值下约 +10 题;SFT 数据里这条链为 0 条,需先补支撑集,见 Design Doc P3) |

| 7 | **eval 时存 top-k logprob**(dump 补丁,`patches/rl/0009` 那个形状) | 小 | §9.2 那条「批组成 → 归约顺序 → 平局翻转」目前只能从结果反推;存了 logprob 才能直接读出分叉点两个候选 token 的间距,把因果钉死 |

> **不建议做的:** 换工具(正交路径,混在一起不可归因)、在 `fit` 题上调 prompt
> (P6 判定为工具集缺口 —— 模型缺的是信息,不是分布)。

---

## 附录 A:产物清单

```
training/SFT/
  env/2x_A6000_REPORT.md              环境构建(阶段 A–C)+ 30 步探测
  env/training_report/REPORT.md        4× A6000 全量 SFT 执行报告
  env/bundle/                          4.0 GB 可迁移环境包 + RESTORE.sh
  model_rl_start/                      SFT checkpoint 本体(π_ref)
  SFT训练学习笔记.md                    配置来源、显存账、ZeRO 取舍

training/GFlowRL/
  A40/P7_GFLOWRL_训练报告.md            训练报告
  A40/metrics/p7_metrics.csv           85 行,每行一步
  A40/logs/ · A40/config/ · A40/scripts/   as-run 日志 / 环境快照 / 脚本
  diff/*.patch                         三个 patch = 全部代码改动
  docs/                                C 路闸门 / 计划 / 步骤 23 结果

eval/GFlowRL/
  P7_GFLOWRL_EVAL报告.md               eval 报告
  P4_P5_P6_vs_P7_对比分析.md            逐样本对比
  P7_GFLOWRL_EVAL/
    dumps/<run>/<benchmark>/0.jsonl    完整轨迹(含工具真实返回),对错都在里面
    parsed/<benchmark>.jsonl           结构化记录,与 p4/parsed 同构
    parsed_runC/robospatial.jsonl      robospatial 第二次
    logs/ · gates/ · scripts/ · gpu/ · config/ · MANIFEST.txt
    analysis/compare_p4_p7.py          逐样本配对
    analysis/robospatial_divergence.py 分歧样本的方向与题型
    analysis/p6_criteria_p4_vs_p7.py   P6 判据 A/B、题型三分、Vacant 透传-vs-改点
    analysis/significance_p4_p7.py     McNemar + 符号检验

spacetools-repro/                      GitHub: qyYue1389/spacetools-repro
  records/                             29 份报告(P2–P7)+ CHANGES.md + PROVENANCE.txt
  p4/dumps/run1..4/ · p4/parsed/ · p4/logs/     官方 ckpt 的轨迹与结构化记录
  p6/                                  消融与探针(fp32 / gmu025 / passk / swap / probes / manual)
  tools/parse_dump.py · tools/p5/ · tools/p6/ · tools/p7/
```

## 附录 B:可复算的入口

```bash
# 从 dumps 重新生成 parsed(不需要 GPU)
python3 spacetools-repro/tools/parse_dump.py \
    eval/GFlowRL/P7_GFLOWRL_EVAL/dumps/runB-rerun4/cvb3ddepth/0.jsonl \
    --emit eval/GFlowRL/P7_GFLOWRL_EVAL/parsed

# 重跑全部对比分析(不需要 GPU)
cd eval/GFlowRL/P7_GFLOWRL_EVAL/analysis
python3 compare_p4_p7.py            # 逐样本配对 + 总表
python3 p6_criteria_p4_vs_p7.py     # P6 判据 A/B + 题型 + Vacant
python3 significance_p4_p7.py       # McNemar + 符号检验
python3 robospatial_divergence.py   # 分歧样本

# 裸机 -> 九个 benchmark 的分数(需要 4 张 ≥40 GB 的 sm_80/86/89 卡)
export HF_TOKEN=...
P7_STEP=85 bash eval/GFlowRL/P7_GFLOWRL_EVAL/scripts/EVAL_FROM_SCRATCH.sh
```

**口径(引用任何分数都要带上):**

```
gpu_memory_utilization=0.545(KV 池对齐到 24 GB)· NUM_GPUS=4 EVAL_GPUS=1
4× RTX A6000 · vlm num_gpus=1.0 · greedy 解码
ckpt: qzpm55555/spacetools-p7-gflowrl-cprime-8xa40 @ global_step_85
env:  qzpm55555/spacetools-eval-env
data: siyich/spacetools-eval-benchmarks @ 1d539ac9
```
