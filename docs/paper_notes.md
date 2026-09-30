# 项目论文笔记(2 篇)

> 用途:新会话开始时先读这份笔记,即可接上上下文,不必重读 PDF。
> 覆盖:`2512.04069v2.pdf`(SpaceTools / DIRL)、`2607.13394v1.pdf`(GFlowRL)。
> 记录时间:2026-08-24(含 SpaceTools 代码库勘察)

---

## 一、SpaceTools:Tool-Augmented Spatial Reasoning via Double Interactive RL
arXiv 2512.04069v2 · NVIDIA + University of Michigan · 2026-06 修订 · **已被 CVPR 2026 接收**

### 问题
VLM 的定性视觉理解不错,但缺乏具身应用所需的**度量级精确空间推理**(距离、位姿、抓取、遮挡、3D 关系)。
两条已有路线都不理想:
- 在任务数据上做 SFT → 需要大规模标注,每加一种感知能力(深度、pointing、3D)几乎都要改架构或重新收数据;
- 让 VLM 调用视觉工具 → 但现有做法要么靠手工 prompt,要么写死固定工具流水线,模型无法自行发现最优工具组合。

ViGoRL 证明了 RL 能学会用**单个**视觉工具(裁剪),但工具一多,动作空间组合爆炸,朴素 RL 探索直接失败。

### 方法:DIRL(Double Interactive RL)
核心洞察:**单工具 IRL 可收敛且能教会 grounding;多工具 IRL 能精炼推理但需要好的初始化**。所以把 RL 用两次:

1. **Teaching phase(教学阶段)**
   - 先用 IRL 训一个只会用 pointing 工具(RoboRefer)的 single-tool 专家 → 生成 2k 条 grounded reasoning 轨迹;
   - 再用 frontier model(**Claude Sonnet 4.5**)接上全套工具解题,只保留答对的轨迹 → 6k 条;
   - 两者按 1:3 混成 8k 教学集,对 base model 做 SFT(学会工具签名、输出格式、信息流)。
2. **Exploration phase(探索阶段)**
   - 从 SFT 权重继续做 IRL(GRPO + KL),开放全部工具,精炼工具链式调用与纠错策略。

对话格式:`<think>` / `<tool_call>` / `<answer>`,多轮直到出答案或到 T_max。

**奖励设计**(全部归一化到 [0,1]):选择题二值;2D bbox 用 MIoU;pointing 用 NNDC(到目标区域质心的归一化负距离,并与二值项取 max 做 clip);位姿用 8 个投影角点凸包 IoU;抓取用 NNCE。试过加 format reward,无增益,最终未用。

### Toolshed(系统贡献,已开源)
基于 Ray 的分布式工具托管平台,解决"训练时在线调用重型 CV 工具"的工程瓶颈:
- 执行与 policy 推理循环解耦(避免一次阻塞调用卡死整个 batch);
- 每个工具多 actor 异步并行、资源隔离、**Python 环境隔离**(解决多 CV 模型依赖冲突);
- 支持文本/图像/结构化变量(点云等)跨节点传递;与 VERL 的异步多轮 rollout 天然契合。
- 工具:SAM2 分割、point1(RoboRefer)/point2(Molmo) 两套 pointing、DepthPro 深度+点云、3D bbox 拟合、GraspGen 抓取、image_ops、code_executor;机器人工具:capture_image / capture_depth / execute_grasp / place_object(另有 mock 版本用于训练)。
- 效率:8 并发调用下比朴素 HTTP 部署快 3.2×;整条 pipeline 延迟 20.2s → 10.6s。

### 实验结果
Base model = **Qwen2.5-VL-3B-Instruct**(只训 LLM 部分,2.55B 可训参数;视觉编码器冻结)。

- 空间推理 benchmark(RoboSpatial-Home / BLINK / RefSpatial / CVBench / BOP-ASK)几乎全 SOTA:
  RoboSpatial overall **70.0**(超 Gemini-ER 1.5 +7.5);Pose **34.37**(超 Claude Sonnet 4.5 +24.4);Grasp-SR **50.0**(超 GPT-5 +8.3)。
- 同一 8k 数据、同一 base model 的无工具对照:比 tool-free SFT **+12%**、比 tool-free RL **+16%**(RoboSpatial)。
- 真机操作(Kinova Jaco + ZED2 + CuRobo,机器人本身作为工具):Pick 86%、Relational Pick 83%、Pick&Place 86%,均优于 Claude Sonnet 4.5 + Toolshed 与 GPT-5 + Toolshed;TTFM 10s(它们 30–36s)。π0.5 全 0。
- 消融(Table 4):去掉 IRL teacher → 52.48→41.68;去掉 universal teacher → 42.86(pose 崩到 8.92);去掉 Stage-2 IRL → 50.99。Tool SFT / Tool NIRL 分别低 13.4 / 14.4 分。**直接对所有任务所有工具做 IRL → 19.79,几乎不可学**。
- 有意思的泛化:只在 RoboSpatial 上用 pointing 工具做 IRL 的模型,在未见的 RefSpatial 上拿到 34.3%,而其他微调法都是 0。
- 零样本给 GPT-5 / Claude 挂上 Toolshed:需要精确几何的任务(RefSpatial、pose)明显涨,但 RoboSpatial/BLINK 这类高层任务反而略降——模型**过度调用工具、误读工具输出**。

### 局限
只做短/中时程任务;工具输出主要是文本/结构化变量,尚未充分利用图像型工具输出;训练时用 mock robot(真机在环太慢);抓取仍是最弱项(30/60 失败里 23 例是工具错);grasp/pose 的瓶颈在杂乱场景下的目标检测。

### 代码库(2026-08-24 查阅)
主仓库 <https://github.com/spacetools/SpaceTools> · 项目页 <https://spacetools.github.io/> · 当时 34 stars / 1 fork。
是一个 **submodule 聚合仓库**,三个子模块 + 建环境脚本:

| 组件 | 地址 |
|---|---|
| Toolshed(工具托管) | `NVlabs/SpaceTools-Toolshed` |
| SFT(LLaMA-Factory fork) | `ChicyChen/SpaceTools-SFT` |
| RL + eval(verl fork) | `ChicyChen/SpaceTools-RL` |
| SFT 数据集 | HF `siyich/spacetools-sft`(~7900) |
| RL 数据(point tools) | HF `siyich/spacetools-rlpointtools`(~4000) |
| RL 数据(full tools) | HF `siyich/spacetools-rlfulltools`(~5500) |
| 评测 benchmark | HF `siyich/spacetools-eval-benchmarks` |
| **预训练 checkpoint** | HF `siyich/spacetools-ckpt` |

**四步流水线**(README 把论文两阶段拆成 4 步):Step 1 point-tool RL(GRPO,只用 `detect_one`)→ Step 2 teacher 数据采集(Claude + Toolshed)→ Step 3 SFT → Step 4 full-tool RL(GRPO,11–17 工具在线)。
**Step 1–2 的产出已预先发布在 SFT 数据集里,多数人可直接从 Step 3 开始。**

复现成本:推荐 **2 节点 × 8×A100-80G**(一节点跑 Toolshed 工具 actor,一节点训练);Step 1 ~10–15h(15 epoch),SFT ~3–4h(单节点 8 卡,3000 步),Step 4 ~8–12h(1 epoch ≈ 86 步,每步 5–7 分钟)。~100GB 存储。所有脚本支持自动 resume(为 SLURM 短墙钟设计)。
需手动下两个工具权重:`Zhoues/RoboRefer-8B-SFT`、Apple `depth_pro.pt`。

**比论文多出来的实现细节**:
- 两套工具配置:`toolshed_v1_config.yaml` = 11 工具(纯推理),`toolshed_v2_config.yaml` = 17 工具(含机器人)。SFT 也分 v1/v2 两版。
- RL 侧是 verl 的新 agent loop:`AgentLoopManager → ToolAgentLoop`,推理用 **sglang**,工具调用解析用 **hermes** 格式。这对应论文附录里"新版 verl 的 agent loop 缓解了训练不稳定"那段。
- SFT 明确 vision tower 冻结、只训语言模型,工具 schema 注入 system prompt。
- 评测 9 个 benchmark key:`robospatial` / `reflocation` / `refplacement` / `refunseen` / `blinkdepth` / `cvb2drelation` / `cvb3ddepth` / `boppose` / `bopgrasp`。
- 六个 conda 环境(sft / rl / tool-roborefer / tool-vlm / tool-bbox / tool-graspgen),统一 Python 3.11 + Ray 2.47.1 保证跨环境兼容。
- 作者声明正在把 Toolshed 的多轮工具训练**上游合并**进 LLaMA-Factory 和 verl,目标是最终不需要 fork。

**Toolshed 单独看**(可复用性最高的一块):继承 `BaseTool` + `@tool_method` 装饰器 + Google 风格 docstring(自动转成 LLM schema),返回 `ToolResult`(value / text / images / variables)。启动方式三种:`toolshed-launch --config`、Python `start_toolkit()`、`web_ui.py`。要求 Linux + Python 3.11 + CUDA 11.8+,视觉工具需 ≥40GB VRAM。**README 里没有明确的 license 声明**(主仓库和 Toolshed 都没看到),要用得先确认许可。另外文档自己提醒 code_executor **没有沙箱**,执行生成代码有安全风险。

---

## 二、GFlowRL:Scaling Distribution-Matching RL to Large Language Models
arXiv 2607.13394v1 · Microsoft Research · 2026-07

### 问题
GRPO/PPO 这类**奖励最大化**目标会把概率质量压到单一高奖励模式上,导致 mode collapse、解法多样性丧失。GFlowNet 提供另一条路:**按奖励比例采样**(distribution matching),天然保留多条高奖励推理路径。
但已有的 GFlowNet 式 LLM 后训练(FOR、FlowRL)都要学一个 prompt 条件的配分函数 Z_φ(FlowRL 用 3 层 MLP 接在 prompt 最后隐状态上),在真实后训练规模下会炸。

### 根因诊断:learned partition function 是罪魁
**学习时程失配**——policy 是预训练好的数十亿参数模型,只需微调几百步;Z_φ 是随机初始化、要从零学一个复杂量,同样只有几百步。于是训练大部分时间里 log Z_φ 基本就是个噪声函数。

两条实证:
1. **它没用**:把 Z_φ 换成 `N(0.5, 1)` 的随机采样,性能不降反微升(36.19 vs 35.61)。合成三峰高斯实验里 FlowRL 与 FlowRL-RandomLogZ 行为几乎一模一样,都拟合不出多峰结构。
2. **它有害**:421 步里 FlowRL 有 **55 步梯度范数 ≥ 1e6**,均值 3.2e14、最大 9.6e16;GRPO / GFlowRL 的均值分别只有 0.24 / 0.07,最大值 <6.2。

### 方法:GFlowRL
把学到的 log Z_φ(x) 换成**批内 Monte Carlo 估计**——反正 GRPO 式训练每个 prompt 本来就要采 G 条 rollout:

```
Z_t(x) = (1/G) Σ_i [ β·r(x,y_i) + log π_ref(y_i|x) − log π_old(y_i|x) ]
```

这是利用 TB 最优点的性质:每条轨迹隐含同一个目标值。Z_t 以 **stop-gradient baseline** 的形式进入残差,不带任何梯度,于是辅助网络、它的 optimizer 状态、分布式同步全部消失。

两个稳定器:
- **importance sampling 修正** rollout policy 与 trainer policy 的漂移,`w = min(π_θ/π_old, 1+ε)`;
- **非对称 flow-gap clipping**:把在 rollout policy 下评估的 flow gap g 裁到 `[−ε_low, +ε_high]`,ε_low < ε_high(数学任务上正确解早期欠采样,往上推的力度应大于往下压)。
- 另外按 response 长度归一化 log 概率,防长序列主导 loss。

理论(Appendix B):clip 未激活且不做长度归一化时,零损失的自洽不动点满足 `π_θ ∝ π_ref · exp(β r)`,即保住了 TB 的目标不动点;不动点处 flow gap = 0 所以 clip 在最优点邻域天然失效,不改变稳态分布。长度归一化会引入与长度相关的轻微偏差(各 rollout 长度接近时可忽略),作者把它定位成工程折中。

### 实验结果
- **7B 数学**(Qwen2.5-7B,6 个 benchmark,Avg@16):GFlowRL **40.92**,vs GRPO 32.48(+8.44)、FlowRL 35.63(+5.29)。32B 上同样最好(50.42 vs FlowRL 48.39)。
- **代码**(DeepSeek-R1-Distill-Qwen-7B):LiveCodeBench 38.62、Codeforces 1646 Elo / 88.0 百分位、HumanEval+ 84.93,三项全胜。
- **14B Codeforces:2048 Elo**,超 DeepCoder-14B +112、FlowRL-14B +144、OpenAI o1 +157,距 o3-mini(2073)仅 25 分。408 题逐题对比 DeepCoder:GFlowRL 独解 108 题,DeepCoder 独解 2 题。
- **红队攻击**(SEMA setting,奖励更稀疏更噪):AdvBench ASR@1 平均 **82.5%**、HarmBench **79.5%**,超前 SOTA 多轮攻击 SEMA +2.4 / +4.5;**FlowRL 在这个 setting 直接不收敛**。
- **MoE**:Qwen3-30B-A3B 数学 78.32(backbone 74.52、GRPO 75.78),Codeforces 1999 Elo(仅 3B 激活参数,超 o1 +108);Qwen3-235B-A22B 直接复用 30B 超参,只训 30 步(GRPO 训了 100 步)仍取得 83.35 vs GRPO 82.40。**FlowRL 在所有 MoE 设置下均不收敛**。作者称这是首个在 dense 和 sparse 架构上都能稳定训练的 GFlowNet 式 RL 算法。
- 消融:β 在 [1,10] 内不敏感,峰值 β=8;去掉 flow-gap clipping → 掉 3.9 分,梯度均值涨 6.3×、最大值近 3 倍。多样性打分(GPT-o4-mini,1–5):GRPO 1.21 / PPO 1.15 / FlowRL 2.64 / **GFlowRL 3.93**。
- 另一条对照:ConstantLogZ 只有 25.34(backbone 23.02),FOR 29.53 —— 说明**怎么估 log Z 才是关键**,不是简单去掉就行。

### 局限
批内 MC 估计在 group size 小时方差偏大(靠两个稳定器压住);只验证了数学/代码/红队三个语言域,是否推广到 agentic、多模态 RL 未知。

代码计划开源于 <https://github.com/microsoft/gflowrl> —— **2026-08-24 查证仍为 404,尚未放出**。

---

## 三、两篇的关系

表面上一篇是"VLM + 视觉工具的空间推理",一篇是"LLM 后训练的 RL 算法",但放在一起看有几条清晰的连线:

| 维度 | SpaceTools / DIRL | GFlowRL |
|---|---|---|
| RL 算法 | GRPO + KL(奖励最大化) | GFlowNet TB 目标(分布匹配) |
| 核心痛点 | 多工具动作空间组合爆炸,朴素 RL 探索失败 | 学习式配分函数 Z_φ 带来梯度爆炸,规模化失败 |
| 解法思路 | **课程化**:先窄后宽,单工具 IRL → SFT 初始化 → 全工具 IRL | **做减法**:删掉辅助网络,用 rollout group 的批内 MC 估计替代 |
| 稳定性手段 | 好的初始化防止探索坍塌;奖励归一化到 [0,1] | IS 修正 + 非对称 flow-gap clipping;估计量与 r/β 同量级 |
| 规模 | 3B VLM | 7B–235B,dense + MoE |

**共同主题**:两篇都在处理"**RL 在大动作空间/大模型上不稳定**",而且给出的答案都不是加机制,而是——SpaceTools 把探索**分阶段**做,GFlowRL 把不该学的东西**删掉**。GFlowRL 结论那句话说得很直白:"scaling GFlowNet-style RL 更多取决于识别原目标里哪些部分在这个 regime 里是不必要的,而非添加辅助机械。"DIRL 的消融(直接全工具 IRL → 19.79)本质是同一类失败:优化信号被淹没在过大的搜索空间/方差里。

**可能的交叉点(尚无人做,值得想)**:
1. SpaceTools 的 Stage-2 IRL 用的是 GRPO,而工具调用天然存在**多条等价有效的工具链**(先 point 再 segment vs 先 depth 再 point)。GRPO 会坍塌到单一工具流水线;换成 GFlowRL 的分布匹配目标,理论上能保留多样的工具编排策略——这正是 GFlowRL 多样性打分 3.93 vs GRPO 1.21 想要的那种效果。
   *工程上现在是可行的*:两边都建在 verl 上(SpaceTools 是 verl fork + Toolshed;GFlowRL 也基于 verl),且 SpaceTools 已放出 SFT checkpoint 和 full-tool RL 数据集(~5500),等于 Step 4 之前的东西都白送。把 GFlowRL 的 loss 换进 `run_rl.sh` 是一个规模可控的实验。
2. SpaceTools 论文自己在 limitation 里提到"stepwise reward 可能对大规模多工具动作空间更有效",而 GFlowRL 是 trajectory-level 的;两者对"如何在长工具链上做信用分配"的答案是互补的。
3. GFlowRL 在**奖励噪声大**的场景(红队)才显出对 FlowRL 的绝对优势。工具调用的奖励同样噪声大(工具本身会出错——SpaceTools 60 次抓取失败里 23 次是工具错),这是 GFlowRL 的主场。
4. 共同引用基础:GRPO(Shao et al. 2024)、VERL 框架、Qwen 系列 backbone。SpaceTools 用 Claude Sonnet 4.5 当 universal teacher,GFlowRL 用它当红队 victim/baseline 之一。

---

## 四、快速索引(想细查时看 PDF 哪部分)

**SpaceTools**:Alg. 1(多轮工具循环)/ §4.1 DIRL 两阶段 / §4.3 奖励公式 / Table 2 主结果 / Table 3 真机 / Table 4 消融 / Table 5 给 GPT-5、Claude 挂工具 / Appx A.2 完整工具 API / Fig. 7 system prompt / Appx E.4 奖励与 prompt 消融。

**GFlowRL**:§3.1 为什么 Z_φ 失败(Table 1 梯度统计)/ Eq. 4 批内估计量 / Eq. 5–8 完整 loss / §4.2 random-log Z 诊断 / Table 2–5 主结果 / Table 6–7 消融 / Appx B 不动点证明 / Appx D 全部超参 / Appx I 与 DeepCoder 的逐题定性对比。
