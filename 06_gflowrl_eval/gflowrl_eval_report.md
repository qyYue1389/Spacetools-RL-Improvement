# P7 GFlowRL(C′)checkpoint eval 报告

对象:`qzpm55555/spacetools-p7-gflowrl-cprime-8xa40` 的 `global_step_85`
执行:2026-09-17 ~ 09-18 · RunPod 4× RTX A6000
配套:`05_gflowrl_training/gflowrl_training_report_8xA40.md`(训练侧)、`03_sft_eval/sft_eval_report.md`(起点)、`P4→P5 交接文档(未收录)`(官方 ckpt 对照)

---

## 摘要

把 SpaceTools Step 4 的 GRPO 换成 GFlowRL 的 **C′** 目标、训了 85 步之后,在论文 Table 2
的九个 key 上评了一遍,**2121 个样本零 OOM、零截断、零顶轮数、零缺 `<answer>`**。

结论一句话:**在能做前后对比的地方,一项都没有比 SFT 起点更好,也没有更差。**

```
RoboSpatial Overall   SFT 61.00(两次 216/211)  ->  C′ 62.14(两次 215/220)
                      差 4 题,约 1.5 个标准误,不构成有统计意义的提升
                      官方 ckpt 65.43–66.00 的差距(约 11 题)完全没有被追上
RefSpatial 三项       52.58 -> 52.68 简单平均;加权两边都是 53.07
其余五个 benchmark    没有 SFT 起点可比(当初只评了四个 key),
                      与官方 ckpt 持平或略好,主要证明环境复现正确
```

---

## 1. 评什么、怎么评

### 1.1 被评的是一个系统,不是一个模型

策略模型(Qwen2.5-VL-3B,本次是 C′ 的 step85)由 sglang 起,七个工具是 Ray actor,
**各自跑在自己的 conda 环境里**(五个环境的依赖互相冲突:torch 2.3.1 / 2.5.1 / 2.9.1,
numpy 1.26.4 与 2.4.6 并存)。模型看图 + 空间推理问题,`<think>` → 调工具 →
`<tool_response>` 回注 → 继续推理 → `<answer>`,最多 8 轮。

解码是 **greedy**(verl 的 `val_kwargs` 默认 `temperature 0 / top_p 1.0 / n 1 /
do_sample False`;脚本里的 `rollout.n=5` 是训练用的 group size,不作用于评测)。

### 1.2 九个 key

| key | n | 指标 |
|---|--:|---|
| robospatial | 350 | Yes/No 准确率(228)+ 点凸包归属(122)的样本加权 |
| reflocation / refplacement / refunseen | 100 / 100 / 77 | 点凸包归属 |
| blinkdepth | 124 | 选项准确率 |
| cvb2drelation / cvb3ddepth | 650 / 600 | 选项准确率 |
| boppose | 60 | 8 个投影角点两组凸包的 **IoU**,连续值不是准确率 |
| bopgrasp | 60 | 复合分(仓库脚本可拆成 MACE / SR) |

### 1.3 三次运行

```
Run A  2026-09-17 23:40:25 -> 09-18 01:45:27  2 h 05 m   九个 key
       其中五个(robospatial / reflocation / refplacement / refunseen / cvb2drelation)
       零 OOM,结果有效;另外四个被显存问题污染,结果作废(见 §4.2),本报告不记它们的分数
Run B  09-18 03:06:46 -> 03:53:18             46 m       被污染的那四个重跑,零 OOM
Run C  09-18 04:16:51 -> 04:36:36             19 m 45 s  robospatial 第二次,零 OOM
```

判据顺序是项目定死的、**不能颠倒**:**先数 OOM → 再查工具静默错误 → 最后才看分数**。
理由见 §4.2:OOM 不让脚本崩,只让分数静默变低。

---

## 2. 环境与卡怎么配的

### 2.1 机器

```
GPU        4× NVIDIA RTX A6000 · sm_86 · 驱动 580.159.03
           GPU0 / GPU1   49140 MiB   ECC Disabled
           GPU2 / GPU3   46068 MiB   ECC Enabled      <- 同机混合 ECC,见 §4.4
OS         Ubuntu 24.04.3 LTS · glibc 2.39
CPU/RAM    96 核 / 503 GB
磁盘       容器盘 120 G(用 62 G)· /workspace 网络卷
```

### 2.2 软件栈(从 22 GB 环境包还原到 /opt)

```
torch 2.9.1+cu128 (cuda 12.8) · transformers 4.57.1 · ray 2.47.1 · numpy 1.26.4
sglang 0.5.6 · verl 0.8.0.dev
conda envs: spacetools-rl / -tool-vlm / -tool-roborefer / -tool-bbox / -tool-graspgen
```

环境包 `qzpm55555/spacetools-eval-env` 解到 `/opt/conda-st` + `/opt/spacetools`
(conda 环境里烧死了绝对路径,不能换位置),权重按钉死的 revision 拉。

### 2.3 卡怎么分

`NUM_GPUS=4 EVAL_GPUS=1`——四张卡里**一张整卡给策略,三张给工具**。

工具的逻辑预留(Ray `num_gpus`,**不是显存配额**):

```
roborefer        1 actor × 0.6      RoboRefer-8B
vlm              1 actor × 1.0      Molmo-7B-D        <- 本次从 0.6 改成 1.0,见 §3.3
sam2             2 actor × 0.2
depth_estimator  2 actor × 0.2      DepthPro
bounding_box     2 actor × 0.1
vision_ops       1 actor × 0
grasp_generator  1 actor × 0.1
                        合计 2.7  +  策略整卡 1.0  =  3.7  ≤  4.0
```

实测摆放(Run C 的 60 秒采样轨迹峰值):

```
GPU0  20.5 GB   roborefer + 若干小 actor
GPU1  30.9 GB   Molmo 独占                 <- 改动生效的地方
GPU2   8.6 GB   depth / sam2 / bbox / grasp 余下部分
GPU3  41.7 GB   策略(verl FSDP + sglang)
```

**策略侧要的是一张完整的卡,不能是分数。** 拿不到整卡时 verl 会在
`resource_pool_manager.create_resource_pool()` 抛
`ValueError: Total available GPUs X is less than total desired GPUs 1`。

---

## 3. 改了哪些参数,为什么这么改

相对上游 `run_eval.sh` 一共三处,全部有 as-run 备份(`run_eval.sh.orig*`)。

### 3.1 `BENCHMARKS` 的九行路径

```diff
-    [robospatial]="robospatial_home_multiturn/test.parquet"
-    [reflocation]="refspatial_bench/location.parquet"
-    ... 共九行
+    [robospatial]="data/robospatial.parquet"
+    [reflocation]="data/reflocation.parquet"
+    ... 共九行
```

**为什么:** 上游脚本期望嵌套目录结构,而数据集
`siyich/spacetools-eval-benchmarks` 无论 `main` 还是钉死的 `1d539ac9` **都是扁平的**
`data/<key>.parquet`。上游改过数据集结构、脚本没跟上。不改则每次都
`Missing: .../robospatial_home_multiturn/test.parquet` + `exit 1`,一个样本都跑不了。

**对分数的影响:** 无。改的是文件路径,读到的 parquet 是同一个。

### 3.2 `gpu_memory_utilization` 0.5 → 0.545

**为什么:** 这个旋钮是 **整卡比例,不是绝对值**。SFT 基线(RoboSpatial 61.00 ± 0.77)
是在 48 GB 卡 × 0.5 = **24 GB KV 池** 下测出来的。这台机器最小的卡是 46068 MiB
(ECC 打开的那两张),要让池子仍然是 24 GB 就得反算:`24 / 44 = 0.545`。

**为什么非对齐不可:** 池子大小改变 batch 组成 → 浮点归约顺序 → **接近平局的样本翻转**。
不对齐就不是同一个测量,和 SFT 起点比出来的差值里会混进纯机器项。

**对分数的影响:** 有,而且这正是目的——把它压回与基线同一条件。报结果必须带上这个值。

### 3.3 `vlm` 的 `num_gpus` 0.6 → 1.0

```diff
-    'vlm': {'num_actors': 1, 'resources': {'num_gpus': 0.6}, ...}
+    'vlm': {'num_actors': 1, 'resources': {'num_gpus': 1.0}, ...}
```

**为什么:** 声明值与实际显存严重不符。`run_eval.sh` 给 vlm 传的是
`'dtype': 'float16'`,但上游 `vlm.py:116` 写死 `torch_dtype="auto"`,把这个配置吃掉了,
于是 Molmo-7B-D **以 fp32 加载,实测占 30.2 GiB**——而它只声明 0.6。

Ray 的 `num_gpus` 只保证「落在同一张卡上的逻辑份额加起来不超过 1.0」,**不切分显存**。
于是 Ray 完全合法地把 `vlm(0.6) + depth_estimator(0.2) + sam2(0.2) = 1.0` 摆进同一张卡:

```
30.20 + 9.15 + 7.97 = 47.3 GiB / 整卡 47.4 GiB       只剩 70 MiB
工具再要 576 MiB -> torch.OutOfMemoryError
```

改成 1.0 之后 Molmo 独占一张卡,工具需求 2.3 → 2.7,加策略 1.0 仍然 ≤ 4.0。

**对分数的影响:** 不进入分数。它只改 Ray 的摆放,不改任何模型、判分函数或工具行为;
工具在给定输入下是确定性的(SFT eval 在 350 个样本上验过「同一查询零例返回不同结果」)。
它改的是**能不能跑完而不静默掉分**。

### 3.4 没有改、但必须显式给的两件事

```
NUM_GPUS=4 EVAL_GPUS=1          上游默认 8 / 4,是按 8 卡节点写的。
                                4 卡机器上不覆盖 EVAL_GPUS,工具 2.7 + verl 要的 4 = 6.7 > 4.0,
                                Ray 不报错,actor 无限排队
BASH_ENV=/opt/conda-st/etc/profile.d/conda.sh
                                `conda activate` 是 shell 函数,不跨子进程。非交互 bash 会
                                自动 source BASH_ENV,函数因此在子 shell 里存在。不需要改脚本
```

**特意没有改的:** `vlm.py:116` 的 fp32 问题本身。论文结果是在那个行为下跑出来的,
改了就不可比。我们只改 Ray 的预留让它装得下,不改它加载成什么精度。

---

## 4. 运行健康度

### 4.1 门禁(全零才看分数)

```
                      n    OOM  截断  顶轮数  缺<answer>  工具失败样本
robospatial(A)      350     0    0      0        0          0
robospatial(C)      350     0    0      0        0          0
reflocation         100     0    0      0        0          0
refplacement        100     0    0      0        0          0
refunseen            77     0    0      0        0          0
cvb2drelation       650     0    0      0        0          0
blinkdepth          124     0    0      0        0          1
cvb3ddepth          600     0    0      0        0          0
boppose              60     0    0      0        0          0
bopgrasp             60     0    0      0        0         43   <- 领域结果,见 §4.5
```

OOM = 0 是三个独立口径互相印证的:主日志 `grep -c OutOfMemoryError` = 0、
每个 benchmark 自己的 `eval.log` = 0、`parse_dump.py --strict` 的逐样本
`OOM samples` = 0;连 `CUDA out of memory` 的原文都是 0 次。

### 4.2 被作废的那批结果(只记方法,不记数字)

Run A 里 blinkdepth / cvb3ddepth / boppose / bopgrasp 四个 key 命中了 §3.3 那个显存错配。
**表现不是崩溃,是静默掉分**:工具 OOM 之后 Toolshed 把错误包成正常的 `ToolResult`
返回给模型(调用方拿不到任何异常),模型拿着错误字符串继续推理,eval 跑完、不报错、
给出一个看着合理的数字。

错题归因把这件事量化得很清楚:

```
cvb3ddepth   污染那次 243 错 -> 230 归因 OOM,只有 13 个是真错
             重跑        21 错 ->   0 归因 OOM,21 个全是真错
blinkdepth   污染那次  25 错 ->  19 归因 OOM
             重跑        15 错 ->   0
```

还有一条下游效应值得记:污染那次日志里有 590 条
`TypeError: Expected PIL Image, got <class 'str'>`——那是 **OOM 的二阶后果**,
工具返回错误字符串 → 模型把这个字符串当 `$变量` 喂给下一个工具 → TypeError。
重跑之后降到 4 条(而且那 4 条是模型自己传错了参数,集中在 1 个样本,该样本最后还答对了)。

**这批数字全部作废,本报告不引用。** 保留它们的 dump 是因为它是「静默污染长什么样」
的实物样本,对以后判断同类故障有用。

### 4.3 工具调用直方图

```
robospatial(A)   {roborefer: 576}                                      2.00 轮/样本
robospatial(C)   {roborefer: 575}                                      2.00
reflocation      {roborefer: 102}                                      2.01
refplacement     {roborefer: 101}                                      2.01
refunseen        {roborefer:  77}                                      2.00
cvb2drelation    {roborefer: 1306}                                     2.02
blinkdepth       {vision_ops 253, roborefer 247, depth 125, vlm 33}    3.16
cvb3ddepth       {roborefer 1206, vision_ops 1202, depth 600}          3.01
boppose          {roborefer 61, sam2 60, depth 60, bounding_box 60}    5.02
bopgrasp         {roborefer 60, sam2 60, depth 60, grasp_generator 60} 4.90
```

两件事可以从这张表直接读出来:

1. **RoboSpatial / RefSpatial 这五个 key 全程只调 roborefer**,`depth_estimator`
   一次都没有被调用。SFT eval 当时是 350 个样本里调了 1 次。也就是说
   **C′ 训完之后,「该查深度的时候去查深度」这个行为不但没有被放大,反而归零了。**
2. blinkdepth / cvb3ddepth / boppose / bopgrasp 这四个才真正用全套工具——
   这也解释了为什么 §3.3 的显存错配只在它们身上炸,而 SFT 那四个 key 碰都碰不到。

### 4.4 一个未受控的变量:混合 ECC

这台机器 GPU0/1 是 49140 MiB(ECC 关),GPU2/3 是 46068 MiB(ECC 开),差 3072 MiB。
而 `gpu_memory_utilization` 是整卡比例——**策略落在哪张卡上,KV 池就差约 1.7 GB**。

Run C 的显存轨迹显示策略落在 GPU3(46068 MiB 的那张,池子 ≈ 25.1 GB);Run A 当时没采
轨迹,落哪张卡不知道。这在 §5.2 的结论里是一个真实的混淆项,必须写出来。

**下次的做法:** eval 期间就起 1 Hz 采样,并且给策略显式钉卡。

### 4.5 bopgrasp 的 43/60

`grasp_generator` 在 43 个样本上返回 `No collision-free grasps found`(32)或
`Top-down filtering removed all grasps`(11)。**这是领域结果不是故障**——工具正常跑完,
告诉模型这个场景按它拿到的深度和分割找不到可行抓取。P4 修正后的同口径数字是 **41/60**,
两边一致,所以不是本次的问题。

顺带复现了 P4 记录过的一个上游缺陷:`analysis/analyze_grasp_result.py` 只匹配
`No collision-free grasps`,**漏掉 `Top-down filtering removed all`**,因此它报
32 个 error sample,真实是 43。

---

## 5. 结果

### 5.1 九个 key(全部来自零 OOM 的运行)

```
RoboSpatial   VQA      228   71.93 / 72.81   (两次)
              Vacant   122   41.80 / 44.26
              Overall  350   61.43 / 62.86   均值 62.14
RefSpatial    Location 100   52.00
              Placement100   58.00
              Unseen    77   48.05
              三项      277   52.68 简单平均 / 53.07 加权
BLINK Relative Depth   124   87.90
CV-Bench 2D Relation   650   94.62
CV-Bench 3D Depth      600   96.50
BOP-ASK Pose            60   55.73  平均 IoU
BOP-ASK Grasp           60   MACE 44.79 · SR 55.00%
```

口径:`gpu_memory_utilization=0.545` · `NUM_GPUS=4 EVAL_GPUS=1` · 4× A6000 ·
`vlm num_gpus 1.0` · greedy 解码。

### 5.2 三方对照

| Table 2 行 | n | SFT 起点 | 官方 ckpt(P4 复现) | **C′ step85** | 论文 |
|---|--:|--:|--:|--:|--:|
| RoboSpatial VQA | 228 | 71.27 | 73.25–73.68 | **72.37**(两次均值) | 79.38 |
| RoboSpatial Vacant | 122 | 41.80 | 50.82–51.64 | **43.03**(两次均值) | 52.46 |
| **RoboSpatial Overall**‡ | 350 | **61.00 ± 0.77** | **65.43–66.00** | **62.14**(两次均值) | 70.00 |
| RefSpatial Location | 100 | 54.00 | — | 52.00 | — |
| RefSpatial Placement | 100 | 57.00 | — | 58.00 | — |
| RefSpatial Unseen | 77 | 46.75 | — | 48.05 | — |
| **RefSpatial 三项** | 277 | 52.58 / 53.07 | 53.35 / 53.79 | **52.68 / 53.07** | 53.07 |
| BLINK Relative Depth | 124 | 未评 | 86.29–87.90(上限 90.32) | **87.90** | 90.32 |
| CV-Bench 2D Relation | 650 | 未评 | 94.62 | **94.62** | 94.92 |
| CV-Bench 3D Depth | 600 | 未评 | 96.50 | **96.50** | 96.00 |
| BOP-ASK Pose(IoU) | 60 | 未评 | 53.36 | **55.73** | 34.37※ |
| BOP-ASK Grasp MACE | 60 | 未评 | 43.07–46.17 | **44.79** | 43.06 |
| BOP-ASK Grasp SR | 60 | 未评 | 55.00–56.67 | **55.00** | 50.00 |

‡ Overall 是 VQA/Vacant 的样本加权,不是独立测量。
※ 论文的 34.37 在 n=60 下不是比例(60 × 0.3437 非整数),P4 已判定为**指标映射未解决**,
不要当差距读。
RefSpatial 两个数是「简单平均 / 样本加权」。SFT 那一轮只评了四个 key,所以后五行没有起点。

**cvb2drelation 94.62 与 cvb3ddepth 96.50 和 P4 官方 ckpt 逐位相同**——这是这套环境
复现正确最强的一条证据(同一套工具、同一个判分函数、同一批数据)。

### 5.3 RoboSpatial 两次运行

```
              n     Run A           Run C
  VQA       228   164  71.93%    166  72.81%
  Vacant    122    51  41.80%     54  44.26%
  Overall   350   215  61.43%    220  62.86%     均值 217.5 = 62.14%

  恒对 205 · 恒错 120 · 翻转 25       单次可能取值区间 205 ~ 230
  两次生成逐字不同的样本 199/350
```

对照 SFT 基线:`216 / 211`,恒对 199,上界 228,期望 213.5(61.00%),翻转 29 个。

**差 4 题 ≈ 1.5 个标准误,不构成有统计意义的提升:**

```
C′  翻转 25 -> 单次 sd ≈ 2.5 题,两次均值 sd ≈ 1.8 题
SFT 翻转 29 -> 两次均值 sd ≈ 1.9 题
合并 sd ≈ 2.6 题,差 4.0 题 -> 约 1.5 σ,双侧 p ≈ 0.12
两边区间 205–230 与 199–228 几乎完全重叠
```

而且 **SFT 的 61.00 是在另一台机器上测的**,本次没有在这台机器上重测 SFT,叠加 §4.4 的
混合 ECC,这 4 题的差**机器层面的解释和模型层面的解释目前分不开**。要钉死只有一个干净
做法:在同一台机器、同一 session 里把 SFT ckpt 也跑一遍 robospatial。

### 5.4 VQA 混淆矩阵:三个模型在同一个地方失效

```
                        gt_no 准确率     gt_yes 准确率
SFT 起点                  49.21%           80.61%
官方 ckpt(RL 后)         47.62%           83.03%
C′ step85  Run A          50.79%           80.00%
C′ step85  Run C          50.79%           81.21%      <- gt_no 两次逐位相同 32/63
```

**该答 "no" 的题,三个模型都是掷硬币。** SFT 报告当时写下的那条负面预期
(「`gt_no` 的掷硬币水平在论文的 RL 之后依然存在,不要把它列进 RL 的预期收益」)
到这里第三次被确认——**C′ 也没有修掉它**。两次运行之间涨的那点分全部来自 `gt_yes`
和 Vacant 的贴边样本。

---

## 6. 结论

**1. C′ 在能做前后对比的地方没有产生可测的提升。** RoboSpatial 差 4 题(1.5 σ)、
RefSpatial 加权逐位相同、Vacant 与 SFT 起点完全一致。官方 ckpt 领先的那约 11 题没有
被追上任何一题。

**2. 这和训练侧的指标是自洽的,不是意外。** 训练报告里的三条:

- 奖励退化组占 **63–80%**(`reward/degenerate_group_frac` 中位数 0.703)。整组同分时
  奖励项恒为 0,梯度全部来自漂移项——那时候 C′ 做的是向 `π_ref·exp(βr)/Z` 的
  flow matching,**不是在学奖励**。
- `actor/grad_norm` 全程 **103–5229**,而 `grad_clip=1.0`:每次更新都被裁到只剩方向、
  没有幅度。
- 1 epoch / 85 步。

**一个策略在约 70% 的样本上收不到奖励信号、每次更新又只剩方向,跑 85 步之后与起点分不开,
是预期内的结果。**

**3. 工具使用行为不但没被放大,反而收窄了。** SFT 起点在 robospatial 350 个样本里调过
1 次 `depth_estimator`(而且那一次算对了),本次是 **0 次**。SFT 报告把「该调工具时不调」
认定为 RL 应该改、SFT 改不动的那一类问题;C′ 在这一项上给出的是反方向的读数。

**4. 另外五个 benchmark 不能说明 C′ 的好坏,因为没有起点。** 它们与官方 ckpt 持平或略好,
其价值在于证明这套 eval 环境复现正确(两个 CV-Bench 的数与 P4 逐位相同)。

---

## 7. 有意义的问题(与本次结果相关、换机器还会遇到的)

### 7.1 `num_gpus` 是逻辑预留,不是显存配额 —— 而 vlm 的声明差了 5 倍

§3.3 已详述。要点重复一遍,因为它是本次唯一一个真正毁掉过结果的问题:

- Ray 只保证同卡逻辑份额 ≤ 1.0,**不切分显存**;
- `vlm.py:116` 的 `torch_dtype="auto"` 吃掉 `dtype: float16`,Molmo 以 fp32 加载占 30.2 GiB,
  却只声明 0.6;
- 于是 Ray 合法地把 30.2 + 9.15 + 7.97 塞进 47.4 GiB 的一张卡;
- **OOM 不崩溃,只静默掉分。**

它只在真正用全套工具的 benchmark 上触发,所以 SFT eval 那四个 key 从来碰不到——
SFT 报告 §6.2 其实已经记下了这个危险摆放(`GPU1 46.3 GiB / 48 GiB,util 0%`),
只是那张卡上没有人再要显存,它就静静地满着,什么都没发生。

### 7.2 跑完了,退出码却是 15

`run_eval.sh` 打完 `=== EVALUATION COMPLETE ===` 之后,它自己的 cleanup 陷阱
`kill` 后台的 toolshed 进程,`wait` 把 SIGTERM 带出来,脚本退出码 15。

**按退出码判死会把一批完全有效的结果判废。** 三次运行的退出码都是 15,三次都跑完了。
正确的判据是**成功标记 + 样本数 + OOM 数 + router 失联数**,不是 `$?`。
(这一条和 SFT 报告 §6.6「验收脚本本身也会给假绿灯」是同一类问题的两面:
一个是失败却 exit 0,一个是成功却 exit 15。)

### 7.3 `run_eval.sh` 没有并发保护,两个实例会互相打死对方

启动命令被上层重试了一次,结果同时起了两个 `run_eval.sh`。两者在开头都执行
`ray stop --force`,于是后起的把先起的 toolshed 连同全部工具 actor 一起杀掉
(`start_toolkit(detached=False)`,父进程一死 actor 全死)。

**表现极其隐蔽**:先起的那个 eval **继续跑**,只是从此每次调工具都拿到
`Could not find ToolRouterActor 'toolshed_router'`,照样产出分数。

**对策:** 启动脚本开头加一个原子锁(`mkdir /root/.lock || exit 0`),并把
`Could not find ToolRouterActor` 的计数加进门禁。两者本次都已加上。

### 7.4 同一台机器上的混合 ECC 让 KV 池不受控

§4.4。`gpu_memory_utilization` 是整卡比例,而这台机器两两卡的可用显存差 3072 MiB,
策略落哪张卡池子就差约 1.7 GB——而池子大小正是会翻转接近平局样本的那个旋钮。
**要和别的机器上的基线比,就得给策略钉卡,或者至少记录它落在哪张卡上。**

### 7.5 上游 `run_eval.sh` 的 BENCHMARKS 映射与数据集实际布局不符

§3.1。这是「拿到机器第一次跑必然 exit 1」的问题,与 checkpoint 无关。

### 7.6 `conda activate` 在非交互子进程里失效

`run_eval.sh:84` 调 `conda activate spacetools-rl`,而 `conda` 是 shell 函数、不跨子进程,
报 `CondaError: Run 'conda init' before 'conda activate'`。
解法是 `BASH_ENV=/opt/conda-st/etc/profile.d/conda.sh`,不需要改脚本。

### 7.7 我们自己的 `parse_dump.py` 吃文件不吃目录

`EVAL_FROM_SCRATCH.sh` 的门禁那一步传的是目录,而 `parse_dump.py` 要的是
`0.jsonl` 文件路径,于是它打印 `not a file` 并报 `CONTAMINATED OR UNREADABLE`。
本次是手工按文件路径重跑的门禁。**这是个会给出「门禁跑过了」错觉的缺陷,要修。**

### 7.8 `analyze_grasp_result.py` 少数一类抓取失败

§4.5。上游只匹配 `No collision-free grasps`,漏掉 `Top-down filtering removed all`,
把 43 报成 32。P4 已记录,本次复现。

---

## 8. 附:产物清单与它们能回答什么

```
dumps/<run>/<benchmark>/0.jsonl    一行一个样本,对错都在里面。字段与 P4 那批逐个相同:
                                   input(~10 KB 完整 prompt)、output(**完整多轮轨迹,
                                   含工具真实返回**)、gts、answer、acc/score/reward/fmt、index
logs/<run>/<benchmark>/eval.log    工具侧视角:actor pid、工具内部重试、OOM 原文
logs/*.log                         三次运行的主日志(含 set -x 全程)
gates/                             parse_dump.py --strict 的逐 benchmark 输出
scripts/                           as-run 的启动脚本 + 上游 run_eval.sh 的 as-run 版与原版
gpu/                               Run C 的 60 秒显存轨迹
```

**dump 和 eval.log 回答的是不同问题**(P4 那个「96 vs 16」的教训):日志记录工具内部
的多次尝试,dump 记录样本级发生了什么。两个都要留。

`image` 字段多数为 null,取图要走 `sample_id → parquet`;benchmark parquet 没有随包
(292 MB),在 HF `siyich/spacetools-eval-benchmarks @ 1d539ac9` 上有钉死版本。

**要生成 P4 那套 enriched `parsed/` 记录**(`vars_exposed` / `vars_used` /
`vars_unused` —— 直接对应 P6 taxonomy 2d「不复用变量」、抽出来的 `question`、
`num_turns_verl_convention`),用 `spacetools-repro` 的 parser 吃这些 dump 即可,
**不需要 GPU**。
