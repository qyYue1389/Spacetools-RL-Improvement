# SFT checkpoint eval 报告

对象:`qzpm55555/spacetools-sft-v1-4xa6000` @ `91fd4bdf6d2ffe3b355cc650e71925c6e91a15c5`
执行:2026-09-11/12 · 4× RTX A6000 · 依据 `SFT eval 交接文档(未收录)`
配套:`03_sft_eval/sft_eval_results.md`(原始数字与出处)、`00_environment/eval_rl_env/env_package_revision_20260912.md`(环境侧改动)

---

## 摘要

自训 SFT checkpoint 在四个空间推理 benchmark 上跑完 627 个样本,**判据通过,可以进 RL**:

```
RoboSpatial   61.0%  [56.9 下界 / 65.1 上界]   两次观测 60.29 / 61.71    判据 ≥60  ✅
RefSpatial    52.58% 简单平均 / 53.07 加权                              判据 ≥48  ✅
```

比这两个数字更有价值的是**失败模式**:RefSpatial 上 SFT 已经与论文里 RL 之后的官方
checkpoint 平手(277 题差 2 题),而 RoboSpatial 落后 15–17 题,落后的原因不是推理弱,
而是**该调工具的时候不调**——350 个样本里 `depth_estimator` 只被调用过 1 次,
而那唯一一次算对了。这正好是 RL 能改、SFT 改不动的那一类问题。

---

## 1. 判据是怎么定的

上一个 session 定的 go/no-go:

| 结果 | 结论 |
|---|---|
| RoboSpatial ≥ 60 **且** RefSpatial ≥ 48 | checkpoint 可用,进 RL |
| RoboSpatial < 55 **或** RefSpatial < 40 | 停下来查,不进 RL |
| 中间带 | 先确认运行健康,再和论文表逐项对齐 |

判读顺序被明确规定**不能颠倒**:先数 OOM → 再查工具静默错误 → 最后才看分数。
理由见 §3.4。

选这四个 key 的原因:它们覆盖论文主表的空间推理能力,且都用不到
`grasp_generator`(七个工具里唯一在这四个 key 上完全用不到、却必须注册进 schema 的,
因为模型见过的接口不能少)。明确排除 `boppose` / `bopgrasp`——前者 metric 映射
复杂、后者会把 `grasp_generator` 拉进来增加变量。

---

## 2. 怎么 eval 的

### 2.1 系统结构

被评的不是一个模型,是一个**模型 + 七个工具**的系统。策略模型
(Qwen2.5-VL-3B,本次的 SFT ckpt)由 sglang 起,工具是七个 Ray actor,
**各自跑在自己的 conda 环境里**:

```
策略模型  Qwen2.5-VL-3B (sglang)            ← 被评的对象
工具                                        conda 环境        num_actors × num_gpus
  roborefer      空间指代定位(RoboRefer-8B)  tool-roborefer    1 × 0.6
  vlm            Molmo-7B-D                   tool-vlm          1 × 0.6
  sam2           分割                         tool-vlm          2 × 0.2
  depth_estimator DepthPro 深度+点云          tool-vlm          2 × 0.2
  bounding_box   3D 有向包围盒                tool-bbox         2 × 0.1
  vision_ops     数组取值等                   tool-bbox         1 × 0
  grasp_generator GraspGen                     tool-graspgen     1 × 0.1
                                                        逻辑预留合计  2.3 张卡
```

五个环境的依赖互相冲突(torch 2.3.1 / 2.5.1 / 2.9.1,numpy 1.26.4 与 2.4.6 并存),
所以必须分环境。Toolshed 通过 Ray 的 `runtime_env={"conda": <env>}` 把每个工具
起在对应环境里(`router.py:158-164`),变量(ndarray)跨环境传递由 Ray 序列化。

**上表这三列不是我们选的,是 `run_eval.sh` 里写死的** —— `VERSION=v1` 分支下那段内联
python 的 `TOOL_CONFIGS` 字典(`run_eval.sh:132-142`),`conda_env` 的归属也在里面:

```python
'sam2': {'num_actors': 2, 'resources': {'num_gpus': 0.2},
         'conda_env': 'spacetools-tool-vlm', 'timeout': 600, 'args': {...}}
```

本次**一个字没改**(改了就没法和论文对口径)。as-run 的那一版存在
`sft-eval-artifacts/config/run_eval.sh.asrun`。

**`num_gpus` 是 Ray 的逻辑预留,不是显存配额 —— 分数不切分显存。** 同一张物理卡上的
actor 共享这张卡的全部显存,Ray 只保证「落在同一张卡上的逻辑份额加起来不超过 1.0」。
所以「合计 2.3 张」只决定**排得下排不下**,不保证任何一张卡不会 OOM。
显存要单独算,见 §6.1(Molmo fp32 实测 30.2 GiB)和 §6.2(实测打包布局)。

### 2.2 交互形态

策略模型看到一张图和一个空间推理问题,按系统提示的四步走:先 `<think>` 内心独白,
需要信息时调工具,拿到工具返回后继续推理,最后 `<answer>` 给答案。最多 8 轮。

一条真实的完整轨迹(RefSpatial-Location,index=0,答对):

```
<think> I need to find the orange box in the image. Looking at the image, I can see
a box with "Nature Valley" and "GRANOOLA" text on it, which appears to be a Nature
Valley granola bar box. This box is in the upper right area of the image. I will use
the roborefer.detect_one tool to locate this box. </think>
<tool_call>
{"name": "roborefer.detect_one", "arguments": {"obj_name": "orange box", "image_index": 0}}
</tool_call>
          ↓ 工具返回
<tool_response>
Detected 1 instance(s) of 'orange box': [(0.95, 0.247)]
</tool_response>
          ↓
<think> I have successfully located the orange box at the coordinates (0.95, 0.247).
Now, I can format this as a list of tuples as requested. </think>
<answer> [(0.95, 0.247)] </answer>
```

### 2.3 口径

- **解码是 greedy**。verl 的 `val_kwargs` 默认 `temperature 0 / top_p 1.0 / n 1 /
  do_sample False`,发布仓库的 `run_eval.sh` 没有覆盖它。脚本里那个 `rollout.n=5`
  是训练用的 group size,不作用于评测。
- **权重全部按钉死的 revision 拉取**(`WEIGHTS_PINS.txt`),包括 benchmark 数据集
  `siyich/spacetools-eval-benchmarks @ 1d539ac9...`。
- **不修 `vlm.py:116` 的 fp32 问题**(见 §6.1)——论文结果是在那个行为下跑出来的,
  口径必须一致。

### 2.4 判分函数(读源码确认,不靠推测)

三个 RefSpatial key 和 RoboSpatial 的点类问题,判据都在
`verl/utils/reward_score/robos_all.py`:

```
compute_points_score(..., point_evaluation_method="convex_hull")
    1.0 if point is within (or on the boundary of) the convex hull of the GT points
    0.0 otherwise
```

**GT**(ground truth,真值)给的是 **10 个点**,因为「空位」是一片区域而不是一个点,
数据集用 10 个采样点来描述它。判分取这 10 个点的**凸包**(能把它们全包住的最小凸多边形),
只问一件事:**预测点在不在这个多边形里**。在就 1 分,不在就 0 分,没有中间分。

以 index=1 为例,GT 的 10 个点聚在 `x ∈ [0.701, 0.757]`、`y ∈ [0.615, 0.649]`:

```
  y=0.60     ·  ← 模型答的 (0.715, 0.6)
  y=0.615  ┌──●──────────●──┐   ← 凸包
           │ ●  ●   ●  ●    │      10 个 GT 点在里面
  y=0.649  └──────●─────────┘
```

x 在范围内,但 y=0.6 比最上面那个 GT 点还高 0.015,**落在凸包外 → 0 分**。

**这不是「到最近 GT 点的距离」,这两个判据不等价。** 数据里有直接反例:idx 87 两次运行,
到最近 GT 点 0.0605 判**对**、0.0191 判**错** —— 更近反而错了。一个点可以刚好在凸包外
却擦着某个顶点很近,也可以在凸包正中间却离那 10 个采样点都不近。
**所以分析错题「差多少」必须算到凸包边界,算到最近点是错的度量。**

「用仓库自己的 `_convex_hull` / `_is_point_in_convex_polygon` 重算,
**122/122 完全一致、零分歧**」的意思是:没有凭读代码就相信自己理解对了,而是从 rollouts
取出预测点与 GT 点、import 仓库那两个函数重算一遍「在不在里面」,再和 verl 落盘的 `acc`
逐个比对。这同时确认两件事 —— 判据理解正确,且 `acc` 真的就是凸包归属而非别的东西。

RoboSpatial 的 Yes/No 类问题走 `compute_yesno_score`,按 GT 形态天然分成
**VQA 228 题**(`Yes`/`No`)和 **Vacant 122 题**(点列表),与 P5 的切分逐个吻合。

---

## 3. 流程:从裸机到分数

### 3.1 机器验收

```
GPU 架构   A100(sm_80) / A6000(sm_86) / L40S(sm_89) 可用
           H100(sm_90) 不可用 —— 编译时没带 +PTX,没有能 JIT 到 Hopper 的中间码
glibc      不能老于 Ubuntu 24.04 的 2.39
驱动       ≥ 550
容器盘     ≥ 80 GB;网络卷 ≥ 100 GB
还原路径   必须 /opt/conda-st + /opt/spacetools,conda 环境里烧死了绝对路径
```

**实测要加一条:交付的机器可能根本没装驱动。** 本次拿到的机器 `lspci` 能看到
4 张 GA102GL,但 `/dev/nvidia*` 不存在、内核模块没加载、`dpkg -l | grep nvidia`
是 0 个包。装 `nvidia-driver-570-server`(实得 580.173.02)、DKMS 编译、
`modprobe` 之后四卡正常。**所以第一条检查应该是 `nvidia-smi -L` 而不是 `nvidia-smi`**,
后者不存在时的报错(`command not found`)容易被当成 PATH 问题。

最终环境与打包机的对照:

| | 打包机 | 本次 |
|---|---|---|
| GPU | RTX A6000 | RTX A6000 ✓ |
| 驱动 | 580.159.04 | 580.173.02 ✓ 同分支 |
| OS | Ubuntu 24.04.3 | 24.04.2 ✓ |
| glibc | — | 2.39 ✓ |

### 3.2 环境还原

```
hf download qzpm55555/spacetools-eval-env     21 GB,6 个切片
sha256sum -c                                   12/12 OK
FETCH_WEIGHTS.sh                               79 GB,按钉死 revision
RESTORE.sh                                     拼接 22102227452 字节(与 README 记录逐字节一致)
                                               解到 /opt,五环境就位
POSTRESTORE.sh                                 ← 本次之后新增,见 00_environment/eval_rl_env/env_package_revision_20260912.md
VERIFY.sh                                      四项验收
```

权重落地实测:

```
/workspace/hf           54G   ← MANIFEST 记 54G
/workspace/checkpoints  34G   ← 25.7G 基础权重 + 7.6G SFT ckpt
depth_pro.pt sha256     3eb35ca6...85c0ce   实际 == 期望
```

SFT checkpoint 确认:

```
architectures   Qwen2_5_VLForConditionalGeneration
model_type      qwen2_5_vl        dtype  bfloat16        7.6 GB
num_attention_heads 16   num_hidden_layers 36
revision        91fd4bdf...a15c5  ← 用 SHA 而非 main,否则结果不可复现
```

### 3.3 验收(四项)

```
五环境 import 闸门     全通
架构实扫               4619 个 .so · 108 个含 cubin · 自编扩展缺 sm_8x 共 0 个
                       (1439+591+886+519+1184 / 37+14+21+1+35)
七工具冒烟             7/7,真加载权重真出结果
Ray 跨环境链           5 工具 · 失败环数 0 · ndarray 双向传递正确
```

三个数字(4619 / 108 / 0)与打包时的基线**逐字对上**,说明这套环境在新机器上可复现。

跨环境链这一项测的是 eval 的真实形态——A 工具产出的变量喂给 B 工具,而 A、B 在
不同 conda 环境里:

```
sam2 (tool-vlm, numpy 1.26.4)  产出 $segmentation_mask (bool ndarray)
     → bounding_box (tool-bbox, numpy 2.4.6)  吃它
depth_estimator (tool-vlm, np 1.26.4)  产出 $depth_map (float ndarray)
     → vision_ops (tool-bbox, np 2.4.6)  吃它
```

### 3.4 跑 eval,以及为什么判读顺序不能颠倒

```bash
NUM_GPUS=4 EVAL_GPUS=1 \
ROBOREFER_MODEL=/workspace/checkpoints/RoboRefer-8B-SFT \
DEPTH_CHECKPOINT=/workspace/checkpoints/depth_pro.pt \
bash run_eval.sh /workspace/checkpoints/sft-ckpt \
     robospatial reflocation refplacement refunseen
```

627 个样本,32 分 41 秒。

**判读第一步是数 OOM,不是看分数**:OOM 的样本会算进分母,分数被稀释成一个
「偏低但看着合理」的数字。

**第二步查工具有没有静默返回错误,这一步最容易被跳过、后果最贵。** toolshed 的工具
出错时**不抛异常**,而是把错误信息包成正常的 `ToolResult` 返回;router 也会把 actor
里的异常吞掉包成正常返回。表现是这样的:

```
CALL_OK   depth_estimator -> ModuleNotFoundError: No module named 'numpy._core.numeric'
                             返回类型: ToolResult
```

注意那是 `CALL_OK`。**调用方拿不到任何异常。** 如果某个工具在整个 eval 里一直返回
报错文本,模型就一直拿着垃圾输入推理,eval 会跑完、不报错、给出一个看着正常的
分数——然后你会去怀疑 checkpoint。

所以除了日志层面的 `grep -cE "Error:|ERROR:toolshed"`,本次还在**落盘的 rollouts 上**
逐样本核了工具响应(见 §4.3)。注意那个 grep 用的是**子串**匹配:`TypeError:`
里含 `Error:`,所以它抓得到;换成更「严谨」的前缀匹配反而会漏。

---

## 4. 结果

### 4.1 分数

| | n | run 1 | run 2 | 恒对 | 恒错 | 翻转 | 期望 ± sd |
|---|--:|--:|--:|--:|--:|--:|--:|
| RoboSpatial · Overall | 350 | 61.71% (216) | 60.29% (211) | 199 | 122 | 29 | **61.00% ± 0.77 pp** |
| RoboSpatial · VQA | 228 | 70.61% (161) | 71.93% (164) | 152 | 55 | 21 | 71.27% |
| RoboSpatial · Vacant | 122 | 45.08% (55) | 38.52% (47) | 47 | 67 | 8 | 41.80% |
| RefSpatial · Location | 100 | 54.00% (54) | — | | | | |
| RefSpatial · Placement | 100 | 57.00% (57) | — | | | | |
| RefSpatial · Unseen | 77 | 46.75% (36) | — | | | | |
| RefSpatial · 三项 | 277 | 52.58 简单 / 53.07 加权 | — | | | | |

**RoboSpatial 必须按区间报。** 判据 60% = 210/350 落在**翻转带内**(199 恒对 ~ 228 上界),
单次运行读到线上还是线下取决于那 29 个翻转样本怎么落:观测翻转成功率 ≈ 0.50,
清关需要中 11 个,**单次读到线下的概率约 6.8%**。
5 次取中位数读到线下的概率只有 0.28%,所以补跑到 5 次不会改变结论。

### 4.2 与论文 / 官方 ckpt 的逐项对照

| Table 2 行 | n | 论文 | P4/P5 复现<br>(官方 ckpt,RL 之后) | **本 SFT ckpt** | 少答对几题 |
|---|--:|--:|--:|--:|--:|
| RoboSpatial · VQA | 228 | 79.38 | 73.25–73.68 | 71.27 | 4–6 题 |
| RoboSpatial · Vacant | 122 | 52.46 | 50.82–51.64 | 41.80 | 11–12 题 |
| RoboSpatial · Overall‡ | 350 | 70.00 | 65.43–66.00 | 61.00 | 15–17 题 |
| RefSpatial · 三项 | 277 | 53.07 | 53.35 简单 / 53.79 加权 | 52.58 / 53.07 | **2 题** |

‡ Overall 是 VQA/Vacant 的样本加权,不是独立测量。

**换算成题数是必要的**:不同 benchmark 的 n 差一个量级,pp 不可直接比较。
RefSpatial 的 −0.77 pp 换算过来是 **277 题里差 2 题**,统计上不可分辨
(n=277、p≈0.53 时标准误约 3.0 pp,差距是 0.26 个标准误);
RoboSpatial 的 −4 pp 是 **350 题里差 14 题**,那才是真差距。

### 4.3 运行健康:为什么可以相信上面的数字

```
OOM                          0
样本级工具错误响应            0 / 627
每个样本都有工具调用          627 / 627
缺 <answer>                   0
轮数耗尽(8 轮上限)           0
截断的工具响应                0
畸形 tool call                0
行数对账      350 + 100 + 100 + 77 = 627 ✓
```

工具调用分布与 P5 记录的官方 ckpt 行为几乎重合:

```
robospatial   {1 次: 124, 2 次: 225, 3 次: 1}   平均 1.649 次/样本
                                                  P5(官方 ckpt): 1.63 次/样本
reflocation   {1 次: 98, 2 次: 2}                平均 1.020
refplacement  {1 次: 100}                        平均 1.000     P5: 1.00
refunseen     {1 次: 77}                         平均 1.000     P5: 1.00
```

实际被调用的工具只有三个:

```
roborefer          627 / 627 样本
depth_estimator      1 次
vision_ops           1 次
sam2 / vlm / bounding_box / grasp_generator    0 次
```

---

## 5. 轨迹分析:分数背后发生了什么

### 5.1 三个 RefSpatial key:策略是 RoboRefer 的一层薄包装

277/277 样本只有一种链路:`roborefer×1@2t`。§2.2 那条轨迹就是全部形态——
调一次 `detect_one`,把返回的坐标抄进 `<answer>`。

> **链路签名记法**(沿用 P5 §5.3):`工具×调用次数@N t`,N 是 assistant 轮数。
> 1 次调用必然占 2 轮 —— 一轮发起调用,工具结果以 **user 轮**回注,再一轮作答。
> 调用数与轮数的比值能看出模型是否并行:`roborefer×2@2t` 是两次调用塞在同一轮
> (§5.2),`depth×1+roborefer×2+vision_ops×2@3t` 是 5 次调用只用 3 轮(§5.4)。
> 日志里 verl 打的 `num_turns=4` 数的是全部轮次(含 user 轮),描述同一条轨迹。

**这解释了为什么 SFT 在这一档能和 RL 后的官方 ckpt 平手(差 2 题)**:
任务退化成「调用一次工具并原样转述」,SFT 完全学得会,RL 没有可加力的地方。
论文 Table 2 里 RoboRefer-8B-SFT 单跑 RefSpatial 是 48.37,SpaceTools-3B(RL 之后)
是 53.07。**48.37 → 52.58 这一段不需要 RL 就能拿到**,由本次 eval 证实:我们的纯 SFT
ckpt 套上同一个 roborefer 就到 52.58,剩下到 53.07 只差 0.49 pp(约 2 题,在噪声里)。

「把工具接进来」指的是这两个系统在**结构**上差什么:

```
RoboRefer-8B-SFT 单跑   图 + 问题原文 → RoboRefer-8B → 点                    48.37
SpaceTools-3B           图 + 问题 → 3B 策略自己写 obj_name
                                  → 同一个 RoboRefer-8B → 点
                                  → 策略抄进 <answer>                        53.07
```

同一个感知模型,前面多了一层 3B 的策略 —— 策略比工具还小,却涨了 4.7 pp。
**涨的不是感知能力,是那层包装做的事。**

那 4 pp 具体由什么贡献,**尚未拆解**。候选是「策略自己改写 `obj_name`」
(§5.5 量到:答对的样本 80% 用了 >30 字符的查询,答错的只有 51%)和「输出格式归一化」;
校验/重试这一条实测不成立 —— §5.5 显示策略对工具返回不做任何校验。

> 注:48.37 是从 P5 报告转引论文 Table 2 的基线行,本次未自行复现,也未核对它是否
> 用了相同的 parquet 与判分函数。

### 5.2 RoboSpatial VQA:两次定位 + 定性推理

答对的样本(index=122,GT `Yes`,问「纸巾盒能放到椅子后面吗」):

```
<think> ... 1. Locate the tissue box  2. Locate the chair
             3. Assess the available space behind the chair ... </think>
<tool_call> roborefer.detect_one  obj_name="tissue box" </tool_call>
<tool_call> roborefer.detect_one  obj_name="chair" </tool_call>
   → Detected 1 instance(s) of 'tissue box': [(0.306, 0.531)]
   → Detected 1 instance(s) of 'chair':      [(0.477, 0.517)]
<think> Tissue box at (0.306, 0.531) - left-center, on what appears to be a desk
        Chair at (0.477, 0.517) - center of the image
        "Behind the chair" would mean the area further away from the camera ...
        Looking at the image, behind the chair there appears to be wall space and
        potentially some floor space ... </think>
```

答对了,但注意推理的性质:**它拿到的是两个 2D 质心,然后靠「看图觉得」补完了
深度和尺寸判断。** 两个坐标对「能不能放进去」这个问题几乎没有信息量。

### 5.3 关键反例:该调深度工具而不调(index=128,GT `No`,答错)

问「杯子能放到音箱前面吗」:

```
<tool_call> roborefer.detect_one  obj_name="cup" </tool_call>
<tool_call> roborefer.detect_one  obj_name="speaker" </tool_call>
   → Detected 1 instance(s) of 'cup':     [(0.78, 0.494)]
   → Detected 1 instance(s) of 'speaker': [(0.154, 0.491)]
<think> The question asks if the cup can fit "in front of" the speaker.
        This means: is there space between the speaker and the camera/viewer?
        ... There is desk space in front of the speaker
        ... There appears to be adequate desk space in front of the speaker ... </think>
答:Yes        真值:No
```

**它自己把问题正确地翻译成了深度问题**(「音箱和相机之间有没有空间」),
**然后用两个 y 值几乎相同的 2D 坐标去回答它**(0.494 vs 0.491),
最后用「看起来桌面有空间」收尾。

`depth_estimator` 和 `vision_ops` 就在它的 schema 里,而且组合起来正好能回答这个问题。
它没有调。

### 5.4 唯一一次调了完整链路的样本,算对了(index=176,acc=1.0)

350 个样本里只有这一个走了跨三个 conda 环境的完整链路:

```
depth_estimator.estimate_depth_with_pointcloud        (tool-vlm, numpy 1.26.4)
   → Generated 1428x1932 depth map and 2,758,896 3D points.
     Depth range in point cloud: 1.93m to 5.44m (mean: 3.39m).
     Use $depth_map, $point_cloud and $focal_length_px to reference the results.
roborefer.detect_one  "bed"     → [(0.336, 0.551)]
roborefer.detect_one  "table"   → [(0.844, 0.592)]
vision_ops.index_at   $depth_map @ (0.336, 0.551)   (tool-bbox, numpy 2.4.6)
   → Pixel value at (0.336, 0.551) is 2.7431180477142334
vision_ops.index_at   $depth_map @ (0.844, 0.592)
   → Pixel value at (0.844, 0.592) is 2.944430351257324
```

拿到两个物体的**真实米数**(2.74 m vs 2.94 m)再比较,答对。

**§5.3 和 §5.4 放在一起就是本次 eval 最重要的一条发现:能力是有的,触发率是 1/350。**

### 5.5 Vacant:把问题里的空间条件整句塞进「物体名」参数,然后不做校验

先把两个词在这个样本里分别指什么说清:

**空间谓词** = 问题里描述**空间关系**的那部分文字。index=1 这道题要找的是「一个能放
东西的空位,要靠近吸尘器、并且在它左边」,其中

```
close to and to the left of the vacuum
```

就是空间谓词。它描述的是物体之间的**关系**(靠近、在左边),不是某一个物体。

**指代检测器** = `roborefer` 这个工具(底层是 RoboRefer-8B)。它的职责是「给一个
描述物体的名词短语,返回那个物体在图上的坐标」。它的接口签名是

```
roborefer.detect_one(obj_name=..., image_index=...)
                     ^^^^^^^^ 参数名就叫 obj_name —— 物体名
```

模型做的事,是把整句空间谓词连同前缀一起填进了 `obj_name` 这个本该放物体名的槽位:

```
<tool_call>
{"name": "roborefer.detect_one", "arguments": {"obj_name":
  "point within the vacant space suitable for placing an object close to and to
   the left of the vacuum", "image_index": 0}}
</tool_call>
   → Detected 1 instance(s) of '...': [(0.715, 0.6)]
<think> The tool detected a point at (0.715, 0.6) which is suitable for placing an
        object close to and to the left of the vacuum. This point is within the
        vacant space and is left of the vacuum. </think>
<answer>[(0.715, 0.6)]</answer>

GT 凸包(10 点): x ∈ [0.701, 0.757],  y ∈ [0.615, 0.649]
预测 (0.715, 0.6):x 在范围内,y 差 0.015 落在凸包外  → 0 分
```

两处值得注意:

1. **它把整个空间谓词当成物体名交给了 `detect_one`** —— 等于把「哪里算空位、哪边算
   左边」这部分空间推理整体外包给了 roborefer,自己只做转述。有意思的是这个做法
   **统计上是有效的**:答对的 Vacant 样本里
   80% 用了 >30 字符的 `obj_name`(平均 54 字符),答错的只有 51%(平均 38 字符)——
   查询写得越完整越容易对。roborefer 本来就是指代表达模型,吃得下长谓词。
2. **它对工具返回不做任何校验**,直接断言「This point is within the vacant space」——
   它没有任何依据这么说。而 `vision_ops.index_at` + `$depth_map` 正好可以验证那个点
   落在哪个平面上。

Vacant 的错题整体上是「差一点」:67 个错题到凸包**边界**的距离
中位数 0.0761,16% 在 0.02 以内、36% 在 0.05 以内、61% 在 0.10 以内。

### 5.6 VQA 混淆矩阵:学会了先验,没学会判别

**混淆矩阵**是把「真值」和「模型答了什么」交叉成一张表:**行是真值,列是模型的答案**,
对角线上的格子是答对的,非对角线是答错的。VQA 的答案只有 Yes / No 两种,所以是 2×2。

它比单个准确率有用,因为「70.61%」这一个数分不出两种完全不同的情况:**两类都一般**,
还是**一类很好、另一类掷硬币**。这两种情况该采取的行动不一样。

228 道 VQA 题:

```
                 模型答 no    模型答 yes   |  这一行答对的比例
  真值 no             31           32      |   49.21%   (共 63 题)
                   ┗━ 对 ┛      ┗━ 错 ┛
  真值 yes            32          133      |   80.61%   (共 165 题)
                   ┗━ 错 ┛      ┗━ 对 ┛
  ────────────────────────────────────────
  模型答案的分布    no 63       yes 165
  真值的分布        no 63       yes 165     ← 与上一行完全相同
```

**预测的边缘分布和真值的边缘分布完全相同(165/63),但 `gt_no` 的准确率恰好是
掷硬币(49.21%)。** 模型学到了「多久该说 yes」这个先验,没学到「什么时候该说 no」
这个判别信号——SFT 的似然目标可以只靠匹配边缘分布就拿到不错的 loss。

#### 一对样本把这件事说透了

index=127 和 128 是**同一张桌面照片**上相邻的两道题 —— 证据是两题里 roborefer
检测到的 speaker 坐标完全相同,都是 `(0.154, 0.491)`。句式相同,模型都答 **Yes**,
而真值一个 Yes 一个 No。

```
index=127   gt=Yes   acc=1.0
问题: Can the speaker fit in front of the monitor?
  → roborefer.detect_one("speaker")  → [(0.154, 0.491)]
  → roborefer.detect_one("monitor")  → [(0.553, 0.194)]
  <think> ... The space in front of the monitor appears to be very limited -
          there's just a small amount of desk space between the monitor base and
          where the camera is positioned.
          Given the speaker's size and the limited space in front of the monitor,
          it would likely fit, though it would be a very tight fit. </think>
  <answer> Yes </answer>                                     ← 蒙对了

index=128   gt=No    acc=0.0
问题: Can the cup fit in front of the speaker?
  → roborefer.detect_one("cup")      → [(0.78, 0.494)]
  → roborefer.detect_one("speaker")  → [(0.154, 0.491)]      ← 同一个场景
  <think> ... They are at similar vertical positions (y ≈ 0.49)
          The question asks if the cup can fit "in front of" the speaker. This
          means: is there space between the camera viewpoint and the speaker ...
          There appears to be desk space in front of the speaker ...
          it appears there would be sufficient room to place the cup there. </think>
  <answer> Yes </answer>                                     ← 错
```

两条轨迹的**结构完全一致**:两次 roborefer 取两个 2D 质心 → 用「看图觉得」补完深度
和尺寸 → 答 Yes。127 里它甚至一路推向「空间非常有限、会很挤」,结论仍然是
"would likely fit";128 里换成「看起来桌面有空间」,同样 Yes。

**两题都没有查过深度。** 128 里它自己把问题翻译对了 ——
"is there space between the camera viewpoint and the speaker" —— 这是个深度问题,
而它手上只有 x/y,两个物体的 y 还几乎相同(0.494 vs 0.491)。
`depth_estimator` 就在它的 schema 里,§5.4 证明它会用,这里没调。

所以模型不是在判别,是在**输出这个句式的先验答案**。混淆矩阵的形状正是这个行为的
统计后果:边缘分布能对上,`gt_no` 那一行掉到掷硬币。

对照 P5 对官方(RL 之后)checkpoint 的同一张表:`gt_no` 47.62%、`gt_yes` 83.03%。
**本 ckpt 在 `gt_no` 上其实略好(+1 题),落后的 4–6 题全部来自 `gt_yes`。**

这一条同时给出一个**负面预期**:`gt_no` 的掷硬币水平在论文的 RL 之后**依然存在**,
所以不要指望 RL 能修掉它——那更可能是任务/数据层面的性质。

---

## 6. Repo 自身的问题(本次踩到并确认的)

这些不是环境搭建问题,是上游代码/脚本自带的,换任何机器都会遇到。

### 6.1 `vlm.py:116` 把配置里的 dtype 吃掉了

`run_eval.sh` 给 vlm 工具传的是 `'dtype': 'float16'`,但 `vlm.py:116` 写死
`torch_dtype="auto"`,于是 Molmo 以 **fp32** 加载。实测单卡占用 **30.2 GiB**。

**本次不修**——论文结果是在这个行为下跑出来的。但它的代价是实测级别的:
Molmo 与两个 DepthPro actor 同卡时该卡合计 **46.3 GiB / 48 GiB**,
**40 GB 的卡装不下**(这也从实测上确认了「4× A100 40GB 不够」这个判断)。
而在这四个 benchmark 上 vlm 一次都没被调用过,也就是说将近一整张 A6000
全程 0% 利用率地常驻着。省卡的第一抓手就是这里。

### 6.2 `run_eval.sh` 的两个默认值会让 Ray 静默挂起

```bash
NUM_GPUS="${NUM_GPUS:-8}"      # :50   默认 8
EVAL_GPUS="${EVAL_GPUS:-4}"    # :51   默认 4 → trainer.n_gpus_per_node
```

脚本是按 8 卡节点写的。4 卡机器上不覆盖 `EVAL_GPUS`,七工具的 2.3 加上 verl 要的 4
等于 6.3 > 4.0,**Ray 不报错,actor 无限排队**。必须 `NUM_GPUS=4 EVAL_GPUS=1`。

即便设成 1 仍然偏紧:**策略侧要的是一张完整的卡,不能是分数**,需要 Ray 恰好把 2.3
压成 1.0/1.0/0.3 留一张整卡出来。拿不到整卡时的表现是
`verl/utils/device.py` 的 `is_cuda_available = torch.cuda.is_available()`
(**模块导入时求值**)为 False,`get_device_name()` 永久返回 `cpu`,于是
`fsdp_workers.py:158` 拼出非法的 backend 串:

```
ValueError: Duplicate device type cpu in backend string: nccl.
    The custom backend string argument is invalid: cpu:gloo,cpu:nccl.
```

实测布局(四卡,全工具存活):

```
GPU0  20.0 GiB  util 80%    roborefer 18.6 GiB + 两个小 actor + sglang http
GPU1  46.3 GiB  util  0%    Molmo 30.2 + 12.3 + 3.8     ← 几乎满,全程空转
GPU2   0.7 GiB  util  0%
GPU3  41.6 GiB  util 44%    verl FSDP 15.0 + sglang scheduler 26.7
```

### 6.3 `conda activate` 在子进程里失效

`run_eval.sh:84` 调 `conda activate spacetools-rl`,但 `conda` 是 shell 函数,
不跨子进程:

```
CondaError: Run 'conda init' before 'conda activate'
```

解法是 `BASH_ENV=/opt/conda-st/etc/profile.d/conda.sh`——非交互 bash 会自动 source 它,
函数因此在子 shell 里存在。不需要改脚本。

### 6.4 benchmark 数据集的目录结构和脚本的映射对不上

`run_eval.sh` 的 `BENCHMARKS` 映射期望嵌套结构
(`robospatial_home_multiturn/test.parquet`、`refspatial_bench/location.parquet` …),
而数据集**无论 `main` 还是钉死的 `1d539ac9...` 都是扁平的** `data/<benchkey>.parquet`。
上游改过结构,脚本没跟上,直接跑必然:

```
Missing: .../benchmarks/robospatial_home_multiturn/test.parquet
exit 1
```

顺带:`run_eval.sh` 自己 `snapshot_download` 时**不带 revision**(拉 `main`),
与 `WEIGHTS_PINS.txt` 钉死的 revision 不一致。本次核对过这四个文件在两个 revision 上
逐字节同大小,所以没造成口径差异,但这是个隐患。

### 6.5 `RESTORE.sh` 校验的文件名和包里的不一致

`RESTORE.sh` 第 1 步执行 `sha256sum -c SHA256SUMS`,而包里给的叫
`SHA256SUMS.remote`,照文档跑必然 die 在「包损坏了」。

### 6.6 `28_chain.sh` 失败时仍然 `exit 0`

跨环境链测试把原始输出写 `/workspace/logs/chain_raw.log`,而**这个目录不在 MANIFEST
列出的必备目录里**(只列了 `checkpoints` 和 `hf`)。目录不存在时后面所有 grep 读空:

```
28_chain.sh: line 33: /workspace/logs/chain_raw.log: No such file or directory
  链上失败环数:
  ✗ 链没通
```

**然后它 `exit 0`**,于是 VERIFY 的汇总里这一项显示「退出码 0」并打出「✓ 三项全过」。
这一项**根本没测,却报了通过**。

同一个脚本还有第二个触发条件:它用 `ray.init(num_cpus=8, num_gpus=1, ...)`,
而连接**已存在**的集群时 Ray 会直接拒绝这两个参数。上一次 eval 留下的
`/root/tmp/ray/ray_current_cluster` 地址文件就足以触发:

```
Connecting to existing Ray cluster at address: ...:6379
ValueError: When connecting to an existing cluster, num_cpus and num_gpus must not be provided.
```

### 6.7 五个 conda 环境的 Python 补丁版本不一致

```
spacetools-rl               3.11.16   ← Ray 集群头节点
spacetools-tool-vlm         3.11.0
spacetools-tool-roborefer   3.11.16
spacetools-tool-bbox        3.11.0
spacetools-tool-graspgen    3.11.16
```

Ray 的 `check_version_info` 默认比完整版本串,`3.11.0 ≠ 3.11.16` 抛 RuntimeError,
那两个环境里的 **8 个 actor 全部入不了集群**。

**这个故障原来的三项验收全都抓不到**——跨环境链测试在未修状态下是**通过**的,
它只在 eval / RL 规模下暴露。而 `RESTORE.sh` 的环境检查把这个版本差异
**原样打印出来还判了 OK**(只验 `bin/python` 可执行)。

影响面:四个空间推理 key 只用 roborefer + 各一次 depth/vision_ops,受影响有限;
依赖全套工具的 key(`blinkdepth` / `cvb3ddepth` / `boppose` / `bopgrasp`)会受影响;
**RL 阶段这两个环境里有 23 个 actor。**

修法与验证见 `00_environment/eval_rl_env/env_package_revision_20260912.md`,已随 `POSTRESTORE.sh` 推到 HF repo,
`VERIFY.sh` 也加了第 0 项前置检查把这个故障挡在 eval 之前。

### 6.8 roborefer 的 deepspeed 硬依赖 `CUDA_HOME`

llava 的**推理**路径模块级无条件 `import deepspeed.comm`
(`llava/train/sequence_parallel/globals.py:22`,即环境 README 洞 #17),
deepspeed 在 import 时读 `CUDA_HOME`。打包机上有系统 `/usr/local/cuda-12.8`,
命中 torch 的第三条 fallback;只装驱动的机器上没有。

真实 eval 路径走 Ray 的 `runtime_env={"conda": ...}`,会激活 conda,
`$CONDA_PREFIX/bin/nvcc` 在 PATH 上,torch 能解析出 `CUDA_HOME`,**所以 eval 不受影响**;
但 `04_smoke.sh:42` 直连 env 的 python、不激活 conda,冒烟会在 roborefer 上失败。

---

## 7. 结论

**1. checkpoint 可用,可以进 RL。** 两项判据都过,但两项的「过」不是一回事:

- **RefSpatial 余量宽。** 判据 48%,实测 52.58%,4.58 pp ≈ **13 题**的余量。把 277 题里
  最贴边的十几题全算成错也还在线上,所以这一项跑哪一次都过。
- **RoboSpatial 是「期望过线」,不是「每次都过线」。** 判据 60% = 210/350,而单次运行的
  可能取值只能落在 **199 ~ 228** 这个区间里(199 = 两次都对的恒对样本,228 = 恒对 + 29
  个翻转样本全中),**210 正好落在区间内部**——这就是「判据落在翻转带内」的意思。
  含义是:单次读数在线上还是线下,由那 29 个翻转样本这一次怎么落决定,不由 checkpoint
  决定。要够 210 需要翻转样本中 11 个落对,单次落对率 ≈ 0.5,算下来**一次运行读到线下
  的概率约 6.8%**;期望值 213.5 题(61.00%)在线上,所以按期望是过的。
  实际两次观测正是一上一下:61.71%(216)和 60.29%(211)。

  这一条的实践后果:**只跑一次、运气差,就会读到 59.x%,把一个可用的 checkpoint 判成
  不可用。** 所以下面第 2 条要求按区间报、按期望做决定。补跑到 5 次取中位数可以把误判
  概率压到 0.28%,但它压的只是读数噪声,不改变 61.00% 这个结论(§4.1),所以没补。

**2. 分数必须按区间报。** 策略侧不是逐位确定的:两次运行 350 个样本里 29 个翻转、
199/350 条生成不同。来源已定位——**工具侧零非确定性**(350 个样本里零例「同一个查询
返回不同结果」),全部来自策略 rollout(sglang 的连续批处理使批组成逐次不同,
归约顺序变化,近似平局处 argmax 翻转)。Vacant 是噪声大头,因为它的判据是
**连续预测上的二值凸包归属**,贴边样本被极小扰动推过边线就翻。

**3. 运行侧没有可疑之处**,七项健康指标全为零,工具直方图与官方 ckpt 的行为几乎重合。
**所以差距是真差距,不是运行事故。**

**4. 差距的性质已经定位到样本级**:不是推理能力弱,是**工具使用策略保守**。
`depth_estimator` 在 350 个样本里只触发 1 次,而那一次算对了;需要深度信息的
VQA 题(§5.3)它自己都把问题翻译成了深度问题,却用 2D 坐标硬答。

---

## 8. 为什么需要继续做 RL

### 8.1 SFT 目标和任务目标之间有结构性缺口

SFT 学的是「模仿示范轨迹的 token 分布」,而这个任务的目标是「答对」。两者在
**工具调用的决策**上分道扬镳:

- 调一次 roborefer 就作答,和调 depth + 两次 vision_ops 再作答,**在 SFT 的 loss 里
  没有本质区别**——都只是不同的 token 序列,谁在训练集里出现得多就学谁。
- 但在**答对**这个目标下两者天差地别:§5.3 少调一次深度就错,§5.4 调了就对。
- SFT 也没有任何机制告诉模型「你对工具返回的断言是无依据的」(§5.5 那句
  "This point is within the vacant space")。奖励信号才会。

§5.6 的混淆矩阵是同一件事的另一个切面:**预测的边缘分布和真值边缘分布完全一致
(165/63),而 `gt_no` 恰好是掷硬币。** SFT 可以靠拟合先验拿到好 loss,
判别能力却没上去。per-sample 的奖励不接受这种交换。

### 8.2 空间已经量化过,不是「也许还能涨」

| | 当前 | 官方 ckpt(RL 后) | 论文 | 空间 |
|---|--:|--:|--:|---|
| RoboSpatial Overall | 61.00 | 65.43–66.00 | 70.00 | **9 pp** |
| RefSpatial 三项 | 52.58 | 53.35 | 53.07 | ~0 |

**空间全在 RoboSpatial,而且恰好落在 RL 该起作用的地方。** 对照两档的链路复杂度:

```
RefSpatial    277/277 单一链路,1.00 次调用/样本   → SFT 已饱和,RL 无从加力
RoboSpatial   1.649 次/样本,主链路仅覆盖 64%      → 有决策空间,RL 的作用域
```

**「主链路覆盖率」= 出现最多的那一种链路签名占了多少样本。** RoboSpatial 的每样本调用
次数分布是 `{1 次: 124, 2 次: 225, 3 次: 1}`(§4.3),最多的那一种(`roborefer×2`)
**225/350 = 64%**,剩下 36% 走的是别的形态;RefSpatial 是 277/277 全走
`roborefer×1@2t`,覆盖率 100%。

这个数回答的是「**还有没有决策空间**」:

- 覆盖率 100% = 模型对每个样本都做同一件事,策略上没有分歧,**RL 能改的只剩抄写方式**。
- 覆盖率 64% = 模型自己在样本之间就已经在做不同的选择,那么「什么情况下该走哪条链」
  是一个真实存在、而且当前选得不够好的决策——§5.3 是选错的那一侧(该调深度没调),
  §5.4 是选对的那一侧(调了就对)。**这正是 per-sample 奖励可以重新加权的东西。**

**平手处恰是 RL 无用之处,落后处恰是 RL 有用之处** —— 这是支持进 RL 最强的一条证据。

### 8.3 一条负面预期,应该提前写下来

`gt_no` 的掷硬币水平在论文的 RL 之后**依然存在**(官方 ckpt 47.62%,本 ckpt 49.21%)。
**不要把它列进 RL 的预期收益。** 它更可能是任务/数据层面的性质。

### 8.4 对 RL 奖励设计的一条输入

Vacant 的奖励是**连续预测上的二值凸包归属判据**。67 个错题里 16% 在凸包边界
0.02 以内、36% 在 0.05 以内。这意味着 RL 在这一类任务上拿到的是**不连续、
信息量低**的奖励:差 0.001 和差 0.5 都是 0 分,梯度上无法区分「快对了」和「完全错」。
若要在 RoboSpatial 上取得进展,这一格的奖励塑形值得单独考虑
(例如对凸包外的点给一个随距离衰减的部分奖励)。

---

## 9. SFT 训练到什么程度才值得做 RL

从本次的数据可以抽出几条可检验的闸门。按重要性排:

### 闸门一:工具调用的**格式**必须已经稳定(硬性)

```
本次实测   每样本都有工具调用     627 / 627
           畸形 tool call         0
           缺 <answer>             0
           轮数耗尽(8 轮上限)     0
```

这是 RL 的前提而不是目标。如果 SFT 还没把调用格式学稳,RL 的 rollout 会有很大
比例因格式错误直接拿 0 奖励,梯度信号几乎全是噪声,GRPO 那种组内比较的
advantage 会大面积归零。**这一项不满分就继续 SFT,不要上 RL。**

### 闸门二:至少一条**多工具链路**必须可用,哪怕触发率很低

§5.4 的 index=176 证明这个 ckpt 能跑
`depth_estimator → $depth_map → vision_ops.index_at` 跨三个 conda 环境的链路
并算对。**能力存在但触发率是 1/350——这正是 RL 要放大的东西。**

反过来说:如果一条多工具链路一次都跑不通(不是不调,而是调了就错),
那是 SFT 覆盖不足或工具接口没学会,RL 只会把一个坏策略强化得更自信。

### 闸门三:目标 benchmark 上**不能已经饱和**

RefSpatial 是反例:SFT 已经和 RL 后的官方 ckpt 平手(277 题差 2 题),
链路 277/277 无变化。**在这种任务上做 RL 是浪费算力**——没有可探索的决策空间。

判断方法就是本次用的:看**每样本工具调用次数的分布**和**主链路覆盖率**。
`{1 次: 100}` 这种分布说明没有决策;`{1: 124, 2: 225, 3: 1}` 才有。

### 闸门四:失败模式要是**决策型**的,不是**能力型**的

本次把 15–17 题的差距定位到了样本级:模型把问题翻译对了、工具也有,就是不调
(§5.3)。这是决策型,RL 的 credit assignment 正对着它。

如果失败模式是「调了工具、拿到正确返回、仍然推不出答案」,那是推理能力不足,
应该继续 SFT 或换基座,RL 的收益会小很多。**区分这两者必须看轨迹,看不出来就
不要凭分数决定上 RL。**

### 一个量化参考

本次通过闸门时的位置:

```
格式健康度      7 / 7 项为零
工具调用覆盖    100% (627/627)
多工具链路      可用,触发率 1/350
目标 benchmark  61.0% vs RL 后 65.4–66.0,空间 9 pp,未饱和
主链路覆盖率    64%(有决策空间)
失败模式        决策型(该调不调),已定位到样本
```

**四条闸门全过、且空间 ≥ 5 pp,就值得上 RL。** 缺任何一条都应该先补 SFT。

---

## 10. 其他值得记的

**判据要写成区间,不要写成点阈值。** 本次的 `RoboSpatial ≥ 60` 正好落在测量的
翻转带内(199 恒对 / 228 上界),导致判据在形式上无法回答问题——第一次读 61.71
像是有 6 题余量,第二次 60.29 只剩 1 题。**后续闸门建议写成「期望值 ≥ X 且
恒对下界 ≥ Y」,或者直接规定「取 N 次中位数」。**

**`Error:` 的 grep 要用子串匹配。** `TypeError:` 里含 `Error:`,子串匹配抓得到;
改成前缀匹配(`startswith("Error:")`)会漏掉整类 Python 内建异常名。

**验收脚本本身也会给假绿灯。** §6.6 是一个具体案例:脚本打印「✗ 链没通」的同时
`exit 0`,汇总据此打「✓ 三项全过」。**消费退出码的一方应该同时检查输出里的成功
标记**,不要只信 `$?`。本次之后 `VERIFY.sh` 已改成这样。

**跑 eval 的机器在跑之前要确认没有残留的 Ray 集群**,包括没有残留的
`ray_current_cluster` 地址文件——它足以让下一次验收的第 3 项失败(§6.6)。
`ray stop --force` 报「57/58 停掉」不必担心,剩下那个通常是 zombie。

**GPU 轨迹要在 eval 期间就采。** 本次的单卡峰值是在运行中途抓到的;跑完再查
`nvidia-smi` 只能看到 0 MiB。建议按 P4 的做法起 1 Hz 采样落盘,它同时也是
「Ray 把 actor 怎么打包的」这个问题的唯一答案来源。

**A100 和 A6000 共用同一个环境包。** 架构实扫确认自编扩展缺 sm_8x 为 0,
A100(sm_80)/ A6000(sm_86)/ L40S(sm_89)都在覆盖范围内,只有 H100(sm_90)不行。
换这两种卡只需要改 `NUM_GPUS` / `EVAL_GPUS` 和重算 GPU 预算,包不用动。

**RL 阶段的 GPU 预算必须按「全工具存活」重算。** eval 规模下 4 张卡已经偏紧
(工具 2.3 + 策略整卡 1.0,且需要 Ray 恰好压对);RL 的 actor 数量是
roborefer 6、sam2/depth/bbox/graspgen 各 5、vision_ops 8,预留约 7 张卡以上。
**4 卡结构上装不下。**
