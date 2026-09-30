# SFT checkpoint eval 结果:可以进 RL(按区间报,不按点值)

跑于 2026-09-11/12 · 依据 `SFT eval 交接文档(未收录)` · RoboSpatial 两次运行

---

## 0. 判据结论

**通过,但必须按区间读。**

| | 判据 | 实测 | 结论 |
|---|---|---|---|
| RoboSpatial | ≥ 60 | 期望 **61.00%**,两次 60.29 / 61.71 | ✅ 期望过线,单次约 7% 概率读到线下 |
| RefSpatial | ≥ 48 | **52.58%** 简单 / 53.07 加权 | ✅ 余量 13 个样本 |

**不要报 RoboSpatial 的单次点值。** 第一次 61.71 看着有 6 个样本余量,第二次 60.29
只剩 1 个。测量散布与判据余量同量级,点值会诱导错误读法。

推荐报法:

```
RoboSpatial  61.0%  [恒对下界 56.86 · 恒错上界 65.14]  两次观测 60.29 / 61.71
```

---

## 1. 分数

### 两次运行(RoboSpatial)

| | n | run 1 | run 2 | 恒对 | 恒错 | 翻转 | 期望 ± sd |
|---|--:|--:|--:|--:|--:|--:|--:|
| RoboSpatial · Overall | 350 | 61.71% (216) | 60.29% (211) | 199 | 122 | 29 | **61.00% ± 0.77 pp** |
| RoboSpatial · VQA | 228 | 70.61% (161) | 71.93% (164) | 152 | 55 | 21 | 71.27% ± 1.00 pp |
| RoboSpatial · Vacant | 122 | 45.08% (55) | 38.52% (47) | 47 | 67 | 8 | 41.80% ± 1.16 pp |
| RefSpatial · Location | 100 | 54.00% (54) | — | | | | |
| RefSpatial · Placement | 100 | 57.00% (57) | — | | | | |
| RefSpatial · Unseen | 77 | 46.75% (36) | — | | | | |
| RefSpatial · 三项 | 277 | 52.58 简单 / 53.07 加权 | — | | | | |

判据 60% = 210/350 落在**翻转带内**(199 恒对 ~ 228 上界),清关需要 29 个翻转里中 11 个,
观测翻转成功率 ≈ 0.50(run1 中 17/29、run2 中 12/29),单次读到线下概率 **6.8%**。
5 次取中位数读到线下的概率 0.28%,所以**补跑到 5 次没有意义** —— 判据不会因此翻。

### 与论文 / 官方 ckpt 复现的对照

| Table 2 行 | n | 论文 | P4/P5 复现<br>(官方 ckpt) | **本 SFT ckpt** | 少答对几题 |
|---|--:|--:|--:|--:|--:|
| RoboSpatial · VQA | 228 | 79.38 | 73.25–73.68 | 71.27(期望) | 4–6 题 |
| RoboSpatial · Vacant | 122 | 52.46 | 50.82–51.64 | 41.80(期望) | 11–12 题 |
| RoboSpatial · Overall‡ | 350 | 70.00 | 65.43–66.00 | 61.00(期望) | 15–17 题 |
| RefSpatial · 三项 | 277 | 53.07 | 53.35 简单 / 53.79 加权 | 52.58 / 53.07 | **2 题** |

‡ Overall 是 VQA/Vacant 的样本加权,非独立测量(P5 §1.2)。
VQA / Vacant 按 GT 形态拆(`Yes`/`No` → VQA,点列表 → Vacant),得 **228 / 122**,与 P5 逐个吻合。

**官方 ckpt 是 RL 之后的模型,本 ckpt 是纯 SFT** —— 这是拿起点比成品。差距落点有解释:

| | 推理链 | 差距 | 读法 |
|---|---|--:|---|
| RefSpatial | 277/277 单一链路 `roborefer×1@2t` | 2 题 | 策略是 RoboRefer 的薄包装,SFT 够用,RL 无从加力 |
| RoboSpatial | 1.65 次/样本,主链路仅覆盖 64% | 15–17 题 | 链路多样、需要决策,正是 RL 的作用域 |

**平手处恰是 RL 无用之处,落后处恰是 RL 有用之处。** 对「该不该进 RL」是正面信号。

> RefSpatial 加权 53.07 与论文 53.07 相同是巧合,不构成证据(P5 §6)。

---

## 2. 非确定性:来源在策略侧,工具侧确定

两次运行 350 样本逐样本对比:

```
完全相同(tool_call + tool_response + answer)   289
C  tool_call 就不同        (策略,调用之前)      11
B  同 call,response 不同   (roborefer 侧)        0     ← 零
A  同 call 同 response,答案不同 (策略)           50
逐字节相同的完整生成                            151
```

**工具侧零非确定性。** `<think>` 文本会飘(289 − 151 = 138 个样本措辞不同但落点相同),
工具交互与最终答案一致。P5 §4.3 的非确定性来源在这条链路上表现为 **sglang 的批式推理**:
连续批处理使批组成逐次不同,归约顺序变化,近似平局处 argmax 翻转。

**不建议把 sglang 弄成逐位确定**(batch=1 牺牲吞吐,或动 radix cache 等推理路径)。
承认散布、报区间 —— P5 对小 benchmark 已是这个处方。

### Vacant 为什么是噪声大头:二值区域判据 + 连续预测

打分函数 `verl/utils/reward_score/robos_all.py`:

```
"point lies within (or on the boundary of) the convex hull of the ground-truth"
compute_points_score(..., point_evaluation_method="convex_hull")
    1.0 if point is within convex hull, 0.0 otherwise
```

**判的是预测点是否落在 10 个 GT 点的凸包内,二值。**
用仓库自己的 `_convex_hull` / `_is_point_in_convex_polygon` 重算,
**与落盘 acc 在 122/122 上完全一致、零分歧**(与 P5 复核 pose 指标同一手法)。

**这条容易读错,所以单独强调:GT 给的 10 个点是可接受区域的采样,不是容差圆心。**
判据是凸包归属,不是「到最近 GT 点的距离」。反例就在数据里:idx 87 在 run1 距最近
GT 点 0.0605 判**对**,run2 距 0.0191 判**错** —— 离某个 GT 点更近,却落在凸包外。
**所以分析 Vacant 的错题必须算到凸包边界的距离,算到最近点的距离是错的度量。**

8 个翻转的 Vacant 样本全是「run1 在凸包内 → run2 在凸包外」,出界距离很小:

```
idx  87  0.0011    idx 107  0.0188    idx 105  0.0460    idx   5  0.0684
idx  22  0.0169    idx 100  0.0456    idx 106  0.0477    idx  45  0.1004
```

八个里五个在 0.05 以内,idx 87 只差 0.0011。**贴着凸包边界的样本被策略侧极小扰动
推过边线,二值分数就翻。** 坐标位移方向各不相同(均值仅 +0.009/−0.014),不是整体漂移。

> 八个全部同向在独立掷币模型下概率 1/128,但这 8 个是**按「两次之间有差异」筛出来的**,
> 筛选以两次结果为条件,同向更可能是选择效应而非样本相关。第三次运行会给出一组
> 不同的翻转样本来检验。

按正确判据(到凸包**边界**的距离),67 个错的 Vacant 样本:

```
中位数 0.0761   p25 0.0283   p75 0.1790   最小 0.0018   最大 0.5157
距凸包边 ≤0.02   11 题 (16%)      ≤0.05   24 题 (36%)      ≤0.10   41 题 (61%)
```

---

## 3. 运行健康

两次运行都通过 P5 §5.1 的五项:

```
                     run 1      run 2
OOM                    0          0
工具错误响应(样本级)   0/627      0/350
缺 <answer>             0          —
轮数耗尽(8 轮上限)     0          —
截断的工具响应          0          —
畸形 tool call          0          —
每样本都有工具调用    627/627      —
```

工具直方图与 P5 记录的官方 ckpt 行为几乎重合:

```
robospatial  本次 1.654 次/样本 · 64.3% 恰好 2 次    P5: 1.63 次/样本 · 62.3% 主链路
refplacement 100/100 单次   refunseen 77/77 单次     P5: 100%
reflocation  98 单次 + 2 双次                        P5: 100% 单次
```

四个 benchmark 实际调用的工具只有三个:

```
roborefer        627 / 627 样本
depth_estimator    1 次(robospatial index=176)
vision_ops         1 次(同一样本)
sam2 / vlm / bounding_box / grasp_generator   0 次
```

### 日志里的 `Error:` 计数逐条查明不影响样本

两次运行各有 **8 条** `RuntimeError: Version mismatch`,逐条同型、可复现,
全部来自 §4 那个 Python 版本问题,**一条都没碰到任何样本**(依据:627/627 样本
都有工具调用、0 条工具响应含错误文本,且唯一用到受影响环境的那个样本
tool_response 逐条核过、`acc=1.0`)。

---

## 4. Python 版本不一致:已修,而且修完暴露了一个更紧的约束

### 问题

```
cluster 启动于  Python 3.11.16   (spacetools-rl)
这些进程启动于  Python 3.11.0    (spacetools-tool-vlm / spacetools-tool-bbox)
Ray 2.47.1 两侧相同
```

Ray 的 `check_version_info` 默认比完整 Python 版本串,`3.11.0 ≠ 3.11.16` 直接拒绝入集群。
八条错误正好对应那两个环境的八个 actor(vlm 环境 sam2×2+depth×2+vlm×1=5,
bbox 环境 bbox×2+vision_ops×1=3)。

`RESTORE.sh` 的环境检查把这个版本差异**原样打印出来还判了 OK**(只验 `bin/python` 可执行),
是一个假绿灯。

### 修法(偏离 #6):Ray 自带的 minor 档

`ray/_private/utils.py:1502` 的 `check_version_info` 支持
`python_version_match_level="minor"`,此时只 `logger.warning` 不抛;
但 `node.py:454`(由 `worker.py:2437` 的 `connect` 调用)没传这个参数,默认 `"patch"`。
在**两个 3.11.0 环境**里给这一个调用点补上该参数,备份在 `node.py.orig`。

**为什么不升 Python:**

```
CondaToSNonInteractiveError: Terms of Service have not been accepted for:
    https://repo.anaconda.com/pkgs/main   https://repo.anaconda.com/pkgs/r
```

接受 Anaconda 商业条款是组织的法务决定;而且重新求解会动到
numpy 1.26.4 / transformers 4.53.2 这些钉死的版本,那才是本项目最怕的静默改口径。

**为什么 minor 档是安全的、不是绕过:**

```
spacetools-rl    bytecode magic 3495 | pickle proto 4 | 3.11.16
tool-vlm         bytecode magic 3495 | pickle proto 4 | 3.11.0
tool-bbox        bytecode magic 3495 | pickle proto 4 | 3.11.0
```

magic 与 pickle 协议两侧相同,code object 和 pickle 互通 —— 这正是 Ray 提供 minor 档的场景。
**该改动只放宽一个版本检查,不可能改变数值。**

### 三重验证

```
单元    minor 档接受 3.11.0 vs 3.11.16(仅 warning);默认 patch 档仍然拒绝
VERIFY  三项真绿:环境闸门 RC=0 · 4619 个 .so 缺 sm_8x 0 · 七工具 7/7(首次与基线一致)
        跨环境链 4 环 STEP_OK,sam2/depth 在 tool-vlm、bbox/vision_ops 在 tool-bbox,
        两个补丁环境都成功入集群
端到端  RuntimeError: Version mismatch  8 → 0
        Python patch version mismatch (warning)  0 → 8
        10 个 tool actor 全部加入(之前只有一部分)
```

### ⚠️ 修完暴露的约束:全工具存活时 4 张卡排不开策略

blinkdepth(用来端到端验证的 benchmark)两次都死在 verl 侧:

```
ValueError: Duplicate device type cpu in backend string: nccl.
    The custom backend string argument is invalid: cpu:gloo,cpu:nccl
    fsdp_workers.py:158  backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}"
ActorDiedError: WorkerDict.__init__()
```

`verl/utils/device.py:104` 的 `is_cuda_available = torch.cuda.is_available()` 是
**模块导入时求值**,所以 WorkerDict 进程导入那一刻没有可见 GPU,`get_device_name()`
就永久返回 `cpu`,拼出非法的 `cpu:gloo,cpu:nccl`。

根因是 GPU 预算:

```
工具逻辑预留 2.3  +  策略要一整张 1.0  =  3.3 ≤ 4.0
```

逻辑总量够,但**策略要的是一张完整的卡**,需要 Ray 把 2.3 压进 3 张、留一张整的。

**之前四次 eval 成功,是因为工具的真实占用小于 2.3。** nvidia-smi 的进程归属里当时只数到
**7 个 ToolActor**,而配置有 10 个 actor(其中 `vision_ops` 不占卡),即有占卡的 actor
当时不在集群里 —— 正是被版本检查挡住的那些。补丁修好后 10 个全部加入,预留升到完整 2.3,
Ray 再压不出一张完整空卡。

**所以这不是补丁引入的 bug,是补丁取消了一个一直在替我们腾卡的故障。**

对已有分数**无影响**:那四个 benchmark 只调用了 roborefer / depth_estimator / vision_ops,
三者全部成功返回真结果(见 §3 的直方图)。但这是一条必须记进出处的事实:
**那四次 eval 是在部分工具 actor 缺席的情况下跑的。**

---

## 5. 另一个假绿灯:残留的 Ray 地址文件

`VERIFY.sh` 第三项一度报出自相矛盾的结果 —— 同时出现 `✗ 链没通` 和
「跨环境链 退出码 0」,汇总还打了 `✓ 三项全过`。

根因:`/root/tmp/ray/ray_current_cluster`(上一次 eval 留下的 19 字节地址文件)
让 `28_chain.sh` 的 `ray.init(num_cpus=8, num_gpus=1, ...)` 以为集群还在:

```
Connecting to existing Ray cluster at address: 172.27.124.124:6379
ValueError: When connecting to an existing cluster, num_cpus and num_gpus must not be provided.
```

python 块在第 1 秒就死了,而 `28_chain.sh` 仍然 `exit 0`。

**结论:有 Ray 头节点在跑、或刚杀掉但没清 temp dir 时,不能跑 VERIFY.sh。**
清掉 `/root/tmp/ray` 后重跑,链正常通过(ROUTER_OK 33.2 秒,四环 STEP_OK)。

`ray stop --force` 报「57/58 停掉」不必担心:剩下那个是 zombie(已 defunct,不占资源)。

---

## 6. GPU 用法(实测,1 Hz × 75 采样)

```
卡        4× RTX A6000 (49140 MiB each)
驱动      580.173.02   (打包机 580.159.04,同分支)
NUM_GPUS  4     ← 必须覆盖,默认 8
EVAL_GPUS 1     ← 必须覆盖,默认 4
耗时      run1 四项 32 分 41 秒 · run2 单项(robospatial)20 分 55 秒
```

### Ray 实际怎么打包的(部分工具 actor 缺席时的布局)

```
GPU0   20525 MiB (20.0 GiB)   util 峰100% 均80%
       roborefer          18628 MiB      ToolActor ×2  634 MiB ×2
       SGLangHttpServer     604 MiB

GPU1   47441 MiB (46.3 GiB)   util 0%   ← 几乎满,全程空转
       ToolActor          30926 MiB (30.2 GiB)  ← Molmo
       ToolActor          12574 MiB (12.3 GiB)
       ToolActor           3920 MiB ( 3.8 GiB)

GPU2     725 MiB ( 0.7 GiB)   util 0%
GPU3   42616 MiB (41.6 GiB)   util 峰100% 均44%
       WorkerDict         15310 MiB  ← verl FSDP
       sglang::scheduler  27292 MiB  ← 策略 rollout
```

**交接文档 §2 的推算得到实测确认。** 它预测最坏打包
`vlm 0.6 + depth 0.2 + depth 0.2` = 32.11 + 7.94×2 = **47.99 GiB**,据此判 4× A100 40GB
「不够」。GPU1 上就是这个打包,实测 **46.3 GiB**,差 1.7 GiB。
**那个坏打包并没有被第 4 张卡避开 —— 它真的发生了,只是 A6000 的 48 GB 刚好装下。
换 40 GB 的卡这张必 OOM。**

Molmo 实测 30.2 GiB,印证 fp32 那个洞(`vlm.py:116` 吃掉 `dtype: float16`)。

> 冒烟阶段报的 `vlm 峰值 7.66 GiB` 不矛盾:那是
> `torch.cuda.max_memory_allocated()`,只计 torch allocator,不是卡上总占用。

**GPU1 那 46.3 GiB 全程 0% 利用率** —— robospatial 350 个样本里只有 1 个用到
depth/vision_ops,将近一整张 A6000 为一次调用常驻。这是「省卡两个抓手」
(修 fp32、按 benchmark 裁工具集)的实测依据。

### `EVAL_GPUS` 默认值会让 Ray 永久挂起

```
七工具逻辑预留(run_eval.sh:132-142,与交接文档 §2 表一致)
roborefer 1×0.6 + vlm 1×0.6 + sam2 2×0.2 + depth 2×0.2
        + bbox 2×0.1 + vision_ops 1×0 + graspgen 1×0.1  = 2.3
默认 EVAL_GPUS=4  →  2.3 + 4 = 6.3 > 4.0   Ray 不报错,actor 无限排队
设成 1            →  2.3 + 1 = 3.3 ≤ 4.0   逻辑上够,但见 §4 的整卡约束
```

---

## 7. 出处

```
SFT ckpt   qzpm55555/spacetools-sft-v1-4xa6000
           commit 91fd4bdf6d2ffe3b355cc650e71925c6e91a15c5
           Qwen2_5_VLForConditionalGeneration · bfloat16 · 7.6 GB
           num_attention_heads=16 · num_hidden_layers=36

环境包     qzpm55555/spacetools-eval-env
           spacetools-envs-20260910-0959.tar.zst
           22102227452 字节(拼接后与 README 记录逐字节一致)· sha256 12/12 OK

仓库       SpaceTools           17d585539b6cc32f2b3c068ea2591762b4583fd9
           SpaceTools-RL        54270e82443d3d2a4c2a737c2d3b33314a991fcc
           SpaceTools-Toolshed  4f0512d092f53abc1e6c5c934245bf83a211466a
           GraspGen             9b3cfc1e5b664698e047ddd482832f6e7796380c
           RoboRefer            d97a995ad28376720a4c8beb64915c58ed16c844

数据       siyich/spacetools-eval-benchmarks @ 1d539ac935872c7aa712c85a77bf4b0cb469c8e8
           350 / 100 / 100 / 77 = 627 ✓ · blinkdepth 124 ✓
           (该 revision 与 main 在这些文件上逐字节同大小)

机器       4× RTX A6000 · Ubuntu 24.04.2 · glibc 2.39 · 驱动 580.173.02 · CUDA 13.0
解码       greedy(verl val_kwargs 默认 temperature 0 / do_sample False,
           run_eval.sh 未覆盖)—— 但策略侧仍非逐位确定,见 §2
工具状态   ⚠️ 四个 benchmark 的两次运行都在部分工具 actor 缺席下完成(见 §4)
```

---

## 8. 为了跑起来做的偏离(仓库代码未改,配置/库补丁六处)

| # | 偏离 | 为什么必须 |
|---|---|---|
| 1 | 装 `nvidia-driver-570-server`(实得 580.173.02) | 机器交付时**驱动完全没装**:卡在 PCI 可见,但无 `/dev/nvidia*`、无内核模块、`dpkg -l` 里 0 个 nvidia 包 |
| 2 | `NUM_GPUS=4 EVAL_GPUS=1` | 默认 8 / 4,后者会让 Ray 永久挂起(§6) |
| 3 | `BASH_ENV=/opt/conda-st/etc/profile.d/conda.sh` | `run_eval.sh:84` 的 `conda activate` 在子进程失效,conda 是 shell 函数不跨进程 |
| 4 | benchmark 软链接到脚本期望的嵌套路径 | `BENCHMARKS` 映射期望 `refspatial_bench/location.parquet` 之类,数据集(main 与钉死版本**都**是)扁平 `data/<key>.parquet`,不处理直接 `Missing: / exit 1`。没改 `run_eval.sh`,因为其 commit 记在 MANIFEST |
| 5 | roborefer 环境加 `activate.d/zz_cuda_home.sh` | `04_smoke.sh:42` 直连 env python 不激活 conda,torch 找不到 nvcc → deepspeed(llava 推理硬依赖,README 洞 #17)抛 `MissingCUDAException`。打包机有系统 `/usr/local/cuda-12.8` 命中 torch 第三条 fallback。真实 eval 走 Ray `runtime_env={"conda":...}` 会激活,本来就能解析 |
| 6 | **两个 3.11.0 环境的 `ray/_private/node.py:454` 补 `python_version_match_level="minor"`** | 见 §4。不改数值,只放宽版本检查;备份 `node.py.orig` |

**没做的事**:没改 `vlm.py:116` 的 fp32(交接文档要求保持口径),没改任何仓库代码,
没装系统 CUDA toolkit,没接受 Anaconda ToS。

偏离 #5 与 #6 已随 `POSTRESTORE.sh` 推到 HF repo,换机器时自动生效,
并由 `VERIFY.sh` 的第 0 项前置检查把关 —— 详见 `00_environment/eval_rl_env/env_package_revision_20260912.md`。

---

## 9. 下一步

**可以进 RL。** 按优先级:

1. **RL 的 GPU 预算必须按「全工具存活」重算,而且要保证策略有一张专属整卡。**
   §4 的教训:4 张卡在 eval 规模下就已经排不开(工具 2.3 + 策略整卡 1.0,
   需要 Ray 恰好压成 1.0/1.0/0.3/空);RL 的 actor 数量远大于此
   (sam2/depth/bbox/graspgen 各 5、vision_ops 8),必然更紧。
   可考虑的抓手:修 `vlm.py:116` 的 fp32(Molmo 实测 30.2 GiB,减半能腾出 15 GiB)、
   按 benchmark 裁工具集、或显式把策略绑到独立卡上。

2. **RoboSpatial 一律按区间报**(61.0% [56.9 / 65.1],两次 60.29 / 61.71)。
   **不要补跑到 5 次** —— 5 次中位数读到线下的概率只有 0.28%,判据不会翻。
   若想量化「真实散布是否比 ±0.8 宽」(两次运行只能看到翻转集的**下界**),
   一次运行就能拿到大部分信息:看 199 个「恒对」里有多少在第三次翻掉。
   这个数字有超出本判据的意义 —— 若真实散布明显更宽,本项目所有单次数字
   (含 P4/P5)都该配更宽的误差棒。

3. **RL 阶段起 1 Hz GPU 轨迹采样**(P4 在 `p4/logs/` 就是这么做的)。

4. **环境包下次重建时把五个环境的 Python 统一到 3.11.16**,这样偏离 #6 可以撤掉。

### 对 RL 奖励设计的一条输入

Vacant 的奖励是**连续预测上的二值凸包归属判据**(§2)。贴边样本(67 个错题里
16% 在 0.02 以内、36% 在 0.05 以内)会因极小扰动在 0 和 1 之间跳。这既是 eval 噪声
的来源,也意味着 RL 在这类任务上拿到的是**不连续、信息量低的奖励信号**。
若 RL 要在 RoboSpatial 上取得进展,这一格的奖励塑形值得单独考虑。
