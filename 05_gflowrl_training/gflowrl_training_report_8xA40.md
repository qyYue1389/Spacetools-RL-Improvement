# P7 训练报告:SpaceTools Step 4 用 GFlowRL C′ 替换 GRPO

> 训练时间:2026-09-16 08:35:04 → 2026-09-17 06:32:35 UTC(21 小时 57 分)
> 机器:RunPod 8× A40 48 GB(已于本报告生成后释放)
> 结果:`FULL_DONE_PASS`,85 步跑完,退出码 0
> 产物:`qzpm55555/spacetools-p7-gflowrl-cprime-8xa40`(HF 私有 repo)
> 本文件与同目录下的原始文件一起构成这次训练的完整记录

---

## 0. 一句话

把 SpaceTools 的 Step 4 RL 从 GRPO 换成 GFlowRL(arXiv:2607.13394)的 **C′ 变体**,
在自训 SFT checkpoint 上跑满 1 个 epoch(85 步),全程无崩溃、无 OOM(训练侧)、
无中断,三个 checkpoint(85 / 60 / 30)已 merge 成 HF 格式并上传。
**效果未知** —— eval 还没跑。

---

## 1. 训练对象与配置

### 1.1 基线与目标

| | |
|---|---|
| base model | `qzpm55555/spacetools-sft-v1-4xa6000` @ `91fd4bdf6d2ffe3b355cc650e71925c6e91a15c5` |
| 架构 | Qwen2.5-VL-3B-Instruct(4.066 B 参数,视觉编码器冻结,可训 2.55 B) |
| 算法 | GFlowRL Eq. 4/6/7/8,变体 `cprime` |
| 对照 | 论文的 GRPO(本次**没有**跑,只训了 GFlowRL 单臂) |
| 数据 | `siyich/spacetools-rlfulltools`,train.parquet **5500 行** |

### 1.2 为什么是 C′

GFlowRL 的 Eq. 4(in-batch log Z)和 Eq. 5/6(flow gap)各自可以做 / 不做序列长度
归一化,三种组合:

| 变体 | Eq.4 | Eq.6 | 结论 |
|---|---|---|---|
| `paper` | 不归一 | 归一 | 漂移项退化成组内公共平移,在 30–49% 的组上把奖励抹平 |
| `normalized` | 归一 | 归一 | 有不动点,但那个不动点是 \|y\|≈334 处的点质量(纯奖励最大化) |
| **`cprime`** | **不归一** | **不归一** | **唯一同时有 Prop. B.1 解析不动点、且"整组被 clip 成同一个值"率为 0 的** |

不动点已用 `tools/p7/p7_fixedpoint_check.py` 数值验到 **7.39e-13**。
代码里 `masked_sum`(而非 `masked_mean`)是 C′ 和 `normalized` 的**唯一**区别,改了不报错,
所以这一行是 load-bearing 的,动过就必须重跑 fixedpoint check。

### 1.3 超参(as-run)

```
算法侧
  loss_mode              gflowrl
  variant                cprime
  beta                   8.0
  eps_low / eps_high     0.2 / 0.28        (Eq.7 非对称 clip)
  eps_is                 0.2               (Eq.8 IS 权重下限)
  kl_loss_coef           0                 ← 被 gflowrl wrapper 覆盖
  use_kl_loss            True              ← 必须保持 True,见 §3.4

训练侧(全部继承自上游 run_rl.sh,一个字没改)
  train_batch_size       64
  rollout.n              5                 (G=5,论文是 16;改了两臂就不可比)
  ppo_mini_batch_size    64                (= train_batch_size → on-policy)
  ppo_micro_batch_size_per_gpu  2
  lr                     1e-6
  grad_clip              1.0
  total_epochs           1
  max_prompt/response_length    8192 / 8192
  max_assistant_turns    8
  max_parallel_calls     8
  max_tool_response_length      2048
```

**85 步不是 86。** 5500 ÷ 64 = 85.9,verl 丢掉不满一个 batch 的尾巴,所以一个 epoch
正好 85 步。交接文档里写的 86 是估算值,实际以 `latest_checkpointed_iteration.txt` 为准。

---

## 2. 机器与环境

### 2.1 硬件

```
GPU        8× NVIDIA A40 (48 GB, GA102, sm_86)
驱动       580.159.04 · CUDA 13.0
CPU/RAM    96 核 / 503 GB
容器盘     120 GB(/opt 的五个 conda 环境在这里)
网络卷     250 GB(/workspace:权重、数据、checkpoint)
OS         Ubuntu 24.04.3 · glibc 2.39
拓扑       GPU0-3 同 NUMA0(PXB 互联),GPU4-7 同 NUMA1,两组之间是 SYS
```

拓扑那一行后来变得很重要:训练用的 4 张卡跨不跨 NUMA 会影响 NCCL,而这台机器的
**PCIe P2P 实际是坏的**(见 §5.1)。

### 2.2 软件栈

```
torch 2.9.1+cu128 · cuda 12.8 · cudnn 9.16.0
transformers 4.57.1 · numpy 1.26.4 · ray 2.47.1
verl 0.8.0.dev · sglang 0.5.6
```

五个 conda 环境,互相依赖冲突(torch 2.3.1 / 2.5.1 / 2.9.1 并存,numpy 1.26 与 2.x 并存),
所以工具必须分环境跑:

```
spacetools-rl              训练主环境(verl + sglang)
spacetools-tool-roborefer  RoboRefer-8B
spacetools-tool-vlm        Molmo-7B-D · SAM2 · DepthPro
spacetools-tool-bbox       3D bbox · vision_ops
spacetools-tool-graspgen   GraspGen
```

环境不是现装的,是从 `qzpm55555/spacetools-eval-env`(22 GB 打包)还原到 `/opt` 的
—— conda 环境里烧死了绝对路径,必须还原到 `/opt/conda-st`,换路径全线崩。

---

## 3. 卡怎么分、工具怎么部署

### 3.1 8 张卡的划分

```
GPU 0-3   Toolshed 工具 actor      TOOL_GPUS=4
GPU 4-7   训练(FSDP + sglang)     TRAIN_GPUS=4
```

这个划分不是上游的做法。上游 `run_rl.sh` 把**同一个** `GPUS_PER_NODE` 同时交给
Toolshed 的 STRICT_PACK placement group 和 `trainer.n_gpus_per_node`。论文用 2 个节点,
PG 落在 node1、trainer 占 node2,所以从没暴露;**单节点上这就是把同一批卡预订两次**,
结果是 PG 吃掉全部卡、trainer 拿不到,或者 PG 永远 ready 不了 —— 死锁,不报错。
拆成 `TOOL_GPUS` / `TRAIN_GPUS` 是我们改的(commit `574ae81`,详见 §4)。

### 3.2 工具 actor 的部署

Toolshed 是 Ray 之上的工具托管层:每个工具是一组 Ray actor,跑在自己的 conda 环境里,
变量(ndarray、点云)通过 Ray 序列化跨环境传递。

上游 `TOOL_CONFIGS` 是按论文的 2 节点规模写的,逻辑需求 **7.8 张卡**:

```
roborefer 6×0.6 + vlm 2×0.6 + sam2 5×0.2 + depth 5×0.2
        + bbox 5×0.1 + vision_ops 8×0 + grasp 5×0.1  = 7.8
```

我们只有 4 张卡给工具,所以加了自动缩放(commit `3e1d02c`),实际落地的是:

```
scaled tool actors by 0.500 to fit tool_gpus=4.0:
  {'roborefer': 3, 'vlm': 1, 'sam2': 2, 'depth_estimator': 2,
   'bounding_box': 2, 'vision_ops': 4, 'grasp_generator': 2}
  -> demand 3.70 logical GPUs
```

**`num_gpus` 是 Ray 的逻辑预留,不是显存配额。** 同一张物理卡上的 actor 共享全部显存,
Ray 只保证「落在同一张卡上的分数加起来 ≤ 1.0」。所以 3.70 只决定"排得下排不下"。

工具侧一共注册了 **48 个方法**(`config/rl_toolshed_config.yaml`),按工具分:
`roborefer.*`、`vlm.*`(Molmo)、`sam2.*`、`depth_estimator.*`、`bounding_box.*`、
`vision_ops.*`、`grasp_generator.*`。模型每轮可以并发调用最多 8 个,最多 8 轮。

### 3.3 训练侧的 4 张卡

```
FSDP 全分片(param_offload=False, optimizer_offload=False)
ref model 的参数 offload 到 CPU (ref.fsdp_config.param_offload=True)
sglang rollout: tensor_model_parallel_size=1, gpu_memory_utilization=0.7, max_num_seqs=256
```

实测峰值显存 **26.5 → 35.3 GB**(48 GB 上限),CPU 侧峰值 151 GB。

### 3.4 一个必须记住的约束

`kl_loss_coef=0` 但 `use_kl_loss=True` 看起来矛盾,其实两者都必要:

- `use_kl_loss=False` 会让 `dp_actor` 根本不去算 `ref_log_prob`,而 GFlowRL 的
  `d = Σ(log π_ref − log π_old)` 正需要它 → 跑不起来
- `kl_loss_coef≠0` 会**重复计入**参考项,因为 d 本身就是 Eq.4/6 里的那个 KL 型量
  → 不报错,但训的是另一个目标

这两种错都是静默的,所以我们在 `ray_trainer.py` 里加了会抛异常的 guard(commit `0b5098a`)。

---

## 4. 相对原 repo 改了什么、为什么

基线是上游 `SpaceTools-RL`。我们的分支 `repro-4xa6000`,HEAD = `c6fef78`,
工作区干净(只有一个未跟踪的 `images` 软链)。**九个 commit,共 6 个文件、+366/−8 行。**
完整 diff 见 `code/p7_all_changes.diff`,完整 commit message 见 `code/p7_commits.txt`。

### 4.1 实现 GFlowRL 本身(2 个 commit)

| commit | 改了什么 | 为什么 |
|---|---|---|
| `33036a5` | `ray_trainer.compute_gflowrl_flow_gap()` + `core_algos.compute_policy_loss_gflowrl()` | 论文 Eq.4/6/7 是无梯度的(只依赖 π_old、π_ref、r,actor 更新期间全冻结),所以放在 driver 上每步算一次;Eq.8 的带梯度部分注册成 `loss_mode='gflowrl'`。三个 variant 开关让三种归一化组合不用改代码就能比 |
| `0b5098a` | `run_rl_gflowrl.sh` 薄封装 + §3.4 那个 guard | 两臂对比只有在"除了目标函数以外全都一样"时才有意义。复制 400 行 `run_rl.sh` 一定会漂移,所以 wrapper 只有一个 7 行的 overrides 数组,靠 `run_rl.sh` 现成的 `"$@"` 透传;hydra 对重复 key 取最后一个,所以这里的 `kl_loss_coef=0` 压得住 `run_rl.sh` 里的 `=0.01` |

**`advantages` 被复用了。** GFlowRL 没有 advantage 这个东西,但注册损失的签名里没有别的
per-sequence 槽位,所以 g̃ 被广播进 `batch["advantages"]`。后果:trainer 打印的
`critic/advantages/*` 那几个指标报的是 g̃ 的统计量,不是 GRPO 的优势。读日志时别看错。

### 4.2 修上游的 bug(5 个 commit)

这些都不是 GFlowRL 相关的,是**在单节点、小机器上跑上游代码必然会撞的**:

| commit | 症状 | 根因 |
|---|---|---|
| `574ae81` | 单节点上 Ray 死锁,不报错 | `GPUS_PER_NODE` 同时给了 PG 和 trainer,同一批卡预订两次(§3.1) |
| `dfbde8e` | `TypeError: PolicyLossConfig.__init__() got an unexpected keyword argument 'gflowrl'`,而且是在 Toolshed 起来几分钟后、在 Ray worker 里抛 | `PolicyLossConfig` 是结构化 dataclass,经 `omega_conf_to_dataclass → hydra.utils.instantiate` 把 `policy_loss` 下每个 key 当 kwarg 传进去,没声明的 key 不是被忽略而是报错。顺带修了一个更安静的半边:driver 从裸 DictConfig 读 beta/eps,而 `core_algos` 从 dataclass 读 `eps_is` —— dataclass 里没这个字段,于是 `GF_EPS_IS` 被静默丢弃,两个配置源对同一个损失的说法不一致 |
| `3e1d02c` | `assert _demand <= tool_gpus` 直接拒绝启动 | 上游 `num_actors` 是按 2 节点写的(7.8 逻辑卡),小机器上不缩放就起不来(§3.2) |
| `5fa00d8` | 加载 20 分钟后 `FileNotFoundError: 'images/xxx.png'` | parquet 里是相对路径,靠 cwd 解析;`ray start` 是在调用者的 cwd 起的 raylet,而 `cd "$VERL_DIR"` 在它之后。把 `cd` 提到 `ray start` 之前 |
| `4474e67` | 成功的 run 报失败;**而且 `ray stop --force` 从没执行过,每次跑完都留一个活着的 Ray 集群** | `cleanup()` 杀掉自己的后台任务然后 `wait`,`wait` 对刚被信号杀掉的任务返回 128+SIGTERM,`set -e` 在 EXIT trap 里同样生效,于是 cleanup 死在 `wait` 那一行 —— 后面的 `ray stop` 再也跑不到。用最小复现(`scripts/exitprobe.sh`)验证过:加 `set +e` 作为 cleanup 第一行就好了 |

### 4.3 加观测(2 个 commit)

| commit | 加了什么 | 为什么 |
|---|---|---|
| `3e93509` | `compute_group_degeneracy_metrics()`(组退化率 / rollout 退化率 / 组内奖励极差)+ 两个 IS 权重守卫 + 修正 `actor/ppo_kl` 的定义 | 奖励退化率是 C′ 可能赢 GRPO 的**全部理由**,而之前引用的 56.9% / 75.8% 来自 eval benchmark 的采样,不是 Step 4 的训练集。调用点放在 `compute_advantage` 之后、gflowrl 分支之前,这样两臂都会记录 —— 只有一臂的数字证明不了两臂的差异。退化按"组内极差严格为 0"判定,而不是方差,因为 pointing / IoU 奖励是连续值 |
| `c6fef78` | β 分解的诊断埋点(见 §5.4) | 见下 |

---

## 5. 训练里用到的优化手段,以及它们各自起了什么作用

| 手段 | 在哪配的 | 起了什么作用 |
|---|---|---|
| **冻结视觉编码器** | `model.freeze_vision_model=true` | 4.066 B 参数里只训 2.55 B。省显存、省算力,也是论文的做法 |
| **FSDP 全分片** | verl 默认 + `param_offload=False` | 参数/梯度/优化器状态在 4 张卡上分片。不 offload 是因为 48 GB 够,offload 会把 PCIe 变成瓶颈(而这台机器的 P2P 本来就坏) |
| **ref model 参数 offload 到 CPU** | `ref.fsdp_config.param_offload=True` | ref 只在每步算一次 log_prob,常驻显存不划算。代价是每步 `timing_s/ref ≈ 122 s` |
| **梯度检查点** | `enable_gradient_checkpointing=True` | 用重算换激活显存。8192+8192 的上下文下这是能跑起来的前提 |
| **remove padding(序列打包)** | `use_remove_padding=True` | 变长序列拼成连续 token 流,避免 pad 位置的无效计算 |
| **flash-attention-2** | `override_config.attn_implementation` | 注意力显存从 O(L²) 降到 O(L) |
| **micro-batch 2 + mini-batch 64** | `ppo_micro_batch_size_per_gpu=2`, `ppo_mini_batch_size=64` | mini_batch == train_batch 让这一步是**严格 on-policy** 的,于是 Eq.8 的 IS 权重恒等于 1(`is_weight_min=1.0`, `collapsed_frac=0.0`)。这不只是省事:一旦为了省显存把 mini_batch 调小,IS 权重就会真的开始工作,每 token −3e-3 的漂移就能把一条 1e3 token 的序列的 w 压到 0.02 —— 它还在 batch 里,但贡献接近 0,有效 batch 悄悄缩水。那两个守卫指标就是为了让这件事被看见 |
| **sglang 异步多轮 rollout** | `rollout.name=sglang`, `max_parallel_calls=8` | 工具调用不阻塞整个 batch。每步 `timing_s/gen ≈ 322 s` 里包含了全部工具往返 |
| **Toolshed 多 actor + 环境隔离** | `TOOL_CONFIGS` | 同一工具多副本并行消化排队;conda 隔离解决 5 套互斥依赖。论文实测 8 并发下比朴素 HTTP 快 3.2× |
| **NCCL 走共享内存** | `NCCL_P2P_DISABLE=1` | **不是优化,是绕过硬件故障**(§5.1)。代价是 all-reduce 过主机内存,更慢 |
| **checkpoint 清道夫** | `scripts/janitor2.py`(自研) | `save_freq=5` × 每个 ckpt 44 GB,250 GB 的卷在第 20 步就会满。清道夫只留最新的完整 ckpt(给自动 resume 用),并在第 30/60 步另存**只含权重分片**的快照(15 GB,不含 7 GB/rank 的 `optim_*.pt`)。先存档再删除,只在最新那个已写到 ≥25 GB 时才动手 |

没用到的(记录一下,避免以后重复讨论):没有用 LoRA(全参数微调)、没有用 offload 优化器、
没有开 `expandable_segments`(它和 sglang 的 TorchMemorySaver 冲突)、没有调 `rollout.n`
(改了两臂就不可比)。

### 性能实测(最后一步)

```
timing_s/step          981 s   (全程均值 925 s,中位 ~930 s)
  ├ gen               322 s   rollout + 工具往返
  ├ update_actor      369 s
  ├ ref               122 s
  ├ save_checkpoint    39 s
  └ 其余(adv 0.07s 等)
perf/total_num_tokens        1,360,105 / 步
perf/throughput              346 token/s
perf/mfu/actor               0.145
显存峰值                      35.3 GB / 48 GB
```

85 步 × 925 s ≈ **21.8 小时**,与墙钟 21 小时 57 分吻合。

---

## 6. `p7_metrics.csv` 怎么读

文件:`metrics/p7_metrics.csv`,85 行,每行一步。由 `scripts/p7tab.py` 从训练日志解析而来
(脚本按需运行,不是守护进程,重跑任何次数都不影响训练)。

### 6.1 列的定义

| 列 | 日志里的 key | 含义 |
|---|---|---|
| `step` | — | 训练步 1–85 |
| `wall_s` | `timing_s/step` | 该步墙钟秒数 |
| `degen` | `reward/degenerate_group_frac` | **奖励退化组占比**:组内 5 个 rollout 的奖励极差严格为 0 的组的比例。GRPO 在这些组上梯度恒为 0 |
| `sat` | `gflowrl/clip_saturation` | Eq.7 的 clip 饱和率:g 落在 [−0.2, +0.28] 之外、被截断的样本比例。**贴到 1.0 意味着幅度信息全丢,只剩符号** |
| `sat_b2/b4/b16` | `gflowrl/sat_at_beta_*` | 把 β 换成 2/4/16 重算的饱和率(复用同一批 r 和 d,不额外前向)。**`sat_b8` 必须与 `sat` 逐位相等** —— 这是诊断块自带的自检,本次训练用的 β 就是 8 |
| `reward` | `gflowrl/reward_term_abs_mean` | flow gap 的**奖励半边** \|mean_group(β·r) − β·r_i\| 的均值,正比于 β |
| `drift` | `gflowrl/drift_term_abs_mean` | flow gap 的**漂移半边** \|mean_group(d) − d_i\| 的均值,与 β 无关 |
| `rew/drift` | 二者之比 | **奖励信号相对漂移信号的强弱。< 1 = 奖励被漂移淹没** |
| `sign_ok` | `gflowrl/sign_agree_nondegenerate` | 在**非退化组**上,clip 之后的 g 与奖励项同号的比例。→ 0.5 说明漂移项翻转了奖励的方向 |
| `d_seq` | `gflowrl/d_seq_abs_mean` | \|d\| = \|Σ(log π_ref − log π_old)\| 的均值,即策略离参考策略多远。**它随训练单调上升是正常的** |
| `grad_norm` | `actor/grad_norm` | 裁剪**前**的梯度范数 |
| `score` | `critic/score/mean` | 该步 rollout 的平均奖励(0–1 之间的任务奖励之和) |
| `mem_gb` | `perf/max_memory_allocated_gb` | 单卡 torch allocator 峰值 |

### 6.2 β 分解 —— 读这份 CSV 之前必须知道的

`z_t` 是组内的**算术**均值,所以 Eq.6 可以精确拆成两个可加的半边,且 g 关于 β 线性:

```
g_i = [mean_group(β·r) − β·r_i]  +  [mean_group(d) − d_i]
       └── 奖励项,正比于 β ──┘     └── 漂移项,与 β 无关 ──┘
```

两个直接后果,写结论时必须带上:

1. **在奖励退化的组上,奖励项恒为 0,g 全部来自漂移项。** 那时候 C′ 做的是向
   `π_ref·exp(βr)/Z` 的 flow matching,**不是在学奖励**。本次实测退化组中位数
   **70.3%**(范围 57.8–85.9%)—— 也就是说每个 batch 的大多数组都是这种情况。
   不能把它说成"GRPO 给 0 梯度而 C′ 还在学奖励"。
2. **调小 β 治不了 clip 饱和。** 它只缩小奖励那一半,漂移项纹丝不动,于是幸存下来的
   梯度反而**更**被漂移主导。`sat_b2/b4/b16` 三列就是每步实算的反证:
   本次 `sat_b2` 中位 0.478、`sat_b4` 0.500、`sat`(β=8)0.516、`sat_b16` 0.525 ——
   β 差 8 倍,饱和率只动了 5 个百分点。**β 要按 `rew/drift` 选,不能按 `sat` 选。**

### 6.3 本次 85 步的实际读数

```
                 最小      中位      最大
degen           0.578    0.703    0.859
sat             0.303    0.516    0.613
sat_b2          0.278    0.478    0.562
sat_b4          0.291    0.500    0.597
sat_b16         0.325    0.525    0.622
reward          0.290    0.672    1.695
drift           0.000    0.339    0.441
rew/drift       0.858    2.014    4.941
sign_ok         0.689    1.000*   1.000     (*中位 0.837)
d_seq           0.000    0.374    0.478
grad_norm     103.075  180.312 5229.205
score           0.581    0.806    1.083
mem_gb         26.478   34.618   35.339
wall_s            862      930      981
```

怎么判读:

- **`sat` 稳定在 0.5 附近,没有向 1.0 漂** —— 幅度信息还在,clip 没有把梯度退化成纯符号。
- **`rew/drift` 中位 2.01,85 步里只有 step 76 那一步跌破 1(0.86)** —— 奖励信号整体
  压得住漂移。趋势上前 10 步中位 2.125 → 后 10 步 1.754,在缓慢下降但幅度不大。
- **`d_seq` 从 0 涨到 ~0.41 后趋平** —— 第 1 步策略等于参考策略所以是 0,之后正常拉开。
- **`grad_norm` 中位 180,而 `grad_clip=1.0`。** 这意味着**每一次更新都被裁到只剩方向、
  没有幅度**。step 74 那次 5229 是全程唯一超过 1000 的。这条必须写进结论:
  在这个配置下,学习率和梯度大小之间的关系被 clip 完全接管了。
- **`score` 没有可靠的上升趋势。** 前 10 步中位 0.766、后 10 步 0.820,但前半 0.809、
  后半 0.796 —— 这个量级的来回就是噪声。85 步、又几乎每步纯方向更新,**训练奖励
  在这次运行里没有明显改善**。eval 上大概率也不会看到大幅提升,心里要有数。
- **`mem_gb` 26.5 → 35.3 平稳**,离 48 GB 有余量,没有 OOM 风险。

---

## 7. 值得记录的问题(硬件 / 上游代码导致的)

### 7.1 【硬件】容器里的 PCIe P2P 是坏的 —— 最严重的一个

**症状:** trainer 起来之后第一个 NCCL collective 永远不完成,10 分钟后 watchdog 报

```
Last enqueued NCCL work: 1, last completed NCCL work: -1
```

而 `nvidia-smi topo -p2p r` 显示全是 OK。

**定位:** 没有靠反复起 15 分钟的训练去试,而是写了一个 4 卡 all-reduce 的最小探针
(`scripts/ncclprobe.py` + `scripts/nprobe.sh`),90 秒一轮跑了 5 个变体:

```
A  默认                          -> 40 s 超时
B  NCCL_IB_DISABLE=1             -> 40 s 超时   (所以不是 IB/RoCE 的事)
C  B + NCCL_SOCKET_IFNAME=eth0   -> 40 s 超时
D  B + NCCL_P2P_DISABLE=1        -> 7 秒 ALLREDUCE_OK   ← 就是它
E  B + NCCL_SHM_DISABLE=1        -> 40 s 超时   (所以 SHM 正是能用的那条路)
```

**根因:** 容器里 ACS / IOMMU 没关的典型表现 —— P2P 在拓扑上"可用",实际传输挂死。
这是**平台的问题,不是代码的问题**,换一台机器可能就没有。

**代价:** 关掉 P2P 后 NCCL 走共享内存(过主机内存中转),比 P2P 慢。本次的每步 925 s
是在这个前提下测的,**不能直接拿去和别的机器比吞吐**。

**教训:** 租到新机器后,跑训练之前先花 90 秒跑一次 all-reduce 探针。这次如果不做,
就是每次改一个环境变量等 15 分钟,一天就没了。

### 7.2 【上游代码】单节点上把同一批卡预订两次 → 死锁且不报错

见 §4.2 的 `574ae81`。论文用 2 节点,这条路径永远不会暴露;**任何单节点复现都会撞上**。
失败模式是"卡住不动",没有任何错误信息 —— 这是最难查的那一类。

### 7.3 【上游代码】`cleanup()` 在 `set -e` 下自己把自己中止,导致 Ray 集群残留

见 §4.2 的 `4474e67`。表面症状只是"成功的 run 报失败"(看起来只是难看),
**真正的危害是 `ray stop --force` 从来没有执行过** —— 每次跑完都留下一个活着的 Ray 集群,
下一次启动会撞上它。我们是在一次成功的冒烟结束 2 小时 44 分之后,发现集群还活着才意识到的。

推论:凡是"退出码不对但看起来无害"的信号,都要查到根因再决定要不要忽略。

### 7.4 【上游代码】raylet 的 cwd 决定了数据能不能找到

见 §4.2 的 `5fa00d8`。parquet 里的图片是相对路径 `images/xxx.png`,靠进程 cwd 解析;
而 `ray start` 起 raylet 时的 cwd 是调用者的 cwd,`cd "$VERL_DIR"` 在它之后。
后果:**加载 20 分钟之后**才报 `FileNotFoundError`。这种"很晚才失败"的 bug 特别贵。

### 7.5 【上游代码】结构化配置对未知 key 是报错而不是忽略,而且在 Ray worker 里才炸

见 §4.2 的 `dfbde8e`。附带的那个"半边"更阴险:两个配置读取源(driver 读裸 DictConfig、
`core_algos` 读 dataclass)对同一个损失的参数集合理解不一致,`eps_is` 被静默丢弃。
**同一份配置有两个读取路径时,必须有一个检查断言两边的 key 集合一致。**

### 7.6 【上游代码】环境变量从 eval 的 shell 泄漏到训练

`VERL_DUMP_TOKEN_DIAGNOSTICS` 是为 eval 手工设的,它会打开 `ray_trainer._validate()` 里的
token 诊断;而 `test_freq` 会让 validation 在训练期间也跑,于是训练里会多出
"把 rollout 引擎睡眠 + 一整趟 `compute_log_prob`"的开销。同一个 shell 先跑 eval 再跑训练
就会中招。已在 `574ae81` 里显式 unset。

### 7.7 【工具侧】运行时错误很多,但都被正确吞掉了

整个训练日志(9.8 MB)里的统计:

```
No collision-free grasps found          3389 次
  其中作为 RuntimeError 抛出            2346 次
Top-down filtering removed all grasps    130 次
selected index k out of range             24 次
CUDA error: invalid configuration argument 12 次  (depth_estimator.get_depth_map)
Mask removed all points                   10 次
torch.cuda.OutOfMemoryError                2 次  (工具侧,见下)
ERROR:toolshed.tools.sam2                  3 次
达到 8 轮上限                              2 次
截断的工具响应                             0 次
训练侧 OOM / NCCL WARN / 崩溃              0
```

**这些不是训练故障。** Toolshed 把工具异常转成文本返回给模型,模型可以换个工具或换个
参数重试 —— 这正是 tool-augmented RL 要学的东西。`grasp_generator` 那 3389 次尤其正常:
它是在真实点云上做碰撞检测,"这个物体周围没有无碰撞抓取位姿"是**几何事实**,不是 bug。

**但有两条值得单独记:**

1. 两次工具侧 OOM,一次试图分配 **14.36 GiB**,另一次试图分配 **1481.25 GiB** ——
   都发生在 `torch.cdist` 里。1481 GiB 显然不是显存不够的问题,而是**输入退化**
   (点云点数异常膨胀)导致 cdist 的中间矩阵爆炸。这是工具的鲁棒性问题,
   在更大规模的训练里可能会更频繁。
2. 12 次 `CUDA error: invalid configuration argument`,全部在 `depth_estimator.get_depth_map`,
   典型是零尺寸输入导致的 kernel 启动参数非法。

两者都被吞掉了,不影响本次结果;但如果以后要统计"工具成功率",这些必须计入分母。

### 7.8 【平台】HF 的 5000 resolve / 5 分钟配额

下训练数据(5425 个文件)时撞上 429,下载速度掉到 0.5 MB/s。不是代码问题,
是 Hub 的限速。解决办法是分批 + 落到容器盘再整体拷贝,以及**下好之后别再让脚本去重下**
(`run_rl.sh` 有 `os.path.isfile` 检查,把数据软链到它期望的位置就会直接命中)。

### 7.9 【上游默认值】`grad_clip=1.0` 与 `grad_norm` 差两个数量级

不是 bug,是配置事实,但影响对结果的解释:`grad_norm` 中位 180、最大 5229,而裁剪阈值是 1.0。
**每一次更新都只保留方向。** 这在报告里必须说明,否则"学习率 1e-6"这个数字会被误读。

---

## 8. 其他值得记的

### 8.1 checkpoint 的构成与磁盘

verl 的 FSDP checkpoint 每步 **44 GB**(交接文档估的 34 GB 是错的,实测才知道):

```
model_world_size_4_rank_*.pt     4 × 4.07 GB   ← merge 成 HF 格式只需要这些
optim_world_size_4_rank_*.pt     4 × ~7 GB     ← 只有续训要,eval 用不到
extra_state_* / fsdp_config.json 很小
huggingface/                     config + tokenizer
```

`verl/model_merger/fsdp_model_merger.py` 只读 `model_world_size_*_rank_*.pt`
(L91、L149 两处),所以**权重快照 15 GB 就够 merge + eval**,不必留 optim。
这是清道夫能在 250 GB 卷上活下来的关键。

### 8.2 merge 与上传

```
merge:  python3 -m verl.model_merger merge --backend fsdp \
            --local_dir <ckpt>/actor --target_dir <out>
        每个约 40 秒,产出 7.6 GB 的 HF 格式(2 个 safetensors 分片)
上传:   三个 ckpt + provenance,共 54 个文件 / 22.8 GiB
校验:   HF 记录的 sha256 与本地逐字节比对,10 个大文件全部一致,0 个不符
```

### 8.3 这次训练**没有**做的事

- 没有跑 GRPO 对照臂(用户明确要求先只跑单臂)
- 没有在训练中跑 validation(`test_freq=-1`、`val_before_train=False`)—— 省时间,
  而且 eval 要在同一台机器上单独做才有可比性
- 没有多 seed(单 seed)
- 没有 early stopping / lr schedule 调整

### 8.4 eval 的口径(还没跑)

主结果对照 SFT 起点,以及 P4 复现的官方 ckpt 与论文 Table 2:

| | SFT 起点 | 官方 ckpt(RL 后) | 论文 |
|---|---|---|---|
| RoboSpatial Overall | 61.00 ± 0.77 | 65.43–66.00 | 70.00 |
| RefSpatial 三项 | 52.58 简单平均 | 53.35 | 53.07 |

两个必须注意的点:

1. **SFT 基线是在 4× A6000 上、`gpu_memory_utilization=0.5`(= 24 GB KV 池)下测的。**
   这个旋钮是**整卡比例**不是绝对值,换卡不调它池子就变了;池子大小 → batch 组成 →
   浮点归约顺序 → 接近平局的样本翻转。`eval/EVAL_FROM_SCRATCH.sh` 会按卡容量反算,
   把池子压回 24 GB。
2. RefSpatial 上 SFT 已经和 RL 后的官方 ckpt 平手(277 题差 2 题),**那一项本来就没有
   可探索的空间**。要看的是 RoboSpatial。

### 8.5 所有产物在哪

```
HF 私有 repo   qzpm55555/spacetools-p7-gflowrl-cprime-8xa40
  global_step_85/ 60/ 30/        三个 HF 格式 ckpt,各 7.6 GB
  provenance/                    p7_metrics.csv · as-run 脚本 · 完整日志 · commit SHA
  bundle/p7_bundle.tar.gz        本目录的完整版(含 9.8 MB 训练日志全文)
  EVAL_FROM_SCRATCH.sh           裸机 → 九个 benchmark,一条命令
  parse_dump.py                  eval 的健康门禁
本地(本目录)                    精简版,不含完整训练日志 —— 那份在 HF 的 bundle 里
```

---

## 9. 下一步

1. 租 **4 张 A6000**(和 SFT 基线同卡),跑 `EVAL_FROM_SCRATCH.sh`,`P7_STEP=85`,
   九个 benchmark,约 2 小时 10 分。
2. 主结果只报 step 85。若明显不佳,再评 60 / 30,并在报告里写明一共评了几个 checkpoint、
   哪些是事后挑的。
3. 若 P7 与 61.00 的差距在 1–2 个点以内,补跑 SFT 那一臂(同机同 session)做配对检验,
   否则"你换机器了吧"这个质疑无法反驳。
4. 结论措辞:在 ~70% 的退化组上,C′ 的梯度是向 `π_ref·exp(βr)/Z` 的 flow matching,
   **不是**奖励学习;且每步更新都被 `grad_clip=1.0` 裁成纯方向。
