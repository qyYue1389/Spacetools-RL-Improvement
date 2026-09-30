# GFlowRL —— Step 4 训练用到的全部东西

> ⚠ **`code/` 里是快照,不是正本。** 正本在 `../../SpaceTools-RL/`,
> 训练跑的是正本。**在这里改文件不会影响任何一次训练。**
> 快照过没过期:`bash verify_copies.sh`

本目录回答一个问题:**要把 GFlowRL(C′ 臂)跑起来,涉及哪些文件、它们干什么、
怎么确认没被改坏。**

```
README.md            ← 本文件
MANIFEST.txt         code/ 每个文件的 sha256 与它在正本里的路径
verify_copies.sh     快照 vs 正本,逐文件 OK / DRIFT
run_checks.sh        跑五个自检(跑的是 spacetools-repro 里的正本脚本)
code/                实现与启动脚本的快照
diff/                三份补丁:两个 feat commit + 当前未提交的修复
docs/                为什么是 C′、闸门、不动点自检、作业书
```

---

## 1. 实现:两半

| 文件 | 位置 | 干什么 |
|---|---|---|
| `verl/trainer/ppo/ray_trainer.py` | `compute_gflowrl_flow_gap()` L221 | **不带梯度的一半。** Eq.4 组内 log Z → Eq.6 flow gap → Eq.7 非对称 clip,结果广播进 `batch["advantages"]` |
| | `fit()` 里 L1824 | 调用点 + 两个「静默错配」的守卫(`use_kl_loss=False` / `kl_loss_coef≠0` 都 raise) |
| `verl/trainer/ppo/core_algos.py` | `compute_policy_loss_gflowrl()` L2237 | **带梯度的一半。** Eq.8,注册名 `"gflowrl"` |
| `verl/workers/config/actor.py` | `PolicyLossConfig.gflowrl` | 让 hydra 能设 `policy_loss.gflowrl.*`。**没有这个字段,启动就崩** |
| `verl/trainer/config/actor/actor.yaml` | `policy_loss.gflowrl: {}` | 同上,yaml 侧声明 |

```
d_i   = masked_sum(log π_ref − log π_old)          整条求和,不除 |y|
Z_t   = mean over uid group (β·r_i + d_i)          Eq.4
g_i   = Z_t − d_i − β·r_i                          Eq.6
g̃_i   = clip(g_i, −ε_low, +ε_high)                 Eq.7
L     = mean_i  w_i · (g̃_i + masked_sum(log π_θ − log π_old))²      Eq.8
w_i   = clamp(exp(整条 log ratio).detach(), max = 1 + ε_is)
```

### 一个配置相关的事实:`w` 在当前配置下恒为 1

`train_batch_size=64` × `rollout.n=5` = 320 条,而 `ppo_mini_batch_size=64` 经
`fsdp_workers.py:249` 乘 5 除卡数之后正好等于每卡的序列数 ⇒ **每个 rollout batch 只切出
一个 mini-batch**,加上 `ppo_epochs=1`,`dp_actor.py:550` 的 `on_policy` 成立,于是
`old_log_prob = log_prob.detach()`,`log_ratio` 数值恒为 0(梯度不为 0)。

    w = clamp(exp(0), max=1+eps_is) = 1      <- eps_is 不起作用
    loss 数值 = mean(g~^2)                    <- 梯度 = mean(2*g~_i * grad masked_sum(log pi_theta))

不是缺陷:一步优化本来就没有 `pi_theta/pi_old` 失配可纠。但**这是配置依赖的** ——
改 `ppo_mini_batch_size`(为了省显存最容易动的就是它)或 `ppo_epochs` 就会翻转。
`p7_onpolicy_check.py` 守这一条。

**为什么是这个拆法**:`g̃` 只依赖 `π_old`、`π_ref`、`r`,在 actor update 期间全部冻结,
所以每步在 driver 上算一次就够;而 driver 上 `old_log_probs` / `ref_log_prob` /
`token_level_scores` / `uid` 都是现成的。**因此 actor 端不需要原始 `r`,
`dp_actor.py` 一行没改。**

### variant

| variant | Eq.4 的 d | Eq.5/6 的 d | 用途 |
|---|---|---|---|
| **`cprime`** | **`d`** | **`d`** | **训练臂,默认** |
| `paper` | `d` | `d/L` | 论文原样,诊断臂。不动点 loss 4.0e-2,clip 100% 饱和 |
| `normalized` | `d/L` | `d/L` | 两边都归一,诊断臂。5.1e-2 / 84% |

---

## 2. 启动

```bash
# GRPO 臂 —— 原样,零代码改动(但要按下面加 kl_loss_coef=0)
bash examples/toolshed/run_rl.sh actor_rollout_ref.actor.kl_loss_coef=0

# C′ 臂
bash examples/toolshed/run_rl_gflowrl.sh
```

`run_rl_gflowrl.sh` 是**薄封装,不是 `run_rl.sh` 的副本** —— 两臂要可比,复制 400 行
迟早会漂。全部 delta 是七行 hydra override,经 `run_rl.sh` 结尾的 `"$@"` 透传。

可调环境变量:`GF_BETA`(默认 8.0)· `GF_EPS_LOW` 0.2 · `GF_EPS_HIGH` 0.28 ·
`GF_EPS_IS` 0.2 · `GF_VARIANT` cprime。

单机还要设 GPU 拆分(见 §5):`TOOL_GPUS` / `TRAIN_GPUS`。

---

## 3. 跑之前跑一遍自检

```bash
bash run_checks.sh
```

| 脚本 | 拦什么 |
|---|---|
| `p7_config_check` | `policy_loss.gflowrl.*` 这条配置路径通不通;wrapper 传的 key 与代码读的 key 对不对得上;**两臂的 delta 是否只有目标函数**(override 越界、rollout correction 被单边打开,都在这里拦) |
| `p7_guard_check` | 两个静默错配的守卫会不会真的 raise |
| `p7_gpusplit_check` | 单机 GPU 拆分;两节点行为必须不变 |
| `p7_fixedpoint_check` | **改过那两个函数之后必测。** cprime 在 Prop. B.1 不动点上 loss 必须 ≈ 0 |
| `p7_degenerate_check` | C′ 在奖励简并组上给不给非零梯度 |
| `p7_onpolicy_check` | batch 配置是否仍让 `on_policy` 成立。成立则 Eq.8 的 `w` 恒为 1、`eps_is` 不起作用;不成立则两者都变成真问题 |

这些脚本用 `ast` / 正则**从真实源文件里抠代码再执行**,测的是仓库里那一份。
所以 `run_checks.sh` 只是 cd 过去跑正本,这里不放副本。

---

## 4. 三个最容易被改坏的地方

1. **`masked_sum` 不能换成 `masked_mean`。** 那是 C′ 与 `normalized` 的**全部**差别,
   换了不报错,只是悄悄训练另一个目标。→ `p7_fixedpoint_check` 会从 7.4e-13 跳到 5e-2。
2. **`w` 必须 `.detach()`。** 它是纠正采样分布的重要性权重,不是被求导的目标。
3. **`use_kl_loss` 必须保持 True,`kl_loss_coef` 必须为 0。**
   `dp_actor.py:528` 只在前者打开时才把 `ref_log_prob` 放进 batch;而 `d` 本身就是
   Eq.4/6 里的 KL 型量,再加显式 KL 是重复计。**开关决定算不算,系数决定罚不罚。**
   两个都有守卫兜底。

---

## 5. 训练时从第 1 步就要盯的指标

```
gflowrl/clip_saturation        离线测到 77.9%。长期贴近 1.0 ⇒ g̃ 被顶死,
                               奖励的大小信息丢光只剩符号,那就不是 GFlowRL 了
gflowrl/is_weight_min          当前配置下这两个应当恒为 1.0 / 0.0,
gflowrl/is_weight_collapsed_frac   log_ratio_abs_mean 恒为 0 —— 因为一个 rollout batch
                               只走一步优化,on_policy 成立,old_log_prob = log_prob.detach(),
                               所以 w = exp(0) = 1,eps_is 完全不起作用(见 §5.1)。
                               ⚠ 它们不是诊断,是**断言**:一旦动起来,说明有人改了
                               ppo_mini_batch_size 或 ppo_epochs,on_policy 翻转了,
                               而那时 w 变成 exp(上千 token 的 logprob 差之和)、只 clip 上界,
                               每 token 漂移 −3e-3 就能把某条序列的 w 压到 ~0.02 ——
                               序列还在 batch 里但什么都不贡献,有效 batch 静默缩水
gflowrl/Z_t_mean               Var = O(1/G);G=5 比论文 G=16 噪 3.2×
reward/degenerate_group_frac   组内 r 全同的组占比。**两臂都记**,调用点在 gflowrl
reward/degenerate_rollout_frac 分支之前 —— 只有一臂有这个数的话它什么都证明不了。
reward/group_range_mean        P7 此前引用的 56.9% / 75.8% 是 eval 分布上的估计,
                               训练分布上的真实值只能从这里读
gflowrl/variant_is_cprime      必须恒为 1.0
[GFlowRL] WARNING              response_mask 像是 attention-mask fallback。
                               只 warn 不 raise,且完全不调工具的 rollout 会误报 ——
                               别无视,去对一眼 num_turns 分布
```

---

## 6. GPU 拆分(单机必看)

原 `run_rl.sh` 把**同一个 `GPUS_PER_NODE`** 同时给了 Toolshed 的 placement group
和 `trainer.n_gpus_per_node`。两节点没问题(PG 落节点 1、训练在节点 2);
**单机时同一批卡被要了两次**,PG 抓走全部,训练侧一张拿不到。

现在拆成 `TOOL_GPUS` / `TRAIN_GPUS`,**两节点行为逐字不变**,单机默认 2 / 其余。

⚠ **压 `num_actors` 时必须同步抬 `TOOL_GPUS`。** Ray 的 `num_gpus` 是**逻辑预留**,
和显存无关:原配置七个工具合计 **7.8 个逻辑 GPU**,压到 13 个 actor 是 **3.0**。
PG 只预留 2 而 actor 要 3.0,PG 会 ready 而 actor 永远排不进去 —— heredoc 里的
assert 就是拦这个的。

---

## 7. diff/

| 文件 | 内容 |
|---|---|
| `01-gflowrl-policy-loss.patch` | commit `33036a51`,两半实现 + variant 开关 |
| `02-wrapper-and-guard.patch` | commit `0b5098ae`,wrapper + 两个守卫 |
| `03-uncommitted-fixes.patch` | **尚未 commit**:`STRICT_PACK` 拆分、`PolicyLossConfig.gflowrl` 字段、`actor/ppo_kl` 口径、两个 IS 权重指标 |
| `HEAD.txt` | 生成时的 HEAD 与工作树状态 |

---

## 8. docs/

| 文件 | 读它干什么 |
|---|---|
| `P7_STEP_C12_RESULTS.md` | **实现文档。** 自检怎么做的、结果、四条未排除的风险 |
| `P7_ROUTE_C_GATE.md` | 为什么选 C′、β 的下界推导、饱和率 77.9% 这个必须一起报的代价 |
| `P7_ROUTE_C_PLAN.md` | 作业书:全部超参、统计设计与预登记。**§2「实现:改哪里」已被实际实现超越,别照它动手** |
| `P7_STEP23_RESULTS.md` | 奖励简并率、长度分布、clip 饱和的离线测量 |

**一条不要读错的**:`P7_STEP_C12_RESULTS.md` §3 测到 C′ 在奖励简并组上有非零梯度
(mean|g̃| 0.167 vs 非退化 0.247)。那个信号**全部来自 drift 项,一点不来自 reward** ——
它在做分布匹配,不在做排序。**这是机制,不是预期收益。**
