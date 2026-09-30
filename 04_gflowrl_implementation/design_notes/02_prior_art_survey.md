# P7 可借鉴的现成实现(2026-09-01 检索)

> 目的:P7 走 A′ 路线,需要一个**外部参照**来把「我们的 loss 写得对不对」与
> 「算法在多轮工具轨迹上行不行」分开。本文只记检索结论与可借鉴的具体位置。
>
> ⚠ **检索不是不存在性证明。** 下面是若干次 web 检索 + 直接访问仓库的结果,
> 用的是英文关键词。「没找到」应读作「没找到」,不是「没有」。

---

## 0. 结论一句话

**GFlowRL 本身仍然没有公开实现,但它的直接前作 FlowRL 有,而且建在 verl 上、
license 是 Apache-2.0 —— 那正是我们要的 90% 管道。**
更巧的是:FlowRL 需要改三个文件,其中两个**只是为了安置 `Z_φ` 那个网络**,
而 GFlowRL 删掉的恰好就是它。**我们的改动比 FlowRL 更小。**

| 仓库 | 是什么 | 能借什么 | license / 规模 |
|---|---|---|---|
| `microsoft/gflowrl` | 论文声明的官方实现 | **仍然 404**(2026-09-01 再确认) | — |
| **`Xuekai-Zhu/FlowRL`** | GFlowRL 的直接前作,也是它的 baseline | **verl 集成、TB loss、长度归一化与 mask 的写法** | **Apache-2.0** · 180★ · 101 commits |
| `bbartoldson/TBA` | Trajectory Balance with Asynchrony | **TB-VarGrad**:Eq. 4 估计量的同族 | Apache-2.0 · 32★ · 自建框架 |
| `Yu-Fangxu/FoR` | Flow of Reasoning(ICML 2025) | 单一共享标量 logZ,即论文的 ConstantLogZ 对照 | ICML 2025 |
| f-TB / DevGrad(arXiv 2605.15417) | f-散度族 TB,ICML 2026 | **把批内常数当作要优化的量**,而非取均值 | **无代码** |
| 我们自己的 `SpaceTools-RL` | verl 0.8.0.dev fork | **`register_policy_loss` 与 `ref_log_prob` 已经就位** | 已在本机 |

**没有找到**:任何 SpaceTools 的第三方复现;任何 GFlowNet 式目标用在**多轮工具轨迹**上的实现。
后者与论文 Appendix A 自己那句「unclear whether ... extends to broader agentic or
multimodal RL settings」一致 —— **A′ 的空位仍然是空的。**

---

## 1. FlowRL:最值得抄的一个

`https://github.com/Xuekai-Zhu/FlowRL` · Apache-2.0 · 建在 **verl** 上
(README 给两条路:verl 0.4.0 复现论文,或接最新 verl)。
仓库带一份 **`FLOWRL_SIMPLE_GUIDE.md`**,直接给出要改的文件与代码。

### 1.1 它改哪三个文件,以及为什么我们只需要改一个

    verl/workers/fsdp_workers.py            <- 安置 ProjZModule(3 层 MLP 的 Z_φ)
    verl/workers/sharding_manager/fsdp_vllm.py  <- 让 Z_φ 参与 FSDP/vLLM 的分片
    verl/workers/actor/dp_actor.py          <- compute_flowrl_objective,真正的 loss

> **前两个文件的存在理由就是 `Z_φ`。** GFlowRL 的全部贡献是把 `Z_φ` 换成 Eq. 4 的批内
> Monte Carlo 估计,「the auxiliary network and its optimizer state in distributed
> training are all removed」。**所以我们只需要 `dp_actor.py` 那一处。**

### 1.2 一处**外部确认**:`|y|` 就是 `response_mask` 的和

`P7_DECISION.md` §5.1 第 2 条是我们自己从 Remark B.4 推出来的:
`|y_i|` 必须是策略生成的 token 数,不能是原始 response 长度。
FlowRL 的实现独立地这么写:

    avg_logpf    = verl_F.masked_mean(logpf,  response_mask, axis=1)     # 长度归一化 = 按 mask 计数
    avg_logp_ref = verl_F.masked_mean(logf_ref, response_mask, axis=1)
    ...            verl_F.masked_sum (logpf - logpf_old, response_mask, axis=1)   # 未归一化的那一支

**`masked_mean` 的分母就是 mask 里 1 的个数。** 这是对我们那条推论的一次独立佐证,
而且顺带说明:归一化(`masked_mean`)与未归一化(`masked_sum`)两支在同一份实现里并存 ——
与 Eq. 4 无归一化、Eq. 5/6 有归一化的那个不对称对得上(见 `P7_STEP23_RESULTS.md` §2.5 的量纲失配)。

---

## 2. TBA:Eq. 4 估计量的同族参照

`https://github.com/bbartoldson/TBA` · Apache-2.0 · arXiv:2503.18929

它用的是 **TB-VarGrad**。GFlowRL 论文原文:估计量「builds on the log-partition
variance loss of Zhang et al. [2023]」,而「**Rather than minimize the variance** of these
targets across trajectories sharing the same condition, **we take their in-batch mean**
as a stop-gradient baseline」。

> **所以 TBA 与 GFlowRL 是同一个量的两种用法:VarGrad 最小化它的方差,GFlowRL 取它的均值。**
> TBA 是这条支路上唯一能跑的公开实现,值得读它怎么算 `Ẑ(x)`、怎么配 replay buffer。
> 缺点:自建框架(不是 verl),规模小(32★),要移植。

---

## 3. f-TB / DevGrad:对判据 (i) 最相关的一篇(但没有代码)

arXiv:2605.15417(ICML 2026)。它把 TB 推广到整个 f-散度族,并提出 **DevGrad**:
不再假定配分函数已知,而是**解一个最优的批内常数**

    min_C  (1/B) Σ_i  L_f( Δ(y_i) + C )        使得梯度系数在批内求和为零

> **这正是我们的 `Z_t`,只不过用「优化」代替了「取算术平均」。**
> 判据 (i) 问的是「`G=5` 下批内均值这个估计量还能不能用」——
> DevGrad 给出的是同一个常数的另一种估计法,**是这条判据天然的对照组**。
> 论文未给代码;但公式简单到可以直接实现在我们的合成检验里
> (`tools/p7/p7_fixedpoint.py` 已经有现成的框架)。

---

## 4. 我们自己的 verl fork 已经准备好了

`SpaceTools-RL` @ `f0742338` · **verl 0.8.0.dev**。实地查过:

    verl/trainer/ppo/core_algos.py
      POLICY_LOSS_REGISTRY / register_policy_loss / get_policy_loss_fn   已存在
      已注册 11 个:vanilla · dppo_tv · dppo_kl · gspo · sapo · gpg ·
                    clip_cov · kl_cov · geo_mean · cispo · bypass_mode

    注册接口的签名(不含我们要的两样东西):
      policy_loss_fn(old_log_prob, log_prob, advantages, response_mask,
                     loss_agg_mode, config, rollout_is_weights)

    verl/workers/actor/dp_actor.py
      L516  select_keys = [responses, response_mask, input_ids, attention_mask,
                           position_ids, old_log_probs, advantages]
      L528  if self.config.use_kl_loss:  select_keys.append("ref_log_prob")   <- 已就位
      L615  policy_loss_fn(...) 的调用点
      L648  ref_log_prob = model_inputs["ref_log_prob"]                        <- 同一作用域

**因此最小改动路径(四步,约 10 行):**

1. `actor.use_kl_loss=True` 且 `kl_loss_coef=0` —— **`ref_log_prob` 就被选进 micro-batch,
   而 KL 罚项乘 0 不生效**。GFlowRL 本来就需要 ref 前向,顺手拿到。
2. `select_keys.append("token_level_scores")` —— 一行。`advantages` 是组内归一化过的,
   GFlowRL 要的是**原始** `r`;`token_level_scores` 在 `ray_trainer.py:1528` 写入 batch。
3. `@register_policy_loss("gflowrl")` 写在 `core_algos.py`。
4. 在 L615 的调用点把 `ref_log_prob` 与 `token_level_scores` 传进去(签名要扩,或用闭包)。

> **注意 `bypass_mode` 这个已注册的 loss**(`core_algos.py:2236`):它把
> `old_log_prob = rollout_log_prob`,并显式处理 IS 权重与 rejection mask。
> GFlowRL 的 `w_i = min(π_θ/π_old, 1+ε)` 与它同类,**其实现值得先读一遍再自己写**。

---

## 5. 对 P7 的净影响

- **A′ 的空位得到外部确认**:没有找到任何 GFlowNet 式目标用在多轮工具轨迹上的实现。
- **「loss 写得对不对」现在有了两道保险**:合成不动点检验(已完成)+ FlowRL 的
  `compute_flowrl_objective` 逐行对照(TB 部分与 GFlowRL 只差 `Z` 的来源)。
- **`|y|` 那条推论被独立佐证**(§1.2),可以从「待复核」降为「已确认」。
- **工程量下修**:改动比 FlowRL 小(不需要 `fsdp_workers.py` 与 `fsdp_vllm.py`),
  registry 与 `ref_log_prob` 都已在我们的 fork 里。
- **判据 (i) 多了一个对照组**:DevGrad(f-TB)。

## 6. 链接

- FlowRL <https://github.com/Xuekai-Zhu/FlowRL>
- TBA <https://github.com/bbartoldson/TBA> · <https://tba-llm.github.io/>
- FoR <https://github.com/Yu-Fangxu/FoR>
- f-TB <https://arxiv.org/abs/2605.15417>
- verl core_algos <https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/core_algos.py>
- GFlowRL 论文 <https://arxiv.org/abs/2607.13394> · 官方代码 `microsoft/gflowrl` **仍 404**
