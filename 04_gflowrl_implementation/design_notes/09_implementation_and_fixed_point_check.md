# P7 路线 C — 步骤 1&2 结果：GFlowRL 实现与不动点自检

状态：**通过**。实现落地，不动点精确为 0，梯度方向正确。
日期：2026-09-03
代码：`SpaceTools-RL` @ f945bb6b + 2 处未提交改动
脚本：`tools/p7/p7_fixedpoint_check.py`、`tools/p7/p7_degenerate_check.py`

---

## 1. 实现落在哪两处

按 P7_ROUTE_C_PLAN 的设计，loss 拆成两半：**不带梯度的那半在 driver 上算一次，带梯度的那半在 actor 里算。**

### 1.1 `verl/trainer/ppo/ray_trainer.py` — `compute_gflowrl_flow_gap()`

每个训练 step 在 driver 上执行一次，产出 Eq. 4 → Eq. 6 → Eq. 7：

- `Z_t` = 组内 `mean(β·r_i + d_i)`（Eq. 4 的 in-batch MC 估计，`uid` 分组）
- `g_i = Z_t − d_i − β·r_i`（Eq. 6）
- `g̃_i = clip(g_i, −ε_low, +ε_high)`（Eq. 7，非对称）
- 写进 `batch["advantages"]`，按 `response_mask` 广播到 token 维

`variant` 三选一，是这次复现的核心开关：

| variant | Eq. 4 里的 d | Eq. 5/6 里的 d | 说明 |
|---|---|---|---|
| `paper` | `d`（不归一） | `d/L` | 论文原样 |
| `normalized` | `d/L` | `d/L` | 两边都归一 |
| **`cprime`** | **`d`** | **`d`** | **两边都不归一 ← 我们训练用这个** |

调用点接在 `fit()` 里现有 `compute_advantage(...)` 之后，只在 `policy_loss.loss_mode == "gflowrl"` 时触发；不开这个开关，代码路径和原来一字不差。

### 1.2 `verl/trainer/ppo/core_algos.py` — `compute_policy_loss_gflowrl()`

用 `@register_policy_loss("gflowrl")` 注册，Eq. 8 的带梯度部分：

```
L = mean_i  w_i · ( g̃_i + [log π_θ(y_i|x) − log π_old(y_i|x)] )²
w_i = min( π_θ/π_old , 1+ε_is )        # stop-gradient
```

两个实现要点：

- **`masked_sum`，不是 `masked_mean`。** 这是 C′ 和论文写法的**全部**差别。这里若用 `masked_mean` 会悄悄变成 `normalized` 配置，且**不会报错**——这是本实现最容易被改坏的一行，改动前请先看 §2 的不动点检查。
- `w_i` 带 `.detach()`。它是纠正采样分布的重要性权重，不是被求导的目标的一部分。

### 1.3 一个防呆护栏

`compute_gflowrl_flow_gap` 里加了对 `response_mask` 的检查：如果全 batch 里「policy token 数 < 非 pad token 数」的样本占比 < 50%，打印 WARNING。

针对的是 `ray_trainer.py:157` 那个静默 fallback——`response_mask` 缺失时会退化成 attention mask，`|y_i|` 就会把**工具返回的 token 也算进去**，`d` 和 `L` 一起错，而且不报错。

**已知误报风险：** 完全不调用工具的 rollout 天然满足 `L == 非pad`。若某个 batch 里多数 rollout 不调工具，这条 WARNING 会误报。它只 warn 不 raise，所以最坏情况是噪声；但**看到它别直接无视**，去对一眼 `num_turns` 分布。

---

## 2. 不动点自检（步骤 2）

### 2.1 方法：测的是仓库里的代码，不是它的副本

脚本用 `ast.get_source_segment` 从两个**真实源文件**里抠出这两个函数的源码文本，再 `exec` 到打了桩的命名空间里（`masked_sum` / `masked_mean` / `DataProto` 是桩）。

这么绕一圈是因为：**把 loss 重打一遍再测它，测的是重打的那份对不对，不是仓库里那份对不对。** 后者才是要上机器跑的。

### 2.2 构造

Prop. B.1 说 Eq. 8 的不动点是 `π_θ(y|x) = π_ref(y|x)·exp(β·r(x,y))/Z(x)`。在那一点，对每条采样轨迹

```
log π_θ(y_i) − log π_ref(y_i) = β·r_i − log Z(x)
```

所以直接**解析地**造出这个 π_θ：给定任意 `π_ref` 和任意 `r`，令 `log π_θ = log π_ref + (β·r − log Z)/|y_i|` 逐 token 摊开。用 Eq. 4 的 in-batch 常数当 `log Z`。

设置：`G=8`，4 个 prompt，序列长度 12–40 **随机不等长**（不等长是关键，等长会让三个 variant 全都通过，测不出东西），其中一组 reward 全同（退化组）。构造精度 `max|err| = 3.8e-6`。

### 2.3 结果

| variant | `max\|g̃_i\|` | clip 饱和 | loss | 不动点 |
|---|---|---|---|---|
| **`cprime`** | **2.4e-06** | **0.0000** | **7.4e-13** | **PASS** |
| `paper` | 2.0e-01（顶到 ε_low） | 1.0000 | 4.0e-02 | FAIL |
| `normalized` | 2.8e-01（顶到 ε_high） | 0.8438 | 5.1e-02 | FAIL |

**`cprime` 精确为 0**（7.4e-13 是 fp32 噪声），clip 完全不激活——和 Remark B.3 一致。

**另外两个的 FAIL 不是 bug，是 P7 步骤 2–3 那个结论第一次跑通真实代码路径的独立复现。** 之前是离线脚本算出来的，这次是从仓库源码里抠出来的函数算出来的，结论一致：

- `paper` 混用 `Z_t`（序列和量纲）和 `d/L`（每 token 量纲），两边尺度差一个 `L`，`g` 直接顶死 clip，100% 饱和。
- `normalized` 两边量纲一致，但 `g_i = mean_j(β·r_j + d_j/L_j) − (β·r_i + d_i/L_i)`，而不动点上恒定的是 `β·r + d` 而非 `β·r + d/L`——`d/L` 随 `L` 变。**只要长度不等，解析不动点就不存在。**

这正是当初否掉 `paper` / `normalized`、选 C′ 的理由。现在这个理由有了可执行的形式：**任何人改了这两个函数，跑一遍 `p7_fixedpoint_check.py`，loss 不再是 0 就说明改错了。**

### 2.4 梯度检查

把 θ 从不动点上扰动 0.01：

- loss = 1.64e-03 > 0 ✓
- ‖∂L/∂log π‖ = 7.46e-02 > 0 ✓
- **padding 位置上的梯度 = 0.000e+00**（精确）✓
- IS 权重截断比例 = 0.0000（小扰动下不该截断）✓

---

## 3. 附带确认：C′ 在 reward 退化组上不是零梯度

P6 测到 56.3–79.0% 的组 reward 全同。GRPO 的 advantage 是 `(r − mean)/std`，**这些组的梯度精确为 0**——超过一半的 rollout 预算被丢掉。

用同一套真实代码测（`p7_degenerate_check.py`，6 组，3 组退化，`β=8`，含 policy drift）：

| | mean \|g̃\| | GRPO advantage |
|---|---|---|
| 退化组（50% rollout） | **0.1671** | **0（丢弃）** |
| 非退化组 | 0.2467 | 0.81–0.91 |
| `\|g̃\| < 1e-6` 的 rollout 占比 | **0.0000** | 0.5000 |

C′ 在退化组上给出的梯度是非退化组的 **0.68 倍**——不是零，也不是噪声级。

**但这里必须说清楚它是什么、不是什么。**

退化组上 `r` 是常数，所以 `g_i = mean_j(d_j) − d_i`，**信号全部来自 drift 项 `d = log π_ref − log π_old`，一点不来自 reward。** 它做的事是把组内 G 条轨迹的 drift 拉向组均值，即**把 π_θ 往 `π_ref·exp(βr)/Z` 这个分布上拽**。

所以：它**不是**在这些组里分辨「哪个答案更好」——答案 reward 全同，没有好坏可分。它是分布匹配项，在做形状，不在做排序。

这是 GFlowRL 可能赢 GRPO 的一条**具体的、机制层面的**理由（用上了 GRPO 扔掉的一半预算），但它离「所以 accuracy 会涨」还隔着一步，那一步只能由训练跑出来。**不要把这条当成预期收益写进任何结论。**

---

## 4. 这一步没能排除的风险

按 §3.3 纪律，把上界和代价一并报出来：

1. **clip 饱和高。** 合成数据 72.9%，C′ gate 在真实 rollout 上测到 77.9%。`g̃` 顶死在 ±ε 时，loss 变成 `(±ε + log_ratio)²`——**reward 的大小信息基本丢光，活下来的只有符号。** 这是 C′ 的已知代价（P7_ROUTE_C_GATE.md），不是这一步能解决的，但训练时 `gflowrl/clip_saturation` 必须逐 step 记录。若它长期贴近 1.0，等于在跑一个符号化的 KL 正则，不是 GFlowRL。
2. **`β` 只有下界。** gate 推出 `drift/reward ∝ 1/β² ⇒ β ≥ 2`，上界没有。默认取 8.0，**这是继承论文的值，不是我们测出来的值。**
3. **合成不等于真实。** 上面全部是合成 rollout。真实数据里 `d` 的分布、`L` 的分布、退化率都会不同。步骤 2 证明的是**代码实现了它声称实现的数学**，不是这套数学在 SpaceTools 上有效。
4. **`ref_log_prob` 必须真的在 batch 里。** `dp_actor.py:528` 只在 `use_kl_loss` 打开时才追加它。开 gflowrl 时若忘了这个开关，`compute_gflowrl_flow_gap` 会 KeyError——这个是会崩的，属于好错误，但值得先知道。

---

## 5. 下一步

步骤 1&2 完成。步骤 3 起需要 GPU 和数据：

- 下载 SFT 数据（6.32 GB）+ RL 数据（3.38 GB）
- 跑 SFT（4 卡约 8–12 h，不需要工具常驻）→ SFT baseline eval
- **烟测 Step 4 能否塞进 4×A100-40GB**（这是当前最大的未知；工具占卡 + 60.6 GB optimizer state）
- GFlowRL 训练 → Table 2 九个 benchmark eval → 对 SFT baseline 和 P4 数字

`model_dtype=bf16`（偏离 [20]）**绝不带进训练**。
