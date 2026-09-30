# P7 步骤 4 结果:`response_mask` 与 `|y_i|` 的定义

> `P7_DECISION.md` §6 第 4 步。**零 GPU。** 写于 2026-09-01。
> 这一步是 GPU 前向 pass 的**正确性前置**:前向要算的 `log π_ref` / `log π_old`
> 必须只算 assistant token,mask 错了整个 `Z_t` 分解就是废的。

---

## 摘要

**`|y_i| = response_mask.sum()`,而它确实只数策略生成的 token** —— 这次是从
verl 源码逐行确认的,不是从 FlowRL 的 `masked_mean` 用法反推的。

**但这一步挖出两个此前不知道的东西,都会改变后面的做法:**

1. **`response_mask` 这个名字在同一份代码里有两种相反的语义。**
   agent loop 给的那个 = 「策略生成的 token」;`ray_trainer.py` 的 fallback
   `compute_response_mask()` = `attention_mask[:, -L:]`,**工具响应 token 全是 1**。
   用错就是把 `|y|` 放大 1.14–2.40×(步骤 2 已量过这个倍数)。
2. **`p4/dumps/` 只存文本,不存 token id 与 mask。**
   所以对已记录轨迹做前向,就必须**从文本重新推导 mask** —— 正是 P3 明令禁止的事。
   **出路是在第二组采样开机之前先加一个 dump 补丁**,见 §4。

---

## 1. 源码证据:mask 到底数哪些 token

`SpaceTools-RL @ f0742338` · verl 0.8.0.dev

    verl/experimental/agent_loop/tool_agent_loop.py
      L287-289   agent_data.response_ids = output.token_ids            <- sglang 实际生成的
                 agent_data.response_mask += [1] * len(response_ids)

      L430-431   response_ids = apply_chat_template(add_messages, ...)  <- 工具响应,整块模板化
                 agent_data.response_mask += [0] * len(response_ids)

      L462-463   同上,interaction(user)分支

    verl/experimental/agent_loop/agent_loop.py
      L291-317   apply_chat_template(..., add_generation_prompt=True)
      L631       response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
      L463 注释   "1 for LLM generated tokens, 0 for observation/padding tokens"

**三条推论,都来自上面的行:**

- **`mask = 1` ⟺ sglang 自己吐出来的 token。** 不含任何模板化内容。
- **下一轮的 `<|im_start|>assistant\n` 落在 `mask = 0` 的块里** ——
  因为 `add_generation_prompt=True`,它是工具响应那一块模板的尾巴,不是生成出来的。
- **padding 也是 0**(L631 又乘了一次 `attention_mask`)。

> 这同时从 token 层面解释了 P3 那个 bug 的根源:**第一轮的
> `<|im_start|>assistant` 在 prompt 里**(初始 `apply_chat_template` 也带
> `add_generation_prompt=True`),所以 `output` 是从第一轮 assistant 的**正文中间**开始的。
> `parse_dump.py` 的 `leading_is_assistant` 处理与这里的 token 事实一致。

---

## 2. ⚠ 同名不同义:两个 `response_mask`

    verl/trainer/ppo/ray_trainer.py
      L111-126  def compute_response_mask(data):
                    return attention_mask[:, -response_length:]        <- 工具 token 也是 1
      L157-158  if "response_mask" not in data.batch.keys():
                    data.batch["response_mask"] = compute_response_mask(data)   <- 静默 fallback

    verl/experimental/agent_loop/agent_loop.py
      L800      "response_mask": response_mask,   # [bsz, response_length]  <- 真正想要的那个

SpaceTools 走 agent loop,所以 batch 里**本来就有**正确的那一个,fallback 不会触发。
**但它是静默的**:一旦哪天 batch 里没有这个键(换 rollout 路径、改 dataproto、
自己拼 batch 做离线前向),`compute_response_mask()` 会**不报错地**给出一个语义相反的 mask。

> **这正是 CHANGES.md §10 那条「不要拿静态声明推断运行时行为」的又一个实例,
> 而且这次两个东西连名字都一样。**

**处方 —— 写进 loss 的运行时守卫**(多轮样本上必然成立):

    # 多轮轨迹里,策略 token 一定严格少于全部非 padding token
    L_resp = response_mask.size(1)
    n_policy = response_mask.sum(-1)
    n_nonpad = attention_mask[:, -L_resp:].sum(-1)
    multi_turn = (num_turns > 2)
    assert (n_policy[multi_turn] < n_nonpad[multi_turn]).all(), \
        "response_mask looks like the attention-mask fallback: |y| would include tool tokens"

单轮样本上两者相等是正常的,所以判据要挂在多轮样本上。
按步骤 2 的实测,若守卫失效,`|y|` 会被放大 **1.14×(robospatial)/ 1.41×(blinkdepth)/
2.40×(boppose)**,而且**按 benchmark 系统性不同**。

---

## 3. 结构核对:`p4/parsed/` 全量

| benchmark | n | 轮数交叉校验 | 截断 | 轮数耗尽 | assistant 轮 mean/min/max | 工具轮 mean |
|---|--:|--:|--:|--:|--:|--:|
| `blinkdepth` | 124 | 124/124 | 0 | 0 | 3.13 / 1 / 5 | 2.13 |
| `bopgrasp` | 60 | 60/60 | 0 | 0 | 4.95 / 4 / 5 | 3.95 |
| `boppose` | 60 | 60/60 | 0 | 0 | 5.03 / 5 / 6 | 4.03 |
| `cvb2drelation` | 650 | 650/650 | 0 | 0 | 2.06 / 2 / 6 | 1.06 |
| `cvb3ddepth` | 600 | 600/600 | 0 | 0 | 3.04 / 3 / 5 | 2.04 |
| `reflocation` | 100 | 100/100 | 0 | 0 | 2.00 / 2 / 2 | 1.00 |
| `refplacement` | 100 | 100/100 | 0 | 0 | 2.00 / 2 / 2 | 1.00 |
| `refunseen` | 77 | 77/77 | 0 | 0 | 2.00 / 2 / 2 | 1.00 |
| `robospatial` | 350 | 350/350 | 0 | 0 | 2.00 / 2 / 2 | 1.00 |
| **合计** | **2121** | **2121/2121** | **0** | **0** | | |

**`assistant 轮 = 工具轮 + 1` 在九个 benchmark 上恒成立。**
所以 mask 的形状是严格交替的 `1…1 / 0…0 / 1…1 / … / 1…1`,以 1-块结尾 ——
没有连续 assistant 轮,也没有结尾的工具轮。这与 agent loop 的结构一致。

复算:`python3 -c` 遍历 `p4/parsed/*.jsonl` 的 `num_turns_agree` /
`num_turns_derived` / `num_user_turns_derived`(字段由 `parse_dump.py` 生成)。

---

## 4. ⚠ 这一步改变了 GPU 前向 pass 的规格

**问题:`p4/dumps/` 与 `p6/passk/` 存的是 `input` / `output` 文本,没有 token id,也没有 mask。**

所以「对已记录轨迹做一次前向」这件事,实际需要:

1. 重新 tokenize 文本 —— 而 `output` 是 `skip_special_tokens=False` 解码出来的,
   重编码**不保证**与原 token 序列逐位相同(空白、特殊 token 边界);
2. **从文本重新推导 mask** —— 正是 `P7_HANDOFF.md` §2① 明令禁止的
   「不要自己重写 assistant/tool 的切分」。

**出路(推荐):在第二组采样开机之前,先加一个 P3 风格的 dump 补丁。**
`patches/rl/0008` 已经示范了做法:在 `_dump_generations()` 里防御式地多写两个字段。
这次要写的是:

    base_data["response_mask_rle"]  # [[value, run_length], ...],游程编码
    base_data["n_policy_tokens"]    # = response_mask.sum(),即 |y_i|

游程编码是因为一条 4 轮轨迹只有约 8 段,体积可忽略,而且能**逐位重建** mask。
数据在 `test_batch.batch["response_mask"]` 里(agent loop 放的那个),
与 `sample_turns` / `sample_uids` 在同一个作用域,补丁形状一致。

> **这样一来,前向 pass 消费的是 verl 自己的 mask,而不是我们重新推导的。**
> 代价:第二组采样必须**先打补丁再跑**,否则跑完还得重跑。**这是排期上的一条依赖。**

---

## 5. 局限

- **本步没有实际跑过 tokenizer**,`|y|` 的 token 数仍未测(本机无 tokenizer,
  `models/` 在 GPU 机器上)。步骤 2 的字符代理仍然是目前唯一的量级依据。
- §2 的守卫**没有在真实 batch 上跑过**,它是从源码推出来的写法,首次训练/前向时要验证。
- §4 的补丁**只写了规格,没有实现**(按约定,不在路线敲定前写训练侧代码)。
- 轮数交叉校验来自 `parse_dump.py` 自己的字段,与 verl 的 `num_turns` 对账;
  它验证的是**轮的个数**,不是**token 边界**。token 边界只有拿到真 mask 才能验。
