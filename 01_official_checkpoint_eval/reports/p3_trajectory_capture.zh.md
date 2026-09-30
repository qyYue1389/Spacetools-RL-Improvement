# P3:trajectory 采集

> 本文是 `P3_NOTES.md` 的中文版。代码标识符、文件路径、配置键、字段名、
> 错误原文与 commit hash 一律保留英文。

## 译法对照

凡是做过取舍的译法都列在这里,附英文原文。对着原版核对时以此为准。

### 通用

| 中文 | 英文原文 | 说明 |
|---|---|---|
| 上游 | upstream | 指 verl / Toolshed 等原始仓库的状态 |
| 落盘 | dump / written | verl 把样本写成 JSONL |
| 污染 | contamination | 一次运行的数字不可信,而不是崩溃 |
| 静默 / 静默错掉 | silent / silently wrong | 没有报错,但结果是错的 |
| 守门员 | guard | 指 `--strict` |
| 线索 | lead | 指向原因但不是结论 |
| 物有所值 | earned its keep | |
| **分数** | **score** | 评测得分 |
| **区间 / 波动带** | **band** | `--compare` 报的上下界 |
| **极差** | **spread** | 多次抽样中最大与最小之差 |

### 本文特有

| 中文 | 英文原文 | 说明 |
|---|---|---|
| trajectory 采集 | trajectory capture | 标题里的 trajectory 保留英文,它是本项目的对象名 |
| 切分器 | splitter | `parse_transcript` 里按 `<\|im_start\|>` 切分的那部分 |
| 派生字段 | derived fields | 解析器算出来的,与直接从 JSON 读出来的相对 |
| 报数上的取舍 | reporting choices | |
| 富化 | enriched | `--emit` 产出的记录 |
| 按引用 | by reference | Toolshed 交回 `$depth_map` 这种名字而不是数组本身 |
| 交叉校验 | cross-check | 解析器的轮数 vs verl 的 `num_turns` |
| 回归测试 | regression test | P2 手算值 |
| 断点续跑 | resume | |
| 分类学 | taxonomy | P6 的错误分类体系 |
| 符合性 | conformance | 对照计划逐项核对 |
| 被取代 | superseded | 计划里的某项任务因上游已实现而作废 |
| 未构建 | not built | 有意不做,不是遗漏 |
| 告诫 | caveat | |
| 判决 | verdict | 「是线索而不是判决」 |

## P3 最后变成了什么

计划原本假定 P3 意味着给 `ToolAgentLoop` 挂 hook 来记录每一轮。
先读 verl,让其中大部分都变得没有必要。

`_dump_generations`(`verl/trainer/ppo/ray_trainer.py:390`)本来就会在
`trainer.validation_data_dir` 下面为每个样本写一行 JSONL,携带
`input`、`output`、`gts`、`score`、`step`、`messages`,
以及来自 `extra_info` 的 `image` / `answer` / `index` 字段。
它还有一个分支会把 `multi_modal_data` 里的图像存成 PNG——
但那个分支在这条 eval 路径上**从不触发**,因为 `multi_modal_data` 在这里
根本到不了 `_dump_generations`。这没问题,也不需要补丁;见下面的「符合性」一节。

决定性的细节是第 601 行:

    # Use skip_special_tokens=False to preserve tool call tokens in the output
    output_texts = [self.tokenizer.decode(ids, skip_special_tokens=False) ...]

因为响应是保留特殊 token 解码的,而且 verl 的多轮实现把工具结果
保留在响应的 token 序列里,**整份 transcript 已经躺在 `output` 这个字符串里了**——
每一个 `<think>`、`<tool_call>`、`<tool_response>` 和 `<answer>`,顺序完整。
什么都不需要挂 hook。

所以 P3 就是一个小补丁加一个离线脚本。

## 补丁:`patches/rl/0008`

有两个字段验证循环本来就在收集,却从来没有传给 dump:

| 字段 | 在哪里收集 | 为什么我们想要它 |
|----------------|-----------------------------------|----------------|
| `sample_uids`  | `ray_trainer.py:614`(`extend`) | 稳定的身份标识 |
| `sample_turns` | `ray_trainer.py:655`(`append`) | 交叉校验 |

真正重要的是轮数。它让解析器可以把自己对 transcript 的分解与 verl 的计数相比,
于是解析器的 bug 会表现为一次**响亮的不匹配**,而不是悄悄错掉的统计数字。
`parse_dump.py` 会报 `turn-count mismatch`,而 `--strict` 把它当作污染处理。

`sample_turns` 是按 batch `append` 而不是 `extend` 的,所以它到手时是一个
「数组的列表」,使用前要先 flatten。两处添加都放在 `try/except` 里面:
**这是诊断代码,绝不能有能力弄坏一次 eval。**

解析器并不依赖这个补丁——不管有没有它,解析器都会从 transcript 推导轮数。
打上补丁只是买到那个交叉校验。

## 脚本:`tools/parse_dump.py`

**只用标准库**,所以它能在五个环境中的任何一个里运行。它有两个职责。

### 1. P4 的守门员 —— 每个 benchmark 跑完就执行

    python tools/parse_dump.py "$EVAL_DIR/blinkdepth/0.jsonl" --strict

一次 out-of-memory **不会让 eval 崩溃**。工具返回一个错误字符串,
模型把它当作一次工具响应读下去并继续,最后这个 benchmark 报出一个
看似合理、实则悄悄错掉的数字。P2 有一次运行 OOM 了 68 次,因此丢了 5 分。
`--strict` 在遇到 OOM、畸形的 tool call、或轮数不匹配时以 1 退出。

**把非零退出读成「这个 benchmark 还没有结果」,而不是「有个警告」。**

### 2. P6 的输入

    python tools/parse_dump.py "$EVAL_DIR"/*/0.jsonl --emit parsed/

写出富化后的记录,其中 trajectory 已被分解、每一种失败模式都已标记,
于是错误分类再也不需要重读日志或碰 GPU。
只想要标记而不要完整 trajectory 时,加 `--emit-slim`。

### 派生字段

| 字段 | 怎么来的 |
|---------------------------|-----|
| `trajectory` | 按 `<\|im_start\|>` 切分,然后逐轮提取标签 |
| `num_turns_derived` | assistant 轮的数量 |
| `tool_stats` | tool call 的名字,按顶层工具分组 |
| `tool_failures` | 匹配失败模式的工具响应 |
| `oom` | `OutOfMemoryError` / `CUDA out of memory` |
| `truncated_tool_response` | 字面的 `...(truncated)` —— `tool_agent_loop.py:515` 在 middle 模式下插入的东西 |
| `hit_max_turns` | `num_turns >= 8`(`run_eval.sh` 的 `max_assistant_turns`) |
| `parse_ok` | 存在一个 `<answer>` 块 |
| `truncated_generation` | 最后一个 assistant 轮既没有 tool call 也没有 answer |
| `chain_signature` | 例如 `depth_estimatorx1+roboreferx2+vision_opsx2@3t`,用来把相同的工具使用模式分到一组 |
| `num_turns_verl_convention` | `assistant + user + 1`,唯一可以与 verl 落盘的 `num_turns` 相比的形式 |
| `vars` / `arg_vars` | 一次工具响应**发布**的 `$variables` / 一次 tool call **读回**的 |
| `vars_exposed` / `vars_used` / `vars_unused` | 逐样本汇总;`vars_unused` 是分类学 2d 的线索 |
| `vars_phantom` | 被引用但从未被任何工具发布过——名字是模型自己发明的。分类学 2c,而且比 `vars_unused` 更干净的信号 |
| `question` | 单独的问题本身,从约 10 KB 的渲染后 prompt 中提取 |

**OOM 的分类顺序排在 tool failure 之前**,所以一个因 OOM 而失败的响应
不会同时被算作「工具拒绝了这个任务」。它们是两回事:
**一个是运行污染,另一个是给 P6 的信号。**

### 报数上的取舍

`--compare` 接受同一个 benchmark 的多次重复运行,打印**区间而不是点值**,
因为 P2 在完全相同的配置下把 blinkdepth 测出了 108 / 107 / 111 个正确。
**比波动带还小的差异不是差异。**

错题分解把**有基础设施原因的**失败(工具错误、截断、轮数耗尽、缺失 answer、OOM)
与 `clean` 的分开——后者是「一切都正常工作却仍然答错」。
clean 的那一批才是 P6 真正的工作所在:真正的工具不准,或者真正的推理错误。

`-v` 会加上 grasp 的成功/失败拆分,那是「失败检测是否有效」最便宜的一个检查。

**不要把 P2 的 27 succeeded / 33 errored 当作期望值——它是错的。**
真实的拆分是 **19 / 41**,这一点由一次完全独立于解析器的原始文本搜索确认:
33 个样本命中 `No collision-free grasps`,另有 8 个命中
`Top-down filtering removed all`,而 P2 把后者算成了成功。
见 `CHANGES.md` §7.3 以及 `P2_RESULTS.md` 顶部的更正块。

## 验证解析器

`P2_RESULTS.md` 里已经有针对 blinkdepth **run4** dump 手工算出的统计量:
16 个错、零个被截断的工具响应、零次轮数耗尽、一个工具错误、
每个样本都产出了 `<answer>`,而且 16 个错题里有 12 个
正好用了 `depth x1, roborefer x2, vision_ops x2`、2 轮。

把解析器跑在那份 dump 上,这些数字应当原样回来。它们是独立推导出来的,
所以是一份**免费的回归测试**——在信任解析器处理任何新东西之前,先用它。

    python tools/parse_dump.py <run4>/blinkdepth/0.jsonl -v

来自 `P2_RESULTS.md` 的期望值:

    score >= 0.50        108/124 = 87.10%
    OOM samples          0
    truncated responses  0
    hit max turns        0
    no <answer>          0
    tool failures        1 sample
    wrong samples        16
      chains             12 x depth_estimatorx1+roboreferx2+vision_opsx2@3t

**重要的是那个计数,不是 `@Nt` 后缀。** P2 的手工分析写的是 `@2t`,
因为它数的是*调用了工具的*轮;而解析器数的是 assistant 轮,
第三轮正是那个发出 `<answer>` 却不调用任何工具的轮。两种数法指的是同样那十二个样本。

如果 chain signature 的计数对不上,那么在读任何别的东西之前,
transcript 切分器(splitter)就已经错了——而这正是它在 Phase V1 首次运行时发生的事,
并且顺带带出了三个 bug。见 `CHANGES.md` §7。

## 与计划中 P3 一节的符合性(conformance)

在 Phase V 之后,逐字段对着计划 artifact 重新核对过。

### 计划里的字段表

| 计划的字段 | 状态 |
|--------------|--------|
| `trajectory`(think / tool_calls / tool_results) | 已完成 |
| `tool_stats` / `tool_errors` | 已完成(`tool_stats`、`tool_failures`) |
| `num_turns` / `hit_max_turns` / `parse_ok` | 已完成,三种轮数约定分开保留 |
| `truncated_tool_response` | 已完成,依据字面标记而不是长度启发式 |
| `latency_ms` | **未构建** —— 见下文 |
| `image_path` | **已关闭,且不需要** —— 见下文 |

### `image_path`:靠测量关闭,而不是靠补丁

计划指出 `image` 是 null,并要求「打个小补丁」。
它在**目前测量过的八个 benchmark 中的七个**上都是 null,而原因并不是 bug:
`multi_modal_data` 在这条 eval 路径上根本到不了 `_dump_generations`,
所以 `ray_trainer.py:432` 那个存 PNG 的分支从不触发。
在一次完整运行的日志里,`multi_modal_data` 出现零次。

`boppose` 是例外,而它是**更糟的一种非 null**:它携带的是
`images/scene_000001_frame_000000.png` 这样的字符串,从 parquet 的 `extra_info` 复制而来。
**这样的文件并不存在**——它是一条指向原始 BOP 数据集布局的悬空相对路径。
信它比信 null 更糟。

而且打补丁本来就是错的修法。**`sample_id` 就是 parquet 的行索引**,
对着 blinkdepth 的 ground truth 逐条验证过 124/124,唯一且连续。
所以原图可以精确地这样取回:

    parquet[sample_id]["images"][0]["image"]     # base64 data URI

这是免费的、无损的,而且避免了在每份 dump 旁边再写一份 292 MB 的图像副本。
P6 的 prediction-vs-GT 叠图需要的一切都齐了。

有一条告诫值得带着:`make_subset.py` 产出的子集用 `head(N)`,
所以子集的第 *i* 行就是完整文件的第 *i* 行,`sample_id` 对两者含义相同。
**任何人一旦把子集改成随机抽样,这个等价关系就失效。**

### `pred`:推迟到 P5/P6,而且推迟不损失任何东西

计划的 schema 把 `raw_answer`(`<answer>` 里的文本)与 `pred`(结构化预测)分开。
只发出 `raw_answer` 是有意为之。

`gt` 与 `raw_answer` 在每个 benchmark 家族里都是**相同的格式**,
所以一个解析器可以对称地处理两侧:

    blinkdepth    gt "B"                              raw_answer "B"
    robospatial   gt "[(0.383, 0.873), ...]"          raw_answer "[(0.338, 0.934)]"
    bopgrasp      gt "Grasp center: [...], Left ..."   raw_answer "Grasp center: [...], ..."

所以 `pred` 纯粹是对记录已经携带的字段做离线文本解析——
它在 eval 时不需要捕获任何东西。现在写等于去猜那六个 dump 还不存在的 benchmark
的答案格式。它属于 P5/P6 的分析阶段,不在这台机器上。

### 计划里的七项 P3 任务

| # | 任务 | 状态 |
|---|------|--------|
| 1 | 给 `ToolAgentLoop` 挂 hook,逐轮抓 think / tool_call / tool_response | **被取代(superseded)** —— verl 已经落盘整份 transcript(`skip_special_tokens=False`);不需要 hook |
| 2 | 大变量只存摘要(shape、range、mean) | **上游已满足** —— Toolshed 从不内联数组;一次响应写的是 "Use `$depth_map` (numpy array, 364x504) ... Depth range: 3.09m to 1094.68m (mean: 20.95m)"。**摘要本身就是响应**。解析器现在还额外记录了两侧的变量名(见下) |
| 3 | 显式记录工具错误 | 已完成(`tool_failures`,OOM 单独分类) |
| 4 | 标记被 `max_tool_response_length=2048` 截断的响应 | 已完成 |
| 5 | pose / grasp 的 prediction-vs-GT 叠图 | **推迟到 P6**,而且现在可证明可行:`sample_id` → parquet → 图像 |
| 6 | 按 `sample_id` 做样本级断点续跑 | **未构建** —— verl 每个 benchmark 写一个 `{step}.jsonl`,而不是逐样本追加,所以续跑属于外层 shell 循环(跳过 dump 已存在的 benchmark),不属于 eval |
| 7 | 进 P4 之前,在 P2 样本上验证 dump 完全可解析 | **已完成,而且物有所值** —— 它抓到了三个真实的 bug;见 `CHANGES.md` §7 |

### 核查任务 2 时顺带加上的:变量复用追踪

任务 2 结果是被 Toolshed 的设计满足了,但核查它的过程带出了一样东西——
计划在它的 schema 里要求过(`"vars": [...]`),而且 P6 直接需要它。
Toolshed 把大结果**按引用(by reference)**交回:

    Use $depth_map (numpy array, 364x504) to reference the depth data
    and $focal_length_px (float) to reference the focal length.

而后续调用把它们读回来:

    vision_ops.index_at  arguments {"data": "$depth_map", "u": 0.443, ...}

两侧现在都记录了——每次工具响应的 `vars`、每次 tool call 的 `arg_vars`,
并逐样本汇总为 `vars_exposed` / `vars_used` / `vars_unused`。

**`vars_unused` 才是重点**:一个工具发布过、而模型从未读回的变量,
正是**分类学(taxonomy)2d「重算了自己已经有的东西」**的签名。
它是线索(lead)而不是判决(verdict)——在 blinkdepth 上,124 个样本里有 106 个正好留下一个未使用的变量,
而且总是 `focal_length_px`,那个 benchmark 确实不需要它。
值得看的是尾巴:本次运行测到 11 个样本有三个未使用变量、1 个有四个,
那正是 roborefer 的检测结果被发布之后、又被一次 vlm 重试取代的地方。

### 另外加的

`question`(从渲染后的 prompt 里提取,而不是把整整约 10 KB 都带着——
那会在九个 benchmark 上多出约 21 MB,而那些文本除了最后几行以外每一行都一样),
以及 `reward` 的透传。

## 有意不构建的

**逐工具 `latency_ms`。** 那需要给 Toolshed 挂埋点,而 P2 已经表明
瓶颈是策略生成而不是工具:轮数变化 2.5 倍、像素数变化 16 倍时,
单样本成本始终维持在 3.3-3.7 s。

**样本级断点续跑。** verl 每个 benchmark 写一个 `{global_steps}.jsonl`,
而不是逐样本追加,所以续跑属于外层 shell 循环——
跳过 dump 已经存在的 benchmark——而不属于 eval。

**pose / grasp 的 prediction-vs-GT 叠图。** 对 P6 值得有(论文 Fig. 10-12 那种风格),
但它们是从 parquet 的 ground truth 加上解析出的预测渲染的,与 eval 完全解耦。
推迟它们不阻塞 P4。
