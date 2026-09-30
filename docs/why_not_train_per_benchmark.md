# 为什么不能按 benchmark 分开训练

写于 2026-09-15 · 起因:为了省卡,考虑「训练某个 benchmark 时只部署那个 benchmark
需要的工具」· 相关 `GFlowRL/README.md`、`RL 训练准备交接文档(未收录)`

---

## 结论

**机制上做得到,但不该做。** 四条理由,任何一条单独成立都足够;第 2 和第 3 条是硬的,
第 4 条说明它连省钱这个初衷都达不到。

末尾第 6 节记了这个想法里**唯一值得单独算的变体**,以及第 7 节一个能拿到同样省卡效果、
且零偏离的替代做法。

---

## 1. 机制上做得到 —— 有先例

工具集不是靠提示词约束的,是**在启动工具服务时就只注册想要的那几个**:

```
examples/toolshed/run_rl_roborefer.sh      ← Step 1 就是这么干的,只注册 roborefer
examples/toolshed/generate_toolshed_config.py
       从正在运行的服务反查工具 schema,再生成交给 verl 的 YAML
```

所以没注册的工具**真的会从 system prompt 的 `<tools>` 段里消失**,模型抽不出那个工具名;
就算凭空编一个调用,也没有服务器接,直接报错。

这一节的意思只是:下面三条反对理由**不是"做不到"**,是"做了会坏"。

---

## 2. 没有「benchmark 的训练集」这个东西

这是前提层面的错位,不是权衡。

```
训练数据    spacetools-rlfulltools
run_rl.sh:335   RL_PARQUET="$EXPERIMENT_DIR/rl_data/data/train.parquet"
run_rl.sh:439   data.train_files="[$RL_PARQUET]"        ← 一个池化文件
```

**benchmark 是 eval 侧的概念** —— 九个 key,各自一个 `data/<key>.parquet`,由 Step 5 的
`run_eval.sh` 读取。训练侧只有一个混合池 `train.parquet`。

所以「按 benchmark 分开训练」第一步就得**自己把训练池切开**,而:

- 按什么切不清楚。`train.parquet` 里有没有能对应到 benchmark 的列(`data_source` 之类),
  没有验证过。
- 切完每块剩多少条不清楚。全量 5425 条 / `train_batch_size=64` ≈ 85–86 步 = 1 epoch。
  切成 N 块之后每块可能只剩十几步,**不够训**。
- 就算切得开,切出来的也是「训练分布的一个子集」,不是「那个 benchmark 的训练集」。
  两者不是一回事。

---

## 3. 它把 Step 4 变回 Step 1

**Step 4 的全部目的就是学工具编排** —— 在 11–17 个工具里选、串起来、出错了怎么救。
只部署某个子集需要的工具,模型就没有可选的了,搜索空间被人为塌缩。

论文自己的消融给了这条的方向(Table 4):

```
完整四步流水线                       52.48
从 base model 直接跑全工具 GRPO      19.79     ← 搜索空间太大,学不动
```

Step 1 之所以**只开一个工具**,正是因为搜索空间必须小到「瞎试也能撞对」,退化组才不会
把训练卡死。而 Step 4 的前提恰好相反:模型已经会用工具了,这时候把全部工具打开,
让它自己探索比示范更好的编排。

**把工具集缩回子集,等于把 Step 4 的目标换成了 Step 1 的目标。** 训出来的是一组窄专家,
而 Step 5 的 eval 要的是**一个模型**在九个 benchmark 上的表现。

---

## 4. prompt 分布与 eval 不匹配,外加灾难性遗忘

工具清单是注入 system prompt 的,而且是 `generate_toolshed_config.py` 从运行中的服务
**反查**出来的 —— 部署什么,prompt 里就写什么。

于是分阶段训练会有两个连带后果:

**① 最终 ckpt 是在一个 eval 时不存在的 prompt 分布下训出来的。**
每个阶段模型看到的 `<tools>` 段都不同,也就是**任务格式不同**。而 Step 5 的 eval 一律
开全部工具,prompt 里是完整清单。训练时从没见过完整清单的模型,在 eval 时面对的是
一个陌生格式。

**② 顺序训练会遗忘。** 前一阶段学会的工具用法,在后一阶段既没有对应的工具可调、
也没有奖励维持它。最后一个阶段的工具会被过度强化,之前的会退化。

这两条合起来的意思是:**分阶段训练得到的最后那个 checkpoint,不是「学会了全部工具的
模型」,而是「刚学完最后一个子集的模型」。**

---

## 5. 成本反而更高

初衷是省卡。但账要算完整的:

```
省的     每次训练少部署几个工具 → 工具侧少占 1–2 张卡
付的     子集数 N 倍的训练次数
```

工具侧省下的是**每次运行的卡数**,而切分付出的是**运行次数**。后者是乘法。

而且省下来的那部分本身有限:工具显存的大头是三个模型(见第 6 节),多数
benchmark 至少要 pointing,也就是最大头之一的 `roborefer` 砍不掉。

---

## 6. 唯一值得单独算的变体:能不能砍掉 `vlm`

这个想法里真正有价值的部分不是「分 benchmark」,而是**「最占显存的那个工具是不是必需的」**。

单实例显存(SFT eval 实测,GiB):

```
vlm (Molmo fp32)     30.2      ← 单个最大头
roborefer            17.2
depth_estimator       7.9
sam2 / bbox / grasp   1.3 / 0.5 / 1.0 each
vision_ops            0
```

压缩到 13 actor 的布局合计 **86.0 GiB**,其中 `vlm` 占 35%。砍掉它:

```
86.0 → 55.8 GiB     一张 80 GB 卡装得下 roborefer×2
                    或者两张 48 GB 卡能给 roborefer 更多副本
```

这正好解掉「卡少时 `roborefer` 只能留 1 个副本」这个痛点 —— 而 `roborefer` 是调用最
频繁的工具,它的并发直接决定整步时间。

**但两件事挡在前面:**

**① 不能假设用不到它。** `p6/passk` 的 system prompt 里,`vlm` 和 `roborefer` 是**并列的
pointing 选项**:

```
{"name": "vlm.detect_one",       "description": "Detect *one* instance of *obj_name*..."}
{"name": "roborefer.detect_one", "description": "Detect *one* instance of *obj_name*..."}
```

模型完全可能在一部分样本上走了 `vlm`。这是**可查的**:数一下 SFT 起点轨迹里
`<tool_call>` 中 `vlm.*` 的占比。数据在 `eval/SFT/sft-eval-artifacts/rollouts/`
—— 注意 `p6/passk/*/0.jsonl` **不够用**,那里面只有 prompt 和最后一轮,
中间几轮的工具调用不在里面。

**② 就算占比是 0,这仍然是「换工具」。** 用户 2026-09-02 已定不做,理由是
「换目标函数与换工具正交,混在一起不可归因」。而且 P6 归因里那 84.7% 正是
工具错 + 工具集缺口 —— 动工具集就动了那个归因的地基。

**所以它要么不走,要么当成另一个实验走。**(「Step 4 只开 pointing 工具」本身是个
干净的问题,但那是另一篇的事。)

---

## 7. 想省卡,零偏离的做法

不动工具集,只压 `num_actors`:

```python
# roborefer 6→2 · vlm 2→1 · sam2 5→2 · depth 5→2
# bbox 5→2 · grasp 5→2 · vision_ops 8→2
# 共 13 actor,并发只降 2.8×,逻辑 GPU 需求 7.8 → 3.0
```

**这是零偏离的** —— 工具清单不变、prompt 不变、一个数都不变,只是让 rollout 排队。
它省卡的收益和「少部署工具」重叠很大,而代价只是时间,并且时间的代价**烟测三步就能
量出来**(看 Toolshed 日志的 actor 忙/闲占比)。

⚠ 压 `num_actors` 时必须同步抬 `TOOL_GPUS`:Ray 的 `num_gpus` 是**逻辑预留**,不是显存
配额。PG 预留少于 actor 索取的总量时,PG 会 ready 而 actor 永远排不进去
—— `run_rl.sh` 的 heredoc 里有一条 assert 拦这个。

---

## 引用来源

| 断言 | 出处 |
|---|---|
| 工具集靠注册决定,有单工具先例 | `examples/toolshed/run_rl_roborefer.sh`、`generate_toolshed_config.py` |
| 训练数据是一个池化 parquet | `run_rl.sh:335`、`run_rl.sh:439` |
| eval 按 benchmark 分 parquet | `交接 RL 训练准备.md` §6 数据 |
| 5425 条 / 85–86 步 = 1 epoch | 同上 |
| 跳过前三步只得 19.79 | 论文 Table 4 |
| 单实例显存 | SFT eval 实测,`sft-eval-artifacts/gpu/gputrace_1hz.log` |
| `vlm` 与 `roborefer` 并列为 pointing 选项 | `p6/passk/robospatial/0.jsonl` 的 system prompt |
| `p6/passk` 不含中间轮的工具调用 | 2026-09-15 实查该文件结构 |
| 换工具已被排除 | 用户 2026-09-02 决定,记于 `交接 RL 训练准备.md` §3 |
| 压 `num_actors` 零偏离 / `TOOL_GPUS` 断言 | `交接 RL 训练准备.md` §4.1、§5.8 |
