# P7 P0 结果:三臂同机同池重评与改点归因

跑于 2026-09-23 · 4× RTX 6000 Ada · 依据 `P7 问题与优化方向(已由 docs/design_doc_gflowrl_optimization.md 取代)` 的 P0 计划
原始产物:`/workspace/exp/p0/`(12 次运行的 dump)、`/workspace/parsed/`(富化记录)

---

## 0. 一句话

**三个 checkpoint 在同一台机器、同一 session、同一个 KV 池下各跑 3 次之后:整体上 P4(GRPO)与 C′ 分不开(p=0.28),但把 robospatial 拆成 VQA 与 Vacant 之后,两个方向相反的显著效应浮出来——C′ 在 VQA 上略好,在 Vacant 上显著更差(p=0.0004);而 Vacant 的全部差距由"改工具给的点"这一个行为解释,且该行为是从 SFT 起点继承的,GRPO 压住了,C′ 没压住。**

---

## 1. 怎么跑的

| | |
|---|---|
| 机器 | 4× RTX 6000 Ada(sm_89,49140 MiB,四卡容量一致、无混合 ECC)· Ubuntu 24.04 · 驱动 610.43.02 |
| 环境 | `qzpm55555/spacetools-eval-env`,`VERIFY.sh` 四项全过 |
| KV 池 | 整卡 47 GB × `gpu_memory_utilization=0.511` = **24.5 GB**,三臂完全一致 |
| 规模 | 12 次独立调用 · 5097 个样本 · 每次约 24 分钟 · 合计约 4 h 10 m |
| 顺序 | run1(sft→p4→cp)→ run2 → run3 → ref1,**按 run 交叉排**,不按 ckpt 分组 |
| dump | 全部保留 |

三个臂的出处(按 revision 钉死):

| 臂 | repo | revision |
|---|---|---|
| SFT 起点 | `qzpm55555/spacetools-sft-v1-4xa6000` | `91fd4bdf` |
| P4 官方 GRPO | `siyich/spacetools-ckpt` | `f953b1a1` |
| C′ step85 | `qzpm55555/spacetools-p7-gflowrl-cprime-8xa40` | `global_step_85` |

### 1.1 两处偏离(必须记入 PROVENANCE)

**① `vlm` 的 `num_gpus` 0.6 → 1.0。** 第一次跑 blinkdepth 时 Ray 把 `vlm 0.6 + depth 0.2 + depth 0.2 = 1.0` 打包进同一张卡:Molmo fp32 32.11 + DepthPro 7.94×2 = 48 GiB,而 torch 可见容量 47.40 GiB。**88 次 OOM 被 toolshed 包成正常 `ToolResult` 吞掉**(README 洞 #22),blinkdepth 静默掉到 66.94%(正常带 85–88%)。把 `vlm` 提到 1.0 强制 Molmo 独占一张卡,显存分布回到 P7 那次健康运行的样子(18.7 / 31.3 / 9.1 / 43.1 GB)。**这只改 Ray 的放置提示,不改任何计算。**

同时加了 `ulimit -c 0`:P7_GPU_RESULTS 记过一次 OOM 写了 50 GB 的 core 把容器盘撑满。

**② 官方 ckpt 的 config 补丁。** `siyich/spacetools-ckpt` 发布时**没有**施加上游 `docs/SETUP.md` 记录、`run_sft.sh` 对自训 ckpt 自动施加的修正,直接加载死在 `RuntimeError: Unimplemented model type: qwen2_5_vl_text`。用 P4 当初同一个脚本 `spacetools-repro/tools/fix_checkpoint.py` 原样施加三处:

- 删 `config.json` 的 `text_config`(有它在,transformers 4.57.1 会把 `AutoConfig` 解析成文本子模型)
- 设 `tie_word_embeddings=True`(脚本先逐位比对 `lm_head.weight` 与 `embed_tokens.weight`,确认 identical 才设)
- `preprocessor_config.json` 换成基座模型的非 Fast 版本(Fast processor 的 image token 数与视觉编码器不符)

**权重一个字节没动**,原件备份为 `*.orig`。环境版本与 P4 当初完全一致(sglang 0.5.6 / transformers 4.57.1,见 `records/env-freeze/spacetools-rl.txt`),所以这不是版本漂移,是发布的 ckpt 本身缺修。

---

## 2. 健康门禁

**基础设施层全零:** 27 个 dump 全部通过 `parse_dump.py --strict`,OOM 0、截断 0、malformed tool_call 0。

**模型行为层 6 例**(约 5000 样本的 0.12%,分布 **SFT 3 / P4 3 / C′ 0**):

```
grasp_generator.compute_grasp  ×3   在 blinkdepth(深度题)上调抓取工具
laser_radar.detect_one         ×1   工具不存在(v1 工具集里没有)
depth_estimator.index_at       ×1   方法张冠李戴(index_at 属于 vision_ops)
vision_ops.index_at            ×1   把字符串 "$point_cloud" 当变量传(P4/P5 记过的老相识)
```

逐个核对过:**没有一个落在被报告的分歧对里**,所有结论不受影响。唯一擦边的是 blinkdepth #22(SFT run1 出错、推向错误、有利于 P4),而那组结论本来就是"无差别"(p=0.85)。

**这不是流水线污染,是工具纪律问题。** P4 当初报"健康指标无一例外全为零"用的是基础设施层的口径,建议以后分两层报。

---

## 3. 主表

| | robospatial /350 | 恒对 | 恒错 | 翻转 | blinkdepth /124 | 恒对 | 翻转 | RefSpatial /277 |
|---|---|--:|--:|--:|---|--:|--:|--:|
| SFT 起点 | 215 · 211 · 213 → **213.0** | 185 | 108 | 57 | 107 · 109 · 108 → **108.0** | 104 | 7 | 147 |
| P4 官方 GRPO | 223 · 229 · 228 → **226.7** | 205 | 102 | 43 | 105 · 110 · 111 → **108.7** | 103 | 9 | 149 |
| C′ step85 | 223 · 221 · 221 → **221.7** | 203 | 105 | 42 | 111 · 113 · 110 → **111.3** | 109 | 5 | 148 |

**跨机器可比性坐实了:** SFT 起点的 215 / 211 / 213 与旧机器(A6000,sm_86)上的 216 / 211 几乎重合。换卡、换架构、换机器都没有走样。

**RefSpatial 三臂平手**(147 / 149 / 148),印证 §10.3「这项没有策略自由度」。

**分辨率(现在是量化的):** 三次均值的 sd = √(翻转带/3)/2 → robospatial 上 SFT 2.18 / P4 1.89 / C′ 1.87 题;**两臂之差的 sd ≈ 2.7 题,2σ ≈ 5.3 题**。P4 单次能差 6 题(223 vs 229),**单次读数依然不可用**。

---

## 4. 逐样本配对检验

两种算法都列,真相在中间:

- **稳定核**:只用两边 3 次都一致的样本。保守 —— 丢掉了"在 A 下摇摆、在 B 下变恒对"的样本。
- **三次配对合并**:run i 对 run i 逐样本比,三次分歧合并。用满数据,但同一批样本数了三遍,**偏乐观**。

| 对照 | 稳定核 McNemar | 三次配对合并 |
|---|---|---|
| robospatial SFT→P4 | 净 +10,p=0.052 | 71 vs 112,净 +41,**p=0.003** |
| robospatial SFT→C′ | 净 +3,p=0.250 | 47 vs 73,净 +26,**p=0.022** |
| robospatial P4→C′ | 净 −5,p=0.405 | 93 vs 78,净 −15,p=0.284 |
| blinkdepth SFT→P4 | 净 −2,p=0.500 | 13 vs 15,净 +2,p=0.851 |
| blinkdepth SFT→C′ | 净 +1,p=1.000 | 3 vs 13,净 +10,**p=0.021** |
| blinkdepth P4→C′ | 净 +3,p=0.250 | 6 vs 14,净 +8,p=0.115 |

---

## 5. 关键一步:拆成 VQA 与 Vacant

整体看 P4 与 C′ 分不开(p=0.284)。**拆开之后是两个方向相反的效应在互相抵消。**

| | VQA /228 | Vacant /122 |
|---|---|---|
| SFT 起点 | 163 · 160 · 164 → **162.3** | 52 · 51 · 49 → **50.7** |
| P4 官方 GRPO | 160 · 166 · 167 → **164.3** | 63 · 63 · 61 → **62.3** |
| C′ step85 | 167 · 167 · 170 → **168.0** | 56 · 54 · 51 → **53.7** |

| 对照 | VQA(三次合并) | Vacant(三次合并) |
|---|---|---|
| SFT→P4 | 56 vs 62,净 +6,p=0.65 | 15 vs 50,净 +35,**p=0.00002** |
| SFT→C′ | 42 vs 59,净 +17,p=0.111 | 5 vs 14,净 +9,p=0.064 |
| P4→C′ | 54 vs 65,净 +11,p=0.359 | 39 vs 13,净 **−26**,**p=0.0004** |

**读法:C′ 在需要推理的 VQA 上比 GRPO 好 3.7 题(不显著),在 pointing 的 Vacant 上比 GRPO 差 8.6 题(显著)。** 后者量更大,所以整体是 P4 领先 5 题;两个效应方向相反,所以合在一起看不出显著性。§10 当初的"一项都不显著"有一部分是这么来的。

> **方法论:以后判读必须按 VQA / Vacant 分开。** 这是本次最有转移价值的一条 —— 合并子集会掩盖方向相反的真实效应。

---

## 6. 改点归因(P2 的问题)

判据:答案里的坐标是否**全部**出现在工具返回的坐标集合里(3 位小数精确匹配)。三次的平均。

**判别口径已校准:** 这套判别在 P4 上复现出 §10.6 记录的 **透传 101 / 改点 21 / 改点正确率 28.6%**,与报告吻合。

| | 透传(每次) | 透传正确率 | 改点 | 改点正确率 | 改点落在 0.05 网格上 |
|---|--:|--:|--:|--:|--:|
| **SFT 起点** | **72.7** | 56.9% | **49.3** | 18.9% | 18.7 |
| P4 官方 GRPO | **100.3** | 56.1% | **21.7** | 27.7% | 7.0 |
| C′ step85 | 77.0 | 57.6% | 44.0 | 21.2% | 18.3 |

### 结论:是"GRPO 压住了、C′ 没压住",不是"C′ 学会了改点"

**SFT 起点就已经改 49.3 个点,比 C′ 的 44.0 还多。** GRPO 把它压到 21.7,C′ 几乎没动。0.05 网格指纹(P6 量化过的"目测"签名)给出同一个排序:起点 18.7 → GRPO 7.0 → C′ 18.3。

**反事实核算:** C′ 若保持 GRPO 的透传比例,Vacant = 100.3×57.6% + 21.7×21.2% ≈ **62.4 题**,而 P4 实测 **62.3 题**。P4 与 C′ 在 Vacant 上的 8.6 题差距,一题不剩地由透传/改点比例解释。

### 这改变了三件事

1. **C′ 的目标函数没有"鼓励改点"。** 原先把它列为"C′ 相对 GRPO 唯一可测的有害变化"是读反了。
2. **它变成 P1(信号饥饿)的直接证据。** 改点错误率约 79%,奖励本应强烈压制;GRPO 在同数据、同 G=5、同 1 epoch 下压住了,C′ 没有 —— 与 §8.4 的读数(70.3% 组退化、梯度只剩漂移项、grad_clip=1.0)是同一件事的两面。
3. **不要写奖励塑形。** 既然 GRPO 同配置就能压住,这是信号问题而非奖励缺项;把启发式写进奖励会掩盖 P1 的效果。降级为"只在 P1 之后仍然存在时才考虑"。

### C′ 并非全面落后

同一批 dump 上:C′ 在 VQA 上 +3.7 题、blinkdepth 上 +2.6 题、翻转带最窄(robospatial 42、blinkdepth 5,三臂最低)、模型侧工具误用 0 例(SFT 和 P4 各 3 例)。**C′ 的问题集中在 pointing 的一个具体行为上。**

---

## 6b. 工具调用:depth_estimator 的真实调用数(2026-09-23 用本批 dump 复算)

这是第一次有 SFT 起点的 dump 可以查这件事。

| robospatial(n=350) | depth_estimator 调用 | 工具调用总数 |
|---|---|---|
| SFT 起点 | **0 · 0 · 0** | 577 · 577 · 577 |
| P4 官方 GRPO | 2 · 2 · 1 | 576 · 575 · 573 |
| C′ step85 | 0 · 0 · 1 | 578 · 576 · 579 |

**起点本身就不调。** 不是两条 RL 臂把它压没的,是这条链路从来没进过支撑集 —— §10.4 的判断成立,P3 维持最后。
(纠一处:§10.4 记的「SFT 起点是 1 次」来自旧机器单次运行,三次实测是 0/0/0。)

**两条旁证:**

1. **robospatial 上七个工具只用了 roborefer 一个**(578 次调用全是它,1.65 次/样本),其余六个一次没用。
   这让 §6 的改点问题更清楚:模型手上只有 roborefer 给的那个点,改它就是纯目测,没有第二个信息源可以交叉验证。
2. **对照 blinkdepth(n=124)**:depth_estimator 调用 SFT 121/122/120、P4 124/123/123、C′ 122/122/122 —— 几乎每样本一次。
   **模型会调这个工具,只是在 robospatial 的题面下不会想到。** 这把「数据分布问题」坐实了,不是能力问题。

复算方式:`parse_dump.py --emit-slim` 之后聚合每条记录的 `tool_stats`,零 GPU。
注意 `--emit` 的输出文件名取自输入路径的父目录名,本批是 `dumps.jsonl`;其中的 `correct` 字段
因 benchmark 名推断不同不可直接使用,`tool_stats` 不受影响。

---

## 7. 训练中要加的两个监控量(不需要 GPU)

- Vacant / pointing 类的**改点率**:目标从起点的 ~40% 往 GRPO 的 ~18% 走。
- 答案坐标落在 **0.05 网格**上的比例:上升就是在学"自己目测"。

---

## 8. 没做的事

- **bopgrasp 判分符号:决定不改,标为「不可用」**(2026-09-23)。`parse_dump.py` 统一用 `correct = score ≥ 0.5`,而 bopgrasp 的 `score` 是 RL 的 NCE(越低越好,均值约 2.0,见 `records/CHANGES.md` §7)——那一行既不是准确率、符号还是反的(抓得越差越被判成对)。**但改成 `NCE ≤ 阈值` 只是把一个错数换成另一个可疑的数**:P5 已证明 NCE 由位置主导、SR 由朝向主导,两者排序相反(朝向中位错 63.9° 的回退答案 NCE 反而更低)。正确做法是报连续 NCE + 符号检验或拆成 MACE / SR,只在真要拿 bopgrasp 做 GRPO vs C′ 对照时才做。目前 bopgrasp 只是「环境复现正确」的旁证,不参与任何结论。
- RefSpatial 只跑了 1 次(分歧本就极少:sft↔p4 共 4 个、p4↔cp 共 3 个)。
- 未做 P2 的"强制透传上界检查"。

---

## 9. 复算

```bash
# 健康门禁
python parse_dump.py --strict /workspace/exp/p0/<arm>/<run>/<bench>/0.jsonl

# 主表 + 翻转带 + 逐样本配对
python /root/flips.py        # 稳定核 McNemar + 三次区间
python /root/paired.py       # 三次配对合并 + sd
python /root/split2.py       # VQA / Vacant 拆分
python /root/p2.sh           # 透传 / 改点表
```

脚本全部只吃 dump,零 GPU。
