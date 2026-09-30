# SFT schema 口径比对结果

> 比的是:`run_sft.sh` 拼进 SFT system prompt 的那段 tool schema 文本,
> 与 **P4 eval 时模型真正看到的**那段,是否逐字一致。
> 数据来源:`spacetools-repro/p4/dumps/run1/blinkdepth/0.jsonl` 第 0 行的 `input` 字段。
> 日期:2026-09-06

---

## 结论:**不一致,两处。用 `toolshed_v1_config.p4.yaml`,不要用仓库里的静态版。**

| | 仓库静态版<br>`SpaceTools-RL/examples/toolshed/toolshed_v1_config.yaml` | **P4 实际** |
|---|---|---|
| schema 条数 | 11 | 11 ✓ |
| 方法名集合 | 相同 | 相同 ✓ |
| **顺序** | 见下 | **不同** ❌ |
| **`vlm.detect_one` / `detect_all` 的 description** | 缺一行 | **多 169 字符 × 2** ❌ |
| `<tools>` 块字节数 | 8257 | **8595** |
| sha256(前 16) | `436221881e80cacd` | `72c71c806f64e162` |

---

## 差异 1:`vlm` 的两个 schema 少了变量说明(**这条有实质影响**)

P4 的 `vlm.detect_one` / `vlm.detect_all` 描述里比仓库版多这么一行:

```
Stored variables: Coordinates in ${obj_name}_detections variable of list of (x, y)
point coordinates in normalized pixel space [0, 1] for use in subsequent operations.
```

**成因**:工具 docstring 里用了 `[[if:vars]] … [[/if:vars]]` 条件块,渲染结果取决于
`no_output_vars`。`run_eval.sh` 给 `vlm` 配的是 `no_output_vars: False`,所以这一行在。
仓库里那份静态 YAML 是在别的配置(或更早的 Toolshed)下导出的,没有它。

> 对照:`roborefer` 的两个 schema **两边完全相同** —— 因为
> `roborefer.py` 里硬写了 `print("WARNING: RoboRefer forcing no_output_vars to True")`,
> 它的 vars 行永远不出现,所以没有分歧。

**为什么这条不能忽略**:P4 的轨迹里模型确实在用
`$red_point_under_label_A_detections` 这种变量 —— 它知道有这些变量,**就是从这一行学来的**。
若 SFT 用仓库静态版,等于训练时的 prompt 从没提过这些变量、eval 时的 prompt 却提了。
**训练/推理口径不一致,而且照例不会报错。**

## 差异 2:顺序不同

```
仓库:  vision_ops · bounding_box · sam2×2 · depth×2 · grasp · roborefer×2 · vlm×2
P4  :  vision_ops · bounding_box · sam2×2 · grasp · vlm×2 · depth×2 · roborefer×2
```

集合相同、内容(除 vlm 外)相同,只是排列不同 —— 大概率反映的是导出那一刻
actor 的就绪顺序。**但它照样改变 system prompt 的字节,所以仍算不一致。**

---

## 一条好消息:**包裹文本完全对得上**

`run_sft.sh` 构造的 `# Tools` 段落头尾,与 verl chat template 渲染出来的
**逐字相同**(已从 dump 的 `input` 前后各 400 字符核对):

```
\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n
\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>
… schemas …
</tools>\n\nFor each function call, return a json object with function name and arguments
within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": …, "arguments": …}\n</tool_call>
```

**所以 `run_sft.sh` 的拼装逻辑没问题,唯一过期的是那份 YAML 里的 schema 列表。**
换掉 YAML 就够了,脚本一行不用改。

---

## 处置

用本目录下的 **`toolshed_v1_config.p4.yaml`**:

```bash
VERSION=v1 \
TOOL_CONFIG=/path/to/toolshed_v1_config.p4.yaml \
    bash scripts/spacetools/run_sft.sh
```

它是**从 P4 dump 的 `input` 反解出来的**,已验证:按 `run_sft.sh` 的方式重建后,
`<tools>` 块与 P4 逐字节相同(8595 字符,sha `72c71c806f64e162`)。
完整 `# Tools` 段 **8952 字符** —— 与方法笔记里记的「约 8.9 KB」对得上。

格式是 `run_sft.sh` 需要的最小形态(它只读 `item["tool_schema"]`):

```yaml
tools:
  - tool_schema: {...}
  - tool_schema: {...}
```

---

## 复核方式

```bash
python3 diff_tool_text.py toolshed_v1_config.repo.yaml toolshed_v1_config.p4.yaml
# 退出码 1 + 列出差异 = 复现本文结论
```

`diff_tool_text.py` 逐字复刻了 `run_sft.sh` 的构造逻辑,比的是**最终进 prompt 的字符串**,
不是 YAML 的行 —— 两份 YAML 的缩进和键序必然不同,裸 `diff` 全是噪声。

---

## 追加发现:**P4 自己的 15 次运行里,schema 顺序有三种**

把全部 15 份 dump 的 `<tools>` 块都抽出来比 sha:

```
run1/blinkdepth      8595  72c71c806f64      run1/refplacement    8595  72c71c806f64
run1/bopgrasp        8595  72c71c806f64      run1/refunseen       8595  72c71c806f64
run1/boppose         8595  0c6f645a3ebe  ←   run1/robospatial     8595  72c71c806f64
run1/cvb2drelation   8595  0c6f645a3ebe  ←   run2/*  run3/*       8595  72c71c806f64
run1/cvb3ddepth      8595  72c71c806f64      run4/blinkdepth      8595  4f58f1391956  ←
run1/reflocation     8595  0c6f645a3ebe  ←
```

**三种 sha,但字节数全是 8595、schema 全是 11 条 —— 已验证是同一组 schema 的不同排列**,
内容一个字都没变。变的只有 `grasp_generator` / `vlm` / `depth_estimator` 这三者的相对位置:

```
72c7… : … sam2×2 · grasp · vlm×2 · depth×2 · roborefer×2
0c6f… : … sam2×2 · vlm×2 · grasp · depth×2 · roborefer×2
4f58… : … sam2×2 · grasp · depth×2 · vlm×2 · roborefer×2
```

`vision_ops` / `bounding_box` / `sam2` 永远在最前(加载最快),
`roborefer` 永远在最后(8B、4 个 shard,最慢)——
**这就是 actor 就绪顺序,每次 Toolshed 重起重来一次。**

### 两条后果

**① SFT 该对齐哪一个顺序?—— 没有「哪一个」。**
eval 自己都没把顺序固定住。所以**内容对齐是必须的(vlm 那两行),顺序对齐是不可能的**。
做法:用 `toolshed_v1_config.p4.yaml` 的内容,顺序取其一并**记进 PROVENANCE**,
说明这是从多数派(15 份里 11 份)取的。

**② 这是 run-to-run 不确定性的第二个来源,此前没有被记录过。**
P2/P4 把同配置三次 108/107/111 归因于「sglang 的 batch 组成改变浮点归约顺序」。
**但 prompt 本身的 token 也在变** —— 三种排列是三串不同的 token,
注意力模式不同,足以推动低置信度 token 翻面。

> 注意 `run1/blinkdepth`(72c7…)与 `run4/blinkdepth`(4f58…)**排列不同** ——
> 而这两次正是那组 108 / 107 / 111 / … 重复实验的一部分。
> **此前把全部散布都算在浮点归约上,可能高估了那一项。**
> 这条不改变任何已发表的结论(散布带宽是实测的),但它给「为什么会散」多了一个候选解释,
> 而且是**可控的**:把 `TOOL_CONFIG` 固定成一份静态 YAML 喂给 eval,这个来源就消失了。

## 遗留

- `toolshed_v2_config.yaml`(17 工具)没查,若将来跑 v2 要重做一遍。
- 上面 ② 只是一个**可能的**来源,没有做控制实验证明它真的影响分数。
  要证:固定 `TOOL_CONFIG` 重跑 blinkdepth 三次,看散布是否收窄。成本约 3 × 5 分钟。
