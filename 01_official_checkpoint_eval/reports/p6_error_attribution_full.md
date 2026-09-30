# P6 错题归因(完整版)

> **范围**:SpaceTools 官方 checkpoint 在 P4 全量评测上的 **2121 个样本**。
> 准确率类七个 benchmark 共 **2001 个样本 / 322 个错题**,逐样本归到唯一一类;
> 连续判分的 `boppose` / `bopgrasp` 另 120 个样本单独处理。
>
> **本文不涉及 P7 / GFlowRL。** 对象是论文发布的那个 checkpoint,不重新训练。
>
> **底稿**:`01_official_checkpoint_eval/reports/p6_error_attribution_report.md`。本文的区别是:**每一条错题都被显式落到一个类上**
> (交叉表可复算),**每一类给两个真实样本**,数字全部从 `p4/parsed/` 当场重跑。
> 复算脚本见附录。

---

## 0. 先说三件影响怎么读这份表的事

**① 322 个错题 100% 是 clean。** OOM、工具响应截断、轮数耗尽、缺 `<answer>`、
畸形 tool call、幻影变量 —— 在错题上命中数**全部为 0**:

```
benchmark          n      对     错    正确率  | 工具失败 OOM 截断 顶轮 无ans 幻影 || clean 错
blinkdepth       124     107     17   86.29%  |     0    0   0    0    0    0  ||   17
cvb2drelation    650     615     35   94.62%  |     0    0   0    0    0    0  ||   35
cvb3ddepth       600     579     21   96.50%  |     0    0   0    0    0    0  ||   21
reflocation      100      54     46   54.00%  |     0    0   0    0    0    0  ||   46
refplacement     100      58     42   58.00%  |     0    0   0    0    0    0  ||   42
refunseen         77      37     40   48.05%  |     0    0   0    0    0    0  ||   40
robospatial      350     229    121   65.43%  |     0    0   0    0    0    0  ||  121
合计            2001            322                                            ||  322
```

**没有一个错题有基础设施上的借口。** 这一条决定了下面整张表的性质:它归的是能力,不是事故。

**② 「工具错」这个标签比「模型无能」宽。** 它包含三种机制,不该一律读成「深度/检测模型估不准」:
真的估错了、点落在物体上但那个像素属于别的东西、以及**两个物体本来就几乎没有裕度**。
§2 会把它拆开量。

**③ 每一类都给两个样本的完整轨迹。** 引用格式统一为:

```
turn N  THINK  模型这一轮的 <think> 原文(过长时用 … 截断,不改写)
        CALL   工具名({参数原文})
        RESP   工具返回原文(过长时用 … 截断)
        ANSWER <answer> 里的内容
```

全部取自 `p4/parsed/<benchmark>.jsonl` 的 `trajectory` 字段,**原文引用、不翻译、不润色**。
只有两处加工:过长处截断(标 `…`)、以及在行尾加中文批注(标 `←`)。
**轨迹里常有计数表看不见的东西** —— 模型是否察觉了工具坏了、它把「后方」翻译成了什么、
它有没有写下自己需要深度却没去取。下面每一类的分析都会指出那一句。

**④ 3b 的归属有争议,两种读法都报。** 若把「坐标系/语义不匹配」算进推理错,
工具错 : 推理错 = **241 : 33 ≈ 7.3 : 1**;若单列(本文的做法),是 **241 : 19 ≈ 12.7 : 1**。
板子打在哪一侧,取决于你认为策略该不该自己去补深度。

---

## 1. 总表:错因 × 题数 × benchmark

**322 / 322 全部归类,零未覆盖、零重复。**

| # | 错因 | 代码 | n | 占 322 | 分布(按 benchmark) |
|---|---|---|--:|--:|---|
| 1 | **工具错 —— 检测定位不准** | `1` | **197** | 61.2% | robospatial 45 · reflocation 44 · refplacement 42 · refunseen 40 · cvb2drelation 26 |
| 2 | **工具错 —— 深度估计不准** | `1` | **35** | 10.9% | cvb3ddepth 21 · blinkdepth 14 |
| 3 | **工具集缺口**(`fit`:没有工具给自由空间) | `5` | **32** | 9.9% | robospatial 32 |
| 4 | **推理错** | `3` | **19** | 5.9% | robospatial 15 · cvb2drelation 3 · blinkdepth 1 |
| 5 | **坐标系 / 语义不匹配** | `3b` | **14** | 4.3% | robospatial 14 |
| 6 | **工具错 —— 检测退化**(不同查询返回同一点) | `1a` | **9** | 2.8% | robospatial 4 · cvb2drelation 2 · reflocation 2 · blinkdepth 1 |
| 7 | **该调没调** | `2a` | **11** | 3.4% | robospatial 11(其中 front/behind 未调深度 8 · 只检了主体 3) |
| 8 | **二维投影不可分**(信息不足) | `2D` | **2** | 0.6% | cvb2drelation 2 |
| 9 | **参数错**(查询串写错) | `2c` | **1** | 0.3% | cvb2drelation 1 |
| 10 | **标注 / 指代歧义** | `6` | **1** | 0.3% | cvb2drelation 1 |
| 11 | **格式错**(答案不在选项集) | `4` | **1** | 0.3% | blinkdepth 1 |

**三个大类合起来是 241 + 32 + 19 = 292,占 90.7%。**

```
工具错合计(1 + 1a)  197 + 35 + 9 = 241   74.8%
推理错(3)                          19    5.9%
工具错 : 推理错 = 12.7 : 1
```

> 论文附录 Table 16 在 grasp 上给的是 30 例中 23 : 7。
> **我们的比例更偏工具侧,而且是在大十倍的样本上。**

**换一个切法 —— 按 benchmark 看每一类占它自己错题的比例:**

| benchmark | 错题 | 工具错 | 工具集缺口 | 推理错 | 3b | 2a | 其它 |
|---|--:|--:|--:|--:|--:|--:|--:|
| reflocation | 46 | **46** | — | — | — | — | — |
| refplacement | 42 | **42** | — | — | — | — | — |
| refunseen | 40 | **40** | — | — | — | — | — |
| cvb3ddepth | 21 | **21** | — | — | — | — | — |
| cvb2drelation | 35 | **28** | — | 3 | — | — | 4 |
| blinkdepth | 17 | **15** | — | 1 | — | — | 1 |
| robospatial | 121 | **49** | **32** | 15 | **14** | **11** | — |

> **除 `robospatial` 外,每一个 benchmark 的错题都被工具错主导(80–100%)。**
> 全部五个非零的「推理错 / 3b / 2a / 工具集缺口」里,**有 72/77 落在 `robospatial` 一家**。
> 这不是巧合:`robospatial` 是唯一一个「图像平面两点比较」这条规则不成立的 benchmark(§5)。

---

## 2. 错因 1:工具错 —— 检测定位不准(197 条,61.2%)

### 2.1 怎么判的

两条判据,都不需要看图:

- **pointing 题**(`reflocation` / `refplacement` / `refunseen` / `robospatial` Vacant):
  模型的答案是不是 RoboRefer 返回点的**原样透传**?是 → 模型没做任何加工,错只能是工具的。
  实测 **276/276 原样透传,一次改动都没有**(三个 RefSpatial);Vacant 上 104/122 透传。
- **关系题**(`cvb2drelation`):模型是不是在执行「比 x / 比 y」这条规则?
  遵守规则却答错 → 检测点不准。该判据的自洽率(在**答对**的样本上规则与模型是否一致)是 **99.83%**,
  规则对 GT 的准确率 **95.4%** —— 图像平面就是 CVBench 2D relation 的正确语义,所以这一判可用。

### 2.2 分析

**这一档是全部错题的主体,而且它的主体是 RoboRefer。** 三个 RefSpatial 上 126 条、
`robospatial` Vacant 45 条、`cvb2drelation` 26 条,合计 197 条全部由检测点的位置决定。

有一个尖锐的推论:在 277 个 RefSpatial 样本上,**策略相对于「直接调 RoboRefer」的全部贡献
就是怎么措辞那个 `obj_name`** —— 它不做任何后处理。论文 Table 2 里 RoboRefer-8B-SFT 单跑
RefSpatial 是 48.37,SpaceTools-3B 是 53.07,**那 +4.7 全部来自查询构造,没有一分来自推理。**

间距分布也支持这个判法:`cvb2drelation` 上答对组的判定轴间距中位 **0.410**,
遵守规则却答错组 **0.192** —— 错的那批点本来就挤在一起。

### 2.3 样本 A —— `cvb2drelation #10`:规则执行得没错,点错了

> *Considering the relative positions of the **rug** (annotated by the red box) and the **pillow** …
> where is the rug located with respect to the pillow?*   `(A) above  (B) below`   **GT = A**

**轨迹全文**(链路 `roborefer×2@2t`):

```
turn 1  THINK  Looking at this image, I need to identify the positions of the rug (marked by
               the red box) and the pillow to determine their spatial relationship. Let me
               detect the locations of these objects:
        CALL   roborefer.detect_one({"obj_name": "rug in red box", "image_index": 0})
        RESP   Detected 1 instance(s) of 'rug in red box': [(0.377, 0.794)]
        CALL   roborefer.detect_one({"obj_name": "pillow", "image_index": 0})
        RESP   Detected 1 instance(s) of 'pillow': [(0.552, 0.42)]

turn 2  THINK  Based on the detection results: - The rug (in the red box) is located at
               coordinates (0.377, 0.794) - The pillow is located at coordinates (0.552, 0.42)
               The vertical coordinate (v-value) indicates the vertical position in the image:
               - Rug: v = 0.794 (lower in the image) - Pillow: v = 0.42 (higher in the image)
               Since the rug has a higher v-value (0.794) compared to the pillow (0.42),
               the rug is positioned lower in the …
        ANSWER B
```

**模型把两个 y 比了大小,比对了** —— 轨迹里那一段推理是教科书式的,一步没错。
而 GT 说地毯在枕头**上方** —— 在图像平面语义下这只能意味着
两个检测点里至少有一个不在它该在的地方(地毯是一大片区域,RoboRefer 给的是其中一个点)。
**推理侧无可指摘。**

### 2.4 样本 B —— `refplacement #0`:模型就是一根管子

> *Please point out the **free space on the top of the wooden shelf and beside the picture frame**.*

**轨迹全文**(链路 `roborefer×1@2t`,`score = 0.0`):

```
turn 1  THINK  I need to find the free space on the top of the wooden shelf and beside the
               picture frame. The image shows a wooden shelf on the right side and a picture
               frame on the wall. I will use the roborefer.detect_one tool to locate these
               objects and then find the free space around them.
        CALL   roborefer.detect_one({"obj_name": "free space on the top of the wooden shelf
                                     and beside the picture frame", "image_index": 0})
        RESP   Detected 1 instance(s) of 'free space on the top of the wooden shelf and
               beside the picture frame': [(0.816, 0.485)]

turn 2  THINK  I have found the free space on the top of the wooden shelf and beside the
               picture frame at (0.816, 0.485). This point is within the normalized pixel
               space of the image. Now, I can format this point as a list of tuples as
               requested.
        ANSWER [(0.816, 0.485)]
```

**注意 turn 1 的 THINK 里那句 "and then find the free space around them" —— 它本来打算做第二步,
然后 turn 2 直接把返回值抄了。** 模型做的唯一一件事,是把题面整句塞进 `obj_name`,
再把返回值原样抄进 `<answer>`。
链路 `roborefer×1@2t`,没有第二次调用、没有校验、没有后处理。
这 42 条 `refplacement` 错题**每一条都是这个形状**。

> 所以这一类的 headroom 不在策略上。P6 §6.2 实测:把 RoboRefer 整体换成 Molmo-7B-D
> **更差 −14.61 pp**,但逐样本 oracle 选工具 **+10.83 pp** —— 动作是**补上**(router / ensemble),
> 不是换掉。

---

## 3. 错因 2:工具错 —— 深度估计不准(35 条,10.9%)

### 3.1 怎么判的

`blinkdepth` 与 `cvb3ddepth` 走同一条链(`depth_estimator×1 + roborefer×2 + vision_ops×2`),
题目问哪个更近。规则:**选测得深度更小的那个**。

| benchmark | 可判定 | 遵守规则 | 遵守却答错 → 工具错 | 违反且答错 → 推理错 |
|---|--:|--:|--:|--:|
| `blinkdepth` | 114/124 | 109(95.6%) | **14** | **1** |
| `cvb3ddepth` | 599/600 | 598(**99.8%**) | **21** | **0** |

**`cvb3ddepth` 的 21 个错题 100% 是 DepthPro 的错,推理侧零错误。**

### 3.2 分析:这 35 条里有一半根本没有裕度

按相对间距 `|d_A − d_B| / min(d_A, d_B)` 拆开这 35 条:

| 相对间距 | 条数 | 读法 |
|---|--:|---|
| **< 10%(近平局)** | **10** | 更像「没有裕度」而非「估错」 |
| 10 – 50% | 17 | |
| **> 50%(自信地反了)** | **8** | |

对照更有说服力:**答对样本的相对间距中位 83.3%,这 35 条错题中位 25.3%。**
**大多数深度错题发生在两个物体本来就很接近的地方。**

> 这条直接给「换深度工具」定了界:**略强一点的工具能翻回近平局那 10 条,
> 吃掉自信错的那 8 条需要显著更强的模型。**

还有一个天然对照:两个 benchmark 同链、同规则、推理保真度都接近 100%,而分数差十个点——
**那十个点纯粹是 DepthPro 在不同图像上的准确率差异**,与编排、与策略无关。

### 3.3 样本 A —— `cvb3ddepth #86`:自信地反了(相对间距 91%)

> *Which object is closer to the camera, the **lamp** (red box) or the **refrigerator** (blue box)?*
> `(A) lamp  (B) refrigerator`   **GT = A**

**轨迹全文**(链路 `depth_estimator×1 + roborefer×2 + vision_ops×2@3t`):

```
turn 1  THINK  I need to determine which object is closer to the camera by analyzing the depth
               information. Let me follow the systematic process: First, I'll estimate the depth
               map of the image, then locate both objects and compare their depth values.
        CALL   depth_estimator.estimate_depth({"image_index": 0})
        RESP   Estimated depth map for 1036x756 image. Focal length: 920.9px.
               Depth range: 1.84m to 16.34m (mean: 5.86m). Use $depth_map …
        CALL   roborefer.detect_one({"obj_name": "lamp (highlighted by a red box)", …})
        RESP   Detected 1 instance(s) of 'lamp (highlighted by a red box)': [(0.368, 0.144)]
        CALL   roborefer.detect_one({"obj_name": "refrigerator (highlighted by a blue box)", …})
        RESP   Detected 1 instance(s) of 'refrigerator …': [(0.6, 0.469)]

turn 2  THINK  Now I'll get the depth values at both detected locations:
        CALL   vision_ops.index_at({"data": "$depth_map", "u": 0.368, "v": 0.144})
        RESP   Pixel value at (0.368, 0.144) is 4.53564453125
        CALL   vision_ops.index_at({"data": "$depth_map", "u": 0.6, "v": 0.469})
        RESP   Pixel value at (0.600, 0.469) is 2.378054141998291

turn 3  THINK  Based on the depth analysis: - Lamp (red box): depth = 4.54 meters
               - Refrigerator (blue box): depth = 2.38 meters The refrigerator has a smaller
               depth value (2.38m) compared to the lamp (4.54m), which means the refrigerator
               is closer to the camera.
        ANSWER B
```

**这条轨迹是这套系统「正确工作」时的标准形状** —— 三轮、五次调用、一句比大小。
两个检测点也都对(审阅卡看图核实过:红框是画面顶部的吸顶灯,`(0.368, 0.144)` 落在框内)。

**模型做的全部事情就是把 4.536 和 2.378 比大小,而它比对了。**
红框框住的是画面顶部的一盏**吸顶灯**,物理上离相机很近;GT 说它更近,意味着实际 < 2.378 m,
而 DepthPro 报了 4.536 m —— **相对间距 91%,不是差一点,是差了近两倍且方向反了。**
单目深度在纹理稀少的天花板灯具上把「投影得小」读成「远」,是它已知的失效模式。

### 3.4 样本 B —— `blinkdepth #5`:近平局,谁来都得抛硬币

> *Which point is closer to the camera?*   `(A) A is closer  (B) B is closer`   **GT = A**

**轨迹全文**(链路 `depth_estimator×1 + roborefer×2 + vision_ops×2 + vlm×2@4t`):

```
turn 1  THINK  Looking at the image, I can see: - Point A is labeled near the top center,
               appearing to be on an orange in the background - Point B is labeled in the upper
               right area, also on an orange in the background … Both points appear to be on
               background oranges, but it's not immediately obvious which is closer. I need to
               use tools to determine t…
        CALL   depth_estimator.estimate_depth({"image_index": 0})
        RESP   … Depth range: 0.45m to 1.16m (mean: 0.66m) …
        CALL   roborefer.detect_one({"obj_name": "a", …})   RESP  [(0.315, 0.045)]
        CALL   roborefer.detect_one({"obj_name": "b", …})   RESP  [(0.635, 0.106)]

turn 2  THINK  Now let me get the depth values at these two points:
        CALL   vision_ops.index_at({"data": "$depth_map", "u": 0.315, "v": 0.045})
        RESP   Pixel value at (0.315, 0.045) is 0.8774694800376892
        CALL   vision_ops.index_at({"data": "$depth_map", "u": 0.635, "v": 0.106})
        RESP   Pixel value at (0.635, 0.106) is 0.8658632040023804

turn 3  THINK  The depth values are very close: - Point A: 0.877m - Point B: 0.866m
               The difference is only about 0.01m, which is within the margin where depth
               estimation might not be reliable. According to the guidelines, when depth values
               are too close (within 0.5m), I should reconsider the visual appearance and use
               the VLM detector for verification. Let me try the VLM detector:
        CALL   vlm.detect_one({"obj_name": "a", …})   RESP  [(0.378, 0.055)]
        CALL   vlm.detect_one({"obj_name": "b", …})   RESP  [(0.638, 0.052)]

turn 4  THINK  … Point A appears to be on an orange that is slightly more forward in the scene,
               while Point B is on an orange that is very close to the background. The depth
               values being so close (within 0.5m) makes …
        ANSWER B
```

**两个点的深度差 1.16 厘米,而整幅图的深度跨度只有 71 厘米(相对间距 1.3%)。**
这不是「DepthPro 估错了」,是**这道题在这个工具的分辨率下不可判**。

**而轨迹给出了一个计数表看不见的事实:模型自己发现了这一点。**
turn 3 它明确写出「差值只有约 0.01 m,落在深度估计可能不可靠的范围内」,
并按系统提示里的规则改调 `vlm` 复检、最后转成视觉判断 —— **它做了能做的全部动作,还是错了。**

> 把它和 `#86` 放在同一个「工具错」标签下,是标签的粗糙,不是结论的错误 ——
> **所以 §3.2 那张间距表必须和计数一起引用。** 顺带一提,判据把这一条计为「遵守规则」
> 是因为它最终的答案(B)与「选更小的」一致;**但轨迹显示它并不是靠那条规则得到 B 的。**
> 这是判据的已知误差方向(自洽率 95.6% 的那 4.4%),这里正好抓到一例。

---

## 4. 错因 3:工具集缺口 —— `fit` 题(32 条,9.9%)

### 4.1 怎么判的

不靠判据,靠**工具直方图**:题目需要的量,模型从来没有去取过,而且**没有任何单个工具能返回它**。

```
robospatial VQA 的 fit 题   n = 105   正确率 73/105 = 69.5%
                            **未调用 depth_estimator 105/105**
                            GT = no 的正确率  4/18 = 22%
                            模型答 yes 83/105
```

### 4.2 分析

问的是「放不放得下」,而工具返回的是「在哪」。**`bounding_box` 能给物体尺寸、`depth_estimator`
能给深度,但没有任何工具返回「某个方位上有多少自由空间」** —— 这个量要靠组合多个工具再算,
而模型从没这么做过。

后果非常整齐:**模型在 fit 题上几乎总是答 yes**(105 题答了 83 次),
所以 GT = yes 的题大半蒙对,GT = no 的题塌到 22%。
P5 记过的「GT 为 `no` 的题停在随机水平」,**拆开看就集中在这里**
(fit 22% vs 关系题 58% vs front/behind 64%)。

> **这条路指向加工具或加组合能力,不是调 prompt、不是换 pointing 工具。**
> 只要 fit 题不再一律答 yes,`robospatial` VQA 相对论文的 6.46 pp 缺口就填平了。

### 4.3 样本 A —— `robospatial #128`:典型形状(该答 no,答了 yes)

> *Can the **cup** fit **in front of** the **speaker**?*   **GT = No**,模型答 **Yes**

**轨迹全文**(链路 `roborefer×2@2t`):

```
turn 1  THINK  I need to analyze whether the cup can fit in front of the speaker. To do this,
               I should: 1. Locate the cup in the image 2. Locate the speaker in the image
               3. Assess the spatial relationship and available space
               Let me detect both objects first.
        CALL   roborefer.detect_one({"obj_name": "cup", "image_index": 0})
        RESP   Detected 1 instance(s) of 'cup': [(0.78, 0.494)]
        CALL   roborefer.detect_one({"obj_name": "speaker", "image_index": 0})
        RESP   Detected 1 instance(s) of 'speaker': [(0.154, 0.491)]

turn 2  THINK  Now I have the positions: - Cup is at (0.78, 0.494) - this is on the right side
               of the image - Speaker is at (0.154, 0.491) - this is on the left side …
               The question asks if the cup can fit "in front o…
        ANSWER Yes
```

**turn 1 的 THINK 把计划写得很清楚:第 3 步是 "Assess the spatial relationship and
available space"。然后它只拿到了两个点位,第 3 步无从做起,turn 2 直接作答。**

没有任何工具告诉过它:speaker 前面有多少空地、cup 有多大。
拿到的是「在哪」,问的是「放不放得下」——**它需要的量根本没有工具返回。**

### 4.4 样本 B —— `robospatial #122`:反方向也错,说明不是单纯的 yes 偏置

> *Can the **tissue box** fit **behind** the **chair**?*   **GT = Yes**,模型答 **No**

**轨迹全文**(链路 `roborefer×2@2t`):

```
turn 1  THINK  I need to analyze whether the tissue box can fit behind the chair. To do this,
               I should: 1. Locate the tissue box 2. Locate the chair 3. Assess the spatial
               relationship and available space behind the chair
        CALL   roborefer.detect_one({"obj_name": "tissue box", …})  RESP  [(0.306, 0.531)]
        CALL   roborefer.detect_one({"obj_name": "chair", …})       RESP  [(0.477, 0.517)]

turn 2  THINK  Now I have the positions: - Tissue box: (0.306, 0.531) … - Chair: (0.477, 0.517)
               … "Behind the chair" from the camera's perspective would mean further away from
               the c…
        ANSWER No
```

**逐字相同的计划(连 THINK 的措辞都几乎一样),同样只有两个点位,这次它答了 no,一样错。**

把这两条放在一起才准确:**不是「模型有个 yes 偏置所以错」,而是「它在瞎猜,
而猜的先验偏 yes」** —— 先验帮它在 GT=yes 的 87 题上蒙到不少,在 GT=no 的 18 题上塌到 4/18。
**两个方向上它都没有可依据的量。**

---

## 5. 错因 4:推理错(19 条,5.9%)

### 5.1 怎么判的

反过来用前面两条判据:**违反规则**(深度题没选更小的、关系题没按坐标比)
或**自行改动工具的输出**(pointing 题不透传),且答错。

分布:`robospatial` Vacant 15 · `cvb2drelation` 3 · `blinkdepth` 1。
**15/19 集中在 Vacant 一处。**

### 5.2 分析:它不是在算,是在目测

Vacant 上有一条极干净的统计:

| | n | 正确率 | 两个坐标都是 0.05 的整数倍 |
|---|--:|--:|--:|
| 原样透传 | 104 | **54.8%** | 2 条(**2%**) |
| 模型自行改点 | 18 | **27.8%** | 6 条(**33%**) |

**一改就减半,而且 0.05 整数倍的比例差了 16 倍。**

透传时坐标带着工具的三位小数(`0.271, 0.870`);一旦模型自己动手,就变成
`(0.40, 0.70)`、`(0.50, 0.85)` 这种**眼估的整数**。
**所以「推理错」在 Vacant 上的具体含义是:它不是在算,是在目测。**
而 Vacant 的 GT 是一条几十像素宽的窄带,目测到 0.05 的粒度不够用。

### 5.3 样本 A —— `robospatial #11`:方向对了,距离没拿准

> *In the image, there is a **cup**. Pinpoint one point within the **vacant space** suitable for
> placing an object **close to and to the right of** the cup.*

**轨迹全文**(链路 `roborefer×1@2t`,**整个样本只有一次工具调用**):

```
turn 1  THINK  I need to find a vacant space that is close to and to the right of the cup.
               First, I should locate the cup in the image to understand its position, then
               identify an appropriate vacant space to its right. Looking at the image, I can
               see a white mug/cup on the desk in the lower left portion of the image.
               Let me detect its exact location first.
        CALL   roborefer.detect_one({"obj_name": "cup", "image_index": 0})
        RESP   Detected 1 instance(s) of 'cup': [(0.271, 0.87)]   ✓ 精准落在桌上那只杯子上

turn 2  THINK  The cup is located at approximately (0.271, 0.87), which is in the lower left
               area of the desk. Now I need to find a vacant space that is: 1. Close to the cup
               2. To the right of the cup (higher x-coordinate) Looking at the desk surface,
               to the right of the cup (x > 0.271), there appears to be vacant desk space.
               The desk extends to the right with some items like a keyboard and mouse further
               right. A suitable vacant sp…
        ANSWER [(0.50, 0.85)]
```

```
GT 是 23 个可接受点组成的一片区域 —— 杯子右边那条桌面窄带:
  x 范围 0.327 – 0.558      y 范围 0.878 – 0.989
```

**turn 2 的 THINK 里没有一个算式。** 它说「看桌面、杯子右边好像有空位」,
然后给了一个数 —— **这就是「目测」在轨迹里的样子。**

**它没有照抄工具给的点,自己动了手,而且看清楚是这样错的:**
`x = 0.50` 落在 GT 范围内(0.327–0.558),**方向完全正确;差的是 y** ——
GT 的 y 全部 ≥ 0.878,模型给 0.85,**高了约 0.03,落在那条窄带的上沿之外,擦边没中。**
**它知道往右,没拿准往前多少。** 两个坐标都是 0.05 的整数倍。

### 5.4 样本 B —— `robospatial #3`:同样的形状,差得更远

> *In the image, there is a **speaker**. Pinpoint one point within the vacant space suitable for
> placing an object **close to and behind** the speaker.*

**轨迹全文**(链路 `roborefer×1@2t`):

```
turn 1  THINK  I need to find a vacant space that is close to and behind the speaker. First, I
               should detect where the speaker is located … Looking at the image, I can see a
               white speaker with a mesh/grill design on the left side of the desk. There's
               also a monitor, some papers, a mouse, and a cup visible …
        CALL   roborefer.detect_one({"obj_name": "speaker", "image_index": 0})
        RESP   Detected 1 instance(s) of 'speaker': [(0.154, 0.491)]

turn 2  THINK  The speaker is detected at position (0.154, 0.491) … Behind the speaker (to the
               right in the image) would be the area where the monitor and other desk items are
               located. To find a vacant space that is: - Close to the speaker - Behind it
               (to the right in the image) I should look a…
        ANSWER [(0.40, 0.70)]
```

```
GT 区域(10 个点):x 0.251 – 0.400   y 0.416 – 0.455
```

**注意 turn 2 自己写的定义:「behind = to the right in the image」。**
它把「后方」翻译成了图像里的右边 —— x = 0.40 因此正好压在 GT 的右边界上,**方向判对了**;
而 y 它给了 0.70,GT 的 y 最大才 0.455,**差了 0.25**。

又是两个 0.05 的整数倍。

**两条放在一起就能看出机制:模型对「哪个方向」判断得不错,对「多远」完全靠估,
而 Vacant 的判分恰恰是对后者敏感的。**

---

## 6. 错因 5:坐标系 / 语义不匹配(14 条,4.3%)

### 6.1 怎么判的

**这一类判据判不了,必须看图。** 触发条件是:两个检测点都对、规则也执行得没错,
但**那条规则不是这道题的语义**。

判据 C 给出了这一类存在的硬证据:

| | 可判定 | 自洽率 | **规则对 GT** | 模型对 GT |
|---|--:|--:|--:|--:|
| `cvb2drelation` | 627/650 | 99.83% | **95.4%** | 94.7% |
| `robospatial` VQA 关系题 | 87/87 | 98.55% | **79.3%** | 79.3% |

两边模型都在执行同一条规则(自洽率都 ≈99%),但**规则对 GT 的准确率差了 16 个点**。
`robospatial` 上规则对 GT **69/87**、模型对 GT **69/87**,**68/69 是同一批样本** ——
**不是接近,是相同:模型在这类题上就是一次两点坐标比较,上限就是那条规则的上限 79.3%。**

### 6.2 分析:两个亚型

**① 图像 y ≠ 3D 上方(11 条)。** 图像 y 这一个数**同时编码了「更高」和「更远」**,
而工具只返回二维点 —— 这个歧义在模型拿到的数据里无解。

**② 相机系 ≠ benchmark 的参考系(3 条)。** 左右题上出现系统性反转,
而且间距 0.40–0.59,**不可能是检测误差**。

### 6.3 样本 A —— `robospatial #283`(亚型 ①)

> *Is the **lamp** above the **bed**?*   **GT = No**,模型答 **Yes**

**轨迹全文**(链路 `roborefer×2@2t`):

```
turn 1  THINK  I need to determine if the lamp is positioned above the bed in the image …
               From my initial observation: - The lamp appears to be a tall floor lamp with a
               rectangular lampshade, positioned in the center-right area of the image
               - The bed is visible with bedding on the right side of the image
        CALL   roborefer.detect_one({"obj_name": "lamp", …})  RESP  [(0.561, 0.234)]  ✓ 落在落地灯上
        CALL   roborefer.detect_one({"obj_name": "bed",  …})  RESP  [(0.703, 0.775)]  ✓ 落在床上

turn 2  THINK  Now I have the coordinates: - Lamp: (0.561, 0.234) … - Bed: (0.703, 0.775) …
               In image coordinates, a lower y-value means higher up in the image … Since the
               lamp has y-coordinate 0.234 and the bed has y-coordinate 0.775, the lamp is
               indeed positioned above the bed **in the image** - it app…
        ANSWER Yes
```

**turn 1 的 THINK 里它自己就说了「a tall floor lamp」—— 它看出那是一盏落地灯。**
然后 turn 2 里它把问题换成了「in the image 里谁的 y 更小」,这一换就错了。

**两个点都对,规则也执行得没问题,但那条规则不是这道题的语义。**
落地灯**立在床边的地上**,不在床的上方;它在图像里更高,**只是因为它离相机更远**。

它既不是工具错(点是对的),也不是普通推理错(规则没被违反),所以单列。

### 6.4 样本 B —— `robospatial #347`(亚型 ②)

> *Is the **toilet** right of the **shelf**?*   **GT = Yes**,模型答 **No**

**轨迹全文**(链路 `roborefer×2@2t`):

```
turn 1  THINK  I need to determine if the toilet is to the right of the shelf in this bathroom
               image … From my initial observation: - I can see a toilet in the lower left
               portion of the image - There appears to be a wooden shelf unit on the right side
        CALL   roborefer.detect_one({"obj_name": "toilet", …})  RESP  [(0.215, 0.766)]
        CALL   roborefer.detect_one({"obj_name": "shelf",  …})  RESP  [(0.801, 0.709)]

turn 2  THINK  Now I have the coordinates: - Toilet: (0.215, 0.766) - x-coordinate is 0.215
               - Shelf: (0.801, 0.709) - x-coordinate is 0.801 In normalized coordinates, the
               x-axis goes from 0 (left) to 1 (right). Since the toilet's x-coordinate (0.215)
               is less than the shelf's x-coordinate (0.801), the toilet is to the LEFT of the
               shelf, not to the right. Therefore, the answer … is No.
        ANSWER No
```

**这段推理挑不出毛病 —— 它甚至在 turn 1 就目视确认了「马桶在画面左下、架子在右侧」,
和检测点一致。** 判定轴上的间距 0.586。

**间距 0.586,跨了大半张图 —— 不可能是检测误差。**
GT 说马桶在右,意味着这道题的「左右」不是相机看过去的左右
(`#295` microwave/fridge 间距 0.567、`#310` mouse/keyboard 间距 0.400,
**三条方向全部与相机系相反**)。

**这三条就是 §6.1 那个 79.3% vs 95.4% 的直接来源。**

---

## 7. 错因 6:工具错 —— 检测退化(9 条,2.8%)

### 7.1 定义

**RoboRefer 对不同的查询返回同一个点。** 它和「定位不准」不同:不准是给了一个错的答案,
退化是**根本没有在回答那个查询**。跨三个 benchmark 出现,共 9 条。

### 7.2 分析

值得注意的是**模型的应对:它会主动换 3–6 种措辞去绕**。这与「措辞是它唯一的手段」这条结论
互为佐证 —— 而这个失败模式恰好是**措辞打不穿的**。

还有一个放大效应:**一次坏检测可以污染多个样本**(下面样本 B)。
所以本文的「工具责任份额」是**按错题**而不是按**缺陷**计的 —— 同一个工具缺陷会被重复计入。

### 7.3 样本 A —— `cvb2drelation #621`:六种查询,同一个点

> *…where is the **bottle** (red box) located with respect to the **vase**?*  **GT = A**,模型答 **B**

**轨迹全文**(链路 `roborefer×6@6t` —— 六轮,全是同一件事):

```
turn 1  THINK  Looking at this bathroom image, I need to identify the bottle (in the red box)
               and the vase, then determine their relative positions.
        CALL   roborefer.detect_one({"obj_name": "bottle in red box"})  RESP  [(0.558, 0.594)]
        CALL   roborefer.detect_one({"obj_name": "vase"})               RESP  [(0.558, 0.594)]

turn 2  THINK  The detection results show both objects at the same location (0.558, 0.594),
               which seems unusual. Let me look more carefully at the image … Let me try
               detecting the bottle more specifically:
        CALL   roborefer.detect_one({"obj_name": "red bottle on counter"}) RESP [(0.558, 0.594)]

turn 3  THINK  Let me try detecting the soap or other items on the counter that might be in the
               red box area:
        CALL   roborefer.detect_one({"obj_name": "soap in red box"})    RESP  [(0.558, 0.594)]

turn 4  THINK  Looking at the image more carefully … Let me detect the items on the counter
               more specifically:
        CALL   roborefer.detect_all({"obj_name": "bottles on counter"}) RESP  [(0.558, 0.594)]

turn 5  THINK  Let me try detecting the small items that might be in the red box area:
        CALL   roborefer.detect_all({"obj_name": "small items on counter"})
                                                                RESP  [(0.654, 0.626)]

turn 6  THINK  Based on the visual analysis of the image, the red box is located on the bathroom
               counter in the center-left area … the most direct comparison is to the …
        ANSWER B
```

**turn 2 的第一句话就是证据:"both objects at the same location (0.558, 0.594), which seems
unusual"** —— **模型明确地识别出了工具坏了**,然后用四轮、四种措辞去绕,前五次拿到完全相同的坐标。
它**知道**结果有问题并且一直在尝试,**但没有任何手段能让工具改口**。
最后 turn 6 它放弃求证,改用「visual analysis」硬答。

### 7.4 样本 B —— `robospatial #273` + `#274`:一次坏检测,连错两题

> `#273` *Is the **orange** left of the **bottle**?*    **GT = Yes**,模型答 **No**
> `#274` *Is the **water pitcher** left of the **bottle**?*  **GT = Yes**,模型答 **No**

**两条轨迹并排**(都是 `roborefer×2@2t`):

```
#273
turn 1  THINK  I need to determine if the orange is to the left of the bottle … There's an
               orange on the counter - There's a water bottle (gray/silver colored) on the
               counter - Various other items like a paper towel holder, utensils …
        CALL   roborefer.detect_one({"obj_name": "orange"})  RESP  [(0.477, 0.552)]
        CALL   roborefer.detect_one({"obj_name": "bottle"})  RESP  [(0.082, 0.498)]  ← 错检
turn 2  THINK  … Since the orange's x-coordinate (0.477) is greater than the bottle's
               x-coordinate (0.082), the orange is to the RIGHT of the bottle …
        ANSWER No

#274
turn 1  THINK  I need to determine if the water pitcher is to the left of the bottle … 1. There's
               a water pitcher/jug on the left side of the counter … 2. There's a gray water
               bottle (Hydro Flask) in the center-right area …
        CALL   roborefer.detect_one({"obj_name": "water pitcher"})  RESP  [(0.266, 0.509)]
        CALL   roborefer.detect_one({"obj_name": "bottle"})         RESP  [(0.082, 0.498)]  ← 同一点
turn 2  THINK  … Since 0.266 > 0.082, the water pitcher is actually to the RIGHT of the bottle …
        ANSWER No
```

**`#274` 的 THINK 里模型自己写了「gray water bottle (Hydro Flask) in the center-right area」
—— 它看到的瓶子在画面中右,而工具返回的是 x = 0.082(最左)。**
两处认知直接打架,模型没有察觉,照着坐标算下去了。

`'bottle'` 被检到了 x = 0.082(画面最左侧,人工审阅认定是洗手液),
于是**这张图上凡是「X 在 bottle 左边吗」的题,全部会被判成 no。**
两道题,一个缺陷。

---

## 8. 错因 7:该调没调(11 条,3.4%)

### 8.1 定义

**工具表里有能直接给出答案所需量的工具,模型一次都没调。**
与「工具集缺口」(§4)的区别就在这里:那 105 道 `fit` 是**没有工具可调**,
这 11 条是**有工具却没调**。**前者是工具集的错,后者是模型的错。**

两个亚型:**front/behind 未调深度 8 条** · **只检测了主体、参照物从未获取 3 条**。

### 8.2 分析

最刺眼的一个数:**`robospatial` 全部 350 个样本,0 个调用过 `depth_estimator`** ——
而它就在该 benchmark 的工具表里。29 道 `front/behind` 问的就是深度序,**29 次一次都没调**,
正确率 72.4%。

这一类是**唯一一处「存在已知有效替代链、策略却从不走」的地方**,
也因此是奖励塑形最直接的靶子(上界约 +8 题)。

### 8.3 样本 A —— `robospatial #246`:只拿到了比较的一半

> *Is the **paper towel** above the **counter**?*   **GT = Yes**,模型答 **No**

**轨迹全文**(链路 `roborefer×1@2t`,**一共只有一次工具调用**):

```
turn 1  THINK  I need to determine if the paper towel is above the counter. The paper towel is
               located on the counter, and I need to check if it is positioned above the counter
               surface. Let me use the detection tool to locate the paper towel.
        CALL   roborefer.detect_one({"obj_name": "paper towel", "image_index": 0})
        RESP   Detected 1 instance(s) of 'paper towel': [(0.293, 0.481)]

turn 2  THINK  The detection shows that the paper towel is at coordinates (0.293, 0.481), which
               means it is positioned in the upper-left area of the counter. However, the
               question asks if the paper towel is "above" the counter. Looking at the image,
               the paper towel is sitting on top of the counter surface. The paper towel appears
               to be at a similar or slightly elevated position above the counter surface, but
               it's not actually "abov…
        ANSWER No
```

**turn 1 的 THINK 里它就写死了计划:「locate the paper towel」——
从头到尾没有打算去取 counter 的位置。** turn 2 拿一个点做了一次不可能成立的比较,
最后靠「Looking at the image」目测收尾。

**参照物 `counter` 的位置从未被获取。** 比较两个物体的上下关系,
却只拿到了其中一个的坐标 —— **另一半是猜的。**
`#272`(bottle / water)、`#301`(game controller / sofa)是同一形状。

### 8.4 样本 B —— `robospatial #230`:问深度序,却只拿两个二维点

> *Is the **speaker** in front of the **cup**?*   **GT = Yes**,模型答 **No**

**轨迹全文**(链路 `roborefer×2@2t` —— **没有调 `depth_estimator`**):

```
turn 1  THINK  I need to determine if the speaker is in front of the cup. To do this, I should:
               1. Locate the speaker in the image 2. Locate the cup in the image
               3. **Compare their positions to determine which is closer to the camera**
        CALL   roborefer.detect_one({"obj_name": "speaker"})  RESP  [(0.154, 0.491)]
        CALL   roborefer.detect_one({"obj_name": "cup"})      RESP  [(0.78, 0.494)]

turn 2  THINK  Now I have the positions: - Speaker: (0.154, 0.491) - This is on the left side of
               the image - Cup: (0.78, 0.494) - This is on the right side of the image …
               **To determine which is "in front of" the other, I need to consider depth.**
               The speaker appears to be sitti…
        ANSWER No
```

**它两次写下自己需要深度**(turn 1 第 3 步、turn 2 "I need to consider depth"),
**而 `depth_estimator.estimate_depth` 就在它的工具表里,它一次都没调。**
最后靠两个 x 坐标和一句「appears to be」作答。

「in front of」问的是**深度序**,而两个二维点里没有深度。
`depth_estimator.estimate_depth` 就在工具表里,一次调用就能拿到
(`cvb3ddepth` 上 600 道题全都是这么做的,正确率 96.5%)。
**它在这里没调,于是只能拿 x 坐标硬答。**

> 对照 §3 里 `cvb3ddepth` 的链路 `depth_estimator×1 + roborefer×2 + vision_ops×2` ——
> **同一个模型、同一批工具,在另一个 benchmark 上就会正确地摆出这三步。**
> 所以这不是能力缺失,是**编排在 `robospatial` 上塌到了 `roborefer×2` 一条链**。

---

## 9. 错因 8:二维投影不可分(2 条)

**两条全部列出。** 定义:两个检测点**都对**,但它们在判定轴上的差小到二维投影里
**不含判定所需的信息** —— 与 3b 的区别是这里不涉及语义错位,纯粹是信息量不够。

### 9.1 `cvb2drelation #412`

> *…where is the **person** (red box) located with respect to the **cell phone**?*  **GT = B**,模型答 **A**

```
turn 1  CALL   roborefer.detect_one({"obj_name": "person in red box"})  RESP  [(0.58, 0.545)]
        CALL   roborefer.detect_one({"obj_name": "cell phone"})         RESP  [(0.253, 0.539)]
turn 2  THINK  Comparing the vertical positions (v-coordinates): - Person in red box: v = 0.545
               - Cell phone: v = 0.539 Since the v-coordinate increases downward in image
               coordinates, a higher v-value means a lower position in the image …
        ANSWER A
```

**判定轴(y)上相差 0.006** —— 模型把 0.545 和 0.539 当成了一个可判的差。

### 9.2 `cvb2drelation #478`

> *…where is the **person** (red box) located with respect to the **bottle**?*  **GT = A**,模型答 **B**

```
turn 1  CALL   roborefer.detect_one({"obj_name": "person in red box"})  RESP  [(0.507, 0.431)]
        CALL   roborefer.detect_one({"obj_name": "bottle"})             RESP  [(0.503, 0.85)]
turn 2  THINK  Comparing the x-coordinates (horizontal position): - Person in red box: x = 0.507
               - Bottle: x = 0.503 The person in the red box has a slightly higher x-coordinate
               (0.507 > 0.503), meaning they are positioned **slightly** to the right …
        ANSWER B
```

**判定轴(x)上相差 0.004,而两者的 y 差了 0.42 —— 人和瓶子是竖直叠置的。**
模型自己都写了 "slightly",还是照着这个差下了判断。

**0.004–0.006 的间距在一张归一化到 [0,1] 的图上是 2–3 个像素。**
这两条既不能怪检测(点是对的),也不能怪推理(规则执行了),
**它们是「用二维点回答这道题」这个方案本身的边界。**

---

## 10. 错因 9–11:各 1 条(全部列出)

### 10.1 `2c` 参数错 —— `cvb2drelation #169`

> *…where is the **fluorescent tube** located with respect to the **tank**?*  **GT = A**,模型答 **B**

```
turn 1  THINK  Looking at this image, I need to identify the positions of the fluorescent tube
               and the tank … Let me detect both objects to get their precise locations:
        CALL   roborefer.detect_one({"obj_name": "fluentice tube"})   ← fluorescent 拼成 fluentice
        RESP   Detected 1 instance(s) of 'fluentice tube': [(0.84, 0.631)]
        CALL   roborefer.detect_one({"obj_name": "tank"})   RESP  [(0.532, 0.461)]

turn 2  THINK  Based on the detection results: - The fluorescent tube is located at coordinates
               (0.84, 0.631) - The tank is located at (0.532, 0.461) … the tube (0.631) is
               positioned lower/below the tank (0.461) in the image.
        ANSWER B
```

**THINK 里它写的是 "fluorescent tube",`obj_name` 里打成了 `fluentice tube`
—— 错发生在从思考到调用的那一步,而且它后面一直以为自己查的是荧光灯管。**

**模型把查询串写错了,而 RoboRefer 照样返回了一个点。**
工具没有任何「我不认识这个词」的信号 —— 于是**错误静默传播**,
后面的坐标比较是在一个无关的点上做的。

> 这是整份归因里唯一一条**纯粹的参数错**,但它指出的接口问题是普遍的:
> 检测工具对无意义查询不报错(对比 `reflocation #6`,那里工具明确返回了
> `Detected 0 instance(s)`,模型仍自行给了一个点 —— 那条被归到 `1a`)。

### 10.2 `6` 标注 / 指代歧义 —— `cvb2drelation #263`

> *…where is the **plant** (annotated by the red box) located with respect to the **plants**?*
> **GT = B**,模型答 **A**

```
turn 1  THINK  … I need to identify the plant in the red box and compare its position to other
               plants in the scene.
        CALL   roborefer.detect_all({"obj_name": "plant"})          RESP  [(0.855, 0.674)]
turn 2  THINK  Let me also detect the plant more specifically by looking at the red box area:
        CALL   roborefer.detect_one({"obj_name": "plant in red box"}) RESP [(0.845, 0.18)]
turn 3  THINK  … - The plant in the red box is at (0.845, 0.18) - in the upper right portion
               - The other plant detected is at (0.855, 0.674) - in the middle-right portion
               Comparing the x-coordinates … Since the red box plant has a smaller x-coordinate
               (0.845) compared to the other plan…
        ANSWER A
```

题面的选项是 `(A) left` / `(B) right`。两个点都对(红框内那株在上、前景那株在下),
而它们的 **x 相差 0.010** —— 模型据此判「红框内的略偏左」答 A,GT 说 B。
题面里的 `the plants`(复数)究竟指哪一株存疑,所以归到标注/指代歧义,
**不计入模型或工具的过失。**

> ⚠ **复核时发现人工判定的证据串写错了轴。** `p6/manual/verdicts.jsonl` 里这一条写的是
> 「图像上红框内的更高 ⇒ 规则给 above,GT=below」,**而这道题问的是 left/right,不是 above/below**
> (原始 `question` 字段:`(A) left\n(B) right`)。
>
> 分类结论本身不受影响(两点都对、GT 相反、指代存疑,仍是 `6`),
> **但按 0.010 的 x 间距看,它同样落在判据 C 的「规则退化」区间(阈值 0.02)之内** ——
> 也就是说它和 §9 那两条 `2D不可分` 的形状其实更接近。
> **本文按归档的分类计数,不重新裁决;但这条证据需要修。**

### 10.3 `4` 格式错 —— `blinkdepth #69`

> *Which point is closer to the camera?*   `(A) A is closer  (B) B is closer`   **GT = B**

**轨迹全文**(链路 `depth_estimator×1 + roborefer×3 + vision_ops×3@3t`):

```
turn 1  THINK  Looking at the image, I can see three children playing with a bicycle on a
               sidewalk. The children appear to be at roughly similar distances from the camera,
               so I need to use depth estimation to determine which is closer …
        CALL   depth_estimator.estimate_depth({"image_index": 0})
        RESP   … Depth range: 0.89m to 8.40m (mean: 3.28m) …
        CALL   roborefer.detect_one({"obj_name": "child on the left"})    RESP [(0.216, 0.339)]
        CALL   roborefer.detect_one({"obj_name": "child in red jacket"})  RESP [(0.597, 0.506)]
        CALL   roborefer.detect_one({"obj_name": "child on the right"})   RESP [(0.92, 0.652)]

turn 2  THINK  Now let me get the depth values at these three locations:
        CALL   vision_ops.index_at(… 0.216, 0.339)  RESP  3.118241310119629
        CALL   vision_ops.index_at(… 0.597, 0.506)  RESP  2.0428996086120605
        CALL   vision_ops.index_at(… 0.920, 0.652)  RESP  1.346768856048584

turn 3  THINK  Based on the depth analysis: - Child on the left (A): depth = 3.12m
               - Child in red jacket (B): depth = 2.04m - Child on the right (C): depth = 1.35m
               The child on the right has the smallest depth value (1.35m vs 3.12m for child on
               the left), indicating they are closest to the camera.
        ANSWER C      <- **选项只有 (A)/(B)**
```

**看 turn 3 就明白它是怎么走到 `C` 的:它自己给三个小孩编了 A/B/C 三个标签**,
然后忠实地报告了「C 最近」—— **而题面的 (A)/(B) 指的是图上那两个标注点,不是三个小孩。**
整条深度链路执行得毫无问题,错在它重新定义了选项集。

**模型检测了三个小孩、探了三个深度,然后回答了一个不存在的选项。**

这一条还牵出一个流水线问题:`parse_ok` 为真(`<answer>` 标签完好、能解析),
**所以 `--strict` 没有拦住它**,「格式错 = 0」因此漏报。
全量扫描 1374 道有选项的题:`blinkdepth` 1/124 · `cvb2drelation` 0/650 · `cvb3ddepth` 0/600。

> **已修**:`parse_dump.py` 新增 `answer_off_options` 字段,做成 warning 不做污染标记 ——
> 「答出选项之外」是模型的错而不是流水线坏了。

---

## 11. 连续判分的两个 benchmark(另 120 个样本)

`boppose` / `bopgrasp` 没有「对/错」,判据 A、C 都不适用。
**但透传判据适用,而且更干净** —— 两个工具都把最终答案直接以 2D 形式放在返回文本里。

### 11.1 `boppose`:60/60 透传,全部误差都是 `bounding_box` 的

```
逐位同序             56/60
同一集合(允许重排)   60/60        仅重排的样本 #0 #10 #25 #35
```

**模型从不修改角点的取值。** 那 4 次只是重新排序 —— 而指标是凸包 IoU、对顺序不敏感,
所以重排既不改分数,也正好说明**它在试着满足题面「bottom face first, counter-clockwise」
的要求而做不到**。

零分也不是随机的,它跟着**点云拟合退化**走:

| | n | OBB 最短边/最长边(中位) | 比值 < 0.20 |
|---|--:|--:|--:|
| 零分 | 9 | **0.161** | **6/9** |
| 非零 | 51 | 0.306 | 12/51 |

Fisher 精确 **p = 0.0164**。**框被拟合得又扁又薄时,投影凸包就废了。**

#### 样本 A —— `boppose #0`(IoU = 0.000)

**轨迹全文**(链路 `bounding_box×1 + depth_estimator×1 + roborefer×1 + sam2×1@5t`):

```
turn 1  THINK  I'll help you find the 8 normalized corner coordinates of the bounding box for
               the can of sliced mushrooms. Let me process this step by step.
        CALL   roborefer.detect_one({"obj_name": "can of sliced mushrooms"})
        RESP   Detected 1 instance(s): [(0.779, 0.787)]

turn 2  CALL   sam2.segment_from_point({"x": 0.779, "y": 0.787, "image_index": 0})
        RESP   Segmented object … with **IoU score: 0.640**. Use $segmentation_mask …

turn 3  CALL   depth_estimator.estimate_depth_with_pointcloud({"image_index": 0})
        RESP   Generated 1932x1092 depth map and 2,109,744 3D points …

turn 4  CALL   bounding_box.compute_bbox({"point_cloud": "$point_cloud",
                                          "mask": "$segmentation_mask", …})
        RESP   Oriented bounding box computed from **524/2109744 points**.
               Corners in 3D: [[0.326, 0.195, 0.899], [0.34, 0.189, 0.904], …]
               Extent [0.016, 0.017, **0.001**]   最短/最长 = 0.059

turn 5  ANSWER [(0.786, 0.803), (0.797, 0.793), (0.777, 0.781), (0.787, 0.803),
                (0.788, 0.77), (0.798, 0.792), (0.778, 0.781), (0.788, 0.771)]
```

**turn 2 到 turn 5 没有一句 THINK。** 模型在这条链上是纯粹的管道:
检测 → 分割 → 点云 → 拟合 → 抄答案。唯一的「主动动作」是把工具给的八个点里两个对调了
(同一集合、非同序)。

**这个「长方体」是 1.6 cm × 1.7 cm × 1 mm —— 一张纸。**
只有 524 个点参与拟合,OBB 塌成了一个面,投影出来是一条细缝,和 GT 的凸包零重叠。

#### 样本 B —— `boppose #28`(IoU = 0.952)

**轨迹全文**(同一条链,逐步对照):

```
turn 1  THINK  I'll help you find the 8 normalized corner coordinates of the oatmeal-raisin
               cookies box. Let me process this step by step.          <- 与 #0 逐字同构
        CALL   roborefer.detect_one({"obj_name": "oatmeal-raisin cookies"})
        RESP   [(0.479, 0.417)]

turn 2  CALL   sam2.segment_from_point({"x": 0.479, "y": 0.417, …})
        RESP   Segmented object … with **IoU score: 0.971**          <- #0 是 0.640

turn 3  CALL   depth_estimator.estimate_depth_with_pointcloud({"image_index": 0})
        RESP   Generated … 2,109,744 3D points …                      <- 与 #0 完全相同

turn 4  CALL   bounding_box.compute_bbox(…)
        RESP   Oriented bounding box computed from **36801/2109744 points**
               Extent [0.214, 0.158, 0.046]   最短/最长 = 0.215        <- #0 是 524 点 / 0.059

turn 5  ANSWER [(0.428, 0.351), (0.445, 0.544), (0.504, 0.276), …]     逐位同序 ✓
```

**模型对这两个样本做的事情完全一样:把工具给的八个点抄下来。**
差别全在 `sam2` 的掩膜质量(0.640 vs 0.971)和进入拟合的点数(524 vs 36,801)。
`#0` 里模型还多做了一件事 —— 把两个点对调,大概是在试着满足题面的顺序要求 ——
**那次重排对分数毫无影响,但它是模型唯一一次「主动做了点什么」的痕迹,而且做了也没用。**

### 11.2 `bopgrasp`:工具失败 40/60,模型有一个完全确定的回退

```
compute_grasp 给出 5 个点的 20 条 -> 逐位透传 19
走失败出口的 40 条 -> 模型把 grasp center 放在 roborefer 检测点上 **40/40(±0.02)**
```

**工具失败时模型不弃答,而是执行一条确定的规则:抓物体中心。**

#### 样本 A —— `bopgrasp #0`(工具成功)

**轨迹全文**(链路 `depth_estimator×1 + grasp_generator×1 + roborefer×1 + sam2×1@5t`):

```
turn 1  THINK  I need to find the grasp plane for the carton of orange juice. To do this, I'll
               need to: 1. Detect the orange juice carton location 2. Segment it to get a mask
               3. Estimate depth and generate a point cloud 4. Compute the grasp pose …
        CALL   roborefer.detect_one({"obj_name": "carton of orange juice"})  RESP [(0.381,0.428)]
turn 2  THINK  Good, I found the orange juice carton at (0.381, 0.428). Now I need to segment it …
        CALL   sam2.segment_from_point(…)   RESP  … **IoU score: 0.932** …
turn 3  THINK  Great! I have the segmentation mask. Now I need to estimate depth …
        CALL   depth_estimator.estimate_depth_with_pointcloud(…)  RESP  … 937,664 3D points …
turn 4  THINK  Perfect! Now I have: - $segmentation_mask - $point_cloud - $focal_length_px
               Now I can compute the grasp pose …
        CALL   grasp_generator.compute_grasp(…)
        RESP   **Generated collision-free grasp with confidence 0.953** from 13885/937664 masked
               points. Filtered 100 → 8 collision-free grasps.
               Projected 2D gripper points: [(0.375, 0.190), (0.422, 0.200), (0.323, 0.17…
turn 5  THINK  Excellent! The grasp generator has provided me with the five normalized (x, y)
               points … Let me format this according to the user's requested format.
        ANSWER Grasp center: [0.375, 0.190], Left finger base: [0.422, 0.200],
               Right finger base: [0.323, 0.178], Left finger tip: [0.407, 0.300],
               Right finger tip: [0.310, 0.291]          <- 逐位透传 compute_grasp
```

```
GT center        (0.358, 0.257)     中心距 0.069
夹爪轴与 GT 夹角   17.1°              NCE = 1.31
```

#### 样本 B —— `bopgrasp #1`(工具失败 → 回退)

**轨迹全文**(前四轮与 `#0` 同构,第五轮岔开):

```
turn 1  THINK  I need to find the grasp plane points for the cylindrical Parmesan cheese
               container … 1. Detect 2. Segment 3. Estimate depth 4. Compute grasp 5. Extract …
        CALL   roborefer.detect_one({"obj_name": "grated Parmesan cheese container"})
        RESP   [(0.73, 0.594)]
turn 2  CALL   sam2.segment_from_point(…)  RESP  … **IoU score: 0.516** …   <- #0 是 0.932
turn 3  CALL   depth_estimator.estimate_depth_with_pointcloud(…)  RESP  … 937,664 3D points …
turn 4  THINK  Perfect! Now I have … Now I can compute the grasp pose for the Parmesan …
        CALL   grasp_generator.compute_grasp(…)
        RESP   **Error: RuntimeError: No collision-free grasps found.
               All grasps collide with other objects.**
turn 5  THINK  The grasp generator couldn't find a collision-free grasp, likely because the
               Parmesan cheese container is surrounded by other objects. **Since I was instructed
               not to call the grasp tool more than once if it fails, I need to estimate the
               grasp pose myself based on the image.** Looking at the Parmesan cheese container
               at position (0.73, 0.594), it appears to be a cylindrical container. For a
               typical parallel-jaw gripper g…
        ANSWER Grasp center: [0.73, 0.594], Left finger base: [0.70, 0.594],
               Right finger base: [0.76, 0.594], Left finger tip: [0.67, 0.594],
               Right finger tip: [0.79, 0.594]
```

**turn 5 把那条回退规则写得明明白白:「我被要求工具失败后不要再调第二次,所以只能自己估」。**
于是它把 grasp center 原封不动放在检测点上(**位移 0.0000**),
五个点的 **y 坐标全是 0.594** —— 一把水平的、张在物体中心的夹爪。

```
GT center        (0.730, 0.518)     中心距 0.076
夹爪轴与 GT 夹角   54.9°              NCE = 1.23
```

**这一对把整个矛盾压进了两个样本。** `#1` 的朝向错了 **54.9°**
(几乎垂直于正确的夹持方向,现实中抓不起来),
**而 NCE 给它 1.23,比工具真正算出抓取的 `#0`(1.31)还低。**
**一条能抓的和一条抓不起来的,奖励函数判后者更好。**

全组统计:

| | 工具成功(n=20) | 回退(n=40) |
|---|--:|--:|
| grasp center 到 GT 的距离(中位) | 0.116 | **0.092** |
| 夹爪轴与 GT 的夹角(中位) | **18.3°** | 63.9° |
| 夹角 < 30° | **16/20** | 7/40 |
| NCE(越低越好) | 1.38 | **1.07** |

**回退赢在位置、输在朝向** —— 这正好解释了两个指标为什么打架:
**NCE 主要由位置决定,SR 由朝向决定。**

> **一条能要,一条不能。**
> **不能**读成「回退比工具好」—— 两组是不同的场景,工具在哪些场景失败并不随机,这是有混淆的对比。
> **能**读的是:**这个 RL 奖励(NCE)对夹爪朝向不敏感。**
> 一个朝向中位错 64° 的答案,在 NCE 上比工具生成的真抓取还低。
>
> ⚠ **`p4/parsed/bopgrasp.jsonl` 的 `correct` 字段没有意义** ——
> 它是 `score >= 0.5` 的通用判据,而 `bopgrasp` 的 `score` 是 NCE、**越低越好**。

---

## 12. 第四类错误:一致性(不在上面 11 类里)

采样实验(`n=5, T=1.0`)显示还有一类**不属于任何单次运行的归因**:
同一道题模型自己给不出稳定答案。关键的一刀是按「五次的 `<tool_call>` 是否逐字相同」再切一次 ——
**调用相同 ⇒ 五次面对同一批证据,翻转只可能翻在决策层**:

| | n | 分裂 | 其中工具调用**逐字相同** | 调用不同 |
|---|--:|--:|--:|--:|
| RoboSpatial VQA | 228 | 100(43.9%) | **46** | 54 |
| RoboSpatial Vacant | 122 | 51(41.8%) | **0** | 51 |
| `blinkdepth` | 124 | 30(24.2%) | **3** | 27 |

- **VQA 的 46 个是纯决策翻转**:同一张图、同一批 `roborefer` 返回,给出相反的 yes/no。
- **Vacant 一个都没有**:51 次翻转全部伴随不同的 `obj_name` —— 那不是「投五票」,
  是「问五个不同的问题」。
- **`blinkdepth` 只有 3 个**,反过来印证了深度判据:给定同样两个深度读数,
  「选更小的」几乎不会被违反。

> **⚠ 现象在,收益不在。** 多数表决@5 在第二组独立采样上复核后:
> **0/12 次配对比较显著,VQA 的方向在两组之间翻了号。**
> **正确表述:多数表决@5 在这三个 benchmark 上没有效应。**
> 模型确实在同一批证据上摇摆,但**摇摆是对称的,投票捞不回来**。

---

## 13. 汇总:这份表推出来的三件事

**① 责任份额压倒性地在工具侧,而且集中在一个工具。**

| 工具 | 直接造成的错题 | 占 322 |
|---|--:|--:|
| **RoboRefer** | 126(RefSpatial 三项)+ 45(Vacant 透传)+ 26(cvb2d)+ 9(退化)= **206** | **64.0%** |
| **DepthPro** | 21(`cvb3ddepth`)+ 14(`blinkdepth`)= **35** | 10.9% |
| `bounding_box` | `boppose` 全部误差(另计) | — |
| `grasp_generator` | 40/60 直接失败(另计) | — |

> **口径**:这里按**逐样本归类**统计,所以 RoboRefer 的份额(64.0%)比底稿摘要里的
> 53.4% 高 —— 底稿只计了 RefSpatial 三项 + Vacant 透传,本表另外计入了
> `cvb2drelation` 的 26 条(遵守规则却答错 = 检测点不准)和 9 条检测退化。
> **两个数不矛盾,是统计范围不同;引用时必须带口径。**

**但这块 headroom 不能靠替换单个工具吃掉**:实测整体换成 Molmo **更差 −14.61 pp**,
逐样本 oracle 却 **+10.83 pp**。要么做 router / ensemble,要么引入显著更强的第三个工具。
**oracle 上界不是可实现收益。**

**② 除 `robospatial` 外,「推理」是一条能写成代码的规则,而模型逐字执行它。**
pointing 上 276/276 原样透传;`boppose` 上 60/60;深度题上 95.6% / 99.8% 遵守规则。
**这意味着在这些 benchmark 上换任何策略模型,结果都一样。**

**③ 唯一有真实策略缺口的是 `robospatial`,而它的缺口有明确的形状:**

| 题类 | n | 正确率 | 未调深度 | GT=`no` 正确率 | 做到 100% 能补 |
|---|--:|--:|--:|--:|--:|
| 关系题(2D 可判定) | 87 | 79.3% | — | 18/31 = 58% | +18 |
| `front/behind`(需深度序) | 29 | 72.4% | **29/29** | 7/11 = 64% | **+8**(有工具没调) |
| **`fit`(需自由空间范围)** | 105 | 69.5% | **105/105** | **4/18 = 22%** | **+32**(没工具可调) |
| 检测对不上 | 7 | 57.1% | — | — | +3 |

**可操作的优先级只有两条:**
1. **`front/behind` 的 8 条** —— 唯一一处「存在已知有效替代链、策略却从不走」,
   奖励塑形直接可打,上界 +8 题。
2. **`fit` 的 32 条** —— 需要加工具或加组合能力,**不是调 prompt、不是换 pointing 工具**。

---

## 14. 局限

- **人工归类只有一轮标注,κ 算不出来。** 32 条是单标注、无第二标注者,
  **没有标注一致性度量**。补救是逐条证据公开在 `p6/manual/verdicts.jsonl`。
- **`3b` 的归属有争议**(见 §0 第 ④ 条),两种读法都报。
- **一处归档证据需要修**:`p6/manual/verdicts.jsonl` 里 `cvb2drelation #263` 的证据串
  写成了 above/below,而该题选项是 left/right(见 §10.2)。分类结论不受影响。
- **判据与轨迹偶有出入**:`blinkdepth #5` 被判为「遵守规则」,但轨迹显示它最终是靠视觉推理
  得到同一个答案的(见 §3.4)。这是判据已知误差方向上的一例,自洽率 95.6% 量化了它的上界。
- **一次坏检测可以污染多个样本**(§7.4),所以「工具责任份额」是**按错题**而非按**缺陷**计的。
- **「工具错」是个粗标签**,§3.2 的间距表必须和计数一起引用,否则会把「没有裕度」读成「估不准」。
- **`bopgrasp` 的两组对比有混淆**(工具在哪些场景失败并不随机)。
- **本文不覆盖 accuracy 与偏差归因**,那是 P5 的内容;**也不涉及 P7 / GFlowRL。**
- **与底稿的一处差异**:`01_official_checkpoint_eval/reports/p6_error_attribution_report.md` 把人工那批记成 31 条,
  `p6/manual/verdicts.jsonl` 实际 **32 条**(`pending31.json` 也是 32 个条目)。
  本文按 32 条计,总数仍是 322,**各类计数与底稿摘要表逐项一致**。

---

## 附录:复算

### A.1 已有脚本(在 `spacetools-repro` 仓库根目录跑)

| 脚本 | 产出 |
|---|---|
| `tools/p6/p6_inventory.py` | §0 那张 clean 表 |
| `tools/p6/p6_split.py` | 判据 A(深度)· 判据 B(pointing) |
| `tools/p6/p6_relations.py` | 判据 C(关系题),内置自洽率与语义两道校验;Vacant 透传 vs 改点 |
| `tools/p6/p6_continuous.py` | §11 的两个连续 benchmark |
| `tools/p6/p6_consistency.py` | §12 一致性 |

> ⚠ `p6_inventory.py` / `p6_split*.py` 把路径写成 `$HOME/mnt/Agentic RL/spacetools-repro/p4/parsed`。
> 换机器后用软链接绕开,不要改归档脚本:
>
> ```bash
> mkdir -p "$HOME/p6shim/mnt" && ln -sfn "$HOME/mnt" "$HOME/p6shim/mnt/Agentic RL"
> HOME="$HOME/p6shim" python3 tools/p6/p6_inventory.py
> ```
>
> `p6_relations.py` / `p6_continuous.py` 用相对路径,在仓库根目录直接跑即可。

### A.2 本文新增:逐样本归类脚本

§1 那张交叉表由一个新脚本生成,它把上述判据合到一起、以**人工判定优先**,
给 322 个错题各分配唯一一个类,并自检「未覆盖 / 重复」:

```
python3 attrib.py p4/parsed p6/manual/verdicts.jsonl
→ 错题总数 322 · 已归类 322 · 未覆盖 0 · 多出 0
```

脚本与 `p6_split.py` 的唯一实现差异,在深度判据的一处修补:

```python
# 取「实际被探过深度」的前两个检测点 —— roborefer 退化时模型会改用 vlm 重检,
# 此时前两个检测点并未进入 index_at,必须按 pix 的键来对齐
probed = [p for p in d if p in pix]
dA, dB = (pix[probed[0]], pix[probed[1]]) if len(probed) >= 2 else (pix.get(d[0]), pix.get(d[1]))
```

**不打这个补丁,`blinkdepth #119` 会落进「判据跳过」而无法归类**
(该样本 RoboRefer 对 A/B 返回同一点 `(0.55, 0.283)`,模型改用 `vlm` 重检后才拿到可用坐标)。
打上之后 322/322 全覆盖,且各类计数与底稿摘要表逐项一致。

### A.3 数据

| 数据 | 路径 |
|---|---|
| 逐样本记录(全部统计的来源) | `spacetools-repro/p4/parsed/` |
| 人工归类:待判定 / 结论 / 审阅卡 | `spacetools-repro/p6/manual/` |
| 工具对调的探测与产出 | `spacetools-repro/p6/probes/` · `p6/swap/` |
| 采样实验(两组) | `spacetools-repro/p6/passk/` · `p6/passk2/` |

**除审阅卡外,每一个数字都可以从仓库副本直接重算,不需要 GPU,也不需要重跑评测。**
