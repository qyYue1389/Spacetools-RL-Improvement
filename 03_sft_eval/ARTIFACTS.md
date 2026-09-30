# SFT eval 原始产物 · 2026-09-11/12

来自 `qzpm55555/spacetools-sft-v1-4xa6000` @ `91fd4bdf6d2ffe3b355cc650e71925c6e91a15c5`
在 4× RTX A6000 上的 eval。分析结论见 `03_sft_eval/sft_eval_report.md`。

机器已 terminate,这里是全部可留存的东西。

---

## rollouts/ —— 轨迹,最有价值的部分

```
reflocation.jsonl         100 行
refplacement.jsonl        100 行
refunseen.jsonl            77 行
robospatial_run1.jsonl    350 行   ← 第一次运行
robospatial_run2.jsonl    350 行   ← 第二次运行,同一 ckpt、同一配置
```

每行是一个样本的完整记录,JSON:

| 字段 | 内容 |
|---|---|
| `index` | 样本序号,两次运行之间可对齐 |
| `input` | 完整 prompt:system 提示 + 七工具的 schema + 图(base64)+ 问题,约 10 KB |
| `output` | **模型的生成**,含 `<think>` / `<tool_call>` / `<tool_response>` / `<answer>` 全部轮次 |
| `gts` | 真值。VQA 是 `Yes`/`No`;Vacant 是 10 个点;部分 RefSpatial 是 base64 PNG |
| `acc` `score` `reward` | 判分结果,本任务下三者相同 |

**robospatial 的两份可以逐样本对比**,这是量化非确定性的数据来源:
两次之间 29 个样本翻转、199/350 条生成逐字节不同,而工具响应零差异。

### VQA / Vacant 怎么拆

按 `gts` 的形态,不需要额外元数据:

```python
import re, json
def is_vacant(r):                       # 点列表 → Vacant(122 题)
    return bool(re.search(r"\(\s*-?[0-9.]*\s*,", r["gts"]))
# 其余是 VQA(228 题),gts 为 Yes / No
```

### 判分函数

`verl/utils/reward_score/robos_all.py`:

- 点类问题:`compute_points_score(..., point_evaluation_method="convex_hull")`
  —— **预测点是否落在 10 个 GT 点的凸包内,二值**。
  注意不是「到最近 GT 点的距离」:一个点可以离某个 GT 点很近却在凸包外而判 0。
  分析错题要算到**凸包边界**的距离,用仓库自己的 `_convex_hull` /
  `_is_point_in_convex_polygon`(已验证与落盘 acc 122/122 一致)。
- Yes/No 问题:`compute_yesno_score`

---

## logs/ —— eval 日志

```
run1_all4benchmarks.log    15.8 MB   四个 benchmark 的完整运行(32 分 41 秒)
run2_robospatial.log        8.7 MB   robospatial 补跑
blinkdepth_attempt1.log             blinkdepth 两次尝试,都死在 verl 侧
blinkdepth_attempt2.log             (GPU 预算问题,见报告 §6.2)
perbench_<key>.log                  verl 按 benchmark 分开落的日志
```

日志里带大量 ANSI 和进度条。有用的 grep:

```bash
grep -oE "val-core/[A-Za-z0-9/-]+/acc/mean@1:[0-9.]+" run1_all4benchmarks.log | sort -u
grep -c OutOfMemoryError run1_all4benchmarks.log
grep -cE "Error:|ERROR:toolshed" run1_all4benchmarks.log    # 子串匹配,能抓到 TypeError:
grep -c "Version mismatch" run1_all4benchmarks.log
```

---

## verify/ —— 验收产物

```
VERIFY_run_initial.log              首次四项验收
VERIFY_run_after_patch.log          Ray Python 补丁之后的验收(七工具 7/7)
<tool>.log × 7                      七个工具各自的冒烟输出(真加载权重、真出结果)
chain_raw.log                       Ray 跨 conda 环境工具链测试的原始输出
chain_rerun_after_cleanup.log       清掉残留 ray 地址文件后重跑
ray_python_patch.log                minor 档补丁的应用与回读验证
```

---

## gpu/ —— GPU 轨迹,不可再生

```
gputrace_1hz.log        eval 运行期间 1 Hz × 75 采样,每卡 used_MiB + util%
                        末尾带进程级归属(哪个 actor 在哪张卡、占多少)
gputrace_summary.log    每卡峰值/均值汇总
```

**跑完再查 nvidia-smi 只能看到 0 MiB**,所以这份只能在运行中采。
后续 eval / RL 建议常态化这一步。

---

## config/ —— 出处

```
toolshed_config.yaml    本次实际生效的七工具配置(由 generate_toolshed_config.py 产出)
run_eval.sh.asrun       跑的时候那一版 run_eval.sh
WEIGHTS_PINS.txt        七个模型权重钉死的 revision
pkg_MANIFEST.txt        环境包清单(含五环境 Python 版本表、GPU 预算实测)
SHA256SUMS.remote       环境包各文件哈希
POSTRESTORE.sh          还原后要补的三件事(已推到 HF repo)
VERIFY.sh.new           改成四项之后的验收脚本(已推到 HF repo)
```

---

## scripts/ 与 analysis/ —— 分析脚本及其输出,可复用

| 脚本 | 做什么 |
|---|---|
| `audit2.sh` | 逐样本核工具调用与错误文本(判读第二步,只看日志不够) |
| `health.sh` | 五项健康指标 + 每样本工具调用次数分布 |
| `nondet.sh` | 把两次运行的分歧拆成策略侧 / 工具侧 |
| `stable.sh` | 恒对 / 恒错 / 翻转分解,给出期望值与判据通过概率 |
| `hull.sh` | 用仓库自己的几何函数按凸包判据重算 Vacant |
| `vacant.sh` | Vacant 错题的偏差分布 |
| `split.sh` | 按 GT 形态拆 VQA / Vacant |
| `material.sh` | VQA 混淆矩阵 + 代表性轨迹导出 |
| `compare.sh` | 两次运行逐样本对比 |
| `gputrace.sh` | 1 Hz GPU 采样 |

脚本里的路径是那台机器上的绝对路径,复用时要改。
`analysis/` 是它们当时的输出,可以直接对照报告里的数字。
