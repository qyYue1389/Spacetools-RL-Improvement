#!/usr/bin/env python3
"""P6 §2b:一致性(第四类错误)—— 只读 dump,不需要 GPU 也不需要 conda 环境。

输入是实验 C 的 pass@k dump(`p6/passk/<bench>/*.jsonl`,每个样本 5 行)。
做三件事:

  1. 把每个样本的 5 次按对错分成 全对 / 全错 / 分裂;
  2. **把分裂样本再按「五次的 <tool_call> 是否逐字相同」切一刀** ——
     调用相同 ⇒ 五次面对同一批证据,翻转只可能翻在决策层;
     调用不同 ⇒ 每次问的都不一样,五次之间没有共同证据,无从表决;
  3. 报「五次里多数正确」的样本数,并与每一次 greedy 运行做**逐样本配对检验**
     (双尾精确 McNemar)。**不要只比范围** —— 一组 5 次采样只给出一个表决值,
     拿它和 greedy 的多个值比范围,等于把有散布的一侧和没散布的一侧对比。
     **注意**:点预测(Vacant)没有「多数」可投,那一行只是描述统计,不是策略。

用法(在仓库根目录跑,greedy 参照按相对路径找,缺了就跳过那一次比较):

    python3 tools/p6/p6_consistency.py --root p6/passk

一条陷阱:dump 里的 `answer` 字段是 **GT 的副本**,不是模型的答案
(全部记录 `answer == gts`)。判对错只能看 `acc`。本脚本会先断言这一点。
"""
import argparse, json, glob, os, re, collections, statistics as st
from math import comb

TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
BINARY = {"yes", "no", "true", "false", "a", "b"}

# greedy 参照运行的 dump 目录。**必须逐样本配对比较,不能只比范围** ——
# 一组 5 次采样只给出一个表决值,拿它和 greedy 的多个值比范围,
# 等于把有散布的一侧和没散布的一侧对比。见 P6_NOTES.md §2b。
GREEDY_DIRS = {
    "blinkdepth": ["p4/dumps/run1/blinkdepth", "p4/dumps/run4/blinkdepth",
                   "p6/gmu025/run5/blinkdepth", "p6/gmu025/run6/blinkdepth"],
    "robospatial": ["p4/dumps/run1/robospatial", "p4/dumps/run2/robospatial",
                    "p6/gmu025/run3/robospatial", "p6/gmu025/run8/robospatial"],
}


def mcnemar_exact(b, c):
    """双尾精确 McNemar。b/c 是两侧的独对数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(path):
    rows = []
    for f in sorted(glob.glob(os.path.join(path, "*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            rows.extend(json.loads(l) for l in fh if l.strip())
    return rows


def split_key(gt):
    """robospatial 的 parquet 把 VQA 与 Vacant 混在一起,按 GT 的形状分。"""
    return "VQA" if str(gt).strip().lower() in BINARY else "Vacant"


def report(label, samples, bench, keep):
    """samples: {index: [row, ...]};keep 是本组的 index 集合,用于筛 greedy 参照。"""
    n = len(samples)
    tab = collections.Counter()
    for rows in samples.values():
        corr = [r["acc"] >= 0.5 for r in rows]
        calls = {tuple(c.strip() for c in TOOL_CALL.findall(r["output"])) for r in rows}
        state = "全对" if all(corr) else ("全错" if not any(corr) else "分裂")
        tab[(state, len(calls) == 1)] += 1

    avg1 = st.mean(st.mean(r["acc"] >= 0.5 for r in rows) for rows in samples.values())
    p5 = st.mean(any(r["acc"] >= 0.5 for r in rows) for rows in samples.values())
    maj_ok = {i: sum(r["acc"] >= 0.5 for r in rows) >= 3
              for i, rows in samples.items()}
    maj = sum(maj_ok.values())

    split = tab[("分裂", True)] + tab[("分裂", False)]
    print(f"\n=== {label}  n={n}")
    print(f"  avg@1 {avg1*100:6.2f}%   pass@5 {p5*100:6.2f}%   多数正确 {maj}/{n} "
          f"({maj/n*100:.2f}%)")
    print(f"  全对 {tab[('全对',True)]+tab[('全对',False)]:4}  "
          f"全错 {tab[('全错',True)]+tab[('全错',False)]:4}  "
          f"分裂 {split:4} ({split/n*100:.1f}%)")
    print(f"  分裂样本里:五次工具调用**逐字相同** {tab[('分裂',True)]:4}  "
          f"(纯决策翻转)   调用不同 {tab[('分裂',False)]:4}")

    dirs = [d for d in GREEDY_DIRS.get(bench, []) if os.path.isdir(d)]
    if dirs:
        print("  逐样本配对(多数表决@5 vs 每一次 greedy,双尾精确 McNemar):")
        sig = 0
        nets = []
        for d in dirs:
            g = {r["index"]: (r["acc"] >= 0.5) for r in load(d)}
            idx = [i for i in keep if i in g]
            b = sum(1 for i in idx if maj_ok[i] and not g[i])
            c = sum(1 for i in idx if g[i] and not maj_ok[i])
            p = mcnemar_exact(b, c)
            sig += p < 0.05
            nets.append(b - c)
            print(f"    vs {os.path.basename(os.path.dirname(d)):6}"
                  f"(={sum(g[i] for i in idx):4}) 表决独对 {b:3} · greedy 独对 {c:3}"
                  f"   净 {b-c:+3}   p={p:.4f}{'  *' if p < 0.05 else ''}")
        if sig == len(dirs):
            verdict = "全部显著"
        elif all(x > 0 for x in nets):
            verdict = "**方向一致为正但未达显著,不构成结论**"
        elif all(x <= 0 for x in nets):
            verdict = "**无增益**(净差 ≤ 0)"
        else:
            verdict = "**方向不一致,无效应**"
        print(f"  => {sig}/{len(dirs)} 次比较显著。{verdict}")
        print("     注意:各次比较**共用同一个表决臂**(只跑了一组 5 次采样),"
              "不是独立重复;要定案需要第二组采样。")

    if label.endswith("Vacant"):
        print("  ⚠ 点预测没有「多数」可投,上面的『多数正确』只是描述统计,不是可实现策略")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="p6/passk")
    args = ap.parse_args()

    for bench in sorted(os.listdir(args.root)):
        path = os.path.join(args.root, bench)
        if not os.path.isdir(path):
            continue
        rows = load(path)
        if not rows:
            continue

        bad = [r for r in rows if str(r["answer"]) != str(r["gts"])]
        assert not bad, (f"{bench}: `answer` 不再等于 `gts`({len(bad)} 条)。"
                         "本脚本假定 answer 是 GT 的副本,若上游改了含义,先确认再往下读。")

        by_idx = collections.defaultdict(list)
        for r in rows:
            by_idx[r["index"]].append(r)
        ks = {len(v) for v in by_idx.values()}
        if ks != {5}:
            print(f"[warn] {bench}: 每样本次数 = {ks},不是 5")

        if bench == "boppose":
            # 连续指标,对错分档无意义;只报 avg@1 与 best@5,并提醒选择偏倚
            avg = st.mean(st.mean(r["acc"] for r in v) for v in by_idx.values())
            best = st.mean(max(r["acc"] for r in v) for v in by_idx.values())
            print(f"\n=== boppose  n={len(by_idx)}(连续 IoU,不做对错分档)")
            print(f"  avg@1 {avg*100:.2f}   best@5 {best*100:.2f}")
            print("  ⚠ best@5 有选择偏倚:即使分数纯为噪声,max 也高于均值。**不能当收益**")
            continue

        groups = collections.defaultdict(dict)
        for i, v in by_idx.items():
            gt = v[0]["gts"]
            sub = split_key(gt)
            # 只有 robospatial 需要再分;单一类型的 benchmark 不加后缀
            groups[sub][i] = v
        if len(groups) == 1:
            s = next(iter(groups.values()))
            report(bench, s, bench, set(s))
        else:
            for sub in ("VQA", "Vacant"):
                if sub in groups:
                    report(f"{bench}:{sub}", groups[sub], bench, set(groups[sub]))


if __name__ == "__main__":
    main()
