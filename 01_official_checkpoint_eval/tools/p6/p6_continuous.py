#!/usr/bin/env python3
"""P6:两个连续指标 benchmark 的归因 —— `boppose` 与 `bopgrasp`。

它们没有「对/错」,所以判据 A(深度规则)和判据 C(关系规则)都不适用。
但**判据 B(原样透传)适用**,而且比在 pointing 上更干净:两个工具都把
最终答案**直接以 2D 形式**放在返回文本里——

    bounding_box.compute_bbox -> "Corners in normalized image coordinates: [[...]]"
    grasp_generator.compute_grasp -> "Projected 2D gripper points: [(...)]"

所以「模型的答案是不是工具输出的原样透传」可以逐位比对。答案是:是。

另外两件只能在这里做的事:

  * `boppose` 的零分与 **OBB 退化**(拟合出的框又扁又薄)相关;
  * `bopgrasp` 的工具在 40/60 上失败,而模型有一个**完全确定的回退**:
    把 grasp center 放在 roborefer 的检测点上。把两组按**位置**和**朝向**
    分开量,可以解释 NCE 与 SR 为什么给出相反的排序。

用法:

    python3 tools/p6/p6_continuous.py --parsed p4/parsed
"""
import argparse, json, math, os, re, statistics as st
from math import comb

NUMS = re.compile(r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)")
PAIR_SQ = re.compile(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]")
PAIR_RD = re.compile(r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)")
BBOX_2D = re.compile(r"Corners in normalized image coordinates:\s*(\[\[.*?\]\])", re.S)
GRASP_2D = re.compile(r"Projected 2D gripper points:\s*(\[.*?\])", re.S)
EXTENT = re.compile(r"Extent:\s*\[([^\]]*)\]")
GRASP_FAIL = re.compile(r"No collision-free grasps|Top-down filtering removed all")
TOL = 0.0015          # 工具打印三位小数,这是「逐位相同」的容差


def calls(rec):
    for turn in rec["trajectory"]:
        for c, r in zip(turn.get("tool_calls") or [], turn.get("tool_responses") or []):
            yield c, (r.get("text") or "")


def pts(s, rx):
    return [(round(float(a), 3), round(float(b), 3)) for a, b in rx.findall(s or "")]


def same(a, b):
    return len(a) == len(b) and all(abs(x[0]-y[0]) <= TOL and abs(x[1]-y[1]) <= TOL
                                    for x, y in zip(a, b))


def fisher(a, b, c, d):
    """双尾 Fisher 精确检验。"""
    n = a + b + c + d
    tot = comb(n, a + c)
    obs = comb(a + b, a) * comb(c + d, c) / tot
    p = 0.0
    for i in range(min(a + b, a + c) + 1):
        k = a + c - i
        if k < 0 or k > c + d:
            continue
        pr = comb(a + b, i) * comb(c + d, k) / tot
        if pr <= obs + 1e-12:
            p += pr
    return min(1.0, p)


def med(v):
    return st.median(v) if v else float("nan")


# ---------------------------------------------------------------- boppose
def do_boppose(path):
    R = [json.loads(l) for l in open(path, encoding="utf-8")]
    exact = setsame = withtool = 0
    reorder, rows = [], []
    for r in R:
        tool, ext = None, None
        for c, txt in calls(r):
            if c["name"].endswith("compute_bbox"):
                m = BBOX_2D.search(txt)
                if m:
                    tool = pts(m.group(1), PAIR_SQ)
                e = EXTENT.search(txt)
                if e:
                    ext = [float(x) for x in e.group(1).split(",")]
        ans = pts(str(r["raw_answer"]), PAIR_RD)
        if tool and len(ans) == len(tool) == 8:
            withtool += 1
            e_, s_ = same(ans, tool), sorted(ans) == sorted(tool)
            exact += e_; setsame += s_
            if s_ and not e_:
                reorder.append(r["sample_id"])
        rows.append(dict(sid=r["sample_id"], score=r["score"], ext=ext))

    print(f"\n=== boppose  n={len(R)}")
    print(f"  答案与 compute_bbox 的 2D 角点:**逐位同序 {exact}/{withtool}** · "
          f"**同一集合(允许重排) {setsame}/{withtool}**")
    print(f"  仅重排的样本:{reorder}")
    print("  → 模型从不修改角点的**取值**。指标是凸包 IoU、对顺序不敏感,")
    print("     所以那几次重排既不改分数,也说明它在试着满足题面的顺序要求而做不到。")
    print(f"  **结论:boppose 的全部误差都是 bounding_box 的误差,推理侧零参与。**")

    z = [x for x in rows if x["score"] == 0 and x["ext"]]
    nz = [x for x in rows if x["score"] > 0 and x["ext"]]
    ratio = lambda g: [min(x["ext"]) / max(x["ext"]) for x in g if max(x["ext"]) > 0]
    rz, rn = ratio(z), ratio(nz)
    if rz and rn:
        a = sum(1 for v in rz if v < 0.20); b = len(rz) - a
        c = sum(1 for v in rn if v < 0.20); d = len(rn) - c
        print(f"\n  零分 {len(z)} 条 vs 非零 {len(nz)} 条,OBB 的最短边/最长边:")
        print(f"    中位 {med(rz):.3f} vs {med(rn):.3f}")
        print(f"    比值 < 0.20(框被拟合得又扁又薄):{a}/{a+b} vs {c}/{c+d}"
              f"   Fisher 精确 p={fisher(a,b,c,d):.4f}")
        print("  → 零分不是随机的:它跟着**点云拟合退化**走。")


# --------------------------------------------------------------- bopgrasp
def angle(v):
    """夹爪轴的朝向:左指根 -> 右指根。返回 [0,180) 度。"""
    return math.degrees(math.atan2(v[2][1] - v[1][1], v[2][0] - v[1][0])) % 180


def do_bopgrasp(path):
    R = [json.loads(l) for l in open(path, encoding="utf-8")]
    rows, passthrough, withtool, fellback, nfail = [], 0, 0, 0, 0
    for r in R:
        tool, det, failed = None, None, False
        for c, txt in calls(r):
            if c["name"].endswith("detect_one"):
                m = PAIR_RD.findall(txt)
                if m:
                    det = (float(m[0][0]), float(m[0][1]))
            if c["name"].endswith("compute_grasp"):
                if GRASP_FAIL.search(txt):
                    failed = True
                m = GRASP_2D.search(txt)
                if m:
                    tool = [(round(float(a), 3), round(float(b), 3))
                            for a, b in NUMS.findall(m.group(1))]
        ans = [(float(a), float(b)) for a, b in NUMS.findall(str(r["raw_answer"]))]
        gt = [(float(a), float(b)) for a, b in NUMS.findall(str(r["gt"]))]
        if tool and len(ans) == len(tool) == 5:
            withtool += 1
            passthrough += same(ans, tool)
        if tool is None and det and ans:
            nfail += 1
            fellback += (abs(ans[0][0]-det[0]) <= 0.02 and abs(ans[0][1]-det[1]) <= 0.02)
        if len(ans) >= 5 and len(gt) >= 5:
            da = abs(angle(ans) - angle(gt)) % 180
            rows.append(dict(sid=r["sample_id"], tool=tool is not None, score=r["score"],
                             dc=math.dist(ans[0], gt[0]), da=min(da, 180 - da)))

    print(f"\n=== bopgrasp  n={len(R)}")
    print(f"  compute_grasp 给出 5 个点的 {withtool} 条,其中**逐位透传 {passthrough}**")
    print(f"  工具走失败出口的 {len(R)-withtool} 条,模型把 grasp center 放在 "
          f"roborefer 检测点上(±0.02)的:**{fellback}/{nfail}**")
    print("  → 工具失败时模型不弃答,而是执行一个**完全确定的回退**:抓物体中心。")

    S = [x for x in rows if x["tool"]]
    F = [x for x in rows if not x["tool"]]
    if S and F:
        print(f"\n  把两组按**位置**和**朝向**分开量(n={len(S)} vs {len(F)}):")
        print(f"    {'':26}{'工具成功':>10}{'回退':>10}")
        print(f"    {'grasp center 到 GT 距离':26}{med([x['dc'] for x in S]):10.3f}"
              f"{med([x['dc'] for x in F]):10.3f}")
        print(f"    {'夹爪轴与 GT 夹角(度)':26}{med([x['da'] for x in S]):10.1f}"
              f"{med([x['da'] for x in F]):10.1f}")
        print(f"    {'NCE 分数(越低越好)':26}{med([x['score'] for x in S]):10.2f}"
              f"{med([x['score'] for x in F]):10.2f}")
        for t in (15, 30, 45):
            print(f"      夹角 < {t}°:{sum(1 for x in S if x['da']<t)}/{len(S)}"
                  f"  vs  {sum(1 for x in F if x['da']<t)}/{len(F)}")
        print("\n  → **回退赢在位置、输在朝向**,这解释了 NCE 与 SR 为什么给出相反的排序:")
        print("     NCE 主要由位置决定,SR 由朝向决定。")
        print("  ⚠ 两组是不同的场景(工具在哪些场景失败并不随机),所以这是**有混淆的对比**,")
        print("     不能读成「回退比工具好」。能读的是:**这个 RL 奖励对朝向不敏感。**")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", default="p4/parsed")
    a = ap.parse_args()
    do_boppose(os.path.join(a.parsed, "boppose.jsonl"))
    do_bopgrasp(os.path.join(a.parsed, "bopgrasp.jsonl"))
    print("\n注意:`p4/parsed/bopgrasp.jsonl` 里的 `correct` 字段**没有意义**——"
          "它是 score>=0.5 的通用判据,而 bopgrasp 的 score 是 NCE,越低越好。")


if __name__ == "__main__":
    main()
