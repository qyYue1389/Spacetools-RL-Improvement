#!/usr/bin/env python3
"""P6:关系题的自动判据(重建版)—— 只读 `p4/parsed/`,不需要 GPU 也不需要人工。

覆盖两个原先退回人工的 benchmark:

  * `cvb2drelation`  —— 旧判据自洽率只有约 93%,被判定不可用而退回人工。
    本版把它修到 **99.83%**。三处修法都对应旧笔记点名的那两个误差来源:
      1. 主体/参照从 **"where is the X located with respect to the Y"** 这一小句取,
         不要从开头 "relative positions of A and B" 取——后者的次序与前者不一定相同;
      2. 剥掉 "(annotated by the red box)" 这类修饰语再与 `obj_name` 匹配;
      3. **逐样本读选项字母到词的映射**,不要假定 (A)=left / (A)=above。
  * `robospatial` VQA 的关系题 —— 新增。自洽率 98.5%。

判据本身很朴素:两个检测点,在图像平面上比 x 或比 y。它的价值全在**校验**:
> **在模型答对的样本上,规则与模型是否一致?** 不一致率就是判据自身的误差上界。
> 旧判据栽在这一步(93%),而错题总数只有 35 —— 信噪比不够,所以必须退回人工。
> 一个不够准的自动判据比没有更糟。

另外报两件事,它们不是判据、但决定结论怎么写:

  * **规则对 GT 的准确率**。它回答「二维图像平面是不是这个 benchmark 的正确语义」。
    `cvb2drelation` 95.4%(是),`robospatial` VQA 78.6%(**不是**)。
    所以「遵守规则却答错」在前者可判为工具错,在后者**不能**——那里还混着
    坐标系/语义不匹配(分类 3b)与标注问题(分类 6),必须看图才能分开。
  * **判定轴上的间距**。间距接近 0 时规则退化成掷硬币,那些样本要单独拿出来,
    不能算进「遵守」或「违反」。

用法(在仓库根目录):

    python3 tools/p6/p6_relations.py --parsed p4/parsed [--dump-cases out.jsonl]
"""
import argparse, json, os, re, collections, statistics as st

PT = re.compile(r"\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)")
# robospatial:"Is the A <rel> the B?"
RS_Q = re.compile(r"^Is the (.+?) (above|below|behind|in front of|left of|right of) the (.+?)\?", re.I)
RS_FIT = re.compile(r"\bfit\b", re.I)
# cvb2drelation:"where is the SUBJ located with respect to the REF?"
CV_Q = re.compile(r"where is the (.+?) located with respect to the (.+?)\?", re.I)
CV_OPT = re.compile(r"\(([AB])\)\s*([^()]+?)(?=\s*\([AB]\)|$)")

# 只有这四种关系能从两个 2D 点判定。front/behind 需要深度,见 §「结构性缺口」。
DECIDABLE = {
    "above":    ("y", lambda a, b: a[1] < b[1]),
    "below":    ("y", lambda a, b: a[1] > b[1]),
    "left of":  ("x", lambda a, b: a[0] < b[0]),
    "right of": ("x", lambda a, b: a[0] > b[0]),
}
TIE = 0.02          # 判定轴上的间距小于此值 -> 规则退化,单独归档


def norm(s):
    s = s.lower()
    s = re.sub(r"\(annotated by the red box\)", "", s)
    return re.sub(r"\s+", " ", s).strip(" .")


def detections(rec):
    """按调用顺序收集 {obj_name: 第一个返回点}。"""
    out = {}
    for turn in rec["trajectory"]:
        calls = turn.get("tool_calls") or []
        resps = turn.get("tool_responses") or []
        for c, r in zip(calls, resps):
            if not c["name"].endswith(("detect_one", "detect_all")):
                continue
            pts = PT.findall(r.get("text") or "")
            if pts:
                out.setdefault(norm(c["arguments"].get("obj_name", "")),
                               (float(pts[0][0]), float(pts[0][1])))
    return out


def lookup(det, name):
    """精确优先;否则唯一的子串匹配。**匹配不唯一就返回 None**——
    宁可判为不可判定,也不要猜错主体/参照,那正是旧判据的死因。"""
    if name in det:
        return det[name]
    cand = [v for k, v in det.items() if name in k or k in name]
    return cand[0] if len(cand) == 1 else None


def judge(rows, label, out=None):
    dec = [x for x in rows if x["rule"]]
    if not dec:
        print(f"\n=== {label}: 没有可判定样本"); return
    corr = [x for x in dec if x["correct"]]
    wrong = [x for x in dec if not x["correct"]]
    agree_c = sum(1 for x in corr if x["rule"] == x["ans"])

    print(f"\n=== {label}")
    print(f"  可判定 {len(dec)}/{len(rows)}")
    print(f"  [判据校验] 答对样本 n={len(corr)}:规则与模型一致 {agree_c} "
          f"({100*agree_c/len(corr):.2f}%),不一致 {len(corr)-agree_c}"
          f"   <- 判据自身误差上界")
    if agree_c / len(corr) < 0.97:
        print("  ⚠ 自洽率低于 97%,判据不可用,退回人工(旧 cvb2drelation 判据就死在这里)")

    rule_gt = sum(1 for x in dec if x["rule"] == x["gt"])
    print(f"  [语义检验] 规则对 GT {rule_gt}/{len(dec)} = {100*rule_gt/len(dec):.1f}%"
          f" · 模型对 GT {len(corr)}/{len(dec)} = {100*len(corr)/len(dec):.1f}%")
    if abs(rule_gt - len(corr)) <= 2:
        print("  → 模型的成绩与规则的成绩几乎相同:**在这类题上模型就是这条规则**,"
              "它的上限就是规则的上限")
    if rule_gt / len(dec) < 0.90:
        print("  ⚠ 图像平面规则不是这个 benchmark 的正确语义。"
              "「遵守规则却答错」**不能**直接判为工具错——还混着坐标系/语义(3b)与标注(6)")

    tie   = [x for x in wrong if x["margin"] is not None and x["margin"] < TIE]
    keep  = [x for x in wrong if x not in tie]
    obey  = [x for x in keep if x["rule"] == x["ans"]]
    viol  = [x for x in keep if x["rule"] != x["ans"]]
    print(f"  [错题 n={len(wrong)}] 遵守规则 {len(obey)} · 违反规则 {len(viol)} · "
          f"间距<{TIE} 规则退化 {len(tie)}")

    if obey:
        mo = [x["margin"] for x in obey if x["margin"] is not None]
        mc = [x["margin"] for x in corr if x["margin"] is not None]
        print(f"  [间距] 答对组中位 {st.median(mc):.3f} · 遵守却答错组中位 {st.median(mo):.3f}")
        for t in (0.05, 0.10):
            print(f"        <{t:.2f}: 答对组 {100*sum(1 for m in mc if m<t)/len(mc):.0f}% · "
                  f"答错组 {100*sum(1 for m in mo if m<t)/len(mo):.0f}%")

    if out is not None:
        tie_ids = {x["sid"] for x in tie}
        for x in wrong:
            v = ("规则退化(间距≈0,不计入遵守/违反)" if x["sid"] in tie_ids else
                 ("遵守规则" if x["rule"] == x["ans"] else "违反规则"))
            out.append({**x, "benchmark": label, "verdict": v})


def do_cvb2d(path, out):
    rows = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        m = CV_Q.search(r["question"])
        if not m:
            continue
        subj, ref = norm(m.group(1)), norm(m.group(2))
        opts = {l.upper(): w.strip().lower() for l, w in CV_OPT.findall(r["question"])}
        det = detections(r)
        pa, pb = lookup(det, subj), lookup(det, ref)
        rule = margin = None
        axis = "x" if "left" in opts.values() else "y"
        if pa and pb:
            if axis == "x":
                word, margin = ("left" if pa[0] < pb[0] else "right"), abs(pa[0] - pb[0])
            else:
                word, margin = ("above" if pa[1] < pb[1] else "below"), abs(pa[1] - pb[1])
            hit = [l for l, w in opts.items() if w == word]
            rule = hit[0] if hit else None
        rows.append(dict(sid=r["sample_id"], rule=rule, axis=axis, margin=margin,
                         gt=str(r["gt"]).strip().upper()[:1],
                         ans=str(r["raw_answer"]).strip().upper()[:1],
                         correct=r["correct"], q=r["question"][:90]))
    judge(rows, "cvb2drelation", out)


def do_robospatial(path, out):
    rows, fit, fb, unmatched = [], [], [], []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if str(r["gt"]).strip().lower() not in ("yes", "no"):
            continue                                   # Vacant,不在本判据范围
        if RS_FIT.search(r["question"]):
            fit.append(r); continue
        m = RS_Q.match(r["question"].strip())
        if not m:
            continue
        A, rel, B = norm(m.group(1)), m.group(2).lower(), norm(m.group(3))
        det = detections(r)
        pa, pb = lookup(det, A), lookup(det, B)
        if rel not in DECIDABLE:
            fb.append(r); continue
        if not (pa and pb):
            unmatched.append(r); continue
        axis, fn = DECIDABLE[rel]
        margin = abs(pa[0] - pb[0]) if axis == "x" else abs(pa[1] - pb[1])
        rows.append(dict(sid=r["sample_id"], rule=("yes" if fn(pa, pb) else "no"),
                         axis=axis, margin=margin, rel=rel,
                         gt=str(r["gt"]).strip().lower(),
                         ans=str(r["raw_answer"]).strip().lower(),
                         correct=r["correct"], q=r["question"][:90]))
    judge(rows, "robospatial VQA · 关系题", out)

    # ---- 结构性缺口:题目需要的信息,模型从来没有去取 ----
    print("\n=== robospatial VQA · 结构性缺口(不靠判据,靠工具直方图)")
    for name, group, why in [
        ("front/behind", fb, "需要深度序;depth_estimator 就在工具表里"),
        ("fit(自由空间)", fit, "需要自由空间范围;没有单个工具直接给,但 bbox+depth 可逼近"),
    ]:
        if not group:
            continue
        c = sum(1 for r in group if r["correct"])
        nod = sum(1 for r in group if "depth" not in r["chain_signature"])
        no_c = [r for r in group if str(r["gt"]).lower() == "no"]
        print(f"  {name:16} n={len(group):3}  正确率 {c}/{len(group)} = {100*c/len(group):.1f}%"
              f"  **未调用深度工具 {nod}/{len(group)}**"
              f"  GT=no 正确率 {sum(1 for r in no_c if r['correct'])}/{len(no_c)}")
    if unmatched:
        print(f"  {'检测对不上':16} n={len(unmatched):3}  "
              f"正确率 {sum(1 for r in unmatched if r['correct'])}/{len(unmatched)}  <- 需看图")


def do_vacant(path):
    """RoboSpatial Vacant:透传 vs 自行改点,以及「改点时用眼估的整数坐标」。

    Vacant 的答案是一个点,所以判据 B(原样透传)直接适用。已知「模型一改点
    正确率减半」;这里再问一句**它改成了什么样的数**——如果覆盖工具靠的是目测
    而不是计算,那么坐标会落在 0.05 这种人为粒度上。
    """
    import math
    keep, chg = [], []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if str(r["gt"]).strip().lower() in ("yes", "no"):
            continue                                   # VQA,不在本节范围
        det = detections(r)
        tool = next(iter(det.values()), None)
        a = PT.findall(str(r["raw_answer"]))
        if not tool or not a:
            continue
        ap = (float(a[0][0]), float(a[0][1]))
        (chg if math.dist(ap, tool) >= 0.02 else keep).append((ap, r["correct"]))

    def rnd(v, step=0.05):
        return abs(v / step - round(v / step)) < 1e-6

    print("\n=== robospatial Vacant · 透传 vs 改点")
    print(f"  {'':14}{'n':>5}{'正确率':>12}{'两坐标都是 0.05 整数倍':>24}")
    for g, name in ((keep, "原样透传"), (chg, "模型自行改点")):
        if not g:
            continue
        ok = sum(1 for _, c in g if c)
        r5 = sum(1 for ap, _ in g if rnd(ap[0]) and rnd(ap[1]))
        print(f"  {name:14}{len(g):5}{ok:6} ({100*ok/len(g):4.1f}%){r5:14} ({100*r5/len(g):3.0f}%)")
    print("  → 透传时坐标带着工具的三位小数;一旦模型自己动手,坐标就落在人为粒度上。")
    print("     **它不是在算,是在目测**——而 Vacant 的 GT 是一条几十像素宽的窄带。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", default="p4/parsed")
    ap.add_argument("--dump-cases", help="把错题逐条写成 jsonl,供人工归类使用")
    a = ap.parse_args()
    out = [] if a.dump_cases else None

    do_cvb2d(os.path.join(a.parsed, "cvb2drelation.jsonl"), out)
    do_robospatial(os.path.join(a.parsed, "robospatial.jsonl"), out)
    do_vacant(os.path.join(a.parsed, "robospatial.jsonl"))

    if a.dump_cases:
        os.makedirs(os.path.dirname(a.dump_cases) or ".", exist_ok=True)
        with open(a.dump_cases, "w", encoding="utf-8") as f:
            for x in out:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"\n写入 {a.dump_cases}({len(out)} 条错题)")


if __name__ == "__main__":
    main()
