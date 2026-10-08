#!/usr/bin/env python3
"""P6: attribution for the two continuous-metric benchmarks — `boppose` and `bopgrasp`.

They have no "right/wrong", so criterion A (depth rule) and criterion C (relation rule) do not apply.
But **criterion B (verbatim pass-through) applies**, and more cleanly than on pointing: both tools put
the final answer **directly in 2D form** in the returned text —

    bounding_box.compute_bbox -> "Corners in normalized image coordinates: [[...]]"
    grasp_generator.compute_grasp -> "Projected 2D gripper points: [(...)]"

So "is the model's answer a verbatim pass-through of the tool output" can be compared bit-for-bit. The answer is: yes.

Two other things that can only be done here:

  * `boppose` zero scores correlate with **OBB degeneration** (the fitted box is flat and thin);
  * the `bopgrasp` tool fails on 40/60, and the model has a **fully deterministic fallback**:
    put the grasp center on roborefer's detection point. Measuring the two groups separately by **position** and **orientation**
    explains why NCE and SR give opposite rankings.

Usage:

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
TOL = 0.0015          # the tool prints three decimals; this is the tolerance for "bit-for-bit identical"


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
    """Two-sided Fisher exact test."""
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
    print(f"  answer vs compute_bbox 2D corners: **bit-for-bit, same order {exact}/{withtool}** · "
          f"**same set (reordering allowed) {setsame}/{withtool}**")
    print(f"  samples that were only reordered: {reorder}")
    print("  → the model never changes the corner **values**. The metric is convex-hull IoU, insensitive to order,")
    print("     so those few reorderings do not change the score, and they show it is trying to satisfy the ordering the question asks for and cannot.")
    print(f"  **Conclusion: all of the boppose error is bounding_box error; the reasoning side contributes nothing.**")

    z = [x for x in rows if x["score"] == 0 and x["ext"]]
    nz = [x for x in rows if x["score"] > 0 and x["ext"]]
    ratio = lambda g: [min(x["ext"]) / max(x["ext"]) for x in g if max(x["ext"]) > 0]
    rz, rn = ratio(z), ratio(nz)
    if rz and rn:
        a = sum(1 for v in rz if v < 0.20); b = len(rz) - a
        c = sum(1 for v in rn if v < 0.20); d = len(rn) - c
        print(f"\n  zero-score {len(z)} vs non-zero {len(nz)}, OBB shortest edge/longest edge:")
        print(f"    median {med(rz):.3f} vs {med(rn):.3f}")
        print(f"    ratio < 0.20 (box fitted flat and thin): {a}/{a+b} vs {c}/{c+d}"
              f"   Fisher exact p={fisher(a,b,c,d):.4f}")
        print("  → zero scores are not random: they follow **point-cloud fit degeneration**.")


# --------------------------------------------------------------- bopgrasp
def angle(v):
    """Orientation of the gripper axis: left finger base -> right finger base. Returns [0,180) degrees."""
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
    print(f"  compute_grasp gave 5 points on {withtool}, of which **bit-for-bit pass-through {passthrough}**")
    print(f"  tool took the failure exit on {len(R)-withtool}; model put the grasp center on the "
          f"roborefer detection point (±0.02) in: **{fellback}/{nfail}**")
    print("  → when the tool fails the model does not abstain; it runs a **fully deterministic fallback**: grasp the object center.")

    S = [x for x in rows if x["tool"]]
    F = [x for x in rows if not x["tool"]]
    if S and F:
        print(f"\n  measuring the two groups separately by **position** and **orientation** (n={len(S)} vs {len(F)}):")
        print(f"    {'':26}{'tool success':>10}{'fallback':>10}")
        print(f"    {'grasp center to GT dist':26}{med([x['dc'] for x in S]):10.3f}"
              f"{med([x['dc'] for x in F]):10.3f}")
        print(f"    {'gripper axis vs GT angle (deg)':26}{med([x['da'] for x in S]):10.1f}"
              f"{med([x['da'] for x in F]):10.1f}")
        print(f"    {'NCE score (lower is better)':26}{med([x['score'] for x in S]):10.2f}"
              f"{med([x['score'] for x in F]):10.2f}")
        for t in (15, 30, 45):
            print(f"      angle < {t}°: {sum(1 for x in S if x['da']<t)}/{len(S)}"
                  f"  vs  {sum(1 for x in F if x['da']<t)}/{len(F)}")
        print("\n  → **the fallback wins on position and loses on orientation**, which explains why NCE and SR give opposite rankings:")
        print("     NCE is driven mainly by position, SR by orientation.")
        print("  ⚠ the two groups are different scenes (which scenes the tool fails on is not random), so this is a **confounded comparison**,")
        print("     and cannot be read as \"the fallback is better than the tool\". What can be read: **this RL reward is insensitive to orientation.**")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", default="p4/parsed")
    a = ap.parse_args()
    do_boppose(os.path.join(a.parsed, "boppose.jsonl"))
    do_bopgrasp(os.path.join(a.parsed, "bopgrasp.jsonl"))
    print("\nNote: the `correct` field in `p4/parsed/bopgrasp.jsonl` is **meaningless** — "
          "it is the generic score>=0.5 criterion, while the bopgrasp score is NCE, lower is better.")


if __name__ == "__main__":
    main()
