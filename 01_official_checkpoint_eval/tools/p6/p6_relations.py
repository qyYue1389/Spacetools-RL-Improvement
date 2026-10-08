#!/usr/bin/env python3
"""P6: automatic criterion for relation questions (rebuilt version) — reads only `p4/parsed/`, needs neither GPU nor manual work.

Covers two benchmarks that were previously sent back to manual review:

  * `cvb2drelation`  — the old criterion had a self-consistency rate of only about 93%, was judged unusable and sent back to manual review.
    This version fixes it to **99.83%**. All three fixes target the two error sources named in the old notes:
      1. take subject/reference from the clause **"where is the X located with respect to the Y"**,
         not from the leading "relative positions of A and B" — the latter's order is not necessarily the same as the former's;
      2. strip modifiers like "(annotated by the red box)" before matching against `obj_name`;
      3. **read the option-letter-to-word mapping per sample**, do not assume (A)=left / (A)=above.
  * `robospatial` VQA relation questions — new. Self-consistency rate 98.5%.

The criterion itself is simple: two detection points, compare x or y in the image plane. All of its value is in the **validation**:
> **On samples the model got right, do the rule and the model agree?** The disagreement rate is an upper bound on the criterion's own error.
> The old criterion failed at this step (93%), and there are only 35 wrong answers in total — not enough signal-to-noise, so it had to go back to manual review.
> An automatic criterion that is not accurate enough is worse than none.

Two more things are reported; they are not criteria, but they decide how the conclusion is written:

  * **Rule accuracy against GT**. It answers "is the 2D image plane the correct semantics for this benchmark".
    `cvb2drelation` 95.4% (yes), `robospatial` VQA 78.6% (**no**).
    So "obeys the rule yet answers wrong" can be classed as tool error in the former, but **not** in the latter — there it is still mixed with
    frame/semantics mismatch (class 3b) and label problems (class 6), which can only be separated by looking at the image.
  * **Margin on the deciding axis**. When the margin is near 0 the rule degenerates into a coin flip; those samples must be pulled out separately
    and cannot be counted as "obey" or "violate".

Usage (from the repo root):

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

# only these four relations can be decided from two 2D points. front/behind need depth, see § "structural gap".
DECIDABLE = {
    "above":    ("y", lambda a, b: a[1] < b[1]),
    "below":    ("y", lambda a, b: a[1] > b[1]),
    "left of":  ("x", lambda a, b: a[0] < b[0]),
    "right of": ("x", lambda a, b: a[0] > b[0]),
}
TIE = 0.02          # margin on the deciding axis below this -> rule degenerates, filed separately


def norm(s):
    s = s.lower()
    s = re.sub(r"\(annotated by the red box\)", "", s)
    return re.sub(r"\s+", " ", s).strip(" .")


def detections(rec):
    """Collect {obj_name: first returned point} in call order."""
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
    """Exact match first; otherwise a unique substring match. **If the match is not unique, return None** —
    better to call it undecidable than to guess subject/reference wrong; that is exactly what killed the old criterion."""
    if name in det:
        return det[name]
    cand = [v for k, v in det.items() if name in k or k in name]
    return cand[0] if len(cand) == 1 else None


def judge(rows, label, out=None):
    dec = [x for x in rows if x["rule"]]
    if not dec:
        print(f"\n=== {label}: no decidable samples"); return
    corr = [x for x in dec if x["correct"]]
    wrong = [x for x in dec if not x["correct"]]
    agree_c = sum(1 for x in corr if x["rule"] == x["ans"])

    print(f"\n=== {label}")
    print(f"  decidable {len(dec)}/{len(rows)}")
    print(f"  [criterion check] correct samples n={len(corr)}: rule agrees with model {agree_c} "
          f"({100*agree_c/len(corr):.2f}%), disagrees {len(corr)-agree_c}"
          f"   <- upper bound on the criterion's own error")
    if agree_c / len(corr) < 0.97:
        print("  ⚠ self-consistency below 97%, criterion unusable, back to manual review (the old cvb2drelation criterion died here)")

    rule_gt = sum(1 for x in dec if x["rule"] == x["gt"])
    print(f"  [semantics check] rule vs GT {rule_gt}/{len(dec)} = {100*rule_gt/len(dec):.1f}%"
          f" · model vs GT {len(corr)}/{len(dec)} = {100*len(corr)/len(dec):.1f}%")
    if abs(rule_gt - len(corr)) <= 2:
        print("  → the model's score is almost the same as the rule's: **on this question type the model is this rule**,"
              "its ceiling is the rule's ceiling")
    if rule_gt / len(dec) < 0.90:
        print("  ⚠ the image-plane rule is not the correct semantics for this benchmark."
              "\"obeys the rule yet answers wrong\" **cannot** be classed directly as tool error — it is still mixed with frame/semantics (3b) and labels (6)")

    tie   = [x for x in wrong if x["margin"] is not None and x["margin"] < TIE]
    keep  = [x for x in wrong if x not in tie]
    obey  = [x for x in keep if x["rule"] == x["ans"]]
    viol  = [x for x in keep if x["rule"] != x["ans"]]
    print(f"  [wrong answers n={len(wrong)}] obeys rule {len(obey)} · violates rule {len(viol)} · "
          f"margin<{TIE} rule degenerates {len(tie)}")

    if obey:
        mo = [x["margin"] for x in obey if x["margin"] is not None]
        mc = [x["margin"] for x in corr if x["margin"] is not None]
        print(f"  [margin] correct group median {st.median(mc):.3f} · obeyed-but-wrong group median {st.median(mo):.3f}")
        for t in (0.05, 0.10):
            print(f"        <{t:.2f}: correct group {100*sum(1 for m in mc if m<t)/len(mc):.0f}% · "
                  f"wrong group {100*sum(1 for m in mo if m<t)/len(mo):.0f}%")

    if out is not None:
        tie_ids = {x["sid"] for x in tie}
        for x in wrong:
            v = ("rule degenerates (margin≈0, not counted as obey/violate)" if x["sid"] in tie_ids else
                 ("obeys rule" if x["rule"] == x["ans"] else "violates rule"))
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
            continue                                   # Vacant, outside the scope of this criterion
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
    judge(rows, "robospatial VQA · relation questions", out)

    # ---- structural gap: information the question needs that the model never fetched ----
    print("\n=== robospatial VQA · structural gap (not from the criterion, from the tool histogram)")
    for name, group, why in [
        ("front/behind", fb, "needs depth order; depth_estimator is right there in the tool list"),
        ("fit (free space)", fit, "needs free-space extent; no single tool gives it directly, but bbox+depth can approximate it"),
    ]:
        if not group:
            continue
        c = sum(1 for r in group if r["correct"])
        nod = sum(1 for r in group if "depth" not in r["chain_signature"])
        no_c = [r for r in group if str(r["gt"]).lower() == "no"]
        print(f"  {name:16} n={len(group):3}  accuracy {c}/{len(group)} = {100*c/len(group):.1f}%"
              f"  **depth tool not called {nod}/{len(group)}**"
              f"  GT=no accuracy {sum(1 for r in no_c if r['correct'])}/{len(no_c)}")
    if unmatched:
        print(f"  {'detection mismatch':16} n={len(unmatched):3}  "
              f"accuracy {sum(1 for r in unmatched if r['correct'])}/{len(unmatched)}  <- needs looking at the image")


def do_vacant(path):
    """RoboSpatial Vacant: pass-through vs point override, and "round coordinates eyeballed during point override".

    The Vacant answer is a point, so criterion B (verbatim pass-through) applies directly. It is known that "once the model overrides the point,
    accuracy halves"; here we also ask **what kind of number it changes it to** — if overriding the tool relies on eyeballing
    rather than computation, the coordinates will land on an artificial granularity such as 0.05.
    """
    import math
    keep, chg = [], []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if str(r["gt"]).strip().lower() in ("yes", "no"):
            continue                                   # VQA, outside the scope of this section
        det = detections(r)
        tool = next(iter(det.values()), None)
        a = PT.findall(str(r["raw_answer"]))
        if not tool or not a:
            continue
        ap = (float(a[0][0]), float(a[0][1]))
        (chg if math.dist(ap, tool) >= 0.02 else keep).append((ap, r["correct"]))

    def rnd(v, step=0.05):
        return abs(v / step - round(v / step)) < 1e-6

    print("\n=== robospatial Vacant · pass-through vs point override")
    print(f"  {'':14}{'n':>5}{'accuracy':>12}{'both coords multiple of 0.05':>24}")
    for g, name in ((keep, "verbatim pass-through"), (chg, "model point override")):
        if not g:
            continue
        ok = sum(1 for _, c in g if c)
        r5 = sum(1 for ap, _ in g if rnd(ap[0]) and rnd(ap[1]))
        print(f"  {name:14}{len(g):5}{ok:6} ({100*ok/len(g):4.1f}%){r5:14} ({100*r5/len(g):3.0f}%)")
    print("  → on pass-through the coordinates carry the tool's three decimals; once the model does it itself, they land on an artificial granularity.")
    print("     **it is not computing, it is eyeballing** — and the Vacant GT is a narrow band a few dozen pixels wide.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed", default="p4/parsed")
    ap.add_argument("--dump-cases", help="write the wrong answers one per line as jsonl, for manual classification")
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
        print(f"\nwrote {a.dump_cases} ({len(out)} wrong answers)")


if __name__ == "__main__":
    main()
