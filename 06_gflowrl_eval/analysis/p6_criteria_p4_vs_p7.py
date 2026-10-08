#!/usr/bin/env python3
"""Run P6's three criteria + the fit/Vacant analysis on the parsed records of both P4 and P7.

Original P6 scripts: tools/p6/p6_split.py (criteria A/B), tools/p6/p6_split3.py (question type and Vacant).
Merged here into one, with the parsed directory parameterized, for per-sample comparison.
"""
import json, os, re, math, statistics
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root
from collections import Counter

DIRS = [("P4(official checkpoint)", f"{ROOT}/01_official_checkpoint_eval/p4/parsed"),
        ("P7(C' step85)", f"{ROOT}/06_gflowrl_eval/parsed")]
DET = re.compile(r"Detected (\d+) instance\(s\)[^:]*:\s*(\[.*?\])")
PT  = re.compile(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)")
PIX = re.compile(r"Pixel value at \(([\d.]+), ([\d.]+)\) is ([\d.eE+-]+)")
load = lambda d, b: [json.loads(l) for l in open(f"{d}/{b}.jsonl", encoding="utf-8")]

def qtype(r):
    q = (r.get("question") or "").lower()
    if "fit" in q: return "fit (needs free space)"
    if "in front of" in q or "behind" in q: return "front/behind (needs depth order)"
    return "relation (decidable in 2D)"

def yesno(r):
    t = str(r.get("raw_answer") or "").strip().lower()
    return "yes" if "yes" in t else ("no" if "no" in t else "?")

print("=" * 78)
print("P6's criteria rerun on P4 and P7")
print("=" * 78)

# ---------- criterion A: do depth questions follow "pick the one measured closer" ----------
print("\n### criterion A — depth questions: does the model follow \"pick the one with smaller measured depth\" verbatim")
for b in ["blinkdepth", "cvb3ddepth"]:
    print(f"  {b}")
    for tag, d in DIRS:
        rs = load(d, b); follow = viol = skip = 0
        for r in rs:
            dets = []; pix = {}
            for t in r["trajectory"]:
                for c, x in zip(t.get("tool_calls", []), t.get("tool_responses", [])):
                    if x is None: continue
                    txt = x.get("text") or ""
                    if c["name"] in ("roborefer.detect_one", "vlm.detect_one"):
                        m = DET.search(txt)
                        if m:
                            p = PT.findall(m.group(2))
                            if p: dets.append((round(float(p[0][0]), 3), round(float(p[0][1]), 3)))
                    if c["name"] == "vision_ops.index_at":
                        m = PIX.search(txt)
                        if m: pix[(round(float(m.group(1)), 3), round(float(m.group(2)), 3))] = float(m.group(3))
            ans = (r["raw_answer"] or "").strip().upper()
            ans = ans[1] if ans.startswith("(") and len(ans) > 1 else (ans[0] if ans else "")
            if len(dets) < 2 or len(pix) < 2 or ans not in "AB": skip += 1; continue
            dA, dB = pix.get(dets[0]), pix.get(dets[1])
            if dA is None or dB is None: skip += 1; continue
            rule = "A" if dA < dB else "B"
            follow += (ans == rule); viol += (ans != rule)
        n = follow + viol
        print(f"    {tag:16s} follows {follow:4d} breaks {viol:3d} undecidable {skip:3d}"
              f"   follow rate {100*follow/n if n else 0:.1f}%")

# ---------- criterion B: is pointing a verbatim pass-through of the tool output ----------
print("\n### criterion B — pointing questions: is the answer a verbatim pass-through of the tool output")
for b in ["reflocation", "refplacement", "refunseen"]:
    line = f"  {b:14s}"
    for tag, d in DIRS:
        rs = load(d, b); same = tot = 0
        for r in rs:
            det = None
            for t in r["trajectory"]:
                for x in t.get("tool_responses", []):
                    m = DET.search((x or {}).get("text") or "")
                    if m and det is None:
                        p = PT.findall(m.group(2))
                        if p: det = (round(float(p[0][0]), 3), round(float(p[0][1]), 3))
            p = PT.findall(r["raw_answer"] or "")
            if det is None or not p: continue
            tot += 1
            same += ((round(float(p[0][0]), 3), round(float(p[0][1]), 3)) == det)
        line += f"   {tag} pass-through {same}/{tot}"
    print(line)

# ---------- robospatial VQA question types ----------
print("\n### robospatial VQA by question type (the three-way split of P6 §6.4)")
print(f"  {'question type':24s} {'n':>4s}  {'P4 accuracy':>10s} {'P7 accuracy':>10s}   {'P4 ans yes':>8s} {'P7 ans yes':>8s}")
tabs = {}
for tag, d in DIRS:
    rs = load(d, "robospatial")
    vqa = [r for r in rs if str(r["gt"]).strip().lower() in ("yes", "no")]
    t = {}
    for r in vqa:
        k = qtype(r); e = t.setdefault(k, [0, 0, 0, 0])
        e[0] += 1; e[1] += bool(r["correct"]); e[2] += (yesno(r) == "yes")
        e[3] += (str(r["gt"]).strip().lower() == "yes")
    tabs[tag] = t
for k in sorted(tabs[DIRS[0][0]], key=lambda x: -tabs[DIRS[0][0]][x][0]):
    a = tabs[DIRS[0][0]][k]; c = tabs[DIRS[1][0]][k]
    print(f"  {k:24s} {a[0]:4d}  {100*a[1]/a[0]:9.1f}% {100*c[1]/c[0]:9.1f}%   "
          f"{a[2]:4d}/{a[0]:<4d} {c[2]:4d}/{c[0]:<4d}   (GT=yes {a[3]})")

# ---------- Vacant: pass-through vs point override ----------
print("\n### robospatial Vacant: when the model changes the point given by roborefer, does it make it better or worse")
for tag, d in DIRS:
    rs = load(d, "robospatial")
    vac = [r for r in rs if str(r["gt"]).strip().startswith("[")]
    gtpts = lambda s: [(float(x), float(y)) for x, y in PT.findall(s)]
    mind = lambda p, G: min(math.hypot(p[0]-g[0], p[1]-g[1]) for g in G) if G else None
    mod = []; passthru = 0; passthru_ok = 0
    for r in vac:
        det = None
        for t in r["trajectory"]:
            for x in t.get("tool_responses", []):
                m = DET.search((x or {}).get("text") or "")
                if m and det is None:
                    p = PT.findall(m.group(2))
                    if p: det = (round(float(p[0][0]), 3), round(float(p[0][1]), 3))
        p = PT.findall(r["raw_answer"] or "")
        if det is None or not p: continue
        a = (round(float(p[0][0]), 3), round(float(p[0][1]), 3))
        if a == det:
            passthru += 1; passthru_ok += bool(r["correct"]); continue
        G = gtpts(str(r["gt"]))
        if not G: continue
        mod.append((mind(det, G), mind(a, G), r["correct"]))
    better = sum(1 for d0, d1, _ in mod if d1 < d0)
    worse  = sum(1 for d0, d1, _ in mod if d1 > d0)
    okmod  = sum(1 for *_, ok in mod if ok)
    print(f"  {tag:16s} pass-through {passthru:3d} (accuracy {100*passthru_ok/passthru if passthru else 0:.1f}%)  "
          f"point override {len(mod):3d} (accuracy {100*okmod/len(mod) if mod else 0:.1f}%)  "
          f"better {better} worse {worse}  "
          f"median distance of original point {statistics.median(d0 for d0,_,_ in mod):.4f} -> after change {statistics.median(d1 for _,d1,_ in mod):.4f}")

# ---------- chain collapse overview ----------
print("\n### tool-orchestration collapse degree (the quantity in item 1 of P6 §7)")
print(f"  {'benchmark':14s} {'n':>4s}  {'P4 main chain':>9s} {'P7 main chain':>9s}  {'P4 kinds':>7s} {'P7 kinds':>7s}")
for b in ["robospatial", "reflocation", "refplacement", "refunseen",
          "blinkdepth", "cvb2drelation", "cvb3ddepth", "boppose", "bopgrasp"]:
    row = [b, None, None, None, None, None]
    for j, (tag, d) in enumerate(DIRS):
        rs = load(d, b)
        c = Counter(r["chain_signature"] for r in rs)
        row[1] = len(rs)
        row[2 + j] = 100.0 * c.most_common(1)[0][1] / len(rs)
        row[4 + j] = len(c)
    print(f"  {row[0]:14s} {row[1]:4d}  {row[2]:8.1f}% {row[3]:8.1f}%  {row[4]:7d} {row[5]:7d}")
