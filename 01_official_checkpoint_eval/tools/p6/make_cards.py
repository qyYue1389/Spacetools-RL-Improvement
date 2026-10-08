#!/usr/bin/env python3
"""Render review cards for the P6 wrong answers awaiting manual classification: original image + points returned by tools + question/GT/model answer/criterion.

The question a card answers is always one of three:
  1 tool error       — some point landed on the wrong object
  3b frame/semantics — the points are all correct, but the GT does not use image-plane semantics
  6 label problem    — points correct, semantics correct too; the GT itself is questionable
"""
import base64, io, json, re, os, sys
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--data", default="../eval-benchmarks/data", help="data/ directory of eval-benchmarks")
_ap.add_argument("--parsed", default="p4/parsed")
_ap.add_argument("--out", default="p6/manual/cards")
_ap.add_argument("--pending", default="p6/manual/pending31.json")
_A = _ap.parse_args()
DATA, PARSED, OUT = _A.data, _A.parsed, _A.out
PT = re.compile(r"\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)")

# must use a font with CJK glyphs, otherwise the caption text is all boxes
CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
CJKB = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
DJ = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
def _f(size, bold=False):
    for path in ([CJKB, CJK] if bold else [CJK, CJKB]) + [DJ]:
        if os.path.exists(path):
            try: return ImageFont.truetype(path, size)
            except Exception: pass
    return ImageFont.load_default()
font, fontb, fonts = _f(15), _f(16, True), _f(13)

COLORS = [(0, 122, 255), (255, 122, 0), (0, 190, 90), (200, 0, 160)]


def load_img(cell):
    if hasattr(cell, "tolist"):
        cell = cell.tolist()
    if isinstance(cell, (list, tuple)) and cell:
        cell = cell[0]
    if isinstance(cell, dict):
        for k in ("bytes", "image", "path"):
            if cell.get(k) is not None:
                cell = cell[k]; break
    if isinstance(cell, str):
        if cell.startswith("data:"):
            cell = cell.split(",", 1)[1]
        cell = base64.b64decode(cell)
    return Image.open(io.BytesIO(cell)).convert("RGB")


def detections(rec):
    out = []
    for turn in rec["trajectory"]:
        for c, r in zip(turn.get("tool_calls") or [], turn.get("tool_responses") or []):
            if c["name"].endswith(("detect_one", "detect_all")):
                pts = PT.findall(r.get("text") or "")
                out.append((c["arguments"].get("obj_name", ""), pts[0] if pts else None,
                            (r.get("text") or "")[:70]))
            elif c["name"].endswith("index_at"):
                a = c["arguments"]
                u, v = a.get("u"), a.get("v")
                txt = (r.get("text") or "")[:70]
                out.append((f"index_at -> {txt.split('is')[-1].strip()[:12]}",
                            (u, v) if u is not None else None, txt))
    return out


def wrap(d, text, w, f):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if d.textlength(t, font=f) <= w:
            cur = t
        else:
            lines.append(cur); cur = wd
    if cur:
        lines.append(cur)
    return lines


def card(case, rec, row, path):
    im = load_img(row["images"])
    W = 760
    im = im.resize((W, int(im.height * W / im.width)))
    dets = detections(rec)
    pts = [(n, p, t) for n, p, t in dets if p]

    d = ImageDraw.Draw(im)
    for i, (name, p, _) in enumerate(pts[:4]):
        x, y = float(p[0]) * im.width, float(p[1]) * im.height
        c = COLORS[i % 4]
        r = 11
        d.ellipse([x-r, y-r, x+r, y+r], outline=c, width=4)
        d.line([x-r-7, y, x+r+7, y], fill=c, width=2)
        d.line([x, y-r-7, x, y+r+7], fill=c, width=2)
        lab = f"{i+1}"
        d.rectangle([x+r+4, y-r-4, x+r+22, y-r+16], fill=c)
        d.text((x+r+9, y-r-3), lab, fill=(255, 255, 255), font=fontb)

    # caption
    lines = []
    lines.append(("B", f"[{case['bench']}] #{case['sid']}   —— {case['why']}"))
    for L in wrap(d, case["q"].replace("\n", " "), W-24, font):
        lines.append(("", L))
    m = f"  margin {case['margin']:.3f}" if case.get("margin") is not None else ""
    lines.append(("B", f"GT = {case['gt']}    model = {case['ans']}    rule = {case.get('rule')}{m}"))
    lines.append(("", "tool returned:"))
    for i, (name, p, t) in enumerate(pts[:4]):
        lines.append(("S", f"  [{i+1}] {name} -> ({p[0]}, {p[1]})"))
    for name, p, t in dets:
        if p is None:
            lines.append(("S", f"  (no point) {name} -> {t}"))

    lh = 22
    H = im.height + 16 + lh * len(lines) + 14
    cv = Image.new("RGB", (W, H), (250, 250, 248))
    cv.paste(im, (0, 0))
    dd = ImageDraw.Draw(cv)
    y = im.height + 12
    for kind, t in lines:
        f = {"B": fontb, "S": fonts}.get(kind, font)
        dd.text((12, y), t, fill=(20, 20, 24), font=f)
        y += lh if kind != "S" else lh - 4
    cv.save(path, quality=92)


def main():
    os.makedirs(OUT, exist_ok=True)
    cases = json.load(open(_A.pending))
    cases += [dict(bench="blinkdepth", sid=s, why="criterion A undecidable (3 index_at calls)",
                   margin=None, rule=None, ans="?", gt="?", q="") for s in (69, 86)]
    parsed, dfs = {}, {}
    made = []
    for c in cases:
        b = c["bench"]
        if b not in parsed:
            parsed[b] = {json.loads(l)["sample_id"]: json.loads(l)
                         for l in open(f"{PARSED}/{b}.jsonl")}
            dfs[b] = pd.read_parquet(f"{DATA}/{b}.parquet")
        rec = parsed[b][c["sid"]]
        if not c["q"]:
            c["q"] = rec["question"][:160]
            c["gt"] = str(rec["gt"])[:40]; c["ans"] = str(rec["raw_answer"])[:40]
        p = f"{OUT}/{b}_{c['sid']:03d}.jpg"
        card(c, rec, dfs[b].iloc[c["sid"]], p)
        made.append(p)
    json.dump(cases, open(os.path.join(OUT, "pending_all.json"), "w"), ensure_ascii=False, indent=1)
    print(f"{len(made)} cards -> {OUT}/")
    for p in made:
        print(" ", p)


if __name__ == "__main__":
    main()
