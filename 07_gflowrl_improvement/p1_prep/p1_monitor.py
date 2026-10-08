#!/usr/bin/env python3
"""P1 monitor: from verl rollout / eval dumps (JSONL), compute the point-override rate and the 0.05-grid fraction for pointing questions.

During training set trainer.rollout_data_dir=<dir>; each step writes <dir>/<step>.jsonl; eval dumps have the same format.
Classification definition matches P0 (p2.sh): all answer coordinates appear in the set of coordinates returned by the tool (exact match to 3 decimals) = pass-through, otherwise = point override.

Usage:
  p1_monitor.py [--window 10] [--upper] [--boot 2000] <dir or jsonl ...>
    --window  steps per window (eval dumps have no step number; each file is its own window)
    --upper   forced pass-through: replace point-override samples with the first point of the last tool return, and recompute accuracy with the same convex-hull scoring as training

  Column "tool pt hit" = fraction where the first point of the last tool return falls inside the GT convex hull (only for questions whose GT is a list of points).
  Column "object-only query" = among samples that called a tool, fraction whose query names only the anchor object (e.g. "cup") — the primary monitoring metric of P1.
  P0: SFT 34.4% · C′ 28.1% · GRPO 10.9%; the Vacant gap is entirely determined by it; point override and tool-point hits are both its consequences.
"""
import argparse, glob, json, os, random, re, sys
from collections import defaultdict

NUM = re.compile(r'\(\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*\)')
TOOL = re.compile(r'<tool_response>(.*?)</tool_response>', re.S)
ANS = re.compile(r'<answer>(.*?)</answer>', re.S)
EXCLUDE = re.compile(r'Grasp center|eight normalized|cuboid corners|yes or no|\(A\)', re.I)
RS_VAC = re.compile(r'Pinpoint several points within the vacant space', re.I)
CALL = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.S)
# query contains a location / vacant-space description = target; names only the anchor object (e.g. "cup") = object. P0: object queries tool-point hit 0–3%, target 56%
TGT = re.compile(r'\b(point|vacant|space|spot|area|place|placing|free|empty|left of|right of|in front of|behind|above|below|next to)\b', re.I)


def query_type(out):
    qs = []
    for c in CALL.findall(out):
        try:
            qs.append(json.loads(c).get('arguments', {}).get('obj_name', '') or '')
        except Exception:
            pass
    if not qs:
        return None
    return 'target' if any(TGT.search(q) for q in qs) else 'object'


def pts(s):
    return [(float(a), float(b)) for a, b in NUM.findall(s or '')]


def r3(ps):
    return set((round(x, 3), round(y, 3)) for x, y in ps)


def category(r):
    g = r.get('gts')
    if isinstance(g, str) and g.strip().lower() in ('yes', 'no'):
        return None
    if EXCLUDE.search(r.get('input', '')) or not pts(str(g)):
        return None
    return 'rs_vacant' if RS_VAC.search(r.get('input', '')) else 'point'


def classify(out):
    blocks = [pts(b) for b in TOOL.findall(out)]
    tool = [p for b in blocks for p in b]
    last = next((b for b in reversed(blocks) if b), [])
    a = ANS.findall(out)
    ans = pts(a[-1]) if a else []
    if not ans:
        return 'no-answer', last, ans
    if not tool:
        return 'no-tool', last, ans
    # the tool point in the return value = the last tool return that has coordinates (forced pass-through uses its first point)
    return ('passthrough' if r3(ans) <= r3(tool) else 'modified'), last, ans


def grid05(ans):
    return bool(ans) and all(abs(v * 20 - round(v * 20)) < 1e-6 for p in ans for v in p)


# ---- same convex-hull scoring as verl/utils/reward_score/robos_all.py (convex_hull, first point only)
def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def hull(ps):
    ps = sorted(ps)
    if len(ps) <= 1:
        return ps
    lo, up = [], []
    for p in ps:
        while len(lo) >= 2 and _cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(ps):
        while len(up) >= 2 and _cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def inside(p, poly):
    if not poly:
        return False
    if len(poly) == 1:
        return p == poly[0]
    if len(poly) == 2:
        (x0, y0), (x1, y1) = poly
        if abs((x1 - x0) * (p[1] - y0) - (y1 - y0) * (p[0] - x0)) > 1e-9:
            return False
        return (p[0] - x0) * (p[0] - x1) + (p[1] - y0) * (p[1] - y1) <= 1e-9
    sign = None
    for i in range(len(poly)):
        cp = _cross(poly[i], poly[(i + 1) % len(poly)], p)
        if abs(cp) < 1e-12:
            continue
        if sign is None:
            sign = cp > 0
        elif sign != (cp > 0):
            return False
    return True


def load(paths):
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, '*.jsonl'))) if os.path.isdir(p) else [p]
    for f in files:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                r['_file'] = os.path.basename(f)
                yield r


def rate(rows):
    pt = sum(k == 'passthrough' for k, *_ in rows)
    md = sum(k == 'modified' for k, *_ in rows)
    return md / (pt + md) if pt + md else float('nan')


def boot_se(by_prompt, n_boot):
    keys = list(by_prompt)
    if len(keys) < 2:
        return float('nan')
    vals = []
    for _ in range(n_boot):
        rows = [x for k in random.choices(keys, k=len(keys)) for x in by_prompt[k]]
        v = rate(rows)
        if v == v:
            vals.append(v)
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--window', type=int, default=10)
    ap.add_argument('--upper', action='store_true')
    ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args()
    random.seed(0)

    win = defaultdict(lambda: defaultdict(list))   # (window, cat) -> prompt -> rows
    for r in load(a.paths):
        cat = category(r)
        if cat is None:
            continue
        k, tool, ans = classify(r.get('output', ''))
        step = r.get('step', 0) or 0
        w = r['_file'] if step == 0 else f'step {step // a.window * a.window}-{step // a.window * a.window + a.window - 1}'
        ok = float(r.get('score', 0)) >= 0.5
        gt = pts(str(r['gts']))
        # convex-hull scoring only makes sense for RoboSpatial-style GT (a list of points); RefSpatial GT is a single point, not counted
        hit = inside(tool[0], hull(gt)) if (tool and len(gt) >= 3) else None
        forced = hit if (k == 'modified' and hit is not None) else ok
        win[(w, cat)][r.get('input', '') + str(r.get('gts'))].append(
            (k, ok, grid05(ans) if k == 'modified' else False, forced, hit, query_type(r.get('output', ''))))

    hdr = f'{"window":24s} {"cat":10s} {"prompts":>7s} {"rollouts":>8s} {"pass-through":>6s} {"override":>6s} {"override rate":>7s} {"±SE":>6s} {"grid":>5s} {"pass-through correct":>6s} {"override correct":>6s} {"tool pt hit":>7s} {"object-only query":>7s}'
    if a.upper:
        hdr += f' {"actual correct":>6s} {"forced pass-through correct":>9s}'
    print(hdr)
    for (w, cat), bp in sorted(win.items()):
        rows = [x for v in bp.values() for x in v]
        pt = [x for x in rows if x[0] == 'passthrough']
        md = [x for x in rows if x[0] == 'modified']
        hitr = lambda xs: (lambda h: f'{100 * sum(h) / len(h):.1f}%' if h else '-')([x[4] for x in xs if x[4] is not None])
        objr = lambda xs: (lambda q: f'{100 * sum(x == "object" for x in q) / len(q):.1f}%' if q else '-')([x[5] for x in xs if x[5]])
        acc = lambda xs: f'{100 * sum(x[1] for x in xs) / len(xs):.1f}%' if xs else '-'
        line = (f'{w:24s} {cat:10s} {len(bp):7d} {len(rows):8d} {len(pt):6d} {len(md):6d} '
                f'{100 * rate(rows):6.1f}% {100 * boot_se(bp, a.boot):5.1f} {sum(x[2] for x in md):5d} {acc(pt):>6s} {acc(md):>6s} '
                f'{hitr(rows):>7s} {objr(rows):>7s}')
        if a.upper:
            line += f' {sum(x[1] for x in rows):6d} {sum(x[3] for x in rows):9d}'
        print(line)


if __name__ == '__main__':
    main()
