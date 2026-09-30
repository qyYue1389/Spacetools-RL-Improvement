#!/usr/bin/env python3
"""P1 监控:从 verl 的 rollout / eval dump(JSONL)算 pointing 题的改点率与 0.05 网格比例。

训练时开 trainer.rollout_data_dir=<dir>,每步产出 <dir>/<step>.jsonl;eval dump 格式相同。
判别口径与 P0(p2.sh)一致:答案坐标全部出现在工具返回坐标集合里(3 位小数精确匹配)= 透传,否则 = 改点。

用法:
  p1_monitor.py [--window 10] [--upper] [--boot 2000] <dir 或 jsonl ...>
    --window  每几步一个窗口(eval dump 没有步数,每个文件自成一窗)
    --upper   强制透传:把改点样本换成最后一次工具返回的第一个点,用训练同款凸包判分重算正确率

  列「工具点中」= 最后一次工具返回的第一个点落在 GT 凸包内的比例(只对 GT 为一串点的题)。
  列「只问物体」= 调了工具的样本里,查询只写锚物体名(如 "cup")的比例 —— P1 的主监控量。
  P0:SFT 34.4% · C′ 28.1% · GRPO 10.9%;Vacant 的差距全部由它决定,改点、工具点命中都是它的后果。
"""
import argparse, glob, json, os, random, re, sys
from collections import defaultdict

NUM = re.compile(r'\(\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*\)')
TOOL = re.compile(r'<tool_response>(.*?)</tool_response>', re.S)
ANS = re.compile(r'<answer>(.*?)</answer>', re.S)
EXCLUDE = re.compile(r'Grasp center|eight normalized|cuboid corners|yes or no|\(A\)', re.I)
RS_VAC = re.compile(r'Pinpoint several points within the vacant space', re.I)
CALL = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.S)
# 查询里带位置 / 空位描述 = target;只写锚物体名(如 "cup")= object。P0:object 查询工具点命中 0–3%,target 56%
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
    # 返回值里的工具点 = 最后一次有坐标的工具返回(强制透传时用它的第一个点)
    return ('passthrough' if r3(ans) <= r3(tool) else 'modified'), last, ans


def grid05(ans):
    return bool(ans) and all(abs(v * 20 - round(v * 20)) < 1e-6 for p in ans for v in p)


# ---- 与 verl/utils/reward_score/robos_all.py 相同的凸包判分(convex_hull,只看第一个点)
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
        # 凸包判分只对 RoboSpatial 式 GT(一串点)有意义;RefSpatial 的 GT 是单点,不算
        hit = inside(tool[0], hull(gt)) if (tool and len(gt) >= 3) else None
        forced = hit if (k == 'modified' and hit is not None) else ok
        win[(w, cat)][r.get('input', '') + str(r.get('gts'))].append(
            (k, ok, grid05(ans) if k == 'modified' else False, forced, hit, query_type(r.get('output', ''))))

    hdr = f'{"window":24s} {"cat":10s} {"prompts":>7s} {"rollouts":>8s} {"透传":>6s} {"改点":>6s} {"改点率":>7s} {"±SE":>6s} {"网格":>5s} {"透传对":>6s} {"改点对":>6s} {"工具点中":>7s} {"只问物体":>7s}'
    if a.upper:
        hdr += f' {"实际对":>6s} {"强制透传对":>9s}'
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
