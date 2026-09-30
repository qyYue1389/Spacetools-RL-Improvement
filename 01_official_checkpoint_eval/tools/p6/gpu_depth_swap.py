#!/usr/bin/env python3
"""P6 实验 B:深度工具对比 —— 同样不需要 policy。

为什么可以绕开 policy:P6 的离线分析证明,在 cvb3ddepth 上模型 99.8%、
blinkdepth 上 95.6% 严格遵守同一条规则 —— **回答测得深度更小的那个点**。
所以「换一个深度工具能涨多少分」= 用同样的两个探测点跑新深度工具,
重新套这条规则。探测点直接取自 P4 的 dump(模型当时实际探的点),
所以 pointing 那一环保持不变,变量只有深度工具本身。

**代理的误差已量化**:用基线深度重放规则得 674/714,而基线实际是 677/714,
差 3 个样本(0.4%)—— 正是那几个「违反规则却蒙对」的。对比时用重放值当基准,
不要用 677,否则会把这 0.4% 算到新工具头上。

用法(Toolshed 必须已经起来):

    conda run -n spacetools-rl python tools/p6/gpu_depth_swap.py \
        --probes p6/probes/depth_probes.jsonl \
        --data-dir /workspace/eval-benchmarks \
        --out p6/swap/depth_baseline.jsonl

先不带 --new-depth 跑一次:它用**现有的 depth_estimator** 重跑,
应当重现 674/714。对不上就说明这条通路有问题,别往下走。
换工具时把新的深度工具注册进 Toolshed,再用 --tool 指定它的名字。
"""
import argparse, json, os, re, io, base64, time
from collections import defaultdict

BENCH2PARQUET = {
    "blinkdepth": "data/blinkdepth.parquet",
    "cvb3ddepth": "data/cvb3ddepth.parquet",
}
PIXVAL = re.compile(r"is\s+([\d.eE+-]+)")


def load_image(cell):
    from PIL import Image
    import numpy as np
    # 同 gpu_pointing_swap.py:pandas 把 parquet 的 list<struct> 读成
    # dtype=object 的 ndarray,不是 list。
    if isinstance(cell, np.ndarray):
        cell = cell.reshape(-1)[0] if cell.size else None
    if isinstance(cell, (list, tuple)) and cell:
        cell = cell[0]
    if isinstance(cell, dict):
        for k in ("bytes", "image", "path"):
            if k in cell and cell[k] is not None:
                cell = cell[k]; break
    if isinstance(cell, str):
        if cell.startswith("data:"):
            cell = cell.split(",", 1)[1]
        cell = base64.b64decode(cell)
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(cell)).convert("RGB")
    raise TypeError(f"无法识别的 image 单元格:{type(cell)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tool", default="depth_estimator", help="Toolshed 里深度工具的名字")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd, ray
    from toolshed import get_toolkit

    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)
    toolkit = get_toolkit()
    depth_tool = getattr(toolkit, args.tool)
    vops = toolkit.vision_ops

    probes = [json.loads(l) for l in open(args.probes, encoding="utf-8")]
    if args.limit:
        probes = probes[: args.limit]
    by_bench = defaultdict(list)
    for p in probes:
        by_bench[p["benchmark"]].append(p)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fout = open(args.out, "w", encoding="utf-8")
    summary = {}

    for bench, ps in by_bench.items():
        df = pd.read_parquet(os.path.join(args.data_dir, BENCH2PARQUET[bench]))
        n = ok = replay_base = failed = 0
        t0 = time.time()
        for p in ps:
            row = df.iloc[p["sample_id"]]
            dA = dB = None
            try:
                img = load_image(row["images"])
                res = depth_tool.estimate_depth(img)
                dm = (res.variables or {}).get("depth_map")
                if dm is None:
                    failed += 1
                else:
                    rA = vops.index_at(dm, p["pointA"][0], p["pointA"][1])
                    rB = vops.index_at(dm, p["pointB"][0], p["pointB"][1])
                    mA, mB = PIXVAL.search(rA.text or ""), PIXVAL.search(rB.text or "")
                    if mA and mB:
                        dA, dB = float(mA.group(1)), float(mB.group(1))
                    else:
                        failed += 1
            except Exception as e:
                failed += 1
                print(f"  [EXC] {bench}#{p['sample_id']}: {e}", flush=True)

            pred = None if dA is None else ("A" if dA < dB else "B")
            base_pred = "A" if p["baseline_depthA"] < p["baseline_depthB"] else "B"
            gt = p["gt"].strip().upper()[:1]
            n += 1
            ok += (pred == gt)
            replay_base += (base_pred == gt)
            fout.write(json.dumps({**p, "tool": args.tool, "new_depthA": dA,
                                   "new_depthB": dB, "new_pred": pred,
                                   "replay_baseline_pred": base_pred},
                                  ensure_ascii=False) + "\n")
            if n % 50 == 0:
                print(f"  {bench} {n}/{len(ps)}  {time.time()-t0:.0f}s", flush=True)
        summary[bench] = (n, ok, replay_base, failed)
        print(f"[{bench}] n={n}  新工具 {ok} ({100*ok/n:.2f}%)  "
              f"基线重放 {replay_base} ({100*replay_base/n:.2f}%)  失败 {failed}  "
              f"{time.time()-t0:.0f}s", flush=True)
    fout.close()

    print("\n=== 汇总 ===")
    print(f"{'benchmark':14}{'n':>6}{'新工具':>10}{'基线重放':>11}{'差值':>10}")
    tn = tok = tb = 0
    for b, (n, ok, rb, failed) in summary.items():
        print(f"{b:14}{n:6}{100*ok/n:9.2f}%{100*rb/n:10.2f}%{100*(ok-rb)/n:+9.2f}")
        tn += n; tok += ok; tb += rb
    print(f"{'合计':14}{tn:6}{100*tok/tn:9.2f}%{100*tb/tn:10.2f}%{100*(tok-tb)/tn:+9.2f}")
    print(f"\n写入 {args.out}")
    print("\n注意:基线重放列应当精确等于 —— blinkdepth 97/115 (84.35%)、"
          "cvb3ddepth 577/599 (96.33%)、合计 674/714 (94.40%)。")
    print("它是纯算术(不调工具),所以必然复现;对不上说明 probes 文件被改过。")
    print("而『新工具』一列不带 --tool 时应当≈重放值,差异来自 DepthPro 自身的非确定性。")


if __name__ == "__main__":
    main()
