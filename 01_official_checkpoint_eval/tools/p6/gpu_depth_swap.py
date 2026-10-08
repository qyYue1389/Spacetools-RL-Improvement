#!/usr/bin/env python3
"""P6 experiment B: depth-tool comparison — likewise does not need the policy.

Why the policy can be bypassed: the P6 offline analysis showed that the model strictly follows one rule on 99.8% of cvb3ddepth and
95.6% of blinkdepth — **answer with the point whose measured depth is smaller**.
So "how much does swapping in another depth tool gain" = run the new depth tool on the same two probe points,
then reapply this rule. The probe points come straight from the P4 dump (the points the model actually probed at the time),
so the pointing stage stays unchanged and the only variable is the depth tool itself.

**The proxy's error is quantified**: replaying the rule with the baseline depth gives 674/714, while the actual baseline is 677/714,
a gap of 3 samples (0.4%) — exactly the few that "broke the rule but guessed right". When comparing, use the replay value as the reference,
not 677, otherwise this 0.4% gets attributed to the new tool.

Usage (Toolshed must already be up):

    conda run -n spacetools-rl python tools/p6/gpu_depth_swap.py \
        --probes p6/probes/depth_probes.jsonl \
        --data-dir /workspace/eval-benchmarks \
        --out p6/swap/depth_baseline.jsonl

First run once without --new-depth: it reruns with the **existing depth_estimator**,
and should reproduce 674/714. If it doesn't match, this path is broken; don't go further.
To swap tools, register the new depth tool in Toolshed, then pass its name with --tool.
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
    # Same as gpu_pointing_swap.py: pandas reads the parquet list<struct> as
    # an ndarray with dtype=object, not a list.
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
    raise TypeError(f"unrecognized image cell: {type(cell)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tool", default="depth_estimator", help="name of the depth tool in Toolshed")
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
        print(f"[{bench}] n={n}  new tool {ok} ({100*ok/n:.2f}%)  "
              f"baseline replay {replay_base} ({100*replay_base/n:.2f}%)  failed {failed}  "
              f"{time.time()-t0:.0f}s", flush=True)
    fout.close()

    print("\n=== Summary ===")
    print(f"{'benchmark':14}{'n':>6}{'new tool':>10}{'baseline replay':>11}{'diff':>10}")
    tn = tok = tb = 0
    for b, (n, ok, rb, failed) in summary.items():
        print(f"{b:14}{n:6}{100*ok/n:9.2f}%{100*rb/n:10.2f}%{100*(ok-rb)/n:+9.2f}")
        tn += n; tok += ok; tb += rb
    print(f"{'total':14}{tn:6}{100*tok/tn:9.2f}%{100*tb/tn:10.2f}%{100*(tok-tb)/tn:+9.2f}")
    print(f"\nwrote {args.out}")
    print("\nNote: the baseline-replay column should equal exactly — blinkdepth 97/115 (84.35%), "
          "cvb3ddepth 577/599 (96.33%), total 674/714 (94.40%).")
    print("It is pure arithmetic (no tool calls), so it must reproduce; a mismatch means the probes file was modified.")
    print("The 'new tool' column without --tool should be ≈ the replay value; the difference comes from DepthPro's own nondeterminism.")


if __name__ == "__main__":
    main()
