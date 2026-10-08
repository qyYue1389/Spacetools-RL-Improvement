#!/usr/bin/env python3
"""P6 experiment A: pointing-tool comparison — needs neither the policy nor sglang.

Why the policy can be bypassed: the P6 offline analysis showed that on the three RefSpatial benchmarks
the model is 100% pass-through (276/276, the answer is bit-for-bit equal to what roborefer returned),
and 82.8% pass-through on robospatial Vacant. So "how much does swapping in another pointing tool gain"
can be measured directly: the same obj_name query → new tool → scored with verl's own scoring function.

Usage (in the spacetools-rl environment; Toolshed must already be up):

    conda run -n spacetools-rl python tools/p6/gpu_pointing_swap.py \
        --tool vlm \
        --probes p6/probes/pointing_probes.jsonl \
        --data-dir /workspace/eval-benchmarks \
        --out p6/swap/pointing_vlm.jsonl

--tool takes roborefer / vlm. First run once with --tool roborefer as a **regression check**:
it should reproduce the baseline accuracy (see baseline_correct in the probes); if it doesn't match,
this offline path itself is broken and none of the later comparisons can be trusted.
"""
import argparse, json, os, re, sys, io, base64, time
from collections import defaultdict

PT = re.compile(r"\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)")


class _Trainer:
    """Replicates the training_args.trainer that default_compute_score sees during eval.

    In the scoring branch taken by RoboSpatial/* and BLINK/Spatial_Relation,
    format_score_val is only assigned inside the `if training_args and hasattr(training_args,'trainer')`
    block (the other six defaults in the same block are all initialized before the if). When an offline call does not pass
    training_args, that branch's `format_score=format_score_val` raises
    UnboundLocalError outright — all 122 robospatial probes fail to score.

    Both values are pinned by evidence, not chosen here:
      format_score                 verl/trainer/config/ppo_trainer.yaml:230 = 0.0
      allow_last_response_fallback ppo_trainer.yaml:233 defaults to false,
                                   but examples/toolshed/run_eval.sh:297 overrides it to true
    """
    format_score = 0.0
    allow_last_response_fallback = True


class _EvalTrainingArgs:
    trainer = _Trainer()


_EVAL_TRAINING_ARGS = _EvalTrainingArgs()

BENCH2PARQUET = {
    "reflocation": "data/reflocation.parquet",
    "refplacement": "data/refplacement.parquet",
    "refunseen": "data/refunseen.parquet",
    "robospatial": "data/robospatial.parquet",
}


def load_image(cell):
    """The shape of the parquet images column is not fully consistent across benchmarks; try each kind here."""
    from PIL import Image
    import numpy as np
    # pandas reads a parquet list<struct> column as an ndarray with dtype=object,
    # not a list — the original isinstance(cell, (list, tuple)) didn't recognize it, so
    # the whole chain fell through to the TypeError at the end on the first hop, and the tool was never called once.
    if isinstance(cell, np.ndarray):
        cell = cell.reshape(-1)[0] if cell.size else None
    if isinstance(cell, (list, tuple)) and cell:
        cell = cell[0]
    if isinstance(cell, dict):
        for k in ("bytes", "image", "path"):
            if k in cell and cell[k] is not None:
                cell = cell[k]
                break
    if isinstance(cell, str):
        if cell.startswith("data:"):
            cell = cell.split(",", 1)[1]
        cell = base64.b64decode(cell)
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(cell)).convert("RGB")
    raise TypeError(f"unrecognized image cell: {type(cell)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True, choices=["roborefer", "vlm"])
    ap.add_argument("--probes", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N rows, for smoke tests")
    args = ap.parse_args()

    import pandas as pd
    import ray
    from toolshed import get_toolkit
    from verl.utils.reward_score import default_compute_score

    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)
    toolkit = get_toolkit()          # connects to the router already started by run_eval.sh
    tool = getattr(toolkit, args.tool)

    probes = [json.loads(l) for l in open(args.probes, encoding="utf-8")]
    if args.limit:
        probes = probes[: args.limit]
    by_bench = defaultdict(list)
    for p in probes:
        by_bench[p["benchmark"]].append(p)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fout = open(args.out, "w", encoding="utf-8")
    summary = {}
    printed_shape = False

    for bench, ps in by_bench.items():
        df = pd.read_parquet(os.path.join(args.data_dir, BENCH2PARQUET[bench]))
        n = ok = base_ok = failed = 0
        t0 = time.time()
        for p in ps:
            row = df.iloc[p["sample_id"]]
            if not printed_shape:
                print(f"[shape] images cell type = {type(row['images'])}", flush=True)
                printed_shape = True
            try:
                img = load_image(row["images"])
                res = tool.detect_one(img, p["obj_name"])
                m = PT.findall(res.text or "")
                if not m:
                    failed += 1
                    pt = None
                else:
                    pt = [round(float(m[0][0]), 4), round(float(m[0][1]), 4)]
            except Exception as e:                       # a tool failure is also a result
                failed += 1
                pt = None
                res = type("R", (), {"text": f"EXC {e}"})()

            score = 0.0
            if pt is not None:
                sol = f"<answer>[({pt[0]}, {pt[1]})]</answer>"
                gt = row["reward_model"]["ground_truth"]
                ei = row["extra_info"] if "extra_info" in row else None
                try:
                    r = default_compute_score(row["data_source"], sol, gt,
                                              extra_info=ei,
                                              training_args=_EVAL_TRAINING_ARGS)
                    score = float(r["score"] if isinstance(r, dict) else r)
                except Exception as e:
                    print(f"  [score failed] {bench}#{p['sample_id']}: {e}", flush=True)
            n += 1
            ok += score >= 0.5
            base_ok += p["baseline_correct"]
            fout.write(json.dumps({**p, "tool": args.tool, "new_point": pt,
                                   "new_score": score, "text": (res.text or "")[:200]},
                                  ensure_ascii=False) + "\n")
            if n % 25 == 0:
                print(f"  {bench} {n}/{len(ps)}  {time.time()-t0:.0f}s", flush=True)
        summary[bench] = (n, ok, base_ok, failed)
        print(f"[{bench}] n={n}  new tool {ok} ({100*ok/n:.2f}%)  "
              f"baseline {base_ok} ({100*base_ok/n:.2f}%)  tool failed {failed}  "
              f"{time.time()-t0:.0f}s", flush=True)
    fout.close()

    print("\n=== Summary ===")
    print(f"{'benchmark':16}{'n':>5}{'new tool':>10}{'baseline':>10}{'diff':>10}")
    tn = tok = tb = 0
    for b, (n, ok, base_ok, failed) in summary.items():
        print(f"{b:16}{n:5}{100*ok/n:9.2f}%{100*base_ok/n:9.2f}%{100*(ok-base_ok)/n:+9.2f}")
        tn += n; tok += ok; tb += base_ok
    print(f"{'total':16}{tn:5}{100*tok/tn:9.2f}%{100*tb/tn:9.2f}%{100*(tok-tb)/tn:+9.2f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
