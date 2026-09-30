#!/usr/bin/env python3
"""P6 实验 A:pointing 工具对比 —— 不需要 policy,不需要 sglang。

为什么可以绕开 policy:P6 的离线分析证明,在三个 RefSpatial benchmark 上
模型是 100% 透传(276/276,答案逐位等于 roborefer 的返回),
robospatial Vacant 上是 82.8% 透传。所以「换一个 pointing 工具能涨多少分」
可以直接测量:同样的 obj_name 查询 → 新工具 → 用 verl 自己的打分函数打分。

用法(在 spacetools-rl 环境里,Toolshed 必须已经起来):

    conda run -n spacetools-rl python tools/p6/gpu_pointing_swap.py \
        --tool vlm \
        --probes p6/probes/pointing_probes.jsonl \
        --data-dir /workspace/eval-benchmarks \
        --out p6/swap/pointing_vlm.jsonl

--tool 取 roborefer / vlm。先跑一次 --tool roborefer 做**回归校验**:
它应当重现基线正确率(见 probes 里的 baseline_correct),对不上就说明
这条离线通路本身有问题,后面的对比都不能信。
"""
import argparse, json, os, re, sys, io, base64, time
from collections import defaultdict

PT = re.compile(r"\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)")


class _Trainer:
    """复刻 eval 时 default_compute_score 看到的 training_args.trainer。

    RoboSpatial/* 与 BLINK/Spatial_Relation 走的打分分支里,
    format_score_val 只在 `if training_args and hasattr(training_args,'trainer')`
    块内赋值(同块的其它六个默认值都在 if 之前初始化)。离线调用不传
    training_args 时,该分支的 `format_score=format_score_val` 直接
    UnboundLocalError —— 122 个 robospatial 探测全部打分失败。

    两个值都由证据钉死,不是这里的选择:
      format_score                 verl/trainer/config/ppo_trainer.yaml:230 = 0.0
      allow_last_response_fallback ppo_trainer.yaml:233 默认 false,
                                   但 examples/toolshed/run_eval.sh:297 覆盖为 true
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
    """parquet 的 images 列形状在不同 benchmark 上不完全一致,这里逐种试。"""
    from PIL import Image
    import numpy as np
    # pandas 把 parquet 的 list<struct> 列读成 dtype=object 的 ndarray,
    # 不是 list —— 原来的 isinstance(cell, (list, tuple)) 认不出来,于是
    # 整条链在第一跳就掉到末尾的 TypeError,工具一次都没被调用。
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
    raise TypeError(f"无法识别的 image 单元格:{type(cell)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True, choices=["roborefer", "vlm"])
    ap.add_argument("--probes", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条,用于冒烟")
    args = ap.parse_args()

    import pandas as pd
    import ray
    from toolshed import get_toolkit
    from verl.utils.reward_score import default_compute_score

    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)
    toolkit = get_toolkit()          # 连到 run_eval.sh 已经起好的 router
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
                print(f"[shape] images 单元格类型 = {type(row['images'])}", flush=True)
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
            except Exception as e:                       # 工具失败也是一种结果
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
                    print(f"  [score 失败] {bench}#{p['sample_id']}: {e}", flush=True)
            n += 1
            ok += score >= 0.5
            base_ok += p["baseline_correct"]
            fout.write(json.dumps({**p, "tool": args.tool, "new_point": pt,
                                   "new_score": score, "text": (res.text or "")[:200]},
                                  ensure_ascii=False) + "\n")
            if n % 25 == 0:
                print(f"  {bench} {n}/{len(ps)}  {time.time()-t0:.0f}s", flush=True)
        summary[bench] = (n, ok, base_ok, failed)
        print(f"[{bench}] n={n}  新工具 {ok} ({100*ok/n:.2f}%)  "
              f"基线 {base_ok} ({100*base_ok/n:.2f}%)  工具失败 {failed}  "
              f"{time.time()-t0:.0f}s", flush=True)
    fout.close()

    print("\n=== 汇总 ===")
    print(f"{'benchmark':16}{'n':>5}{'新工具':>10}{'基线':>10}{'差值':>10}")
    tn = tok = tb = 0
    for b, (n, ok, base_ok, failed) in summary.items():
        print(f"{b:16}{n:5}{100*ok/n:9.2f}%{100*base_ok/n:9.2f}%{100*(ok-base_ok)/n:+9.2f}")
        tn += n; tok += ok; tb += base_ok
    print(f"{'合计':16}{tn:5}{100*tok/tn:9.2f}%{100*tb/tn:9.2f}%{100*(tok-tb)/tn:+9.2f}")
    print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
