#!/usr/bin/env python3
"""Build a truncated copy of the eval benchmark tree for smoke tests.

run_eval.sh points data.val_files straight at a parquet and has no sample-count
option, so the only way to run N samples is to hand it a parquet with N rows.

Minimum size: run_eval.sh passes data.train_batch_size=32 and points both
train_files and val_files at the same parquet, so verl builds a train dataloader
even under val_only=true. Fewer than 32 rows yields zero batches and fails with
"Train dataloader is empty!".

Usage:
    python make_subset.py bopgrasp 32 [--out /workspace/eval-subset]

Then run the eval against it:
    DATA_DIR=/workspace/eval-subset bash examples/toolshed/run_eval.sh <ckpt> bopgrasp
"""

import argparse
import os
import shutil
from collections import Counter

import pandas as pd

SRC = "/workspace/eval-benchmarks"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", help="benchmark key, e.g. bopgrasp")
    ap.add_argument("n", type=int, help="number of rows to keep")
    ap.add_argument("--out", default="/workspace/eval-subset")
    ap.add_argument("--src", default=SRC)
    args = ap.parse_args()

    src = os.path.join(args.src, "data", f"{args.benchmark}.parquet")
    if not os.path.isfile(src):
        raise SystemExit(f"not found: {src}")

    dst_dir = os.path.join(args.out, "data")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f"{args.benchmark}.parquet")

    TRAIN_BATCH_SIZE = 32  # must match data.train_batch_size in run_eval.sh
    if args.n < TRAIN_BATCH_SIZE:
        raise SystemExit(
            f"n={args.n} is below run_eval.sh's data.train_batch_size="
            f"{TRAIN_BATCH_SIZE}; verl would build an empty train dataloader and "
            f'fail with "Train dataloader is empty!". Use {TRAIN_BATCH_SIZE} or more.'
        )

    df = pd.read_parquet(src)
    # Take the head rather than a random sample: a fixed prefix keeps repeated
    # smoke runs comparable to each other.
    sub = df.head(args.n)
    sub.to_parquet(dst, index=False)

    # head(N) keeps repeated smoke runs comparable, but it is NOT a random
    # sample: robospatial's first 32 rows turned out to be 100% "vacant" (point)
    # questions with no VQA (yes/no) among them, so comparing the subset score
    # against the paper's mixed Overall number is meaningless. Report whatever
    # structure is visible so the caller does not misread the result.
    if "reward_model" in df.columns:
        kinds = Counter()
        for v in sub["reward_model"]:
            g = str(v.get("ground_truth", "")).strip()
            if g.lower() in ("yes", "no"):
                kinds["yes/no"] += 1
            elif g.startswith(("[(", "(")):
                kinds["point-list"] += 1
            else:
                kinds["other"] += 1
        if len(kinds) == 1:
            only = next(iter(kinds))
            print(f"  WARNING: all {len(sub)} rows share one answer type ({only}).")
            print("  This subset is not representative; do not compare it against a")
            print("  benchmark-wide number that mixes answer types.")
        else:
            print(f"  answer types in subset: {dict(kinds)}")

    print(f"{args.benchmark}: {len(df)} -> {len(sub)} rows")
    print(f"  wrote {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")
    print(f"  run with: DATA_DIR={args.out} bash examples/toolshed/run_eval.sh <ckpt> {args.benchmark}")


if __name__ == "__main__":
    main()
