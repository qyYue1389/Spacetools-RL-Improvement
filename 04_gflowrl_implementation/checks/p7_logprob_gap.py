#!/usr/bin/env python3
"""P7 · rollout(sglang) 与 trainer(FSDP) 之间 logprob 差的分布形状。

**这是第二组采样这一趟最重要的产出。** 判据 i-b 的落点就是它:
`p7_estimators.py` §4 测出三种噪声分布给三种相反的排序 ——
高斯 -> 算术平均最优;重尾 -> median/huber 好 0.6x;**双峰 -> 算术平均好 3.5-7.4x**。
该选哪个批内常数完全取决于真实漂移的分布形状,而那只有前向 pass 能给。

⚠ 一个必须写进结论的限定:P4/P6 的数据里**没有训练发生过**,
`π_old = π_ref = 同一个 ckpt`,所以 GFlowRL 的漂移项 `log π_ref − log π_old`
在这些数据上**恒等于 0**。本脚本量的是 **rollout/trainer 的框架差**,
不是训练漂移 —— 它是漂移的**第一个真实代理**(GFlowRL 的 IS 权重正是为它设的)。

同时兑现 P7_STEP23_RESULTS.md §1.2 的待办:把那里的**字符**代理换成**真实 token 数**。

用法:  python3 tools/p7/p7_logprob_gap.py <dump 目录或 jsonl> [...]
"""
import argparse
import json
import math
import os
import statistics as st
from collections import defaultdict


def iter_dump(path):
    if os.path.isdir(path):
        path = os.path.join(path, "0.jsonl")
    with open(path) as fh:
        for line in fh:
            yield json.loads(line)


def rle_sum(rle):
    """Reconstruct sum(mask) from the run-length encoding."""
    return sum(run for val, run in rle if val == 1)


def quantiles(xs, qs=(0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)):
    s = sorted(xs)
    n = len(s)
    return {q: s[min(n - 1, max(0, int(round(q * (n - 1)))))] for q in qs}


def moments(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / n
    sd = math.sqrt(var) if var > 0 else 0.0
    if sd == 0:
        return m, sd, 0.0, 0.0
    skew = sum((x - m) ** 3 for x in xs) / n / sd**3
    kurt = sum((x - m) ** 4 for x in xs) / n / sd**4 - 3.0  # excess
    return m, sd, skew, kurt


def bimodality(xs):
    """Sarle's bimodality coefficient: (skew^2 + 1) / kurt_excess+3, corrected for n.

    > 5/9 = 0.5556 is the usual flag for a bimodal (or heavily non-normal) shape.
    Reported next to skew/kurtosis, not instead of them -- one number cannot
    distinguish 'two peaks' from 'one flat peak'.
    """
    n = len(xs)
    _, _, skew, kurt = moments(xs)
    if n < 4:
        return float("nan")
    num = skew**2 + 1
    den = (kurt + 3) + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return num / den if den else float("nan")


def histogram(xs, bins=25):
    lo, hi = min(xs), max(xs)
    if hi <= lo:
        return [(lo, hi, len(xs))]
    w = (hi - lo) / bins
    counts = [0] * bins
    for x in xs:
        k = min(bins - 1, int((x - lo) / w))
        counts[k] += 1
    return [(lo + i * w, lo + (i + 1) * w, c) for i, c in enumerate(counts)]


def render_hist(rows, width=48):
    top = max(c for _, _, c in rows) or 1
    out = []
    for a, b, c in rows:
        bar = "#" * int(round(width * c / top))
        out.append(f"  [{a:+9.5f}, {b:+9.5f})  {c:6d}  {bar}")
    return "\n".join(out)


def analyse(path, label):
    per_token_gap = []     # every token, one entry
    per_seq_gap_sum = []   # per rollout, summed over its policy tokens
    per_seq_gap_mean = []  # per rollout, per-token mean
    per_token_lp_trainer = []
    n_policy, n_resp = [], []
    missing = defaultdict(int)
    n = 0

    for d in iter_dump(path):
        n += 1
        rlp = d.get("rollout_log_probs")
        tlp = d.get("trainer_log_probs")
        npt = d.get("n_policy_tokens")
        nrt = d.get("n_response_tokens")
        rle = d.get("response_mask_rle")

        if npt is None:
            missing["n_policy_tokens"] += 1
        else:
            n_policy.append(npt)
        if nrt is None:
            missing["n_response_tokens"] += 1
        else:
            n_resp.append(nrt)
        if rle is not None and npt is not None and rle_sum(rle) != npt:
            missing["rle_mismatch"] += 1

        if not rlp or not tlp:
            missing["logprobs"] += 1
            continue
        if len(rlp) != len(tlp):
            missing["length_mismatch"] += 1
            continue
        if npt is not None and len(rlp) != npt:
            missing["logprob_len_vs_npt"] += 1

        gaps = [a - b for a, b in zip(rlp, tlp)]
        per_token_gap.extend(gaps)
        per_seq_gap_sum.append(sum(gaps))
        per_seq_gap_mean.append(sum(gaps) / len(gaps))
        per_token_lp_trainer.extend(tlp)

    print("=" * 76)
    print(f"{label}   ({n} rollouts)")
    print("=" * 76)
    if missing:
        print("  incomplete records:", dict(missing))

    if n_policy:
        print(f"\n|y| = n_policy_tokens (real tokens, replaces the character proxy)")
        print(f"  mean {st.mean(n_policy):8.1f}   median {st.median(n_policy):8.1f}"
              f"   min {min(n_policy)}   max {max(n_policy)}")
    if n_resp and n_policy and len(n_resp) == len(n_policy):
        mult = [r / p for r, p in zip(n_resp, n_policy) if p]
        print(f"  raw response / policy tokens:  median {st.median(mult):.2f}x"
              f"   mean {st.mean(mult):.2f}x")
        print(f"  (P7_STEP23_RESULTS.md §1.2 measured this in CHARACTERS as"
              f" 1.14x / 1.41x / 2.40x)")

    if per_token_lp_trainer:
        m, sd, sk, ku = moments(per_token_lp_trainer)
        print(f"\nper-token log pi (trainer arm), magnitude check for T6")
        print(f"  mean {m:+.4f}   sd {sd:.4f}   n={len(per_token_lp_trainer)}")
        print(f"  => a |y|~{st.median(n_policy):.0f} sequence carries"
              f" log pi ~ {m * st.median(n_policy):+.1f} in total,"
              f" vs beta*r in [0, 8]")

    if not per_token_gap:
        print("\n  NO logprob pairs -- nothing to say about the distribution shape.")
        return

    for name, xs in (
        ("per-TOKEN gap  (log pi_rollout - log pi_trainer)", per_token_gap),
        ("per-SEQUENCE summed gap", per_seq_gap_sum),
        ("per-SEQUENCE per-token mean gap", per_seq_gap_mean),
    ):
        m, sd, sk, ku = moments(xs)
        bc = bimodality(xs)
        q = quantiles(xs)
        print(f"\n{name}   n={len(xs)}")
        print(f"  mean {m:+.6f}   sd {sd:.6f}   median {st.median(xs):+.6f}")
        print(f"  skew {sk:+.3f}   excess kurtosis {ku:+.3f}   Sarle BC {bc:.4f}"
              f"   {'-> flags NON-GAUSSIAN/bimodal (>0.5556)' if bc > 5 / 9 else ''}")
        print("  quantiles: " + "  ".join(f"p{int(k*100)}={v:+.5f}" for k, v in q.items()))

    print(f"\nper-TOKEN gap histogram")
    print(render_hist(histogram(per_token_gap)))

    # The threshold that matters: P7_CRITERION_IB.md §3 puts the point where the
    # Eq.4 / Eq.5-6 normalisation mismatch starts wiping out the reward signal at
    # roughly 5e-4 .. 7.5e-4 nat per token.
    absmean = sum(abs(x) for x in per_seq_gap_mean) / len(per_seq_gap_mean)
    print(f"\nvs the 5e-4 nat/token threshold from P7_CRITERION_IB.md §3")
    print(f"  mean |per-token gap| per rollout: {absmean:.3e} nat/token")
    print(f"  => {'ABOVE' if absmean > 5e-4 else 'below'} the ~5e-4 threshold")
    print("  (that threshold was derived for log pi_ref - log pi_old, i.e. TRAINING")
    print("   drift.  This gap is the framework mismatch, a different quantity --")
    print("   comparable in units and magnitude, not the same thing.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    args = ap.parse_args()
    for p in args.dumps:
        analyse(p, os.path.basename(os.path.dirname(p.rstrip("/"))) or p)


if __name__ == "__main__":
    main()
