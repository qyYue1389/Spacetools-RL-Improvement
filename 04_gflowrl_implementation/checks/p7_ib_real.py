#!/usr/bin/env python3
"""P7 criterion i-b (real half): replace the synthetic drift in p7_estimators.py with **measured** logprob gaps.

The conclusion of section 4 of p7_estimators.py: which within-batch constant to pick **depends entirely on the shape of the drift distribution** —
Gaussian -> arithmetic mean is best; heavy-tailed -> median/huber better by 0.6x; bimodal -> arithmetic mean better by 3.5-7.4x.
That half was done on synthetic noise, with the note "must be rerun on real values after the forward pass". **This script is that rerun.**

⚠ **What is measured is not training drift.** No training happened in the P4/P6/passk2 data,
π_old = π_ref = the same ckpt, so GFlowRL's log π_ref − log π_old is identically 0.
This script uses **the logprob gap between rollout (sglang) and trainer (FSDP)** —
GFlowRL's IS weight w_i exists precisely for it, and it is present even with zero training;
it is the **first real proxy** for drift. Units and magnitude are comparable; it is **not the same quantity**.

The two synthetic assumptions replaced:
  (a) distribution shape — synthetic used independent Gaussian x |y|; the real shape is whatever the data says;
  (b) |y| — synthetic used a character proxy; here it is n_policy_tokens, the real token count.

Usage:  python3 tools/p7/p7_ib_real.py <dump dir or jsonl> [...]
"""
import argparse
import json
import os

import numpy as np

EPS_LOW, EPS_HIGH = 0.2, 0.28          # paper Table 9
RNG = np.random.default_rng(20260902)


def c_mean(T):
    return T.mean(axis=-1)


def c_median(T):
    return np.median(T, axis=-1)


def c_huber(T, delta=EPS_HIGH):
    c = np.median(T, axis=-1)
    for _ in range(60):
        r = T - c[..., None]
        w = np.where(np.abs(r) <= delta, 1.0, delta / np.maximum(np.abs(r), 1e-12))
        c = (w * T).sum(-1) / np.maximum(w.sum(-1), 1e-12)
    return c


EST = {"mean(GFlowRL Eq.4)": c_mean, "median(DevGrad)": c_median, "huber(DevGrad)": c_huber}


def load(path, G=5):
    """-> R [n,G] rewards, L [n,G] real token count |y|, D [n,G] measured logprob gap (summed over the whole sequence)."""
    if os.path.isdir(path):
        path = os.path.join(path, "0.jsonl")
    g = {}
    for line in open(path):
        d = json.loads(line)
        g.setdefault(d["index"], []).append(d)
    R, L, D = [], [], []
    dropped = 0
    for v in g.values():
        if len(v) != G or any(("rollout_log_probs" not in x or "trainer_log_probs" not in x) for x in v):
            dropped += 1
            continue
        R.append([x["score"] for x in v])
        L.append([max(x["n_policy_tokens"], 1) for x in v])
        D.append([sum(a - b for a, b in zip(x["rollout_log_probs"], x["trainer_log_probs"])) for x in v])
    if dropped:
        print(f"  (dropped {dropped} incomplete group(s))")
    return np.array(R, float), np.array(L, float), np.array(D, float)


def shape_report(D, L):
    for name, x in (("per-sequence summed", D.ravel()), ("per-token mean", (D / L).ravel())):
        m, sd = x.mean(), x.std()
        sk = ((x - m) ** 3).mean() / sd**3 if sd else 0.0
        ku = ((x - m) ** 4).mean() / sd**4 - 3 if sd else 0.0
        n = len(x)
        bc = (sk**2 + 1) / ((ku + 3) + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        flag = "  <- bimodal flag" if bc > 5 / 9 else ""
        print(f"  {name:22s} mean {m:+.5f}  sd {sd:.5f}  skew {sk:+.2f}"
              f"  exkurt {ku:+.1f}  SarleBC {bc:.4f}{flag}")


def estimator_variance(D, G=5, trials=20000):
    """Where criterion i-b lands: resample G samples from the **measured** drift distribution and compare the variance of the three constants."""
    pool = D.ravel()
    idx = RNG.integers(0, len(pool), size=(trials, G))
    T = pool[idx]
    print(f"  resample {trials} times x G={G}, sampling with replacement from {len(pool)} measured drift values")
    base = None
    for name, fn in EST.items():
        v = float(np.var(fn(T)))
        base = v if base is None else base
        print(f"    {name:22s} Var {v:10.5f}   relative to mean {v / base:5.2f}x")


def saturation_and_collapse(R, L, D, beta):
    T = beta * R + D                       # the averaged term of Eq.4 (not normalized by |y|)
    keep = {}
    print(f"  beta={beta}")
    for name, fn in EST.items():
        C = fn(T)[:, None]
        g = C - D / L - beta * R           # Eq.6: the D side is normalized by |y|
        gt = np.clip(g, -EPS_LOW, EPS_HIGH)
        sat = float(((g < -EPS_LOW) | (g > EPS_HIGH)).mean())
        same = float((gt.max(1) - gt.min(1) < 1e-12).mean())
        keep[name] = gt
        print(f"    {name:22s} clip saturated {100 * sat:5.1f}%   whole group g identical {100 * same:5.1f}%")
    dis = float((np.sign(keep["mean(GFlowRL Eq.4)"]) != np.sign(keep["huber(DevGrad)"])).mean())
    print(f"    rollouts where mean and huber give opposite update directions: {100 * dis:.1f}%")


def normalised_eq4(R, L, D, beta=8.0):
    for norm in (False, True):
        T = beta * R + (D / L if norm else D)
        C = c_mean(T)[:, None]
        g = C - D / L - beta * R
        gt = np.clip(g, -EPS_LOW, EPS_HIGH)
        sat = float(((g < -EPS_LOW) | (g > EPS_HIGH)).mean())
        same = float((gt.max(1) - gt.min(1) < 1e-12).mean())
        tag = "after length normalization" if norm else "Eq.4 as is   "
        print(f"    {tag}  clip saturated {100 * sat:5.1f}%   whole group g identical {100 * same:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--beta", type=float, nargs="*", default=[1.0, 8.0])
    args = ap.parse_args()

    for p in args.dumps:
        key = os.path.basename(os.path.dirname(p.rstrip("/"))) or p
        print("=" * 74)
        print(key)
        print("=" * 74)
        R, L, D = load(p)
        if len(R) == 0:
            print("  no complete groups with logprobs\n")
            continue
        print(f"  groups {len(R)}   G={R.shape[1]}   |y| median {np.median(L):.0f} tokens\n")
        print("[1] distribution shape of the measured drift proxy (input to criterion i-b)")
        shape_report(D, L)
        print("\n[2] where criterion i-b lands: variance of the three constants on the **measured** distribution")
        estimator_variance(D)
        print("\n[3] clip saturation and whole-group collapse (real drift)")
        for b in args.beta:
            saturation_and_collapse(R, L, D, b)
        print("\n[4] candidate fix: apply length normalization to Eq.4 as well (beta=8, real drift)")
        normalised_eq4(R, L, D)
        print()


if __name__ == "__main__":
    main()
