#!/usr/bin/env python3
"""C′ gate: compare the three length-normalization configs once, on **real** data.

The preliminary gate of `records/P7_ROUTE_C_PLAN`. **Zero GPU.**

A′ has already shown: the paper as-is (A) and "normalized Eq.4" (B) each have a pathology,
while "normalize neither side" (C′) is healthy on both metrics — but the cost of C′ is exactly the problem the paper originally
introduced length normalization to solve: "long sequences dominating the loss".
**This script measures that cost, on our own length distribution.**

The flow gap of the three configs (Eq. 6, d_i := log π_ref − log π_old):

    A  paper as-is     Z_t = mean_j(β·r_j + d_j)          g_i = Z_t − d_i/L_i − β·r_i
    B  normalized Eq.4 Z_t = mean_j(β·r_j + d_j/L_j)      g_i = Z_t − d_i/L_i − β·r_i
    C′ neither side    Z_t = mean_j(β·r_j + d_j)          g_i = Z_t − d_i     − β·r_i

The second term of Eq. 8 (the one carrying the gradient) is correspondingly (1/L_i)·log(π_θ/π_old) (A, B)
or log(π_θ/π_old) (C′). **Length domination happens in this term, not in the flow gap** —
the flow gap is clamped by the clip, the update term is not.

⚠ **`d_i` is a proxy, not the true value.** No training happened in P4/P6/passk2 (π_old = π_ref = the same ckpt),
   so the true drift is always 0. Here the **logprob difference between rollout (sglang) and trainer (FSDP)** is used instead:
   comparable in units and magnitude, it is the quantity the IS weight `w_i` exists for, but it is **not training drift**.
   Convention: d_i := Σ_masked (trainer_lp − rollout_lp).

Usage: python3 tools/p7/p7_cprime_gate.py [--beta 8]
"""
import argparse, json, os, statistics as st
from collections import defaultdict

EPS_LOW, EPS_HIGH = 0.2, 0.28                 # paper Table 9
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BENCH = ["robospatial", "blinkdepth", "boppose"]


def load(key, root):
    g = defaultdict(list)
    with open(os.path.join(root, key, "0.jsonl")) as f:
        for line in f:
            o = json.loads(line)
            if "rollout_log_probs" not in o:
                continue
            rl, tr = o["rollout_log_probs"], o["trainer_log_probs"]
            n = min(len(rl), len(tr))
            g[o["index"]].append({
                "r": o["score"],
                "L": max(int(o["n_policy_tokens"]), 1),
                "d": sum(tr[i] - rl[i] for i in range(n)),      # unnormalized, summed over the whole sequence
            })
    return [v for v in g.values() if len(v) >= 2]


def gaps(group, beta, cfg):
    r = [x["r"] for x in group]; L = [x["L"] for x in group]; d = [x["d"] for x in group]
    G = len(group)
    if cfg == "B":
        Z = sum(beta*r[i] + d[i]/L[i] for i in range(G)) / G
    else:
        Z = sum(beta*r[i] + d[i] for i in range(G)) / G
    if cfg == "Cp":
        return [Z - d[i] - beta*r[i] for i in range(G)]
    return [Z - d[i]/L[i] - beta*r[i] for i in range(G)]


def share_top_decile(weights, lengths):
    """Sort by |y|; the share of total weight taken by the longest 10% of rollouts."""
    pairs = sorted(zip(lengths, weights))
    k = max(1, len(pairs) // 10)
    tot = sum(w for _, w in pairs)
    top = sum(w for _, w in pairs[-k:])
    return top / tot if tot else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=8.0)
    ap.add_argument("--root", default=os.path.join(REPO, "01_official_checkpoint_eval", "p6", "passk2"))
    a = ap.parse_args()
    B = a.beta
    print(f"# C′ gate · beta={B:g} · clip [-{EPS_LOW}, +{EPS_HIGH}] · real |y| and logprob difference between the two sides\n")

    for key in BENCH:
        groups = load(key, a.root)
        allL = [x["L"] for g in groups for x in g]
        alld = [x["d"] for g in groups for x in g]
        n = sum(len(g) for g in groups)
        print("=" * 76)
        print(f"## {key}   prompts={len(groups)}  rollouts={n}")
        print(f"   |y| token   median {st.median(allL):.0f}   mean {st.mean(allL):.0f}"
              f"   min {min(allL)}  max {max(allL)}   max/min {max(allL)/min(allL):.1f}×")
        print(f"   d whole-seq sum  median {st.median([abs(x) for x in alld]):.4f}"
              f"   per-token median {st.median([abs(x['d'])/x['L'] for g in groups for x in g]):.2e}\n")

        # ---- section A: flow gap health ----
        print(f"   A. flow gap health (starting point, π_θ=π_old, update term is 0)")
        print(f"      {'config':>16} {'saturation':>9} {'whole group same':>10} {'within-group |g| range median':>20}")
        for cfg, name in (("A", "A paper as-is"), ("B", "B normalized Eq.4"), ("Cp", "C′ normalize neither")):
            sat = tot = uni = 0; rng = []
            for g in groups:
                gg = gaps(g, B, cfg)
                gt = [min(max(x, -EPS_LOW), EPS_HIGH) for x in gg]
                sat += sum(1 for x in gg if x < -EPS_LOW or x > EPS_HIGH); tot += len(gg)
                uni += (max(gt) - min(gt) < 1e-12)
                rng.append(max(gg) - min(gg))
            print(f"      {name:>16} {100*sat/tot:8.1f}% {100*uni/len(groups):9.1f}%"
                  f" {st.median(rng):20.4f}")

        # ---- section A2: within-group variance decomposition — is the reward signal still there ----
        print(f"\n   A2. variance decomposition of within-group g: reward term vs drift term (reported separately for degenerate/non-degenerate)")
        print(f"       g_i = [Z − β·r_i] + [drift term]; asks whether drift drowns out the reward")
        print(f"       {'':>14} {'groups':>6} {'Var(reward term)':>13} {'Var(drift term) A':>15}"
              f" {'Var(drift term) C′':>16} {'drift/reward C′':>14}")
        for label, want_deg in (("reward-degenerate groups", True), ("non-degenerate groups", False)):
            vr, vdA, vdC, ratio, cnt = [], [], [], [], 0
            for g in groups:
                r = [x["r"] for x in g]
                deg = (max(r) - min(r)) < 1e-9
                if deg != want_deg:
                    continue
                cnt += 1
                L = [x["L"] for x in g]; d = [x["d"] for x in g]
                rew = [-B*x for x in r]                       # Z is a constant, does not change variance
                dA  = [-d[i]/L[i] for i in range(len(g))]
                dC  = [-d[i]       for i in range(len(g))]
                vr.append(st.pvariance(rew)); vdA.append(st.pvariance(dA)); vdC.append(st.pvariance(dC))
                if st.pvariance(rew) > 1e-12:
                    ratio.append(st.pvariance(dC) / st.pvariance(rew))
            if not cnt:
                continue
            rr = f"{st.median(ratio):.3f}" if ratio else "— (reward variance is 0)"
            print(f"       {label:>14} {cnt:6d} {st.median(vr):13.4f} {st.median(vdA):15.2e}"
                  f" {st.median(vdC):16.4f} {rr:>14}")

        # ---- section B: length domination ----
        print(f"\n   B. length domination: Eq.8 update term, how much of the loss the longest 10% of rollouts take")
        print(f"      update term = (1/L)·Σδ_t (A, B)  or  Σδ_t (C′); loss ∝ the square of this term")
        print(f"      both δ assumptions reported: independent across tokens (∝√L) and fully aligned (∝L)")
        eq = share_top_decile([1.0]*len(allL), allL)
        rows = [
            ("normalized · independent",   [1.0/L for L in allL]),
            ("normalized · aligned",   [1.0   for L in allL]),
            ("unnormalized · independent", [float(L) for L in allL]),
            ("unnormalized · aligned", [float(L)**2 for L in allL]),
        ]
        print(f"      {'case':>16} {'loss share of longest 10%':>20} {'relative to even split (10%)':>16}")
        for name, w in rows:
            s = share_top_decile(w, allL)
            print(f"      {name:>16} {100*s:19.1f}% {s/0.1:15.2f}×")

        # within group (only relative weights inside the same prompt affect that prompt's gradient direction)
        ratios_ind, ratios_coh = [], []
        for g in groups:
            Ls = [x["L"] for x in g]
            m = st.mean(Ls)
            ratios_ind.append(max(Ls) / m)
            ratios_coh.append((max(Ls) / m) ** 2)
        print(f"      relative weight of the longest rollout in the group (vs group mean):"
              f" independent {st.median(ratios_ind):.2f}×   aligned {st.median(ratios_coh):.2f}×")

        # ---- section C: how sensitive this cost is to length spread ----
        med = st.median(allL)
        print(f"\n   C. sensitivity of the cost to length spread (unnormalized · aligned, worst case)")
        print(f"      reference: **log-uniform** distribution (not our shape, only to see sensitivity)")
        print(f"      {'length spread max/min':>18} {'loss share of longest 10%':>20}")
        import random
        rng = random.Random(20260902)
        for spread in (1.0, 3.0, 10.0, 20.0, 100.0):
            import math
            if spread == 1.0:
                Ls = [med]*len(allL)
            else:                      # log-uniform, keep the median unchanged
                Ls = [med*math.exp(rng.uniform(-0.5, 0.5)*math.log(spread)) for _ in allL]
            sh = share_top_decile([x**2 for x in Ls], Ls)
            print(f"      {spread:18.0f}× {100*sh:19.1f}%")
        print(f"      **real distribution of this benchmark** (max/min = {max(allL)/min(allL):.1f}×, "
              f"but much more concentrated than log-uniform) -> **{100*share_top_decile([float(x)**2 for x in allL], allL):.1f}%**")
        print()


if __name__ == "__main__":
    main()
