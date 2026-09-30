#!/usr/bin/env python3
"""C′ 闸门:长度归一化的三个配置,在**真实**数据上比一次。

`records/P7_ROUTE_C_PLAN` 的前置闸门。**零 GPU。**

A′ 已经证明:论文原样(A)与「归一化 Eq.4」(B)各有一种病,
而「两边都不归一化」(C′)在两个指标上都健康 —— 但 C′ 的代价正是论文当初
引入长度归一化要解决的问题:「long sequences dominating the loss」。
**本脚本量那个代价,在我们自己的长度分布上。**

三个配置的 flow gap(Eq. 6,d_i := log π_ref − log π_old):

    A  论文原样    Z_t = mean_j(β·r_j + d_j)          g_i = Z_t − d_i/L_i − β·r_i
    B  归一化 Eq.4 Z_t = mean_j(β·r_j + d_j/L_j)      g_i = Z_t − d_i/L_i − β·r_i
    C′ 两边都不    Z_t = mean_j(β·r_j + d_j)          g_i = Z_t − d_i     − β·r_i

Eq. 8 的第二项(带梯度的那个)相应地是 (1/L_i)·log(π_θ/π_old)(A、B)
或 log(π_θ/π_old)(C′)。**长度主导发生在这一项,不在 flow gap** ——
flow gap 被 clip 钳住了,update 项没有。

⚠ **`d_i` 是代理不是真值。** P4/P6/passk2 里没有训练发生过(π_old = π_ref = 同一 ckpt),
   真实漂移恒为 0。这里用 **rollout(sglang)与 trainer(FSDP)的 logprob 差**代替:
   单位与量级可比,是 IS 权重 `w_i` 设立的那个量,但**不是训练漂移**。
   约定 d_i := Σ_masked (trainer_lp − rollout_lp)。

用法: python3 tools/p7/p7_cprime_gate.py [--beta 8]
"""
import argparse, json, os, statistics as st
from collections import defaultdict

EPS_LOW, EPS_HIGH = 0.2, 0.28                 # 论文 Table 9
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
                "d": sum(tr[i] - rl[i] for i in range(n)),      # 未归一化,整条求和
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
    """按 |y| 排序,最长 10% 的 rollout 占总权重的比例。"""
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
    print(f"# C′ 闸门 · beta={B:g} · clip [-{EPS_LOW}, +{EPS_HIGH}] · 真实 |y| 与两侧 logprob 差\n")

    for key in BENCH:
        groups = load(key, a.root)
        allL = [x["L"] for g in groups for x in g]
        alld = [x["d"] for g in groups for x in g]
        n = sum(len(g) for g in groups)
        print("=" * 76)
        print(f"## {key}   prompts={len(groups)}  rollouts={n}")
        print(f"   |y| token   中位 {st.median(allL):.0f}   均值 {st.mean(allL):.0f}"
              f"   min {min(allL)}  max {max(allL)}   max/min {max(allL)/min(allL):.1f}×")
        print(f"   d 整条求和  中位 {st.median([abs(x) for x in alld]):.4f}"
              f"   每 token 中位 {st.median([abs(x['d'])/x['L'] for g in groups for x in g]):.2e}\n")

        # ---- A节:flow gap 健康度 ----
        print(f"   A. flow gap 健康度(起点,π_θ=π_old,update 项为 0)")
        print(f"      {'配置':>16} {'饱和率':>9} {'整组同值':>10} {'组内 |g| 极差 中位':>20}")
        for cfg, name in (("A", "A 论文原样"), ("B", "B 归一化Eq.4"), ("Cp", "C′ 都不归一化")):
            sat = tot = uni = 0; rng = []
            for g in groups:
                gg = gaps(g, B, cfg)
                gt = [min(max(x, -EPS_LOW), EPS_HIGH) for x in gg]
                sat += sum(1 for x in gg if x < -EPS_LOW or x > EPS_HIGH); tot += len(gg)
                uni += (max(gt) - min(gt) < 1e-12)
                rng.append(max(gg) - min(gg))
            print(f"      {name:>16} {100*sat/tot:8.1f}% {100*uni/len(groups):9.1f}%"
                  f" {st.median(rng):20.4f}")

        # ---- A2节:组内方差分解 —— 奖励信号还在不在 ----
        print(f"\n   A2. 组内 g 的方差分解:奖励项 vs 漂移项(分简并/非简并报)")
        print(f"       g_i = [Z − β·r_i] + [漂移项];问漂移会不会把奖励淹掉")
        print(f"       {'':>14} {'组数':>6} {'Var(奖励项)':>13} {'Var(漂移项) A':>15}"
              f" {'Var(漂移项) C′':>16} {'漂移/奖励 C′':>14}")
        for label, want_deg in (("奖励简并组", True), ("非简并组", False)):
            vr, vdA, vdC, ratio, cnt = [], [], [], [], 0
            for g in groups:
                r = [x["r"] for x in g]
                deg = (max(r) - min(r)) < 1e-9
                if deg != want_deg:
                    continue
                cnt += 1
                L = [x["L"] for x in g]; d = [x["d"] for x in g]
                rew = [-B*x for x in r]                       # Z 是常数,不改方差
                dA  = [-d[i]/L[i] for i in range(len(g))]
                dC  = [-d[i]       for i in range(len(g))]
                vr.append(st.pvariance(rew)); vdA.append(st.pvariance(dA)); vdC.append(st.pvariance(dC))
                if st.pvariance(rew) > 1e-12:
                    ratio.append(st.pvariance(dC) / st.pvariance(rew))
            if not cnt:
                continue
            rr = f"{st.median(ratio):.3f}" if ratio else "—(奖励方差为 0)"
            print(f"       {label:>14} {cnt:6d} {st.median(vr):13.4f} {st.median(vdA):15.2e}"
                  f" {st.median(vdC):16.4f} {rr:>14}")

        # ---- B节:长度主导 ----
        print(f"\n   B. 长度主导:Eq.8 的 update 项,最长 10% 的 rollout 占多少 loss")
        print(f"      update 项 = (1/L)·Σδ_t (A、B)  或  Σδ_t (C′);loss ∝ 该项的平方")
        print(f"      两种 δ 假设都报:token 间独立(∝√L)与完全同向(∝L)")
        eq = share_top_decile([1.0]*len(allL), allL)
        rows = [
            ("归一化 · 独立",   [1.0/L for L in allL]),
            ("归一化 · 同向",   [1.0   for L in allL]),
            ("未归一化 · 独立", [float(L) for L in allL]),
            ("未归一化 · 同向", [float(L)**2 for L in allL]),
        ]
        print(f"      {'情形':>16} {'最长10%的 loss 份额':>20} {'相对均分(10%)':>16}")
        for name, w in rows:
            s = share_top_decile(w, allL)
            print(f"      {name:>16} {100*s:19.1f}% {s/0.1:15.2f}×")

        # 组内(同一 prompt 内部的相对权重才影响该 prompt 的梯度方向)
        ratios_ind, ratios_coh = [], []
        for g in groups:
            Ls = [x["L"] for x in g]
            m = st.mean(Ls)
            ratios_ind.append(max(Ls) / m)
            ratios_coh.append((max(Ls) / m) ** 2)
        print(f"      组内最长 rollout 的相对权重(对组内均值):"
              f" 独立 {st.median(ratios_ind):.2f}×   同向 {st.median(ratios_coh):.2f}×")

        # ---- C节:这个代价对长度散布有多敏感 ----
        med = st.median(allL)
        print(f"\n   C. 代价对长度散布的敏感性(未归一化·同向,最坏情形)")
        print(f"      参照:**对数均匀**分布(不是我们的形状,只为看敏感度)")
        print(f"      {'长度散布 max/min':>18} {'最长10%的 loss 份额':>20}")
        import random
        rng = random.Random(20260902)
        for spread in (1.0, 3.0, 10.0, 20.0, 100.0):
            import math
            if spread == 1.0:
                Ls = [med]*len(allL)
            else:                      # 对数均匀,保持中位不变
                Ls = [med*math.exp(rng.uniform(-0.5, 0.5)*math.log(spread)) for _ in allL]
            sh = share_top_decile([x**2 for x in Ls], Ls)
            print(f"      {spread:18.0f}× {100*sh:19.1f}%")
        print(f"      **本 benchmark 的真实分布**(max/min = {max(allL)/min(allL):.1f}×,"
              f"但比对数均匀集中得多)-> **{100*share_top_decile([float(x)**2 for x in allL], allL):.1f}%**")
        print()


if __name__ == "__main__":
    main()
