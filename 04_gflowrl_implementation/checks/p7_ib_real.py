#!/usr/bin/env python3
"""P7 判据 i-b(真实一半):用**实测**的 logprob 差替掉 p7_estimators.py 的合成漂移。

p7_estimators.py 第 4 节的结论是:该选哪个批内常数,**完全取决于漂移的分布形状** ——
高斯 -> 算术平均最优;重尾 -> median/huber 好 0.6x;双峰 -> 算术平均好 3.5-7.4x。
那一半是在合成噪声上做的,并写明「前向 pass 之后必须用真值重跑」。**本脚本就是重跑。**

⚠ **量的不是训练漂移。** P4/P6/passk2 的数据里没有训练发生过,
π_old = π_ref = 同一个 ckpt,所以 GFlowRL 的 log π_ref − log π_old 恒等于 0。
本脚本用的是 **rollout(sglang)与 trainer(FSDP)之间的 logprob 差** ——
GFlowRL 的 IS 权重 w_i 正是为它设立的,零训练下它也存在,
是漂移的**第一个真实代理**。单位与量级可比,**不是同一个量**。

替掉的两条合成假设:
  (a) 分布形状 —— 合成用独立高斯 x |y|;真实形状由数据说了算;
  (b) |y| —— 合成用字符代理,这里是 n_policy_tokens,真 token 数。

用法:  python3 tools/p7/p7_ib_real.py <dump 目录或 jsonl> [...]
"""
import argparse
import json
import os

import numpy as np

EPS_LOW, EPS_HIGH = 0.2, 0.28          # 论文 Table 9
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
    """-> R [n,G] 奖励, L [n,G] 真 token 数 |y|, D [n,G] 实测 logprob 差(整条求和)."""
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
    """判据 i-b 的落点:在**实测**漂移分布上重抽 G 个样本,比三个常数的方差。"""
    pool = D.ravel()
    idx = RNG.integers(0, len(pool), size=(trials, G))
    T = pool[idx]
    print(f"  重抽 {trials} 次 x G={G},从 {len(pool)} 条实测漂移里有放回抽样")
    base = None
    for name, fn in EST.items():
        v = float(np.var(fn(T)))
        base = v if base is None else base
        print(f"    {name:22s} Var {v:10.5f}   相对 mean {v / base:5.2f}x")


def saturation_and_collapse(R, L, D, beta):
    T = beta * R + D                       # Eq.4 的被平均项(未按 |y| 归一化)
    keep = {}
    print(f"  beta={beta}")
    for name, fn in EST.items():
        C = fn(T)[:, None]
        g = C - D / L - beta * R           # Eq.6:D 这一侧按 |y| 归一化
        gt = np.clip(g, -EPS_LOW, EPS_HIGH)
        sat = float(((g < -EPS_LOW) | (g > EPS_HIGH)).mean())
        same = float((gt.max(1) - gt.min(1) < 1e-12).mean())
        keep[name] = gt
        print(f"    {name:22s} clip 饱和 {100 * sat:5.1f}%   整组 g 同值 {100 * same:5.1f}%")
    dis = float((np.sign(keep["mean(GFlowRL Eq.4)"]) != np.sign(keep["huber(DevGrad)"])).mean())
    print(f"    mean 与 huber 给出相反更新方向的 rollout: {100 * dis:.1f}%")


def normalised_eq4(R, L, D, beta=8.0):
    for norm in (False, True):
        T = beta * R + (D / L if norm else D)
        C = c_mean(T)[:, None]
        g = C - D / L - beta * R
        gt = np.clip(g, -EPS_LOW, EPS_HIGH)
        sat = float(((g < -EPS_LOW) | (g > EPS_HIGH)).mean())
        same = float((gt.max(1) - gt.min(1) < 1e-12).mean())
        tag = "长度归一化后" if norm else "原样 Eq.4   "
        print(f"    {tag}  clip 饱和 {100 * sat:5.1f}%   整组 g 同值 {100 * same:5.1f}%")


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
        print(f"  groups {len(R)}   G={R.shape[1]}   |y| 中位 {np.median(L):.0f} tokens\n")
        print("[1] 实测漂移代理的分布形状(判据 i-b 的输入)")
        shape_report(D, L)
        print("\n[2] 判据 i-b 的落点:三个常数在**实测**分布上的方差")
        estimator_variance(D)
        print("\n[3] clip 饱和与整组塌缩(真实漂移)")
        for b in args.beta:
            saturation_and_collapse(R, L, D, b)
        print("\n[4] 候选修法:Eq.4 也做长度归一化(beta=8,真实漂移)")
        normalised_eq4(R, L, D)
        print()


if __name__ == "__main__":
    main()
