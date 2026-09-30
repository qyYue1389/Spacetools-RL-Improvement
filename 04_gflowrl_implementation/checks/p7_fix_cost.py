#!/usr/bin/env python3
"""P7 · 候选修法的不动点代价(P7_CRITERION_IB.md 第 6 节的待办)。

第 6 节提出:把 Eq.4 的 logprob 项也按 |y| 归一化,
    C_norm = mean_i[ beta*r_i + (log pi_ref - log pi_old)_i / |y_i| ]
合成上把「整组 g 被削成同值」从 36%-93% 压回 0.0%,真实漂移上也压回 0.0%(见 P7_GPU_RESULTS 第 4.5 节)。
**但那一节只测了症状消失,没测代价。** 本脚本测代价。

沿用 tools/p7/p7_fixedpoint.py 的世界构造与约定(小世界 |Y|=64,可精确归一化)。

先把代数写出来,再用数值验:

  零损失(Eq.5 带长度归一化)要求对每个 y:
      Z + (1/L_y)(log pi_theta,y - log pi_ref,y) - beta*r_y = 0
  =>  log pi_theta,y = log pi_ref,y + L_y*(beta*r_y - Z) - c(Z)
      其中 c(Z) = logsumexp(...) 是归一化常数。
  代回残差:
      residual_i = Z_t - Z - c(Z)/L_i
  L 不同时,要它对所有 i 为零,需要 **c(Z)=0 且 Z_t=Z** 两件事同时成立。

  c(Z)=0 是一个单调方程,**一元一根,generically 可解** -> 记作 Z*。
  剩下的 Z_t = Z* 就是「估计量自洽」那一步:

    Eq.4 原样(未归一化):
      Z_t = E[beta*r] - E[L*beta*r] + Z*E[L]     -> 一般 != Z*   (T3b 的「不动点不存在」)
    Eq.4 归一化后(第 6 节的修法):
      Z_t = E[ beta*r + (log pi_ref - log pi_theta)/L ] = E[ beta*r + Z - beta*r ] = Z   ✓ 恒等

  **所以修法在代数上恢复了不动点的存在性。代价不在存在性,在不动点的位置。**
"""
import numpy as np

RNG = np.random.default_rng(20260902)
BETA = 8.0
EPS_LOW, EPS_HIGH, EPS_IS = 0.2, 0.28, 0.2


def lse(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())


def lse_norm(v):
    return v - lse(v)


def make_world(n=64, lo=200, hi=3200, seed=None):
    rng = RNG if seed is None else np.random.default_rng(seed)
    logpi_ref = lse_norm(rng.normal(0, 1.0, n))
    r = rng.uniform(0.0, 1.0, n)
    lens = rng.integers(lo, hi, n).astype(float)
    return logpi_ref, r, lens


def c_of_Z(Z, logpi_ref, r, lens, beta=BETA):
    """未归一化构造的 log 归一化常数 c(Z);零损失需要 c(Z)=0。"""
    return lse(logpi_ref + lens * (beta * r - Z))


def solve_Zstar(logpi_ref, r, lens, beta=BETA):
    """c(Z)=0 的根。c 关于 Z 单调递减,二分。"""
    lo, hi = -50.0, 50.0
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if c_of_Z(mid, logpi_ref, r, lens, beta) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_fixed_point(Z, logpi_ref, r, lens, beta=BETA):
    return lse_norm(logpi_ref + lens * (beta * r - Z))


def Zt_paper(logpi, logpi_ref, r, beta=BETA):
    """Eq.4 原样:无长度归一化。"""
    return float(np.sum(np.exp(logpi) * (beta * r + logpi_ref - logpi)))


def Zt_fixed(logpi, logpi_ref, r, lens, beta=BETA):
    """第 6 节修法:Eq.4 的 logprob 项也按 |y| 归一化。"""
    return float(np.sum(np.exp(logpi) * (beta * r + (logpi_ref - logpi) / lens)))


def residual(logpi, logpi_ref, r, Zt, lens, beta=BETA):
    return Zt + (logpi - logpi_ref) / lens - beta * r


def loss_at(logpi, logpi_ref, r, Zt, lens, beta=BETA, clip=True):
    g = residual(logpi, logpi_ref, r, Zt, lens, beta)
    gt = np.clip(g, -EPS_LOW, EPS_HIGH) if clip else g
    return float(np.sum(np.exp(logpi) * gt ** 2)), g


def shape(logpi):
    p = np.exp(logpi)
    H = float(-(p * np.log(np.maximum(p, 1e-300))).sum())
    return H, float(np.exp(H)), float(p.max())


def report_world(tag, logpi_ref, r, lens):
    print("=" * 78)
    print(f"{tag}   |Y|={len(r)}  beta={BETA:g}  |y| 范围 [{lens.min():.0f}, {lens.max():.0f}] "
          f"中位 {np.median(lens):.0f}")
    print("=" * 78)

    Zs = solve_Zstar(logpi_ref, r, lens)
    lp = build_fixed_point(Zs, logpi_ref, r, lens)
    print(f"  解 c(Z)=0 得 Z* = {Zs:.6f}   残余 c(Z*) = {c_of_Z(Zs, logpi_ref, r, lens):.2e}")

    zt_fix = Zt_fixed(lp, logpi_ref, r, lens)
    zt_pap = Zt_paper(lp, logpi_ref, r)
    print(f"\n  [存在性] 在该点上两种 Eq.4 各自给出的 Z_t,与 Z* 比:")
    print(f"    第 6 节修法 (归一化)   Z_t = {zt_fix:14.6f}   |Z_t - Z*| = {abs(zt_fix - Zs):.3e}")
    print(f"    论文原样   (未归一化)  Z_t = {zt_pap:14.6f}   |Z_t - Z*| = {abs(zt_pap - Zs):.3e}")

    l_fix, _ = loss_at(lp, logpi_ref, r, zt_fix, lens, clip=False)
    l_pap, _ = loss_at(lp, logpi_ref, r, zt_pap, lens, clip=False)
    print(f"\n  [零损失] 用各自自洽的 Z_t 代回 Eq.8(不 clip):")
    print(f"    第 6 节修法   loss = {l_fix:.3e}   {'<- 精确零点' if l_fix < 1e-16 else ''}")
    print(f"    论文原样      loss = {l_pap:.3e}")

    print(f"\n  [代价] 不动点长什么样(与 pi_ref、与 Prop.B.1 的 beta-tilt 点比):")
    lp_beta = lse_norm(logpi_ref + BETA * r)
    for nm, v in (("pi_ref", logpi_ref), ("Prop.B.1  pi_ref*exp(beta*r)", lp_beta),
                  ("修法的不动点", lp)):
        H, eff, mx = shape(v)
        print(f"    {nm:30s} H={H:7.4f}  有效支撑 {eff:8.3f}/{len(r)}  max p = {mx:.6f}")


def main():
    print(__doc__)

    logpi_ref, r, lens = make_world()
    report_world("A. 合成世界(p7_fixedpoint.py 的原始设定,|y| 200-3200)", logpi_ref, r, lens)

    # 用实测 |y| 的量级重做:robospatial 中位 334 token,组内相对极差 ~0.41
    lp_ref2, r2, _ = make_world(seed=7)
    lens2 = RNG.integers(250, 450, len(r2)).astype(float)
    report_world("B. 实测量级(|y| ~ 250-450 token,robospatial 中位 334)", lp_ref2, r2, lens2)

    # 同质长度扫描:看不动点的塌缩如何随 L 增长
    print("=" * 78)
    print("C. 同质长度扫描:不动点的塌缩程度 vs L(L 相同则一切都可解析)")
    print("=" * 78)
    lp_ref3, r3, _ = make_world(seed=11)
    print(f"    {'L':>8} {'Z*':>12} {'loss(修法)':>14} {'H':>9} {'有效支撑':>10} {'max p':>10}")
    for Lc in (1.0, 2.0, 5.0, 10.0, 50.0, 334.0):
        lens3 = np.full(len(r3), Lc)
        Zs = solve_Zstar(lp_ref3, r3, lens3)
        lp3 = build_fixed_point(Zs, lp_ref3, r3, lens3)
        zt = Zt_fixed(lp3, lp_ref3, r3, lens3)
        l3, _ = loss_at(lp3, lp_ref3, r3, zt, lens3, clip=False)
        H, eff, mx = shape(lp3)
        print(f"    {Lc:8.1f} {Zs:12.5f} {l3:14.3e} {H:9.4f} {eff:10.3f} {mx:10.6f}")
    H0, e0, m0 = shape(lp_ref3)
    Hb, eb, mb = shape(lse_norm(lp_ref3 + BETA * r3))
    print(f"    {'pi_ref':>8} {'':12} {'':14} {H0:9.4f} {e0:10.3f} {m0:10.6f}")
    print(f"    {'beta-tilt':>8} {'':12} {'':14} {Hb:9.4f} {eb:10.3f} {mb:10.6f}   <- Prop.B.1 的目标")

    print("""
=== 判读 ===
[存在性] 修法**恢复了自洽的零损失不动点**:Z_t(归一化 Eq.4) 与 Z* 恒等(数值上到机器精度),
         而论文原样的 Eq.4 在同一点上给出的 Z_t 与 Z* 相差一个可观的量 —— 这正是 T3(b)
         「L != 1 时自洽零点不存在」的另一面。**修法把它修好了。**

[代价]   代价不在存在性,在**不动点的位置**。零损失条件本身(Eq.5 的长度归一化)迫使
         log pi* = log pi_ref + L*(beta*r - Z),也就是 Remark B.4 那句
         「inverse temperature L*beta」。L ~ 300 token 时 exp(L*beta*r) 是一个硬 argmax:
         不动点塌成接近点质量,**分布匹配退化为奖励最大化** —— 恰恰是 GFlowRL 要避免的东西。

         注意这与 P7_STEP23_RESULTS 第 2.3 节收回的那条**不矛盾**:那条说的是
         **论文原样**下极小点落在 c ~ 1,不是 c = L —— 因为那里根本没有零点,
         极小值由「自洽」与「零残差」两个不相容条件折中而来。修法让零点真的存在了,
         于是 Remark B.4 的 L*beta 读法**在修法之下回来了**。

[所以]   第 6 节那条修法**不是免费的**。它把「整组 g 被削成同值」压到 0,
         代价是把不动点推向 argmax。两者是同一个长度归一化的两面。
         真正一致的替代是**两边都不归一化**(回到 Prop. B.1 的设定,tilt = beta,
         不动点熵与 pi_ref 同量级)—— 代价是长序列在 loss 里权重更大,
         而那正是论文当初引入长度归一化要解决的问题。
""")


if __name__ == "__main__":
    main()
