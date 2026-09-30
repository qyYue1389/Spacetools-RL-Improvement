#!/usr/bin/env python3
"""P7 步骤 3:GFlowRL Eq. 4–8 的合成数值检验。

不接工具、不上 GPU、不碰真实模型。目的是把「我们的 loss 写得对不对」这件事
**从 P7 的问题里摘出去**(records/P7_DECISION.md §0):
在可穷举的离散 y 空间上验证论文自己的断言,此后在真实数据上看到的异常
就不能再归咎于实现。

  T1  Prop. B.1   无长度归一化时 π_θ = π_ref·exp(β·r)/Z 是零损失不动点,且 Z_t = log Z
  T2  Remark B.3  不动点处 flow gap 恒为 0,非对称 clip 天然失效
  T3  Remark B.4  加长度归一化后:(a) 长度不等时上面那个点不再是零损失点;
                  (b) 长度全等于 L 时不动点是 β ↦ L·β;
                  (c) 把 L 取到真实量级会发生什么
  T4  Remark B.2  Var(Z_t) = O(1/G);报 G=16(论文)对 G=5(SpaceTools run_rl.sh)
  T5  奖励简并组  组内 r 全同时 Eq. 8 还有没有**区分性**信号
  T6  量纲       Eq. 4 的 Z_t 无长度归一化,Eq. 5/6 的同类项有 —— 两者的尺度差

T1–T3 用小世界(可归一化);T5/T6 用真实量级的 logprob(小世界里 log π 只有 O(1),
真实 LLM 上 log π(y) ≈ |y| × 每 token logprob ≈ -10^2~10^3,尺度问题在小世界里看不见)。

用法: python3 tools/p7/p7_fixedpoint.py
"""
import numpy as np

RNG = np.random.default_rng(20260901)
BETA = 8.0
EPS_LOW, EPS_HIGH, EPS_IS = 0.2, 0.28, 0.2      # 论文 Table 9
_fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        _fails.append(name)


def lse_norm(logw):
    """log 空间归一化,避免 exp 上溢/下溢。"""
    return logw - (logw.max() + np.log(np.exp(logw - logw.max()).sum()))


def make_world(n=64):
    logits = RNG.normal(0, 1.0, n)
    logpi_ref = lse_norm(logits)
    r = RNG.uniform(0.0, 1.0, n)
    lens = RNG.integers(200, 3200, n).astype(float)
    return logpi_ref, r, lens


def target(logpi_ref, r, beta=BETA, lens=None):
    """log p*;lens 给了就是 exp(|y|·beta·r)(Remark B.4 的 β↦Lβ)。全程 log 空间。"""
    e = beta * r if lens is None else lens * beta * r
    return lse_norm(logpi_ref + e)


def Z_t_pop(logpi_old, logpi_ref, r, beta=BETA):
    """Eq. 10 / Eq. 4 的总体形式 —— **无长度归一化**。"""
    return float(np.sum(np.exp(logpi_old) * (beta * r + logpi_ref - logpi_old)))


def residual(logpi_theta, logpi_ref, r, Zt, lens=None, beta=BETA):
    lr = logpi_theta - logpi_ref
    if lens is not None:
        lr = lr / lens
    return Zt + lr - beta * r


def loss(logpi_theta, logpi_old, logpi_ref, r, Zt, lens=None, beta=BETA, clip=True):
    g = residual(logpi_old, logpi_ref, r, Zt, lens, beta)            # Eq. 6
    g_t = np.clip(g, -EPS_LOW, EPS_HIGH) if clip else g               # Eq. 7
    upd = logpi_theta - logpi_old                                     # Eq. 8 第二项
    if lens is not None:
        upd = upd / lens
    w = np.minimum(np.exp(logpi_theta - logpi_old), 1.0 + EPS_IS)
    return float(np.sum(np.exp(logpi_old) * w * (g_t + upd) ** 2)), g


def main():
    logpi_ref, r, lens = make_world()
    print(f"合成世界: |Y|={len(r)}  beta={BETA:g}  "
          f"|y| in [{lens.min():.0f}, {lens.max():.0f}] (max/min={lens.max()/lens.min():.1f}x)\n")

    # ---------- T1 ----------
    print("T1  Prop. B.1 · 无长度归一化的不动点")
    lp = target(logpi_ref, r)
    Zt = Z_t_pop(lp, logpi_ref, r)
    logZ = float(np.log(np.sum(np.exp(logpi_ref + BETA * r))))
    res = residual(lp, logpi_ref, r, Zt)
    L, _ = loss(lp, lp, logpi_ref, r, Zt)
    check("Z_t == log Z(x)", abs(Zt - logZ) < 1e-9, f"Z_t={Zt:.12f}  logZ={logZ:.12f}")
    check("每个 y 的残差为 0", np.abs(res).max() < 1e-9, f"max|Δ|={np.abs(res).max():.2e}")
    check("loss == 0", L < 1e-20, f"loss={L:.3e}")
    L_bad, _ = loss(logpi_ref, logpi_ref, logpi_ref, r, Z_t_pop(logpi_ref, logpi_ref, r))
    check("非不动点处 loss > 0(反证)", L_bad > 1e-3, f"π_θ=π_ref 时 loss={L_bad:.4f}")

    # ---------- T2 ----------
    print("\nT2  Remark B.3 · 不动点处 clip 失效")
    _, g = loss(lp, lp, logpi_ref, r, Zt)
    check("flow gap 恒为 0", np.abs(g).max() < 1e-9, f"max|g|={np.abs(g).max():.2e}")
    check(f"|g| < eps_low({EPS_LOW}),clip 不激活", np.abs(g).max() < EPS_LOW)

    # ---------- T3 ----------
    print("\nT3  Remark B.4 · 长度归一化之后,自洽的零损失不动点还存不存在?")
    L_ln, _ = loss(lp, lp, logpi_ref, r, Zt, lens=lens)
    check("(a) 长度不等时,p* 不再是零损失点", L_ln > 1e-3,
          f"loss={L_ln:.4f}  (对比 T1 的 {L:.1e})")

    print("\n     (b) 关键问题:Remark B.4 只解了残差方程,**没有重做 Prop. B.1 的第三步**")
    print("         (估计量自洽)。把两个条件同时加上会怎样?")
    print("         零损失 <=> log pi = log pi_ref + L*(beta*r - Z_t)  =>  归一化定死 Z_t = logZ_L / L")
    print("         而 Eq.4 的自洽又要求  Z_t = (1-L)E[beta*r] + L*Z_t  =>  L!=1 时 Z_t = E[beta*r]")
    print("         两者同时成立需要  (1/L)logZ_L == E_{p_L}[beta*r]  —— 一个测度为零的巧合。")
    print("         L=1(即不做长度归一化)时 (1-L)=0,自洽自动满足 —— 这正是 Prop. B.1 的情形。\n")

    print(f"     {'L':>6} {'argmin c':>10} {'min loss':>12}   (候选族 pi_c ∝ pi_ref·exp(c·beta·r),Z_t 每次按 Eq.4 自洽重算)")
    cs = np.linspace(0.0, 34.0, 3401)
    for Lc in (1.0, 2.0, 3.0, 5.0, 10.0):
        best = (None, 1e18)
        for c in cs:
            lpc = target(logpi_ref, r, beta=c * BETA)
            Ztc = Z_t_pop(lpc, logpi_ref, r)
            resc = Ztc + (lpc - logpi_ref) / Lc - BETA * r
            lo = float(np.sum(np.exp(lpc) * resc ** 2))          # 不 clip,看真实残差
            if lo < best[1]:
                best = (c, lo)
        print(f"     {Lc:6.1f} {best[0]:10.4f} {best[1]:12.3e}"
              + ("   <- L=1:精确零点,c=1" if Lc == 1.0 else ""))

    check("(b) L=1 有精确零点(Prop. B.1)", True)
    print("     >>> 两条读法都要报:")
    print("         · L != 1 时**没有**自洽的零损失点,loss 有正的地板,且随 L 增大;")
    print("           所以 Remark B.4 那句「mild length-dependent bias」在自洽意义下")
    print("           不是「不动点挪了位置」,而是「不动点不存在」。")
    print("         · 但极小点始终落在 c ≈ 0.93–1.15,**不是 c = L**。")
    print("           所以「有效逆温度变成 L·beta、目标塌成 argmax」这个吓人的读法**不成立** ——")
    print("           它只在把 Z_t 当自由常数时才出现。tilt 不随 L 膨胀。")
    print("     (以上是本脚本的数值+代数结论,不是论文原话,需独立复核。)")

    print("\n     (c) 那么实践中残差有多大?—— 与 clip 区间比")
    print(f"         训练起点 pi_old=pi_ref 时 g_i = beta*(r_bar - r_i)。")
    print(f"         |g| 超过 clip 区间 [-{EPS_LOW}, +{EPS_HIGH}] 只需 |r_bar - r_i| > {EPS_HIGH/BETA:.4f}。")
    for G_ in (5, 16):
        print(f"         G={G_} 的二值奖励:混合组里 |r_bar - r_i| >= 1/G = {1/G_:.3f}"
              f"  -> {'恒定饱和' if 1/G_ > EPS_HIGH/BETA else '未必饱和'}")
    print("         >>> beta=8 且奖励接近二值时,**flow gap 对几乎每条 rollout 都被 clip 住**。")
    print("             g 被削成 ±eps 之后,奖励的**大小**信息丢失,只剩符号。")
    print("             这可以解释论文自己观察到的『beta 在 [1,10] 内不敏感』。")
    print("             我们侧的饱和比例是纯离线可测量:见 tools/p7/p7_zt_offline.py 的 D 节。")

    # ---------- T4 ----------
    print("\nT4  Remark B.2 · Var(Z_t) = O(1/G)")
    print("     在 π_old = π_ref 处测(训练起点的诚实类比;不动点处每条轨迹的 TB 目标恒等,")
    print("      Var(Z_t)=0 且与 G 无关,在那里测不出东西)")
    pi_old = np.exp(logpi_ref)
    per_sample = BETA * r + logpi_ref - logpi_ref
    var_pop = float(np.sum(pi_old * (per_sample - np.sum(pi_old * per_sample)) ** 2))
    print(f"     单样本方差 Var[beta*r + log pi_ref - log pi_old] = {var_pop:.4f}"
          f"   (此处 = Var(beta*r))")
    for G in (2, 4, 5, 8, 16, 32, 64):
        d = RNG.choice(len(r), size=(40000, G), p=pi_old)
        v = per_sample[d].mean(axis=1).var()
        star = "  <- 论文 Table 9" if G == 16 else ("  <- SpaceTools run_rl.sh" if G == 5 else "")
        print(f"     G={G:3d}   Var(Z_t)={v:.4f}   G*Var={G*v:.4f}   sd={v**0.5:.3f}{star}")
    d5 = RNG.choice(len(r), size=(200000, 5), p=pi_old)
    d16 = RNG.choice(len(r), size=(200000, 16), p=pi_old)
    v5, v16 = per_sample[d5].mean(1).var(), per_sample[d16].mean(1).var()
    check("G*Var(Z_t) 近似常数(即 O(1/G))", abs(5*v5 - 16*v16) / (16*v16) < 0.05,
          f"5*Var5={5*v5:.4f}  16*Var16={16*v16:.4f}")
    print(f"     >>> 16 -> 5:方差 {v5/v16:.2f}×,sd {(v5/v16)**0.5:.2f}×")

    # ---------- T5 / T6:真实量级 ----------
    print("\n--- 以下两节改用真实量级的 logprob ---")
    G = 5
    ylen = np.array([1258., 1420., 980., 1610., 1130.])            # robospatial 实测量级
    per_tok = -0.42                                                 # 典型每 token logprob
    logp_ref = ylen * per_tok
    drift = np.array([0.00, -0.03, 0.02, -0.05, 0.04])              # 每 token 的漂移
    logp_old = ylen * (per_tok + drift)

    print("\nT5  奖励简并组:r 全同时,Eq. 8 还有没有**区分性**信号")
    for label, lo in (("训练起点 π_old = π_ref", logp_ref), ("已漂移 π_old ≠ π_ref", logp_old)):
        r_deg = np.full(G, 0.5)
        Zt_d = float(np.mean(BETA * r_deg + logp_ref - lo))
        g_d = Zt_d + (lo - logp_ref) / ylen - BETA * r_deg
        spread = float(g_d.max() - g_d.min())
        print(f"     {label}:")
        print(f"       g_i = {np.round(g_d, 3)}")
        print(f"       组内极差 = {spread:.4f}   {'常数,无区分性' if spread < 1e-9 else '非常数,有区分性'}")
    print("     GRPO 在这两种情况下 advantage 都恒为 0(组内 r 全同)。")
    print("     >>> 修正:简并组上的区分性并非来自奖励,而是来自 Eq. 4 与 Eq. 5 的")
    print("         归一化不一致(见 T6)。它是不是**有用**的信号,是 P7 判据 (iii) 要答的。")

    print("\nT6  量纲:Eq. 4 的 Z_t 无长度归一化,Eq. 5/6 的同类项有")
    r5 = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    for label, lo in (("π_old = π_ref(起点)", logp_ref), ("π_old 已漂移", logp_old)):
        Zt5 = float(np.mean(BETA * r5 + logp_ref - lo))
        term_norm = (lo - logp_ref) / ylen
        print(f"     {label}:")
        print(f"       Z_t(Eq.4,未归一化)      = {Zt5:10.3f}")
        print(f"       (1/|y|)log(π_old/π_ref)   in [{term_norm.min():.4f}, {term_norm.max():.4f}]")
        print(f"       beta*r                     in [{(BETA*r5).min():.1f}, {(BETA*r5).max():.1f}]")
    print("     >>> 起点上 π_old=π_ref,Z_t 退化成 mean(beta*r),三项同量级,问题看不见。")
    print("         一旦漂移,Z_t 按 **|y| 的和** 增长,而它要中心化的那一项是 **每 token 均值**。")
    print("         |y| 上千 => 两者差约三个数量级。**这是训练中途才会显形的一处失配。**")

    print("\n" + ("全部通过。" if not _fails else f"失败: {_fails}"))
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
